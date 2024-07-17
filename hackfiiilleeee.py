import streamlit as st
import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from html_temp import css, bot_temp, user_temp

def get_website_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        texts = soup.stripped_strings
        return "\n".join(texts)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching website content: {e}")
        return None

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2048,
        chunk_overlap=128,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3", model_kwargs={"temperature": 0.5,"top_k" :5, "max_length":1000}, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory)
    return qa_chain

def extract_helpful_answer(response_text):
    helpful_answer_prefix = "Helpful Answer:"
    if helpful_answer_prefix in response_text:
        return response_text.split(helpful_answer_prefix)[-1].split('\n')[0].strip()
    else:
        return "No helpful answer found."

def handle_userio(question):
    response = st.session_state.conversation({'query': question})
    st.session_state.chat_history = response['chat_history']

    new_chat_history = []
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            new_chat_history.append(user_temp.replace("{{MSG}}", message.content))
        else:
            helpful_answer = extract_helpful_answer(message.content)
            new_chat_history.append(bot_temp.replace("{{MSG}}", helpful_answer))
    
    st.session_state.chat_history = new_chat_history[::-1]

    for message in st.session_state.chat_history:
        st.write(message, unsafe_allow_html=True)

def save_email(email):
    with open("user_emails.txt", "a") as f:
        f.write(email + "\n")

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email)

def main():
    load_dotenv()
    global HUGGINGFACEHUB_API_TOKEN
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    st.set_page_config(page_title="Chat over Website: ChatAura")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Welcome to ChatAura! :smile:")

    st.sidebar.header("Enter Details:")

    email = st.sidebar.text_input("Enter your email:")
    if email:
        if is_valid_email(email):
            st.sidebar.success("Email is valid.")
        else:
            st.sidebar.error("Enter valid Email ID.")

    url = st.sidebar.text_input("Enter the website URL:")
    if url and email and is_valid_email(email):
        if st.sidebar.button("Process"):
            save_email(email)
            with st.spinner("Processing data, Please wait ..."):
                raw_text_data = get_website_text(url)
                if raw_text_data:
                    text_chunks = get_text_chunks(raw_text_data)
                    vectorstore = get_vectorstore(text_chunks)
                    st.sidebar.success("Website data processed successfully!")
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                else:
                    st.sidebar.warning("No text extracted from the website. Please check the URL and try again.")

    question = st.text_input("Ask a question here:")
    if question and st.session_state.conversation:
        handle_userio(question)

if __name__ == '__main__':
    main()