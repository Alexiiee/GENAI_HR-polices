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
        response.raise_for_status()  # Raise error for bad response
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching website content: {e}")
        return None

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3", model_kwargs={"temperature": 0.7}, huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    memory = ConversationBufferMemory(
                memory_key='chat_history', return_messages=True)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory)
    return qa_chain

def extract_helpful_answer(response_text):
    helpful_answer_prefix = "Helpful Answer:"
    if helpful_answer_prefix in response_text:
        return response_text.split(helpful_answer_prefix)[-1].strip()
    return "No helpful answer found."

def handle_userio(question):
    response = st.session_state.conversation({'query': question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_temp.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            helpful_answer = extract_helpful_answer(message.content)
            st.write(bot_temp.replace("{{MSG}}", helpful_answer), unsafe_allow_html=True)

def save_email(email):
    with open("user_emails.txt", "a") as file:
        file.write(email + "\n")

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email)

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat over Website: ChatAura")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Welcome to ChatAura! 😊")

    question = st.text_input("Ask a question here:")
    if question:
        handle_userio(question)

    with st.sidebar:
        st.header("Enter the below details:")
        website_url = st.text_input("Website URL")

        email = st.text_input("Enter your email:")
        if email:
            if is_valid_email(email):
                st.success("Email is valid.")
            else:
                st.error("Enter valid Email ID.")

        if st.button("Process") and website_url and is_valid_email(email):
            with st.spinner("Processing website..."):
                raw_text_data = get_website_text(website_url)

                if raw_text_data:
                    text_chunks = get_text_chunks(raw_text_data)
                    vectorstore = get_vectorstore(text_chunks)
                    st.success("Website processed successfully!")
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                else:
                    st.warning("No text extracted from the website. Please check the URL and try again.")

if __name__ == '__main__':
    main()