import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from html_temp import css, bot_temp, user_temp


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1024,
        chunk_overlap=256,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3", model_kwargs={"temperature": 0.2}, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)
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
    return response_text.strip()

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
    
    # Reverse the order to display the latest message on top
    st.session_state.chat_history = new_chat_history[::-1]

    for message in st.session_state.chat_history:
        st.write(message, unsafe_allow_html=True)

def main():
    load_dotenv()
    global HUGGINGFACEHUB_API_TOKEN
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    st.set_page_config(page_title="Chat over PDF: ChatAura")
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
        st.header("List of your documents")
        pdf_docs = st.file_uploader(":warning: Upload PDF file only", accept_multiple_files=True, type="pdf")

        if pdf_docs and st.button("Upload now"):
            with st.spinner("Processing data..."):
                # Get text from the PDFs
                raw_text_data = get_pdf_text(pdf_docs)

                if raw_text_data:
                    # Get text chunks from the data
                    text_chunks = get_text_chunks(raw_text_data)

                    # Create a vector store for different chunks of data
                    vectorstore = get_vectorstore(text_chunks)
                    st.success("File uploaded successfully!")

                    # Introducing memory in the chatbot - Conversational History
                    st.session_state.conversation = get_conversation_chain(vectorstore)

                else:
                    st.warning("No text extracted from the PDF(s). Please check the file(s) and try again.")


if __name__ == '__main__':
    main()