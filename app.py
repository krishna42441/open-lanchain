import streamlit as st
import os
import requests  # Import the requests library
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import retrieval_qa
from langchain_community.vectorstores import FAISS
import time

# Set USER_AGENT environment variable
os.environ['USER_AGENT'] = "my_custom_user_agent"

# Your API key
groq_api_key = "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"

# Initialize session state
if "vector" not in st.session_state:
    try:
        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./my_content")
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vector = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    except requests.exceptions.ConnectionError as e:
        st.error(f"Failed to connect to the embeddings service: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()

# Streamlit UI
st.title("ChatGroq Demo")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-It")

prompt_template = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only
pretend you are sai krishna veeramaneni who is in the document and answer 

please provide accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

documents_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, documents_chain)

user_input = st.text_input("Input your question here")

if user_input:
    start = time.process_time()
    try:
        response = retrieval_chain.invoke({"input": user_input})
        print("response time: ", time.process_time() - start)
        st.write(response['answer'])
    except Exception as e:
        st.error(f"An error occurred during query execution: {e}")
