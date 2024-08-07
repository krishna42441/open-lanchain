import streamlit as st
import os
from dotenv import load_dotenv
import requests
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
import time
import gc

# Load environment variables from .env file
load_dotenv()

# Set USER_AGENT environment variable
os.environ['USER_AGENT'] = "my_custom_user_agent"

# Load the API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("API key not found. Please set the GROQ_API_KEY environment variable.")
    st.stop()

# Load documents efficiently
@st.cache_resource
def load_documents():
    loader = PyPDFDirectoryLoader("./my_content")
    docs = loader.load()[:10]  # Load only 10 documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(final_documents, embeddings)
    return vector_store

def manage_memory():
    # Run garbage collection
    gc.collect()

st.session_state.vector = load_documents()

# Streamlit UI
st.title("langchain Bot")
st.write("This is a chatbot which uses langchain and Gemma model to answer any question based on a document in this case which is my resume. You can ask me any question on personal or professional questions.")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-It")

prompt_template = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context.
Pretend you are Sai Krishna and the document is your resume which has educational background, technical skills, projects, and experience. Be creative and understand the question asked. If you don't know the answer, give a response to approach Sai Krishna Veeramaneni at v.krishna2727@gmail.com.

Please provide accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
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
        # Manage memory after processing
        manage_memory()
    except Exception as e:
        st.error(f"An error occurred during query execution: {e}")
        # Manage memory in case of error
        manage_memory()
