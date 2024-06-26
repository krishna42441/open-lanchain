import streamlit as st
import os
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

# Set USER_AGENT environment variable
os.environ['USER_AGENT'] = "my_custom_user_agent"

# Your API key
groq_api_key = "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"

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
    # Clear cache if necessary
    st.caching.clear_cache()
    # Run garbage collection
    gc.collect()

st.session_state.vector = load_documents()

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
        # Manage memory after processing
        manage_memory()
    except Exception as e:
        st.error(f"An error occurred during query execution: {e}")
        # Manage memory in case of error
        manage_memory()
