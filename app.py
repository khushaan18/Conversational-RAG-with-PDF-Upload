import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="PDF Chat", layout="wide")

st.title("Conversational-RAG-with-PDF-Upload")
st.write("Upload a PDF and ask questions")

# API Key
groq_api_key = st.text_input("Enter Groq API Key", type="password")

if groq_api_key:

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-120b"
    )

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = splitter.split_documents(docs)

        # Embeddings
        from langchain_community.embeddings import FakeEmbeddings

        embeddings = FakeEmbeddings(size=384)

        # Vector DB
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        # Chat input
        query = st.text_input("Ask a question")

        if query:
            docs = retriever.get_relevant_documents(query)

            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
            Answer the question using the context below.
            If not found, say "I don't know".

            Context:
            {context}

            Question:
            {query}
            """

            response = llm.invoke(prompt)

            st.subheader("Answer:")
            st.write(response.content)

else:
    st.warning("Please enter your Groq API Key")
