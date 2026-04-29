import streamlit as st
import tempfile
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings

# Load env
load_dotenv()

st.set_page_config(page_title="Multi-PDF Chat", layout="wide")

st.title("📄 Chat with Multiple PDFs")
st.write("Upload PDFs and ask questions")

# API key
groq_api_key = st.text_input("Enter Groq API Key", type="password")

if groq_api_key:

    # 🔥 Try your model, fallback if fails
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="openai/gpt-oss-120b"
        )
        st.success("Using model: openai/gpt-oss-120b")
    except Exception:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-8b-8192"
        )
        st.warning("Fallback to llama3-8b-8192")

    # Upload multiple PDFs
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:

        # Build vector DB only once
        if "vectorstore" not in st.session_state:

            documents = []

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    temp_path = tmp.name

                loader = PyPDFLoader(temp_path)
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name

                documents.extend(docs)

            # Split text
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = splitter.split_documents(documents)

            # Lightweight embeddings (no crash)
            embeddings = FakeEmbeddings(size=384)

            # Create vector DB
            st.session_state.vectorstore = FAISS.from_documents(
                splits, embeddings
            )

        retriever = st.session_state.vectorstore.as_retriever()

        # User question
        query = st.text_input("Ask a question")

        if query:
            # ✅ Updated LangChain API
            docs = retriever.invoke(query)

            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
            Answer the question using the context below.
            If the answer is not in the context, say "I don't know".

            Context:
            {context}

            Question:
            {query}
            """

            response = llm.invoke(prompt)

            st.subheader("🧠 Answer")
            st.write(response.content)

            # Show sources
            st.subheader("📚 Sources")
            sources = set([doc.metadata.get("source", "Unknown") for doc in docs])
            for src in sources:
                st.write(f"- {src}")

else:
    st.warning("Please enter your Groq API Key")
