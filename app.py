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

st.title("Conversational-RAG-with-PDF-Upload")
st.write("Upload PDFs and chat with history")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# API key input
groq_api_key = st.text_input("Enter Groq API Key", type="password")

if groq_api_key:

    # LLM (use stable model; your model may fail)
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="openai/gpt-oss-120b"
        )
    except:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-8b-8192"
        )

    # Upload PDFs
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and st.session_state.vectorstore is None:

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

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = splitter.split_documents(documents)

        # Safe embeddings
        embeddings = FakeEmbeddings(size=384)

        st.session_state.vectorstore = FAISS.from_documents(
            splits, embeddings
        )

        st.success("Documents processed successfully!")

    # Chat UI
    query = st.chat_input("Ask something about your PDFs...")

    if query and st.session_state.vectorstore:

        retriever = st.session_state.vectorstore.as_retriever()

        # Get docs (new API)
        docs = retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        # 🔥 Include chat history in prompt
        history_text = ""
        for msg in st.session_state.chat_history:
            history_text += f"{msg['role']}: {msg['content']}\n"

        prompt = f"""
        You are a helpful assistant.

        Chat History:
        {history_text}

        Use the context below to answer.
        If not found, say "I don't know".

        Context:
        {context}

        Question:
        {query}
        """

        response = llm.invoke(prompt)

        # Save history
        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response.content
        })

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

else:
    st.warning("Please enter your Groq API Key")
