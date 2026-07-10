<h1 align="center">📄 Conversational RAG with PDF Upload</h1>

<p align="center">
  Chat with multiple PDFs at once — powered by a Retrieval-Augmented Generation pipeline with persistent conversational memory.
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain" />
<img src="https://img.shields.io/badge/Groq%20API-F55036?style=for-the-badge&logo=groq&logoColor=white" alt="Groq" />
<img src="https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white" alt="FAISS" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
</p>

---

## 🚀 Overview

**Conversational RAG with PDF Upload** is a Streamlit application that lets you upload one or more PDF documents and chat with them in natural language. It combines document chunking, vector similarity search with FAISS, and Groq-hosted LLMs to answer questions grounded in your uploaded files — while keeping track of the full conversation history for follow-up questions.

## ✨ Features

- 📁 **Multi-PDF upload** — process several PDF documents in a single session
- ✂️ **Automatic chunking** — documents are split into overlapping chunks (1000 chars, 200 overlap) using `RecursiveCharacterTextSplitter`
- 🔍 **Vector search** — chunks are indexed in a FAISS vector store for fast semantic retrieval
- 💬 **Conversational memory** — full chat history is tracked in session state and fed back into each prompt for context-aware follow-ups
- ⚡ **Groq-powered inference** — uses `openai/gpt-oss-120b` by default, with automatic fallback to `llama3-8b-8192`
- 🖥️ **Simple chat UI** — built entirely with Streamlit's native `st.chat_input` / `st.chat_message` components

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **App Framework** | Streamlit |
| **LLM Orchestration** | LangChain |
| **LLM Inference** | Groq API (`openai/gpt-oss-120b`, fallback `llama3-8b-8192`) |
| **Vector Store** | FAISS |
| **Document Loading** | PyPDFLoader (pypdf) |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` |
| **Environment Management** | python-dotenv |

## 📂 Project Structure

```
├── app.py               # Main Streamlit application
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (not committed)
└── README.md
```

## ⚙️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/khushaan18/<repo-name>.git
   cd <repo-name>
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file in the project root:
   ```env
   HF_TOKEN=your_huggingface_token
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_PROJECT="Conversational RAG with PDF Upload"
   LANGCHAIN_TRACING_V2=true
   ```
   > ⚠️ Never commit your `.env` file. Add it to `.gitignore` and keep your keys private.

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

## ▶️ Usage

1. Launch the app and enter your **Groq API key** in the input field
2. Upload one or more PDF files
3. Wait for the "Documents processed successfully!" confirmation
4. Ask questions about your documents in the chat box — the assistant will answer using retrieved context and prior conversation history

## 📝 Notes & Limitations

- The current build uses `FakeEmbeddings` as a placeholder for vector generation. For accurate semantic retrieval, swap this out for a real embedding model (e.g. `langchain-huggingface`'s `HuggingFaceEmbeddings`, which is already included in `requirements.txt`).
- A Groq API key must be entered manually in the UI at runtime; it is not read from the `.env` file.
- LangSmith tracing is enabled via environment variables for observability into the RAG pipeline.

## 📫 Contact

<p align="center">
  <a href="mailto:khushaansaini62@gmail.com"><img src="https://img.shields.io/badge/Email-khushaansaini62%40gmail.com-blue?style=for-the-badge&logo=gmail" alt="Email" /></a>
  <a href="https://www.linkedin.com/in/khushaan-saini-86a0ba275"><img src="https://img.shields.io/badge/LinkedIn-khushaan--saini-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn" /></a>
</p>
