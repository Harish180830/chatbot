"""
Multi-Modal RAG Chatbot
------------------------
Upload PDFs, Word docs, text files, images, and videos.
The app extracts text from all of them, stores it in a vector database,
and lets you chat with your documents using a Groq-hosted LLM.

HOW EACH FILE TYPE IS HANDLED:
- PDF / DOCX / TXT -> text extracted directly (LangChain document loaders)
- Images (jpg/png)  -> text extracted using OCR (pytesseract)
- Videos (mp4/mov)  -> audio is pulled out, then transcribed using
                       Groq's Whisper API (same API key as your chat LLM,
                       so you only need one key)

Run with:  streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from groq import Groq
from PIL import Image
import pytesseract
from moviepy import VideoFileClip

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.chains import RetrievalQA


# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(page_title="Multi-Modal RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Modal RAG Chatbot")
st.caption("Chat with PDFs, Word docs, text files, images, and videos — powered by Groq + LangChain")


# ---------------------------------------------------------------------
# SESSION STATE (keeps data alive between reruns / chat turns)
# ---------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


# ---------------------------------------------------------------------
# SIDEBAR: API KEY + FILE UPLOAD
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get a free key at console.groq.com",
    )

    st.divider()
    st.header("📁 Upload Files")

    uploaded_files = st.file_uploader(
        "Add PDFs, DOCX, TXT, images, or videos",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "mp4", "mov", "avi"],
        accept_multiple_files=True,
    )

    process_btn = st.button("🚀 Process Files", use_container_width=True)

    if st.session_state.vectorstore is not None:
        st.success("✅ Knowledge base ready")


# ---------------------------------------------------------------------
# LOADER FUNCTIONS
# Each one takes a saved-to-disk file path and returns LangChain Documents
# ---------------------------------------------------------------------

def load_pdf(path, filename):
    loader = PyPDFLoader(path)
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = filename
    return docs


def load_docx(path, filename):
    loader = Docx2txtLoader(path)
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = filename
    return docs


def load_txt(path, filename):
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = filename
    return docs


def load_image(path, filename):
    """Extract any visible text from an image using OCR."""
    image = Image.open(path)
    text = pytesseract.image_to_string(image)
    if not text.strip():
        text = "[No readable text was found in this image.]"
    return [Document(page_content=text, metadata={"source": filename, "type": "image"})]


def load_video(path, filename, client):
    """Pull audio out of the video, then transcribe it with Groq Whisper."""
    audio_path = path + ".mp3"
    clip = VideoFileClip(path)
    clip.audio.write_audiofile(audio_path, logger=None)
    clip.close()

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
        )

    os.remove(audio_path)
    text = transcript.text if transcript.text.strip() else "[No speech detected in this video.]"
    return [Document(page_content=text, metadata={"source": filename, "type": "video"})]


def load_any_file(uploaded_file, groq_client):
    """Dispatch a Streamlit UploadedFile to the right loader based on extension."""
    suffix = "." + uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            return load_pdf(tmp_path, uploaded_file.name)
        elif suffix == ".docx":
            return load_docx(tmp_path, uploaded_file.name)
        elif suffix == ".txt":
            return load_txt(tmp_path, uploaded_file.name)
        elif suffix in (".png", ".jpg", ".jpeg"):
            return load_image(tmp_path, uploaded_file.name)
        elif suffix in (".mp4", ".mov", ".avi"):
            return load_video(tmp_path, uploaded_file.name, groq_client)
        else:
            return []
    finally:
        os.remove(tmp_path)


# ---------------------------------------------------------------------
# BUILD THE VECTOR STORE (RAG "knowledge base") FROM UPLOADED FILES
# ---------------------------------------------------------------------

def build_vectorstore(uploaded_files, groq_api_key):
    groq_client = Groq(api_key=groq_api_key)

    all_docs = []
    progress = st.progress(0.0, text="Reading files...")

    for i, f in enumerate(uploaded_files):
        st.write(f"Processing **{f.name}**...")
        docs = load_any_file(f, groq_client)
        all_docs.extend(docs)
        progress.progress((i + 1) / len(uploaded_files), text=f"Processed {f.name}")

    if not all_docs:
        return None

    # Split long text into overlapping chunks so retrieval is more precise
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)

    # Embeddings turn text into vectors so we can search by meaning, not just keywords
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def build_qa_chain(vectorstore, groq_api_key):
    # Groq deprecated the llama-3.3-70b-versatile / llama-3.1-8b-instant chat
    # models. openai/gpt-oss-120b is their current recommended general-purpose
    # model (still runs on Groq's fast inference, just a different model id).
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-120b",
        temperature=0.2,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return chain


# ---------------------------------------------------------------------
# HANDLE "PROCESS FILES" BUTTON
# ---------------------------------------------------------------------
if process_btn:
    if not groq_api_key:
        st.sidebar.error("Please enter your Groq API key first.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one file.")
    else:
        with st.spinner("Building knowledge base..."):
            vectorstore = build_vectorstore(uploaded_files, groq_api_key)
            if vectorstore is None:
                st.sidebar.error("No text could be extracted from these files.")
            else:
                st.session_state.vectorstore = vectorstore
                st.session_state.qa_chain = build_qa_chain(vectorstore, groq_api_key)
                st.session_state.chat_history = []
                st.sidebar.success(f"Indexed {len(uploaded_files)} file(s) successfully!")


# ---------------------------------------------------------------------
# CHAT INTERFACE
# ---------------------------------------------------------------------
st.divider()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s}")

user_question = st.chat_input("Ask something about your uploaded files...")

if user_question:
    if st.session_state.qa_chain is None:
        st.warning("Upload and process some files first (see the sidebar).")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"query": user_question})
                answer = result["result"]
                sources = sorted({
                    doc.metadata.get("source", "unknown")
                    for doc in result.get("source_documents", [])
                })
                st.markdown(answer)
                if sources:
                    with st.expander("📚 Sources"):
                        for s in sources:
                            st.write(f"- {s}")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
