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

# moviepy changed its import path between v1 and v2 - this works with either
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LangChain moved these between versions - try the new location, fall back to the old one
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(page_title="Multi-Modal RAG Chatbot", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------------
# THEME: two CSS blocks.
# - LOGIN_CSS: dark glass card (black/blue/white), used only on the login page.
# - APP_CSS: light blue/white/black theme, used after the user signs in.
# Base app colors also come from .streamlit/config.toml (Streamlit's native
# theming - more stable across Streamlit versions than CSS overrides alone).
# ---------------------------------------------------------------------
LOGIN_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at top, #16222A 0%, #0D1117 100%);
    }

    /* Center the login card both horizontally and vertically */
    [data-testid="stAppViewContainer"] .main .block-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 88vh;
        padding-top: 0;
        padding-bottom: 0;
    }

    .login-card {
        width: 420px;
        max-width: 90vw;
        padding: 2.75rem 2.5rem;
        background: linear-gradient(155deg, rgba(30, 41, 59, 0.85), rgba(13, 17, 23, 0.9));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 22px;
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(14px);
    }
    .login-title {
        text-align: center;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 1.7rem;
        margin-bottom: 0.25rem;
    }
    .login-subtitle {
        text-align: center;
        color: #9FB3C8;
        font-size: 0.88rem;
        margin-bottom: 1.75rem;
    }

    /* Tabs */
    .login-card [data-baseweb="tab-list"] {
        background: transparent;
        gap: 0.5rem;
    }
    .login-card [data-baseweb="tab"] {
        color: #9FB3C8;
        font-weight: 600;
    }
    .login-card [aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom-color: #2E86C1 !important;
    }

    /* Field labels */
    .login-card [data-testid="stWidgetLabel"] p {
        color: #E5EAF0;
        font-weight: 600;
        font-size: 0.92rem;
    }

    /* Input boxes - dark, rounded, matches reference */
    .login-card input {
        background-color: #141A24 !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        padding: 0.7rem 1rem !important;
    }
    .login-card input::placeholder {
        color: #5C6B7A !important;
    }

    /* Login / Create Account button - black-to-blue gradient, pill shaped */
    .login-card .stButton > button,
    .login-card .stFormSubmitButton > button {
        width: 100%;
        background: linear-gradient(90deg, #0D1B2A 0%, #1B4965 55%, #2E86C1 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 0;
        font-weight: 700;
        letter-spacing: 0.3px;
        margin-top: 0.5rem;
    }
    .login-card .stButton > button:hover,
    .login-card .stFormSubmitButton > button:hover {
        background: linear-gradient(90deg, #000000 0%, #163A5F 55%, #2E86C1 100%);
        color: #FFFFFF;
    }

    .login-caption { text-align: center; color: #5C6B7A; }
</style>
"""

APP_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at top, #16222A 0%, #0D1117 100%);
    }

    /* Headings - bright and clearly visible */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    p, span, label, .stMarkdown, .stCaption {
        color: #E5EAF0 !important;
    }

    /* Sidebar - dark glass panel matching the login card */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.9), rgba(13, 17, 23, 0.95));
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] * {
        color: #E5EAF0 !important;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }

    /* Text / password inputs - dark, matches login page inputs */
    input, textarea {
        background-color: #141A24 !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }
    input::placeholder { color: #7C8CA0 !important; }

    /* File uploader dropzone */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #141A24 !important;
        border: 1px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
    }

    /* Buttons - black-to-blue gradient, matches login button */
    .stButton > button,
    .stFormSubmitButton > button,
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(90deg, #0D1B2A 0%, #1B4965 55%, #2E86C1 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }
    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: linear-gradient(90deg, #000000 0%, #163A5F 55%, #2E86C1 100%) !important;
        color: #FFFFFF !important;
    }

    /* Chat bubbles - dark cards with a blue accent border */
    [data-testid="stChatMessage"] {
        background-color: #141A24 !important;
        border-radius: 14px;
        border: 1px solid rgba(46, 134, 193, 0.35);
        color: #FFFFFF !important;
    }
    [data-testid="stChatMessage"] * { color: #FFFFFF !important; }

    /* Chat input box */
    [data-testid="stChatInput"] {
        background-color: #141A24 !important;
        border-radius: 50px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #FFFFFF !important;
    }

    /* Inline code / tags (e.g. "int data", "Node *next") - bright green on dark, high contrast */
    code {
        background-color: #0B2B1E !important;
        color: #4ADE80 !important;
        border-radius: 5px;
        padding: 0.15rem 0.4rem;
    }

    /* Success / error banners - readable on dark background */
    [data-testid="stAlert"] {
        background-color: #141A24 !important;
        border-radius: 10px;
    }
    [data-testid="stAlert"] p { color: #FFFFFF !important; }

    /* Expander (Sources) */
    [data-testid="stExpander"] {
        background-color: #141A24 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
    }
</style>
"""


# ---------------------------------------------------------------------
# SESSION STATE (keeps data alive between reruns / chat turns)
# ---------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "users" not in st.session_state:
    # Demo-only in-memory user store. Resets whenever the app restarts -
    # this is NOT persistent storage, just enough for a working login flow.
    st.session_state.users = {}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


# ---------------------------------------------------------------------
# LOGIN / SIGN UP PAGE
# ---------------------------------------------------------------------
def show_login_page():
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🤖 RAG Chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="login-subtitle">Sign in to chat with your documents, images, and videos</div>',
        unsafe_allow_html=True,
    )

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("🔒 Login", use_container_width=True)

        if submitted:
            users = st.session_state.users
            if username in users and users[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_signup:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username", placeholder="Enter username")
            new_password = st.text_input("Choose a password", type="password", placeholder="Enter password")
            confirm_password = st.text_input("Confirm password", type="password", placeholder="Re-enter password")
            signup_submitted = st.form_submit_button("Create Account", use_container_width=True)

        if signup_submitted:
            if not new_username or not new_password:
                st.error("Username and password can't be empty.")
            elif new_password != confirm_password:
                st.error("Passwords don't match.")
            elif new_username in st.session_state.users:
                st.error("That username is already taken.")
            else:
                st.session_state.users[new_username] = new_password
                st.success("Account created! Go to the Login tab to sign in.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="login-caption">⚠️ Demo login only — accounts are stored in memory and reset '
        'when the app restarts. Don\'t reuse a real password here.</p>',
        unsafe_allow_html=True,
    )


if not st.session_state.authenticated:
    show_login_page()
    st.stop()

st.markdown(APP_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# MAIN APP HEADER
# ---------------------------------------------------------------------
header_col1, header_col2 = st.columns([5, 1])
with header_col1:
    st.title("🤖 Multi-Modal RAG Chatbot")
    st.caption(f"Signed in as **{st.session_state.username}** — chat with PDFs, DOCX, TXT, images, and videos")
with header_col2:
    if st.button("Log Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()


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


RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant answering questions about the user's uploaded "
    "files (documents, OCR'd images, and video transcripts).\n\n"
    "You have two modes. Pick exactly one:\n"
    "- [DOCUMENT] mode: use this if the context below actually contains "
    "information relevant to the question. Answer using ONLY that context — "
    "do not add outside facts in this mode.\n"
    "- [GENERAL] mode: use this if the context does NOT contain relevant "
    "information. In this mode, ignore the context and answer the question "
    "yourself using your own general knowledge, as a normal helpful AI "
    "assistant would.\n\n"
    "Your response MUST start with exactly one tag on its own first line — "
    "either [DOCUMENT] or [GENERAL] — followed by a blank line, then the "
    "answer.\n\n"
    "Format the answer itself in structured Markdown:\n"
    "- Start with a one-sentence direct answer\n"
    "- Follow with a '**Key Points**' section as bullet points, if there is "
    "more than one relevant fact\n"
    "- Keep it concise; use short bullets, not long paragraphs\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def parse_answer(raw_answer):
    """Split off the [DOCUMENT] / [GENERAL] mode tag the LLM was asked to prefix."""
    text = raw_answer.strip()
    if text.upper().startswith("[GENERAL]"):
        return "general", text[len("[GENERAL]"):].strip()
    if text.upper().startswith("[DOCUMENT]"):
        return "document", text[len("[DOCUMENT]"):].strip()
    # Model didn't follow the format - fall back to treating it as a document answer
    return "document", text


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

    # LCEL chain: retrieved context + question -> prompt -> llm -> plain text.
    # Built this way instead of the older RetrievalQA helper, which pulls in
    # a legacy pydantic-heavy base class that breaks on newer Python versions.
    answer_chain = RAG_PROMPT | llm | StrOutputParser()

    return {"retriever": retriever, "answer_chain": answer_chain}


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
        if msg["role"] == "assistant" and msg.get("mode"):
            if msg["mode"] == "general":
                st.caption("🌐 General AI knowledge — not from your uploaded files")
            else:
                st.caption("📄 Answered from your uploaded files")
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("mode") == "document" and msg.get("sources"):
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
                retriever = st.session_state.qa_chain["retriever"]
                answer_chain = st.session_state.qa_chain["answer_chain"]

                docs = retriever.invoke(user_question)
                context = format_docs(docs)
                raw_answer = answer_chain.invoke({"context": context, "question": user_question})
                mode, answer = parse_answer(raw_answer)

                sources = sorted({
                    doc.metadata.get("source", "unknown") for doc in docs
                })

                if mode == "general":
                    st.caption("🌐 General AI knowledge — not from your uploaded files")
                else:
                    st.caption("📄 Answered from your uploaded files")

                st.markdown(answer)
                if mode == "document" and sources:
                    with st.expander("📚 Sources"):
                        for s in sources:
                            st.write(f"- {s}")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources, "mode": mode}
        )
