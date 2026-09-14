import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f1117;
    }
    section[data-testid="stSidebar"] {
        background-color: #161925;
        border-right: 1px solid #262b3d;
    }
    .app-title {
        font-size: 1.9rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
        color: #f5f6fa;
    }
    .app-subtitle {
        color: #9aa1b5;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .pill-ready {
        background-color: #17351f;
        color: #4ade80;
        border: 1px solid #245c33;
    }
    .pill-empty {
        background-color: #35291a;
        color: #fbbf24;
        border: 1px solid #5c4a24;
    }
    .doc-card {
        background-color: #1b1f2e;
        border: 1px solid #2a2f45;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
        color: #c6cbe0;
    }
    div[data-testid="stChatMessage"] {
        background-color: #161925;
        border-radius: 12px;
        padding: 0.4rem 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Cached resources (loaded once per process, shared across sessions)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    # Imported here, not at the top of the file, so this heavy import only
    # happens after the loading text below is already on screen.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_llm():
    from langchain_groq import ChatGroq

    return ChatGroq(model="openai/gpt-oss-120b", temperature=0)


@st.cache_resource(show_spinner=False)
def get_prompt():
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a strictly context-grounded assistant.

Your task is to answer the user's question using ONLY the information contained in the provided context.

Rules:

1. Treat the context as your ONLY source of truth.
2. Do NOT use outside knowledge, prior knowledge, assumptions, guesses, or general world knowledge.
3. You may combine and reason over multiple pieces of information in the context, but every factual claim in your answer must be supported by the context.
4. Do NOT fill in missing information or infer facts that are not reasonably supported by the context.
5. If the context does not contain enough information to answer the question, reply exactly:

"Information not found in the provided context. Please ask a question related to the provided context, or rephrase your question if you believe the information is present."

6. Answer concisely and directly.
""",
            ),
            (
                "user",
                """Context:
{context}

Question:
{query}""",
            ),
        ]
    )


# This block runs BEFORE any LangChain/Chroma import happens, so the loading
# text appears on screen immediately, and the heavy imports only kick in
# once the spinner is already visible to the user.
loading_placeholder = st.empty()
if "rag_ready" not in st.session_state:
    with loading_placeholder.container():
        st.markdown(
            """
            <div style="display:flex; flex-direction:column; align-items:center;
                        justify-content:center; height:60vh; text-align:center;">
                <div style="font-size:2rem;">📄🔎</div>
                <div style="font-size:1.2rem; font-weight:600; color:#f5f6fa; margin-top:0.6rem;">
                    Loading the RAG system...
                </div>
                <div style="color:#9aa1b5; margin-top:0.3rem;">
                    First load can take up to ~30 seconds while things warm up.
                    Subsequent loads will be much faster.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.spinner(""):
            get_embeddings()
            get_llm()
            get_prompt()
    st.session_state.rag_ready = True
    loading_placeholder.empty()

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def reset_session():
    # Actually delete the old collection's data from chromadb, not just drop
    # our Python reference to it -- otherwise it lingers in the process-wide
    # shared client and can leak into the next document's answers.
    if st.session_state.vector_store is not None:
        try:
            st.session_state.vector_store.delete_collection()
        except Exception:
            pass

    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.doc_name = None
    st.session_state.num_chunks = 0
    st.session_state.collection_name = None
    st.session_state.messages = []

    # Bump the file_uploader's key so Streamlit renders it as a brand-new,
    # empty widget -- this is what actually clears the "x" file from view.
    st.session_state.uploader_key += 1


def build_vector_store(uploaded_file):
    """Read the uploaded PDF, chunk it, embed it, and build an in-memory
    Chroma store scoped to this browser session only."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = get_embeddings()

        # Delete the previous collection's data before creating a new one,
        # so old vectors never linger in the shared chromadb client.
        if st.session_state.vector_store is not None:
            try:
                st.session_state.vector_store.delete_collection()
            except Exception:
                pass

        # A fresh, unique collection name per upload guarantees this upload
        # can never attach to (or mix with) any previous collection.
        collection_name = f"session_{uuid.uuid4().hex}"

        # No persist_directory -> ephemeral, in-memory Chroma client.
        # Nothing is written to disk and it disappears when the session ends.
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
        )

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.8},
        )

        st.session_state.vector_store = vector_store
        st.session_state.retriever = retriever
        st.session_state.doc_name = uploaded_file.name
        st.session_state.num_chunks = len(chunks)
        st.session_state.collection_name = collection_name
        st.session_state.messages = []  # fresh chat for the new document

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def answer_query(query: str) -> str:
    docs = st.session_state.retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = get_prompt().invoke({"context": context, "query": query})
    response = get_llm().invoke(final_prompt)
    return response.content


# ----------------------------------------------------------------------------
# Sidebar — upload + status
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="app-title">📄 RAG Chat</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Chat with any PDF, privately, per session.Recommeded pdf pages:<0-100></div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key=f"pdf_uploader_{st.session_state.uploader_key}",
    )

    if uploaded_file is not None:
        is_new_file = uploaded_file.name != st.session_state.doc_name
        if is_new_file:
            with st.spinner("Reading, chunking, and embedding your document..."):
                build_vector_store(uploaded_file)
            st.success(f"Indexed **{uploaded_file.name}**")
    elif st.session_state.doc_name is not None:
        # The user clicked the "x" to remove the uploaded file — fully
        # reset the session so nothing lingers (vector store, chat, etc.)
        reset_session()
        st.rerun()

    st.divider()

    if st.session_state.retriever is not None:
        st.markdown('<span class="status-pill pill-ready">● Ready</span>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="doc-card">
                <b>Document:</b> {st.session_state.doc_name}<br>
                <b>Chunks indexed:</b> {st.session_state.num_chunks}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Clear document & chat", use_container_width=True):
            reset_session()
            st.rerun()
    else:
        st.markdown('<span class="status-pill pill-empty">● No document loaded</span>', unsafe_allow_html=True)
        st.caption("Upload a PDF above to start chatting.")

    st.divider()
    st.caption(
        "Your file and its index live only in this browser session's memory. "
        "Nothing is saved to disk or shared with other users."
    )

# ----------------------------------------------------------------------------
# Main chat area
# ----------------------------------------------------------------------------
st.markdown('<div class="app-title">Ask your document</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Answers are grounded only in the PDF you uploaded.</div>',
    unsafe_allow_html=True,
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.retriever is None:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    query = st.chat_input("Ask a question about your document...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_query(query)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})