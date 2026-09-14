# RAG Chat

Upload a PDF, ask questions, get answers grounded only in that document.

## Stack

- **Streamlit** - UI
- **Groq** (`openai/gpt-oss-120b`) - LLM
- **HuggingFace** (`sentence-transformers/all-MiniLM-L6-v2`) - embeddings
- **Chroma** - vector store (in-memory, per session)
- **LangChain** - glue

## How it works

1. Upload a PDF from the sidebar.
2. It gets split into chunks, embedded, and stored in a fresh in-memory Chroma collection.
3. Ask a question. The app retrieves the most relevant chunks (MMR search) and asks the LLM to answer using only that context.
4. Remove the file or click "Clear document & chat" and everything for that session is wiped clean.

Nothing is saved to disk. Each session's document and chat are private to that session.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your-groq-key
HUGGINGFACEHUB_ACCESS_TOKEN=your-Hugging-face-key
```

## Run

```bash
streamlit run app.py
```

## Notes

- First load takes a bit longer as models warm up.
- New upload replaces the previous one and resets the chat.