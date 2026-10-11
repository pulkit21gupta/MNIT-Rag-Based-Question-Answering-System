"""
FastAPI Backend for MNIT RAG Assistant
Serves the chat UI and exposes /api/chat, /api/status, /api/sources endpoints.
Auto-generated API docs available at http://localhost:5000/docs
"""

import os
import sys
import traceback
import threading
import warnings
from contextlib import asynccontextmanager

# Fix Windows console Unicode encoding FIRST
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Config ─────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
CHROMA_PERSIST_DIR = r"D:\Rag Model\rag\chroma_db"
TXT_PATH           = r"D:\Rag Model\rag\mnit data"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
LLM_MODEL          = "openai/gpt-oss-120b"

# ── Thread-safe shared state ────────────────────────────────────
STATE = {
    "ready":     False,
    "message":   "Initialising...",
    "error":     None,
    "rag_chain": None,
    "retriever": None,
}

# ── Pydantic models ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    file: str
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]

class StatusResponse(BaseModel):
    ready: bool
    message: str
    error: str | None = None

# ── RAG helpers ─────────────────────────────────────────────────

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def init_rag():
    """Initialise all RAG components in a background thread."""
    try:
        print("[*] Loading embedding model...", flush=True)
        STATE["message"] = "Loading embedding model..."

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[OK] Embeddings loaded", flush=True)

        if os.path.exists(CHROMA_PERSIST_DIR):
            print("[*] Loading existing ChromaDB...", flush=True)
            STATE["message"] = "Loading vector store..."
            vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings,
            )
        else:
            print("[*] Building ChromaDB from documents...", flush=True)
            STATE["message"] = "Building vector store..."
            loader = DirectoryLoader(
                TXT_PATH,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
            )
            docs   = loader.load()
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            ).split_documents(docs)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
            )
        print("[OK] Vector store ready", flush=True)

        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        print(f"[*] Initialising LLM: {LLM_MODEL}...", flush=True)
        STATE["message"] = "Initialising LLM..."
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=LLM_MODEL,
            temperature=0.2,
            max_tokens=1024,
        )
        print("[OK] LLM ready", flush=True)

        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant for MNIT Jaipur (Malaviya National Institute of Technology).
Use the following context to answer the question accurately and helpfully.
If the answer is not in the context, say so honestly.

Context:
{context}

Question: {question}

Answer:"""
        )

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Update state — set ready last so all fields are populated first
        STATE["rag_chain"] = rag_chain
        STATE["retriever"] = retriever
        STATE["message"]   = "Ready"
        STATE["ready"]     = True
        print("[OK] RAG system ready! Visit http://localhost:5000", flush=True)
        print("[*] API docs available at http://localhost:5000/docs", flush=True)

    except Exception as e:
        STATE["error"]   = str(e)
        STATE["message"] = f"Error: {e}"
        print(f"[ERR] RAG init failed: {e}", flush=True)
        traceback.print_exc()


# ── Lifespan (startup / shutdown) ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start RAG init in background thread on app startup."""
    if not GROQ_API_KEY:
        print("[ERR] GROQ_API_KEY not set in .env!", flush=True)
        STATE["message"] = "Error: GROQ_API_KEY not set"
        STATE["error"]   = "GROQ_API_KEY not set"
    else:
        t = threading.Thread(target=init_rag, daemon=True)
        t.start()
    yield  # app runs here
    print("[*] Shutting down MNIT RAG server...", flush=True)


# ── FastAPI app ─────────────────────────────────────────────────
app = FastAPI(
    title="MNIT RAG Assistant",
    description="RAG-based Q&A system for MNIT Jaipur powered by Groq + ChromaDB",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Routes ──────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def index():
    """Serve the chat UI."""
    return FileResponse("static/index.html")


@app.get("/api/status", response_model=StatusResponse, tags=["System"])
async def api_status():
    """Check if the RAG system is ready."""
    return StatusResponse(
        ready=STATE["ready"],
        message=STATE["message"],
        error=STATE["error"],
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def api_chat(body: ChatRequest):
    """
    Ask a question about MNIT Jaipur.
    Returns an answer and the source documents used.
    """
    if not STATE["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"System not ready yet: {STATE['message']}"
        )

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    rag_chain = STATE["rag_chain"]
    retriever = STATE["retriever"]

    try:
        answer      = rag_chain.invoke(question)
        source_docs = retriever.invoke(question)

        sources, seen = [], set()
        for doc in source_docs:
            src = os.path.basename(doc.metadata.get("source", "Unknown"))
            if src not in seen:
                seen.add(src)
                sources.append(SourceItem(
                    file=src,
                    snippet=doc.page_content[:150].strip(),
                ))

        return ChatResponse(answer=answer, sources=sources)

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/sources", tags=["System"])
async def api_sources():
    """Return list of all indexed knowledge-base files."""
    try:
        import chromadb
        client      = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collections = client.list_collections()
        files = set()
        for col in collections:
            items = col.get(include=["metadatas"])
            for m in items["metadatas"]:
                src = m.get("source", "")
                if src:
                    files.add(os.path.basename(src))
        return {"sources": sorted(files)}
    except Exception as exc:
        return {"sources": [], "error": str(exc)}


# ── Entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("[*] Starting MNIT RAG server at http://localhost:5000", flush=True)
    print("[*] API docs at http://localhost:5000/docs", flush=True)
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        log_level="warning",
    )
