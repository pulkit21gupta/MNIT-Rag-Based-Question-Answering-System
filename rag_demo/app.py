"""
FastAPI Backend for MNIT RAG Assistant
Hybrid search: ChromaDB (local KB) + Tavily (live web) combined.
Auto-generated API docs: http://localhost:5000/docs
"""

import os
import sys
import traceback
import threading
import warnings
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Config ─────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY")
CHROMA_PERSIST_DIR = r"D:\Rag Model\rag\chroma_db"
TXT_PATH           = r"D:\Rag Model\rag\mnit data"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
LLM_MODEL          = "openai/gpt-oss-120b"
TAVILY_MAX_RESULTS = 3

# ── Shared state ────────────────────────────────────────────────
STATE = {
    "ready":      False,
    "message":    "Initialising...",
    "error":      None,
    "retriever":  None,
    "llm":        None,
    "tavily":     None,
}

# ── Pydantic models ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    file: str
    snippet: str
    type: str = "local"   # "local" | "web"
    url: str | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    local_results: int = 0
    web_results: int = 0

class StatusResponse(BaseModel):
    ready: bool
    message: str
    tavily_enabled: bool = False
    error: str | None = None

# ── RAG init ────────────────────────────────────────────────────

def init_rag():
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
                TXT_PATH, glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
            )
            docs   = loader.load()
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            ).split_documents(docs)
            vectorstore = Chroma.from_documents(
                documents=chunks, embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
            )
        print("[OK] Vector store ready", flush=True)

        STATE["retriever"] = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        print(f"[*] Initialising LLM: {LLM_MODEL}...", flush=True)
        STATE["message"] = "Initialising LLM..."
        STATE["llm"] = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=LLM_MODEL,
            temperature=0.2,
            max_tokens=1024,
        )
        print("[OK] LLM ready", flush=True)

        # Tavily setup
        if TAVILY_API_KEY:
            from tavily import TavilyClient
            STATE["tavily"] = TavilyClient(api_key=TAVILY_API_KEY)
            print("[OK] Tavily web search enabled", flush=True)
        else:
            print("[WARN] TAVILY_API_KEY not set — web search disabled", flush=True)

        STATE["message"] = "Ready"
        STATE["ready"]   = True
        print("[OK] RAG system ready! Visit http://localhost:5000", flush=True)
        print("[*] API docs at http://localhost:5000/docs", flush=True)

    except Exception as e:
        STATE["error"]   = str(e)
        STATE["message"] = f"Error: {e}"
        print(f"[ERR] RAG init failed: {e}", flush=True)
        traceback.print_exc()

# ── Search helpers ───────────────────────────────────────────────

def search_local(question: str) -> list[SourceItem]:
    """Search ChromaDB local knowledge base."""
    try:
        docs = STATE["retriever"].invoke(question)
        sources, seen = [], set()
        for doc in docs:
            src = os.path.basename(doc.metadata.get("source", "Unknown"))
            if src not in seen:
                seen.add(src)
                sources.append(SourceItem(
                    file=src,
                    snippet=doc.page_content[:150].strip(),
                    type="local",
                ))
        return docs, sources
    except Exception as e:
        print(f"[WARN] Local search error: {e}", flush=True)
        return [], []


def search_web(question: str) -> tuple[str, list[SourceItem]]:
    """Search the official MNIT website via Tavily (mnit.ac.in only)."""
    if not STATE["tavily"]:
        return "", []
    try:
        results = STATE["tavily"].search(
            query=question,
            max_results=TAVILY_MAX_RESULTS,
            search_depth="basic",
            include_domains=["mnit.ac.in"],   # Official MNIT website only
        )
        context_parts = []
        sources = []
        for r in results.get("results", []):
            title   = r.get("title", "")
            url     = r.get("url", "")
            content = r.get("content", "")
            if content:
                context_parts.append(f"[Web: {title}]\n{content}")
                sources.append(SourceItem(
                    file=title or url,
                    snippet=content[:150].strip(),
                    type="web",
                    url=url,
                ))
        return "\n\n".join(context_parts), sources
    except Exception as e:
        print(f"[WARN] Tavily search error: {e}", flush=True)
        return "", []


def build_hybrid_answer(question: str, local_docs, web_context: str) -> str:
    """Build a combined answer from local + web context."""
    local_context = "\n\n".join(doc.page_content for doc in local_docs)

    has_local = bool(local_context.strip())
    has_web   = bool(web_context.strip())

    if has_local and has_web:
        combined = (
            f"--- MNIT Knowledge Base ---\n{local_context}\n\n"
            f"--- Web Search Results ---\n{web_context}"
        )
        source_note = "Use information from both the MNIT knowledge base AND the web search results."
    elif has_local:
        combined    = local_context
        source_note = "Use information from the MNIT knowledge base."
    elif has_web:
        combined    = web_context
        source_note = "Use information from the web search results."
    else:
        return "I could not find relevant information from either the knowledge base or the web."

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant for MNIT Jaipur. {source_note}
Provide a comprehensive, accurate answer. If sources conflict, prefer the MNIT knowledge base.

Context:
{context}

Question: {question}

Answer:"""
    )

    chain = prompt | STATE["llm"] | StrOutputParser()
    return chain.invoke({
        "context":     combined,
        "question":    question,
        "source_note": source_note,
    })

# ── Lifespan ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not GROQ_API_KEY:
        STATE["message"] = "Error: GROQ_API_KEY not set"
        STATE["error"]   = "GROQ_API_KEY not set"
    else:
        threading.Thread(target=init_rag, daemon=True).start()
    yield
    print("[*] Shutting down...", flush=True)

# ── FastAPI app ──────────────────────────────────────────────────

app = FastAPI(
    title="MNIT RAG Assistant",
    description="Hybrid RAG: ChromaDB local knowledge base + Tavily live web search, powered by Groq LLM",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Routes ───────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("static/index.html")


@app.get("/api/status", response_model=StatusResponse, tags=["System"])
async def api_status():
    return StatusResponse(
        ready=STATE["ready"],
        message=STATE["message"],
        tavily_enabled=STATE["tavily"] is not None,
        error=STATE["error"],
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def api_chat(body: ChatRequest):
    """
    Ask a question. Searches both the MNIT knowledge base AND the web (Tavily),
    then combines both results into one comprehensive answer.
    """
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail=f"System not ready: {STATE['message']}")

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Run local + web search in parallel
        local_docs, local_sources = [], []
        web_context, web_sources  = "", []

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_local = executor.submit(search_local, question)
            future_web   = executor.submit(search_web, question)

            local_docs, local_sources = future_local.result()
            web_context, web_sources  = future_web.result()

        # Build combined answer
        answer  = build_hybrid_answer(question, local_docs, web_context)
        sources = local_sources + web_sources

        return ChatResponse(
            answer=answer,
            sources=sources,
            local_results=len(local_sources),
            web_results=len(web_sources),
        )

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/sources", tags=["System"])
async def api_sources():
    """Return list of all indexed local knowledge-base files."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        files  = set()
        for col in client.list_collections():
            items = col.get(include=["metadatas"])
            for m in items["metadatas"]:
                src = m.get("source", "")
                if src:
                    files.add(os.path.basename(src))
        return {"sources": sorted(files), "tavily_enabled": STATE["tavily"] is not None}
    except Exception as exc:
        return {"sources": [], "error": str(exc)}


# ── Entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("[*] Starting MNIT RAG server (Hybrid: Local + Tavily Web Search)", flush=True)
    print("[*] Chat UI  : http://localhost:5000", flush=True)
    print("[*] API Docs : http://localhost:5000/docs", flush=True)
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False, log_level="warning")
