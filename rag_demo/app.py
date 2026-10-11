"""
Flask Backend for MNIT RAG Assistant
Serves the chat UI and exposes /api/chat, /api/status, /api/sources endpoints.
"""

import os
import sys
import traceback
import threading
import warnings

# Fix Windows console Unicode encoding FIRST
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
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

# ── Thread-safe shared state (mutable dict, not bare globals) ───
STATE = {
    "ready":       False,
    "message":     "Initialising...",
    "error":       None,
    "rag_chain":   None,
    "retriever":   None,
}

# ── Flask app ───────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="static")
CORS(app)

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

        # Load or create vector store
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

        # Update state atomically
        STATE["rag_chain"] = rag_chain
        STATE["retriever"] = retriever
        STATE["message"]   = "Ready"
        STATE["ready"]     = True   # set last so other fields are populated first
        print("[OK] RAG system ready! Visit http://localhost:5000", flush=True)

    except Exception as e:
        STATE["error"]   = str(e)
        STATE["message"] = f"Error: {e}"
        print(f"[ERR] RAG init failed: {e}", flush=True)
        traceback.print_exc()


# ── Routes ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "ready":   STATE["ready"],
        "message": STATE["message"],
        "error":   STATE["error"],
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    if not STATE["ready"]:
        return jsonify({"error": f"System not ready: {STATE['message']}"}), 503

    data     = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Empty question."}), 400

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
                sources.append({
                    "file":    src,
                    "snippet": doc.page_content[:150].strip(),
                })

        return jsonify({"answer": answer, "sources": sources})

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/sources")
def api_sources():
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
        return jsonify({"sources": sorted(files)})
    except Exception as exc:
        return jsonify({"sources": [], "error": str(exc)})


# ── Entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    if not GROQ_API_KEY:
        print("[ERR] GROQ_API_KEY not set in .env!")
        sys.exit(1)

    # Start RAG init in background thread
    t = threading.Thread(target=init_rag, daemon=True)
    t.start()

    print("[*] Starting MNIT RAG server at http://localhost:5000", flush=True)
    # use_reloader=False is critical — reloader kills daemon threads on Windows
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
