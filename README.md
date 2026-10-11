# MNIT RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot for MNIT Jaipur — answers questions about hostels, placements, academics, departments, sports, and more.

Uses a hybrid search approach: local knowledge base (ChromaDB) combined with live official web search (Tavily restricted to mnit.ac.in).

![Tech Stack](https://img.shields.io/badge/LLM-Groq-orange) ![Embeddings](https://img.shields.io/badge/Embeddings-SentenceTransformers-blue) ![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-green) ![Framework](https://img.shields.io/badge/Backend-FastAPI-009688) ![WebSearch](https://img.shields.io/badge/WebSearch-Tavily-purple)

---

## Features

- **Hybrid Search** — Simultaneously searches local knowledge base AND live mnit.ac.in (official only)
- **Beautiful Web UI** — Dark-mode chat interface with markdown rendering and source citations
- **Two source types** — Local sources (teal chips) and official web sources (green clickable chips)
- **Fast LLM** — Powered by Groq API (`openai/gpt-oss-120b`)
- **Local Embeddings** — `all-MiniLM-L6-v2` runs fully on CPU, no paid embedding API needed
- **Persistent Vector Store** — ChromaDB auto-saves and reloads on restart
- **Auto API Docs** — Interactive Swagger UI at `/docs` (powered by FastAPI)
- **Official sources only** — Tavily restricted to `mnit.ac.in`, no third-party sites

---

## How It Works

```
User Question
     |
     |---- ChromaDB Search (local MNIT .txt files) ----+
     |                                                   |
     |---- Tavily Web Search (mnit.ac.in only) ---------+
                                                         |
                                              Combined Context
                                                         |
                                            Groq LLM (openai/gpt-oss-120b)
                                                         |
                                         Answer + Source Citations
```

Sources in the chat UI:
- **Teal chips** — local knowledge base files
- **Green chips** — official mnit.ac.in pages/PDFs (clickable, open URL)

---

## Project Structure

```
rag/
├── rag_demo/
│   ├── app.py              # FastAPI backend (hybrid RAG + Tavily)
│   ├── rag_demo.py         # CLI version (local KB only)
│   ├── requirements.txt    # Python dependencies
│   ├── .env.example        # API key template
│   ├── cmds.txt            # Quick command reference
│   ├── run_rag.bat         # Windows one-click launcher
│   └── static/
│       ├── index.html      # Chat UI
│       ├── style.css       # Dark-mode styles
│       └── app.js          # Frontend logic
├── mnit data/              # Local knowledge base (.txt files)
└── chroma_db/              # Auto-generated vector store (gitignored)
```

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/pulkit21gupta/MNIT-Rag-Based-Question-Answering-System.git
cd MNIT-Rag-Based-Question-Answering-System
```

### 2. Install dependencies
```bash
pip install -r rag_demo/requirements.txt
```

### 3. Set up API keys
```bash
copy rag_demo\.env.example rag_demo\.env
```
Open `rag_demo/.env` and fill in both keys:
```
GROQ_API_KEY="your_groq_key_here"       # from console.groq.com
TAVILY_API_KEY="your_tavily_key_here"   # from app.tavily.com
```

> Note: Tavily key is optional — the system still works using the local knowledge base if no Tavily key is provided.

### 4. Run the Web UI
```bash
cd rag_demo
python app.py
```
Open your browser:
- **http://localhost:5000** — Chat UI
- **http://localhost:5000/docs** — Interactive API docs (Swagger UI)

Wait ~20 seconds after starting for the RAG system to initialise.

### 5. (Optional) Run CLI version
```bash
python rag_demo.py
```

---

## Knowledge Base Topics

| File | Topic |
|------|-------|
| `Mnit hostels.txt` | Hostel facilities, capacity, rules |
| `Mnit placements rag.txt` | Placement stats, recruiters, packages |
| `academics.txt` | Departments, programs, academics |
| `aide department.txt` | AIDE (AI & Data Engineering) department |
| `mnit computer center.txt` | Computer centre, labs, IT staff, support contacts |
| `Mnit administration.txt` | Administration info |
| `banks at mnit.txt` | Banks and ATMs on campus |
| `mnit guest house.txt` | Guest house facilities |
| `mnit cultural society.txt` | Cultural clubs and events |
| `sports.txt` | Sports facilities and teams |
| `Mnit international alumni.txt` | International and alumni affairs |
| `Mnit post office.txt` | Post office details |

To add new topics, drop a `.txt` file in the `mnit data/` folder and delete `chroma_db/` to trigger a rebuild.

---

## Configuration

Edit these constants at the top of `app.py`:

```python
LLM_MODEL          = "openai/gpt-oss-120b"    # Groq model
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"        # Local embeddings
CHROMA_PERSIST_DIR = r"path\to\chroma_db"      # Vector store location
TXT_PATH           = r"path\to\mnit data"      # Knowledge base folder
TAVILY_MAX_RESULTS = 3                          # Web results per query
# Web search domain restriction:
include_domains    = ["mnit.ac.in"]             # Official MNIT only
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq API (`openai/gpt-oss-120b`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| Web Search | Tavily (restricted to `mnit.ac.in`) |
| Orchestration | LangChain (LCEL) |
| Backend | FastAPI + Uvicorn (ASGI) |
| Frontend | HTML / CSS / Vanilla JS |
| API Docs | Swagger UI (auto-generated at `/docs`) |

---

## Author

**Pulkit Gupta** — [github.com/pulkit21gupta](https://github.com/pulkit21gupta)

---

## License

MIT License — free to use, modify, and distribute.
