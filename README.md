# MNIT RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot for MNIT Jaipur — answers questions about hostels, placements, academics, departments, sports, and more from a curated knowledge base.

![Tech Stack](https://img.shields.io/badge/LLM-Groq-orange) ![Embeddings](https://img.shields.io/badge/Embeddings-SentenceTransformers-blue) ![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-green) ![Framework](https://img.shields.io/badge/Backend-FastAPI-009688)

---

## Features

- **Beautiful Web UI** — Dark-mode chat interface with markdown rendering and source citations
- **RAG Pipeline** — Retrieves relevant chunks before answering (no hallucinations)
- **Fast LLM** — Powered by Groq API (`openai/gpt-oss-120b`)
- **Local Embeddings** — `all-MiniLM-L6-v2` runs fully on CPU, no paid embedding API needed
- **Persistent Vector Store** — ChromaDB auto-saves and reloads on restart
- **Expandable Knowledge Base** — Drop new `.txt` files in the data folder to add topics
- **Auto API Docs** — Interactive Swagger UI available at `/docs` (powered by FastAPI)

---

## Project Structure

```
rag/
├── rag_demo/
│   ├── app.py              # Flask backend (Web UI + REST API)
│   ├── rag_demo.py         # CLI version (terminal chat)
│   ├── requirements.txt    # Python dependencies
│   ├── .env.example        # API key template
│   ├── cmds.txt            # Quick command reference
│   ├── run_rag.bat         # Windows batch launcher
│   └── static/
│       ├── index.html      # Chat UI
│       ├── style.css       # Dark-mode styles
│       └── app.js          # Frontend logic
├── mnit data/              # Knowledge base (.txt files)
│   ├── Mnit hostels.txt
│   ├── Mnit placements rag.txt
│   ├── academics.txt
│   └── ...
└── chroma_db/              # Auto-generated (gitignored)
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

### 3. Set up your API key
```bash
copy rag_demo\.env.example rag_demo\.env
```
Open `rag_demo/.env` and add your key from [console.groq.com](https://console.groq.com/):
```
GROQ_API_KEY="your_groq_api_key_here"
```

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

## How It Works

```
Knowledge Base (.txt files)
        |
  Text Splitting (1000 char chunks, 200 overlap)
        |
  Local Embeddings (all-MiniLM-L6-v2)
        |
  ChromaDB Vector Store  <-- persisted on disk
        |
User Question --> Semantic Search (top 4 chunks)
        |
  Groq LLM (openai/gpt-oss-120b) + Retrieved Context
        |
     Answer + Source Citations
```

---

## Knowledge Base Topics

| File | Topic |
|------|-------|
| `Mnit hostels.txt` | Hostel facilities, capacity, rules |
| `Mnit placements rag.txt` | Placement stats, recruiters, packages |
| `academics.txt` | Departments, programs, academics |
| `aide department.txt` | AIDE department details |
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

Edit these constants at the top of `app.py` or `rag_demo.py`:

```python
LLM_MODEL          = "openai/gpt-oss-120b"    # Groq model
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"        # Local embeddings
CHROMA_PERSIST_DIR = r"path\to\chroma_db"      # Vector store location
TXT_PATH           = r"path\to\mnit data"      # Knowledge base folder
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq API (`openai/gpt-oss-120b`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
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
