# RAG Demo 🚀

A Retrieval-Augmented Generation (RAG) demo using:
- **LLM**: Llama 3.3 70B via [Groq API](https://console.groq.com/)
- **Embeddings**: `all-MiniLM-L6-v2` via Sentence Transformers (runs **locally**)
- **Vector Store**: ChromaDB (persistent)
- **Document Source**: PDF files

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure your API key
```bash
# Copy the example env file
copy .env.example .env
```
Then open `.env` and replace `your_groq_api_key_here` with your actual key from [console.groq.com](https://console.groq.com/).

### 3. Set your PDF path
Open `rag_demo.py` and update the `PDF_PATH` variable:
```python
PDF_PATH = r"C:\path\to\your\document.pdf"
```

### 4. Run
```bash
python rag_demo.py
```

---

## How It Works

```
PDF → Chunks → Local Embeddings → ChromaDB
                                      ↓
          Question → Retriever (top 3 chunks) → Groq LLM → Answer
```

1. **Load** - PDF is loaded and split into overlapping chunks (1000 chars, 200 overlap)
2. **Embed** - Each chunk is embedded locally using `all-MiniLM-L6-v2`
3. **Store** - Embeddings stored in ChromaDB (persisted to `./chroma_db`)
4. **Retrieve** - On each question, top 3 relevant chunks are fetched
5. **Generate** - Llama 3.3 70B answers using retrieved context

> **Note**: On second run, the existing ChromaDB vector store is reused automatically — no need to re-embed the PDF.

---

## Project Structure

```
rag_demo/
├── rag_demo.py        # Main script
├── requirements.txt   # Dependencies
├── .env.example       # API key template
├── .env               # Your actual API key (gitignored)
└── chroma_db/         # Auto-created vector store
```
