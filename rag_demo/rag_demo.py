"""
RAG Demo using:
- LLM: Llama 3.3 70B via Groq API
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2) - Local
- Vector Store: ChromaDB
- Data Source: PDF files and/or plain TXT files (folder supported)
"""

import os
import sys
from dotenv import load_dotenv

# Fix Windows console Unicode encoding issues
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# LangChain imports
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set in .env file or environment
PDF_PATH = r"D:\Rag Model\rag\mnit data"  # Path to folder containing PDF(s)
TXT_PATH = r"D:\Rag Model\rag\mnit data"  # Path to folder containing TXT file(s) (can be same as PDF_PATH)
CHROMA_PERSIST_DIR = r"D:\Rag Model\rag\chroma_db"  # Directory to persist ChromaDB

# Embedding model (runs locally)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM model via Groq
LLM_MODEL = "openai/gpt-oss-120b"

# ============================================================
# 1. LOAD DOCUMENTS (PDF + TXT)
# ============================================================

def load_pdf(pdf_path: str):
    """Load all PDFs from a folder and return documents."""
    if not os.path.exists(pdf_path):
        print(f"⚠️  PDF path not found, skipping: {pdf_path}")
        return []
    print(f"📄 Loading PDFs from folder: {pdf_path}")
    loader = PyPDFDirectoryLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from {len(set(d.metadata.get('source','') for d in documents))} PDF file(s)")
    return documents

def load_txt(txt_path: str):
    """Load .txt file(s) — accepts a single file OR a folder of .txt files."""
    if not os.path.exists(txt_path):
        print(f"⚠️  TXT path not found, skipping: {txt_path}")
        return []

    # --- Single file ---
    if os.path.isfile(txt_path):
        print(f"📝 Loading single TXT file: {txt_path}")
        loader = TextLoader(txt_path, encoding="utf-8")
        documents = loader.load()
        print(f"✅ Loaded 1 TXT file")
        return documents

    # --- Directory of files ---
    print(f"📝 Loading TXT files from folder: {txt_path}")
    loader = DirectoryLoader(
        txt_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} TXT file(s)")
    return documents

# ============================================================
# 2. SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks for better retrieval."""
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# ============================================================
# 3. CREATE LOCAL EMBEDDINGS
# ============================================================

def get_embeddings():
    """Initialize local sentence-transformer embeddings."""
    print(f"🔢 Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ Embeddings loaded")
    return embeddings

# ============================================================
# 4. CREATE/LOAD CHROMADB VECTOR STORE
# ============================================================

def create_vector_store(chunks, embeddings):
    """Create ChromaDB vector store from document chunks."""
    print("🗄️ Creating ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    print("✅ Vector store created and persisted")
    return vectorstore

def load_vector_store(embeddings):
    """Load existing ChromaDB vector store."""
    print("🗄️ Loading existing ChromaDB vector store...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    print("✅ Vector store loaded")
    return vectorstore

# ============================================================
# 5. INITIALIZE GROQ LLM
# ============================================================

def get_llm():
    """Initialize Groq LLM with Llama 3.3 70B."""
    print(f"🤖 Initializing LLM: {LLM_MODEL} via Groq")
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.2,  
        max_tokens=1024,
    )
    print("✅ LLM initialized")
    return llm

# ============================================================
# 6. CREATE RAG CHAIN (Modern LCEL)
# ============================================================

def format_docs(docs):
    """Format retrieved documents into a string."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(vectorstore, llm):
    """Create a RAG chain for question answering using LCEL."""

    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3} 
    )

    prompt = ChatPromptTemplate.from_template(
        """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """
    )

    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG chain created")
    return rag_chain, retriever

# ============================================================
# 7. QUERY THE RAG SYSTEM
# ============================================================

def query_rag(rag_chain, retriever, question: str):
    """Query the RAG system and return answer with sources."""
    print(f"\n❓ Question: {question}")
    print("-" * 50)

    try:
        answer = rag_chain.invoke(question)
        source_docs = retriever.invoke(question)
    except Exception as e:
        print(f"❌ Error querying RAG system: {e}")
        return None, []

    print(f"\n💬 Answer: {answer}")
    print("\n📚 Sources:")
    for i, doc in enumerate(source_docs, 1):
        source = doc.metadata.get('source', 'N/A')
        page = doc.metadata.get('page', 'N/A')
        print(f"  [{i}] {source} | Page {page}: {doc.page_content[:100]}...")

    return answer, source_docs

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main function to run the RAG demo."""

    print("=" * 60)
    print("🚀 RAG DEMO - Groq + ChromaDB + Local Embeddings")
    print("=" * 60)

    # Check API key
    if not GROQ_API_KEY:
        print("❌ Error: GROQ_API_KEY not found!")
        print("   Set it in .env file or as environment variable")
        return

    # Initialize embeddings (local)
    embeddings = get_embeddings()

    # Check if vector store already exists
    if os.path.exists(CHROMA_PERSIST_DIR):
        print("\n📂 Found existing vector store")
        vectorstore = load_vector_store(embeddings)
    else:
        # Load PDFs and TXT files, then merge
        pdf_docs = load_pdf(PDF_PATH)
        txt_docs = load_txt(TXT_PATH)
        documents = pdf_docs + txt_docs

        if not documents:
            print("❌ Error: No documents found in PDF_PATH or TXT_PATH.")
            print("   Update PDF_PATH and/or TXT_PATH in the config section.")
            return

        chunks = split_documents(documents)
        vectorstore = create_vector_store(chunks, embeddings)

    # Initialize LLM
    llm = get_llm()

    # Create RAG chain
    rag_chain, retriever = create_rag_chain(vectorstore, llm)

    print("\n" + "=" * 60)
    print("💡 RAG System Ready! Enter your questions below.")
    print("   Type 'quit' to exit")
    print("=" * 60)

    # Interactive query loop
    while True:
        question = input("\n🔍 Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        if not question:
            continue

        query_rag(rag_chain, retriever, question)

if __name__ == "__main__":
    main()
