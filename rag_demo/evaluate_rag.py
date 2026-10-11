"""
RAG Evaluation Script
- Runs 20 predefined questions through the existing RAG pipeline
- Compares model answers with ground-truth answers extracted from the data files
- Saves a structured report: question | model output | real (ground-truth) answer
"""

import os
import json
import datetime
from dotenv import load_dotenv

# LangChain imports (same as rag_demo.py)
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ============================================================
# CONFIGURATION  (reuse same paths as rag_demo.py)
# ============================================================

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
DATA_PATH      = r"D:\Rag Model\rag\mnit data"
CHROMA_DIR     = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"
OUTPUT_FILE     = "rag_evaluation_report.md"

# ============================================================
# 20 QUESTIONS WITH GROUND-TRUTH ANSWERS
# (Drawn from different source files)
# ============================================================

QA_PAIRS = [
    # ── Academic Section ───────────────────────────────────────────────────────
    {
        "no": 1,
        "source_file": "Academic Section.txt",
        "question": "Who is the Dean of Academic Affairs at MNIT Jaipur?",
        "ground_truth": "Prof. D. Boolchandani is the Dean of Academic Affairs at MNIT Jaipur. "
                        "His email is dean.acad@mnit.ac.in and his phone numbers are +91-141-2715040 and 2713397."
    },
    {
        "no": 2,
        "source_file": "Academic Section.txt",
        "question": "Is internal branch sliding allowed for B.Tech students at MNIT Jaipur?",
        "ground_truth": "No. Internal sliding or change of allotted branch is not permitted after "
                        "admission in the First Year through CSAB."
    },
    {
        "no": 3,
        "source_file": "Academic Section.txt",
        "question": "How many courses can a student register for in a supplementary examination at MNIT?",
        "ground_truth": "A student can register for a maximum of 3 courses in a supplementary examination."
    },
    {
        "no": 4,
        "source_file": "Academic Section.txt",
        "question": "What is the CGPA conversion formula used at MNIT Jaipur?",
        "ground_truth": "MNIT Jaipur does not prescribe any official conversion formula. "
                        "However, for reference one can use: Percentage = CGPA × 10."
    },
    {
        "no": 5,
        "source_file": "Academic Section.txt",
        "question": "When is the Institute Assistantship of an M.Tech student terminated?",
        "ground_truth": "The Institute Assistantship of an M.Tech/M.Plan student is terminated "
                        "when the student attains CGPA less than 6 but above 5.5 in a semester."
    },
    # ── Administration ─────────────────────────────────────────────────────────
    {
        "no": 6,
        "source_file": "Administration.txt",
        "question": "Who is the current Director of MNIT Jaipur?",
        "ground_truth": "Prof. N. P. Padhy is the Director of MNIT Jaipur. "
                        "He can be reached at director@mnit.ac.in."
    },
    {
        "no": 7,
        "source_file": "Administration.txt",
        "question": "Who is the Chairman of the Board of Governors of MNIT Jaipur?",
        "ground_truth": "Ar. Habeeb Khan is the Chairman of the Board of Governors of MNIT Jaipur."
    },
    {
        "no": 8,
        "source_file": "Administration.txt",
        "question": "Who is the Dean of Research and Consultancy at MNIT Jaipur?",
        "ground_truth": "Prof. Lava Bhargava is the Dean of Research and Consultancy. "
                        "His email is dean.rnc@mnit.ac.in."
    },
    # ── CSE Department ─────────────────────────────────────────────────────────
    {
        "no": 9,
        "source_file": "CSE department.txt",
        "question": "When was the Department of Computer Science and Engineering established at MNIT Jaipur?",
        "ground_truth": "The Department of Computer Science and Engineering came into existence in 1994. "
                        "The first degree programme offered was B.E. (Computer Engineering) in affiliation "
                        "with the University of Rajasthan, and the first batch graduated in 1998."
    },
    {
        "no": 10,
        "source_file": "CSE department.txt",
        "question": "What programmes does the CSE department at MNIT Jaipur currently offer?",
        "ground_truth": "The CSE department currently offers: B.Tech (Computer Science & Engineering), "
                        "M.Tech (Computer Science & Engineering), M.Tech (Computer Science & Information "
                        "Security), and Ph.D."
    },
    {
        "no": 11,
        "source_file": "CSE department.txt",
        "question": "Who is the Head of the CSE Department at MNIT Jaipur?",
        "ground_truth": "Dr. Mushtaq Ahmed is the Head of the Department of Computer Science and Engineering "
                        "at MNIT Jaipur. His email is hod.cse@mnit.ac.in / mahmed.cse@mnit.ac.in."
    },
    # ── Cultural Society ───────────────────────────────────────────────────────
    {
        "no": 12,
        "source_file": "Cultural society.txt",
        "question": "What is Blitzschlag and when is it held?",
        "ground_truth": "Blitzschlag is the largest annual cultural festival of Rajasthan and the flagship "
                        "celebration of MNIT Jaipur. For 2026 it is scheduled from 6–8 February 2026. "
                        "It features competitions, debates, performances, and PRONITES."
    },
    {
        "no": 13,
        "source_file": "Cultural society.txt",
        "question": "How many active clubs does the Cultural Society of MNIT Jaipur have?",
        "ground_truth": "The Cultural Society of MNIT Jaipur hosts eighteen (18) active clubs that meet "
                        "regularly throughout the year."
    },
    # ── Hostel ─────────────────────────────────────────────────────────────────
    {
        "no": 14,
        "source_file": "hostel.txt",
        "question": "What is the total accommodation capacity for boys and girls in MNIT hostels?",
        "ground_truth": "The current accommodation capacity is 3,058 for boys and 1,509 for girls."
    },
    {
        "no": 15,
        "source_file": "hostel.txt",
        "question": "What is the name and capacity of the largest boys' hostel at MNIT Jaipur?",
        "ground_truth": "H9-Aurobindo is the largest boys' hostel with a capacity of 975 students."
    },
    # ── Sports ─────────────────────────────────────────────────────────────────
    {
        "no": 16,
        "source_file": "Sports.txt",
        "question": "What are the morning and evening timings for sports facilities at MNIT Jaipur?",
        "ground_truth": "Morning: 6:00 AM to 8:00 AM. Evening: 4:00 PM to 10:00 PM."
    },
    {
        "no": 17,
        "source_file": "Sports.txt",
        "question": "What is the mission statement of the Physical Education and Sports department at MNIT?",
        "ground_truth": 'The mission statement is "FIT MNIT FIT NATION" — an active, healthy and inclusive '
                        'campus where sport, physical activity, and active recreation provide a common platform '
                        'for participation, fun, and achievement.'
    },
    # ── Dispensary ─────────────────────────────────────────────────────────────
    {
        "no": 18,
        "source_file": "dispensary.txt",
        "question": "What are the dispensary timings at MNIT Jaipur on Sundays?",
        "ground_truth": "The dispensary is open on Sundays from 9:00 AM to 1:00 PM."
    },
    # ── Administration / General ───────────────────────────────────────────────
    {
        "no": 19,
        "source_file": "Administration.txt",
        "question": "Who is the Chief Vigilance Officer (CVO) at MNIT Jaipur?",
        "ground_truth": "Prof. Vijay Janyani, Professor-HAG in the Department of Electronics and "
                        "Communication Engineering, is the Chief Vigilance Officer. "
                        "His email is cvo@mnit.ac.in."
    },
    # ── Academic Section ───────────────────────────────────────────────────────
    {
        "no": 20,
        "source_file": "Academic Section.txt",
        "question": "Through which exam can a student take admission to M.Sc. programmes at MNIT Jaipur?",
        "ground_truth": "Admission to M.Sc. (Applied Chemistry, Applied Mathematics, and Applied Physics) "
                        "programmes at MNIT Jaipur is done through centralized counseling for M.Sc. admission "
                        "(CCMN) on the basis of IIT JAM score."
    },
]

# ============================================================
# HELPER FUNCTIONS  (copied / adapted from rag_demo.py)
# ============================================================

def load_txt(txt_path: str):
    print(f"📝 Loading TXT files from: {txt_path}")
    loader = DirectoryLoader(
        txt_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} TXT file(s)")
    return docs


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def get_embeddings():
    print(f"🔢 Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✅ Embeddings loaded")
    return embeddings


def get_or_create_vectorstore(embeddings):
    if os.path.exists(CHROMA_DIR):
        print("📂 Loading existing ChromaDB vector store …")
        vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        docs   = load_txt(DATA_PATH)
        chunks = split_documents(docs)
        print("🗄️  Creating ChromaDB vector store …")
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )
    print("✅ Vector store ready")
    return vs


def get_llm():
    print(f"🤖 Initialising LLM: {LLM_MODEL} via Groq")
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.2,
        max_tokens=512,
    )
    print("✅ LLM initialised")
    return llm


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    prompt = ChatPromptTemplate.from_template(
        """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know.

Context:
{context}

Question: {question}

Answer: """
    )
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ============================================================
# EVALUATION RUNNER
# ============================================================

def run_evaluation():
    print("=" * 65)
    print("🧪  RAG EVALUATION — MNIT Jaipur Knowledge Base")
    print("=" * 65)

    if not GROQ_API_KEY:
        print("❌  GROQ_API_KEY not found. Set it in .env and retry.")
        return

    embeddings   = get_embeddings()
    vectorstore  = get_or_create_vectorstore(embeddings)
    llm          = get_llm()
    rag_chain    = create_rag_chain(vectorstore, llm)

    results = []
    total   = len(QA_PAIRS)

    for item in QA_PAIRS:
        no       = item["no"]
        question = item["question"]
        gt       = item["ground_truth"]
        src      = item["source_file"]

        print(f"\n[{no:02d}/{total}] {question}")
        print("-" * 60)

        try:
            model_answer = rag_chain.invoke(question)
        except Exception as exc:
            model_answer = f"ERROR: {exc}"

        print(f"  Model:  {model_answer[:120]}{'…' if len(model_answer) > 120 else ''}")

        results.append({
            "no":           no,
            "source_file":  src,
            "question":     question,
            "model_output": model_answer,
            "ground_truth": gt,
        })

    # ── Write Markdown report ─────────────────────────────────
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# RAG Evaluation Report — MNIT Jaipur",
        f"",
        f"**Generated:** {now}  ",
        f"**LLM:** {LLM_MODEL}  ",
        f"**Embeddings:** {EMBEDDING_MODEL}  ",
        f"**Total Questions:** {total}",
        f"",
        "---",
        "",
    ]

    for r in results:
        lines += [
            f"## Q{r['no']:02d}. {r['question']}",
            f"",
            f"**Source file:** `{r['source_file']}`",
            f"",
            f"### 🤖 Model Output",
            f"",
            f"{r['model_output']}",
            f"",
            f"### ✅ Ground-Truth Answer",
            f"",
            f"{r['ground_truth']}",
            f"",
            "---",
            "",
        ]

    report_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"\n\n{'=' * 65}")
    print(f"✅  Evaluation complete!  Report saved to:")
    print(f"   {report_path}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    run_evaluation()
