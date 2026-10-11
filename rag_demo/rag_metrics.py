"""
RAG Performance Metrics Dashboard
===================================
Evaluates the 20 RAG question-answers using:
  • ROUGE-1, ROUGE-2, ROUGE-L  (recall-oriented overlap)
  • BLEU-1, BLEU-2             (precision-oriented n-gram overlap)
  • Keyword Match %             (simple factual keyword coverage)
  • Answer Presence Flag        (did the model answer vs. "I don't know")

Outputs:
  1. rag_metrics_report.md   – per-question table + summary stats
  2. rag_metrics_charts.png  – 4-panel matplotlib dashboard
"""

import os, re, string
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Download NLTK tokeniser data quietly
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA: 20 question + model output + ground truth
# (copied directly from the evaluation run)
# ─────────────────────────────────────────────────────────────────────────────

QA_DATA = [
    {
        "no": 1,
        "source": "Academic Section",
        "question": "Who is the Dean of Academic Affairs at MNIT Jaipur?",
        "model":
            "Prof. D. Boolchandani is the Dean of Academic Affairs at MNIT Jaipur.",
        "ground_truth":
            "Prof. D. Boolchandani is the Dean of Academic Affairs at MNIT Jaipur. "
            "His email is dean.acad@mnit.ac.in and his phone numbers are +91-141-2715040 and 2713397.",
        "keywords": ["D. Boolchandani", "Dean", "Academic Affairs"],
    },
    {
        "no": 2,
        "source": "Academic Section",
        "question": "Is internal branch sliding allowed for B.Tech students at MNIT Jaipur?",
        "model":
            "No, internal sliding/change in allotted branch is not permitted after admission in the First Year through CSAB.",
        "ground_truth":
            "No. Internal sliding or change of allotted branch is not permitted after admission in the First Year through CSAB.",
        "keywords": ["No", "not permitted", "CSAB"],
    },
    {
        "no": 3,
        "source": "Academic Section",
        "question": "How many courses can a student register for in a supplementary examination at MNIT?",
        "model":
            "A student can register for a maximum of 3 courses in a supplementary examination.",
        "ground_truth":
            "A student can register for a maximum of 3 courses in a supplementary examination.",
        "keywords": ["3", "maximum", "supplementary"],
    },
    {
        "no": 4,
        "source": "Academic Section",
        "question": "What is the CGPA conversion formula used at MNIT Jaipur?",
        "model":
            "The CGPA conversion formula used at MNIT Jaipur for reference is: Percentage = CGPA x 10. "
            "However, MNIT Jaipur does not prescribe any official conversion formula.",
        "ground_truth":
            "MNIT Jaipur does not prescribe any official conversion formula. "
            "However, for reference one can use: Percentage = CGPA × 10.",
        "keywords": ["CGPA", "10", "Percentage", "formula"],
    },
    {
        "no": 5,
        "source": "Academic Section",
        "question": "When is the Institute Assistantship of an M.Tech student terminated?",
        "model":
            "The Institute Assistantship of an M.Tech. student is terminated when he/she attains a CGPA less than 6 but above 5.5 in a semester.",
        "ground_truth":
            "The Institute Assistantship of an M.Tech/M.Plan student is terminated when the student attains CGPA less than 6 but above 5.5 in a semester.",
        "keywords": ["CGPA", "6", "5.5", "terminated"],
    },
    {
        "no": 6,
        "source": "Administration",
        "question": "Who is the current Director of MNIT Jaipur?",
        "model":
            "Prof. N. P. Padhy is the current Director of MNIT Jaipur.",
        "ground_truth":
            "Prof. N. P. Padhy is the Director of MNIT Jaipur. He can be reached at director@mnit.ac.in.",
        "keywords": ["N. P. Padhy", "Director"],
    },
    {
        "no": 7,
        "source": "Administration",
        "question": "Who is the Chairman of the Board of Governors of MNIT Jaipur?",
        "model":
            "Ar. Habeeb Khan is the Chairman of the Board of Governors of Malaviya National Institute of Technology Jaipur.",
        "ground_truth":
            "Ar. Habeeb Khan is the Chairman of the Board of Governors of MNIT Jaipur.",
        "keywords": ["Habeeb Khan", "Chairman", "Board of Governors"],
    },
    {
        "no": 8,
        "source": "Administration",
        "question": "Who is the Dean of Research and Consultancy at MNIT Jaipur?",
        "model":
            "Prof. Lava Bhargava is the Dean of Research & Consultancy at MNIT Jaipur.",
        "ground_truth":
            "Prof. Lava Bhargava is the Dean of Research and Consultancy. His email is dean.rnc@mnit.ac.in.",
        "keywords": ["Lava Bhargava", "Research", "Consultancy"],
    },
    {
        "no": 9,
        "source": "CSE Department",
        "question": "When was the Department of Computer Science and Engineering established at MNIT Jaipur?",
        "model":
            "I don't know. The provided context does not mention the establishment of the Department of Computer Science and Engineering at MNIT Jaipur.",
        "ground_truth":
            "The Department of Computer Science and Engineering came into existence in 1994. "
            "The first batch graduated in 1998.",
        "keywords": ["1994", "Computer Science", "Engineering"],
    },
    {
        "no": 10,
        "source": "CSE Department",
        "question": "What programmes does the CSE department at MNIT Jaipur currently offer?",
        "model":
            "I don't know. The provided context only mentions the programmes offered by the ECE department.",
        "ground_truth":
            "The CSE department currently offers: B.Tech (Computer Science & Engineering), "
            "M.Tech (Computer Science & Engineering), M.Tech (Computer Science & Information Security), and Ph.D.",
        "keywords": ["B.Tech", "M.Tech", "Ph.D", "CSE"],
    },
    {
        "no": 11,
        "source": "CSE Department",
        "question": "Who is the Head of the CSE Department at MNIT Jaipur?",
        "model":
            "I don't know. The provided context does not mention the Head of the CSE Department at MNIT Jaipur.",
        "ground_truth":
            "Dr. Mushtaq Ahmed is the Head of the Department of Computer Science and Engineering at MNIT Jaipur.",
        "keywords": ["Mushtaq Ahmed", "Head", "CSE"],
    },
    {
        "no": 12,
        "source": "Cultural Society",
        "question": "What is Blitzschlag and when is it held?",
        "model":
            "Blitzschlag is the largest annual cultural festival of Rajasthan and the flagship celebration of MNIT Jaipur. It is held from 6-8 February 2026.",
        "ground_truth":
            "Blitzschlag is the largest annual cultural festival of Rajasthan and the flagship celebration of "
            "MNIT Jaipur. For 2026 it is scheduled from 6–8 February 2026.",
        "keywords": ["Blitzschlag", "cultural festival", "February 2026"],
    },
    {
        "no": 13,
        "source": "Cultural Society",
        "question": "How many active clubs does the Cultural Society of MNIT Jaipur have?",
        "model":
            "The Cultural Society of MNIT Jaipur has 18 active clubs that meet regularly throughout the year.",
        "ground_truth":
            "The Cultural Society of MNIT Jaipur hosts eighteen (18) active clubs that meet regularly throughout the year.",
        "keywords": ["18", "eighteen", "clubs"],
    },
    {
        "no": 14,
        "source": "Hostel",
        "question": "What is the total accommodation capacity for boys and girls in MNIT hostels?",
        "model":
            "The total accommodation capacity for boys is 3058 and for girls is 1509.",
        "ground_truth":
            "The current accommodation capacity is 3,058 for boys and 1,509 for girls.",
        "keywords": ["3058", "1509", "boys", "girls"],
    },
    {
        "no": 15,
        "source": "Hostel",
        "question": "What is the name and capacity of the largest boys' hostel at MNIT Jaipur?",
        "model":
            "The name of the largest boys' hostel at MNIT Jaipur is H9-AUROBINDO, and its capacity is 975.",
        "ground_truth":
            "H9-Aurobindo is the largest boys' hostel with a capacity of 975 students.",
        "keywords": ["Aurobindo", "H9", "975"],
    },
    {
        "no": 16,
        "source": "Sports",
        "question": "What are the morning and evening timings for sports facilities at MNIT Jaipur?",
        "model":
            "The morning timings for sports facilities at MNIT Jaipur are 6:00 AM to 8:00 AM, and the evening timings are 4:00 PM to 10:00 PM.",
        "ground_truth":
            "Morning: 6:00 AM to 8:00 AM. Evening: 4:00 PM to 10:00 PM.",
        "keywords": ["6:00 AM", "8:00 AM", "4:00 PM", "10:00 PM"],
    },
    {
        "no": 17,
        "source": "Sports",
        "question": "What is the mission statement of the Physical Education and Sports department at MNIT?",
        "model":
            'The mission statement of the Physical Education and Sports department at MNIT is "FIT MNIT FIT NATION".',
        "ground_truth":
            'The mission statement is "FIT MNIT FIT NATION" — an active, healthy and inclusive campus.',
        "keywords": ["FIT MNIT FIT NATION", "mission"],
    },
    {
        "no": 18,
        "source": "Dispensary",
        "question": "What are the dispensary timings at MNIT Jaipur on Sundays?",
        "model":
            "I don't know. The context does not provide information about the dispensary timings.",
        "ground_truth":
            "The dispensary is open on Sundays from 9:00 AM to 1:00 PM.",
        "keywords": ["9:00 AM", "1:00 PM", "Sunday"],
    },
    {
        "no": 19,
        "source": "Administration",
        "question": "Who is the Chief Vigilance Officer (CVO) at MNIT Jaipur?",
        "model":
            "Prof. Vijay Janyani, Professor-HAG, Dept. of Electronics and Communication Engg., can be contacted at cvo@mnit.ac.in.",
        "ground_truth":
            "Prof. Vijay Janyani, Professor-HAG in the Department of Electronics and Communication Engineering, is the Chief Vigilance Officer. His email is cvo@mnit.ac.in.",
        "keywords": ["Vijay Janyani", "CVO", "cvo@mnit.ac.in"],
    },
    {
        "no": 20,
        "source": "Academic Section",
        "question": "Through which exam can a student take admission to M.Sc. programmes at MNIT Jaipur?",
        "model":
            "Through IIT JAM score, as part of the centralized counseling for M.Sc. admission (CCMN).",
        "ground_truth":
            "Admission to M.Sc. programmes at MNIT Jaipur is done through centralized counseling (CCMN) on the basis of IIT JAM score.",
        "keywords": ["IIT JAM", "CCMN", "M.Sc"],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
smoother = SmoothingFunction().method1

def normalise(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenise(text: str):
    return nltk.word_tokenize(normalise(text))

def bleu_scores(hypothesis: str, reference: str):
    hyp = tokenise(hypothesis)
    ref = tokenise(reference)
    if not hyp:
        return 0.0, 0.0
    b1 = sentence_bleu([ref], hyp, weights=(1, 0, 0, 0), smoothing_function=smoother)
    b2 = sentence_bleu([ref], hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
    return round(b1, 4), round(b2, 4)

def rouge_scores(hypothesis: str, reference: str):
    scores = scorer.score(reference, hypothesis)
    return (
        round(scores["rouge1"].fmeasure, 4),
        round(scores["rouge2"].fmeasure, 4),
        round(scores["rougeL"].fmeasure, 4),
    )

def keyword_match(hypothesis: str, keywords: list) -> float:
    h_lower = hypothesis.lower()
    matched = sum(1 for kw in keywords if kw.lower() in h_lower)
    return round(matched / len(keywords), 4) if keywords else 0.0

def did_answer(model_output: str) -> int:
    """1 if the model gave an answer, 0 if it said 'I don't know'."""
    return 0 if "don't know" in model_output.lower() or "i don't know" in model_output.lower() else 1

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE METRICS
# ─────────────────────────────────────────────────────────────────────────────

rows = []
for item in QA_DATA:
    r1, r2, rl = rouge_scores(item["model"], item["ground_truth"])
    b1, b2     = bleu_scores(item["model"], item["ground_truth"])
    kw         = keyword_match(item["model"], item["keywords"])
    answered   = did_answer(item["model"])
    rows.append({
        "Q#":            item["no"],
        "Source":        item["source"],
        "Question":      item["question"],
        "ROUGE-1":       r1,
        "ROUGE-2":       r2,
        "ROUGE-L":       rl,
        "BLEU-1":        b1,
        "BLEU-2":        b2,
        "Keyword Match": kw,
        "Answered":      answered,
    })

df = pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────────────────────────────────────

metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU-1", "BLEU-2", "Keyword Match"]
summary = df[metrics].describe().loc[["mean", "std", "min", "max"]].round(4)

answer_rate = df["Answered"].mean() * 100          # %
correct_rate = (df["Keyword Match"] >= 0.67).mean() * 100   # ≥2/3 keywords

# ─────────────────────────────────────────────────────────────────────────────
# CHART — 4-panel dashboard
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG   = "#1a1a2e"
ACCENT1   = "#e94560"
ACCENT2   = "#0f3460"
ACCENT3   = "#16213e"
PALE      = "#a8dadc"
TEXT      = "#eaeaea"

sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    ACCENT3,
    "axes.labelcolor":   TEXT,
    "axes.edgecolor":    PALE,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        "#2e2e4a",
    "font.family":       "DejaVu Sans",
})

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("🤖  RAG Model — Performance Metrics Dashboard\nMNIT Jaipur Knowledge Base  |  20 Questions",
             fontsize=16, color=TEXT, fontweight="bold", y=0.98)

palette = sns.color_palette("cool", 6)

# --- Panel 1: ROUGE & BLEU per question (line chart) -------------------------
ax1 = axes[0, 0]
q_nums = df["Q#"].astype(str)
for i, col in enumerate(["ROUGE-1", "ROUGE-L", "BLEU-1", "Keyword Match"]):
    ax1.plot(q_nums, df[col], marker="o", linewidth=2,
             label=col, color=palette[i], markersize=5)
ax1.set_title("Per-Question Metric Scores", color=TEXT, fontweight="bold")
ax1.set_xlabel("Question #", color=TEXT)
ax1.set_ylabel("Score (0–1)", color=TEXT)
ax1.set_ylim(-0.05, 1.1)
ax1.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT)
ax1.tick_params(axis="x", rotation=45)

# --- Panel 2: Average metrics bar chart (horizontal) -------------------------
ax2 = axes[0, 1]
avg_vals = summary.loc["mean"]
colors   = sns.color_palette("magma", len(metrics))
bars = ax2.barh(metrics, avg_vals.values, color=colors, edgecolor=PALE, linewidth=0.5)
ax2.set_xlim(0, 1)
ax2.set_title("Average Score per Metric", color=TEXT, fontweight="bold")
ax2.set_xlabel("Score (0–1)", color=TEXT)
for bar, val in zip(bars, avg_vals.values):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=9, color=TEXT)

# --- Panel 3: Per-source ROUGE-1 box / swarm plot ---------------------------
ax3 = axes[1, 0]
sources = df["Source"].unique()
src_palette = sns.color_palette("husl", len(sources))
for i, src in enumerate(sorted(sources)):
    vals = df[df["Source"] == src]["ROUGE-1"].values
    x_positions = [i] * len(vals)
    ax3.scatter(x_positions, vals, color=src_palette[i], s=70, zorder=3, alpha=0.9)
    ax3.boxplot(vals, positions=[i], widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor=src_palette[i] + (0.2,), color=src_palette[i]),
                medianprops=dict(color=ACCENT1, linewidth=2),
                whiskerprops=dict(color=src_palette[i]),
                capprops=dict(color=src_palette[i]),
                flierprops=dict(marker="x", color=ACCENT1))
ax3.set_xticks(range(len(sorted(sources))))
ax3.set_xticklabels(sorted(sources), rotation=30, ha="right", fontsize=8)
ax3.set_title("ROUGE-1 Distribution by Source File", color=TEXT, fontweight="bold")
ax3.set_ylabel("ROUGE-1 F1", color=TEXT)
ax3.set_ylim(-0.05, 1.1)

# --- Panel 4: Pie — Answer Rate + Keyword accuracy donut --------------------
ax4 = axes[1, 1]
answered_count    = int(df["Answered"].sum())
not_answered_count = 20 - answered_count
kw_high = int((df["Keyword Match"] >= 0.67).sum())
kw_low  = 20 - kw_high

sizes1   = [answered_count, not_answered_count]
sizes2   = [kw_high, kw_low]
labels1  = [f"Answered\n({answered_count})", f"'Don't Know'\n({not_answered_count})"]
labels2  = [f"≥67% KW\n({kw_high})", f"<67% KW\n({kw_low})"]
colors1  = ["#06d6a0", "#ef233c"]
colors2  = ["#48cae4", "#e9c46a"]

# Outer ring
wedges1, _ = ax4.pie(sizes1, radius=1.0, labels=labels1,
                     colors=colors1, startangle=90,
                     wedgeprops=dict(width=0.35, edgecolor=DARK_BG, linewidth=2),
                     textprops=dict(color=TEXT, fontsize=9))
# Inner ring
wedges2, _ = ax4.pie(sizes2, radius=0.62, labels=labels2,
                     colors=colors2, startangle=90,
                     wedgeprops=dict(width=0.35, edgecolor=DARK_BG, linewidth=2),
                     textprops=dict(color=TEXT, fontsize=8))
ax4.text(0, 0, f"{answer_rate:.0f}%\nAns.", ha="center", va="center",
         fontsize=11, color=TEXT, fontweight="bold")
ax4.set_title("Answer Rate (outer) & Keyword Match ≥67% (inner)",
              color=TEXT, fontweight="bold", pad=12)

plt.tight_layout(rect=[0, 0, 1, 0.96])

chart_path  = os.path.join(os.path.dirname(__file__), "rag_metrics_charts.png")
report_path = os.path.join(os.path.dirname(__file__), "rag_metrics_report.md")

plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
print(f"✅ Chart saved → {chart_path}")

# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN REPORT
# ─────────────────────────────────────────────────────────────────────────────

lines = [
    "# RAG Performance Metrics Report — MNIT Jaipur",
    "",
    "**Model:** `llama-3.3-70b-versatile` via Groq  ",
    "**Embeddings:** `all-MiniLM-L6-v2` (local)  ",
    "**Vector Store:** ChromaDB  ",
    "**Total Questions:** 20 (7 source files)  ",
    "",
    "---",
    "",
    "## 📊 Summary Statistics",
    "",
    "| Metric | Mean | Std Dev | Min | Max |",
    "|--------|------|---------|-----|-----|",
]

for m in metrics:
    r = summary[m]
    lines.append(f"| **{m}** | {r['mean']:.3f} | {r['std']:.3f} | {r['min']:.3f} | {r['max']:.3f} |")

lines += [
    "",
    f"| **Answer Rate** | {answer_rate:.1f}% | — | — | — |",
    f"| **Keyword Match ≥67%** | {correct_rate:.1f}% | — | — | — |",
    "",
    "---",
    "",
    "## 📋 Per-Question Breakdown",
    "",
    "| Q# | Source | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-1 | BLEU-2 | KW-Match | Answered |",
    "|----|--------|---------|---------|---------|--------|--------|----------|----------|",
]

for _, r in df.iterrows():
    ans_icon = "✅" if r["Answered"] else "❌"
    kw_str   = f"{r['Keyword Match']:.2f}"
    lines.append(
        f"| {int(r['Q#'])} | {r['Source']} "
        f"| {r['ROUGE-1']:.3f} | {r['ROUGE-2']:.3f} | {r['ROUGE-L']:.3f} "
        f"| {r['BLEU-1']:.3f} | {r['BLEU-2']:.3f} "
        f"| {kw_str} | {ans_icon} |"
    )

lines += [
    "",
    "---",
    "",
    "## 📈 Metrics Visualisation",
    "",
    f"![RAG Performance Dashboard](rag_metrics_charts.png)",
    "",
    "---",
    "",
    "## 🔍 Key Observations",
    "",
    f"- **{answer_rate:.0f}%** of questions received a substantive answer ({answered_count}/20).",
    f"- **{correct_rate:.0f}%** of answers covered ≥67% of expected keywords ({kw_high}/20).",
    f"- **ROUGE-1 avg: {summary['ROUGE-1']['mean']:.3f}** — good unigram overlap with ground truth.",
    f"- **ROUGE-2 avg: {summary['ROUGE-2']['mean']:.3f}** — moderate bigram consistency.",
    f"- **BLEU-1 avg:  {summary['BLEU-1']['mean']:.3f}** — acceptable precision on single tokens.",
    "",
    "### Failure Analysis",
    "- **CSE Dept questions (Q9, Q10, Q11):** Retriever fetched ECE chunks instead of CSE — "
      "likely caused by semantic similarity between the two department files. "
      "Deleting and rebuilding `chroma_db` after separating/tagging chunks would fix this.",
    "- **Dispensary Q18:** The dispensary file is very short; the relevant timing chunk was not "
      "retrieved within `k=4`. Increasing `k` or reducing chunk size helps.",
    "",
]

with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))

print(f"✅ Report saved → {report_path}")
print()
print("=" * 55)
print("  SUMMARY")
print("=" * 55)
print(f"  Answer Rate          : {answer_rate:.1f}%  ({answered_count}/20)")
print(f"  Keyword Match ≥67%   : {correct_rate:.1f}%  ({kw_high}/20)")
print(f"  Avg ROUGE-1          : {summary['ROUGE-1']['mean']:.3f}")
print(f"  Avg ROUGE-2          : {summary['ROUGE-2']['mean']:.3f}")
print(f"  Avg ROUGE-L          : {summary['ROUGE-L']['mean']:.3f}")
print(f"  Avg BLEU-1           : {summary['BLEU-1']['mean']:.3f}")
print(f"  Avg BLEU-2           : {summary['BLEU-2']['mean']:.3f}")
print("=" * 55)
