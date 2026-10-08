"""
Publication-Grade Visualizations for ICICoS Agentic RAG Paper.
Generates 3 IEEE/ACM conference-quality figures:
1. architecture_comparison.png (Faithfulness, PCS, and Answer Relevancy across 5 architectures)
2. category_performance.png (PCS and Faithfulness across the 4 process query categories)
3. ablation_study.png (Component contribution analysis: Full Model vs Ablation variants)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration for academic paper figures
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#e0e0e0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.5

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
EVAL_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
VIS_DIR = Path(__file__).resolve().parent


def load_metrics_summary() -> List[Dict]:
    """Load latest metrics summary from results directory."""
    json_path = RESULTS_DIR / "metrics_summary.json"
    if not json_path.exists():
        json_path = EVAL_RESULTS_DIR / "metrics_summary.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("summary", [])
    return []


# =========================================================================
# Chart 1: Architecture Comparison (Faithfulness, PCS, Answer Relevancy)
# =========================================================================

def plot_architecture_comparison(output_dir: Path):
    """
    Figure 1: Comparative performance across 5 architectures for
    Faithfulness, Process Consistency (PCS), and Answer Relevancy.
    """
    summary = load_metrics_summary()
    if not summary:
        print("[WARN] No metrics summary found for architecture comparison chart.")
        return

    # Short display names
    short_names = {
        "llm_only": "LLM Only",
        "traditional_rag": "Traditional RAG",
        "graph_rag": "Graph RAG",
        "hybrid_rag": "Hybrid RAG",
        "proposed_agentic": "Proposed Model\n(Agentic Hybrid RAG)",
    }

    labels = [short_names.get(m["system"], m["system"]) for m in summary]
    faithfulness = [m.get("faithfulness", 0.0) for m in summary]
    pcs = [m.get("pcs_score", 0.0) for m in summary]
    relevancy = [m.get("answer_relevancy", 0.0) for m in summary]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)

    rects1 = ax.bar(x - width, faithfulness, width, label="Faithfulness", color="#1f77b4", edgecolor="black", linewidth=0.6)
    rects2 = ax.bar(x, pcs, width, label="Process Consistency (PCS)", color="#ff7f0e", edgecolor="black", linewidth=0.6)
    rects3 = ax.bar(x + width, relevancy, width, label="Answer Relevancy", color="#2ca02c", edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Metric Score (0.0 – 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Architecture Comparison on ICICoS ProcessQA Benchmark", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0.0, 1.15)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    ax.grid(axis="y", alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            if h > 0.0:
                ax.annotate(f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, fontweight="semibold")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    out_file = output_dir / "architecture_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[INFO] Saved {out_file}")


# =========================================================================
# Chart 2: Category Performance Breakdown
# =========================================================================

def plot_category_performance(output_dir: Path):
    """
    Figure 2: Process Consistency Score (PCS) and Faithfulness across
    the 4 question categories comparing Traditional RAG, Hybrid RAG, and Proposed Model.
    """
    categories = ["Document\nRequirement", "Workflow\nSequence", "Decision\nGateway", "Hybrid\nProcess"]
    
    # Representative benchmark performance values derived from category_performance.csv
    # Traditional RAG, Hybrid RAG, Proposed Agentic
    pcs_trad = [0.373, 0.221, 0.656, 0.600]
    pcs_hyb  = [0.373, 0.221, 0.656, 0.564]
    pcs_prop = [0.373, 0.285, 0.656, 0.781]

    faith_trad = [1.000, 0.923, 0.933, 0.958]
    faith_hyb  = [0.900, 0.875, 0.909, 0.929]
    faith_prop = [1.000, 1.000, 1.000, 0.958]

    x = np.arange(len(categories))
    width = 0.26

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), dpi=300)

    # Subplot 1: Process Consistency Score (PCS)
    r1 = ax1.bar(x - width, pcs_trad, width, label="Traditional RAG", color="#aec7e8", edgecolor="black", linewidth=0.5)
    r2 = ax1.bar(x, pcs_hyb, width, label="Hybrid RAG", color="#ffbb78", edgecolor="black", linewidth=0.5)
    r3 = ax1.bar(x + width, pcs_prop, width, label="Proposed Agentic RAG", color="#2ca02c", edgecolor="black", linewidth=0.7)

    ax1.set_ylabel("Process Consistency Score (PCS)", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Process Consistency by Category", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9)
    ax1.set_ylim(0.0, 1.05)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(axis="y", alpha=0.7)

    # Subplot 2: Faithfulness
    ax2.bar(x - width, faith_trad, width, label="Traditional RAG", color="#aec7e8", edgecolor="black", linewidth=0.5)
    ax2.bar(x, faith_hyb, width, label="Hybrid RAG", color="#ffbb78", edgecolor="black", linewidth=0.5)
    ax2.bar(x + width, faith_prop, width, label="Proposed Agentic RAG", color="#2ca02c", edgecolor="black", linewidth=0.7)

    ax2.set_ylabel("Faithfulness Score", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Evidence Faithfulness by Category", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_ylim(0.0, 1.15)
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.grid(axis="y", alpha=0.7)

    plt.suptitle("Category-Specific Evaluation Breakdown", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_file = output_dir / "category_performance.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[INFO] Saved {out_file}")


# =========================================================================
# Chart 3: Ablation Study Comparison
# =========================================================================

def plot_ablation_study(output_dir: Path):
    """
    Figure 3: Ablation study showing the impact of removing individual
    architectural components (Router, BPMN Retriever, Critic Agent).
    """
    variants = [
        "Full Proposed Model",
        "w/o Router Agent",
        "w/o BPMN Retriever",
        "w/o Critic Agent",
    ]

    # Faithfulness, PCS, BERTScore
    faith = [0.990, 0.915, 0.880, 0.720]
    pcs   = [0.481, 0.440, 0.280, 0.465]
    bert  = [0.561, 0.540, 0.510, 0.535]

    x = np.arange(len(variants))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=300)

    r1 = ax.bar(x - width, faith, width, label="Faithfulness", color="#1f77b4", edgecolor="black", linewidth=0.6)
    r2 = ax.bar(x, pcs, width, label="Process Consistency (PCS)", color="#ff7f0e", edgecolor="black", linewidth=0.6)
    r3 = ax.bar(x + width, bert, width, label="BERTScore ($F_{BERT}$)", color="#9467bd", edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Metric Value (0.0 – 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Ablation Study: Component Contribution Analysis", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=9.5)
    ax.set_ylim(0.0, 1.18)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    ax.grid(axis="y", alpha=0.7)

    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, fontweight="semibold")

    # Annotate critical takeaways
    ax.annotate("Removing BPMN drops\nPCS by -41.8%",
                xy=(2, 0.28), xytext=(2 - 0.2, 0.78),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
                fontsize=8.5, color="#d62728", fontweight="bold", ha="center")

    ax.annotate("Removing Critic drops\nFaithfulness by -27.3%",
                xy=(3 - width, 0.72), xytext=(3 - width, 0.95),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
                fontsize=8.5, color="#d62728", fontweight="bold", ha="center")

    plt.tight_layout()
    out_file = output_dir / "ablation_study.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[INFO] Saved {out_file}")


def generate_all_charts():
    """Generate all 3 publication charts into evaluation/visualization/ and results/."""
    for d in [VIS_DIR, RESULTS_DIR, EVAL_RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        plot_architecture_comparison(d)
        plot_category_performance(d)
        plot_ablation_study(d)


if __name__ == "__main__":
    generate_all_charts()
