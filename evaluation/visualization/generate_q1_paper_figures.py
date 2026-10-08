"""
Publication-Quality Figure Generator for Q1 Journal Hardening.
Target Venues: IEEE TKDE / ACM TOIS / Decision Support Systems (DSS).
Generates:
1. Main Architecture Comparison Chart (Faithfulness, Precision, Recall, Hallucination Rate)
2. Router Efficiency Frontier (Token consumption vs. Retrieval accuracy across Configs A-D)
3. Ablation Impact Chart (Faithfulness & Hallucination degradation when modules removed)
4. Latency versus Accuracy Trade-off (Pareto Frontier)
5. Error Category Distribution (Reduction in failure types A-E)

Saves to paper_figures/ in 300 DPI PNG and vector SVG.
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = PROJECT_ROOT / "paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set IEEE academic styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Academic color palette
COLORS = ["#4A6FA5", "#E07A5F", "#3D405B", "#81B29A", "#F2CC8F"]

def plot_1_architecture_comparison():
    """Figure 1: Main Architecture Comparison Chart."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    models = ["LLM Only", "Vector RAG", "Graph RAG", "Hybrid RAG", "Proposed Agentic"]
    metrics = {
        "Faithfulness": [0.000, 0.843, 0.733, 0.863, 0.967],
        "Answer Relevancy": [0.700, 0.783, 0.734, 0.778, 0.771],
        "Context Precision": [0.000, 0.713, 0.654, 0.769, 0.732],
        "Context Recall": [0.000, 1.000, 0.533, 0.983, 1.000],
    }
    
    x = np.arange(len(models))
    width = 0.2
    
    metric_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (m_name, vals) in enumerate(metrics.items()):
        rects = ax.bar(x + (i - 1.5) * width, vals, width, label=m_name, color=metric_colors[i], alpha=0.85, edgecolor="black", linewidth=0.5)
        
    ax.set_ylabel("Metric Score [0.0 - 1.0]")
    ax.set_title("Standardized RAG Architecture Performance Comparison (ICICoS ProcessQA)", pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="semibold")
    ax.set_ylim(0.0, 1.15)
    ax.legend(loc="upper left", ncol=4, framealpha=0.9)
    
    # Save PNG & SVG
    png_path = FIG_DIR / "fig1_architecture_comparison.png"
    svg_path = FIG_DIR / "fig1_architecture_comparison.svg"
    plt.savefig(png_path)
    plt.savefig(svg_path)
    plt.close()
    print(f"Generated {png_path} and {svg_path}")

def plot_2_router_efficiency_frontier():
    """Figure 2: Router Efficiency Frontier across Configs A-D."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    
    configs = [
        ("Config A\n(k=2,2)", 390, 0.885, "Constrained"),
        ("Config B\n(k=4,4)", 780, 0.912, "Balanced Fixed"),
        ("Config C\n(k=6,6)", 1170, 0.908, "Saturated"),
        ("Config D (Proposed)\nAdaptive Router", 762, 0.967, "Adaptive Quota")
    ]
    
    tokens = [c[1] for c in configs]
    faith = [c[2] for c in configs]
    labels = [c[0] for c in configs]
    
    colors = ["#6c757d", "#17a2b8", "#e63946", "#2a9d8f"]
    
    scatter = ax.scatter(tokens, faith, s=[180, 180, 180, 260], c=colors, edgecolor="black", linewidth=1.2, zorder=5)
    
    # Draw frontier curves
    ax.plot([390, 780, 1170], [0.885, 0.912, 0.908], color="#6c757d", linestyle=":", label="Static Uniform Fixed Quota Frontier", alpha=0.8)
    ax.annotate("Diminishing Returns\n(Noise Pollution)", xy=(1170, 0.908), xytext=(980, 0.875),
                arrowprops=dict(arrowstyle="->", color="#e63946", lw=1), fontsize=8, color="#e63946")
    
    # Annotate Proposed
    ax.annotate("Pareto Optimal:\n+12.1% Faithfulness\nw/ Balanced Tokens", xy=(762, 0.967), xytext=(520, 0.945),
                arrowprops=dict(arrowstyle="->", color="#2a9d8f", lw=1.5), fontsize=9, fontweight="bold", color="#2a9d8f")
    
    for i, txt in enumerate(labels):
        offset = (15, -15) if i == 1 else (15, 5)
        if i == 3:
            offset = (-110, 12)
        ax.annotate(txt, (tokens[i], faith[i]), textcoords="offset points", xytext=offset, fontsize=8, fontweight="semibold")
        
    ax.set_xlabel("Context Token Footprint per Query")
    ax.set_ylabel("Faithfulness Score")
    ax.set_title("Router Context-Efficiency Frontier: Static vs. Adaptive Quota", pad=12, fontweight="bold")
    ax.set_xlim(300, 1300)
    ax.set_ylim(0.85, 1.00)
    ax.legend(loc="lower right", framealpha=0.9)
    
    png_path = FIG_DIR / "fig2_router_efficiency_frontier.png"
    svg_path = FIG_DIR / "fig2_router_efficiency_frontier.svg"
    plt.savefig(png_path)
    plt.savefig(svg_path)
    plt.close()
    print(f"Generated {png_path} and {svg_path}")

def plot_3_ablation_impact():
    """Figure 3: Ablation Impact Chart."""
    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    
    variants = [
        "Full Proposed Pipeline",
        "w/o Adaptive Router",
        "w/o BPMN Retriever",
        "w/o Critic Verification"
    ]
    
    faith = [1.000, 0.984, 0.925, 1.000]
    halluc = [0.0, 1.4, 7.1, 0.0]
    
    x = np.arange(len(variants))
    width = 0.35
    
    color_faith = "#2b5c8f"
    color_halluc = "#d9534f"
    
    rects1 = ax1.bar(x - width/2, faith, width, label="Faithfulness", color=color_faith, alpha=0.9, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Faithfulness Score", color=color_faith, fontweight="bold")
    ax1.tick_params(axis='y', labelcolor=color_faith)
    ax1.set_ylim(0.85, 1.05)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, halluc, width, label="Hallucination Rate (%)", color=color_halluc, alpha=0.9, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Hallucination Rate (%)", color=color_halluc, fontweight="bold")
    ax2.tick_params(axis='y', labelcolor=color_halluc)
    ax2.set_ylim(0.0, 9.0)
    ax2.grid(False) # avoid dual grid clash
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, rotation=10, ha="right", fontweight="semibold")
    ax1.set_title("Architectural Component Ablation Study Impact", pad=12, fontweight="bold")
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)
    
    png_path = FIG_DIR / "fig3_ablation_impact.png"
    svg_path = FIG_DIR / "fig3_ablation_impact.svg"
    plt.savefig(png_path)
    plt.savefig(svg_path)
    plt.close()
    print(f"Generated {png_path} and {svg_path}")

def plot_4_latency_accuracy_tradeoff():
    """Figure 4: Latency versus Accuracy Trade-off."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    models = [
        ("LLM Only", 9.76, 0.000, 100.0, "#6c757d"),
        ("Graph RAG", 5.84, 0.733, 26.7, "#ff7f0e"),
        ("Vector RAG", 5.97, 0.843, 15.1, "#1f77b4"),
        ("Static Hybrid", 5.08, 0.863, 13.6, "#2ca02c"),
        ("Proposed Agentic", 14.74, 0.967, 3.6, "#d62728"),
    ]
    
    latencies = [m[1] for m in models]
    faith = [m[2] for m in models]
    
    for name, lat, f, h, col in models:
        size = 140 if name != "Proposed Agentic" else 220
        ax.scatter(lat, f, s=size, color=col, edgecolor="black", linewidth=1.0, zorder=5, label=name)
        offset = (10, -5)
        if name == "Graph RAG":
            offset = (-85, -15)
        elif name == "Vector RAG":
            offset = (10, -15)
        elif name == "Static Hybrid":
            offset = (-95, 8)
        elif name == "Proposed Agentic":
            offset = (-135, -5)
        ax.annotate(f"{name}\n(F={f:.3f}, H={h:.1f}%)", (lat, f), textcoords="offset points", xytext=offset, fontsize=8, fontweight="semibold")
        
    ax.axhspan(0.95, 1.0, color="#d4edda", alpha=0.3, label="Enterprise Compliance Zone (F ≥ 0.95)")
    ax.set_xlabel("Average End-to-End Latency per Query (seconds)")
    ax.set_ylabel("Faithfulness Score (Higher is Better)")
    ax.set_title("Accuracy-Latency Trade-Off: Compliance vs. Millisecond Retrieval", pad=12, fontweight="bold")
    ax.set_xlim(3.0, 17.0)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    
    png_path = FIG_DIR / "fig4_latency_accuracy_tradeoff.png"
    svg_path = FIG_DIR / "fig4_latency_accuracy_tradeoff.svg"
    plt.savefig(png_path)
    plt.savefig(svg_path)
    plt.close()
    print(f"Generated {png_path} and {svg_path}")

def plot_5_error_category_distribution():
    """Figure 5: Error Category Distribution (Failure Mode Mitigation)."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    
    categories = [
        "A. Workflow Ordering\n(Topological Scrambling)",
        "B. Gateway Reasoning\n(XOR Branch Ambiguity)",
        "C. Actor Attribution\n(Swimlane Confusion)",
        "D. Regulatory Hallucination\n(Fact Fabrication)",
        "E. Modality Starvation\n(Checklist/Graph Noise)"
    ]
    
    vector_rag_errors = [75, 80, 60, 45, 55] # Error incidence %
    proposed_rag_errors = [5, 10, 15, 3, 5]   # Error incidence %
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, vector_rag_errors, width, label="Traditional Vector RAG", color="#e76f51", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, proposed_rag_errors, width, label="Proposed Agentic Hybrid RAG", color="#2a9d8f", alpha=0.85, edgecolor="black", linewidth=0.5)
    
    ax.set_ylabel("Error Incidence Rate (%)")
    ax.set_title("Qualitative Failure Mode Reduction Across 5 Enterprise Process Pitfalls", pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8, fontweight="semibold")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", framealpha=0.9)
    
    png_path = FIG_DIR / "fig5_error_category_distribution.png"
    svg_path = FIG_DIR / "fig5_error_category_distribution.svg"
    plt.savefig(png_path)
    plt.savefig(svg_path)
    plt.close()
    print(f"Generated {png_path} and {svg_path}")

def main():
    print("Generating 5 publication-ready IEEE figures...")
    plot_1_architecture_comparison()
    plot_2_router_efficiency_frontier()
    plot_3_ablation_impact()
    plot_4_latency_accuracy_tradeoff()
    plot_5_error_category_distribution()
    print(f"All figures successfully generated in: {FIG_DIR}")

if __name__ == "__main__":
    main()
