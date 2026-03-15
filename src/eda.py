import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Visual theme matching the baseline aesthetics
P = {
    "bg":      "#0f1117",
    "panel":   "#1a1d27",
    "accent1": "#f7931a", # Bitcoin Orange
    "accent2": "#00d4ff", # Cyan
    "fraud":   "#ff4757", # Red
    "legit":   "#2ed573", # Green
    "unknown": "#8b8fa8", # Grey
    "text":    "#e8eaf0",
    "sub":     "#8b8fa8",
}

def _apply_theme():
    plt.rcParams.update({
        "figure.facecolor": P["bg"],
        "axes.facecolor":   P["panel"],
        "axes.edgecolor":   "#2a2d3a",
        "axes.labelcolor":  P["text"],
        "xtick.color":      P["sub"],
        "ytick.color":      P["sub"],
        "text.color":       P["text"],
        "grid.color":       "#2a2d3a",
        "grid.linestyle":   "--",
        "grid.alpha":       0.45,
    })

def run_comprehensive_eda(data, output_dir="img/eda"):
    """
    Executes a comprehensive Exploratory Data Analysis (EDA) on the Elliptic 
    Bitcoin PyTorch Geometric Data object, saving plots to the output directory.
    """
    print(f"\n🔍 Initiating Deep EDA for Financial Crime Analytics...")
    os.makedirs(output_dir, exist_ok=True)
    _apply_theme()

    x = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    timesteps = data.timestep.cpu().numpy()
    edge_index = data.edge_index.cpu().numpy()
    
    _plot_class_distribution(y, output_dir)
    _plot_temporal_dynamics(y, timesteps, output_dir)
    _plot_edge_homophily(edge_index, y, output_dir)
    
    print(f"✅ EDA complete. Visualizations saved to {output_dir}/")

def _plot_class_distribution(y, output_dir):
    """Visualizes the extreme class imbalance inherent in AML datasets."""
    illicit_count = (y == 1).sum()
    licit_count = (y == 0).sum()
    unknown_count = (y == -1).sum()
    
    total_labeled = illicit_count + licit_count
    illicit_pct = (illicit_count / total_labeled) * 100 if total_labeled > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ['Illicit (1)', 'Licit (0)', 'Unknown (-1)']
    counts = [illicit_count, licit_count, unknown_count]
    colors = [P["fraud"], P["legit"], P["unknown"]]

    bars = ax.bar(categories, counts, color=colors, edgecolor="none")
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', color=P["text"], fontweight="bold")

    ax.set_title(f"Node Class Distribution\n(Illicit Rate: {illicit_pct:.2f}% of labeled data)", 
                 color=P["accent1"], pad=15, fontweight="bold")
    ax.set_ylabel("Number of Transactions")
    ax.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_class_distribution.png"), dpi=150)
    plt.close()

def _plot_temporal_dynamics(y, timesteps, output_dir):
    """Maps the volume of illicit vs licit transactions across the 49 timesteps."""
    unique_ts = np.sort(np.unique(timesteps))
    
    illicit_trends = []
    licit_trends = []
    
    for ts in unique_ts:
        mask = (timesteps == ts)
        illicit_trends.append((y[mask] == 1).sum())
        licit_trends.append((y[mask] == 0).sum())

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(unique_ts, licit_trends, color=P["legit"], label="Licit", linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel("Time Step (~2 weeks each)")
    ax1.set_ylabel("Licit Transactions Count", color=P["legit"])
    ax1.tick_params(axis='y', labelcolor=P["legit"])
    
    ax2 = ax1.twinx()
    ax2.plot(unique_ts, illicit_trends, color=P["fraud"], label="Illicit", linewidth=2, marker='o', markersize=4)
    ax2.set_ylabel("Illicit Transactions Count", color=P["fraud"])
    ax2.tick_params(axis='y', labelcolor=P["fraud"])
    
    ax1.axvline(x=35, color=P["accent1"], linestyle="--", alpha=0.7, label="Test Split Start (TS 35)")

    plt.title("Temporal Transaction Volume: Licit vs. Illicit\n(Notice the massive drop in illicit activity around TS 43)", 
              color=P["accent1"], pad=15, fontweight="bold")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_temporal_dynamics.png"), dpi=150)
    plt.close()

def _plot_edge_homophily(edge_index, y, output_dir):
    """
    Analyzes 'who transacts with whom'. It looks at all edges where both the 
    sender and receiver have known labels, and categorizes the transaction type.
    """
    src, dst = edge_index[0], edge_index[1]
    
    # Only look at edges where both nodes have known labels (0 or 1)
    valid_mask = (y[src] != -1) & (y[dst] != -1)
    src_valid, dst_valid = y[src][valid_mask], y[dst][valid_mask]
    
    # Count the 4 possible transaction flows
    licit_to_licit = ((src_valid == 0) & (dst_valid == 0)).sum()
    illicit_to_illicit = ((src_valid == 1) & (dst_valid == 1)).sum()
    licit_to_illicit = ((src_valid == 0) & (dst_valid == 1)).sum()
    illicit_to_licit = ((src_valid == 1) & (dst_valid == 0)).sum()

    categories = [
        'Licit → Licit\n(Normal)', 
        'Illicit → Illicit\n(Laundering chain)', 
        'Illicit → Licit\n(Cashing out)', 
        'Licit → Illicit\n(Victim/Payment)'
    ]
    counts = [licit_to_licit, illicit_to_illicit, illicit_to_licit, licit_to_illicit]
    colors = [P["legit"], P["fraud"], P["accent1"], P["accent2"]]

    # We use a log scale on the Y axis because Licit->Licit is massive
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, counts, color=colors, edgecolor="none")
    ax.set_yscale('log')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', color=P["text"], fontweight="bold")

    ax.set_title("Transaction Counterparty Flow (Log Scale)\nProves that Illicit nodes cluster together (Homophily)", 
                 color=P["accent1"], pad=15, fontweight="bold")
    ax.set_ylabel("Number of Edges (Log Scale)")
    ax.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_edge_homophily.png"), dpi=150)
    plt.close()


