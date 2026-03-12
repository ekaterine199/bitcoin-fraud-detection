"""
Baseline Modeler & Evaluator

Trains Random Forest + Gradient Boosting baselines using ONLY node features.
Called automatically from main.py before GNN training.
Reads from data/raw/ directly — no PyTorch required inside this module.

The function run_baseline() returns a dict of minimum target metrics that
main.py can optionally use to compare final GNN results against.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import RAW_DIR, BASE_DIR

warnings.filterwarnings("ignore")

RESULTS_DIR  = os.path.join(BASE_DIR, "results", "baseline")
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Visual theme ───────────────────────────────────────────────────────────────
P = {
    "bg":      "#0f1117",
    "panel":   "#1a1d27",
    "accent1": "#f7931a",
    "accent2": "#00d4ff",
    "fraud":   "#ff4757",
    "legit":   "#2ed573",
    "purple":  "#a29bfe",
    "pink":    "#fd79a8",
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
        "font.family":      "monospace",
    })


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING  (mirrors src/processor.py, no torch)
# ══════════════════════════════════════════════════════════════════════════════

def _load_data():
    """
    Reads the raw CSVs from data/raw/ using the same logic as processor.py.
    Returns X, y, feature_names, time_steps — all numpy, no torch.
    """
    print("   Loading raw CSVs for baseline ...")

    features_df = pd.read_csv(
        os.path.join(RAW_DIR, "elliptic_txs_features.csv"), header=None
    )
    classes_df = pd.read_csv(
        os.path.join(RAW_DIR, "elliptic_txs_classes.csv")
    )

    features_df.rename(columns={0: "tx_id"}, inplace=True)
    classes_df.rename(columns={"txId": "tx_id"}, inplace=True)

    combined_df = pd.merge(features_df, classes_df, on="tx_id", how="left")

    # Same class mapping as processor.py: "1"->1 (illicit), "2"->0 (licit)
    class_map = {"1": 1, "2": 0, "unknown": -1}
    combined_df["class"] = combined_df["class"].map(class_map)

    labelled_df  = combined_df[combined_df["class"] != -1].copy()
    feature_cols = [c for c in labelled_df.columns if c not in ("tx_id", "class")]

    feature_names = np.array(
        ["time_step"]
        + [f"local_{i:02d}" for i in range(1, 94)]
        + [f"agg_{i:02d}"   for i in range(1, 73)]
    )

    X          = labelled_df[feature_cols].values.astype(np.float32)
    y          = labelled_df["class"].values.astype(np.int32)
    time_steps = X[:, 0]

    n_illicit = (y == 1).sum()
    print(f"   Labelled nodes : {len(y):,}  "
          f"({n_illicit:,} illicit / {(y==0).sum():,} licit — "
          f"{n_illicit/len(y)*100:.1f}% illicit)\n")

    return X, y, feature_names, time_steps


# ══════════════════════════════════════════════════════════════════════════════
# 2.  EVALUATION  — accuracy intentionally absent
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_model(name: str, y_true, y_pred, y_prob) -> dict:
    pr_auc  = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    metrics = {
        "model":                name,
        "minority_f1":          round(f1_score(y_true, y_pred, pos_label=1), 4),
        "minority_precision":   round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "minority_recall":      round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "macro_f1":             round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1":          round(f1_score(y_true, y_pred, average="weighted"), 4),
        "pr_auc":               round(pr_auc,  4),
        "roc_auc":              round(roc_auc, 4),
    }

    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  {name}  —  Baseline Evaluation")
    print(sep)
    print(f"  Minority F1-Score  (illicit) : {metrics['minority_f1']:.4f}  <- PRIMARY")
    print(f"  Minority Precision (illicit) : {metrics['minority_precision']:.4f}")
    print(f"  Minority Recall    (illicit) : {metrics['minority_recall']:.4f}")
    print(f"  PR-AUC                       : {metrics['pr_auc']:.4f}  <- PRIMARY")
    print(f"  ROC-AUC                      : {metrics['roc_auc']:.4f}")
    print(f"  Macro F1                     : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1                  : {metrics['weighted_f1']:.4f}")
    print(f"\n  Classification report:")
    print(classification_report(y_true, y_pred,
                                 target_names=["Licit (0)", "Illicit (1)"]))
    return metrics


def _cross_validate(clf, X, y, n_splits=5, name="Model"):
    skf   = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=RANDOM_STATE)
    folds = []
    print(f"  [{name}]  {n_splits}-fold CV ...")
    for fold, (tr, val) in enumerate(skf.split(X, y), 1):
        clf.fit(X[tr], y[tr])
        yp    = clf.predict(X[val])
        yprob = clf.predict_proba(X[val])[:, 1]
        folds.append({
            "minority_f1": f1_score(y[val], yp, pos_label=1),
            "pr_auc":      average_precision_score(y[val], yprob),
        })
        print(f"     Fold {fold}: F1={folds[-1]['minority_f1']:.4f}  "
              f"PR-AUC={folds[-1]['pr_auc']:.4f}")

    f1s    = [f["minority_f1"] for f in folds]
    praucs = [f["pr_auc"]      for f in folds]
    print(f"   -> F1    : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"   -> PR-AUC: {np.mean(praucs):.4f} +/- {np.std(praucs):.4f}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, filename):
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)
    print(f"   Saved -> {path}")
    return path


def _plot_confusion_matrices(cm_data):
    n = len(cm_data)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.8), facecolor=P["bg"])
    if n == 1:
        axes = [axes]

    cmap = sns.diverging_palette(145, 15, s=80, l=40, as_cmap=True)
    for ax, (name, y_true, y_pred) in zip(axes, cm_data):
        cm     = confusion_matrix(y_true, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(cm_pct, annot=False, cmap=cmap, linewidths=2,
                    linecolor=P["bg"], ax=ax, cbar=False, vmin=0, vmax=100)

        labels  = [["TN", "FP"], ["FN", "TP"]]
        colours = [[P["legit"], P["fraud"]], [P["fraud"], P["legit"]]]
        for i in range(2):
            for j in range(2):
                text_color = "#1a1d27"
                ax.text(j+0.5, i+0.30, labels[i][j], ha="center", va="center",
                        fontsize=14, fontweight="bold", color=colours[i][j])
                ax.text(j+0.5, i+0.62,
                        f"{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)",
                        ha="center", va="center", fontsize=9, color=text_color)

        ax.set_xlabel("Predicted", fontsize=10, labelpad=8)
        ax.set_ylabel("Actual",    fontsize=10, labelpad=8)
        ax.set_xticklabels(["Licit", "Illicit"], fontsize=9)
        ax.set_yticklabels(["Licit", "Illicit"], fontsize=9, rotation=0)
        ax.set_title(f"Confusion Matrix — {name}", fontsize=11, pad=12,
                     color=P["accent1"], fontweight="bold")

    fig.suptitle("Baseline Models — Confusion Matrices",
                 fontsize=13, y=1.01, color=P["accent1"], fontweight="bold")
    fig.tight_layout(pad=2)
    return _save(fig, "confusion_matrices.png")


def _plot_pr_curves(entries, y_true_ref):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])

    colors = [P["accent1"], P["accent2"], P["purple"], P["pink"]]
    for idx, (name, y_prob) in enumerate(entries):
        prec, rec, _ = precision_recall_curve(y_true_ref, y_prob)
        auc          = average_precision_score(y_true_ref, y_prob)
        ax.plot(rec, prec, lw=2.5, color=colors[idx % len(colors)],
                label=f"{name}  (PR-AUC = {auc:.3f})")

    baseline = y_true_ref.mean()
    ax.axhline(baseline, color="#636e72", lw=1.5, ls="--",
               label=f"Random baseline ({baseline:.3f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves — Illicit Class (1)",
                 fontsize=12, pad=14, color=P["accent1"], fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.2, facecolor=P["panel"], edgecolor="#444")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    fig.tight_layout(pad=1.5)
    return _save(fig, "pr_curves.png")


def _plot_feature_importance(importances, feature_names, model_name, top_n=20):
    idx      = np.argsort(importances)[::-1][:top_n]
    top_imp  = importances[idx]
    top_feat = feature_names[idx]

    bar_colors = [
        P["purple"]  if "time"  in f else
        P["accent1"] if "local" in f else
        P["accent2"]
        for f in top_feat
    ]

    fig, ax = plt.subplots(figsize=(9, 5.8), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    ax.barh(range(top_n), top_imp[::-1], color=bar_colors[::-1],
            height=0.65, edgecolor="none")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feat[::-1], fontsize=8)
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}",
                 fontsize=12, pad=14, color=P["accent1"], fontweight="bold")

    patches = [
        mpatches.Patch(color=P["purple"],  label="Time step"),
        mpatches.Patch(color=P["accent1"], label="Local features (cols 2-94)"),
        mpatches.Patch(color=P["accent2"], label="Aggregate features (cols 95-166)"),
    ]
    ax.legend(handles=patches, fontsize=8, framealpha=0.2,
              facecolor=P["panel"], edgecolor="#444")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.5)
    return _save(fig, f"feature_importance_{model_name.lower().replace(' ', '_')}.png")


def _plot_metrics_comparison(all_metrics):
    models  = [m["model"]              for m in all_metrics]
    f1s     = [m["minority_f1"]        for m in all_metrics]
    praucs  = [m["pr_auc"]             for m in all_metrics]
    recalls = [m["minority_recall"]    for m in all_metrics]
    precs   = [m["minority_precision"] for m in all_metrics]

    x, w = np.arange(len(models)), 0.18
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])

    for offset, vals, color, label in [
        (-1.5*w, f1s,     P["accent1"], "Minority F1"),
        (-0.5*w, praucs,  P["accent2"], "PR-AUC"),
        ( 0.5*w, recalls, P["purple"],  "Recall (illicit)"),
        ( 1.5*w, precs,   P["pink"],    "Precision (illicit)"),
    ]:
        bars = ax.bar(x + offset, vals, w, label=label,
                      color=color, edgecolor="none")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + w/2, h + 0.004, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7.5, color=P["text"])

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Baseline Models — Key Metrics  (accuracy excluded by design)",
                 fontsize=11, pad=14, color=P["accent1"], fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.2, facecolor=P["panel"],
              edgecolor="#444", loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.5)
    return _save(fig, "metrics_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PUBLIC ENTRY POINT  — called by main.py
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline() -> dict:
    """
    Full baseline pipeline. Called by main.py before GNN training.

    Returns:
        minimum_target (dict) — the best baseline metrics that the GNN must beat.
        Keys: min_minority_f1, min_pr_auc, min_minority_recall,
              min_minority_precision, min_macro_f1, min_roc_auc
    """
    _apply_theme()

    print("\n" + "=" * 60)
    print("  STEP 1: BASELINE MODELS  (node features only)")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    X, y, feature_names, time_steps = _load_data()

    # ── Temporal split (sort by time_step, train on earlier 80%) ──
    sorted_idx = np.argsort(time_steps)
    split      = int(len(sorted_idx) * 0.80)
    train_idx  = sorted_idx[:split]
    test_idx   = sorted_idx[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  Train : {len(X_train):,} nodes  ({y_train.mean()*100:.1f}% illicit)")
    print(f"  Test  : {len(X_test):,}  nodes  ({y_test.mean()*100:.1f}% illicit)\n")

    # ── Random Forest ─────────────────────────────────────────────
    print("── Random Forest ──────────────────────────────────────────")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
    )
    _cross_validate(rf, X_train, y_train, n_splits=5, name="Random Forest")
    rf.fit(X_train, y_train)
    rf_pred    = rf.predict(X_test)
    rf_prob    = rf.predict_proba(X_test)[:, 1]
    rf_metrics = _evaluate_model("Random Forest", y_test, rf_pred, rf_prob)

    # ── Gradient Boosting ─────────────────────────────────────────
    print("\n── Gradient Boosting ──────────────────────────────────────")
    gb = HistGradientBoostingClassifier(
        max_iter=500, max_depth=6, learning_rate=0.05,
        min_samples_leaf=20, class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    _cross_validate(gb, X_train, y_train, n_splits=5, name="GradientBoosting")
    gb.fit(X_train, y_train)
    gb_pred    = gb.predict(X_test)
    gb_prob    = gb.predict_proba(X_test)[:, 1]
    gb_metrics = _evaluate_model("GradientBoosting", y_test, gb_pred, gb_prob)

    # ── Log minimum targets ───────────────────────────────────────
    all_metrics = [rf_metrics, gb_metrics]
    best        = max(all_metrics,
                      key=lambda m: (m["minority_f1"] + m["pr_auc"]) / 2)

    minimum_target = {
        "description":            "GNN must exceed ALL of these baseline metrics",
        "best_baseline_model":    best["model"],
        "min_minority_f1":        best["minority_f1"],
        "min_pr_auc":             best["pr_auc"],
        "min_minority_recall":    best["minority_recall"],
        "min_minority_precision": best["minority_precision"],
        "min_macro_f1":           best["macro_f1"],
        "min_roc_auc":            best["roc_auc"],
        "all_baselines":          all_metrics,
    }

    target_path = os.path.join(RESULTS_DIR, "minimum_target_metrics.json")
    with open(target_path, "w") as f:
        json.dump(minimum_target, f, indent=2)

    print("\n" + "=" * 58)
    print("  MINIMUM TARGETS FOR GNN  (must beat all of these)")
    print("=" * 58)
    print(f"  Best baseline     : {best['model']}")
    print(f"  Minority F1       : {best['minority_f1']:.4f}  <- PRIMARY TARGET")
    print(f"  PR-AUC            : {best['pr_auc']:.4f}  <- PRIMARY TARGET")
    print(f"  Minority Recall   : {best['minority_recall']:.4f}")
    print(f"  Minority Precision: {best['minority_precision']:.4f}")
    print(f"  ROC-AUC           : {best['roc_auc']:.4f}")
    print(f"\n  Saved -> {target_path}\n")

    # ── Visualisations ────────────────────────────────────────────
    print("── Generating visualisations ──────────────────────────────")
    _plot_confusion_matrices([
        ("Random Forest",    y_test, rf_pred),
        ("GradientBoosting", y_test, gb_pred),
    ])
    _plot_pr_curves([
        ("Random Forest",    rf_prob),
        ("GradientBoosting", gb_prob),
    ], y_test)
    _plot_feature_importance(
        rf.feature_importances_, feature_names, "Random Forest", top_n=20
    )
    _plot_metrics_comparison(all_metrics)

    print(f"\n  Baseline complete. Results in: {RESULTS_DIR}\n")
    return minimum_target