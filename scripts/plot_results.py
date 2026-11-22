# Generate all result figures from the validation predictions and tables.
#
# Saves PNGs to results/figures/.  Matplotlib only, no seaborn.
#
# Usage
#   python scripts/plot_results.py

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import roc_auc, roc_curve

FIGURES_DIR = ROOT / "results" / "figures"

DISPLAY_NAMES = {
    "majority": "Majority",
    "rainfall": "Rain rule",
    "place_history": "Place history",
    "naive_bayes": "Naive Bayes",
    "logistic": "Logistic regression",
    "knn": "KNN (k = 5)",
}

METHOD_ORDER = [
    "majority",
    "rainfall",
    "place_history",
    "naive_bayes",
    "logistic",
    "knn",
]

METHOD_COLORS = {
    "majority": "#999999",
    "rainfall": "#2ca02c",
    "place_history": "#d62728",
    "naive_bayes": "#1f77b4",
    "logistic": "#ff7f0e",
    "knn": "#9467bd",
}

METHOD_LINESTYLES = {
    "majority": ":",
    "rainfall": "-.",
    "place_history": "-",
    "naive_bayes": "--",
    "logistic": "-",
    "knn": "--",
}


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_csv(path):
    """Read a CSV and return a list of dicts."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_validation_scores(path=None):
    """Read results/predictions/validation_scores.csv."""
    if path is None:
        path = ROOT / "results" / "predictions" / "validation_scores.csv"
    return read_csv(path)


def read_comparison(path=None):
    """Read results/tables/comparison_main.csv."""
    if path is None:
        path = ROOT / "results" / "tables" / "comparison_main.csv"
    return read_csv(path)


def read_feature_groups(path=None):
    """Read results/tables/feature_groups.csv."""
    if path is None:
        path = ROOT / "results" / "tables" / "feature_groups.csv"
    return read_csv(path)


def read_per_cluster(path=None):
    """Read results/tables/per_cluster.csv."""
    if path is None:
        path = ROOT / "results" / "tables" / "per_cluster.csv"
    return read_csv(path)


# ---------------------------------------------------------------------------
# Figure 1 – ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(scores_rows, output_path, title="ROC curves – 2025 validation"):
    """ROC curve of every method on 2025 validation."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for method in METHOD_ORDER:
        rows = [r for r in scores_rows if r["method"] == method]
        if not rows:
            continue

        y_true = [int(r["actual"]) for r in rows]
        y_score = [float(r["score"]) for r in rows]

        result = roc_curve(y_true, y_score)
        if result is None:
            continue

        fpr, tpr = result
        auc = roc_auc(y_true, y_score)

        label = f"{DISPLAY_NAMES[method]} (AUC = {auc:.3f})"
        ax.plot(
            fpr, tpr,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=2,
            label=label,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Figure 2 – Bar chart of balanced accuracy and macro F1
# ---------------------------------------------------------------------------

def plot_metric_bars(comparison_rows, output_path):
    """Grouped bar chart of balanced accuracy and macro F1 per method."""
    methods = []
    bal_acc = []
    macro_f1 = []

    for method in METHOD_ORDER:
        row = next((r for r in comparison_rows if r["method"] == method), None)
        if row is None:
            continue
        methods.append(DISPLAY_NAMES[method])
        bal_acc.append(float(row["balanced_accuracy"]))
        macro_f1.append(float(row["macro_f1"]))

    n = len(methods)
    x = list(range(n))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(
        [xi - width / 2 for xi in x], bal_acc, width,
        label="Balanced accuracy", color="#1f77b4",
    )
    bars2 = ax.bar(
        [xi + width / 2 for xi in x], macro_f1, width,
        label="Macro F1", color="#ff7f0e",
    )

    # Add value labels on bars.
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 0.01,
            f"{height:.3f}", ha="center", va="bottom", fontsize=7,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 0.01,
            f"{height:.3f}", ha="center", va="bottom", fontsize=7,
        )

    ax.set_ylabel("Score")
    ax.set_title("Balanced accuracy and macro F1 – 2025 validation")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Figure 3 – Feature-group experiment
# ---------------------------------------------------------------------------

def plot_feature_groups(fg_rows, output_path):
    """Grouped bar chart of balanced accuracy by feature group and method."""
    model_methods = ["naive_bayes", "logistic", "knn"]
    groups = ["place", "place_weather", "place_weather_season"]
    group_labels = ["Place", "Place + Weather", "Place + Weather\n+ Season"]

    fig, ax = plt.subplots(figsize=(9, 5))
    n_groups = len(groups)
    n_methods = len(model_methods)
    width = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#9467bd"]

    for i, method in enumerate(model_methods):
        values = []
        for group in groups:
            row = next(
                (r for r in fg_rows
                 if r["method"] == method and r["group"] == group),
                None,
            )
            values.append(float(row["balanced_accuracy"]) if row else 0)

        positions = [g + (i - 1) * width for g in range(n_groups)]
        bars = ax.bar(
            positions, values, width,
            label=DISPLAY_NAMES[method], color=colors[i],
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + 0.01,
                f"{height:.3f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Feature-group experiment – 2025 validation")
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(group_labels)
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Figure 4 – Per-cluster balanced accuracy
# ---------------------------------------------------------------------------

def plot_per_cluster(cluster_rows, output_path):
    """Heatmap-style grouped bar chart of balanced accuracy per cluster."""
    model_methods = ["naive_bayes", "logistic", "knn"]
    baselines = ["majority", "place_history"]
    all_methods = baselines + model_methods

    # Collect the unique cluster IDs in order.
    cluster_ids = []
    for row in cluster_rows:
        cid = row["cluster_id"]
        if cid not in cluster_ids:
            cluster_ids.append(cid)

    n_clusters = len(cluster_ids)
    n_methods = len(all_methods)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#999999", "#d62728", "#1f77b4", "#ff7f0e", "#9467bd"]

    for i, method in enumerate(all_methods):
        values = []
        for cid in cluster_ids:
            row = next(
                (r for r in cluster_rows
                 if r["method"] == method and r["cluster_id"] == cid),
                None,
            )
            if row and row["balanced_accuracy"]:
                values.append(float(row["balanced_accuracy"]))
            else:
                values.append(0)

        positions = [
            c + (i - n_methods / 2 + 0.5) * width
            for c in range(n_clusters)
        ]
        ax.bar(
            positions, values, width,
            label=DISPLAY_NAMES[method], color=colors[i],
        )

    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Per-cluster balanced accuracy – 2025 validation")
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f"Cluster {c}" for c in cluster_ids], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Figure 5 – Validation map
# ---------------------------------------------------------------------------

def plot_validation_map(scores_rows, output_path, method="place_history", title="Validation cells"):
    """Scatter map of validation cells colored by correct / FP / FN.

    Uses one method (default: place_history) for the classification outcome.
    """
    rows = [r for r in scores_rows if r["method"] == method]
    if not rows:
        print(f"No rows for method {method}, skipping map.")
        return

    correct_lat, correct_lon = [], []
    fp_lat, fp_lon = [], []
    fn_lat, fn_lon = [], []

    for r in rows:
        lat = float(r["lat"])
        lon = float(r["lon"])
        actual = int(r["actual"])
        predicted = int(r["predicted"])

        if actual == predicted:
            correct_lat.append(lat)
            correct_lon.append(lon)
        elif actual == 0 and predicted == 1:
            fp_lat.append(lat)
            fp_lon.append(lon)
        else:  # actual == 1 and predicted == 0
            fn_lat.append(lat)
            fn_lon.append(lon)

    fig, ax = plt.subplots(figsize=(8, 10))

    ax.scatter(
        correct_lon, correct_lat, c="#2ca02c", s=12, alpha=0.5,
        label=f"Correct ({len(correct_lat)})", zorder=2,
    )
    ax.scatter(
        fp_lon, fp_lat, c="#d62728", s=20, alpha=0.7,
        marker="^", label=f"False positive ({len(fp_lat)})", zorder=3,
    )
    ax.scatter(
        fn_lon, fn_lat, c="#1f77b4", s=20, alpha=0.7,
        marker="v", label=f"False negative ({len(fn_lat)})", zorder=3,
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    display = DISPLAY_NAMES.get(method, method)
    ax.set_title(f"{title} – {display}")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    scores = read_validation_scores()
    comparison = read_comparison()
    feature_groups = read_feature_groups()
    per_cluster = read_per_cluster()

    plot_roc_curves(scores, FIGURES_DIR / "roc_curves.png")
    plot_metric_bars(comparison, FIGURES_DIR / "metric_bars.png")
    plot_feature_groups(feature_groups, FIGURES_DIR / "feature_groups.png")
    plot_per_cluster(per_cluster, FIGURES_DIR / "per_cluster.png")
    plot_validation_map(scores, FIGURES_DIR / "validation_map.png")

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
