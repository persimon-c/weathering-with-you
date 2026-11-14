import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_data
from src.split import split_data
from src.metrics import evaluate_by_cluster
from src.experiments import predict_methods

METRICS = (
    "accuracy", "balanced_accuracy", "macro_f1", "precision",
    "recall", "f1", "roc_auc",
)
FIELDS = [
    "method", "split", "cluster_id", "threshold_percent", "seed",
    "hyperparameters", "n_train", "n_eval", "n_flooded", "flooded_share",
] + list(METRICS) + ["tp", "tn", "fp", "fn"]


def validation_predictions(train, val, seed=42):
    # Same six methods and settings as every other runner, see src/experiments.py
    return predict_methods(train, val, seed=seed)


def run_per_cluster(data_path, output, seed=42):
    # Use the saved main labels. No dependency on threshold relabeling.
    splits = split_data(*load_data(data_path))
    train, val = splits["train"], splits["val"]
    predictions = validation_predictions(train, val, seed)
    cluster_ids = [info["cluster_id"] for info in val["metadata"]]
    summary = {}
    for label, cluster in zip(val["y"], cluster_ids):
        counts = summary.setdefault(cluster, {"n_eval": 0, "n_flooded": 0})
        counts["n_eval"] += 1
        counts["n_flooded"] += label

    rows = []
    for name, method_seed, settings, predicted, scores in predictions:
        evaluated = evaluate_by_cluster(val["y"], predicted, scores, cluster_ids)
        for cluster, metrics in evaluated["clusters"].items():
            counts = summary[cluster]
            rows.append({
                "method": name, "split": "val", "cluster_id": cluster,
                "threshold_percent": 1.0, "seed": method_seed,
                "hyperparameters": json.dumps(settings, sort_keys=True),
                "n_train": len(train["y"]), **counts,
                "flooded_share": counts["n_flooded"] / counts["n_eval"],
                **{key: metrics[key] for key in METRICS},
                **metrics["confusion_matrix"],
            })

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data/processed/flood_data.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/tables/per_cluster.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = run_per_cluster(args.data, args.output, args.seed)
    print(f"Saved {len(rows)} validation cluster rows to {args.output}")


if __name__ == "__main__":
    main()
