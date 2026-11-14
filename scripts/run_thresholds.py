import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_data
from src.split import split_data
from src.preprocess import fit_preprocess, transform_preprocess
from src.metrics import evaluate
from src.experiments import predict_methods

THRESHOLDS = (0.5, 1.0, 2.0)
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "precision", "recall", "f1", "roc_auc")
FIELDS = ["threshold_percent", "result_type", "method", "split", "seed", "hyperparameters"]
for split in ("train", "val", "test"):
    FIELDS += [f"{split}_rows", f"{split}_flooded", f"{split}_flooded_share"]
FIELDS += list(METRICS) + ["tp", "tn", "fp", "fn"]


def evaluate_threshold(splits, X_train, X_val, seed):
    """Fit all six methods and score this threshold's validation labels."""
    y_val = splits["val"]["y"]

    results = predict_methods(
        splits["train"],
        splits["val"],
        seed=seed,
        prepared=(X_train, X_val),
    )

    rows = []

    for name, method_seed, params, predictions, scores in results:
        metrics = evaluate(
            y_val,
            list(predictions),
            list(scores),
        )

        rows.append({
            "method": name,
            "split": "val",
            "seed": method_seed,
            "hyperparameters": json.dumps(
                params, sort_keys=True
            ),
            **{key: metrics[key] for key in METRICS},
            **metrics["confusion_matrix"],
        })

    return rows


def run_thresholds(data_path, output, seed=42):
    # Features and row membership never depend on the label threshold.
    # Fit preprocessing once, on training rows only, and reuse exactly.
    original = split_data(*load_data(data_path))
    params = fit_preprocess(original["train"]["X"])
    X_train = transform_preprocess(original["train"]["X"], params)
    X_val = transform_preprocess(original["val"]["X"], params)
    rows = []
    for threshold in THRESHOLDS:
        splits = split_data(*load_data(data_path, flood_threshold=threshold))
        summary = {}
        for name, split in splits.items():
            labels = split["y"]
            summary.update({f"{name}_rows": len(labels),
                            f"{name}_flooded": sum(labels),
                            f"{name}_flooded_share": sum(labels) / len(labels)})
        for row in evaluate_threshold(splits, X_train, X_val, seed):
            rows.append({"threshold_percent": threshold,
                         "result_type": "main" if threshold == 1 else "sensitivity",
                         **summary, **row})
        print(f"Completed {threshold:g}%: six validation methods", flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/flood_data.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/thresholds.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_thresholds(args.data, args.output, args.seed)
    print(f"Saved 18 rows to {args.output}")


if __name__ == "__main__":
    main()
