# Usage (from the repository folder):
# python scripts/run_baselines.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
from datetime import datetime, timezone
import csv
import subprocess

from src.data import load_data
from src.split import split_data
from src.metrics import evaluate_by_cluster
from src.baselines import (
    majority_predict,
    majority_predict_proba,
    rain_predict,
    rain_predict_proba,
    choose_rain_cutoff,
    fit_place_history,
    place_history_scores,
    place_history_predict,
)




RUN_COLUMNS = [
    "timestamp_utc", "run_id", "git_commit", "method",
    "split", "seed", "hyperparameters", "n_train", "n_eval",
    "accuracy", "balanced_accuracy", "precision", "recall",
    "f1", "macro_f1", "roc_auc",
    "tp", "tn", "fp", "fn",
    "train_seconds", "predict_seconds",
]



def save_runs_csv(results):
    path = ROOT / "results" / "runs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()

    methods = {
        "majority": {},
        "rainfall": {"cutoff": results["rain_cutoff"]},
        "place_history": {"cutoff": 0.5},
    }

    file_exists = path.exists() and path.stat().st_size > 0

    # Never append rows to a file with a different schema.
    if file_exists:
        with path.open("r", newline="", encoding="utf-8") as file:
            header = next(csv.reader(file), [])
        if header != RUN_COLUMNS:
            raise ValueError("results/runs.csv has an incompatible header")

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_COLUMNS)

        if not file_exists:
            writer.writeheader()

        for method, params in methods.items():
            metrics = results[method]["overall"]
            confusion = metrics["confusion_matrix"]

            row = {
                "timestamp_utc": timestamp,
                "run_id": f"{method}_val_seed0_{timestamp}",
                "git_commit": commit,
                "method": method,
                "split": "val",
                "seed": 0,
                "hyperparameters": json.dumps(params, sort_keys=True),
                "n_train": results["train_samples"],
                "n_eval": results["validation_samples"],
                "train_seconds": None,  # baselines are not timed
                "predict_seconds": None,
            }

            for name in (
                "accuracy", "balanced_accuracy", "precision",
                "recall", "f1", "macro_f1", "roc_auc"
            ):
                row[name] = metrics[name]

            for name in ("tp", "tn", "fp", "fn"):
                row[name] = confusion[name]

            writer.writerow(row)


def main():
    X, y, metadata = load_data(
        ROOT / "data" / "processed" / "flood_data.csv"
    )
    splits = split_data(X, y, metadata)

    train = splits["train"]
    val = splits["val"]

    y_train = train["y"]
    y_val = val["y"]
    X_val = val["X"]

    clusters = [
        row["cluster_id"]
        for row in val["metadata"]
    ]

    majority_class = int(sum(y_train) > len(y_train) / 2)
    prevalence = sum(y_train) / len(y_train)

    majority_predictions = majority_predict(
        X_val, majority_class
    )
    majority_scores = majority_predict_proba(
        X_val, prevalence
    )

    cutoff = choose_rain_cutoff(train["X"], y_train)

    rain_predictions = rain_predict(X_val, cutoff)
    rain_scores = rain_predict_proba(X_val)

    # Learn historical flood rates from training data only.
    place_rates, overall_rate = fit_place_history(
        train["X"], y_train
    )

    # Score 2025 validation cells.
    place_scores = place_history_scores(
        X_val, place_rates, overall_rate
    )

    place_predictions = place_history_predict(
        X_val, place_rates, overall_rate
    )

    results = {
        "dataset": "2025 validation",
        "train_samples": len(y_train),
        "validation_samples": len(y_val),
        "rain_cutoff": cutoff,
        "majority": evaluate_by_cluster(
            y_val,
            majority_predictions,
            majority_scores,
            clusters,
        ),
        "rainfall": evaluate_by_cluster(
            y_val,
            rain_predictions,
            rain_scores,
            clusters,
        ),
        "place_history": evaluate_by_cluster(
            y_val,
            place_predictions,
            place_scores,
            clusters,
        ),
    }

    output = ROOT / "results" / "baselines_val.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("Baseline validation results")
    print(f"Training rows: {len(y_train)}")
    print(f"Validation rows: {len(y_val)}")
    print(f"Rainfall cutoff: {cutoff:.3f}")

    for name in ("majority", "rainfall", "place_history"):
        scores = results[name]["overall"]
        print(f"\n{name.upper()}")
        print(f"Accuracy: {scores['accuracy']:.4f}")
        print(f"Balanced accuracy: {scores['balanced_accuracy']:.4f}")
        print(f"F1: {scores['f1']:.4f}")
        print(f"ROC AUC: {scores['roc_auc']:.4f}")

    save_runs_csv(results)
    print(f"\nSaved results to {output}")
    print("Saved experiment log to results/runs.csv")


if __name__ == "__main__":
    main()
