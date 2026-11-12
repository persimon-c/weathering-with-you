
import json
import math
from pathlib import Path

from src.data import load_data, FEATURES
from src.split import split_data
from src.metrics import evaluate_by_cluster
from src.baselines import (
    majority_predict,
    choose_rain_cutoff,
    rain_predict,
)


def main():
    X, y, metadata = load_data()
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

    # Majority class is learned from training data.
    majority_class = int(sum(y_train) > len(y_train) / 2)
    majority_predictions = [
        majority_class for _ in y_val
    ]

    # Constant probability based on training prevalence.
    prevalence = sum(y_train) / len(y_train)
    majority_scores = [
        prevalence for _ in y_val
    ]

    # Select rainfall cutoff using training data only.
    cutoff = choose_rain_cutoff(train["X"], y_train)
    rain_predictions = rain_predict(X_val, cutoff)

    # Normalize rainfall scores using training-derived bounds.
    rain_index = FEATURES.index("rain_3d")
    
    val_rain = [
        row[rain_index] for row in X_val
    ]

    rain_scores = [
        0.5 + math.atan(value) / math.pi
         for value in val_rain
        ]

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
    }

    output = Path("results/baselines_val.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("Baseline validation results")
    print(f"Training rows: {len(y_train)}")
    print(f"Validation rows: {len(y_val)}")
    print(f"Rainfall cutoff: {cutoff:.3f}")

    for name in ("majority", "rainfall"):
        scores = results[name]["overall"]
        print(f"\n{name.upper()}")
        print(f"Accuracy: {scores['accuracy']:.4f}")
        print(f"Balanced accuracy: {scores['balanced_accuracy']:.4f}")
        print(f"F1: {scores['f1']:.4f}")
        print(f"ROC AUC: {scores['roc_auc']:.4f}")

    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
