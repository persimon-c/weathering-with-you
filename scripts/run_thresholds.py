import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_data
from src.split import split_data
from src.preprocess import (
    fit_preprocess,
    transform_preprocess,
    without_missing_flag,
)
from src.metrics import evaluate
from src.baselines import (
    majority_predict, majority_predict_proba, choose_rain_cutoff,
    rain_predict, rain_predict_proba, fit_place_history,
    place_history_predict, place_history_scores,
)
from src.models.naive_bayes import GaussianNaiveBayes
from src.models.logistic import LogisticRegression
from src.models.knn import KNearestNeighbors

THRESHOLDS = (0.5, 1.0, 2.0)
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "precision", "recall", "f1", "roc_auc")
FIELDS = ["threshold_percent", "result_type", "method", "split", "seed", "hyperparameters"]
for split in ("train", "val", "test"):
    FIELDS += [f"{split}_rows", f"{split}_flooded", f"{split}_flooded_share"]
FIELDS += list(METRICS) + ["tp", "tn", "fp", "fn"]


def evaluate_threshold(splits, X_train, X_val, seed):
    """Fit all six methods on this threshold's training labels; score val."""
    train, val = splits["train"], splits["val"]
    y_train, y_val = train["y"], val["y"]
    raw_train, raw_val = train["X"], val["X"]
    prevalence = sum(y_train) / len(y_train)
    cutoff = choose_rain_cutoff(raw_train, y_train)
    rates, overall = fit_place_history(raw_train, y_train)
    results = [
        ("majority", {"class": int(prevalence > 0.5)},
         majority_predict(raw_val, int(prevalence > 0.5)),
         majority_predict_proba(raw_val, prevalence)),
        ("rainfall", {"cutoff": cutoff}, rain_predict(raw_val, cutoff),
         rain_predict_proba(raw_val)),
        ("place_history", {"cutoff": 0.5},
         place_history_predict(raw_val, rates, overall),
         place_history_scores(raw_val, rates, overall)),
    ]
    models = [
        ("naive_bayes", GaussianNaiveBayes(var_floor=1e-9), {"var_floor": 1e-9, "missing_flag": False}),
        ("logistic", LogisticRegression(lr=0.1, epochs=2000, reg=0.01, class_weight="balanced", seed=seed),
         {"lr": 0.1, "epochs": 2000, "reg": 0.01, "class_weight": "balanced"}),
        ("knn", KNearestNeighbors(k=5), {"k": 5}),
    ]
    for name, model, params in models:
        if name == "naive_bayes":
            model_train = without_missing_flag(X_train)
            model_val = without_missing_flag(X_val)
        else:
            model_train = X_train
            model_val = X_val

        model.fit(model_train, y_train)

        results.append((
            name,
            params,
            model.predict(model_val),
            model.predict_proba(model_val)[:, 1],
        ))

    rows = []
    for name, params, predictions, scores in results:
        metrics = evaluate(y_val, list(predictions), list(scores))
        rows.append({
            "method": name, "split": "val",
            "seed": seed if name in ("naive_bayes", "logistic", "knn") else 0,
            "hyperparameters": json.dumps(params, sort_keys=True),
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
