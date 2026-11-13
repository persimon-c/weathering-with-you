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
from src.metrics import evaluate_by_cluster
from src.baselines import (
    majority_predict, majority_predict_proba, choose_rain_cutoff,
    rain_predict, rain_predict_proba, fit_place_history,
    place_history_predict, place_history_scores,
)
from src.models.naive_bayes import GaussianNaiveBayes
from src.models.logistic import LogisticRegression
from src.models.knn import KNearestNeighbors

METRICS = (
    "accuracy", "balanced_accuracy", "macro_f1", "precision",
    "recall", "f1", "roc_auc",
)
FIELDS = [
    "method", "split", "cluster_id", "threshold_percent", "seed",
    "hyperparameters", "n_train", "n_eval", "n_flooded", "flooded_share",
] + list(METRICS) + ["tp", "tn", "fp", "fn"]


def validation_predictions(train, val, seed=42):
    """Fit on training rows and retain predictions for later error analysis."""
    raw_train, raw_val = train["X"], val["X"]
    y_train = train["y"]
    if not raw_train or not raw_val:
        raise ValueError("Training and validation splits must contain samples.")

    params = fit_preprocess(raw_train)
    X_train = transform_preprocess(raw_train, params)
    X_val = transform_preprocess(raw_val, params)

    prevalence = sum(y_train) / len(y_train)
    majority = int(prevalence > 0.5)
    cutoff = choose_rain_cutoff(raw_train, y_train)
    rates, overall = fit_place_history(raw_train, y_train)
    results = [
        ("majority", 0, {"class": majority},
         majority_predict(raw_val, majority),
         majority_predict_proba(raw_val, prevalence)),
        ("rainfall", 0, {"cutoff": cutoff},
         rain_predict(raw_val, cutoff), rain_predict_proba(raw_val)),
        ("place_history", 0, {"cutoff": 0.5},
         place_history_predict(raw_val, rates, overall),
         place_history_scores(raw_val, rates, overall)),
    ]
    models = [
        ("naive_bayes", GaussianNaiveBayes(var_floor=1e-9),
         {"var_floor": 1e-9}),
        ("logistic", LogisticRegression(
            lr=0.1, epochs=2000, reg=0.01,
            class_weight="balanced", seed=seed),
         {"lr": 0.1, "epochs": 2000, "reg": 0.01,
          "class_weight": "balanced"}),
        ("knn", KNearestNeighbors(k=5), {"k": 5}),
    ]
    for name, model, settings in models:
        model.fit(X_train, y_train)
        results.append((name, seed, settings, list(model.predict(X_val)),
                        list(model.predict_proba(X_val)[:, 1])))
    return results


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
