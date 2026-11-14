# Export every validation prediction for the six main methods.
#
# #28 needs the full score list to draw ROC curves, while results/runs.csv
# stores only summary metrics and validation_errors.csv stores only mistakes.
# The score column contains the raw predicted probability of flood (class 1)
# formatted via repr(float) to preserve full floating-point precision for ROC AUC
# calculation and curve plotting in #28. Consumers should parse with float(row['score']).
#
# Usage
#   python scripts/export_validation_scores.py

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines import (
    choose_rain_cutoff,
    fit_place_history,
    majority_predict,
    majority_predict_proba,
    place_history_predict,
    place_history_scores,
    rain_predict,
    rain_predict_proba,
)
from src.data import FEATURES, load_data
from src.models.knn import KNearestNeighbors
from src.models.logistic import LogisticRegression
from src.models.naive_bayes import GaussianNaiveBayes
from src.preprocess import preprocess_splits, without_missing_flag
from src.split import split_data


METHODS = (
    "majority",
    "rainfall",
    "place_history",
    "naive_bayes",
    "logistic",
    "knn",
)

SCORE_COLUMNS = [
    "method",
    "actual",
    "score",
    "predicted",
    "event_id",
    "cluster_id",
    "lat",
    "lon",
]

SEED = 42
NB_VAR_FLOOR = 1e-9
LOGISTIC_LR = 0.1
LOGISTIC_EPOCHS = 2000
LOGISTIC_REG = 0.01
KNN_K = 5  # the setting selected on 2025 validation in #22


def validate_predictions(y_true, scores, predictions):
    """Reject incomplete or invalid prediction arrays before writing them."""
    if not (len(y_true) == len(scores) == len(predictions)):
        raise ValueError("Labels, scores and predictions must have equal lengths")

    for score in scores:
        value = float(score)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Score outside the probability range, {value}")


def write_validation_scores(path, y_true, metadata, X_raw, method_predictions):
    """Write one row per validation sample and main method."""
    if not (len(y_true) == len(metadata) == len(X_raw)):
        raise ValueError("Labels, metadata and raw rows must have equal lengths")

    method_names = [name for name, _, _ in method_predictions]
    if method_names != list(METHODS):
        raise ValueError(
            f"Expected methods {list(METHODS)}, got {method_names}"
        )

    lat_index = FEATURES.index("lat")
    lon_index = FEATURES.index("lon")

    # Validate every method before opening the file, so a bad method cannot
    # leave a half-written score table.
    for _, scores, predictions in method_predictions:
        validate_predictions(y_true, scores, predictions)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(SCORE_COLUMNS)

        for method, scores, predictions in method_predictions:
            for index, (actual, score, predicted) in enumerate(
                zip(y_true, scores, predictions)
            ):
                info = metadata[index]
                row = X_raw[index]

                writer.writerow(
                    [
                        method,
                        int(actual),
                        repr(float(score)),
                        int(predicted),
                        info["event_id"],
                        info["cluster_id"],
                        row[lat_index],
                        row[lon_index],
                    ]
                )


def build_predictions(data_path, seed=SEED):
    """Train all six main methods and return their validation predictions."""
    X, y, metadata = load_data(data_path)
    splits = split_data(X, y, metadata)
    processed, _ = preprocess_splits(splits)

    train = splits["train"]
    val = splits["val"]

    y_train = train["y"]
    y_val = val["y"]
    X_train_raw = train["X"]
    X_val_raw = val["X"]

    X_train = processed["train"]["X"]
    X_val = processed["val"]["X"]

    # Baselines use the raw features, the same as scripts/run_baselines.py.
    majority_class = int(sum(y_train) > len(y_train) / 2)
    prevalence = sum(y_train) / len(y_train)

    majority_scores = majority_predict_proba(X_val_raw, prevalence)
    majority_predictions = majority_predict(X_val_raw, majority_class)

    cutoff = choose_rain_cutoff(X_train_raw, y_train)
    rain_scores = rain_predict_proba(X_val_raw)
    rain_predictions = rain_predict(X_val_raw, cutoff)

    place_rates, overall_rate = fit_place_history(X_train_raw, y_train)
    place_scores = place_history_scores(X_val_raw, place_rates, overall_rate)
    place_predictions = place_history_predict(
        X_val_raw, place_rates, overall_rate
    )

    # Learned methods use the shared training-only preprocessing.
    # Naive Bayes excludes the soil_wetness_missing indicator via the shared
    # without_missing_flag helper, which also validates the matrix shape.
    naive_bayes = GaussianNaiveBayes(var_floor=NB_VAR_FLOOR)
    naive_bayes.fit(without_missing_flag(X_train), np.asarray(y_train))
    naive_bayes_scores = naive_bayes.predict_proba(without_missing_flag(X_val))[:, 1]
    naive_bayes_predictions = naive_bayes.predict(without_missing_flag(X_val))

    logistic = LogisticRegression(
        lr=LOGISTIC_LR,
        epochs=LOGISTIC_EPOCHS,
        reg=LOGISTIC_REG,
        class_weight="balanced",
        seed=seed,
    )
    logistic.fit(X_train, np.asarray(y_train))
    logistic_scores = logistic.predict_proba(X_val)[:, 1]
    logistic_predictions = logistic.predict(X_val)

    knn = KNearestNeighbors(k=KNN_K)
    knn.fit(X_train, np.asarray(y_train))
    knn_scores = knn.predict_proba(X_val)[:, 1]
    knn_predictions = knn.predict(X_val)

    method_predictions = [
        ("majority", majority_scores, majority_predictions),
        ("rainfall", rain_scores, rain_predictions),
        ("place_history", place_scores, place_predictions),
        ("naive_bayes", naive_bayes_scores.tolist(), naive_bayes_predictions.tolist()),
        ("logistic", logistic_scores.tolist(), logistic_predictions.tolist()),
        ("knn", knn_scores.tolist(), knn_predictions.tolist()),
    ]

    return y_val, val["metadata"], X_val_raw, method_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Export all validation scores for the main comparison."
    )
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "processed" / "flood_data.csv"),
        help="path to processed flood_data.csv",
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "results" / "predictions" / "validation_scores.csv"
        ),
        help="output CSV path (default: results/predictions/validation_scores.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="random seed for logistic regression",
    )
    args = parser.parse_args()

    try:
        y_val, metadata, X_val_raw, method_predictions = build_predictions(
            args.data, seed=args.seed
        )
        write_validation_scores(
            args.output, y_val, metadata, X_val_raw, method_predictions
        )
    except (OSError, ValueError) as error:
        print(f"export_validation_scores.py: {error}", file=sys.stderr)
        return 1

    total = len(y_val) * len(method_predictions)
    print(f"Wrote {total} validation scores to {args.output}")
    for method, scores, _ in method_predictions:
        print(f"{method}: {len(scores)} rows")

    return 0


if __name__ == "__main__":
    sys.exit(main())
