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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import FEATURES, load_data
from src.split import split_data
from src.experiments import METHODS, predict_methods


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
    """Train all six methods and return validation predictions."""
    splits = split_data(*load_data(data_path))
    val = splits["val"]

    results = predict_methods(
        splits["train"],
        val,
        seed=seed,
    )

    method_predictions = [
        (name, scores, predicted)
        for name, _, _, predicted, scores in results
    ]

    return (
        val["y"],
        val["metadata"],
        val["X"],
        method_predictions,
    )


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
