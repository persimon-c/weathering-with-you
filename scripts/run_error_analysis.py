import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_per_cluster import validation_predictions
from src.data import FEATURES, load_data
from src.metrics import confusion_matrix
from src.split import split_data


ERROR_FIELDS = [
    "method",
    "split",
    "event_id",
    "cluster_id",
    "lat",
    "lon",
    "elevation",
    "rain_3d",
    "actual",
    "predicted",
    "score",
    "error_type",
]

GROUP_FIELDS = [
    "method",
    "split",
    "group_type",
    "group",
    "n_eval",
    "n_positive",
    "n_negative",
    "tp",
    "tn",
    "fp",
    "fn",
    "false_negative_rate",
    "false_positive_rate",
    "error_rate",
]


def elevation_group(value):
    if value < 100:
        return "below_100m"
    if value < 500:
        return "100_to_below_500m"
    return "500m_and_above"


def rain_group(value):
    if value < 20:
        return "below_20mm"
    if value < 50:
        return "20_to_below_50mm"
    return "50mm_and_above"


def summarize_errors(val, predictions):

    errors = []
    groups = []
    n = len(val["y"])

    if not n or len(val["X"]) != n or len(val["metadata"]) != n:
        raise ValueError(
            "Validation features, labels and metadata must match."
        )

    for info in val["metadata"]:
        if info["split"] != "val":
            raise ValueError(
                "Error analysis accepts validation rows only."
            )

    indices = {
        name: FEATURES.index(name)
        for name in ("lat", "lon", "elevation", "rain_3d")
    }

    for method, _, _, predicted, scores in predictions:
        if len(predicted) != n or len(scores) != n:
            raise ValueError(
                "Predictions and scores must match validation rows."
            )

        # Validate labels and predictions before counting errors.
        confusion_matrix(val["y"], predicted)

        buckets = {}

        for i, (features, actual, info) in enumerate(
            zip(val["X"], val["y"], val["metadata"])
        ):
            values = {
                name: features[index]
                for name, index in indices.items()
            }

            dimensions = (
                ("cluster", info["cluster_id"]),
                (
                    "cell",
                    f"{values['lat']:g},{values['lon']:g}",
                ),
                (
                    "elevation",
                    elevation_group(values["elevation"]),
                ),
                (
                    "rain_3d",
                    rain_group(values["rain_3d"]),
                ),
            )

            pred = int(predicted[i])

            if actual == 1:
                outcome = "tp" if pred == 1 else "fn"
            else:
                outcome = "fp" if pred == 1 else "tn"

            for key in dimensions:
                counts = buckets.setdefault(
                    key,
                    {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
                )
                counts[outcome] += 1

            if actual != pred:
                errors.append({
                    "method": method,
                    "split": "val",
                    "event_id": info["event_id"],
                    "cluster_id": info["cluster_id"],
                    **values,
                    "actual": actual,
                    "predicted": pred,
                    "score": float(scores[i]),
                    "error_type": outcome,
                })

        for (kind, group), counts in sorted(buckets.items()):
            positive = counts["tp"] + counts["fn"]
            negative = counts["tn"] + counts["fp"]
            total = positive + negative

            false_negative_rate = (
                counts["fn"] / positive
                if positive else None
            )
            false_positive_rate = (
                counts["fp"] / negative
                if negative else None
            )

            groups.append({
                "method": method,
                "split": "val",
                "group_type": kind,
                "group": group,
                "n_eval": total,
                "n_positive": positive,
                "n_negative": negative,
                **counts,
                "false_negative_rate": false_negative_rate,
                "false_positive_rate": false_positive_rate,
                "error_rate": (
                    counts["fp"] + counts["fn"]
                ) / total,
            })

    return errors, groups


def run_error_analysis(data_path, output_dir, seed=42):
    splits = split_data(*load_data(data_path))

    predictions = validation_predictions(
        splits["train"],
        splits["val"],
        seed,
    )

    errors, groups = summarize_errors(
        splits["val"],
        predictions,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = (
        ("validation_errors.csv", ERROR_FIELDS, errors),
        ("error_groups.csv", GROUP_FIELDS, groups),
    )

    for name, fields, rows in tables:
        path = output_dir / name

        with path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fields,
            )
            writer.writeheader()
            writer.writerows(rows)

    return errors, groups


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/processed/flood_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/tables",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    errors, groups = run_error_analysis(
        args.data,
        args.output_dir,
        args.seed,
    )

    print(
        f"Saved {len(errors)} validation error rows "
        f"and {len(groups)} group rows "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()