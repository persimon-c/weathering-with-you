# Train the three learned methods on three feature groups and score each run
# on the 2025 validation split.
#
# The same training-only preprocessing is applied once before the columns are
# selected, so every group uses the shared 15-column processed matrix.
#
# Usage:
#   python scripts/run_feature_groups.py
#   python scripts/run_feature_groups.py --output /tmp/feature_groups.csv

import argparse
import csv
import os
import sys

# Ensure repository root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.run_models import prepare_data, run_model_eval
from src.feature_groups import FEATURE_GROUPS, select_features
from src.experiments import MODEL_METHODS, build_model


TABLE_COLUMNS = [
    "method",
    "group",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
]


def build_models(seed):
    """Create fresh models using the shared settings."""
    models = []

    for name in MODEL_METHODS:
        model, settings = build_model(name, seed=seed)
        models.append((name, model, settings))

    return models


def format_metric(value):
    """Keep table values at the same six decimals used in results/runs.csv."""
    if value is None:
        return ""
    return f"{value:.6f}"


def write_table(rows, output_csv):
    """Write the nine summary rows, replacing any previous table."""
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=TABLE_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                column: format_metric(row[column])
                if column in ("balanced_accuracy", "macro_f1", "roc_auc")
                else row[column]
                for column in TABLE_COLUMNS
            })


def run_feature_groups(X_train, y_train, X_val, y_val,
                       runs_csv, output_csv, seed=42):
    """Run all method and group combinations and write the summary table."""
    rows = []

    for group in FEATURE_GROUPS:
        X_train_group = select_features(X_train, group)
        X_val_group = select_features(X_val, group)

        print(f"\nFeature group: {group} "
              f"({X_train_group.shape[1]} columns)")

        for method_name, model, hyperparameters in build_models(seed):
            run_hyperparameters = dict(hyperparameters)
            run_hyperparameters["group"] = group

            # Logged as in run_models.py, so these rows differ from the old flagged ones
            if method_name == "naive_bayes":
                run_hyperparameters["missing_flag"] = False

            # Naive Bayes must not see the binary soil_wetness_missing
            # indicator. In the flooded class the flag is almost always 0
            # (ocean points are not flooded), giving it near-zero variance.
            # With var_floor=1e-9 this produces massive positive log-odds for
            # every non-ocean row, inflating balanced accuracy by ~7 points
            # and falsely crediting weather features for the gain.
            # Strip the last column of any group that includes the flag;
            # the flag is always appended last in feature_groups.py.
            if method_name == "naive_bayes" and \
                    "soil_wetness_missing" in FEATURE_GROUPS[group]:
                X_train_model = X_train_group[:, :-1]
                X_val_model = X_val_group[:, :-1]
            else:
                X_train_model = X_train_group
                X_val_model = X_val_group

            metrics = run_model_eval(
                method_name,
                model,
                run_hyperparameters,
                seed,
                X_train_model,
                y_train,
                X_val_model,
                y_val,
                runs_csv,
            )

            rows.append({
                "method": method_name,
                "group": group,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "roc_auc": metrics["roc_auc"],
            })

    write_table(rows, output_csv)
    print(f"\nWrote {len(rows)} rows to {output_csv}")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Run the feature-group experiment on the flood data."
    )
    parser.add_argument(
        "--data",
        default=os.path.join(ROOT, "data", "processed", "flood_data.csv"),
        help="path to processed flood_data.csv"
    )
    parser.add_argument(
        "--runs-csv",
        default=os.path.join(ROOT, "results", "runs.csv"),
        help="output CSV file for runs (default: results/runs.csv)"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "results", "tables", "feature_groups.csv"),
        help="summary table output (default: results/tables/feature_groups.csv)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="random seed"
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    X_train, y_train, X_val, y_val = prepare_data(args.data)
    print(f"  Train: {len(y_train)} rows ({100 * y_train.mean():.1f}% flooded)")
    print(f"  Val:   {len(y_val)} rows ({100 * y_val.mean():.1f}% flooded)")

    run_feature_groups(
        X_train,
        y_train,
        X_val,
        y_val,
        args.runs_csv,
        args.output,
        seed=args.seed,
    )

    print("\nCompleted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
