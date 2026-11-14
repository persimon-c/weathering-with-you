# Build the main comparison table from one commit's validation runs.
#
# The six main rows are selected by method, not by "latest run", so extra KNN
# candidates and feature-group rows cannot change the table.
#
# Usage
#   python scripts/compare.py --commit HEAD
#   python scripts/compare.py --commit 40f2c29

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent

METHODS = (
    "majority",
    "rainfall",
    "place_history",
    "naive_bayes",
    "logistic",
    "knn",
)

DISPLAY_NAMES = {
    "majority": "Majority",
    "rainfall": "Rain rule",
    "place_history": "Place history",
    "naive_bayes": "Naive Bayes",
    "logistic": "Logistic regression",
    "knn": "KNN (k = 5)",
}

OUTPUT_COLUMNS = [
    "method",
    "balanced_accuracy",
    "macro_f1",
    "precision",
    "recall",
    "roc_auc",
    "tp",
    "tn",
    "fp",
    "fn",
    "accuracy",
]

NUMERIC_COLUMNS = [
    "balanced_accuracy",
    "macro_f1",
    "precision",
    "recall",
    "roc_auc",
    "accuracy",
]

CONFUSION_COLUMNS = ["tp", "tn", "fp", "fn"]


def resolve_commit(commit):
    """Turn HEAD or a hash into the short hash stored in results/runs.csv."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", commit],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise ValueError(f"Could not resolve commit {commit}") from error


def read_runs(path):
    """Read the run log and check that every table input is present."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    required = {
        "method",
        "git_commit",
        "split",
        "hyperparameters",
        "n_eval",
        *OUTPUT_COLUMNS[1:],
    }

    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise ValueError(f"runs CSV is missing columns {sorted(missing)}")

    if not rows:
        raise ValueError("runs CSV contains no rows")

    return rows


def parse_hyperparameters(value):
    """Read one run's JSON settings, treating a blank field as no settings."""
    if value is None or value.strip() == "":
        return {}

    try:
        settings = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid hyperparameters JSON {value}") from error

    if not isinstance(settings, dict):
        raise ValueError(f"Hyperparameters must be an object, got {value}")

    return settings


def is_main_setting(settings):
    """Feature-group rows carry a group key, main rows do not."""
    return "group" not in settings


def is_knn_five(settings):
    """The KNN row for the setting chosen on validation is k = 5."""
    try:
        return float(settings.get("k")) == 5.0
    except (TypeError, ValueError):
        return False


def select_main_rows(rows, commit):
    """Keep exactly one main validation row per method from one commit."""
    selected = []
    problems = []

    for method in METHODS:
        candidates = []

        for row in rows:
            if row["method"] != method:
                continue
            if row["split"] != "val":
                continue
            if row["git_commit"] != commit:
                continue

            settings = parse_hyperparameters(row["hyperparameters"])

            # Skip the #24 ablation rows so they cannot duplicate a method.
            if not is_main_setting(settings):
                continue

            if method == "knn" and not is_knn_five(settings):
                continue

            candidates.append(row)

        if len(candidates) != 1:
            run_ids = ", ".join(row.get("run_id", "?") for row in candidates)
            problems.append(
                f"{method} expected 1 row, found {len(candidates)} [{run_ids}]"
            )
        else:
            selected.append(candidates[0])

    if problems:
        raise ValueError(
            f"Commit {commit} does not have six unique main validation rows, "
            + "; ".join(problems)
        )

    if len(selected) != len(METHODS):
        raise ValueError(
            f"Expected {len(METHODS)} rows, selected {len(selected)}"
        )

    eval_counts = {row["n_eval"] for row in selected}
    if len(eval_counts) != 1:
        raise ValueError(
            f"Selected rows use different validation sizes, {sorted(eval_counts)}"
        )

    return selected


def format_metric(value):
    """Keep the table at the same six decimals used in results/runs.csv."""
    if value is None or value == "":
        return ""

    return f"{float(value):.6f}"


def format_value(row, column):
    """Format one table cell for CSV or Markdown output."""
    if column == "method":
        return row["method"]

    if column in CONFUSION_COLUMNS:
        return str(int(row[column]))

    return format_metric(row[column])


def write_csv(rows, path):
    """Write the machine-readable comparison table with LF line endings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(OUTPUT_COLUMNS)

        for row in rows:
            writer.writerow(
                [format_value(row, column) for column in OUTPUT_COLUMNS]
            )


def write_markdown(rows, path, commit):
    """Write the report and README version of the same table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_eval = rows[0]["n_eval"]

    lines = [
        "# Main comparison table",
        "",
        f"Source commit `{commit}`. All six rows use the same 2025 validation "
        f"split ({n_eval} rows) and the main feature configuration.",
        "",
        "Feature-group rows are excluded. KNN uses the k = 5 setting chosen "
        "on validation.",
        "",
        "| Method | Balanced accuracy | Macro F1 | Precision | Recall | ROC AUC | TP | TN | FP | FN | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    DISPLAY_NAMES[row["method"]],
                    format_value(row, "balanced_accuracy"),
                    format_value(row, "macro_f1"),
                    format_value(row, "precision"),
                    format_value(row, "recall"),
                    format_value(row, "roc_auc"),
                    format_value(row, "tp"),
                    format_value(row, "tn"),
                    format_value(row, "fp"),
                    format_value(row, "fn"),
                    format_value(row, "accuracy"),
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Build the six-row main comparison table from one commit."
    )
    parser.add_argument(
        "--runs-csv",
        default=str(ROOT / "results" / "runs.csv"),
        help="run log to read (default: results/runs.csv)",
    )
    parser.add_argument(
        "--commit",
        default="HEAD",
        help="source commit or HEAD (default: HEAD)",
    )
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "results" / "tables" / "comparison_main.csv"),
        help="CSV output path (default: results/tables/comparison_main.csv)",
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "results" / "tables" / "comparison_main.md"),
        help="Markdown output path (default: results/tables/comparison_main.md)",
    )
    args = parser.parse_args()

    try:
        commit = resolve_commit(args.commit)
        rows = read_runs(args.runs_csv)
        selected = select_main_rows(rows, commit)
        write_csv(selected, args.output_csv)
        write_markdown(selected, args.output_md, commit)
    except (OSError, ValueError) as error:
        print(f"compare.py: {error}", file=sys.stderr)
        return 1

    print(f"Selected six main rows from commit {commit}")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
