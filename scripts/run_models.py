# Train Gaussian Naive Bayes and Logistic Regression on the training split,
# score on the 2025 validation split, and append results to results/runs.csv.
#
# Usage:
#   python scripts/run_models.py                      # evaluates available models
#   python scripts/run_models.py --model nb           # only Naive Bayes
#   python scripts/run_models.py --model logistic     # only Logistic Regression

import argparse
import csv
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time

import numpy as np

# Ensure repository root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data import load_data
from src.split import split_data
from src.preprocess import preprocess_splits
from src.metrics import evaluate

from src.models.naive_bayes import GaussianNaiveBayes
from src.models.logistic import LogisticRegression


RUNS_HEADER = [
    "timestamp_utc",
    "run_id",
    "git_commit",
    "method",
    "split",
    "seed",
    "hyperparameters",
    "n_train",
    "n_eval",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "roc_auc",
    "tp",
    "tn",
    "fp",
    "fn",
    "train_seconds",
    "predict_seconds",
]


def get_git_commit():
    """Retrieve short git commit hash from HEAD."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True
        ).strip()
        return commit
    except Exception:
        return "unknown"


def prepare_data(data_path):
    """
    Load dataset using src.data, split using src.split, and apply
    the shared preprocessing from src.preprocess (median fill + standardization).
    """
    splits = split_data(*load_data(data_path))
    p, _ = preprocess_splits(splits)
    return p["train"]["X"], np.array(p["train"]["y"]), p["val"]["X"], np.array(p["val"]["y"])


def append_run_to_csv(runs_csv_path, record):
    """Append a single run record to results/runs.csv with header if new."""
    os.makedirs(os.path.dirname(runs_csv_path), exist_ok=True)
    file_exists = os.path.isfile(runs_csv_path)

    with open(runs_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(RUNS_HEADER)

        row = []
        for col in RUNS_HEADER:
            val = record.get(col, "")
            if val is None:
                row.append("")
            elif isinstance(val, float):
                row.append(f"{val:.6f}")
            else:
                row.append(str(val))
        writer.writerow(row)


def run_model_eval(method_name, model, hyperparameters, seed,
                   X_train, y_train, X_val, y_val, out_csv):
    """Fit model, score on val using src.metrics.evaluate, and record to CSV."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{method_name}_val_seed{seed}_{timestamp}"
    commit = get_git_commit()

    print(f"\n=== {method_name} ===")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_val)
    proba = model.predict_proba(X_val)
    predict_s = time.perf_counter() - t0

    # Binary classification: positive class probability
    y_score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba

    metrics = evaluate(list(y_val), list(y_pred), list(y_score))

    cm = metrics["confusion_matrix"]
    print(f"  accuracy:          {metrics['accuracy']:.4f}")
    if metrics["balanced_accuracy"] is not None:
        print(f"  balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  precision:         {metrics['precision']:.4f}")
    print(f"  recall:            {metrics['recall']:.4f}")
    print(f"  f1:                {metrics['f1']:.4f}")
    if metrics["macro_f1"] is not None:
        print(f"  macro_f1:          {metrics['macro_f1']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"  roc_auc:           {metrics['roc_auc']:.4f}")
    print(f"  confusion:         tp={cm['tp']} tn={cm['tn']} fp={cm['fp']} fn={cm['fn']}")
    print(f"  timing:            fit={train_s:.4f}s, predict={predict_s:.4f}s")

    record = {
        "timestamp_utc": timestamp,
        "run_id": run_id,
        "git_commit": commit,
        "method": method_name,
        "split": "val",
        "seed": seed,
        "hyperparameters": json.dumps(hyperparameters),
        "n_train": len(y_train),
        "n_eval": len(y_val),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "macro_f1": metrics["macro_f1"],
        "roc_auc": metrics["roc_auc"],
        "tp": cm["tp"],
        "tn": cm["tn"],
        "fp": cm["fp"],
        "fn": cm["fn"],
        "train_seconds": train_s,
        "predict_seconds": predict_s,
    }

    append_run_to_csv(out_csv, record)
    print(f"  Appended run record to {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Train NB and Logistic Regression on flood data."
    )
    parser.add_argument(
        "--data",
        default=os.path.join(ROOT, "data", "processed", "flood_data.csv"),
        help="path to processed flood_data.csv"
    )
    parser.add_argument(
        "--model", choices=["all", "nb", "logistic"], default="all",
        help="which model to evaluate (default: all available)"
    )
    parser.add_argument(
        "--runs-csv",
        default=os.path.join(ROOT, "results", "runs.csv"),
        help="output CSV file for runs (default: results/runs.csv)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="random seed"
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    X_train_s, y_train, X_val_s, y_val = prepare_data(args.data)
    print(f"  Train: {len(y_train)} rows ({100 * y_train.mean():.1f}% flooded)")
    print(f"  Val:   {len(y_val)} rows ({100 * y_val.mean():.1f}% flooded)")

    # 1. Gaussian Naive Bayes
    if args.model in ("all", "nb"):
        hparams = {"var_floor": 1e-9}
        nb = GaussianNaiveBayes(var_floor=hparams["var_floor"])
        run_model_eval("naive_bayes", nb, hparams, args.seed,
                       X_train_s, y_train, X_val_s, y_val, args.runs_csv)

    # 2. Logistic Regression
    if args.model in ("all", "logistic"):
        hparams = {
            "lr": 0.1,
            "epochs": 2000,
            "reg": 0.01,
            "class_weight": "balanced"
        }
        lr = LogisticRegression(
            lr=hparams["lr"],
            epochs=hparams["epochs"],
            reg=hparams["reg"],
            class_weight=hparams["class_weight"],
            seed=args.seed
        )
        run_model_eval("logistic", lr, hparams, args.seed,
                       X_train_s, y_train, X_val_s, y_val, args.runs_csv)

    print("\nCompleted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
