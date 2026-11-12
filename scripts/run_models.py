# Train Naive Bayes and logistic regression, score on the 2025
# validation split, and save results to results/.
#
# Usage:
#   python scripts/run_models.py                           # real data
#   python scripts/run_models.py --synthetic               # made-up data
#
# The --synthetic flag generates a small dataset that mimics the real
# class balance (~17 % flooded) so the models can be tested before the
# processed CSV exists.

import argparse
import csv
import os
import sys

import numpy as np

# Allow imports from the repository root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from src.models.naive_bayes import GaussianNaiveBayes
except ImportError:
    GaussianNaiveBayes = None

try:
    from src.models.logistic import LogisticRegression
except ImportError:
    LogisticRegression = None


# ------------------------------------------------------------------
# Feature columns from design.md
# ------------------------------------------------------------------

FEATURE_COLS = [
    "lat", "lon", "elevation",
    "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_21d", "rain_30d",
    "humidity", "temperature", "soil_wetness",
    "month_sin", "month_cos",
]

TARGET = "flooded"
SPLIT_COL = "split"


# ------------------------------------------------------------------
# Data loading helpers
# ------------------------------------------------------------------

def load_real_data(path):
    """Load data/processed/flood_data.csv and split by the 'split' column."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    def extract(split_name):
        subset = [r for r in rows if r[SPLIT_COL] == split_name]
        X = np.array([[float(r[c]) for c in FEATURE_COLS] for r in subset])
        y = np.array([int(r[TARGET]) for r in subset])
        return X, y

    X_train, y_train = extract("train")
    X_val, y_val = extract("val")

    return X_train, y_train, X_val, y_val


def make_synthetic_data(seed=42):
    """
    Generate synthetic data that mimics the real dataset's shape:
    ~2000 training rows, ~1000 validation rows, ~17 % positive rate,
    14 features.
    """
    rng = np.random.default_rng(seed)

    def generate(n, pos_rate=0.17):
        n_pos = int(n * pos_rate)
        n_neg = n - n_pos
        # Negative class: centered near zero
        X_neg = rng.standard_normal((n_neg, 14))
        # Positive class: shifted to make it learnable
        X_pos = rng.standard_normal((n_pos, 14)) + 0.8
        X = np.vstack([X_neg, X_pos])
        y = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
        # Shuffle
        idx = rng.permutation(n)
        return X[idx], y[idx].astype(int)

    X_train, y_train = generate(1966)
    X_val, y_val = generate(987)
    return X_train, y_train, X_val, y_val


# ------------------------------------------------------------------
# Standardisation (training stats only)
# ------------------------------------------------------------------

def standardise(X_train, X_val):
    """Zero-mean, unit-variance using training statistics."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0  # constant features stay zero
    return (X_train - mu) / sigma, (X_val - mu) / sigma


# ------------------------------------------------------------------
# Simple metrics (used only when src/metrics.py is not ready)
# ------------------------------------------------------------------

def _confusion(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def _basic_metrics(y_true, y_pred, y_score=None):
    tp, tn, fp, fn = _confusion(y_true, y_pred)
    n = tp + tn + fp + fn

    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    # Balanced accuracy
    tpr = rec
    tnr = tn / (tn + fp) if (tn + fp) else 0
    bal_acc = (tpr + tnr) / 2

    # Macro F1
    prec_0 = tn / (tn + fn) if (tn + fn) else 0
    rec_0 = tn / (tn + fp) if (tn + fp) else 0
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) else 0
    macro_f1 = (f1_0 + f1) / 2

    # ROC AUC (trapezoidal)
    auc = None
    if y_score is not None:
        order = np.argsort(-y_score)
        y_sorted = y_true[order]
        n_pos = y_true.sum()
        n_neg = n - n_pos
        if n_pos > 0 and n_neg > 0:
            tps = np.cumsum(y_sorted)
            fps = np.cumsum(1 - y_sorted)
            tpr_arr = np.concatenate([[0], tps / n_pos])
            fpr_arr = np.concatenate([[0], fps / n_neg])
            auc = float(np.trapezoid(tpr_arr, fpr_arr))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_f1": macro_f1,
        "roc_auc": auc,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


# ------------------------------------------------------------------
# Save results
# ------------------------------------------------------------------

def save_results(name, metrics, out_dir):
    """Write a metric summary CSV to results/."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_val_metrics.csv")

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, val in metrics.items():
            if key == "confusion":
                for k2, v2 in val.items():
                    writer.writerow([f"confusion_{k2}", v2])
            else:
                writer.writerow([key, f"{val:.6f}" if val is not None else ""])

    print(f"  Saved to {path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train NB and LR, score on validation."
    )
    parser.add_argument(
        "--data", default=os.path.join(ROOT, "data", "processed", "flood_data.csv"),
        help="path to the processed CSV"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="use made-up data instead of the real CSV"
    )
    parser.add_argument(
        "--model", choices=["all", "nb", "logistic"], default="all",
        help="which model to evaluate (default: all available)"
    )
    parser.add_argument(
        "--out", default=os.path.join(ROOT, "results"),
        help="output directory (default: results/)"
    )
    args = parser.parse_args()

    # Load data
    if args.synthetic:
        print("Using synthetic data (--synthetic)")
        X_train, y_train, X_val, y_val = make_synthetic_data()
    else:
        if not os.path.isfile(args.data):
            print(f"ERROR: {args.data} not found.")
            print("Run with --synthetic to test on made-up data,")
            print("or wait for issues #3 and #4 to produce the CSV.")
            return 1
        print(f"Loading {args.data}")
        X_train, y_train, X_val, y_val = load_real_data(args.data)

    print(f"  Train: {X_train.shape[0]} rows, "
          f"{y_train.sum():.0f} flooded ({100*y_train.mean():.1f}%)")
    print(f"  Val:   {X_val.shape[0]} rows, "
          f"{y_val.sum():.0f} flooded ({100*y_val.mean():.1f}%)")

    # Standardise
    X_train_s, X_val_s = standardise(X_train, X_val)

    # ---- Gaussian Naive Bayes ----------------------------------------
    if args.model in ("all", "nb") and GaussianNaiveBayes is not None:
        print("\n=== Gaussian Naive Bayes ===")
        nb = GaussianNaiveBayes(var_floor=1e-9)
        nb.fit(X_train_s, y_train)

        nb_pred = nb.predict(X_val_s)
        nb_proba = nb.predict_proba(X_val_s)[:, 1]
        nb_metrics = _basic_metrics(y_val, nb_pred, nb_proba)

        for k, v in nb_metrics.items():
            if k != "confusion":
                print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")
        cm = nb_metrics["confusion"]
        print(f"  confusion: TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}")

        save_results("naive_bayes", nb_metrics, args.out)
    elif args.model == "nb" and GaussianNaiveBayes is None:
        print("ERROR: GaussianNaiveBayes could not be imported.")

    # ---- Logistic Regression -----------------------------------------
    if args.model in ("all", "logistic") and LogisticRegression is not None:
        print("\n=== Logistic Regression (L2, balanced weights) ===")
        lr = LogisticRegression(
            lr=0.1, epochs=2000, reg=0.01,
            class_weight="balanced", seed=42
        )
        lr.fit(X_train_s, y_train)

        lr_pred = lr.predict(X_val_s)
        lr_proba = lr.predict_proba(X_val_s)[:, 1]
        lr_metrics = _basic_metrics(y_val, lr_pred, lr_proba)

        for k, v in lr_metrics.items():
            if k != "confusion":
                print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")
        cm = lr_metrics["confusion"]
        print(f"  confusion: TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}")

        # Loss curve summary
        loss = lr.loss_history_
        print(f"  loss: start={loss[0]:.4f} end={loss[-1]:.4f}")

        save_results("logistic", lr_metrics, args.out)
    elif args.model == "logistic" and LogisticRegression is None:
        print("ERROR: LogisticRegression could not be imported.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
