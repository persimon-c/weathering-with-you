"""Run the 18-point decision-threshold (tau) sweep for fair method comparison.

This script addresses Issue #44: Fair threshold tuning for baselines and learned
models. It is distinct from scripts/run_thresholds.py, which varies the *label
definition* (flood_share > {0.5, 1, 2}%). This script varies the *operating
point* (probability cutoff tau) of each classifier on a fixed 1% label, which
is the standard M4 decision-threshold grid:

    tau in {0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90}

Four methods are tunable under a probability tau:
  - place_history: score = historical flood rate in [0, 1]
  - logistic:      score = sigmoid(w*x + b) in [0, 1]
  - naive_bayes:   score = P(flooded | x) in [0, 1]
  - knn:           score = k-vote fraction in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}

Two methods are structurally degenerate under a probability tau and are excluded
from the grid sweep:
  - majority:   exports a single constant score (training prevalence 0.169888).
                Flips from all-negative to all-positive between tau=0.15 and
                tau=0.20; balanced accuracy is flat 0.5000 at every tau.
  - rainfall:   exports ranking-only scores via atan(rain_3d)/pi in [0.905, 0.998].
                Every grid tau <= 0.90 predicts all-positive; BalAcc = 0.5000.
                The rain rule's real operating point is a mm cutoff (50.98 mm),
                already tuned on training balanced accuracy in run_baselines.py.

Design decisions (see M4-Issue-Plan.md §4):
  D1  New script; run_thresholds.py is untouched (different axis, different test)
  D2  Reuse threshold-independent scores; fit each method once
  D3  Four tunable methods; majority + rainfall documented as not tau-tunable
  D4  GNB true optimum is tau=0.05 (BalAcc 0.6591); issue text says <=0.10
  D5  KNN plateau tau in [0.02, 0.20] -> 0.6647; tau* reported as lowest plateau pt
  D6  Place-only KNN requires a fresh fit (no cached per-row scores available);
      run with --place-only to append place-feature-set rows to the table
  D7  k=51 descoped; k=5 is the cross-k winner from M3
  D8  Selection metric: balanced_accuracy; ties broken by lowest tau (recall-favoring)

Usage:
    python scripts/run_threshold_tuning.py
    python scripts/run_threshold_tuning.py --place-only  # append place-KNN rows
    python scripts/run_threshold_tuning.py --scores results/predictions/validation_scores.csv
    python scripts/run_threshold_tuning.py --output results/tables/threshold_tuning.csv
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_data, FEATURES
from src.split import split_data
from src.preprocess import fit_preprocess, transform_preprocess
from src.experiments import build_model, model_features
from src.metrics import evaluate

TAU_GRID = (
    0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90,
)

# Four methods with well-defined probability scores on [0, 1].
TUNABLE_METHODS = ("place_history", "logistic", "naive_bayes", "knn")

# Methods structurally excluded from the tau grid (see module docstring D3).
NON_TUNABLE_METHODS = ("majority", "rainfall")

SELECTION_METRIC = "balanced_accuracy"  # D8: headline metric for 17%-positive class

FIELDS = [
    "tau", "method", "feature_set", "n_eval",
    "accuracy", "balanced_accuracy", "macro_f1",
    "precision", "recall", "f1", "roc_auc",
    "tp", "tn", "fp", "fn",
    "is_optimal",
]

# Place feature indices in the PREPROCESSED matrix (after transform_preprocess).
# FEATURES = [lat(0), lon(1), elevation(2), rain_1d(3), ...]
_PLACE_INDICES = [0, 1, 2]  # lat, lon, elevation


def load_scores(path):
    """Read validation_scores.csv and return {method: (y_true, scores)} dict.

    Also performs a sanity check: recomputing balanced accuracy at tau=0.50
    must reproduce comparison_main.csv values for the four tunable methods.
    The rainfall method is explicitly exempted because its stored scores are
    ranking-only (atan-transformed mm), not decision probabilities.
    """
    data = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            m = row["method"]
            if m not in data:
                data[m] = {"y": [], "s": []}
            data[m]["y"].append(int(row["actual"]))
            data[m]["s"].append(float(row["score"]))

    # Sanity check: tau=0.50 must reproduce comparison_main.csv for tunable methods.
    # Ground truth from results/tables/comparison_main.csv (committed artifact).
    _EXPECTED_AT_50 = {
        "place_history": 0.781593,
        "logistic":      0.707611,
        "naive_bayes":   0.540372,
        "knn":           0.567485,
    }
    for method, expected in _EXPECTED_AT_50.items():
        if method not in data:
            raise ValueError(f"Method '{method}' not found in {path}")
        y = data[method]["y"]
        s = data[method]["s"]
        pred = [int(sv >= 0.50) for sv in s]
        metrics = evaluate(y, pred, s)
        actual = metrics["balanced_accuracy"]
        if abs(actual - expected) > 1e-4:
            raise ValueError(
                f"Score file sanity check failed for '{method}' at tau=0.50: "
                f"expected BalAcc={expected:.6f}, got {actual:.6f}. "
                f"The score file may be stale or mismatched."
            )

    n = len(next(iter(data.values()))["y"])
    for m in TUNABLE_METHODS:
        if m not in data:
            raise ValueError(f"Required method '{m}' missing from {path}")
        if len(data[m]["y"]) != n:
            raise ValueError(f"Method '{m}' has {len(data[m]['y'])} rows; expected {n}")

    return data


def sweep(y_true, scores, grid=TAU_GRID):
    """Sweep the tau grid and return a list of metric dicts, one per tau."""
    rows = []
    for tau in grid:
        pred = [int(s >= tau) for s in scores]
        metrics = evaluate(y_true, pred, scores)
        rows.append({"tau": tau, **metrics})
    return rows


def select_optimal(rows):
    """Return (tau*, plateau_start, plateau_end, best_bal_acc) using D8 selection rule.

    Selection: argmax balanced_accuracy; ties broken by lowest tau.
    Also records the full plateau span (contiguous run of tau values sharing
    the same balanced accuracy as the optimum).
    """
    best_ba = max(r["balanced_accuracy"] for r in rows if r["balanced_accuracy"] is not None)
    best_rows = [r for r in rows if r["balanced_accuracy"] == best_ba]
    # Ties: lowest tau (recall-favoring, D8)
    best_tau = min(r["tau"] for r in best_rows)

    # Plateau: find contiguous block at the grid level sharing best_ba
    taus = [r["tau"] for r in rows if r["balanced_accuracy"] == best_ba]
    # Sort and find contiguous runs in the grid
    grid_list = list(TAU_GRID)
    plateau_indices = sorted(grid_list.index(t) for t in taus if t in grid_list)
    # Largest contiguous run containing best_tau's index
    best_idx = grid_list.index(best_tau)
    start_idx = best_idx
    end_idx = best_idx
    for i in plateau_indices:
        if i <= best_idx:
            start_idx = min(start_idx, i)
        if i >= best_idx:
            end_idx = max(end_idx, i)
    # Verify they're actually contiguous from start to end
    contiguous = all(j in plateau_indices for j in range(start_idx, end_idx + 1))
    if not contiguous:
        end_idx = best_idx  # fall back to just the optimal point

    plateau_start = grid_list[start_idx]
    plateau_end = grid_list[end_idx]
    return best_tau, plateau_start, plateau_end, best_ba


def run_place_only_knn(data_path, seed=42):
    """Fit KNN k=5 on place-only features and return (y_true, scores).

    Place-only scores are not in validation_scores.csv (which contains only
    the 6 main full-feature methods). This function fits a fresh KNN model
    using only the 3 place columns (lat, lon, elevation) on training data,
    then scores the validation split. This is the correct approach per D6:
    holding k=5 fixed isolates the feature-set effect from the k effect.
    """
    splits = split_data(*load_data(data_path))
    train = splits["train"]
    val = splits["val"]

    # Fit preprocessing on train only (D2 / design.md:218 Decision 3)
    params = fit_preprocess(train["X"])
    X_train_full = transform_preprocess(train["X"], params)
    X_val_full = transform_preprocess(val["X"], params)

    # Select place columns only (indices 0=lat, 1=lon, 2=elevation)
    import numpy as np
    X_train_place = X_train_full[:, _PLACE_INDICES]
    X_val_place = X_val_full[:, _PLACE_INDICES]

    model, _ = build_model("knn", seed=seed, k=5)
    model.fit(X_train_place, train["y"])
    scores = model.predict_proba(X_val_place)[:, 1].tolist()

    return val["y"], scores


def format_row(tau, method, feature_set, n_eval, metrics, is_optimal):
    """Format a single output row with .6f float precision."""
    _FLOAT_COLS = {
        "accuracy", "balanced_accuracy", "macro_f1",
        "precision", "recall", "f1", "roc_auc",
    }
    row = {
        "tau": f"{tau:.2f}",
        "method": method,
        "feature_set": feature_set,
        "n_eval": n_eval,
        "is_optimal": "true" if is_optimal else "false",
    }
    for col in ("accuracy", "balanced_accuracy", "macro_f1",
                "precision", "recall", "f1", "roc_auc"):
        v = metrics.get(col)
        row[col] = ("" if v is None else f"{float(v):.6f}")
    cm = metrics.get("confusion_matrix", {})
    for col in ("tp", "tn", "fp", "fn"):
        row[col] = cm.get(col, "")
    return row


def run_threshold_tuning(scores_path, data_path, output, seed=42,
                         include_place_only=False):
    """Run the full 18-tau sweep and write threshold_tuning.csv.

    Args:
        scores_path: path to results/predictions/validation_scores.csv
        data_path:   path to data/processed/flood_data.csv (for place-only KNN)
        output:      path to write threshold_tuning.csv
        seed:        random seed (not used by tunable methods; for reproducibility)
        include_place_only: if True, append place-feature-set KNN rows (D6)

    Returns:
        list of output row dicts
    """
    score_data = load_scores(scores_path)
    n_eval = len(next(iter(score_data.values()))["y"])

    all_rows = []

    for method in TUNABLE_METHODS:
        y_true = score_data[method]["y"]
        scores = score_data[method]["s"]
        feature_set = "main"

        grid_rows = sweep(y_true, scores)
        best_tau, _, _, _ = select_optimal(grid_rows)

        for gr in grid_rows:
            tau = gr["tau"]
            is_opt = (tau == best_tau)
            all_rows.append(format_row(
                tau, method, feature_set, n_eval, gr, is_opt
            ))

    if include_place_only:
        # D6: Fit place-only KNN, sweep the same 18-tau grid
        y_true, place_scores = run_place_only_knn(data_path, seed=seed)
        grid_rows = sweep(y_true, place_scores)
        best_tau, _, _, _ = select_optimal(grid_rows)

        for gr in grid_rows:
            tau = gr["tau"]
            is_opt = (tau == best_tau)
            all_rows.append(format_row(
                tau, "knn", "place", n_eval, gr, is_opt
            ))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)

    return all_rows


def print_summary(rows, include_place_only=False):
    """Print per-method tau* and budget-parity statement."""
    print()
    print("=" * 60)
    print("THRESHOLD TUNING RESULTS (validation, 1% label threshold)")
    print("=" * 60)

    # Reload score data to get optimal values
    seen = {}
    for row in rows:
        m = row["method"]
        fs = row["feature_set"]
        key = (m, fs)
        if row["is_optimal"] == "true" and key not in seen:
            seen[key] = row

    for key in [(m, "main") for m in TUNABLE_METHODS]:
        if key in seen:
            r = seen[key]
            m, fs = key
            print(
                f"  {m:<18} tau*={r['tau']:>5}  "
                f"BalAcc={r['balanced_accuracy']}  "
                f"recall={r['recall']}  prec={r['precision']}"
            )

    if include_place_only and ("knn", "place") in seen:
        r = seen[("knn", "place")]
        print(
            f"  {'knn (place-only)':<18} tau*={r['tau']:>5}  "
            f"BalAcc={r['balanced_accuracy']}  "
            f"recall={r['recall']}  prec={r['precision']}"
        )

    n_tunable = 4
    print()
    print(
        f"Budget parity: {n_tunable} tunable methods x {len(TAU_GRID)} tau values"
        f" = {n_tunable * len(TAU_GRID)} configurations ({len(TAU_GRID)} per method)"
    )
    print("Non-tunable (excluded from grid):")
    print("  majority  — constant score (training prevalence 0.169888); "
          "BalAcc=0.5000 at every tau")
    print("  rainfall  — ranking-only atan scores in [0.905, 0.998]; "
          "all grid points predict all-positive -> BalAcc=0.5000")
    print("  k=51 descoped: k=5 is the cross-k winner from M3 validation")
    print("=" * 60)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=ROOT / "results" / "predictions" / "validation_scores.csv",
        help="validation scores CSV (default: results/predictions/validation_scores.csv)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "processed" / "flood_data.csv",
        help="processed flood data CSV (default: data/processed/flood_data.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "tables" / "threshold_tuning.csv",
        help="output CSV path (default: results/tables/threshold_tuning.csv)",
    )
    parser.add_argument(
        "--place-only",
        action="store_true",
        default=False,
        help="also run the 18-tau grid on place-only KNN (requires --data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for place-only KNN fit (default: 42)",
    )
    args = parser.parse_args(argv)

    rows = run_threshold_tuning(
        args.scores, args.data, args.output,
        seed=args.seed,
        include_place_only=args.place_only,
    )

    print_summary(rows, include_place_only=args.place_only)
    n_main = len(TUNABLE_METHODS) * len(TAU_GRID)
    n_place = len(TAU_GRID) if args.place_only else 0
    print(f"\nSaved {n_main + n_place} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
