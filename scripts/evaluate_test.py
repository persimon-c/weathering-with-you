"""Score the 2026 test set exactly once with frozen settings (#46).

Every method is fit on the 2022-2024 training rows with the same settings as
validation (src/experiments.py). Tunable methods use their tau* from
results/tables/threshold_tuning.csv; majority and the rain rule keep their own
predictions.

The real run is gated: it refuses before the freeze (Tue 24 Nov 2026 18:00
PHT), off main, with a dirty working tree, or once any test output exists, so
the test set cannot be scored twice. Not part of run_all.py for that reason.

Usage:
    python scripts/evaluate_test.py --dry-run   # validation as a stand-in, temp folder
    python scripts/evaluate_test.py             # the one real test run
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import FEATURES, load_data
from src.split import split_data
from src.experiments import predict_methods
from src.metrics import evaluate, evaluate_by_cluster
from scripts.export_validation_scores import write_validation_scores
from scripts.plot_results import plot_roc_curves, plot_validation_map
from scripts.run_bootstrap import fmt, load_taus

FREEZE = datetime(2026, 11, 24, 18, 0, tzinfo=timezone(timedelta(hours=8)))
SEED = 42
METRICS = ("balanced_accuracy", "macro_f1", "precision", "recall", "roc_auc", "accuracy")
COUNTS = ("tp", "tn", "fp", "fn")
COMPARISON_FIELDS = ["method", "tau", "n_eval"] + list(METRICS) + list(COUNTS)
PER_CLUSTER_FIELDS = ["method", "cluster_id", "n_eval", "n_flooded"] + list(METRICS) + list(COUNTS)


def output_paths(results_dir):
    results_dir = Path(results_dir)
    return {
        "comparison": results_dir / "tables" / "comparison_test.csv",
        "per_cluster": results_dir / "tables" / "per_cluster_test.csv",
        "scores": results_dir / "predictions" / "test_scores.csv",
        "roc": results_dir / "figures" / "roc_curves_test.png",
        "map": results_dir / "figures" / "test_map.png",
    }


def gate_problems(now, branch, dirty, paths):
    """Return the reasons the real test run must not happen (empty = go)."""
    problems = []
    if now < FREEZE:
        problems.append(f"test set is frozen until {FREEZE.isoformat()}")
    if branch != "main":
        problems.append(f"must run on main, not '{branch}'")
    if dirty:
        problems.append("working tree is not clean")
    problems += [f"already scored once: {p} exists" for p in paths.values() if p.exists()]
    return problems


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def score(train, evaluation, taus, seed=SEED):
    """Fit on train and return [(method, scores, predicted)] for the evaluation rows."""
    out = []
    for name, _, _, predicted, scores in predict_methods(train, evaluation, seed=seed):
        if name in taus:
            predicted = [int(s >= taus[name]) for s in scores]
        out.append((name, scores, predicted))
    return out


def unseen_cells(train, evaluation):
    """Count evaluation rows whose cell never appears in training."""
    lat, lon = FEATURES.index("lat"), FEATURES.index("lon")
    seen = {(r[lat], r[lon]) for r in train["X"]}
    return sum((r[lat], r[lon]) not in seen for r in evaluation["X"])


def metric_row(e):
    return {**{k: fmt(e[k]) for k in METRICS}, **e["confusion_matrix"]}


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_split(train, evaluation, taus, paths, label, seed=SEED):
    """Score one split and write the tables, scores and figures."""
    y = evaluation["y"]
    clusters = [m["cluster_id"] for m in evaluation["metadata"]]
    predictions = score(train, evaluation, taus, seed)

    comparison, per_cluster = [], []
    for name, scores, predicted in predictions:
        by_cluster = evaluate_by_cluster(y, predicted, scores, clusters)
        comparison.append({"method": name, "tau": f"{taus[name]:.2f}" if name in taus else "", "n_eval": len(y),
                           **metric_row(by_cluster["overall"])})
        for cluster, e in by_cluster["clusters"].items():
            rows = [i for i, c in enumerate(clusters) if c == cluster]
            per_cluster.append({"method": name, "cluster_id": cluster, "n_eval": len(rows),
                                "n_flooded": sum(y[i] for i in rows), **metric_row(e)})

    # Scores first: it validates every score before opening a file. If a later step
    # fails, delete the outputs it left by hand, or the gate refuses the rerun.
    write_validation_scores(paths["scores"], y, evaluation["metadata"], evaluation["X"],
                            predictions)
    write_csv(paths["comparison"], COMPARISON_FIELDS, comparison)
    write_csv(paths["per_cluster"], PER_CLUSTER_FIELDS, per_cluster)
    with paths["scores"].open(encoding="utf-8", newline="") as f:
        score_rows = list(csv.DictReader(f))
    paths["roc"].parent.mkdir(parents=True, exist_ok=True)
    plot_roc_curves(score_rows, paths["roc"], title=f"ROC curves – {label}")
    plot_validation_map(score_rows, paths["map"], title=f"{label} cells")

    sizes = {c: clusters.count(c) for c in dict.fromkeys(clusters)}
    return comparison, sizes, unseen_cells(train, evaluation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="score validation as a stand-in for test; never touches test")
    parser.add_argument("--out-dir", type=Path,
                        help="dry-run output folder (default: a new temp folder)")
    args = parser.parse_args(argv)
    if args.out_dir and not args.dry_run:
        parser.error("--out-dir needs --dry-run")

    if args.dry_run:
        out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="evaluate_test_dry_"))
        paths, split, label = output_paths(out_dir), "val", "DRY RUN 2025 validation"
    else:
        paths, split, label = output_paths(ROOT / "results"), "test", "2026 test"
        problems = gate_problems(datetime.now(timezone.utc), git("rev-parse", "--abbrev-ref", "HEAD"),
                                 bool(git("status", "--porcelain")), paths)
        if problems:
            print("Refusing to score the test set:", *problems, sep="\n  ", file=sys.stderr)
            return 1

    # Fixed inputs: the real run must read only what the frozen commit holds.
    splits = split_data(*load_data(ROOT / "data/processed/flood_data.csv"))
    taus = load_taus(ROOT / "results/tables/threshold_tuning.csv")
    comparison, sizes, unseen = evaluate_split(splits["train"], splits[split], taus, paths, label)

    n = sum(sizes.values())
    print(f"{label}: {n} rows, {unseen} in cells unseen in training")
    for cluster, size in sizes.items():
        print(f"  cluster {cluster}: {size} rows ({size / n:.1%})")
    for r in comparison:
        print(f"  {r['method']:<14} tau={r['tau'] or '-':<9} BalAcc={r['balanced_accuracy']} "
              f"MacroF1={r['macro_f1']} AUC={r['roc_auc']}")
    print(f"Wrote outputs under {paths['comparison'].parent.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
