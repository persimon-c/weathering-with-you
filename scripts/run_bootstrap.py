"""Paired cluster-bootstrap intervals against Place History on validation (#45).

Each learned method is scored at its tuned tau* from threshold_tuning.csv.
Every draw resamples whole storm clusters, and both methods are scored on the
same draw, so Delta = metric(method) - metric(place_history) is paired.
With only 5 validation clusters the intervals are coarse; the per-cluster
table stays the primary evidence (see results/bootstrap.md).
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import evaluate, cluster_bootstrap

REFERENCE = "place_history"
LEARNED = ("logistic", "naive_bayes", "knn")
DELTA_METRICS = ("balanced_accuracy", "macro_f1")

FIELDS = (
    ["method", "tau", "n_eval", "n_clusters", "balanced_accuracy", "macro_f1", "roc_auc"]
    + [f"delta_{m}{s}" for m in DELTA_METRICS for s in ("", "_lo", "_hi", "_excludes_zero")]
    + ["n_boot", "seed"]
)


def load_scores(path):
    """Return {method: {"y", "s", "cluster"}} in file order."""
    data = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            d = data.setdefault(row["method"], {"y": [], "s": [], "cluster": []})
            d["y"].append(int(row["actual"]))
            d["s"].append(float(row["score"]))
            d["cluster"].append(row["cluster_id"])
    return data


def load_taus(path):
    """Return {method: tau*} for the main feature set."""
    with open(path, encoding="utf-8", newline="") as f:
        return {
            r["method"]: float(r["tau"]) for r in csv.DictReader(f)
            if r["feature_set"] == "main" and r["is_optimal"] == "true"
        }


def effective_clusters(cluster_ids):
    """Kish effective number of clusters, (sum n)^2 / sum n^2."""
    sizes = np.unique(cluster_ids, return_counts=True)[1]
    return sizes.sum() ** 2 / (sizes ** 2).sum()


def fmt(v):
    return "" if v is None else f"{float(v):.6f}"


def run_bootstrap(scores_path, taus_path, output, n_boot=1000, seed=0):
    data = load_scores(scores_path)
    taus = load_taus(taus_path)
    ref = data[REFERENCE]
    y, clusters = ref["y"], ref["cluster"]

    # Pairing is only valid if every method lists the same rows in the same order.
    for m in LEARNED:
        if data[m]["y"] != y or data[m]["cluster"] != clusters:
            raise ValueError(f"Rows of '{m}' do not line up with '{REFERENCE}'.")

    def predict(m):
        return [int(s >= taus[m]) for s in data[m]["s"]]

    ref_pred = predict(REFERENCE)
    base = {"n_eval": len(y), "n_clusters": len(set(clusters)), "n_boot": n_boot, "seed": seed}

    def point_row(m, pred):
        e = evaluate(y, pred, data[m]["s"])
        return e, {"method": m, "tau": f"{taus[m]:.2f}", **base,
                   **{k: fmt(e[k]) for k in ("balanced_accuracy", "macro_f1", "roc_auc")}}

    ref_eval, ref_row = point_row(REFERENCE, ref_pred)
    rows = [ref_row]

    for m in LEARNED:
        pred = predict(m)
        e, row = point_row(m, pred)

        def delta(idx):
            yy = [y[i] for i in idx]
            a = evaluate(yy, [pred[i] for i in idx])
            b = evaluate(yy, [ref_pred[i] for i in idx])
            if a["balanced_accuracy"] is None:
                return None
            return [a[k] - b[k] for k in DELTA_METRICS]

        # Same seed for every method, so all methods see the same cluster draws.
        draws = np.array(cluster_bootstrap(clusters, delta, n_boot=n_boot, seed=seed))
        for j, k in enumerate(DELTA_METRICS):
            lo, hi = np.quantile(draws[:, j], [0.025, 0.975])
            row[f"delta_{k}"] = fmt(e[k] - ref_eval[k])
            row[f"delta_{k}_lo"] = fmt(lo)
            row[f"delta_{k}_hi"] = fmt(hi)
            row[f"delta_{k}_excludes_zero"] = "true" if lo > 0 or hi < 0 else "false"
        rows.append(row)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, restval="", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows, effective_clusters(clusters)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path,
                        default=ROOT / "results/predictions/validation_scores.csv")
    parser.add_argument("--taus", type=Path,
                        default=ROOT / "results/tables/threshold_tuning.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/tables/comparison_validation_final.csv")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rows, n_eff = run_bootstrap(args.scores, args.taus, args.output, args.n_boot, args.seed)
    print(f"Effective number of clusters (Kish): {n_eff:.2f}")
    for r in rows[1:]:
        print(f"  {r['method']:<12} dBalAcc={r['delta_balanced_accuracy']} "
              f"[{r['delta_balanced_accuracy_lo']}, {r['delta_balanced_accuracy_hi']}]  "
              f"dMacroF1={r['delta_macro_f1']} "
              f"[{r['delta_macro_f1_lo']}, {r['delta_macro_f1_hi']}]")
    print(f"Saved {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
