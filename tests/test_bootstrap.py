"""Tests for src.metrics.cluster_bootstrap and scripts/run_bootstrap.py."""

import collections
import csv
import tempfile
import unittest
from pathlib import Path

from src.metrics import cluster_bootstrap
from scripts.run_bootstrap import run_bootstrap

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "results" / "predictions" / "validation_scores.csv"
TAUS = ROOT / "results" / "tables" / "threshold_tuning.csv"

# Synthetic clusters of sizes 1, 2, 3.
CLUSTERS = ["a", "b", "b", "c", "c", "c"]


class TestClusterBootstrap(unittest.TestCase):

    def test_draws_keep_whole_clusters(self):
        seen = []
        cluster_bootstrap(CLUSTERS, lambda idx: seen.append(idx), n_boot=200, seed=1)
        self.assertEqual(len(seen), 200)
        for idx in seen:
            counts = collections.Counter(CLUSTERS[i] for i in idx)
            # Each cluster is drawn whole: its row count is a multiple of its size.
            self.assertEqual(counts["b"] % 2, 0)
            self.assertEqual(counts["c"] % 3, 0)
            # Draws pick as many clusters as there are.
            self.assertEqual(counts["a"] + counts["b"] // 2 + counts["c"] // 3, 3)

    def test_same_seed_same_draws(self):
        stat = lambda idx: sum(idx)
        self.assertEqual(cluster_bootstrap(CLUSTERS, stat, 50, seed=7),
                         cluster_bootstrap(CLUSTERS, stat, 50, seed=7))

    def test_none_draws_are_skipped(self):
        # Statistic undefined whenever cluster "a" is absent.
        values = cluster_bootstrap(CLUSTERS, lambda idx: 1 if 0 in idx else None, 300, seed=3)
        self.assertGreater(len(values), 0)
        self.assertLess(len(values), 300)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            cluster_bootstrap([], len)


class TestRunBootstrap(unittest.TestCase):

    def test_validation_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.csv"
            rows, n_eff = run_bootstrap(SCORES, TAUS, out, n_boot=200, seed=0)
            with out.open(encoding="utf-8", newline="") as f:
                written = list(csv.DictReader(f))

        self.assertEqual([r["method"] for r in written],
                         ["place_history", "logistic", "naive_bayes", "knn"])
        # Point estimates at tau* match the #44 tuning results.
        self.assertEqual(written[0]["balanced_accuracy"], "0.856608")
        self.assertEqual(written[1]["balanced_accuracy"], "0.716786")
        for r in written[1:]:
            lo, hi = float(r["delta_balanced_accuracy_lo"]), float(r["delta_balanced_accuracy_hi"])
            self.assertLessEqual(lo, hi)
            self.assertAlmostEqual(
                float(r["delta_balanced_accuracy"]),
                float(r["balanced_accuracy"]) - float(written[0]["balanced_accuracy"]),
                places=5,
            )
        # 5 validation clusters of sizes 6/96/214/288/383.
        self.assertAlmostEqual(n_eff, 3.42, places=2)


if __name__ == "__main__":
    unittest.main()
