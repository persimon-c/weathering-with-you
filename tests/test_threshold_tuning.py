"""Unit tests for scripts/run_threshold_tuning.py."""

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.run_threshold_tuning import (
    TAU_GRID,
    TUNABLE_METHODS,
    NON_TUNABLE_METHODS,
    FIELDS,
    load_scores,
    sweep,
    select_optimal,
    run_threshold_tuning,
    run_place_only_knn,
)
from src.data import FEATURES
from src.metrics import evaluate

ROOT = Path(__file__).resolve().parent.parent
SCORES_PATH = ROOT / "results" / "predictions" / "validation_scores.csv"
DATA_PATH = ROOT / "data" / "processed" / "flood_data.csv"
COMMITTED_CSV = ROOT / "results" / "tables" / "threshold_tuning.csv"


class TestTauGrid(unittest.TestCase):
    """Grid definition and method lists."""

    def test_tau_grid_has_18_points(self):
        self.assertEqual(len(TAU_GRID), 18)

    def test_tau_grid_values_in_range(self):
        for tau in TAU_GRID:
            self.assertGreater(tau, 0.0)
            self.assertLessEqual(tau, 0.90)

    def test_tau_grid_is_sorted_ascending(self):
        self.assertEqual(list(TAU_GRID), sorted(TAU_GRID))

    def test_tunable_methods_are_four(self):
        self.assertEqual(len(TUNABLE_METHODS), 4)
        self.assertIn("place_history", TUNABLE_METHODS)
        self.assertIn("logistic", TUNABLE_METHODS)
        self.assertIn("naive_bayes", TUNABLE_METHODS)
        self.assertIn("knn", TUNABLE_METHODS)

    def test_non_tunable_methods(self):
        self.assertIn("majority", NON_TUNABLE_METHODS)
        self.assertIn("rainfall", NON_TUNABLE_METHODS)

    def test_tunable_and_non_tunable_disjoint(self):
        self.assertTrue(
            set(TUNABLE_METHODS).isdisjoint(set(NON_TUNABLE_METHODS)),
            "Tunable and non-tunable method sets must not overlap",
        )


class TestSweepAndSelectOptimal(unittest.TestCase):
    """Core sweep logic and optimal-tau selection."""

    def _make_scores(self, n, pattern="alternating"):
        """Generate synthetic (y_true, scores) pairs."""
        if pattern == "alternating":
            y = [i % 2 for i in range(n)]
            # Scores that give a clear peak at tau=0.30
            s = [0.35 if y[i] == 1 else 0.20 for i in range(n)]
        elif pattern == "constant":
            y = [0] * n
            s = [0.15] * n
        elif pattern == "all_positive":
            y = [1] * n
            s = [0.95] * n
        return y, s

    def test_sweep_produces_one_row_per_tau(self):
        y, s = self._make_scores(100)
        rows = sweep(y, s)
        self.assertEqual(len(rows), len(TAU_GRID))

    def test_sweep_at_tau_50_matches_evaluate(self):
        y, s = self._make_scores(100)
        rows = sweep(y, s)
        tau_50_row = next(r for r in rows if abs(r["tau"] - 0.50) < 1e-9)
        pred = [int(sv >= 0.50) for sv in s]
        expected = evaluate(y, pred, s)
        self.assertAlmostEqual(
            tau_50_row["balanced_accuracy"],
            expected["balanced_accuracy"],
            places=10,
        )

    def test_select_optimal_returns_valid_tau(self):
        y, s = self._make_scores(100)
        rows = sweep(y, s)
        best_tau = select_optimal(rows)
        self.assertIn(best_tau, TAU_GRID)
        best_ba = max(r["balanced_accuracy"] for r in rows)
        self.assertEqual(next(r for r in rows if r["tau"] == best_tau)["balanced_accuracy"], best_ba)

    def test_select_optimal_tie_break_lowest_tau(self):
        """When two tau values share the max BalAcc, the lower tau wins (D8)."""
        y = [1, 0, 1, 0] * 50
        # Uniform scores -> tied at every threshold that keeps some positive/negative
        s = [0.50] * 200
        rows = sweep(y, s)
        best_tau = select_optimal(rows)
        # With all scores = 0.50, every threshold gives balanced accuracy 0.50.
        # The lowest tau that gives max BalAcc wins.
        self.assertEqual(best_tau, min(TAU_GRID))


class TestRunThresholdTuning(unittest.TestCase):
    """Integration test and artifact verification."""

    def test_output_has_72_main_rows(self):
        """Main table: 4 methods x 18 tau = 72 rows."""
        if not SCORES_PATH.exists() or not DATA_PATH.exists():
            self.skipTest("Committed score/data files not found")
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "threshold_tuning.csv"
            rows = run_threshold_tuning(
                SCORES_PATH, DATA_PATH, output,
                include_place_only=False,
            )
            self.assertEqual(len(rows), 72)
            with output.open() as f:
                saved = list(csv.DictReader(f))
            self.assertEqual(len(saved), 72)

    def test_fresh_run_matches_committed_csv(self):
        """A fresh run must be byte-for-byte identical to the committed CSV (Finding C)."""
        if not SCORES_PATH.exists() or not DATA_PATH.exists() or not COMMITTED_CSV.exists():
            self.skipTest("Required files not found")
        with tempfile.TemporaryDirectory() as d:
            fresh_output = Path(d) / "threshold_tuning.csv"
            run_threshold_tuning(
                SCORES_PATH, DATA_PATH, fresh_output,
                include_place_only=False,
            )
            self.assertEqual(
                fresh_output.read_bytes(),
                COMMITTED_CSV.read_bytes(),
                "Fresh run output must be byte-identical to committed threshold_tuning.csv",
            )

    def test_output_has_correct_fields(self):
        """Every row must have the full set of FIELDS."""
        if not SCORES_PATH.exists() or not DATA_PATH.exists():
            self.skipTest("Committed score/data files not found")
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "threshold_tuning.csv"
            run_threshold_tuning(SCORES_PATH, DATA_PATH, output)
            with output.open() as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, FIELDS)

    def test_optimal_tau_per_method(self):
        """tau* must match the measured grid optima from committed scores."""
        if not SCORES_PATH.exists() or not DATA_PATH.exists():
            self.skipTest("Committed score/data files not found")
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "threshold_tuning.csv"
            run_threshold_tuning(SCORES_PATH, DATA_PATH, output)
            with output.open() as f:
                rows = list(csv.DictReader(f))
            opt = {
                (r["method"], r["feature_set"]): float(r["tau"])
                for r in rows if r["is_optimal"] == "true"
            }
            # Empirically verified against results/predictions/validation_scores.csv
            self.assertAlmostEqual(opt[("place_history", "main")], 0.30, places=5)
            self.assertAlmostEqual(opt[("logistic", "main")], 0.45, places=5)
            self.assertAlmostEqual(opt[("naive_bayes", "main")], 0.05, places=5)
            self.assertAlmostEqual(opt[("knn", "main")], 0.02, places=5)

    def test_optimal_bal_acc_per_method(self):
        """BalAcc at tau* must match measured values to 4 decimal places."""
        if not SCORES_PATH.exists() or not DATA_PATH.exists():
            self.skipTest("Committed score/data files not found")
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "threshold_tuning.csv"
            run_threshold_tuning(SCORES_PATH, DATA_PATH, output)
            with output.open() as f:
                rows = list(csv.DictReader(f))
            opt_ba = {
                (r["method"], r["feature_set"]): float(r["balanced_accuracy"])
                for r in rows if r["is_optimal"] == "true"
            }
            self.assertAlmostEqual(opt_ba[("place_history", "main")], 0.8566, places=4)
            self.assertAlmostEqual(opt_ba[("logistic", "main")], 0.7168, places=4)
            self.assertAlmostEqual(opt_ba[("naive_bayes", "main")], 0.6591, places=4)
            self.assertAlmostEqual(opt_ba[("knn", "main")], 0.6647, places=4)

    def test_floats_at_6dp(self):
        """Float columns must be formatted to exactly 6 decimal places."""
        if not SCORES_PATH.exists() or not DATA_PATH.exists():
            self.skipTest("Committed score/data files not found")
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "threshold_tuning.csv"
            run_threshold_tuning(SCORES_PATH, DATA_PATH, output)
            with output.open() as f:
                rows = list(csv.DictReader(f))
            float_cols = [
                "accuracy", "balanced_accuracy", "macro_f1",
                "precision", "recall", "f1", "roc_auc",
            ]
            for row in rows:
                for col in float_cols:
                    v = row[col]
                    if v == "":
                        continue
                    # Must be parseable and formatted to 6dp
                    self.assertRegex(v, r"^\d+\.\d{6}$",
                                     msg=f"Column {col} value {v!r} not 6dp")

    def test_is_optimal_exactly_one_per_method_feature_set(self):
        """Exactly one row per (method, feature_set) must have is_optimal=true."""
        if not SCORES_PATH.exists() or not DATA_PATH.exists():
            self.skipTest("Committed score/data files not found")
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "threshold_tuning.csv"
            run_threshold_tuning(SCORES_PATH, DATA_PATH, output)
            with output.open() as f:
                rows = list(csv.DictReader(f))
            from collections import Counter
            counts = Counter(
                (r["method"], r["feature_set"])
                for r in rows if r["is_optimal"] == "true"
            )
            for key, count in counts.items():
                self.assertEqual(count, 1, f"Expected 1 optimal row for {key}, got {count}")

    def test_bit_stable_on_second_run(self):
        """Running twice must produce identical output (bit-stable)."""
        if not SCORES_PATH.exists() or not DATA_PATH.exists():
            self.skipTest("Committed score/data files not found")
        with tempfile.TemporaryDirectory() as d:
            out1 = Path(d) / "run1.csv"
            out2 = Path(d) / "run2.csv"
            run_threshold_tuning(SCORES_PATH, DATA_PATH, out1)
            run_threshold_tuning(SCORES_PATH, DATA_PATH, out2)
            self.assertEqual(out1.read_bytes(), out2.read_bytes())


class TestPlaceOnlyKnn(unittest.TestCase):
    """Place-only KNN integration (D6)."""

    def test_place_only_scores_shape(self):
        """Place-only KNN must return 987 scores matching validation y_true."""
        if not DATA_PATH.exists():
            self.skipTest("flood_data.csv not found")
        y_true, scores = run_place_only_knn(DATA_PATH)
        self.assertEqual(len(y_true), 987)
        self.assertEqual(len(scores), 987)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))

    def test_place_only_optimal_better_than_full_at_tau50(self):
        """Place-only KNN BalAcc at tau=0.50 must exceed full-feature KNN (dilution effect)."""
        if not DATA_PATH.exists():
            self.skipTest("flood_data.csv not found")
        y_true, place_scores = run_place_only_knn(DATA_PATH)
        pred_place = [int(s >= 0.50) for s in place_scores]
        metrics_place = evaluate(y_true, pred_place, place_scores)
        # Full-feature KNN at tau=0.50: 0.567485 (from comparison_main.csv)
        full_feature_ba = 0.567485
        self.assertGreater(
            metrics_place["balanced_accuracy"], full_feature_ba,
            msg="Place-only KNN should outperform full-feature KNN at tau=0.50"
        )

    def test_place_only_with_place_only_flag(self):
        """run_threshold_tuning with --place-only must produce 90 rows."""
        if not SCORES_PATH.exists() or not DATA_PATH.exists():
            self.skipTest("Score/data files not found")
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "threshold_tuning.csv"
            rows = run_threshold_tuning(
                SCORES_PATH, DATA_PATH, output,
                include_place_only=True,
            )
            self.assertEqual(len(rows), 90)  # 4*18 main + 1*18 place-only
            with output.open() as f:
                saved = list(csv.DictReader(f))
            self.assertEqual(len(saved), 90)
            place_rows = [r for r in saved if r["feature_set"] == "place"]
            self.assertEqual(len(place_rows), 18)


if __name__ == "__main__":
    unittest.main()
