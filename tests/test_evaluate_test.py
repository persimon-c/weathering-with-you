"""Tests for scripts/evaluate_test.py. Validation stands in for test; test rows are never scored."""

import csv
from datetime import timedelta
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_test import FREEZE, evaluate_split, gate_problems, output_paths
from scripts.run_bootstrap import load_taus
from src.data import load_data
from src.split import split_data

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "flood_data.csv"
TAUS = ROOT / "results" / "tables" / "threshold_tuning.csv"


class TestGate(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = output_paths(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_open_after_freeze_on_clean_main(self):
        self.assertEqual(gate_problems(FREEZE, "main", False, self.paths), [])

    def test_closed_before_freeze(self):
        problems = gate_problems(FREEZE - timedelta(seconds=1), "main", False, self.paths)
        self.assertEqual(len(problems), 1)
        self.assertIn("frozen", problems[0])

    def test_closed_off_main_or_dirty(self):
        self.assertEqual(len(gate_problems(FREEZE, "feat/x", True, self.paths)), 2)

    def test_closed_once_any_output_exists(self):
        self.paths["map"].parent.mkdir(parents=True)
        self.paths["map"].touch()
        problems = gate_problems(FREEZE, "main", False, self.paths)
        self.assertEqual(len(problems), 1)
        self.assertIn("already scored once", problems[0])


class TestDryRun(unittest.TestCase):

    def test_validation_stand_in_reproduces_tuned_results(self):
        splits = split_data(*load_data(DATA))
        with tempfile.TemporaryDirectory() as tmp:
            paths = output_paths(tmp)
            comparison, sizes, unseen = evaluate_split(
                splits["train"], splits["val"], load_taus(TAUS), paths, "dry run")

            for key in ("roc", "map"):
                self.assertTrue(paths[key].exists())
            with paths["per_cluster"].open(encoding="utf-8", newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 6 * 5)
            with paths["scores"].open(encoding="utf-8", newline="") as f:
                events = {r["event_id"] for r in csv.DictReader(f)}

        self.assertTrue(all(e.startswith("2025") for e in events))
        by_method = {r["method"]: r for r in comparison}
        # Matches comparison_validation_final.csv (tuned) and comparison_main.csv (baselines).
        self.assertEqual(by_method["place_history"]["balanced_accuracy"], "0.856608")
        self.assertEqual(by_method["place_history"]["tau"], "0.30")
        self.assertEqual(by_method["logistic"]["balanced_accuracy"], "0.716786")
        self.assertEqual(by_method["rainfall"]["balanced_accuracy"], "0.508853")
        self.assertEqual(by_method["majority"]["tau"], "")
        self.assertEqual(sum(sizes.values()), 987)
        self.assertEqual(unseen, 2)


if __name__ == "__main__":
    unittest.main()
