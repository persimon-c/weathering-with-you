import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_error_analysis import (
    elevation_group, rain_group, summarize_errors, run_error_analysis,
)
from src.data import FEATURES


class TestErrorAnalysis(unittest.TestCase):
    def sample_validation(self):
        X = []
        for i in range(4):
            row = [1.0] * len(FEATURES)
            for name, value in {
                "lat": 14.25 if i < 2 else 15.25,
                "lon": 121.25, "elevation": 50 if i < 2 else 500,
                "rain_3d": 10 if i < 2 else 50,
            }.items():
                row[FEATURES.index(name)] = value
            X.append(row)
        return {
            "X": X, "y": [1, 1, 0, 0],
            "metadata": [dict(event_id=f"pass_{i}",
                              cluster_id="a" if i < 2 else "b", split="val")
                         for i in range(4)],
        }

    def predictions(self):
        return [("example", 0, {}, [0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])]

    def test_counts_rates_and_error_row_identity(self):
        errors, groups = summarize_errors(self.sample_validation(), self.predictions())
        self.assertEqual([r["event_id"] for r in errors], ["pass_0", "pass_2"])
        self.assertEqual([r["error_type"] for r in errors], ["fn", "fp"])
        for kind in ("cluster", "cell", "elevation", "rain_3d"):
            rows = [r for r in groups if r["group_type"] == kind]
            self.assertEqual(sum(r["n_eval"] for r in rows), 4)
            for outcome in ("tp", "tn", "fp", "fn"):
                self.assertEqual(sum(r[outcome] for r in rows), 1)
        clusters = {r["group"]: r for r in groups if r["group_type"] == "cluster"}
        self.assertEqual(clusters["a"]["false_negative_rate"], 0.5)
        self.assertIsNone(clusters["a"]["false_positive_rate"])
        self.assertEqual(clusters["b"]["false_positive_rate"], 0.5)
        self.assertIsNone(clusters["b"]["false_negative_rate"])
        self.assertEqual(clusters["a"]["error_rate"], 0.5)

    def test_bin_boundaries(self):
        self.assertEqual([elevation_group(v) for v in (99, 100, 499, 500)],
                         ["below_100m", "100_to_below_500m",
                          "100_to_below_500m", "500m_and_above"])
        self.assertEqual([rain_group(v) for v in (19, 20, 49, 50)],
                         ["below_20mm", "20_to_below_50mm",
                          "20_to_below_50mm", "50mm_and_above"])

    def test_rejects_test_rows_and_mismatched_predictions(self):
        val = self.sample_validation()
        val["metadata"][0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation rows only"):
            summarize_errors(val, self.predictions())
        with self.assertRaisesRegex(ValueError, "match validation"):
            summarize_errors(self.sample_validation(), [("bad", 0, {}, [0], [0.1])])

    def test_runner_saves_only_validation_errors_and_blank_undefined_rates(self):
        val = self.sample_validation()
        train = {"X": [[2.0] * len(FEATURES)], "y": [0],
                 "metadata": [dict(event_id="train", cluster_id="train", split="train")]}
        test_info = dict(event_id="test", cluster_id="test", split="test")
        loaded = (train["X"] + val["X"] + [[999.0] * len(FEATURES)],
                  train["y"] + val["y"] + [1],
                  train["metadata"] + val["metadata"] + [test_info])
        
        with tempfile.TemporaryDirectory() as directory, patch(
                "scripts.run_error_analysis.load_data", return_value=loaded), patch(
                "scripts.run_error_analysis.validation_predictions",
                return_value=self.predictions()) as predictor:
            run_error_analysis("unused.csv", directory)
            self.assertEqual(predictor.call_args.args[:2], (train, val))

            with (Path(directory) / "validation_errors.csv").open(newline="") as file:
                errors = list(csv.DictReader(file))

            self.assertEqual({r["event_id"] for r in errors}, {"pass_0", "pass_2"})

            with (Path(directory) / "error_groups.csv").open(newline="") as file:
                groups = list(csv.DictReader(file))

            a = next(r for r in groups if r["group_type"] == "cluster" and r["group"] == "a")
            self.assertEqual(a["false_positive_rate"], "")


if __name__ == "__main__":
    unittest.main()
