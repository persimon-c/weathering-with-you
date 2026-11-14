import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

from scripts.run_feature_groups import run_feature_groups
from src.feature_groups import (
    FEATURE_GROUPS,
    PREPROCESSED_FEATURES,
    select_features,
)


class TestFeatureGroups(unittest.TestCase):

    def test_group_names_and_sizes(self):
        self.assertEqual(
            list(FEATURE_GROUPS),
            ["place", "place_weather", "place_weather_season"]
        )
        self.assertEqual(len(PREPROCESSED_FEATURES), 15)
        self.assertEqual(len(FEATURE_GROUPS["place"]), 3)
        self.assertEqual(len(FEATURE_GROUPS["place_weather"]), 13)
        self.assertEqual(len(FEATURE_GROUPS["place_weather_season"]), 15)

    def test_place_weather_excludes_season_and_includes_missing_flag(self):
        group = FEATURE_GROUPS["place_weather"]

        self.assertIn("soil_wetness_missing", group)
        self.assertNotIn("month_sin", group)
        self.assertNotIn("month_cos", group)

    def test_select_features_uses_processed_column_order(self):
        X = np.arange(15, dtype=float).reshape(1, -1)

        np.testing.assert_array_equal(
            select_features(X, "place"),
            X[:, [0, 1, 2]]
        )
        np.testing.assert_array_equal(
            select_features(X, "place_weather"),
            X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14]]
        )
        np.testing.assert_array_equal(
            select_features(X, "place_weather_season"),
            X
        )

    def test_unknown_group_is_rejected(self):
        with self.assertRaises(ValueError):
            select_features(np.zeros((1, 15)), "weather_only")

    def test_runner_writes_nine_rows_and_group_metadata(self):
        rng = np.random.default_rng(7)
        X_train = rng.normal(size=(60, 15))
        y_train = np.array([0, 1] * 30)
        X_val = rng.normal(size=(20, 15))
        y_val = np.array([0, 1] * 10)

        with tempfile.TemporaryDirectory() as temp_dir:
            runs_csv = os.path.join(temp_dir, "runs.csv")
            output_csv = os.path.join(temp_dir, "feature_groups.csv")

            with redirect_stdout(io.StringIO()):
                rows = run_feature_groups(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    runs_csv,
                    output_csv,
                    seed=42,
                )

            self.assertEqual(len(rows), 9)

            with open(output_csv, newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                self.assertEqual(
                    reader.fieldnames,
                    [
                        "method",
                        "group",
                        "balanced_accuracy",
                        "macro_f1",
                        "roc_auc",
                    ]
                )
                table_rows = list(reader)

            self.assertEqual(len(table_rows), 9)

            expected_pairs = [
                (method, group)
                for group in FEATURE_GROUPS
                for method in ("naive_bayes", "logistic", "knn")
            ]
            self.assertEqual(
                [(row["method"], row["group"]) for row in table_rows],
                expected_pairs
            )

            for row in table_rows:
                for column in ("balanced_accuracy", "macro_f1", "roc_auc"):
                    self.assertNotEqual(row[column], "")

            with open(runs_csv, newline="", encoding="utf-8") as file:
                run_rows = list(csv.DictReader(file))

            self.assertEqual(len(run_rows), 9)

            for row in run_rows:
                self.assertEqual(row["split"], "val")
                self.assertEqual(row["n_train"], "60")
                self.assertEqual(row["n_eval"], "20")
                self.assertIn(
                    "group",
                    json.loads(row["hyperparameters"])
                )


if __name__ == "__main__":
    unittest.main()
