import csv
import json
import os
import tempfile
import unittest

from scripts.compare import (
    METHODS,
    select_main_rows,
    write_csv,
    write_markdown,
)


class TestCompare(unittest.TestCase):

    def make_row(self, method, commit="abc123", group=None, k=None, **overrides):
        settings = {}

        if group is not None:
            settings["group"] = group

        if k is not None:
            settings["k"] = k

        row = {
            "run_id": f"{method}_{commit}_{group}_{k}",
            "git_commit": commit,
            "method": method,
            "split": "val",
            "seed": "42",
            "hyperparameters": json.dumps(settings),
            "n_train": "1966",
            "n_eval": "987",
            "accuracy": "0.800000",
            "balanced_accuracy": "0.700000",
            "precision": "0.600000",
            "recall": "0.500000",
            "f1": "0.545455",
            "macro_f1": "0.650000",
            "roc_auc": "0.750000",
            "tp": "92",
            "tn": "698",
            "fp": "105",
            "fn": "92",
        }

        row.update(overrides)
        return row

    def main_rows(self):
        return [
            self.make_row("majority"),
            self.make_row("rainfall"),
            self.make_row("place_history"),
            self.make_row("naive_bayes"),
            self.make_row("logistic"),
            self.make_row("knn", k=5),
        ]

    def test_selects_exactly_one_row_per_main_method(self):
        rows = self.main_rows()

        selected = select_main_rows(rows, "abc123")

        self.assertEqual(
            [row["method"] for row in selected],
            list(METHODS)
        )

    def test_ignores_other_commits_and_group_rows(self):
        rows = self.main_rows()

        # The same methods at another commit must not be mixed into the table.
        for method in METHODS:
            rows.append(
                self.make_row(method, commit="other", k=5 if method == "knn" else None)
            )

        # Feature-group rows repeat Naive Bayes, logistic and KNN with k = 5.
        rows.append(self.make_row("naive_bayes", group="place"))
        rows.append(self.make_row("logistic", group="place_weather"))
        rows.append(self.make_row("knn", group="place", k=5))

        selected = select_main_rows(rows, "abc123")

        self.assertEqual(len(selected), 6)
        self.assertTrue(
            all(row["git_commit"] == "abc123" for row in selected)
        )
        self.assertTrue(
            all("group" not in json.loads(row["hyperparameters"]) for row in selected)
        )

    def test_ignores_other_knn_values(self):
        rows = self.main_rows()
        rows.append(self.make_row("knn", k=11))
        rows.append(self.make_row("knn", k=21))

        selected = select_main_rows(rows, "abc123")
        knn_row = selected[-1]

        self.assertEqual(
            json.loads(knn_row["hyperparameters"])["k"],
            5
        )

    def test_rejects_duplicate_knn_five(self):
        rows = self.main_rows()
        rows.append(self.make_row("knn", k=5, run_id="duplicate"))

        with self.assertRaises(ValueError):
            select_main_rows(rows, "abc123")

    def test_rejects_missing_method(self):
        rows = self.main_rows()[:-1]

        with self.assertRaises(ValueError):
            select_main_rows(rows, "abc123")

    def test_rejects_different_validation_sizes(self):
        rows = self.main_rows()
        rows[0]["n_eval"] = "986"

        with self.assertRaises(ValueError):
            select_main_rows(rows, "abc123")

    def test_writes_csv_and_markdown(self):
        rows = select_main_rows(self.main_rows(), "abc123")

        with tempfile.TemporaryDirectory() as directory:
            csv_path = os.path.join(directory, "comparison_main.csv")
            md_path = os.path.join(directory, "comparison_main.md")

            write_csv(rows, csv_path)
            write_markdown(rows, md_path, "abc123")

            with open(csv_path, newline="", encoding="utf-8") as file:
                written = list(csv.DictReader(file))

            with open(csv_path, "rb") as file:
                csv_bytes = file.read()

            with open(md_path, encoding="utf-8") as file:
                markdown = file.read()

        self.assertEqual(len(written), 6)
        self.assertEqual(written[0]["method"], "majority")
        self.assertEqual(written[0]["balanced_accuracy"], "0.700000")
        self.assertEqual(written[0]["tp"], "92")
        self.assertNotIn(b"\r\n", csv_bytes)
        self.assertIn("Source commit `abc123`", markdown)
        self.assertIn("KNN (k = 5)", markdown)
        self.assertIn("987 rows", markdown)


if __name__ == "__main__":
    unittest.main()
