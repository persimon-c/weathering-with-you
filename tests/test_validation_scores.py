import collections
import csv
import os
import tempfile
import unittest

from scripts.export_validation_scores import (
    METHODS,
    write_validation_scores,
)
from src.data import FEATURES


class TestValidationScores(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp.name, "validation_scores.csv")

        self.y_true = [0, 1]
        self.metadata = [
            {"event_id": "2025_A", "cluster_id": "1"},
            {"event_id": "2025_B", "cluster_id": "2"},
        ]

        self.X_raw = []
        for latitude, longitude in ((14.5, 121.0), (15.0, 122.0)):
            row = [0.0] * len(FEATURES)
            row[FEATURES.index("lat")] = latitude
            row[FEATURES.index("lon")] = longitude
            self.X_raw.append(row)

    def tearDown(self):
        self.temp.cleanup()

    def method_predictions(self):
        return [
            (method, [0.1, 0.9], [0, 1])
            for method in METHODS
        ]

    def test_writes_one_row_per_method_and_sample(self):
        write_validation_scores(
            self.path,
            self.y_true,
            self.metadata,
            self.X_raw,
            self.method_predictions(),
        )

        with open(self.path, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        with open(self.path, "rb") as file:
            raw = file.read()

        counts = collections.Counter(row["method"] for row in rows)

        self.assertEqual(len(rows), len(METHODS) * len(self.y_true))
        self.assertTrue(all(counts[method] == 2 for method in METHODS))
        self.assertEqual(rows[0]["actual"], "0")
        self.assertEqual(rows[0]["score"], "0.1")
        self.assertEqual(rows[0]["predicted"], "0")
        self.assertEqual(rows[0]["event_id"], "2025_A")
        self.assertEqual(rows[0]["lat"], "14.5")
        self.assertNotIn(b"\r\n", raw)

    def test_rejects_wrong_method_order(self):
        predictions = self.method_predictions()
        predictions[0], predictions[1] = predictions[1], predictions[0]

        with self.assertRaises(ValueError):
            write_validation_scores(
                self.path,
                self.y_true,
                self.metadata,
                self.X_raw,
                predictions,
            )

    def test_rejects_invalid_score_before_writing(self):
        predictions = self.method_predictions()
        predictions[0] = ("majority", [0.1, 1.5], [0, 1])

        with self.assertRaises(ValueError):
            write_validation_scores(
                self.path,
                self.y_true,
                self.metadata,
                self.X_raw,
                predictions,
            )

        self.assertFalse(os.path.exists(self.path))

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            write_validation_scores(
                self.path,
                self.y_true,
                self.metadata[:1],
                self.X_raw,
                self.method_predictions(),
            )


if __name__ == "__main__":
    unittest.main()
