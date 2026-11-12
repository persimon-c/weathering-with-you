
import csv
import os
import tempfile
import unittest

from src.data import load_data, FEATURES


class TestData(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp.name, "test.csv")

    def tearDown(self):
        self.temp.cleanup()

    def write_csv(self, rows):
        columns = FEATURES + [
            "flooded", "event_id", "cluster_id", "split"
        ]

        with open(self.path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def sample_row(self):
        row = {feature: "1.0" for feature in FEATURES}
        row.update({
            "flooded": "1",
            "event_id": "20260101_A",
            "cluster_id": "1",
            "split": "test"
        })
        return row

    def test_valid_data(self):
        self.write_csv([self.sample_row()])

        X, y, metadata = load_data(self.path)

        self.assertEqual(len(X), 1)
        self.assertEqual(len(X[0]), 14)
        self.assertEqual(y, [1])
        self.assertEqual(metadata[0]["split"], "test")

    def test_missing_soil_wetness(self):
        row = self.sample_row()
        row["soil_wetness"] = ""

        self.write_csv([row])

        X, _, _ = load_data(self.path)

        index = FEATURES.index("soil_wetness")
        self.assertIsNone(X[0][index])

    def test_invalid_target(self):
        row = self.sample_row()
        row["flooded"] = "2"

        self.write_csv([row])

        with self.assertRaises(ValueError):
            load_data(self.path)

    def test_invalid_split(self):
        row = self.sample_row()
        row["split"] = "validation"

        self.write_csv([row])

        with self.assertRaises(ValueError):
            load_data(self.path)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_data("nonexistent.csv")

    def test_missing_required_feature(self):
        row = self.sample_row()
        row.pop("rain_3d")

        self.write_csv([row])

        with self.assertRaises(ValueError):
            load_data(self.path)


if __name__ == "__main__":
    unittest.main()
