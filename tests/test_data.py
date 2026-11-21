
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

    def test_threshold_relabels_without_changing_csv(self):
        rows = []

        for share in (0.5, 1.0, 2.0, 2.1):
            row = self.sample_row()
            row["flood_share"] = share
            rows.append(row)

        with open(self.path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        with open(self.path, "rb") as file:
            original = file.read()

        original_X, original_y, original_metadata = load_data(self.path)
        self.assertEqual(original_y, [1, 1, 1, 1])

        expected = {
            0.5: [0, 1, 1, 1],
            1.0: [0, 0, 1, 1],
            2.0: [0, 0, 0, 1]
        }

        for threshold, labels in expected.items():
            with self.subTest(threshold=threshold):
                X, y, metadata = load_data(
                    self.path,
                    flood_threshold=threshold
                )

                self.assertEqual(y, labels)
                self.assertEqual(X, original_X)
                self.assertEqual(metadata, original_metadata)

        with open(self.path, "rb") as file:
            self.assertEqual(file.read(), original)

    def test_relabel_requires_flood_share(self):
        self.write_csv([self.sample_row()])

        with self.assertRaisesRegex(ValueError, "Missing columns"):
            load_data(self.path, flood_threshold=1.0)

    def test_invalid_threshold(self):
        self.write_csv([self.sample_row()])

        for threshold in (-1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    load_data(self.path, flood_threshold=threshold)

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


    def test_processed_dataset_splits_have_zero_overlap(self):
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "processed", "flood_data.csv"
        )
        self.assertTrue(
            os.path.exists(dataset_path),
            f"Processed dataset {dataset_path} must exist in the repository."
        )

        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            train_keys = set()
            val_keys = set()
            test_keys = set()

            train_features = set()
            val_features = set()
            test_features = set()

            for row in reader:
                coord_key = (row["event_id"], row["lat"], row["lon"])
                feature_key = (
                    row["lat"], row["lon"], row["elevation"], row["flooded"],
                    row["rain_1d"], row["rain_3d"], row["rain_7d"], row["rain_14d"],
                    row["rain_21d"], row["rain_30d"], row["humidity"], row["temperature"],
                    row["soil_wetness"], row["month_sin"], row["month_cos"]
                )
                split = row["split"]
                if split == "train":
                    train_keys.add(coord_key)
                    train_features.add(feature_key)
                elif split == "val":
                    val_keys.add(coord_key)
                    val_features.add(feature_key)
                elif split == "test":
                    test_keys.add(coord_key)
                    test_features.add(feature_key)

            # Assert coordinate and pass identity disjointness
            self.assertEqual(len(train_keys & val_keys), 0)
            self.assertEqual(len(train_keys & test_keys), 0)
            self.assertEqual(len(val_keys & test_keys), 0)

            # Assert feature + target tuple disjointness
            self.assertEqual(len(train_features & val_features), 0)
            self.assertEqual(len(train_features & test_features), 0)
            self.assertEqual(len(val_features & test_features), 0)

            # Assert exact row counts per split
            self.assertEqual(len(train_keys), 1966)
            self.assertEqual(len(val_keys), 987)
            self.assertEqual(len(test_keys), 578)
            self.assertEqual(len(train_keys) + len(val_keys) + len(test_keys), 3531)


if __name__ == "__main__":
    unittest.main()
