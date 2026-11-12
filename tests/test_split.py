
import unittest

from src.split import split_data


class TestSplit(unittest.TestCase):

    def setUp(self):
        self.X = [[1], [2], [3], [4]]
        self.y = [0, 1, 0, 1]
        self.metadata = [
            {"split": "train", "cluster_id": "1"},
            {"split": "train", "cluster_id": "1"},
            {"split": "val", "cluster_id": "2"},
            {"split": "test", "cluster_id": "3"}
        ]

    def test_correct_splits(self):
        result = split_data(self.X, self.y, self.metadata)

        self.assertEqual(result["train"]["X"], [[1], [2]])
        self.assertEqual(result["val"]["X"], [[3]])
        self.assertEqual(result["test"]["X"], [[4]])

    def test_labels_preserved(self):
        result = split_data(self.X, self.y, self.metadata)

        self.assertEqual(result["train"]["y"], [0, 1])
        self.assertEqual(result["val"]["y"], [0])
        self.assertEqual(result["test"]["y"], [1])

    def test_metadata_preserved(self):
        result = split_data(self.X, self.y, self.metadata)

        self.assertEqual(
            result["test"]["metadata"][0]["cluster_id"],
            "3"
        )

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            split_data(self.X, self.y[:2], self.metadata)

    def test_empty_dataset(self):
        with self.assertRaises(ValueError):
            split_data([], [], [])

    def test_invalid_split(self):
        metadata = [dict(row) for row in self.metadata]
        metadata[0]["split"] = "validation"

        with self.assertRaises(ValueError):
            split_data(self.X, self.y, metadata)


if __name__ == "__main__":
    unittest.main()
