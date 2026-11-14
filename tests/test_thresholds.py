import csv
import tempfile
import unittest
from pathlib import Path

from scripts.run_thresholds import run_thresholds
from src.data import FEATURES


class TestThresholds(unittest.TestCase):
    def test_all_methods_and_thresholds_preserve_data_and_split(self):

        with tempfile.TemporaryDirectory() as directory:

            data = Path(directory) / "data.csv"
            output = Path(directory) / "tables" / "thresholds.csv"
            columns = FEATURES + ["flood_share", "flooded", "event_id", "cluster_id", "split"]

            with data.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()

                for split, count in (("train", 12), ("val", 8), ("test", 4)):

                    for i in range(count):
                        share = (0.5, 1.0, 2.0, 3.0)[i % 4]
                        row = {name: i + 1 for name in FEATURES}
                        row.update(flood_share=share, flooded=int(share > 1),
                                   event_id=f"{split}_{i}", cluster_id=split, split=split)
                        writer.writerow(row)
                        
            original = data.read_bytes()
            rows = run_thresholds(data, output)
            self.assertEqual(data.read_bytes(), original)

            with output.open(newline="") as file:
                saved = list(csv.DictReader(file))

            self.assertEqual(len(saved), 18)
            expected_methods = {"majority", "rainfall", "place_history", "naive_bayes", "logistic", "knn"}

            for threshold, positive in ((0.5, 9), (1.0, 6), (2.0, 3)):
                group = [row for row in rows if row["threshold_percent"] == threshold]
                self.assertEqual({row["method"] for row in group}, expected_methods)
                self.assertEqual(len(group), 6)

                for row in group:
                    self.assertEqual(row["split"], "val")
                    self.assertEqual(row["train_flooded"], positive)
                    self.assertEqual(row["train_flooded_share"], positive / 12)
                    self.assertEqual(row["result_type"], "main" if threshold == 1 else "sensitivity")

                    for split, count in (("train", 12), ("val", 8), ("test", 4)):
                        self.assertEqual(row[f"{split}_rows"], count)
                    self.assertEqual(sum(row[key] for key in ("tp", "tn", "fp", "fn")), 8)


if __name__ == "__main__":
    unittest.main()
