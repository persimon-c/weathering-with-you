import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_per_cluster import run_per_cluster
from src.data import FEATURES
from src.metrics import evaluate_by_cluster
from src.preprocess import transform_preprocess


class TestPerCluster(unittest.TestCase):
    def test_six_methods_five_clusters_and_single_class_metrics(self):

        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.csv"
            output = Path(directory) / "tables/per_cluster.csv"
            columns = FEATURES + ["flooded", "event_id", "cluster_id", "split"]

            with data.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()

                # No flood_share column: this runner uses saved main labels.
                # No test split: test data is not needed for this analysis.

                for i in range(12):
                    row = {feature: i + 1 for feature in FEATURES}
                    row.update(flooded=i % 2, event_id=f"train_{i}",
                               cluster_id="train", split="train")
                    writer.writerow(row)

                for cluster in range(5):
                    for i in range(4):

                        row = {feature: i + cluster + 1 for feature in FEATURES}
                        row["soil_wetness"] = "" if i == 0 else "0.5"
                        row.update(flooded=0 if cluster == 0 else i % 2,
                                   event_id=f"val_{cluster}_{i}",
                                   cluster_id=str(cluster), split="val")
                        writer.writerow(row)
            original = data.read_bytes()

            with patch("scripts.run_per_cluster.evaluate_by_cluster",
                       wraps=evaluate_by_cluster) as evaluate_mock, patch(
                           "scripts.run_per_cluster.transform_preprocess",
                           wraps=transform_preprocess) as transform_mock:
                rows = run_per_cluster(data, output)
            self.assertEqual(evaluate_mock.call_count, 6)
            self.assertEqual(transform_mock.call_count, 2)
            self.assertEqual([len(call.args[0]) for call in
                              transform_mock.call_args_list], [12, 20])
            
            for call in evaluate_mock.call_args_list:
                self.assertEqual(len(call.args[0]), 20)
                self.assertEqual(call.args[3],
                                 [str(c) for c in range(5) for _ in range(4)])
            self.assertEqual(data.read_bytes(), original)
            self.assertEqual(len(rows), 30)
            expected = {"majority", "rainfall", "place_history",
                        "naive_bayes", "logistic", "knn"}
            self.assertEqual({row["method"] for row in rows}, expected)

            for method in expected:
                group = [row for row in rows if row["method"] == method]
                self.assertEqual({row["cluster_id"] for row in group},
                                 {"0", "1", "2", "3", "4"})
                self.assertEqual(sum(row["n_eval"] for row in group), 20)

                for row in group:
                    self.assertEqual(row["split"], "val")
                    self.assertEqual(row["n_train"], 12)
                    self.assertEqual(row["threshold_percent"], 1.0)
                    self.assertEqual(row["n_eval"], 4)
                    self.assertEqual(sum(row[k] for k in ("tp", "tn", "fp", "fn")), 4)
                    self.assertEqual(row["tp"] + row["fn"], row["n_flooded"])
                    self.assertEqual(row["flooded_share"], row["n_flooded"] / 4)

                    if row["cluster_id"] == "0":
                        self.assertIsNone(row["balanced_accuracy"])
                        self.assertIsNone(row["macro_f1"])
                        self.assertIsNone(row["roc_auc"])

            with output.open(newline="", encoding="utf-8") as file:
                saved = list(csv.DictReader(file))

            self.assertEqual(len(saved), 30)

            for row in saved:
                if row["cluster_id"] == "0":
                    self.assertEqual(row["roc_auc"], "")


if __name__ == "__main__":
    unittest.main()
