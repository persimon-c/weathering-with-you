
import unittest

from src.metrics import (
    confusion_matrix,
    evaluate,
    roc_auc,
    evaluate_by_cluster
)


class TestMetrics(unittest.TestCase):

    def test_confusion_matrix(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 1, 0]

        self.assertEqual(
            confusion_matrix(y_true, y_pred),
            (1, 1, 1, 1)
        )

    def test_balanced_accuracy(self):
        result = evaluate(
            [1, 1, 0, 0],
            [1, 0, 1, 0]
        )

        self.assertAlmostEqual(
            result["balanced_accuracy"], 0.5
        )

    def test_macro_f1(self):
        result = evaluate(
            [1, 1, 0, 0],
            [1, 0, 1, 0]
        )

        self.assertAlmostEqual(
            result["macro_f1"], 0.5
        )

    def test_perfect_auc(self):
        self.assertEqual(
            roc_auc(
                [0, 0, 1, 1],
                [0.1, 0.2, 0.8, 0.9]
            ),
            1.0
        )

    def test_tied_auc(self):
        self.assertEqual(
            roc_auc(
                [0, 1],
                [0.5, 0.5]
            ),
            0.5
        )

    def test_single_class_auc(self):
        self.assertIsNone(
            roc_auc([1, 1], [0.8, 0.9])
        )

    def test_cluster_evaluation(self):
        result = evaluate_by_cluster(
            [1, 0, 1, 0],
            [1, 0, 0, 0],
            [0.9, 0.1, 0.4, 0.2],
            [1, 1, 2, 2]
        )

        self.assertEqual(len(result["clusters"]), 2)
        self.assertEqual(
            result["clusters"][1]["accuracy"], 1.0
        )
        self.assertEqual(
            result["clusters"][2]["accuracy"], 0.5
        )


    # ---------------------------------
    # Confusion matrix edge cases
    # ---------------------------------

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            confusion_matrix([1, 0, 1], [1, 0])

    def test_empty_inputs(self):
        with self.assertRaises(ValueError):
            confusion_matrix([], [])

    def test_invalid_true_label(self):
        with self.assertRaises(ValueError):
            confusion_matrix([1, 2, 0], [1, 0, 0])

    def test_invalid_predicted_label(self):
        with self.assertRaises(ValueError):
            confusion_matrix([1, 0, 0], [1, -1, 0])

    # ---------------------------------
    # ROC AUC edge cases
    # ---------------------------------

    def test_auc_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            roc_auc([0, 1], [0.2])

    def test_auc_empty_inputs(self):
        with self.assertRaises(ValueError):
            roc_auc([], [])

    def test_auc_invalid_label(self):
        with self.assertRaises(ValueError):
            roc_auc([0, 2], [0.2, 0.8])

    def test_auc_invalid_probability(self):
        with self.assertRaises(ValueError):
            roc_auc([0, 1], [-0.1, 1.2])

    def test_auc_all_negative(self):
        self.assertIsNone(
            roc_auc([0, 0, 0], [0.1, 0.2, 0.3])
        )

    def test_auc_all_positive(self):
        self.assertIsNone(
            roc_auc([1, 1, 1], [0.7, 0.8, 0.9])
        )

    def test_auc_completely_reversed(self):
        self.assertEqual(
            roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]),
            0.0
        )

    def test_auc_all_tied(self):
        self.assertEqual(
            roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]),
            0.5
        )

    # ---------------------------------
    # Evaluation edge cases
    # ---------------------------------

    def test_perfect_predictions(self):
        result = evaluate([0, 0, 1, 1], [0, 0, 1, 1])

        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["balanced_accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)

    def test_all_predictions_negative(self):
        result = evaluate([0, 0, 1, 1], [0, 0, 0, 0])

        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f1"], 0.0)
        self.assertEqual(result["balanced_accuracy"], 0.5)

    def test_all_predictions_positive(self):
        result = evaluate([0, 0, 1, 1], [1, 1, 1, 1])

        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["balanced_accuracy"], 0.5)

    def test_evaluate_without_scores(self):
        result = evaluate([0, 1], [0, 1])

        self.assertIsNone(result["roc_auc"])

    # ---------------------------------
    # Cluster evaluation edge cases
    # ---------------------------------

    def test_cluster_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            evaluate_by_cluster(
                [0, 1, 0],
                [0, 1, 0],
                [0.1, 0.9, 0.2],
                [1, 1]
            )

    def test_cluster_mismatched_scores(self):
        with self.assertRaises(ValueError):
            evaluate_by_cluster(
                [0, 1, 0],
                [0, 1, 0],
                [0.1, 0.9],
                [1, 1, 2]
            )


    def test_single_class_cluster(self):
        result = evaluate_by_cluster(
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0.1, 0.2, 0.8, 0.9],
            [1, 1, 2, 2]
        )

        for cluster_id in (1, 2):
            scores = result["clusters"][cluster_id]

            self.assertIsNone(scores["balanced_accuracy"])
            self.assertIsNone(scores["macro_f1"])
            self.assertIsNone(scores["roc_auc"])
            self.assertEqual(scores["accuracy"], 1.0)


    def test_cluster_without_scores(self):
        result = evaluate_by_cluster(
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            None,
            [1, 1, 2, 2]
        )

        self.assertIsNone(result["overall"]["roc_auc"])
        self.assertIsNone(result["clusters"][1]["roc_auc"])
        self.assertIsNone(result["clusters"][2]["roc_auc"])

    def test_single_cluster(self):
        result = evaluate_by_cluster(
            [0, 1],
            [0, 1],
            [0.1, 0.9],
            [5, 5]
        )

        self.assertEqual(len(result["clusters"]), 1)
        self.assertEqual(
            result["clusters"][5]["accuracy"], 1.0
        )
        self.assertEqual(
            result["overall"]["accuracy"], 1.0
        )



    def test_hand_checked_metrics(self):
        y_true = [1, 1, 1, 0, 0]
        y_pred = [1, 1, 0, 1, 0]

        result = evaluate(y_true, y_pred)

        # TP = 2, TN = 1, FP = 1, FN = 1
        self.assertEqual(
            result["confusion_matrix"],
            {"tp": 2, "tn": 1, "fp": 1, "fn": 1}
        )

        self.assertAlmostEqual(result["accuracy"], 3 / 5)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertAlmostEqual(result["balanced_accuracy"], 7 / 12)
        self.assertAlmostEqual(result["macro_f1"], 7 / 12)

    def test_hand_checked_auc(self):
        y_true = [1, 1, 0, 0]
        y_score = [0.9, 0.4, 0.6, 0.2]

        # 3 correctly ordered positive-negative pairs out of 4.
        self.assertAlmostEqual(
            roc_auc(y_true, y_score),
            0.75
        )

    
    def test_unequal_precision_and_recall(self):
        result = evaluate(
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0]
        )

        # TP=1, FN=2, FP=0, TN=2
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1 / 3)
        self.assertAlmostEqual(result["balanced_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["f1"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 7 / 12)
    



if __name__ == "__main__":
    unittest.main()
