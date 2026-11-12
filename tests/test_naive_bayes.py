import math
import unittest
import numpy as np

from src.models.naive_bayes import GaussianNaiveBayes


class TestGaussianNaiveBayes(unittest.TestCase):

    def test_hand_computed_parameters(self):
        # 4 hand-made rows, 2 features, 2 classes
        # Class 0: [1.0, 2.0], [3.0, 4.0]
        # Class 1: [5.0, 6.0], [7.0, 8.0]
        X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
        ])
        y = np.array([0, 0, 1, 1])

        model = GaussianNaiveBayes(var_floor=1e-9)
        model.fit(X, y)

        # Classes
        np.testing.assert_array_equal(model.classes_, [0, 1])

        # Hand-computed priors: P(0) = 2/4 = 0.5, P(1) = 2/4 = 0.5
        expected_log_priors = np.array([math.log(0.5), math.log(0.5)])
        np.testing.assert_allclose(model.log_priors_, expected_log_priors)

        # Hand-computed means:
        # Class 0: mean of [1, 2] and [3, 4] -> [2.0, 3.0]
        # Class 1: mean of [5, 6] and [7, 8] -> [6.0, 7.0]
        expected_means = np.array([
            [2.0, 3.0],
            [6.0, 7.0]
        ])
        np.testing.assert_allclose(model.means_, expected_means)

        # Hand-computed population variances:
        # Class 0: var([1, 3]) = ((1-2)^2 + (3-2)^2) / 2 = 1.0
        #          var([2, 4]) = ((2-3)^2 + (4-3)^2) / 2 = 1.0
        # Class 1: var([5, 7]) = 1.0, var([6, 8]) = 1.0
        expected_variances = np.array([
            [1.0, 1.0],
            [1.0, 1.0]
        ])
        np.testing.assert_allclose(model.variances_, expected_variances)

    def test_predictions(self):
        # Training data
        X_train = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
        ])
        y_train = np.array([0, 0, 1, 1])

        model = GaussianNaiveBayes()
        model.fit(X_train, y_train)

        # Point near class 0 mean [2, 3] should predict class 0
        # Point near class 1 mean [6, 7] should predict class 1
        X_test = np.array([
            [2.0, 3.0],
            [6.0, 7.0]
        ])
        preds = model.predict(X_test)
        np.testing.assert_array_equal(preds, [0, 1])

        proba = model.predict_proba(X_test)
        self.assertEqual(proba.shape, (2, 2))
        # Probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), [1.0, 1.0])
        self.assertGreater(proba[0, 0], proba[0, 1])
        self.assertGreater(proba[1, 1], proba[1, 0])

    def test_variance_floor(self):
        # Constant feature has 0 variance, should be clamped to var_floor
        X = np.array([
            [5.0, 1.0],
            [5.0, 3.0]
        ])
        y = np.array([0, 0])

        model = GaussianNaiveBayes(var_floor=1e-5)
        model.fit(X, y)
        self.assertEqual(model.variances_[0, 0], 1e-5)


if __name__ == "__main__":
    unittest.main()
