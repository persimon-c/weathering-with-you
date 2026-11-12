import math
import unittest
import numpy as np

from src.models.naive_bayes import GaussianNaiveBayes


class TestGaussianNaiveBayes(unittest.TestCase):

    def test_asymmetric_hand_computed_parameters(self):
        # 4 hand-made rows, 2 features, 2 classes: 3 rows of class 0, 1 row of class 1
        # Class 0: [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]
        # Class 1: [10.0, 10.0]
        X = np.array([
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [10.0, 10.0]
        ])
        y = np.array([0, 0, 0, 1])

        var_floor = 1e-9
        model = GaussianNaiveBayes(var_floor=var_floor)
        model.fit(X, y)

        # Classes
        np.testing.assert_array_equal(model.classes_, [0, 1])

        # Hand-computed priors: P(0) = 3/4 = 0.75, P(1) = 1/4 = 0.25
        expected_log_priors = np.array([math.log(0.75), math.log(0.25)])
        np.testing.assert_allclose(model.log_priors_, expected_log_priors)

        # Hand-computed means:
        # Class 0: mean of [1, 2], [2, 4], [3, 6] -> [2.0, 4.0]
        # Class 1: [10.0, 10.0]
        expected_means = np.array([
            [2.0, 4.0],
            [10.0, 10.0]
        ])
        np.testing.assert_allclose(model.means_, expected_means)

        # Hand-computed population variances:
        # Class 0:
        #   feat 0: ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        #   feat 1: ((2-4)^2 + (4-4)^2 + (6-4)^2) / 3 = 8/3
        # Class 1:
        #   Single sample variance is 0.0 -> clamped to var_floor
        expected_variances = np.array([
            [2.0 / 3.0, 8.0 / 3.0],
            [var_floor, var_floor]
        ])
        np.testing.assert_allclose(model.variances_, expected_variances)

    def test_predictions(self):
        # Training data
        X_train = np.array([
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [10.0, 10.0]
        ])
        y_train = np.array([0, 0, 0, 1])

        model = GaussianNaiveBayes()
        model.fit(X_train, y_train)

        # Point near class 0 mean [2, 4] should predict class 0
        # Point near class 1 mean [10, 10] should predict class 1
        X_test = np.array([
            [2.0, 4.0],
            [10.0, 10.0]
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
