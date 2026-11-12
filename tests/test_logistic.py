import unittest
import numpy as np

from src.models.logistic import LogisticRegression


class TestLogisticRegression(unittest.TestCase):

    def test_linearly_separable(self):
        # Linearly separable 2D data
        X = np.array([
            [-2.0, -2.0],
            [-1.5, -1.0],
            [-2.0, -1.0],
            [2.0, 2.0],
            [1.5, 1.0],
            [2.0, 1.0]
        ])
        y = np.array([0, 0, 0, 1, 1, 1])

        model = LogisticRegression(lr=0.5, epochs=500, reg=0.001, seed=42)
        model.fit(X, y)

        preds = model.predict(X)
        np.testing.assert_array_equal(preds, y)

        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (6, 2))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(6))
        # High confidence for class 0 on negative side, class 1 on positive side
        self.assertTrue(np.all(proba[:3, 0] > 0.8))
        self.assertTrue(np.all(proba[3:, 1] > 0.8))

    def test_loss_decreases(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegression(lr=0.1, epochs=100, reg=0.01, seed=42)
        model.fit(X, y)

        self.assertLess(model.loss_history_[-1], model.loss_history_[0])

    def test_balanced_class_weights(self):
        # Heavily imbalanced data: 90 class 0, 10 class 1
        rng = np.random.default_rng(42)
        X0 = rng.standard_normal((90, 2))
        X1 = rng.standard_normal((10, 2)) + 1.5
        X = np.vstack([X0, X1])
        y = np.array([0] * 90 + [1] * 10)

        model = LogisticRegression(
            lr=0.1, epochs=300, reg=0.01, class_weight="balanced", seed=42
        )
        model.fit(X, y)
        preds = model.predict(X)

        # Ensure minority class is predicted with balanced weights
        self.assertGreater(preds[90:].sum(), 0)


if __name__ == "__main__":
    unittest.main()
