import unittest
import numpy as np

from src.models.knn import KNearestNeighbors


class TestKNearestNeighbors(unittest.TestCase):

    def setUp(self):
        # Two groups on a line: 0, 1, 2 are dry and 10, 11 are flooded.
        self.X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
        self.y = np.array([0, 0, 0, 1, 1])

    def test_share_of_flooded_neighbors(self):
        model = KNearestNeighbors(k=3).fit(self.X, self.y)
        proba = model.predict_proba(np.array([[0.5], [10.5], [6.5]]))
        # 6.5 is nearest to 10, 2 and 11 (distances 3.5, 4.5, 4.5).
        np.testing.assert_allclose(proba[:, 1], [0.0, 2 / 3, 2 / 3])
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(3))
        np.testing.assert_array_equal(model.predict(np.array([[0.5], [10.5]])), [0, 1])

    def test_k_one_returns_own_label(self):
        model = KNearestNeighbors(k=1).fit(self.X, self.y)
        np.testing.assert_array_equal(model.predict(self.X), self.y)

    def test_k_larger_than_training_set(self):
        with self.assertRaises(ValueError):
            KNearestNeighbors(k=6).fit(self.X, self.y)


if __name__ == "__main__":
    unittest.main()
