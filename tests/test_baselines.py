
import unittest

from src.data import FEATURES
from src.baselines import (
    majority_predict,
    majority_predict_proba,
    rain_predict,
    choose_rain_cutoff
)


class TestBaselines(unittest.TestCase):

    def setUp(self):
        self.rain_index = FEATURES.index("rain_3d")

    def make_rows(self, rainfall):
        rows = []

        for value in rainfall:
            row = [0.0] * len(FEATURES)
            row[self.rain_index] = value
            rows.append(row)

        return rows

    def test_majority_predictions(self):
        X = self.make_rows([0, 10, 50])

        self.assertEqual(
            majority_predict(X),
            [0, 0, 0]
        )

    def test_majority_scores(self):
        X = self.make_rows([0, 10])

        self.assertEqual(
            majority_predict_proba(X),
            [0.0, 0.0]
        )

    def test_rain_predictions(self):
        X = self.make_rows([5, 20, 50])

        self.assertEqual(
            rain_predict(X, 20),
            [0, 0, 1]
        )

    def test_choose_rain_cutoff(self):
        X = self.make_rows([2, 5, 30, 50])
        y = [0, 0, 1, 1]

        cutoff = choose_rain_cutoff(X, y)

        self.assertEqual(
            rain_predict(X, cutoff),
            y
        )

    def test_invalid_training_lengths(self):
        X = self.make_rows([2, 5])

        with self.assertRaises(ValueError):
            choose_rain_cutoff(X, [0])

    def test_empty_training_data(self):
        with self.assertRaises(ValueError):
            choose_rain_cutoff([], [])


if __name__ == "__main__":
    unittest.main()
