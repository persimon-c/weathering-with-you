
import unittest

from src.data import FEATURES
from src.baselines import (
    majority_predict,
    majority_predict_proba,
    rain_predict,
    choose_rain_cutoff,
    fit_place_history,
    place_history_scores,
    place_history_predict,
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

    def test_place_history_rates(self):
        X = self.make_rows([0, 0, 0])

        lat = FEATURES.index("lat")
        lon = FEATURES.index("lon")

        X[0][lat], X[0][lon] = 14.5, 121.0
        X[1][lat], X[1][lon] = 14.5, 121.0
        X[2][lat], X[2][lon] = 15.0, 122.0

        rates, overall = fit_place_history(X, [1, 0, 1])

        self.assertEqual(rates[(14.5, 121.0)], 0.5)
        self.assertEqual(rates[(15.0, 122.0)], 1.0)
        self.assertAlmostEqual(overall, 2 / 3)

    def test_unseen_cell_fallback(self):
        X_train = self.make_rows([0, 0])
        X_val = self.make_rows([0])

        lat = FEATURES.index("lat")
        lon = FEATURES.index("lon")

        X_val[0][lat] = 99.0
        X_val[0][lon] = 99.0

        rates, overall = fit_place_history(X_train, [1, 0])
        scores = place_history_scores(X_val, rates, overall)

        self.assertEqual(scores, [0.5])

    def test_place_history_predictions(self):
        X = self.make_rows([0, 0])
        lat = FEATURES.index("lat")
        lon = FEATURES.index("lon")

        X[0][lat], X[0][lon] = 14.5, 121.0
        X[1][lat], X[1][lon] = 15.0, 122.0

        rates = {
            (14.5, 121.0): 0.8,
            (15.0, 122.0): 0.2
        }

        self.assertEqual(
            place_history_predict(X, rates, 0.5),
            [1, 0]
        )


if __name__ == "__main__":
    unittest.main()
