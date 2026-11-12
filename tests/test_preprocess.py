
import unittest

import numpy as np

from src.data import FEATURES
from src.preprocess import (
    fit_preprocess,
    transform_preprocess,
    preprocess_splits
)


class TestPreprocess(unittest.TestCase):

    def setUp(self):
        self.soil_index = FEATURES.index("soil_wetness")

        self.X_train = [
            [float(i + 1) for i in range(len(FEATURES))],
            [float(i + 3) for i in range(len(FEATURES))],
            [float(i + 5) for i in range(len(FEATURES))]
        ]

        self.X_train[1][self.soil_index] = None

    def test_training_statistics(self):
        params = fit_preprocess(self.X_train)

        self.assertAlmostEqual(
            params["soil_median"],
            (self.X_train[0][self.soil_index] +
             self.X_train[2][self.soil_index]) / 2
        )

    def test_output_dimensions(self):
        params = fit_preprocess(self.X_train)
        result = transform_preprocess(self.X_train, params)

        self.assertEqual(result.shape, (3, 15))

    def test_missing_indicator(self):
        params = fit_preprocess(self.X_train)
        result = transform_preprocess(self.X_train, params)

        np.testing.assert_array_equal(
            result[:, -1],
            [0, 1, 0]
        )

    def test_no_missing_values_after_transform(self):
        params = fit_preprocess(self.X_train)
        result = transform_preprocess(self.X_train, params)

        self.assertTrue(np.all(np.isfinite(result)))

    def test_validation_does_not_change_statistics(self):
        params = fit_preprocess(self.X_train)
        original_means = params["means"].copy()

        X_val = [[1000.0] * len(FEATURES)]
        X_val[0][self.soil_index] = None

        transform_preprocess(X_val, params)

        np.testing.assert_array_equal(
            params["means"],
            original_means
        )

    def test_empty_training_data(self):
        with self.assertRaises(ValueError):
            fit_preprocess([])

    def test_all_soil_missing(self):
        X = [
            [1.0] * len(FEATURES),
            [2.0] * len(FEATURES)
        ]

        for row in X:
            row[self.soil_index] = None

        with self.assertRaises(ValueError):
            fit_preprocess(X)

    def test_preprocess_splits(self):
        splits = {
            "train": {
                "X": self.X_train,
                "y": [0, 1, 0],
                "metadata": [{}, {}, {}]
            },
            "val": {
                "X": [self.X_train[0]],
                "y": [1],
                "metadata": [{}]
            },
            "test": {
                "X": [self.X_train[2]],
                "y": [0],
                "metadata": [{}]
            }
        }

        processed, params = preprocess_splits(splits)

        self.assertEqual(processed["train"]["X"].shape, (3, 15))
        self.assertEqual(processed["val"]["X"].shape, (1, 15))
        self.assertEqual(processed["test"]["X"].shape, (1, 15))
        self.assertIn("soil_median", params)


if __name__ == "__main__":
    unittest.main()
