#Check shared preprocessing policy and method settings.

import unittest
from unittest.mock import patch

import numpy as np

from src.data import FEATURES
from src.experiments import (
    BASELINES,
    METHODS,
    build_model,
    predict_methods,
)
from src.preprocess import fit_preprocess, transform_preprocess


class TestExperiments(unittest.TestCase):

    def sample_data(self):
        train = {
            "X": [
                [float(i + 1)] * len(FEATURES)
                for i in range(12)
            ],
            "y": [i % 2 for i in range(12)],
        }

        evaluation = {
            "X": [
                [2.0] * len(FEATURES),
                [4.0] * len(FEATURES),
            ],
        }

        return train, evaluation

    def test_baselines_do_not_require_preprocessing(self):
        train, evaluation = self.sample_data()
        soil_index = FEATURES.index("soil_wetness")

        for row in train["X"] + evaluation["X"]:
            row[soil_index] = None

        with patch(
            "src.experiments.fit_preprocess",
            side_effect=AssertionError(
                "Baselines should use raw features."
            ),
        ):
            results = predict_methods(
                train,
                evaluation,
                methods=BASELINES,
            )

        self.assertEqual(
            [result[0] for result in results],
            list(BASELINES),
        )

        for _, seed, _, predictions, scores in results:
            self.assertEqual(seed, 0)
            self.assertEqual(len(predictions), 2)
            self.assertEqual(len(scores), 2)

    def test_prepared_inputs_reused_and_only_nb_loses_flag(self):
        train, evaluation = self.sample_data()

        params = fit_preprocess(train["X"])
        prepared = (
            transform_preprocess(train["X"], params),
            transform_preprocess(evaluation["X"], params),
        )

        original = [matrix.copy() for matrix in prepared]
        received = {}

        class RecordingModel:
            def __init__(self, name):
                self.name = name

            def fit(self, X, y):
                received[self.name] = [X.copy()]

            def predict(self, X):
                received[self.name].append(X.copy())
                return np.zeros(len(X), dtype=int)

            def predict_proba(self, X):
                received[self.name].append(X.copy())
                return np.tile([0.75, 0.25], (len(X), 1))

        def fake_build_model(name, **kwargs):
            return RecordingModel(name), {}

        with patch(
            "src.experiments.fit_preprocess",
            side_effect=AssertionError(
                "Prepared inputs must not refit preprocessing."
            ),
        ), patch(
            "src.experiments.build_model",
            side_effect=fake_build_model,
        ):
            results = predict_methods(
                train,
                evaluation,
                prepared=prepared,
            )

        self.assertEqual(
            [result[0] for result in results],
            list(METHODS),
        )

        for name in ("naive_bayes", "logistic", "knn"):
            expected_train, expected_eval = prepared

            if name == "naive_bayes":
                expected_train = expected_train[:, :-1]
                expected_eval = expected_eval[:, :-1]

            np.testing.assert_array_equal(
                received[name][0],
                expected_train,
            )
            np.testing.assert_array_equal(
                received[name][1],
                expected_eval,
            )
            np.testing.assert_array_equal(
                received[name][2],
                expected_eval,
            )

        for before, after in zip(original, prepared):
            np.testing.assert_array_equal(before, after)

    def test_knn_override_does_not_change_default(self):
        first, first_settings = build_model("knn", k=11)
        second, second_settings = build_model("knn")

        self.assertIsNot(first, second)
        self.assertEqual(first_settings, {"k": 11})
        self.assertEqual(second_settings, {"k": 5})

    def test_unknown_method_is_rejected(self):
        train, evaluation = self.sample_data()

        with self.assertRaises(ValueError):
            build_model("unknown")

        with self.assertRaises(ValueError):
            predict_methods(
                train,
                evaluation,
                methods=("unknown",),
            )


if __name__ == "__main__":
    unittest.main()