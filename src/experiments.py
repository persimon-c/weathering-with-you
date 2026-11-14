"""Shared model settings and predictions for experiment runners."""

from src.baselines import (
    majority_predict,
    majority_predict_proba,
    choose_rain_cutoff,
    rain_predict,
    rain_predict_proba,
    fit_place_history,
    place_history_predict,
    place_history_scores,
)
from src.models.naive_bayes import GaussianNaiveBayes
from src.models.logistic import LogisticRegression
from src.models.knn import KNearestNeighbors
from src.preprocess import (
    fit_preprocess,
    transform_preprocess,
    without_missing_flag,
)

BASELINES = ("majority", "rainfall", "place_history")
MODEL_METHODS = ("naive_bayes", "logistic", "knn")
METHODS = BASELINES + MODEL_METHODS

KNN_KS = (5, 11, 21, 51)
DEFAULT_K = 5


def build_model(name, seed=42, k=DEFAULT_K):
    """Create a fresh model and return its settings."""
    if name == "naive_bayes":
        settings = {
            "var_floor": 1e-9,
            "missing_flag": False,
        }
        model = GaussianNaiveBayes(
            var_floor=settings["var_floor"]
        )

    elif name == "logistic":
        settings = {
            "lr": 0.1,
            "epochs": 2000,
            "reg": 0.01,
            "class_weight": "balanced",
        }
        model = LogisticRegression(**settings, seed=seed)

    elif name == "knn":
        settings = {"k": k}
        model = KNearestNeighbors(**settings)

    else:
        raise ValueError(f"Unknown model: {name}")

    return model, settings


def model_features(name, X):
    """Select inputs from the full preprocessed matrix."""
    if name not in MODEL_METHODS:
        raise ValueError(f"Unknown model: {name}")

    if name == "naive_bayes":
        return without_missing_flag(X)

    return X


def predict_methods(
    train,
    evaluation,
    seed=42,
    methods=METHODS,
    prepared=None,
    k=DEFAULT_K,
):
    """Fit on training rows and predict evaluation rows.

    Returns tuples containing:
    method, seed, settings, predictions, scores.

    prepared optionally supplies (X_train, X_eval), allowing
    threshold experiments to reuse the same preprocessing.
    """
    unknown = set(methods) - set(METHODS)

    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    raw_train = train["X"]
    raw_eval = evaluation["X"]
    labels = train["y"]

    if (
        not raw_train
        or not raw_eval
        or len(raw_train) != len(labels)
    ):
        raise ValueError(
            "Nonempty training/evaluation rows and "
            "matching training labels required."
        )

    if any(name in MODEL_METHODS for name in methods):
        if prepared is None:
            preprocessing = fit_preprocess(raw_train)

            prepared = (
                transform_preprocess(raw_train, preprocessing),
                transform_preprocess(raw_eval, preprocessing),
            )

        X_train, X_eval = prepared

    results = []

    for name in methods:
        if name == "majority":
            prevalence = sum(labels) / len(labels)
            majority = int(prevalence > 0.5)

            settings = {"class": majority}
            predicted = majority_predict(raw_eval, majority)
            scores = majority_predict_proba(raw_eval, prevalence)

        elif name == "rainfall":
            cutoff = choose_rain_cutoff(raw_train, labels)

            settings = {"cutoff": cutoff}
            predicted = rain_predict(raw_eval, cutoff)
            scores = rain_predict_proba(raw_eval)

        elif name == "place_history":
            rates, overall = fit_place_history(
                raw_train, labels
            )

            settings = {"cutoff": 0.5}
            predicted = place_history_predict(
                raw_eval, rates, overall
            )
            scores = place_history_scores(
                raw_eval, rates, overall
            )

        else:
            model, settings = build_model(
                name, seed=seed, k=k
            )

            model_train = model_features(name, X_train)
            model_eval = model_features(name, X_eval)

            model.fit(model_train, labels)

            predicted = model.predict(model_eval).tolist()
            scores = model.predict_proba(
                model_eval
            )[:, 1].tolist()

        method_seed = seed if name in MODEL_METHODS else 0

        results.append((
            name,
            method_seed,
            settings,
            predicted,
            scores,
        ))

    return results