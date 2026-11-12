
import numpy as np

from src.data import FEATURES


def fit_preprocess(X_train):
    """Learn preprocessing statistics from training data only."""
    X = np.array([
        [np.nan if value is None else value for value in row]
        for row in X_train
    ], dtype=float)

    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != len(FEATURES):
        raise ValueError("Invalid training feature matrix.")

    soil_index = FEATURES.index("soil_wetness")

    # Learn the replacement value using training data only.
    valid_soil = X[~np.isnan(X[:, soil_index]), soil_index]

    if len(valid_soil) == 0:
        raise ValueError("Training soil_wetness is entirely missing.")

    soil_median = float(np.median(valid_soil))

    # Fill missing soil wetness before computing statistics.
    X[:, soil_index] = np.where(
        np.isnan(X[:, soil_index]),
        soil_median,
        X[:, soil_index]
    )

    if not np.all(np.isfinite(X)):
        raise ValueError("Features must be finite.")

    means = X.mean(axis=0)
    stds = X.std(axis=0)

    # Avoid dividing by zero for constant features.
    stds[stds == 0] = 1.0

    return {
        "soil_median": soil_median,
        "means": means,
        "stds": stds
    }


def transform_preprocess(X, params):
    """Apply training statistics to any dataset split."""
    X = np.array([
        [np.nan if value is None else value for value in row]
        for row in X
    ], dtype=float)

    if X.ndim != 2 or X.shape[1] != len(FEATURES):
        raise ValueError("Invalid feature matrix.")

    soil_index = FEATURES.index("soil_wetness")

    # Preserve information about missing soil values.
    missing = np.isnan(X[:, soil_index]).astype(float)

    X[:, soil_index] = np.where(
        np.isnan(X[:, soil_index]),
        params["soil_median"],
        X[:, soil_index]
    )

    if not np.all(np.isfinite(X)):
        raise ValueError("Features must be finite.")

    X = (X - params["means"]) / params["stds"]

    # Append one additional feature: 1 = missing, 0 = present.
    return np.column_stack((X, missing))


def preprocess_splits(splits):
    """Fit on train, then transform train, val and test."""
    params = fit_preprocess(splits["train"]["X"])

    processed = {}

    for name in ("train", "val", "test"):
        processed[name] = {
            "X": transform_preprocess(splits[name]["X"], params),
            "y": splits[name]["y"],
            "metadata": splits[name]["metadata"]
        }

    return processed, params
