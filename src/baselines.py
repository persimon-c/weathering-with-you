
from src.data import FEATURES


def majority_predict(X):
    """Always predict not flooded (class 0)."""
    return [0] * len(X)


def majority_predict_proba(X):
    """Return the majority baseline's flood scores."""
    return [0.0] * len(X)


def rain_predict(X, cutoff):
    """Predict flooding when 3-day rainfall exceeds cutoff."""
    rain_index = FEATURES.index("rain_3d")

    return [
        int(row[rain_index] > cutoff)
        for row in X
    ]


def rain_predict_proba(X):
    """Rank samples by raw 3-day rainfall for ROC AUC."""
    rain_index = FEATURES.index("rain_3d")
    rain = [row[rain_index] for row in X]

    if any(value is None for value in rain):
        raise ValueError("Missing rain_3d value.")

    # Scores are normalized later using training statistics.
    return rain


def choose_rain_cutoff(X_train, y_train):
    """Select cutoff maximizing training balanced accuracy."""
    if len(X_train) != len(y_train) or not X_train:
        raise ValueError("Invalid training data.")

    rain_index = FEATURES.index("rain_3d")
    rain = [row[rain_index] for row in X_train]

    if any(value is None for value in rain):
        raise ValueError("Missing rain_3d value.")

    from src.metrics import evaluate

    # Include a cutoff below the smallest value to allow all-positive.
    candidates = [min(rain) - 1] + sorted(set(rain))

    best_cutoff = candidates[0]
    best_score = float("-inf")

    for cutoff in candidates:
        predictions = [
            int(value > cutoff)
            for value in rain
        ]

        score = evaluate(y_train, predictions)["balanced_accuracy"]

        if score is not None and score > best_score:
            best_score = score
            best_cutoff = cutoff

    return best_cutoff
