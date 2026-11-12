
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



def fit_place_history(X_train, y_train):
    """Calculate flood rates for each cell using training data."""
    if len(X_train) != len(y_train) or not X_train:
        raise ValueError("Invalid training data.")

    lat_index = FEATURES.index("lat")
    lon_index = FEATURES.index("lon")

    counts = {}

    for row, label in zip(X_train, y_train):
        cell = (row[lat_index], row[lon_index])

        if cell not in counts:
            counts[cell] = [0, 0]

        counts[cell][0] += label
        counts[cell][1] += 1

    rates = {
        cell: flooded / total
        for cell, (flooded, total) in counts.items()
    }

    overall_rate = sum(y_train) / len(y_train)

    return rates, overall_rate


def place_history_scores(X, rates, overall_rate):
    """Return historical flood rates, using training prevalence for unseen cells."""
    lat_index = FEATURES.index("lat")
    lon_index = FEATURES.index("lon")

    return [
        rates.get(
            (row[lat_index], row[lon_index]),
            overall_rate
        )
        for row in X
    ]


def place_history_predict(X, rates, overall_rate, cutoff=0.5):
    """Convert historical flood scores into binary predictions."""
    scores = place_history_scores(X, rates, overall_rate)

    return [
        int(score >= cutoff)
        for score in scores
    ]
