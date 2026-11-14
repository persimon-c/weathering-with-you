# Feature groups for the M3 experiment.
#
# The groups are selected after preprocessing because the processed matrix
# has one extra column, soil_wetness_missing, that is not in src.data.FEATURES.

from src.data import FEATURES


# The processed matrix keeps the original feature order and appends the
# missing-value indicator as the final column.
PREPROCESSED_FEATURES = FEATURES + ["soil_wetness_missing"]

_PLACE = ("lat", "lon", "elevation")

_WEATHER = (
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "rain_14d",
    "rain_21d",
    "rain_30d",
    "humidity",
    "temperature",
    "soil_wetness",
)

# The ordered group names also define the order of rows in the results table.
FEATURE_GROUPS = {
    "place": _PLACE,
    "place_weather": _PLACE + _WEATHER + ("soil_wetness_missing",),
    "place_weather_season": tuple(PREPROCESSED_FEATURES),
}


def select_features(X, group):
    """Return the processed columns that belong to one feature group."""
    if group not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group: {group}")

    if X.ndim != 2 or X.shape[1] != len(PREPROCESSED_FEATURES):
        raise ValueError(
            f"Expected {len(PREPROCESSED_FEATURES)} processed columns, "
            f"got shape {X.shape}"
        )

    indices = [
        PREPROCESSED_FEATURES.index(name)
        for name in FEATURE_GROUPS[group]
    ]

    return X[:, indices]
