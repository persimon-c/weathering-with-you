# Hyperparameters and seeds

The settings behind every validation row in [`runs.csv`](runs.csv). All methods train on 2022 to 2024 and are scored on 2025. Nothing is chosen on 2026.

## Shared preprocessing

The learned methods (Naive Bayes, logistic regression, KNN) use `src/preprocess.py`, fitted on the training rows only:

- Blank `soil_wetness` is filled with the training median, and `soil_wetness_missing` is added as a 15th feature.
- Every feature is standardized with the training mean and standard deviation.

The baselines use the raw features, so the rain cutoff is in millimeters.

## Methods

| Method | `runs.csv` method | Settings | How they were chosen | Seed |
|---|---|---|---|---|
| Majority | `majority` | Always not flooded | No settings | 0, unused: no randomness |
| Rain rule | `rainfall` | Flooded when `rain_3d` > 50.98 mm | Cutoff with the best balanced accuracy on the training rows, out of every `rain_3d` value seen in training | 0, unused: no randomness |
| Place history | `place_history` | Flooded when the tile's training flood rate is >= 0.5. A tile not seen in training gets the overall training rate | Fixed at 0.5, not tuned | 0, unused: no randomness |
| Gaussian Naive Bayes | `naive_bayes` | `var_floor` 1e-9, class priors from the training rows, flooded when its probability is the larger one | Default floor, only there to avoid dividing by zero. Not tuned | 42, unused: no randomness |
| Logistic regression | `logistic` | `lr` 0.1, `epochs` 2000 full-batch gradient descent steps, `reg` 0.01 (L2 on the weights, not the bias), `class_weight` balanced, decision threshold 0.5 | Not tuned: no search over values has been run yet | 42, used for the starting weights |
| K nearest neighbors | `knn` | `k` 5, Euclidean distance, flooded when at least half of the neighbors flooded | k = 5 has the best validation balanced accuracy out of 5, 11, 21 and 51. All four are in `runs.csv` | 42, unused: no randomness. Distance ties go to the earlier training row |

The decision thresholds of logistic regression and KNN are not in the `hyperparameters` column because they are the 0.5 default. Tuning each method and its threshold on validation is planned for Milestone 4.

## Feature-group experiment

`scripts/run_feature_groups.py` reuses the same training and validation split and the same training-only preprocessing. The feature groups are selected after preprocessing, from the 15-column processed matrix, so `soil_wetness_missing` is available. All nine rows use the settings above, seed 42, and the 2025 validation split. The `group` name is stored in each row's `hyperparameters` JSON, and the compact summary is in [`tables/feature_groups.csv`](tables/feature_groups.csv).

| Group | Columns | Count |
|---|---|---|
| `place` | `lat`, `lon`, `elevation` | 3 |
| `place_weather` | place, the six rain windows, `humidity`, `temperature`, `soil_wetness`, and `soil_wetness_missing` | 13 |
| `place_weather_season` | every processed column, including `month_sin` and `month_cos` | 15 |

`place_weather_season` has the same columns as the main model rows, so its validation scores reproduce the main rows. The 2026 test set is not used.
