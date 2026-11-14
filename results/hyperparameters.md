# Hyperparameters and seeds

The settings behind every validation row in [`runs.csv`](runs.csv). All methods train on 2022 to 2024 and are scored on 2025. Nothing is chosen on 2026.

## Shared preprocessing

The learned methods (Naive Bayes, logistic regression, KNN) use `src/preprocess.py`, fitted on the training rows only:

- Blank `soil_wetness` is filled with the training median, and `soil_wetness_missing` is added as a 15th feature.
- Every feature is standardized with the training mean and standard deviation.

Naive Bayes uses the first 14 standardized features, excluding `soil_wetness_missing` because a near-zero variance binary flag distorts the Gaussian variance floor. Logistic regression and KNN use all 15 features.

The baselines use the raw features, so the rain cutoff is in millimeters.

## Methods

| Method | `runs.csv` method | Settings | How they were chosen | Seed |
|---|---|---|---|---|
| Majority | `majority` | Always not flooded | No settings | 0, unused: no randomness |
| Rain rule | `rainfall` | Flooded when `rain_3d` > 50.98 mm | Cutoff with the best balanced accuracy on the training rows, out of every `rain_3d` value seen in training | 0, unused: no randomness |
| Place history | `place_history` | Flooded when the tile's training flood rate is >= 0.5. A tile not seen in training gets the overall training rate | Fixed at 0.5, not tuned | 0, unused: no randomness |
| Gaussian Naive Bayes | `naive_bayes` | `var_floor` 1e-9, class priors from the training rows, flooded when its probability is the larger one. Uses 14 continuous features (excluding `soil_wetness_missing`) | Default floor, only there to avoid dividing by zero. Not tuned | 42, unused: no randomness |
| Logistic regression | `logistic` | `lr` 0.1, `epochs` 2000 full-batch gradient descent steps, `reg` 0.01 (L2 on the weights, not the bias), `class_weight` balanced, decision threshold 0.5 | Not tuned: no search over values has been run yet | 42, used for the starting weights |
| K nearest neighbors | `knn` | `k` 5, Euclidean distance, flooded when at least half of the neighbors flooded | k = 5 has the best validation balanced accuracy out of 5, 11, 21 and 51. All four are in `runs.csv` | 42, unused: no randomness. Distance ties go to the earlier training row |

The decision thresholds of logistic regression and KNN are not in the `hyperparameters` column because they are the 0.5 default. Tuning each method and its threshold on validation is planned for Milestone 4.
