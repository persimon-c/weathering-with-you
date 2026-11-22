# Hyperparameters and seeds

The settings behind every validation row in [`runs.csv`](runs.csv). All methods train on 2022 to 2024 and are scored on 2025. Nothing is chosen on 2026.

## Shared preprocessing

The learned methods (Naive Bayes, logistic regression, KNN) use `src/preprocess.py`, fitted on the training rows only:

- Blank `soil_wetness` is filled with the training median, and `soil_wetness_missing` is added as a 15th feature. Naive Bayes leaves that flag out: a 0/1 column can have almost no spread in one class, and its bell curve then outweighs every other feature
- Every feature is standardized with the training mean and standard deviation.

The baselines use the raw features, so the rain cutoff is in millimeters.

## Methods

| Method | `runs.csv` method | Settings | How they were chosen | Seed |
|---|---|---|---|---|
| Majority | `majority` | Always not flooded | No settings | 0, unused: no randomness |
| Rain rule | `rainfall` | Flooded when `rain_3d` > 50.98 mm | Cutoff with the best balanced accuracy on the training rows, out of every `rain_3d` value seen in training | 0, unused: no randomness |
| Place history | `place_history` | Flooded when the tile's training flood rate is >= 0.5. A tile not seen in training gets the overall training rate | Fixed at 0.5, not tuned | 0, unused: no randomness |
| Gaussian Naive Bayes | `naive_bayes` | `var_floor` 1e-9, the 14 features without the missing-soil flag, class priors from the training rows, flooded when its probability is the larger one | Default floor, only there to avoid dividing by zero. Not tuned | 42, unused: no randomness |
| Logistic regression | `logistic` | `lr` 0.1, `epochs` 2000 full-batch gradient descent steps, `reg` 0.01 (L2 on the weights, not the bias), `class_weight` balanced, decision threshold 0.5 | Not tuned: no search over values has been run yet | 42, used for the starting weights |
| K nearest neighbors | `knn` | `k` 5, Euclidean distance, flooded when at least half of the neighbors flooded | k = 5 has the best validation balanced accuracy out of 5, 11, 21 and 51. All four are in `runs.csv` | 42, unused: no randomness. Distance ties go to the earlier training row |

The decision thresholds of logistic regression and KNN are not in the `hyperparameters` column because they are the 0.5 default. Tuning each method and its threshold on validation is planned for Milestone 4.

## Decision thresholds (M4 validation tuning)

Milestone 4 systematically tunes the classification decision threshold $\tau$ across the 2025 validation set with the frozen 1.0% flood threshold label. Preprocessing and model parameters remain fit on training data only.

### Threshold grid

An 18-point probability cutoff grid is evaluated:

$$\tau \in \{0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90\}$$

A sample is predicted flooded when $\hat{p} \ge \tau$.

### Validation tuning results

Optimal operating points selected by validation balanced accuracy (ties broken by lowest $\tau$):

| Method | Feature Set | Default BalAcc ($\tau=0.50$) | Tuned $\tau^*$ | Tuned BalAcc | Recall | Precision | F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Place history | `place` (cell ID) | 0.781593 | **0.30** | **0.856608** | 0.831522 | 0.616935 | 0.708333 | Dominant benchmark; lowers FN from 68 to 31 |
| Logistic regression | `place_weather_season` (15) | 0.707611 | **0.45** | **0.716786** | 0.733696 | 0.359043 | 0.482143 | Favorable recall trade-off (+13 TP for +42 FP) |
| Gaussian Naive Bayes | `place_weather_season` (15) | 0.540372 | **0.05** | **0.659108** | 0.798913 | 0.275797 | 0.410042 | Low-$\tau$ optimum (GNB drops missing-soil flag, 14 effective); $\tau=0.10 \to 0.654739$ |
| KNN ($k=5$) | `place_weather_season` (15) | 0.567485 | **0.02** | **0.664654** | 0.782609 | 0.283465 | 0.416185 | Plateau across $\tau \in [0.02, 0.20]$ from discrete votes |
| *KNN ($k=5$, place-only)* | `place` (3) | *0.795238* | ***0.02*** | ***0.853670*** | *0.940217* | *0.480556* | *0.636029* | *Spatial-only KNN approaches Place History (evaluated via `--place-only` ablation; not in 72-row main table)* |

Detailed per-threshold metrics and confusion matrices are saved in [`tables/threshold_tuning.csv`](tables/threshold_tuning.csv) and discussed in [`threshold_tuning.md`](threshold_tuning.md).

### Budget parity

Every tunable method receives equal search budget: exactly 18 probability threshold configurations. Total tuning evaluations: 4 tunable methods $\times$ 18 thresholds = 72 configurations.

### Non-tunable methods

- **Majority baseline:** Predicts all-dry. Exports training prevalence (0.169888); balanced accuracy is flat at 0.5000 across all $\tau$.
- **Rain rule baseline:** Operates on a rainfall millimeter cutoff (50.98 mm, tuned on training data). Its exported scores are ranking-only atan transforms in $[0.905, 0.998]$; all grid thresholds predict all-positive ($\text{BalAcc}=0.5000$).

### Descope and distance dilution notes

- **$k=51$ descope:** $k=5$ was selected as the optimal neighborhood size in Milestone 3 validation. $k=51$ is intentionally descoped from the M4 grid.
- **KNN distance dilution:** Comparing place-only KNN ($k=5$, BalAcc 0.7952 at $\tau=0.50$; 0.8537 at $\tau^*=0.02$) with full-feature KNN ($k=5$, BalAcc 0.5675 at $\tau=0.50$; 0.6647 at $\tau^*=0.02$) demonstrates that 9 noisy weather dimensions dilute the spatial Euclidean metric, confirming the domain finding under identical tuning.

## Feature-group experiment

`scripts/run_feature_groups.py` reuses the same training and validation split and the same training-only preprocessing. The feature groups are selected after preprocessing, from the 15-column processed matrix, so `soil_wetness_missing` is available. All nine rows use the settings above, seed 42, and the 2025 validation split. The `group` name is stored in each row's `hyperparameters` JSON, and the compact summary is in [`tables/feature_groups.csv`](tables/feature_groups.csv).

| Group | Columns | Count |
|---|---|---|
| `place` | `lat`, `lon`, `elevation` | 3 |
| `place_weather` | place, the six rain windows, `humidity`, `temperature`, `soil_wetness`, and `soil_wetness_missing` | 13 |
| `place_weather_season` | every processed column, including `month_sin` and `month_cos` | 15 |

`place_weather_season` has the same columns as the main model rows, so its validation scores reproduce the main rows. The 2026 test set is not used.
