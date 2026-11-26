# 2026 Test Evaluation (M4)

## How it was run

The 2026 test set was scored **once**, on Tue 24 Nov 2026 at 18:21 PHT, after the 18:00 code freeze, with `python scripts/evaluate_test.py` on a clean `main` at frozen commit `8c74e98`. The script's gate refuses to run before the freeze, off `main`, with a dirty tree, or once any test output exists, so the run cannot be repeated. The console output is kept in [`logs/test_eval.log`](logs/test_eval.log).

Every method was fit on the 2022–2024 training rows with the settings chosen on validation. The four tunable methods used their validation $\tau^*$ ([`threshold_tuning.md`](threshold_tuning.md)); majority and the rain rule kept their own rules. Nothing was changed after the test scores were seen.

Outputs: [`tables/comparison_test.csv`](tables/comparison_test.csv), [`tables/per_cluster_test.csv`](tables/per_cluster_test.csv), [`predictions/test_scores.csv`](predictions/test_scores.csv), [`figures/roc_curves_test.png`](figures/roc_curves_test.png) and [`figures/test_map.png`](figures/test_map.png).

## Results

578 rows from 12 passes in 3 storm clusters; 116 rows (20.1%) are flooded.

| Method | $\tau$ | Balanced accuracy | Macro F1 | Precision | Recall | ROC AUC | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority | – | 0.5000 | 0.4442 | 0.000 | 0.000 | 0.5000 | 0 | 462 | 0 | 116 |
| Rain rule | 50.98 mm | 0.5723 | 0.5000 | 0.247 | 0.621 | 0.6023 | 72 | 242 | 220 | 44 |
| **Place history** | 0.30 | **0.8963** | **0.8489** | 0.669 | 0.905 | **0.9375** | 105 | 410 | 52 | 11 |
| Gaussian Naive Bayes | 0.05 | 0.6172 | 0.4622 | 0.256 | 0.862 | 0.6633 | 100 | 172 | 290 | 16 |
| Logistic regression | 0.45 | 0.6648 | 0.5314 | 0.290 | 0.853 | 0.7224 | 99 | 220 | 242 | 17 |
| KNN ($k=5$) | 0.02 | 0.6282 | 0.5075 | 0.271 | 0.793 | 0.6815 | 92 | 214 | 248 | 24 |

Place history is the best method on every metric. It finds 105 of the 116 flooded cells with 52 false alarms. The three learned models find fewer floods (92–100) while raising about 5 times as many false alarms (242–290).

## Cluster 21 dominates the test year

| Cluster | Rows | Share | Flooded | Place history | Logistic | Naive Bayes | KNN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 19 | 53 | 9.2% | 4 | 0.9490 | 0.7041 | 0.5612 | 0.8061 |
| 20 | 66 | 11.4% | 20 | 0.9174 | 0.6848 | 0.6935 | 0.6576 |
| 21 | 459 | 79.4% | 92 | 0.8911 | 0.6569 | 0.6242 | 0.6120 |

Balanced accuracy per cluster. Cluster 21 holds 459 of 578 rows (79.4%), so the pooled test numbers are mostly a Cluster 21 result. Cluster 19 has only 4 flooded rows, so its scores swing on one or two cells; place history and KNN both find all 4 there and differ only in false alarms. The robust finding is the ranking: **place history is first in all three clusters**, by at least 0.14 balanced accuracy.

## From validation (2025) to test (2026)

| Method | Validation BalAcc | Test BalAcc | Change |
|---|---:|---:|---:|
| Place history | 0.8566 | 0.8963 | +0.040 |
| Logistic regression | 0.7168 | 0.6648 | −0.052 |
| KNN ($k=5$) | 0.6647 | 0.6282 | −0.037 |
| Gaussian Naive Bayes | 0.6591 | 0.6172 | −0.042 |
| Rain rule | 0.5089 | 0.5723 | +0.063 |

Both are at the frozen settings: tuned $\tau^*$ for the four tunable methods, the 50.98 mm cutoff for the rain rule. The learned models lose 0.04 to 0.05 balanced accuracy on the new year, while place history gains 0.04, so the gap to the best learned model grows from 0.140 on validation to **0.232** on test. The models that lean on weather transfer worse to a new storm season than the cell's own flood history does. The validation bootstrap ([`bootstrap.md`](bootstrap.md)) already put every learned-minus-place-history interval below zero; the test year agrees.

The rain rule's rise (0.509 → 0.572) is still close to chance (ROC AUC 0.60), and the majority baseline stays at 0.500.

## Spatial coverage

**0 of 578** test rows fall in a cell unseen in training, so place history has a training flood rate for every test cell. The test year measures transfer to **new storms in known places**, not to new places. How the methods do on unseen areas is not measured; the optional 2°×2° block-holdout experiment was descoped (see the limitations in `design.md`).
