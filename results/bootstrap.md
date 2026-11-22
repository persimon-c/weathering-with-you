# Cluster Bootstrap Intervals (M4 Validation)

## Reproduce

```bash
python scripts/run_bootstrap.py
```

Writes [`tables/comparison_validation_final.csv`](tables/comparison_validation_final.csv). It reads the validation scores (`predictions/validation_scores.csv`) and each method's tuned $\tau^*$ (`tables/threshold_tuning.csv`, main feature set). No model is refit, and the 2026 test set is not used.

## Method

- `src/metrics.py::cluster_bootstrap` resamples whole storm clusters with replacement, $B = 1{,}000$ draws, seed 0. Each draw picks as many clusters as the split has (5 on validation), so rows of one storm always stay together.
- Paired difference: on every draw both the learned method and Place History are scored on the same rows, $\Delta = \text{Metric}_\text{learned} - \text{Metric}_\text{place history}$.
- The 95% interval is the 2.5th and 97.5th percentile of the 1,000 draws, $[\Delta_{0.025}, \Delta_{0.975}]$. `excludes_zero` is `true` when the whole interval lies on one side of zero.
- Bootstrapped metrics: balanced accuracy and macro F1, the headline metrics in `design.md`. ROC AUC is reported as a point value only.

## Results

| Method | $\tau^*$ | BalAcc | $\Delta$ BalAcc [95% CI] | Macro F1 | $\Delta$ Macro F1 [95% CI] |
|---|---:|---:|---|---:|---|
| Place history | 0.30 | 0.8566 | reference | 0.8133 | reference |
| Logistic regression | 0.45 | 0.7168 | −0.140 [−0.210, −0.088] | 0.6385 | −0.175 [−0.277, −0.128] |
| Gaussian Naive Bayes | 0.05 | 0.6591 | −0.198 [−0.246, −0.129] | 0.5368 | −0.277 [−0.330, −0.191] |
| KNN ($k=5$) | 0.02 | 0.6647 | −0.192 [−0.242, −0.157] | 0.5505 | −0.263 [−0.325, −0.226] |

Every interval excludes zero: no learned method matches Place History on validation, on either headline metric.

## Caveat: few clusters

The validation year has only 5 storm clusters, of 6, 96, 214, 288 and 383 rows. Their Kish effective number, $(\sum n)^2 / \sum n^2$, is **3.42**, and a draw of 5 clusters from 5 can take only $\binom{9}{5} = 126$ distinct forms. The intervals are therefore coarse and describe variation between these few storms, not a large sample. The per-cluster table ([`tables/per_cluster.csv`](tables/per_cluster.csv)) stays the primary evidence; these intervals are a secondary summary.

## Seed stability of logistic regression

Logistic regression is fit by full-batch gradient descent with L2 regularization, so the seed only sets the near-zero starting weights and training converges to the same solution. Across seeds 0 to 9 at $\tau^* = 0.45$, validation balanced accuracy is 0.716786 every time ($\sigma_\text{BalAcc} = 0.000000$), and ROC AUC varies only in floating point, between 0.752362 and 0.752389 (about 0.00002). The other methods have no random step. A single run per method is therefore enough, and the bootstrap captures the only variation that matters here, between storm clusters.
