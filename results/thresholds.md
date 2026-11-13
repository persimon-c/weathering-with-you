# Threshold sensitivity

## Reproduce

Run from the repository root:

```bash
python scripts/run_thresholds.py
```

The script saves `results/tables/thresholds.csv` with 18 rows:
six methods at each of three flood thresholds.

The 1% threshold remains the main result. The 0.5% and 2%
thresholds are sensitivity checks.

## Labels and dataset

Labels are calculated in memory using:

```python
flooded = int(flood_share > threshold)
```

Both `flood_share` and the threshold are expressed in percent.
For example, `0.5` means half a percent. A value equal to the
threshold is labeled not flooded.

The processed CSV is never modified.

## Splits and preprocessing

All thresholds use the same features, rows, and frozen splits:

- Training: 2022–2024.
- Validation: 2025.
- Test: 2026.

Preprocessing is fitted once on training features and reused
across all thresholds for Naive Bayes, logistic regression,
and KNN. It uses the existing median imputation,
standardization, and missing-soil indicator.

Baselines use raw features, following the existing baseline
runner.

All six methods are fitted using each threshold's training
labels and evaluated on validation data only. Test label
proportions are reported, but test predictions and metrics
remain reserved for Milestone 4.

## Methods and settings

- Majority: predict the training majority class.
- Rainfall: select the rain cutoff using training balanced
  accuracy at each label threshold.
- Place history: calculate training flood rates per cell,
  using the overall training rate for unseen cells.
  Classification cutoff: 0.5.
- Gaussian Naive Bayes: variance floor of 1e-9.
- Logistic regression: learning rate 0.1, 2000 epochs,
  L2 regularization 0.01, balanced class weights,
  seed 42, and classification cutoff 0.5.
- KNN: k=5 and classification cutoff 0.5.

Model settings remain fixed across label thresholds.
Training class weights, rainfall cutoffs, and place-history
rates are recalculated from each threshold's training labels.

## Output columns

- `threshold_percent`: flood label threshold in percent.
- `result_type`: `main` for 1%; otherwise `sensitivity`.
- `method`: method evaluated.
- `split`: evaluation split, always `val`.
- `seed`: seed recorded by the runner.
- `hyperparameters`: model settings and fitted cutoffs.
- `train_rows`, `val_rows`, `test_rows`: split sizes.
- `train_flooded`, `val_flooded`, `test_flooded`:
  number of rows labeled flooded.
- `train_flooded_share`, `val_flooded_share`,
  `test_flooded_share`: fraction of rows labeled flooded.
- `accuracy`, `balanced_accuracy`, `macro_f1`,
  `precision`, `recall`, `f1`, `roc_auc`:
  validation metrics.
- `tp`, `tn`, `fp`, `fn`: validation confusion counts.

Flooded shares are fractions from 0 to 1. Multiply by 100
to display percentages. They describe the proportion of
rows labeled flooded, rather than flooded area within a cell.

Precision, recall, and F1 refer to the flooded class.

## Interpretation

This experiment checks how method performance changes when
the flood label threshold changes.

Scores at different thresholds describe different target
labels. They should not be used to choose whichever threshold
produces the highest score. The main threshold remains 1%.

## Validation

All 101 tests passed. The output contains six methods at each
of 0.5%, 1%, and 2%, for 18 rows total.