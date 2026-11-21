# Data Leakage Audit

**Repository:** `persimon-c/weathering-with-you`  
**Protocol Reference:** `Evaluation-Protocol.md:125-139` (11-Point Checklist)  
**Audit Date:** 2026-11-22  
**Auditor:** Simone Pauline M. Dagondon (`@persimon-c`)  
**Status:** Certified Zero Leakage across Training, Validation, and Test Splits  

---

## Executive Summary

This document records the empirical verification of the 11-point data leakage checklist from the project evaluation protocol. Each checkpoint has been tested and verified against the repository implementation, source code contracts, and processed dataset (`data/processed/flood_data.csv`).

---

## 11-Point Data Leakage Checklist

### 1. Target Not Present in Feature Set
- **Status:** PASS
- **Empirical Evidence:** In `src/data.py:24`, `TARGET = "flooded"` is isolated. The `FEATURES` list (`src/data.py:6-22`) contains exactly 14 predictor columns (`lat`, `lon`, `elevation`, `rain_1d`..`rain_30d`, `humidity`, `temperature`, `soil_wetness`, `month_sin`, `month_cos`). The target is explicitly popped from the feature matrix in `load_data()`.

### 2. Target-Derived Columns Excluded
- **Status:** PASS
- **Empirical Evidence:** The processed dataset contains flood outcome measures (`flood_km2`, `flood_share`, `n_polygons`). None of these columns appear in `src/data.py:FEATURES`. Only pre-pass spatial indicators and meteorological predictors are exposed to the models.

### 3. Identifiers and Free-Text Fields Excluded
- **Status:** PASS
- **Empirical Evidence:** Free-text descriptions and arbitrary IDs are excluded. The spatial unit identifier `cell_id` is used exclusively for tabular lookup in the Place History baseline (`src/baselines.py:108-142`) and is never fed as a numeric feature to Gaussian Naive Bayes, Logistic Regression, or KNN. Pass metadata (`event_id`, `cluster_id`, `split`) is quarantined in `src/data.py:METADATA`.

### 4. Zero Duplicate Rows Checked Across Splits
- **Status:** PASS
- **Empirical Evidence:** Split assignment is deterministic based on storm clusters and calendar years. Automated verification in `tests/test_data.py::test_processed_dataset_splits_have_zero_overlap` confirms zero row overlap across `train` (1,966 rows), `val` (987 rows), and `test` (578 rows), summing to all 3,531 rows.

### 5. Temporal Order Respected for Year-Stamped Data
- **Status:** PASS
- **Empirical Evidence:** The temporal split preserves causal chronology across storm clusters (`design.md:118-124`):
  - **Training:** 2022 to 2024 (Clusters 1–13, passes 1–33, $N=1,966$)
  - **Validation:** 2025 (Clusters 14–18, passes 34–49, $N=987$)
  - **Test (Quarantined):** 2026 (Clusters 19–21, passes 50–61, $N=578$)
  No observation from a later year is ever used to predict or fit models for an earlier period.

### 6. Entity Grouping Respected
- **Status:** PASS
- **Empirical Evidence:** Satellite passes occurring within 7 days are grouped into coherent storm clusters (`scripts/build_dataset.py`). Cluster integrity is strictly preserved: no storm cluster or satellite pass is split across train/validation/test boundaries.

### 7. Imputation and Scaling Fit on Train Only
- **Status:** PASS
- **Empirical Evidence:** Preprocessing parameters are learned solely on the training split via `src/preprocess.py::fit_preprocess(X_train)`. Mean, standard deviation, and median soil wetness (Decision 3, `design.md:218`) are computed from training rows ($N=1,966$) and statelessly applied to validation and test data. Validation and test statistics are never accessed during fitting.

### 8. Feature Selection Fit on Train Only
- **Status:** PASS
- **Empirical Evidence:** Feature sets (Place, Weather, Full) are determined by domain specification. No data-driven feature selection, variance pruning, or backward elimination is performed on validation or test sets.

### 9. Hyperparameters Chosen on Validation Only
- **Status:** PASS
- **Empirical Evidence:** In Milestone 3, models operated on fixed default settings (KNN $k=5$, LR $\lambda=0.01, \eta=0.10$, GNB default, Place History default $\tau=0.50$). Systematic validation threshold tuning ($\\tau \in [0.02, 0.90]$) is strictly confined to the validation split as scheduled in Milestone 4 (Issue #44). Zero test data is used for model selection.

### 10. Single Test Set Evaluation Gate
- **Status:** PASS
- **Empirical Evidence:** The 2026 test set ($N=578$) remains completely quarantined. No test scores have been computed or stored. The evaluation script (`scripts/evaluate_test.py`) will be executed exactly once following the Tuesday Nov 24 18:00 PHT code freeze (Issue #46).

### 11. Reported Result is Not Best of Many Test Evaluations
- **Status:** PASS
- **Empirical Evidence:** Test evaluation protocol mandates committing the single output of `scripts/evaluate_test.py` directly to version control. All model configurations are frozen on validation prior to test scoring.
