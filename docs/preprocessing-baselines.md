### Preprocessing and Baselines

**Preprocessing**
- Missing `soil_wetness` values are replaced using the training-set median.
- A binary indicator identifies rows with missing soil wetness.
- Numerical features are standardized using the training-set mean and standard deviation.
- Training-derived preprocessing parameters are reused for validation and test data to prevent data leakage.

**Baseline Models**
- **Majority Class:** Predicts the majority class determined from the training dataset.
- **3-Day Rainfall Rule:** Predicts flooding when `rain_3d` exceeds a cutoff selected by maximizing training balanced accuracy.

**Validation Results**

| Metric | Majority | Rainfall |
|---|---:|---:|
| Accuracy | 0.8136 | 0.5826 |
| Balanced Accuracy | 0.5000 | 0.5089 |
| F1 | 0.0000 | 0.2590 |
| ROC AUC | 0.5000 | 0.5122 |

The rainfall baseline provides limited predictive discrimination on the 2025 validation set. These results establish benchmarks for comparing Gaussian Naive Bayes, Logistic Regression, and the third AI/ML method.