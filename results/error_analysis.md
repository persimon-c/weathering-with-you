# Per-cluster and error analysis

## Reproduce

Run from the repository root:

```bash
python scripts/run_per_cluster.py
python scripts/run_error_analysis.py
```

Both scripts use the main 1% labels, training data from
2022–2024, and validation data from 2025.

Settings match `results/hyperparameters.md`, including KNN
k=5 and logistic regression seed 42. Preprocessing is fitted
on training features only. Baselines use raw features.

The scripts replace their output tables without modifying
the processed dataset or `results/runs.csv`. Test predictions
and evaluation remain reserved for Milestone 4.

## Output tables

| File | Contents |
|---|---|
| `tables/per_cluster.csv` | 30 rows: six methods on five validation storm clusters, including metrics, confusion counts, sample sizes, and settings |
| `tables/validation_errors.csv` | 1,452 incorrect method-row predictions, including coordinates, elevation, rainfall, and error type |
| `tables/error_groups.csv` | 1,368 summaries by method and cluster, cell, elevation range, or rainfall range |

An observation can be misclassified by several methods.
Therefore, 1,452 error rows do not mean 1,452 distinct
dataset observations.

Each validation row belongs to one group within each
grouping dimension. Do not sum counts across different
dimensions.

## Definitions

- False negative (FN): a flooded row predicted not flooded.
- False positive (FP): a non-flooded row predicted flooded.
- False negative rate (FNR): FN divided by actual positives.
- False positive rate (FPR): FP divided by actual negatives.
- Error rate: (FN + FP) divided by group size.

Rates are fractions from 0 to 1. Undefined rates and metrics
are saved as blank CSV fields. Correct predictions remain
in the denominators.

Elevation groups:

- Below 100 m.
- 100 to below 500 m.
- At least 500 m.

Three-day rainfall groups:

- Below 20 mm.
- 20 to below 50 mm.
- At least 50 mm.

These are descriptive bins, not tuned prediction thresholds
or official hazard classifications.

Cells are identified by center coordinates in
`latitude,longitude` order. Administrative region names
are not assigned without a geographic lookup.

## Validation overview

The validation split contains 987 rows:
184 flooded and 803 not flooded.

| Method | False negatives | False positives |
|---|---:|---:|
| Majority | 184 | 0 |
| Rainfall | 112 | 300 |
| Place history | 68 | 54 |
| Naive Bayes | 104 | 179 |
| Logistic regression | 62 | 199 |
| KNN | 150 | 40 |

Balanced accuracy by storm cluster:

| Method | 14 | 15 | 16 | 17 | 18 |
|---|---:|---:|---:|---:|---:|
| Majority | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| Rainfall | 0.5000 | 0.5071 | 0.5252 | 0.5003 | 0.4255 |
| Place history | 0.8750 | 0.8088 | 0.7424 | 0.8838 | 0.8208 |
| Naive Bayes | 0.5000 | 0.5863 | 0.6023 | 0.6477 | 0.6180 |
| Logistic regression | 0.5000 | 0.7064 | 0.6414 | 0.7711 | 0.6779 |
| KNN | 0.5000 | 0.6015 | 0.5446 | 0.4871 | 0.5882 |

Cluster sizes are 6, 96, 288, 214, and 383 rows for
clusters 14–18, respectively.

## Findings

1. **Small clusters need cautious interpretation.**
   Cluster 14 has only six rows: four positives and two
   negatives. Place history scores 0.8750 balanced accuracy,
   but a single changed prediction can substantially affect
   the result.

2. **Place history leads balanced accuracy in all five clusters.**
   Its scores range from 0.7424 to 0.8838. It misses 68 floods,
   slightly more than logistic regression's 62, but produces
   only 54 false alarms compared with logistic regression's
   199.

3. **Cluster 16 has the largest missed-flood count for every method.**
   It contains 81 of the 184 flooded validation rows.
   Place history misses 39/81 positives (48.1%), compared with
   13/34 (38.2%) in cluster 15. Place history's false alarms
   peak in cluster 18 at 31/338 negatives (9.2%). Its cluster
   17 count is smaller, 16/194, with a similar FPR of 8.2%.

4. **Several methods repeatedly fail on the same cells.**
   At center (8.25, 123.25), place history, Naive Bayes, and
   KNN each miss six floods across seven validation rows;
   logistic regression misses five. At (10.25, 123.25),
   Naive Bayes produces nine false alarms across nine rows,
   and logistic regression produces eight. These shared
   locations show repeated spatial errors without establishing
   a regional cause.

5. **Naive Bayes shows an elevation-associated error pattern.**
   At elevations of at least 500 m, it predicts all 69 rows
   as flooded. It detects all 25 positives but falsely flags
   all 44 negatives, giving 100% FPR. Below 100 m, it misses
   40/46 positives, giving 87.0% FNR.

6. **Logistic regression shows a similar elevation tradeoff.**
   Below 100 m, it misses 37/46 positives (80.4% FNR).
   At elevations of at least 500 m, it misses only 3/25
   positives but falsely flags 38/44 negatives (86.4% FPR).
   These associations do not isolate elevation from correlated
   location and weather features.

7. **KNN misses floods across all rainfall groups.**
   It misses 150/184 positives overall, including all 46
   positives below 100 m. Its FNR is 35/41 (85.4%) below
   20 mm of three-day rain, 54/70 (77.1%) at 20 to below
   50 mm, and 61/73 (83.6%) at 50 mm or above. Its low
   false-positive count comes with poor flood recall.

8. **The rainfall rule produces many high-rain false alarms.**
   Its training-selected cutoff is 50.98 mm. In the group
   with at least 50 mm of three-day rain, it detects 72/73
   positives but falsely flags 300/307 negatives (97.7% FPR).
   All 111 positives below 50 mm are missed. Cluster 18 has
   148 false alarms and balanced accuracy of 0.4255.
   Place history's FPR in the same high-rain group is
   19/307 (6.2%), showing that rain alone poorly separates
   these validation labels.

## Limitations

These findings describe the sampled cells with detected
flood polygons. Repeated passes are correlated, and the
PhilSA labels are not ground-validated.

Weather and elevation come from a coarse grid. Elevation
at the cell center does not describe the terrain of every
flooded polygon.

The results do not establish causation or estimate error
rates for every tile in the Philippines. Model settings and
the main label threshold were not changed based on this
analysis.