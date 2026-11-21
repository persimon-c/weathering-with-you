# Decision Threshold Tuning (M4 Validation)

## Reproduce

Run from the repository root:

```bash
python scripts/run_threshold_tuning.py
```

The script saves `results/tables/threshold_tuning.csv` containing 72 rows:
four tunable methods across an 18-point decision-threshold grid $\tau$,
evaluated on the 2025 validation set with the fixed 1.0% flooded-area label.

To also inspect the place-only KNN threshold sweep (distance dilution ablation):

```bash
python scripts/run_threshold_tuning.py --place-only
```

## Distinction from Label Threshold Sensitivity

This experiment operates along a different axis from `scripts/run_thresholds.py`:

- **Label threshold sensitivity** (`scripts/run_thresholds.py`):
  Varies the *ground-truth label definition* (`flood_share > {0.5, 1.0, 2.0}%`).
  Checks whether findings are sensitive to the chosen definition of a "flooded" cell.
- **Decision threshold tuning** (`scripts/run_threshold_tuning.py`):
  Holds the label definition fixed at the canonical 1.0% threshold and varies
  the classifier's *decision cutoff* $\tau \in [0.02, 0.90]$.
  Finds the optimal operating point on the precision–recall curve.

## The 18-Point Decision-Threshold Grid

Per Milestone 4 protocol (`Evaluation-Protocol.md:117` and `design.md`), every
tunable method receives a uniform 18-point probability cutoff grid:

$$\tau \in \{0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90\}$$

A sample is predicted positive (flooded) if its predicted probability $\hat{p} \ge \tau$.

## Tunable Methods and Optimal Operating Points

Selection metric is **balanced accuracy** (unweighted mean of sensitivity and specificity),
the primary metric for the ~18.6% positive validation class. Ties are broken in favor of
lower $\tau$ (higher recall / disaster safety).

| Method | Feature Set | Default BalAcc ($\tau=0.50$) | Tuned $\tau^*$ | Tuned BalAcc | Recall | Precision | F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **Place History** | Cell ID | 0.781593 | **0.30** | **0.856608** | 0.831522 | 0.616935 | 0.708333 | Dominant baseline; lowers FN from 68 to 31 |
| **Logistic Regression** | Full (15) | 0.707611 | **0.45** | **0.716786** | 0.733696 | 0.359043 | 0.482143 | Balances recall gain against precision cost |
| **Gaussian Naive Bayes** | Place+Weather (14) | 0.540372 | **0.05** | **0.659108** | 0.798913 | 0.275797 | 0.410042 | Optimum at $\tau=0.05$; $\tau=0.10 \to 0.6547$ |
| **KNN ($k=5$)** | Full (15) | 0.567485 | **0.02** | **0.664654** | 0.782609 | 0.283465 | 0.416185 | Plateau across $\tau \in [0.02, 0.20]$ from discrete votes |
| *KNN ($k=5$, place-only)* | Place (3) | *0.795238* | ***0.02*** | ***0.853670*** | *0.940217* | *0.480556* | *0.636029* | *Spatial-only KNN approaches Place History* |

## Analysis and Findings

### 1. Place History Dominance Widens Under Fair Tuning
At the default $\tau=0.50$, Place History achieves 0.7816 balanced accuracy.
When tuned to $\tau^*=0.30$, its balanced accuracy increases by +0.0750 to **0.8566**,
reducing false negatives from 68 down to 31 (recall increases from 63.0% to 83.2%).
The gap between Place History and the best learned model (Logistic Regression, 0.7168)
widens from +0.0740 to **+0.1398**.

### 2. Logistic Regression Precision–Recall Trade-Off
Logistic regression reaches its peak balanced accuracy of 0.7168 at $\tau^*=0.45$.
Shifting $\tau$ from 0.50 to 0.45 increases recall from 66.3% to 73.4% (+13 true positives),
incurring 42 additional false positives (precision drops from 38.0% to 35.9%).
For flood warning where missed inundations are costly, this operating point is favorable.

### 3. Boundary-Pinned Low-$\tau$ Optima for Generative and Distance Models
- **Gaussian Naive Bayes:** The conditional independence assumption over-confidently
  concentrates probabilities toward zero or one. The maximum balanced accuracy occurs
  at $\tau^*=0.05$ (0.6591). At the $\tau \le 0.10$ boundary, it achieves 0.6547.
- **KNN ($k=5$):** Because predictions are fractions of $k=5$ neighbors, scores are
  strictly discrete: $\{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$. Any threshold $\tau \in (0.0, 0.20]$
  predicts flooded if at least 1 neighbor flooded, yielding an identical balanced accuracy
  plateau of **0.6647** across $\tau \in [0.02, 0.20]$. $\tau^*=0.02$ is reported as the
  lowest point of this optimal plateau.

### 4. KNN Distance Dilution Demonstrated at Fixed $k=5$
Holding $k=5$ constant across feature sets:
- **Place only (3 features: lat, lon, elevation):**
  At default $\tau=0.50$: BalAcc = **0.7952**.
  Tuned at $\tau^*=0.02$: BalAcc = **0.8537** (recall 94.0%, approaching Place History).
- **Full features (15 features: place + 9 weather + season):**
  At default $\tau=0.50$: BalAcc = **0.5675**.
  Tuned at $\tau^*=0.02$: BalAcc = **0.6647**.

Adding 9 noisy weather dimensions dilutes the spatial Euclidean distance, pulling nearest
neighbors away from the same geographic cell across past storms and reducing performance
by ~23 points at $\tau=0.50$ and ~19 points when tuned.

## Non-Tunable Methods (Excluded from the Probability Grid)

Two baseline methods are structurally excluded from the probability $\tau$ sweep:

1. **Majority baseline:** Predicts the training majority class (all dry). Exports a
   single constant score equal to training prevalence (0.169888). At $\tau \le 0.15$
   it predicts all flooded; at $\tau \ge 0.20$ it predicts all dry. In both cases,
   balanced accuracy is identically **0.5000**.
2. **Rain rule baseline:** Operates on an empirical rainfall cutoff in millimeters
   (50.98 mm, tuned on training balanced accuracy). Its exported scores are ranking-only
   transformations ($\frac{1}{2} + \frac{\arctan(\text{rain\_3d})}{\pi} \in [0.905, 0.998]$).
   Every grid threshold $\tau \le 0.90$ falls below the minimum score, causing the rule
   to predict all-flooded (balanced accuracy identically **0.5000** across the grid).

## Budget Parity and Descope Notes

- **Fair budget:** Each of the 4 tunable methods is evaluated on exactly 18 configurations
  (18 $\tau$ values), totaling $4 \times 18 = 72$ validation evaluations.
- **$k=51$ descope:** In Milestone 3, $k \in \{5, 11, 21, 51\}$ was evaluated in
  `runs.csv`. $k=5$ won on validation balanced accuracy. $k=51$ is explicitly descoped
  from Milestone 4 tuning to keep focus on decision threshold calibration at the optimal $k$.
