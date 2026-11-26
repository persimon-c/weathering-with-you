# CMSC 170 AI Project: Milestone 4

> **Draft.** Results and code facts are final. Lines marked `TODO` need a member's input before the PDF export.

**Group name:** Weathering With You

**Repo link:** https://github.com/persimon-c/weathering-with-you

**Project page:** https://persimon-c.github.io/weathering-with-you/

**Date:** 2026-11-26

**Members**

- Simone Pauline M. Dagondon
- Andrew Qlyve Paul C. Bundoc
- Juan Carlos J. Bernardino
- Nicholas O. Pacoma

## What was done

1. **One-command reproduction (#43).** `python scripts/run_all.py` checks the environment, runs the 187 unit tests, rebuilds every validation table and figure from the committed dataset, and checks the 8 result tables against committed SHA-256 hashes. A leakage audit covers the 11-point checklist (`results/leakage_audit.md`), and the code is MIT licensed.
2. **Fair threshold tuning (#44).** Every tunable method (place history, Naive Bayes, logistic regression, KNN) gets the same 18-point decision-threshold grid, $\tau \in \{0.02, \dots, 0.90\}$, chosen by validation balanced accuracy (`results/threshold_tuning.md`).
3. **Cluster bootstrap intervals (#45).** Paired 95% intervals for each learned method against place history, resampling whole storm clusters 1,000 times (`results/bootstrap.md`).
4. **One test run (#46).** The 2026 test set was scored once, after the code freeze, by a script that refuses to run twice (`results/test_evaluation.md`).
5. **Project page (#47).** A static GitHub Pages site with all ten required sections, the validation and test figures, and MathJax.
6. **Slides (#48).** `TODO(team)`: `docs/slides.pdf` and the timed rehearsal.

## Who did it

| Member | Task | Evidence |
|---|---|---|
| Simone | Reproduction runner, checksum manifest, license, leakage audit | PR #50, `scripts/run_all.py`, `results/leakage_audit.md` |
| Simone | Fair decision-threshold tuning | PR #51, `scripts/run_threshold_tuning.py`, `results/tables/threshold_tuning.csv` |
| Simone | Ran the single 2026 test evaluation after the freeze | commits `145f924` to `05d7607`, `results/logs/test_eval.log` |
| Nicholas | Cluster bootstrap and paired difference intervals | PR #52, `src/metrics.py::cluster_bootstrap`, `results/tables/comparison_validation_final.csv` |
| Nicholas | Gated single-run test evaluation script and test write-up | PR #53, `scripts/evaluate_test.py`, `results/test_evaluation.md` |
| Nicholas | Reviewed and merged the M4 code PRs, fixed the Windows checksum failure | reviews on PRs #50 and #51 |
| Carlos | Project page: layout, navigation, MathJax, figures, deployment | PR #54, `docs/index.html` |
| Qlyve | Research content for the project page | PR #54 |
| `TODO(team)` | Slides and rehearsal | `docs/slides.pdf` |

## Dataset and task

**Dataset.** Philippine Space Agency (PhilSA) Sentinel-1 flood extent maps from the Humanitarian Data Exchange (CC BY), joined with NASA POWER daily weather and elevation.

**Task.** Predict whether a 0.5° cell is flooded in a satellite pass, from geography and the weather before the pass.

**Target.** `flooded = 1` when more than 1.0% of the cell's area is flooded.

**Rows and features.** 3,531 rows (one cell in one pass), 14 raw features plus a missing-soil flag = 15 model inputs. Split by year: train 2022–2024 (1,966 rows, 13 storm clusters), validation 2025 (987 rows, 5 clusters), test 2026 (578 rows, 3 clusters).

## Methods status

| Method | Family | Status | Notes |
|---|---|---|---|
| Gaussian Naive Bayes | Probabilistic | Tuned | $\tau^* = 0.05$; leaves out the missing-soil flag |
| Logistic regression | Linear | Tuned | L2 $\lambda = 0.01$, learning rate 0.1, 2,000 epochs, balanced class weights, $\tau^* = 0.45$ |
| K nearest neighbors | Instance-based | Tuned | $k = 5$, $\tau^* = 0.02$ (plateau from 0.02 to 0.20) |
| Majority, rain rule | Baselines | Fixed | Rain rule cutoff 50.98 mm, chosen on training rows |
| Place history | Baseline | Tuned | Each cell's training flood rate, $\tau^* = 0.30$ |

All methods and metrics are written from scratch with NumPy. The small MLP was not added (recorded decision 4 in `design.md`).

## Results

### Validation (2025), at each method's tuned threshold

Every method is deterministic at these settings: across seeds 0 to 9, logistic regression's balanced accuracy is 0.716786 every time (standard deviation 0.000000) and its ROC AUC moves by about 0.00002; the other methods have no random step. Uncertainty therefore comes from the storm-cluster bootstrap, not from seeds.

| Method | Balanced accuracy | Macro F1 | ROC AUC | Δ BalAcc vs place history [95% CI] |
|---|---:|---:|---:|---|
| Majority | 0.5000 | 0.4486 | 0.5000 | – |
| Rain rule | 0.5089 | 0.4842 | 0.5122 | – |
| **Place history** | **0.8566** | **0.8133** | **0.9069** | reference |
| Naive Bayes | 0.6591 | 0.5368 | 0.6866 | −0.198 [−0.246, −0.129] |
| Logistic regression | 0.7168 | 0.6385 | 0.7524 | −0.140 [−0.210, −0.088] |
| KNN ($k = 5$) | 0.6647 | 0.5505 | 0.7066 | −0.192 [−0.242, −0.157] |

Every interval lies below zero. With only 5 validation clusters (Kish effective number 3.42), the intervals are coarse, so the per-cluster results are the main evidence: place history leads in all 5 validation clusters.

### Test (2026), scored once at frozen commit `8c74e98`

| Method | Balanced accuracy | Macro F1 | ROC AUC | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Majority | 0.5000 | 0.4442 | 0.5000 | 0 | 0 | 116 |
| Rain rule | 0.5723 | 0.5000 | 0.6023 | 72 | 220 | 44 |
| **Place history** | **0.8963** | **0.8489** | **0.9375** | 105 | 52 | 11 |
| Naive Bayes | 0.6172 | 0.4622 | 0.6633 | 100 | 290 | 16 |
| Logistic regression | 0.6648 | 0.5314 | 0.7224 | 99 | 242 | 17 |
| KNN ($k = 5$) | 0.6282 | 0.5075 | 0.6815 | 92 | 248 | 24 |

Per test cluster, place history is first in all three, by at least 0.14 balanced accuracy (cluster 19: 0.949, cluster 20: 0.917, cluster 21: 0.891). Cluster 21 holds 459 of the 578 rows (79.4%), so the pooled numbers are mostly a cluster 21 result.

From validation to test, the learned models lose 0.04 to 0.05 balanced accuracy while place history gains 0.04. The gap to the best learned model grows from 0.140 to 0.232.

Figures: `results/figures/roc_curves_test.png`, `results/figures/test_map.png`, and the validation figures in `results/figures/`.

### Why each method scored as it did

- **Place beats weather.** Where a cell is decides most of whether it floods: low river basins flood again and again even in moderate rain, and high ground rarely floods even in heavy rain. Place history stores exactly this, a flood rate per cell, and every test cell was seen in training. The rain rule alone is near chance on validation (ROC AUC 0.51).
- **Naive Bayes and logistic regression cannot draw a map.** Naive Bayes fits one bell curve per feature, which over latitude and longitude gives smooth ellipses, not islands and river valleys. Logistic regression draws one straight boundary through the features. Both fall back on elevation, and the validation error analysis (at the default 0.5 threshold) shows it: below 100 m they miss most floods (91% and 80% miss rates), and at 500 m and above they raise false alarms on 84–86% of dry cells.
- **KNN is diluted by weather.** On place alone, KNN's nearest rows are the same cell in earlier storms, so it nearly matches place history (0.854 tuned vs 0.857). Adding nine weather columns pulls neighbours away from that cell and drops it to 0.665 at the same $k = 5$ and the same tuning.
- **Low thresholds.** Naive Bayes and KNN do best at $\tau \le 0.05$ because their scores bunch near 0: Naive Bayes is over-confident, and KNN's score is a vote fraction in steps of 0.2.

## Problems encountered

| Problem | Impact | What was tried | Current status |
|---|---|---|---|
| The rain rule's exported score is a ranking-only transform, not a probability | A probability threshold sweep gives it a flat 0.500 | Kept its own millimetre cutoff, tuned on training rows, and reported it as not $\tau$-tunable | Resolved |
| KNN's score has only 6 values | Many thresholds tie | Report the lowest threshold of the best plateau and document the plateau | Resolved |
| Only 5 validation and 3 test storm clusters | Bootstrap intervals are coarse | Report per-cluster results as the main evidence and the intervals as a caveated summary | Accepted limitation |
| Windows checkouts rewrote the committed tables with CRLF line endings | Checksum verification and 3 tests failed on Windows | `.gitattributes` keeps the generated tables LF | Resolved |
| The plan estimated "about 1.3 effective clusters" | Would understate the validation sample | Computed the Kish effective number from the cluster sizes: 3.42 | Resolved |
| Test year has 0 unseen cells and is 79% one cluster | Test measures new storms in known places only | The 2°×2° block-holdout experiment was descoped for time | Stated as a limitation |

## Limitations

- The labels come from PhilSA's own detection model and are not ground-checked. PhilSA notes that radar underestimates floods in cities and dense vegetation.
- Water in rice paddies is detected as flooding, which ties the label to farming seasons as well as storms.
- NASA POWER is a coarse 0.5° by 0.625° model grid, not rain-gauge data.
- Cells at the edge of a satellite pass are only partly observed.
- The test year holds only 3 storm clusters, one of them 79% of the rows, and every test cell was seen in training, so performance on new places is not measured.

## Possible improvements

- Hold out 2°×2° blocks to measure how the methods do in places with no flood history.
- Give KNN feature weights so place is not drowned out by the weather columns.
- Combine place history with weather, for example as a feature for logistic regression, to test whether weather adds anything once place is known.
- Use rain-gauge or radar rainfall instead of the coarse NASA POWER grid.

## Conclusion

Across 2025 validation and the single 2026 test run, a cell's own flood history predicts flooding far better than any of the weather-based models: balanced accuracy 0.896 against 0.665 for the best learned model on test, with place history first in every storm cluster. In the Philippine archipelago, at this scale, where a cell is matters much more than how much it rained in the days before the pass.

## Plan for the next milestone

M4 is the final milestone. Remaining submission tasks:

| Task | Owner | Due |
|---|---|---|
| Merge the project page (PR #54) and serve GitHub Pages from `main` | Carlos | 2026-11-26 |
| Commit `docs/slides.pdf` and link it from the page | `TODO(team)` | 2026-11-26 |
| Each member checks their row in "Who did it" and `task-log.md` | All | 2026-11-26 |
| Clean-clone check: `python scripts/run_all.py` passes | `TODO(owner)` | 2026-11-26 |
| Peer evaluation forms | All | 2026-11-26 |
| Export this report to PDF and upload the bundle to Google Classroom | `TODO(owner)` | 2026-11-26 23:59 |

## Evidence links

- Repository. https://github.com/persimon-c/weathering-with-you
- Project page. https://persimon-c.github.io/weathering-with-you/
- Results tables. `results/tables/comparison_validation_final.csv`, `results/tables/comparison_test.csv`, `results/tables/per_cluster_test.csv`
- Write-ups. `results/threshold_tuning.md`, `results/bootstrap.md`, `results/test_evaluation.md`, `results/error_analysis.md`
- Task log. `task-log.md`
- Slides. `docs/slides.pdf` (`TODO(team)`)

## AI tool use disclosure

AI assistants (Claude, ChatGPT, GitHub Copilot) were consulted for boilerplate code, plotting templates, code review and documentation drafting. All model and metric code is our own, and every number in this report comes from the committed result tables. `TODO(team)`: confirm this matches the README disclosure.

## Questions for the instructor

1. `TODO(team)`: add any, or delete this section.
