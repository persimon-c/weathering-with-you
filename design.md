# Design

This is the plan for the project from Milestone 2 to presentation day. It records what we predict, from which data, how we split and score, which methods we compare, and what each milestone needs. Change it by group agreement and note each change in `task-log.md`.

## Problem

Typhoons and the southwest monsoon flood large parts of the Philippines every year. The Philippine Space Agency (PhilSA) maps the floods from satellite radar after each event, but only after the fact.

**Research question.** How well can we predict which areas flood in a new storm, and how much does weather before the storm add beyond the place itself?

We ask it this way because of a data check on 4 and 5 November, run on every Sentinel-1 file. It found that where an area is explains most of whether it floods, and that rain before the pass adds very little once the place is known. The project measures that properly and explains why each method behaves the way it does.

## Changes from Milestone 1

The dataset, the task type and the methods are the ones approved at M1. These details changed after the data check, and the M2 report explains them.

| Item | At M1 | Now | Why |
|---|---|---|---|
| Question | Predict if an area flooded from the weather before the flood | How well can we predict flooding in a new storm, and how much does weather add beyond the place | Weather alone predicted about at chance. A 3-day rain rule scored AUC 0.51 on 2025 |
| Data | 87 events, 83 with shapefiles, all sensors | 61 Sentinel-1 passes from 63 shapefiles | Other sensors detect floods differently and publish different map layouts |
| Row | One tile in one flood event | One satellite pass in one 0.5 degree cell with at least one flood polygon | Some events have several passes, and almost every land cell has some detected water |
| Label | Flooded or not | Flooded if more than 1 percent of the cell is flooded | Detection versus none would make nearly every row flooded |
| Features | Rainfall, soil wetness, humidity, temperature | The same weather, plus place (latitude, longitude, elevation) and season | Place explained most of the flooding in the data check |
| Split | Not decided | Train 2022 to 2024, validate 2025, test 2026 | Passes of the same storm must not sit on both sides |
| Methods | Gaussian Naive Bayes, KNN, logistic regression, optional small neural network | Unchanged | |

## Data

| Source | What it gives | Access |
|---|---|---|
| PhilSA flood extent maps on the Humanitarian Data Exchange (HDX) | Flood polygons per satellite pass, the labels | `https://data.humdata.org/organization/philsa`, CC BY, no login |
| NASA POWER daily point API, row 4 of the instructor's list | Rain, humidity, temperature and soil wetness per day, plus grid elevation in the response header | `https://power.larc.nasa.gov/api/temporal/daily/point`, no login |

The instructor approved using the two together as one task.

We use Sentinel-1 passes only. Other sensors publish different map layouts and detect floods differently.

| Fact | Value |
|---|---|
| Passes | 61 Sentinel-1 passes, August 2022 to September 2026, from 63 shapefiles. The three regional pieces of 2024-10-27 are one pass |
| Rows | 3,531, one per pass and 0.5 degree cell, over 253 cells |
| Split sizes | Train 1,966 rows from 33 passes, validation 987 from 16, test 578 from 12 |
| Flooded at the 1 percent threshold | 17.0 percent of training rows, 18.6 percent of validation rows |
| Storm clusters | 21 groups of passes no more than 7 days apart: 13 in training, 5 in validation, 3 in test |

### Getting the data

- **Download by pass, not by HDX page.** Twelve HDX pages hold more than one shapefile, from different sensors or regions. Pick the files named `*_fld_s1_shp*.zip`.
- **Go slow.** HDX blocks scripts that send requests in bursts and answers `202` with an empty body. One request every 15 seconds from one computer works. On a `202`, wait a few minutes and retry.
- **Watch for nested zips.** Five 2026 files end in `.zip.zip` and hold a zip inside a zip.
- **Keep raw files out of git.** The 63 shapefiles are about 430 MB. Put them in `data/raw/`, add that folder to `.gitignore`, and commit only the processed CSV and the download script.

## Task

| Item | Decision |
|---|---|
| Row | One Sentinel-1 pass in one 0.5 degree cell that has at least one PhilSA flood polygon in that pass |
| Target | `flooded` = 1 if the flooded area is more than 1 percent of the cell's area, else 0 |
| Event ID | The pass, written as date and pass time, for example `20260907_1807` |
| Leakage rules | No feature taken from the flood map. No weather from the image date or later. Rain windows end the day before the pass |

Almost every land cell in a pass has some detected water, so "any flood polygon" cannot be the label. The threshold has to be frozen before anyone trains a model. Results at 0.5 and 2 percent are reported as a sensitivity check.

### Computing the flooded area

- Read `.shp` and `.dbf` with `struct`. Every file is polygon type 5 in WGS84.
- Area of each polygon with the shoelace formula, after scaling longitude by the cosine of the latitude so degrees become kilometers.
- Polygons can have holes, so subtract inner rings. Some polygons are wound the wrong way, so take the absolute area per polygon.
- Assign each polygon to the cell that holds its bounding-box center.
- Only 2026 files have a `Class` field, so count every polygon as flooded in every year.

## Features

| Group | Columns | Source |
|---|---|---|
| Place | `lat`, `lon`, `elevation` | Cell center, MERRA-2 elevation from the NASA POWER header |
| Weather | `rain_1d`, `rain_3d`, `rain_7d`, `rain_14d`, `rain_21d`, `rain_30d`, `humidity`, `temperature`, `soil_wetness` | NASA POWER daily. Rain sums end the day before the pass. The rest are the day before |
| Season | `month_sin`, `month_cos` | Month of the pass |

### NASA POWER parameters

| Column | NASA POWER parameter | Unit |
|---|---|---|
| `rain_*` | `PRECTOTCORR`, corrected total precipitation | mm per day, summed over the window |
| `humidity` | `RH2M`, relative humidity at 2 m | percent |
| `temperature` | `T2M`, air temperature at 2 m | degrees Celsius |
| `soil_wetness` | `GWETROOT`, root-zone soil wetness | 0 to 1 |
| `elevation` | Header line "elevation from MERRA-2" | meters |

Request them together with `community=AG`. Use `GWETROOT` rather than `GWETTOP`, the surface layer, because the root zone stays wet over several days of rain and so reflects how saturated the ground is before a flood.

Soil wetness is missing on every day at 8 cells that NASA POWER treats as ocean. Decide one policy before M2 ends, either drop the column or fill it and add a missing flag, and write it in the README.

All three methods use the same columns and the same preprocessing, fitted on training rows only.

## Processed dataset

The pipeline writes one file, `data/processed/flood_data.csv`, and every method reads only that file. Column names are the contract between the pipeline and the models.

| Column | Meaning |
|---|---|
| `event_id` | Pass ID, for example `20260907_1807` |
| `date`, `year` | Image date and its year |
| `cluster_id` | Storm cluster number |
| `split` | `train`, `val` or `test`, decided by year |
| `lon_min`, `lat_min` | South-west corner of the cell |
| `lon`, `lat` | Cell center |
| `elevation` | Meters |
| `n_polygons` | Flood polygons in the cell |
| `flood_km2`, `flood_share` | Flooded area, and the share of the cell in percent |
| `flooded` | Label, 0 or 1 |
| `rain_1d` to `rain_30d` | Millimeters |
| `humidity`, `temperature`, `soil_wetness` | Percent, degrees Celsius, and 0 to 1 |
| `month_sin`, `month_cos` | Season |

## Split

| Split | Years | Use |
|---|---|---|
| Train | 2022 to 2024 | Fit every model and every preprocessing step |
| Validation | 2025 | Choose hyperparameters and the decision threshold |
| Test | 2026 | Score once, after everything is frozen |

Never shuffle events at random. Passes of the same storm are a few days apart, so a random split puts the same storm on both sides and inflates the scores. The year split never cuts a storm cluster.

The test year has only 3 storm clusters, so report results per cluster too, and in M4 add bootstrap intervals that resample clusters.

## Methods

| Method | Family | Why it fits | What the data check suggests |
|---|---|---|---|
| Gaussian Naive Bayes | Probabilistic | Fast baseline, tests whether features carry independent evidence | Weak on place, because bell curves over latitude and longitude cannot draw a map |
| Logistic regression | Linear | Weights show which features matter | Weak on raw place for the same reason, better with elevation |
| K nearest neighbors | Instance-based | Similar places in similar weather should flood alike | Strong on place alone, because a cell's nearest rows are the same cell in other storms. Adding weather columns hurts it unless features are weighted |
| Small MLP, optional | Neural | Can combine place and weather nonlinearly | Add only if M3 is on time |

All four are written from scratch with NumPy. No pandas, scikit-learn or similar.

## Baselines

| Baseline | Rule |
|---|---|
| Majority | Always predict not flooded |
| Rain rule | Flooded if 3-day rain is above a cutoff chosen on training rows |
| Place history | Each cell's flood rate in the training years. This is the bar the methods have to beat |

In the data check on 2025, the rain rule scored AUC 0.51, no better than chance, and place history scored 0.92.

## Metrics

All written from scratch.

- Balanced accuracy and macro F1 as the headline numbers, because only about 17 percent of rows are flooded.
- Precision and recall for the flooded class.
- ROC AUC, which does not depend on the decision threshold.
- Confusion matrix.
- Every metric per storm cluster as well as overall.

## Experiments

| Experiment | Question | Milestone |
|---|---|---|
| Main | Train 2022 to 2024, validate 2025, all methods on the same features | M2 and M3 |
| Feature groups | Place only, then place plus weather, then place plus weather plus season | M3 |
| Threshold | Repeat at 0.5 and 2 percent | M3 |
| New places, optional | Hold out blocks of 2 by 2 degrees, no place history, to see how hard unseen areas are | M4 if time allows |
| Final test | One run on 2026 with everything frozen | M4 |

## Code layout

| Path | Purpose |
|---|---|
| `scripts/download.py` | Fetch the Sentinel-1 shapefiles slowly into `data/raw/` |
| `src/shapefile.py` | Read `.shp` and `.dbf` with `struct` |
| `src/cells.py` | Flooded area per pass and cell |
| `src/weather.py` | Fetch NASA POWER per cell, cache the raw responses, skip the header, treat `-999` as missing |
| `scripts/build_dataset.py` | Join everything into `data/processed/flood_data.csv` |
| `src/data.py` | Load the processed CSV |
| `src/preprocess.py` | Standardize with training statistics, handle missing values |
| `src/split.py` | Split by the `split` column |
| `src/metrics.py` | All metrics |
| `src/models/` | `naive_bayes.py`, `logistic.py`, `knn.py`, and `mlp.py` if added |
| `src/gui.py`, `main.py` | Interface |
| `results/` | Saved predictions, metric tables, figures |

Allowed libraries: the Python standard library, NumPy for arrays and basic linear algebra, and Matplotlib for figures.

### Changes the `carlos` branch needs to match this plan

| File | Now | Plan |
|---|---|---|
| `src/data.py` | Reads `rainfall`, `soil_wetness`, `humidity` and `temperature` | Read the columns above, including place and season |
| `src/split.py` | Shuffles events at random, 80/20 | Split by year through the `split` column |
| `src/metrics.py` | Accuracy, precision, recall, F1 | Add balanced accuracy, macro F1 and ROC AUC |
| `data/processed/flood_data.csv` | Does not exist yet | Built by the pipeline above. This is the M2 critical path |

## Milestone plan

| Milestone | Due | Must have | Owner |
|---|---|---|---|
| M2 | Thu 12 Nov | Download and pipeline code written from scratch, processed CSV, split, metrics, majority baseline, Naive Bayes and logistic regression with first validation results, problems and next plan, task log | To assign |
| M3 | Thu 19 Nov | KNN, all metrics own-code, comparison tables and graphs, feature-group and threshold experiments | To assign |
| M4 | Thu 26 Nov | Tuning on validation only, one test run, per-cluster results and intervals, complete project page | To assign |
| Presentation | To be announced | Slides, every member presents a part, page and repo frozen | All |

If the full download is not finished for M2, build the CSV from the passes already downloaded, report results on those, and say so in the progress report.

## Open decisions

1. Freeze the 1 percent threshold before training.
2. Keep rows as cells with at least one flood polygon, or use every published tile. Published tiles add about 13 percent more rows, nearly all dry, and need each pass's tile list and a tile-number-to-coordinates table.
3. Pick the soil wetness policy.
4. Add the MLP or not, decided at M3.
5. Assign an owner to each milestone task.

## Limitations to state on the page

- The labels come from PhilSA's own detection model and are not ground-checked. PhilSA says radar underestimates floods in cities and dense vegetation.
- Water in rice paddies is detected as flooding, which ties the label to farming seasons as well as storms.
- NASA POWER is a coarse model grid of 0.5 by 0.625 degrees, not rain-gauge data.
- Cells at the edge of a satellite pass are only partly observed.
- The test year holds only 3 storm clusters.

## References

- Philippine Space Agency flood extent maps, Humanitarian Data Exchange, `https://data.humdata.org/organization/philsa`, CC BY.
- NASA POWER Project, daily point API, `https://power.larc.nasa.gov/`.
- CMSC 170 AI Project guidelines.
