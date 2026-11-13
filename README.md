# Weathering With You

CMSC 170 AI Project

## Members

- Simone Pauline M. Dagondon
- Andrew Qlyve Paul C. Bundoc
- Juan Carlos J. Bernardino
- Nicholas O. Pacoma

## Task

Predict whether a 0.5 degree tile in the Philippines flooded in a Sentinel-1 satellite pass, using only the place, the season and the weather before the pass. The plan, features, split and metrics are in [`design.md`](design.md). The data sources, license and column lists are in [`data/README.md`](data/README.md).

## Requirements

- Python 3
- NumPy

Everything else is the Python standard library. There is no pandas and no scikit-learn: the loader, preprocessing, models and metrics are written from scratch.

## Reproduce the results

Run every command from the repository folder.

The processed dataset, `data/processed/flood_data.csv`, is already in the repository, so steps 1 to 3 are only needed to rebuild it from the raw sources.

1. Download the PhilSA flood shapefiles into `data/raw/` (about 50 minutes, see `data/README.md`):
   ```
   python scripts/download.py
   ```
2. Compute the flooded area per pass and tile into `data/interim/flood_cells.csv` (about a minute), then fetch the NASA POWER weather for each tile into the cache in `data/raw/weather/`:
   ```
   python -m src.cells
   python -m src.weather
   ```
3. Build the processed CSV and the frozen split files in `results/splits/`:
   ```
   python scripts/build_dataset.py
   ```
4. Score the three baselines on the 2025 validation set. This writes `results/baselines_val.json` with per-cluster scores and appends one row per baseline to `results/runs.csv`:
   ```
   python scripts/run_baselines.py
   ```
5. Train Gaussian Naive Bayes, logistic regression and K nearest neighbors on 2022 to 2024, score them on 2025, and append one row per model to `results/runs.csv`. KNN adds one row for each k in 5, 11, 21 and 51, and the script prints the k with the best validation balanced accuracy:
   ```
   python scripts/run_models.py
   ```

Every script appends to `results/runs.csv`, so rerunning adds new rows instead of replacing old ones. Each row records the git commit that produced it. The 2026 test set is not scored until Milestone 4.

## Tests

```
python -m unittest discover -s tests -t .
```

## Layout

| Path | Contents |
|---|---|
| `src/` | Shapefile reader, flooded area per tile, weather fetch, loader, split, preprocessing, metrics, baselines |
| `src/models/` | Gaussian Naive Bayes, logistic regression and K nearest neighbors |
| `scripts/` | Download, dataset build, and the baseline and model runners |
| `data/` | Acquisition list, interim and processed tables (`data/raw/` is not committed) |
| `results/` | Frozen splits, `runs.csv` with one row per run, baseline per-cluster scores |
| `tests/` | Unit tests |
| `task-log.md` | Member contributions per milestone |
