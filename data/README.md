# Data

## Source

Flood labels come from the Philippine Space Agency (PhilSA) flood extent maps on the Humanitarian Data Exchange (HDX): `https://data.humdata.org/organization/philsa`. Weather features come from the NASA POWER daily point API: `https://power.larc.nasa.gov/api/temporal/daily/point`.

License: the PhilSA data is CC BY. Credit "Philippine Space Agency (PhilSA), via the Humanitarian Data Exchange" wherever it is used. PhilSA notes that the maps are not validated with ground data. NASA POWER data is free to use with credit to the NASA POWER Project.

## Files

| Path | In git | Contents |
|---|---|---|
| `acquisitions.csv` | Yes | Every PhilSA shapefile on HDX, 97 rows, listed on 4 November 2026 |
| `raw/` | No | The downloaded shapefile zips, about 430 MB |
| `interim/flood_cells.csv` | Yes | Flooded area per satellite pass and 0.5 degree cell, 3,531 rows over 61 passes |
| `processed/flood_data.csv` | Yes | One row per satellite pass and 0.5 degree cell, 3,531 rows and 25 columns. Built by `scripts/build_dataset.py` |

### `interim/flood_cells.csv` columns

Built by `python -m src.cells` from the zips in `raw/`, which takes about a minute.

| Column | Meaning |
|---|---|
| `event_id` | Satellite pass, date and pass time, for example `20260907_1807`. The three 2024-10-27 regional files share one ID |
| `lon_min`, `lat_min` | South-west corner of the 0.5 degree cell |
| `n_polygons` | Flood polygons whose bounding-box center falls in the cell |
| `flood_km2` | Their total area in square km, holes subtracted |
| `flood_share` | `flood_km2` as a percent of the whole cell, sea included |

Only cells with at least one flood polygon appear.

### `acquisitions.csv` columns

| Column | Meaning |
|---|---|
| `package` | HDX dataset name, one per flood date |
| `file` | Shapefile zip name |
| `date`, `hhmm` | Image date and satellite pass time, local time |
| `sensor` | `s1` is Sentinel-1. Others include `rcm` (RADARSAT), `alos2`, `iceye` and `s2` (Sentinel-2) |
| `size` | File size in bytes |
| `url` | Download link |
| `n_maps_zips` | Number of map image zips in the same HDX dataset |

The project uses the 63 rows with sensor `s1`. Twelve HDX datasets hold more than one shapefile, so download by row here, not by HDX dataset. The three `s1` files dated 2024-10-27 are regional pieces of one pass, so the 63 files make 61 passes.

### `processed/flood_data.csv` columns

Built by joining `interim/flood_cells.csv` with the NASA POWER weather cache:

```
python scripts/build_dataset.py
```

It has 3,531 rows and 25 columns. Sources and license are the same as above. Built 2026-11-12. Hashes are in `results/splits/sha256.txt`.

| Column | Meaning |
|---|---|
| `event_id` | Satellite pass, as in the interim table |
| `date`, `year` | Image date and its year |
| `cluster_id` | Storm number, passes no more than 7 days apart share one |
| `split` | `train` for 2022 to 2024, `val` for 2025, `test` for 2026 |
| `lon_min`, `lat_min` | South-west corner of the 0.5 degree cell |
| `lon`, `lat` | Cell center, the weather point |
| `elevation` | Ground elevation at the cell center in meters |
| `n_polygons`, `flood_km2`, `flood_share` | Copied from the interim table |
| `flooded` | 1 when `flood_share` is over 1 percent, else 0 |
| `rain_1d`, `rain_3d`, `rain_7d`, `rain_14d`, `rain_21d`, `rain_30d` | Rain in mm summed over that many days, ending the day before the pass |
| `humidity`, `temperature`, `soil_wetness` | Weather on the day before the pass |
| `month_sin`, `month_cos` | Month as a point on a circle, so December sits next to January |

Every weather window ends the calendar day before the pass, so no feature uses the image date. `soil_wetness` is blank at 9 cells that NASA POWER treats as ocean.

The split is frozen in `results/splits/`. `train.csv`, `validation.csv` and `test.csv` list the row indices of each set, and `sha256.txt` hashes the processed CSV and those three files, so another member can rebuild the exact same rows.

The same cell flooding across passes, and a date with two passes such as `20251104`, are distinct rows, not duplicates. Keep both.

## Getting the raw files

Run this from the repository folder:

```
python scripts/download.py
```

It downloads the 63 `s1` rows of `acquisitions.csv` into `data/raw/`, which git ignores. A full run takes about 50 minutes, most of it the 15-second pauses and the downloads. Run it again after a stop and it skips the files that are already complete. `python scripts/download.py --dry-run` lists what is still missing without downloading.

- HDX blocks scripts that send requests in bursts and answers `202` with an empty body. The script waits 15 seconds between files and waits 2, 5 and then 10 minutes before retrying a file after a `202`.
- A file only counts as complete when its size matches the `size` column and it starts like a zip.
- Five 2026 files end in `.zip.zip` and contain a zip inside a zip. The script downloads them as they are.
