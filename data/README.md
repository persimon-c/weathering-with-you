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
| `processed/flood_data.csv` | Yes, once built | One row per satellite pass and 0.5 degree cell. Columns are listed in `design.md` |

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

## Getting the raw files

Run this from the repository folder:

```
python scripts/download.py
```

It downloads the 63 `s1` rows of `acquisitions.csv` into `data/raw/`, which git ignores. A full run takes about 50 minutes, most of it the 15-second pauses and the downloads. Run it again after a stop and it skips the files that are already complete. `python scripts/download.py --dry-run` lists what is still missing without downloading.

- HDX blocks scripts that send requests in bursts and answers `202` with an empty body. The script waits 15 seconds between files and waits 2, 5 and then 10 minutes before retrying a file after a `202`.
- A file only counts as complete when its size matches the `size` column and it starts like a zip.
- Five 2026 files end in `.zip.zip` and contain a zip inside a zip. The script downloads them as they are.
