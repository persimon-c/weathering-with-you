# Data

## Source

Flood labels come from the Philippine Space Agency (PhilSA) flood extent maps on the Humanitarian Data Exchange (HDX): `https://data.humdata.org/organization/philsa`. Weather features come from the NASA POWER daily point API: `https://power.larc.nasa.gov/api/temporal/daily/point`.

License: the PhilSA data is CC BY. Credit "Philippine Space Agency (PhilSA), via the Humanitarian Data Exchange" wherever it is used. PhilSA notes that the maps are not validated with ground data. NASA POWER data is free to use with credit to the NASA POWER Project.

## Files

| Path | In git | Contents |
|---|---|---|
| `acquisitions.csv` | Yes | Every PhilSA shapefile on HDX, 97 rows, listed on 4 November 2026 |
| `raw/` | No | The downloaded shapefile zips, about 430 MB |
| `processed/flood_data.csv` | Yes, once built | One row per satellite pass and 0.5 degree cell. Columns are listed in `design.md` |

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

Download the `s1` rows of `acquisitions.csv` into `data/raw/`, and keep that folder out of git.

- HDX blocks scripts that send requests in bursts and answers `202` with an empty body. Send one request about every 15 seconds and retry after a few minutes on a `202`.
- Five 2026 files end in `.zip.zip` and contain a zip inside a zip.
