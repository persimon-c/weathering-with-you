# Join the cell table with the cached NASA POWER weather into the processed CSV.
#
# One row per satellite pass and cell, carrying the place, the weather windows
# that end the day before the pass, the season, the label, and the storm cluster
# and year split that keep the evaluation honest.
#
# Usage (from the repository folder):
#   python scripts/build_dataset.py

import argparse
import csv
import datetime
import math
import os
import sys

# The build reads the weather module, which lives one folder up
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.weather import CELL, CACHE_DIR, load_cell


CELLS_PATH = os.path.join(ROOT, "data", "interim", "flood_cells.csv")
OUT_PATH = os.path.join(ROOT, "data", "processed", "flood_data.csv")

THRESHOLD = 1.0                       # flooded once the share passes this percent
RAIN_WINDOWS = [1, 3, 7, 14, 21, 30]  # days summed, ending the day before the pass
CLUSTER_GAP = 7                       # passes further apart than this start a new storm
SPLIT_YEARS = {2022: "train", 2023: "train", 2024: "train",
               2025: "val", 2026: "test"}

COLUMNS = [
    "event_id", "date", "year", "cluster_id", "split",
    "lon_min", "lat_min", "lon", "lat", "elevation",
    "n_polygons", "flood_km2", "flood_share", "flooded",
    "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_21d", "rain_30d",
    "humidity", "temperature", "soil_wetness",
    "month_sin", "month_cos",
]


# Image date of a pass ID, e.g. 20260907_1807 -> date(2026, 9, 7)
def pass_date(event_id):
    text = event_id[:8]
    return datetime.date(int(text[:4]), int(text[4:6]), int(text[6:]))


# Storm cluster per pass: passes no more than CLUSTER_GAP days apart share one
def cluster_by_pass(events):
    clusters = {}
    number = 0
    previous = None

    # sorted by date then pass time, so a run of nearby passes chains together
    for event in sorted(events):
        day = pass_date(event)

        if previous is None or (day - previous).days > CLUSTER_GAP:
            number += 1

        clusters[event] = number
        previous = day

    return clusters, number


# Weather columns for one cell on the day before a pass
def weather_row(cell, day_before):
    elevation, dates, values = cell
    index = {date: i for i, date in enumerate(dates)}

    if day_before not in index:
        raise ValueError(f"No weather for {day_before}")

    end = index[day_before]
    row = {"elevation": elevation}

    # each window covers the days ending the day before the pass
    for window in RAIN_WINDOWS:
        start = end - window + 1
        row[f"rain_{window}d"] = sum(values["PRECTOTCORR"][start:end + 1])

    # the single day before the pass, for the slow fields
    row["humidity"] = values["RH2M"][end]
    row["temperature"] = values["T2M"][end]
    row["soil_wetness"] = values["GWETROOT"][end]

    return row


# A number with a fixed number of decimals, blank when missing
def fmt(value, digits):
    if value is None:
        return ""

    return f"{value:.{digits}f}"


def main():
    # Command-line options
    parser = argparse.ArgumentParser(
        description="Build data/processed/flood_data.csv from the cells and weather."
    )
    parser.add_argument("--cells", default=CELLS_PATH,
                        help="cell table (default: data/interim/flood_cells.csv)")
    parser.add_argument("--cache", default=CACHE_DIR,
                        help="weather cache (default: data/raw/weather)")
    parser.add_argument("--out", default=OUT_PATH,
                        help="output CSV (default: data/processed/flood_data.csv)")
    args = parser.parse_args()

    with open(args.cells, "r", encoding="utf-8", newline="") as file:
        cells = list(csv.DictReader(file))

    # Clusters are decided once over every pass, so a storm keeps one number
    clusters, n_clusters = cluster_by_pass({row["event_id"] for row in cells})

    rows = []
    weather = {}  # cell center -> parsed weather, so each cell is read once

    for cell in cells:
        lon_min = float(cell["lon_min"])
        lat_min = float(cell["lat_min"])
        lon = round(lon_min + CELL / 2, 2)
        lat = round(lat_min + CELL / 2, 2)
        key = (lat, lon)

        if key not in weather:
            weather[key] = load_cell(lat, lon, args.cache)

        day = pass_date(cell["event_id"])
        day_before = (day - datetime.timedelta(days=1)).strftime("%Y%m%d")
        w = weather_row(weather[key], day_before)

        # month as a point on a circle, so December sits next to January
        angle = 2 * math.pi * day.month / 12
        split = SPLIT_YEARS[day.year]

        rows.append({
            "event_id": cell["event_id"],
            "date": day.isoformat(),
            "year": day.year,
            "cluster_id": clusters[cell["event_id"]],
            "split": split,
            "lon_min": f"{lon_min:.1f}",
            "lat_min": f"{lat_min:.1f}",
            "lon": f"{lon:.2f}",
            "lat": f"{lat:.2f}",
            "elevation": fmt(w["elevation"], 4),
            "n_polygons": int(cell["n_polygons"]),
            "flood_km2": f"{float(cell['flood_km2']):.4f}",
            "flood_share": f"{float(cell['flood_share']):.4f}",
            "flooded": 1 if float(cell["flood_share"]) > THRESHOLD else 0,
            "rain_1d": fmt(w["rain_1d"], 2),
            "rain_3d": fmt(w["rain_3d"], 2),
            "rain_7d": fmt(w["rain_7d"], 2),
            "rain_14d": fmt(w["rain_14d"], 2),
            "rain_21d": fmt(w["rain_21d"], 2),
            "rain_30d": fmt(w["rain_30d"], 2),
            "humidity": fmt(w["humidity"], 2),
            "temperature": fmt(w["temperature"], 2),
            "soil_wetness": fmt(w["soil_wetness"], 4),
            "month_sin": f"{math.sin(angle):.6f}",
            "month_cos": f"{math.cos(angle):.6f}",
        })

    # One file for every method, column names are the contract
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Report the split sizes and the cluster spread for a quick check
    counts = {name: 0 for name in ("train", "val", "test")}
    flooded = {name: 0 for name in counts}

    for row in rows:
        counts[row["split"]] += 1
        flooded[row["split"]] += row["flooded"]

    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"{n_clusters} storm clusters")

    for name in ("train", "val", "test"):
        share = 100 * flooded[name] / counts[name] if counts[name] else 0
        print(f"  {name}: {counts[name]} rows, {share:.1f} percent flooded")

    return 0


if __name__ == "__main__":
    sys.exit(main())
