# Flooded area per satellite pass and 0.5 degree cell, from the PhilSA shapefiles.
#
# Usage (from the repository folder):
#   python -m src.cells

import argparse
import csv
import math
import os
import re
import sys

import numpy as np

from src.shapefile import read_dbf, read_shp, read_zip


# Paths relative to the repository root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACQUISITIONS = os.path.join(ROOT, "data", "acquisitions.csv")
RAW_DIR = os.path.join(ROOT, "data", "raw")
OUT_PATH = os.path.join(ROOT, "data", "interim", "flood_cells.csv")

CELL = 0.5                  # cell size in degrees
KM_PER_DEG_LAT = 110.574    # km per degree of latitude
KM_PER_DEG_LON = 111.320    # km per degree of longitude at the equator


# Area of one polygon in square km, holes subtracted
def polygon_area_km2(parts, points):
    # Turn degrees into km around the polygon's mean latitude
    lat0 = math.radians(points[:, 1].mean())
    x = points[:, 0] * KM_PER_DEG_LON * math.cos(lat0)
    y = points[:, 1] * KM_PER_DEG_LAT

    # Shoelace formula per ring; outer rings and holes run in opposite
    # directions, so their signed areas subtract
    bounds = list(parts) + [len(points)]
    total = 0.0

    for start, end in zip(bounds[:-1], bounds[1:]):
        xs, ys = x[start:end], y[start:end]
        total += 0.5 * (np.dot(xs[:-1], ys[1:]) - np.dot(xs[1:], ys[:-1]))

    # Some polygons are wound the other way, so keep the size only
    return abs(total)


# South-west corner of the cell holding the polygon's bounding-box center
def cell_of(points):
    lon = (points[:, 0].min() + points[:, 0].max()) / 2
    lat = (points[:, 1].min() + points[:, 1].max()) / 2
    return math.floor(lon / CELL) * CELL, math.floor(lat / CELL) * CELL


# Area of a whole cell in square km; land area is unknown, so sea counts too
def cell_area_km2(lat_min):
    lat = math.radians(lat_min + CELL / 2)
    return (CELL * KM_PER_DEG_LAT) * (CELL * KM_PER_DEG_LON * math.cos(lat))


# Pass ID from a file name, e.g. 20260907_1807_fld_s1_shp.zip -> 20260907_1807
def event_id(filename):
    match = re.match(r"(\d{8}_\d{4})_", filename)

    if not match:
        raise ValueError(f"No date and pass time in file name: {filename}")

    return match.group(1)


# Add one zip's polygons to the running (count, area) totals per (pass, cell)
def add_file(path, totals):
    shp_data, dbf_data = read_zip(path)
    polygons = read_shp(shp_data)
    _, rows = read_dbf(dbf_data)

    # Every polygon must have a matching attribute row
    if len(rows) != len(polygons):
        raise ValueError(
            f"{path}: {len(polygons)} shapes but {len(rows)} attribute rows"
        )

    # Regional pieces of one pass share an ID, so they add to the same cells
    event = event_id(os.path.basename(path))

    # Files before 2025-11-11 have no Class field, so every polygon counts as flooded
    for polygon in polygons:
        if polygon is None:
            continue

        parts, points = polygon
        key = (event,) + cell_of(points)
        count, area = totals.get(key, (0, 0.0))
        totals[key] = (count + 1, area + polygon_area_km2(parts, points))

    return len(polygons)


# File names of one sensor from acquisitions.csv
def listed_files(path, sensor):
    with open(path, "r", encoding="utf-8", newline="") as file:
        return [
            row["file"] for row in csv.DictReader(file)
            if row["sensor"] == sensor
        ]


def main():
    # Command-line options
    parser = argparse.ArgumentParser(
        description="Flooded area per pass and 0.5 degree cell."
    )
    parser.add_argument("--raw", default=RAW_DIR,
                        help="folder with the downloaded zips (default: data/raw)")
    parser.add_argument("--out", default=OUT_PATH,
                        help="output CSV (default: data/interim/flood_cells.csv)")
    parser.add_argument("--sensor", default="s1",
                        help="sensor code to read (default: s1, Sentinel-1)")
    args = parser.parse_args()

    # Every listed file must be downloaded first
    files = listed_files(ACQUISITIONS, args.sensor)
    missing = [name for name in files
               if not os.path.isfile(os.path.join(args.raw, name))]

    if missing:
        print(f"{len(missing)} files missing in {args.raw}, "
              "run python scripts/download.py first")
        return 1

    # Sum polygons file by file
    totals = {}

    for number, name in enumerate(files, start=1):
        n = add_file(os.path.join(args.raw, name), totals)
        print(f"[{number}/{len(files)}] {name}: {n} polygons")

    # One row per pass and cell, sorted
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["event_id", "lon_min", "lat_min",
                         "n_polygons", "flood_km2", "flood_share"])

        for (event, lon, lat), (count, area) in sorted(totals.items()):
            share = 100 * area / cell_area_km2(lat)
            writer.writerow([event, f"{lon:.1f}", f"{lat:.1f}",
                             count, f"{area:.4f}", f"{share:.4f}"])

    events = {key[0] for key in totals}
    print(f"Wrote {len(totals)} rows for {len(events)} passes to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
