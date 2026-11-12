# Check the NASA POWER cache against the cell list and record a manifest.
# The raw cache lives in data/raw/ and is not committed, so the manifest is the
# committed proof that every cell was fetched and holds the full date range.
#
# Usage (from the repository folder):
#   python scripts/check_weather.py
#   python scripts/check_weather.py --write-manifest

import argparse
import datetime
import hashlib
import json
import os
import sys

# The verifier reuses the fetch module, which lives one folder up
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.weather import (CELLS_PATH, CACHE_DIR, PARAMETERS, START, END,
                         API, cache_path, cell_centers)


MANIFEST_PATH = os.path.join(ROOT, "data", "weather_manifest.json")


# Number of days the range covers, both ends included
def expected_days(start, end):
    first = datetime.date(int(start[:4]), int(start[4:6]), int(start[6:]))
    last = datetime.date(int(end[:4]), int(end[4:6]), int(end[6:]))
    return (last - first).days + 1


# sha256 of a file, to tell whether a rerun produced the same bytes
def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 16), b""):
            digest.update(block)

    return digest.hexdigest()


# Check one cell and return its manifest entry, or raise with the reason
def check_cell(lat, lon, cache_dir, days):
    path = cache_path(lat, lon, cache_dir)

    if not os.path.isfile(path):
        raise ValueError("no cached file")

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # the height sits in the body, a filled height means the cell has none
    elevation = data["geometry"]["coordinates"][2]
    fill = data["header"]["fill_value"]

    if elevation is None or elevation == fill:
        raise ValueError(f"elevation is missing ({elevation})")

    series = data["properties"]["parameter"]
    dates = sorted(series[PARAMETERS[0]])

    # every parameter must report the same days, or the join in the build breaks
    if any(set(series[name]) != set(dates) for name in PARAMETERS):
        raise ValueError("parameters report different days")

    if len(dates) != days or dates[0] != START or dates[-1] != END:
        raise ValueError(f"range is {dates[0]} to {dates[-1]} over {len(dates)} days")

    missing = {name: sum(1 for v in series[name].values() if v == fill)
               for name in PARAMETERS}

    # only soil wetness is expected to be blank, and only at an ocean cell
    for name in ("PRECTOTCORR", "RH2M", "T2M"):
        if missing[name]:
            raise ValueError(f"{name} has {missing[name]} missing days")

    return {
        "elevation": elevation,
        "days": len(dates),
        "missing": missing,
        "sha256": sha256(path),
    }


def main():
    # Command-line options
    parser = argparse.ArgumentParser(
        description="Check the NASA POWER cache for all cell centers."
    )
    parser.add_argument("--cells", default=CELLS_PATH,
                        help="cell list (default: data/interim/flood_cells.csv)")
    parser.add_argument("--cache", default=CACHE_DIR,
                        help="cache folder (default: data/raw/weather)")
    parser.add_argument("--write-manifest", action="store_true",
                        help=f"save the manifest to {MANIFEST_PATH}")
    args = parser.parse_args()

    days = expected_days(START, END)
    centers = cell_centers(args.cells)
    entries = {}
    problems = []

    for lat, lon in centers:
        try:
            entries[f"{lat:.2f},{lon:.2f}"] = check_cell(
                lat, lon, args.cache, days
            )

        except ValueError as error:
            problems.append(f"{lat:.2f},{lon:.2f}: {error}")

    # An ocean cell is one where soil wetness is blank on every day
    ocean = sum(1 for e in entries.values() if e["missing"]["GWETROOT"] == days)

    print(f"{len(centers)} cells, {len(entries)} cached and valid, "
          f"{len(problems)} with a problem")
    print(f"range {START} to {END}, {days} days, {ocean} ocean cells")

    for line in problems:
        print(f"  {line}")

    if args.write_manifest and not problems:
        manifest = {
            "generated": datetime.date.today().isoformat(),
            "api": API,
            "parameters": PARAMETERS,
            "start": START,
            "end": END,
            "days": days,
            "cells": len(centers),
            "ocean_cells": ocean,
            "entries": entries,
        }

        with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")

        print(f"Wrote {MANIFEST_PATH}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
