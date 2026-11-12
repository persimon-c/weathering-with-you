# NASA POWER daily weather per cell center, cached so each cell is fetched once.
# One request per cell covers the whole date range and returns the MERRA-2
# elevation with the daily rain, soil wetness, humidity and temperature.
#
# Usage (from the repository folder):
#   python -m src.weather
#   python -m src.weather --limit 5

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


# Paths relative to the repository root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELLS_PATH = os.path.join(ROOT, "data", "interim", "flood_cells.csv")
CACHE_DIR = os.path.join(ROOT, "data", "raw", "weather")

# Request settings
API = "https://power.larc.nasa.gov/api/temporal/daily/point"
USER_AGENT = "weathering-with-you/1.0 (CMSC 170 class project)"
PARAMETERS = ["PRECTOTCORR", "GWETROOT", "RH2M", "T2M"]  # rain, soil wetness, humidity, temperature
START = "20220701"          # the day before the first pass, leaves room for a 30 day rain window
END = "20260930"            # the day after the last pass
DELAY = 1.0                 # seconds between requests, to stay polite
RETRY_WAITS = [30, 120, 300]  # seconds before each retry
RETRY_HTTP = (429, 500, 502, 503, 504)
CELL = 0.5                  # cell size in degrees, same grid as src/cells.py


# Unique cell centers (latitude, longitude) from the cell list
def cell_centers(path):
    centers = set()

    with open(path, "r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            # a center sits half a cell from the corner kept in the file
            lat = float(row["lat_min"]) + CELL / 2
            lon = float(row["lon_min"]) + CELL / 2
            centers.add((round(lat, 2), round(lon, 2)))

    return sorted(centers)


# Cache file of one cell, named after its center
def cache_path(lat, lon, cache_dir):
    return os.path.join(cache_dir, f"{lat:.2f}_{lon:.2f}.json")


# Raw daily response for one cell center from the NASA POWER point API
def fetch_cell(lat, lon, start, end):
    query = urllib.parse.urlencode([
        ("parameters", ",".join(PARAMETERS)),
        ("community", "AG"),
        ("longitude", lon),
        ("latitude", lat),
        ("start", start),
        ("end", end),
        ("format", "JSON"),
    ])
    request = urllib.request.Request(
        API + "?" + query,
        headers={"User-Agent": USER_AGENT},
    )

    # the whole range for one cell is a small JSON document, so read it at once
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


# Download one cell with retries, True on success
def fetch_to_cache(lat, lon, start, end, cache_dir):
    path = cache_path(lat, lon, cache_dir)

    # First try at once, then wait longer before each retry
    for wait in [0] + RETRY_WAITS:
        if wait:
            print(f"    waiting {wait}s before retrying")
            time.sleep(wait)

        try:
            data = fetch_cell(lat, lon, start, end)

        except urllib.error.HTTPError as error:
            # A bad request or a bad cell will not fix itself on a retry
            if error.code not in RETRY_HTTP:
                print(f"    HTTP {error.code}, giving up on this cell")
                return False

            print(f"    HTTP {error.code}")
            continue

        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as error:
            print(f"    network error: {error}")
            continue

        # Keep the raw response, this is what "fetched once" means. Write to a
        # part file and rename it, so an interrupted run never leaves a half
        # file that the next run would skip
        part = path + ".part"

        with open(part, "w", encoding="utf-8") as file:
            json.dump(data, file)

        os.replace(part, path)

        return True

    return False


# Elevation and the daily series from one raw response
def parse_cell(data):
    # the body holds the MERRA-2 height, ordered longitude, latitude, meters
    elevation = data["geometry"]["coordinates"][2]
    fill = data["header"]["fill_value"]  # a filled day means the value is missing
    days = data["properties"]["parameter"]
    dates = sorted(days[PARAMETERS[0]])  # every parameter reports the same days
    values = {}

    for name in PARAMETERS:
        series = days[name]
        # an ocean cell has no root-zone value, so keep missing days as None
        values[name] = [None if series[day] == fill else series[day] for day in dates]

    return elevation, dates, values


# Elevation and daily series of one cell from the cache, for the dataset build
def load_cell(lat, lon, cache_dir=CACHE_DIR):
    path = cache_path(lat, lon, cache_dir)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"No cached weather for cell {lat:.2f}, {lon:.2f}")

    with open(path, "r", encoding="utf-8") as file:
        return parse_cell(json.load(file))


def main():
    # Command-line options
    parser = argparse.ArgumentParser(
        description="Fetch NASA POWER daily weather per cell center and cache it."
    )
    parser.add_argument("--cells", default=CELLS_PATH,
                        help="cell list (default: data/interim/flood_cells.csv)")
    parser.add_argument("--cache", default=CACHE_DIR,
                        help="cache folder (default: data/raw/weather)")
    parser.add_argument("--start", default=START,
                        help="first day, YYYYMMDD (default: 20220701)")
    parser.add_argument("--end", default=END,
                        help="last day, YYYYMMDD (default: 20260930)")
    parser.add_argument("--delay", type=float, default=DELAY,
                        help="seconds between requests (default: 1)")
    parser.add_argument("--limit", type=int,
                        help="fetch at most this many cells, for a quick test")
    parser.add_argument("--refresh", action="store_true",
                        help="fetch again even when the cache already has the cell")
    args = parser.parse_args()

    centers = cell_centers(args.cells)

    # Skip cells a previous run already cached
    todo = [
        center for center in centers
        if args.refresh or not os.path.isfile(cache_path(*center, args.cache))
    ]

    print(f"{len(centers)} cells, {len(centers) - len(todo)} cached, "
          f"{len(todo)} to fetch")

    # A limit only trims this run, the cached count above stays true
    if args.limit:
        todo = todo[:args.limit]

    os.makedirs(args.cache, exist_ok=True)
    failed = []

    # Fetch the rest, pausing between cells
    for number, (lat, lon) in enumerate(todo, start=1):
        print(f"[{number}/{len(todo)}] {lat:.2f}, {lon:.2f}")

        if not fetch_to_cache(lat, lon, args.start, args.end, args.cache):
            failed.append((lat, lon))

        if number < len(todo):
            time.sleep(args.delay)

    # Report failures so a rerun can pick them up
    print(f"Done: {len(todo) - len(failed)} fetched, {len(failed)} failed")

    for lat, lon in failed:
        print(f"  failed: {lat:.2f}, {lon:.2f}")

    if failed:
        print("Run the script again to retry the failed cells.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
