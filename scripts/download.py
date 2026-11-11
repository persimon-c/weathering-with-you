# Download the PhilSA flood shapefiles listed in data/acquisitions.csv.
# HDX blocks bursts of requests, so files go one at a time with a pause,
# and each file is kept only when it arrived complete.
#
# Usage:
#   python scripts/download.py
#   python scripts/download.py --dry-run

import argparse
import csv
import os
import sys
import time
import urllib.error
import urllib.request


# Paths relative to the repository root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACQUISITIONS = os.path.join(ROOT, "data", "acquisitions.csv")
RAW_DIR = os.path.join(ROOT, "data", "raw")

# Download settings
USER_AGENT = "weathering-with-you/1.0 (CMSC 170 class project)"
DELAY = 15                      # seconds between requests
CHUNK = 1024 * 1024             # read 1 MB at a time
ZIP_MAGIC = b"PK\x03\x04"       # first bytes of every zip file


# Rows of acquisitions.csv for one sensor
def load_rows(path, sensor):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Acquisitions list not found: {path}")

    with open(path, "r", encoding="utf-8", newline="") as file:
        rows = [
            row for row in csv.DictReader(file)
            if row["sensor"] == sensor
        ]

    if not rows:
        raise ValueError(f"No rows with sensor '{sensor}' in {path}")

    return rows


# True when the file exists, has the listed size, and is a zip
def is_complete(path, expected_size):
    if not os.path.isfile(path):
        return False

    if os.path.getsize(path) != expected_size:
        return False

    with open(path, "rb") as file:
        return file.read(4) == ZIP_MAGIC


# Stream one URL into part_path
def fetch(url, part_path):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT}
    )

    # urllib follows HDX's redirect to the storage link
    with urllib.request.urlopen(request, timeout=120) as response:
        with open(part_path, "wb") as file:
            while True:
                chunk = response.read(CHUNK)

                if not chunk:
                    break

                file.write(chunk)


# Download one file, True on success
def download(row, out_dir):
    path = os.path.join(out_dir, row["file"])
    part_path = path + ".part"
    expected_size = int(row["size"])

    try:
        fetch(row["url"], part_path)

    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        print(f"    network error: {error}")
        return False

    # Keep the file only if it arrived whole
    if not is_complete(part_path, expected_size):
        size = os.path.getsize(part_path)
        os.remove(part_path)
        print(f"    got {size} bytes, expected {expected_size}")
        return False

    os.replace(part_path, path)
    return True


def main():
    # Command-line options
    parser = argparse.ArgumentParser(
        description="Download PhilSA flood shapefiles from HDX."
    )
    parser.add_argument("--csv", default=ACQUISITIONS,
                        help="acquisitions list (default: data/acquisitions.csv)")
    parser.add_argument("--out", default=RAW_DIR,
                        help="output folder (default: data/raw)")
    parser.add_argument("--sensor", default="s1",
                        help="sensor code to download (default: s1, Sentinel-1)")
    parser.add_argument("--delay", type=float, default=DELAY,
                        help="seconds between requests (default: 15)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be downloaded and stop")
    args = parser.parse_args()

    rows = load_rows(args.csv, args.sensor)

    # Skip files a previous run already finished
    todo = [
        row for row in rows
        if not is_complete(
            os.path.join(args.out, row["file"]),
            int(row["size"])
        )
    ]

    # Summary of what is left
    total_mb = sum(int(row["size"]) for row in todo) / 1e6
    print(
        f"{len(rows)} '{args.sensor}' files listed, "
        f"{len(rows) - len(todo)} already complete, "
        f"{len(todo)} to download ({total_mb:.0f} MB)"
    )

    if args.dry_run:
        for row in todo:
            print(f"  {row['file']}")
        return 0

    os.makedirs(args.out, exist_ok=True)
    failed = []

    # Download the rest, pausing between files
    for number, row in enumerate(todo, start=1):
        size_mb = int(row["size"]) / 1e6
        print(f"[{number}/{len(todo)}] {row['file']} ({size_mb:.1f} MB)")

        if not download(row, args.out):
            failed.append(row["file"])

        if number < len(todo):
            time.sleep(args.delay)

    # Report failures so a rerun can pick them up
    print(f"Done: {len(todo) - len(failed)} downloaded, {len(failed)} failed")

    for name in failed:
        print(f"  failed: {name}")

    if failed:
        print("Run the script again later to retry the failed files.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
