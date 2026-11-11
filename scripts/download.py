# Download the PhilSA flood shapefiles listed in data/acquisitions.csv.
#
# Usage:
#   python scripts/download.py --dry-run

import argparse
import csv
import os
import sys


# Paths relative to the repository root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACQUISITIONS = os.path.join(ROOT, "data", "acquisitions.csv")
RAW_DIR = os.path.join(ROOT, "data", "raw")

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

    for row in todo:
        print(f"  {row['file']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
