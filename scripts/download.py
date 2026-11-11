"""
Download the PhilSA flood shapefiles listed in data/acquisitions.csv.

HDX blocks clients that send requests in bursts. When it does, it answers
with HTTP 202 and an empty body instead of the file. To stay polite and
still finish, this script:

- downloads one file at a time and waits between requests
- waits a few minutes and retries when HDX answers 202
- skips files that are already downloaded with the right size
- writes to a .part file first, so a stopped run never leaves a
  half-written zip that looks complete

Usage (from anywhere in the repository):
    python scripts/download.py
    python scripts/download.py --dry-run
"""

import argparse
import csv
import os
import sys
import time
import urllib.error
import urllib.request


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACQUISITIONS = os.path.join(ROOT, "data", "acquisitions.csv")
RAW_DIR = os.path.join(ROOT, "data", "raw")

USER_AGENT = "weathering-with-you/1.0 (CMSC 170 class project)"
DELAY = 15                      # seconds between HDX requests
RETRY_WAITS = [120, 300, 600]   # seconds to wait before each retry
RETRY_HTTP = (429, 500, 502, 503, 504)
CHUNK = 1024 * 1024             # read the response 1 MB at a time
ZIP_MAGIC = b"PK\x03\x04"       # every zip file starts with these bytes


class SlowDown(Exception):
    """HDX answered 202, its signal that we are sending requests too fast."""


def load_rows(path, sensor):
    """
    Read data/acquisitions.csv and keep the rows for one sensor.

    Returns:
        A list of rows, each a dict with at least file, size and url.
    """

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


def is_complete(path, expected_size):
    """Check that a file exists, has the listed size, and starts like a zip."""

    if not os.path.isfile(path):
        return False

    if os.path.getsize(path) != expected_size:
        return False

    with open(path, "rb") as file:
        return file.read(4) == ZIP_MAGIC


def fetch(url, part_path):
    """
    Stream one URL into part_path.

    HDX first redirects to a temporary storage link, which urllib follows
    on its own. Raises SlowDown when HDX answers 202 instead of the file.
    """

    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT}
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status == 202:
            raise SlowDown()

        with open(part_path, "wb") as file:
            while True:
                chunk = response.read(CHUNK)

                if not chunk:
                    break

                file.write(chunk)


def download(row, out_dir):
    """
    Download one shapefile zip, retrying after waits if needed.

    Returns:
        True if the file is now complete, False if every attempt failed.
    """

    path = os.path.join(out_dir, row["file"])
    part_path = path + ".part"
    expected_size = int(row["size"])

    # The first attempt goes right away, each retry waits longer
    for wait in [0] + RETRY_WAITS:
        if wait:
            print(f"    waiting {wait // 60} min before retrying")
            time.sleep(wait)

        try:
            fetch(row["url"], part_path)

        except SlowDown:
            print("    HDX asked us to slow down (HTTP 202)")
            continue

        except urllib.error.HTTPError as error:
            # Missing or forbidden files will not appear on a retry
            if error.code not in RETRY_HTTP:
                print(f"    HTTP {error.code}, giving up on this file")
                return False

            print(f"    HTTP {error.code}")
            continue

        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            print(f"    network error: {error}")
            continue

        # Only keep the file if it arrived whole
        if not is_complete(part_path, expected_size):
            size = os.path.getsize(part_path)
            os.remove(part_path)

            if size != expected_size:
                print(f"    got {size} bytes, expected {expected_size}")
            else:
                print("    the file is not a zip")

            continue

        os.replace(part_path, path)
        return True

    return False


def main():
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

    # Skip files that a previous run already finished
    todo = [
        row for row in rows
        if not is_complete(
            os.path.join(args.out, row["file"]),
            int(row["size"])
        )
    ]

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

    for number, row in enumerate(todo, start=1):
        size_mb = int(row["size"]) / 1e6
        print(f"[{number}/{len(todo)}] {row['file']} ({size_mb:.1f} MB)")

        if not download(row, args.out):
            failed.append(row["file"])

        # Pause between requests so HDX does not block us
        if number < len(todo):
            time.sleep(args.delay)

    print(f"Done: {len(todo) - len(failed)} downloaded, {len(failed)} failed")

    for name in failed:
        print(f"  failed: {name}")

    if failed:
        print("Run the script again later to retry the failed files.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
