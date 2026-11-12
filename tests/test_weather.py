# Checks for the NASA POWER fetch module, the parts that do not need the API.
# Standard library unittest only, no extra dependency.
#
# Usage (from the repository folder):
#   python -m unittest discover -s tests -t .

import os
import tempfile
import unittest

from src.weather import cache_path, cell_centers, load_cell, parse_cell


# One raw response with two days, one of them blank for soil wetness
SAMPLE = {
    "geometry": {"coordinates": [117.25, 7.75, 12.5]},
    "header": {"fill_value": -999.0},
    "properties": {"parameter": {
        "PRECTOTCORR": {"20220701": 1.0, "20220702": 2.0},
        "GWETROOT": {"20220701": -999.0, "20220702": 0.5},
        "RH2M": {"20220701": 80.0, "20220702": 81.0},
        "T2M": {"20220701": 28.0, "20220702": 29.0},
    }},
}

# Two identical corners collapse to one center, the offset corner stays
CELLS_CSV = (
    "event_id,lon_min,lat_min,n_polygons,flood_km2,flood_share\n"
    "20220805_0600,122.5,7.0,290,4.4835,0.1469\n"
    "20220805_0600,122.5,7.0,10,0.2,0.01\n"
    "20220901_0600,116.5,4.5,5,0.1,0.005\n"
)


class ParseCellTest(unittest.TestCase):
    # The elevation comes from the body, not the header
    def test_elevation_from_body(self):
        elevation, _, _ = parse_cell(SAMPLE)
        self.assertEqual(elevation, 12.5)

    # A filled day turns into None and a real day keeps its value
    def test_fill_becomes_none(self):
        _, dates, values = parse_cell(SAMPLE)
        self.assertEqual(dates, ["20220701", "20220702"])
        self.assertIsNone(values["GWETROOT"][0])
        self.assertEqual(values["GWETROOT"][1], 0.5)

    # A fully reported parameter keeps every value untouched
    def test_other_parameters_untouched(self):
        _, _, values = parse_cell(SAMPLE)
        self.assertEqual(values["PRECTOTCORR"], [1.0, 2.0])
        self.assertEqual(values["T2M"], [28.0, 29.0])


class CellCentersTest(unittest.TestCase):
    # Duplicate corners collapse and the center sits half a cell from the corner
    def test_center_and_dedup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "cells.csv")
            with open(path, "w", encoding="utf-8") as file:
                file.write(CELLS_CSV)

            self.assertEqual(
                cell_centers(path),
                [(4.75, 116.75), (7.25, 122.75)],
            )


class CachePathTest(unittest.TestCase):
    # The file is named after the cell center
    def test_name(self):
        self.assertEqual(cache_path(7.75, 117.25, "/x/weather"),
                         os.path.join("/x/weather", "7.75_117.25.json"))


class LoadCellTest(unittest.TestCase):
    # A cell with no cache file raises, so the build can report the gap
    def test_missing_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_cell(7.75, 117.25, folder)


if __name__ == "__main__":
    unittest.main()
