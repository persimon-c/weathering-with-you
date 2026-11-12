# Checks for the flooded area per pass and cell, on a tiny shapefile built in memory.
# Standard library unittest only, no extra dependency.
#
# Usage (from the repository folder):
#   python -m unittest discover -s tests -t .

import math
import os
import tempfile
import unittest

import numpy as np

from src.cells import add_file, cell_area_km2, cell_of, event_id, polygon_area_km2
from tests.test_shapefile import DBF, HOLE, SHP, SQUARE, make_dbf, make_zip

# The 0.01 degree square near 7.245 N, in square km
SQUARE_KM2 = (0.01 * 110.574) * (0.01 * 111.320 * math.cos(math.radians(7.245)))


# Ring starts and points of one polygon, as read_shp returns them
def polygon(*rings):
    parts, start = [], 0

    for ring in rings:
        parts.append(start)
        start += len(ring)

    return np.array(parts), np.array([point for ring in rings for point in ring])


class PolygonAreaTest(unittest.TestCase):
    # A plain square matches its size in km
    def test_square(self):
        self.assertAlmostEqual(polygon_area_km2(*polygon(SQUARE)), SQUARE_KM2, places=4)

    # The hole is a quarter of the square, so three quarters are left
    def test_hole_subtracted(self):
        area = polygon_area_km2(*polygon(SQUARE, HOLE))
        self.assertAlmostEqual(area, 0.75 * SQUARE_KM2, places=4)

    # A square wound the other way has the same size
    def test_reversed_winding(self):
        area = polygon_area_km2(*polygon(SQUARE[::-1]))
        self.assertAlmostEqual(area, SQUARE_KM2, places=4)


class CellTest(unittest.TestCase):
    # The polygon goes to the cell holding its bounding-box center
    def test_cell_of(self):
        _, points = polygon(SQUARE)
        self.assertEqual(cell_of(points), (122.5, 7.0))

    # A 0.5 degree cell near 7 N is about 3,000 square km
    def test_cell_area(self):
        self.assertAlmostEqual(cell_area_km2(7.0), 3052.7, places=1)

    # The pass ID is the date and pass time at the start of the name
    def test_event_id(self):
        self.assertEqual(event_id("20260907_1807_fld_s1_shp.zip"), "20260907_1807")

        with self.assertRaises(ValueError):
            event_id("flood_s1_shp.zip")


class AddFileTest(unittest.TestCase):
    # One zip adds its polygon to the right pass and cell, skipping the empty record
    def test_totals(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "20260907_1807_fld_s1_shp.zip")

            with open(path, "wb") as file:
                file.write(make_zip({"flood.shp": SHP, "flood.dbf": DBF}))

            totals = {}
            self.assertEqual(add_file(path, totals), 2)

        self.assertEqual(list(totals), [("20260907_1807", 122.5, 7.0)])
        count, area = totals[("20260907_1807", 122.5, 7.0)]
        self.assertEqual(count, 1)
        self.assertAlmostEqual(area, 0.75 * SQUARE_KM2, places=4)

    # Shapes and attribute rows must line up
    def test_row_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "20260907_1807_fld_s1_shp.zip")
            dbf = make_dbf([("Class", 10)], [["Flood"]])

            with open(path, "wb") as file:
                file.write(make_zip({"flood.shp": SHP, "flood.dbf": dbf}))

            with self.assertRaises(ValueError):
                add_file(path, {})


if __name__ == "__main__":
    unittest.main()
