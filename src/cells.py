# Flooded area per satellite pass and 0.5 degree cell, from the PhilSA shapefiles.

import math
import re

import numpy as np


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

