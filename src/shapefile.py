# Read PhilSA flood shapefiles (.shp and .dbf) straight from their zip files.
# Uses struct and NumPy only.

import io
import struct
import zipfile

import numpy as np


FILE_CODE = 9994    # first four bytes of every .shp file
POLYGON = 5         # shape type for polygons
NULL_SHAPE = 0      # empty record


# Bytes of the .shp and .dbf inside a zip, looking inside a nested zip too
def read_zip(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        shp = [name for name in names if name.lower().endswith(".shp")]
        dbf = [name for name in names if name.lower().endswith(".dbf")]

        # Some 2026 files wrap the shapefile in a second zip
        if not shp:
            inner = [name for name in names if name.lower().endswith(".zip")]

            if len(inner) != 1:
                raise ValueError(f"No .shp found in {path}")

            return read_zip(io.BytesIO(archive.read(inner[0])))

        if len(shp) != 1 or len(dbf) != 1:
            raise ValueError(
                f"Expected one .shp and one .dbf in {path}, "
                f"found {shp} and {dbf}"
            )

        return archive.read(shp[0]), archive.read(dbf[0])


# Polygons of a .shp file as (parts, points), or None for an empty record
# parts: index where each ring starts, points: array of (lon, lat) rows
def read_shp(data):
    # 100-byte header: file code and length are big-endian, the rest little-endian
    file_code, = struct.unpack(">i", data[0:4])

    if file_code != FILE_CODE:
        raise ValueError("Not a shapefile")

    file_length = struct.unpack(">i", data[24:28])[0] * 2   # stored in 16-bit words
    shape_type, = struct.unpack("<i", data[32:36])

    if shape_type != POLYGON:
        raise ValueError(f"Shape type {shape_type}, expected {POLYGON} (polygon)")

    polygons = []
    offset = 100

    while offset < file_length:
        # Record header: record number and content length, big-endian
        _, content_length = struct.unpack(">ii", data[offset:offset + 8])
        start = offset + 8
        offset = start + content_length * 2

        record_type, = struct.unpack("<i", data[start:start + 4])

        # Keep empty records as None so rows still line up with the .dbf
        if record_type == NULL_SHAPE:
            polygons.append(None)
            continue

        # Skip the 32-byte bounding box, then read the part and point counts
        n_parts, n_points = struct.unpack("<ii", data[start + 36:start + 44])
        parts = np.frombuffer(data, "<i4", n_parts, start + 44)
        points = np.frombuffer(
            data, "<f8", 2 * n_points, start + 44 + 4 * n_parts
        ).reshape(n_points, 2)

        polygons.append((parts, points))

    return polygons


# Attribute table of a .dbf file as (field names, one dict per record)
def read_dbf(data):
    # Header: record count, header length and record length
    n_records, header_length, record_length = struct.unpack("<IHH", data[4:12])

    # 32-byte field descriptors, ending at the 0x0D terminator
    fields = []
    offset = 32

    while data[offset] != 0x0D:
        name = data[offset:offset + 11].split(b"\x00")[0].decode("latin-1")
        length = data[offset + 16]
        fields.append((name, length))
        offset += 32

    # Fixed-width records; the first byte is the deletion flag, so values start at 1
    rows = []

    for index in range(n_records):
        position = header_length + index * record_length + 1
        row = {}

        for name, length in fields:
            row[name] = data[position:position + length].decode("latin-1").strip()
            position += length

        rows.append(row)

    return [name for name, _ in fields], rows
