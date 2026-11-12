# Checks for the shapefile reader, on a tiny shapefile built in memory.
# Standard library unittest only, no extra dependency.
#
# Usage (from the repository folder):
#   python -m unittest discover -s tests -t .

import io
import struct
import unittest
import zipfile

from src.shapefile import read_dbf, read_shp, read_zip


# A .shp file of polygon records; a record is a list of rings, or None for an empty one
def make_shp(records):
    body = b""

    for number, rings in enumerate(records, start=1):
        if rings is None:
            content = struct.pack("<i", 0)
        else:
            points = [point for ring in rings for point in ring]
            parts, start = [], 0

            for ring in rings:
                parts.append(start)
                start += len(ring)

            content = struct.pack("<i4d", 5, 0, 0, 0, 0)    # bounding box is not read
            content += struct.pack("<ii", len(rings), len(points))
            content += struct.pack(f"<{len(parts)}i", *parts)
            content += b"".join(struct.pack("<2d", *point) for point in points)

        # Record header is big-endian, length in 16-bit words
        body += struct.pack(">ii", number, len(content) // 2) + content

    # 100-byte file header
    header = struct.pack(">7i", 9994, 0, 0, 0, 0, 0, (100 + len(body)) // 2)
    header += struct.pack("<2i8d", 1000, 5, 0, 0, 0, 0, 0, 0, 0, 0)
    return header + body


# A .dbf file with text fields; fields are (name, length) pairs
def make_dbf(fields, rows):
    header_length = 32 + 32 * len(fields) + 1
    record_length = 1 + sum(length for _, length in fields)

    data = struct.pack("<B3BIHH20x", 3, 126, 11, 12,
                       len(rows), header_length, record_length)

    # 32-byte field descriptors, then the 0x0D terminator
    for name, length in fields:
        data += name.encode("latin-1").ljust(11, b"\x00") + b"C"
        data += struct.pack("<4xBB14x", length, 0)

    data += b"\x0D"

    # Each record starts with a blank deletion flag
    for row in rows:
        data += b" " + b"".join(
            value.encode("latin-1").ljust(length)
            for value, (_, length) in zip(row, fields)
        )

    return data + b"\x1A"


# A zip holding the given files, as bytes
def make_zip(files):
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)

    return buffer.getvalue()


# A 0.01 degree square with a 0.005 degree hole, then an empty record
SQUARE = [(122.74, 7.24), (122.74, 7.25), (122.75, 7.25), (122.75, 7.24), (122.74, 7.24)]
HOLE = [(122.7425, 7.2425), (122.7475, 7.2425), (122.7475, 7.2475), (122.7425, 7.2475), (122.7425, 7.2425)]
SHP = make_shp([[SQUARE, HOLE], None])
DBF = make_dbf([("Class", 10)], [["Flood"], [""]])


class ReadShpTest(unittest.TestCase):
    # Ring starts and points come back as written
    def test_polygon(self):
        polygons = read_shp(SHP)
        parts, points = polygons[0]

        self.assertEqual(list(parts), [0, 5])
        self.assertEqual(points.shape, (10, 2))
        self.assertEqual(tuple(points[6]), HOLE[1])

    # An empty record stays as None so rows line up with the .dbf
    def test_null_record(self):
        self.assertEqual(len(read_shp(SHP)), 2)
        self.assertIsNone(read_shp(SHP)[1])

    # Anything that is not a shapefile is refused
    def test_bad_file_code(self):
        with self.assertRaises(ValueError):
            read_shp(b"\x00" * 100)


class ReadDbfTest(unittest.TestCase):
    # Field names and trimmed values per record
    def test_rows(self):
        names, rows = read_dbf(DBF)

        self.assertEqual(names, ["Class"])
        self.assertEqual(rows, [{"Class": "Flood"}, {"Class": ""}])


class ReadZipTest(unittest.TestCase):
    # The .shp and .dbf come out of a plain zip
    def test_plain(self):
        data = make_zip({"flood.shp": SHP, "flood.dbf": DBF, "flood.prj": b""})
        self.assertEqual(read_zip(io.BytesIO(data)), (SHP, DBF))

    # Some 2026 files hold a zip inside a zip
    def test_nested(self):
        inner = make_zip({"flood.shp": SHP, "flood.dbf": DBF})
        outer = make_zip({"flood.zip": inner})
        self.assertEqual(read_zip(io.BytesIO(outer)), (SHP, DBF))

    # A zip with no shapefile is refused
    def test_no_shp(self):
        with self.assertRaises(ValueError):
            read_zip(io.BytesIO(make_zip({"readme.txt": b"hi"})))


if __name__ == "__main__":
    unittest.main()
