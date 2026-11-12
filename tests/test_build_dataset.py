# Checks for the processed dataset builder, mainly the rain windows.
# Standard library unittest only, no extra dependency.
#
# Usage (from the repository folder):
#   python -m unittest discover -s tests -t .

import datetime
import unittest

from scripts.build_dataset import weather_row


# A 40 day cell with rain equal to the day index, so each sum is predictable
def make_cell():
    start = datetime.date(2026, 1, 1)
    dates = [(start + datetime.timedelta(days=day)).strftime("%Y%m%d")
             for day in range(40)]
    rain = [day for day in range(40)]
    values = {"PRECTOTCORR": rain, "RH2M": rain, "T2M": rain, "GWETROOT": rain}
    return dates, (12.5, dates, values)


class WeatherRowTest(unittest.TestCase):
    # A pass on day 35 reaches back to day 34, never the pass day itself
    def test_windows_end_the_day_before(self):
        dates, cell = make_cell()

        row = weather_row(cell, dates[34])

        self.assertEqual(row["rain_1d"], 34)
        self.assertEqual(row["rain_3d"], 32 + 33 + 34)
        self.assertEqual(row["rain_7d"], sum(range(28, 35)))
        self.assertEqual(row["rain_14d"], sum(range(21, 35)))
        self.assertEqual(row["rain_21d"], sum(range(14, 35)))
        self.assertEqual(row["rain_30d"], sum(range(5, 35)))
        self.assertEqual(row["humidity"], 34)
        self.assertEqual(row["temperature"], 34)
        self.assertEqual(row["soil_wetness"], 34)

    # A missing rain day is skipped instead of breaking the sum
    def test_missing_rain_is_skipped(self):
        dates, cell = make_cell()
        cell[2]["PRECTOTCORR"][30] = None

        row = weather_row(cell, dates[34])

        self.assertEqual(row["rain_7d"], 28 + 29 + 31 + 32 + 33 + 34)


if __name__ == "__main__":
    unittest.main()
