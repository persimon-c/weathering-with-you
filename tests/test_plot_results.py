import os
import tempfile
import unittest
from pathlib import Path

from scripts.plot_results import (
    plot_roc_curves,
    plot_metric_bars,
    plot_feature_groups,
    plot_per_cluster,
    plot_validation_map,
)


class TestPlotResults(unittest.TestCase):

    def test_plot_roc_curves(self):
        scores = [
            {"method": "majority", "actual": "0", "score": "0.1"},
            {"method": "majority", "actual": "1", "score": "0.1"},
            {"method": "place_history", "actual": "0", "score": "0.1"},
            {"method": "place_history", "actual": "1", "score": "0.9"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "roc.png"
            plot_roc_curves(scores, out)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)

    def test_plot_metric_bars(self):
        comparison = [
            {"method": "majority", "balanced_accuracy": "0.5", "macro_f1": "0.45"},
            {"method": "place_history", "balanced_accuracy": "0.78", "macro_f1": "0.79"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "bars.png"
            plot_metric_bars(comparison, out)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)

    def test_plot_feature_groups(self):
        fg = [
            {"method": "naive_bayes", "group": "place", "balanced_accuracy": "0.51"},
            {"method": "naive_bayes", "group": "place_weather", "balanced_accuracy": "0.53"},
            {"method": "naive_bayes", "group": "place_weather_season", "balanced_accuracy": "0.54"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "fg.png"
            plot_feature_groups(fg, out)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)

    def test_plot_per_cluster(self):
        clusters = [
            {"method": "majority", "cluster_id": "14", "balanced_accuracy": "0.5"},
            {"method": "place_history", "cluster_id": "14", "balanced_accuracy": "0.85"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "cluster.png"
            plot_per_cluster(clusters, out)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)

    def test_plot_validation_map(self):
        scores = [
            {"method": "place_history", "actual": "1", "predicted": "1", "lat": "10.0", "lon": "120.0"},
            {"method": "place_history", "actual": "0", "predicted": "1", "lat": "11.0", "lon": "121.0"},
            {"method": "place_history", "actual": "1", "predicted": "0", "lat": "12.0", "lon": "122.0"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "map.png"
            plot_validation_map(scores, out, method="place_history")
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
