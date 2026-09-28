"""Regression for missing AEF coverage across UTM boundaries."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("tminus1", Path(__file__).resolve().parents[1] / "scripts/prepare_tminus1_training.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SourceSelectionTest(unittest.TestCase):
    def test_adjacent_zone_is_retained_and_native_zone_applied_last(self):
        records = {2019: [
            ("EPSG:32636", "native", (35, 40, 36, 41)),
            ("EPSG:32637", "neighbor", (36, 40, 37, 41)),
            ("EPSG:32637", "distant", (40, 40, 41, 41)),
        ], 2020: [("EPSG:32636", "wrong_year", (35, 40, 37, 41))]}
        self.assertEqual(module.covering_aef_urls(records, 2019, (35.9, 40.2, 36.1, 40.8), "EPSG:32636"), ["neighbor", "native"])
        self.assertEqual(module.covering_aef_urls(records, 2018, (35.9, 40.2, 36.1, 40.8), "EPSG:32636"), [])


if __name__ == "__main__":
    unittest.main()
