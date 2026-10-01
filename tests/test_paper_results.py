import json
import statistics
import unittest
from pathlib import Path


RESULTS = Path(__file__).resolve().parents[1] / "results" / "paper"


def load(relative):
    return json.loads((RESULTS / relative).read_text())


class PaperResultsTest(unittest.TestCase):
    """Paper values (10 m inference, threshold 0.30, common valid mask), to three decimals."""

    @classmethod
    def setUpClass(cls):
        cls.table_i = load("table_I/summary_thr0.30.json")["common_mask"]["methods"]

    def test_table_i_pooled_iou(self):
        expected = {"S1-only [10m]": 0.768, "S1+AEF(t) [10m]": 0.851, "OPERA DSWx-S1": 0.747}
        for method, iou in expected.items():
            with self.subTest(method=method):
                self.assertEqual(round(self.table_i[method]["pooled_water_iou"], 3), iou)

    def test_table_i_rows(self):
        expected = {
            "S1-only [10m]": (0.883, 0.854, 0.869, 0.617, 0.527, 0.702),
            "S1+AEF(t) [10m]": (0.891, 0.950, 0.920, 0.741, 0.669, 0.806),
            "OPERA DSWx-S1": (0.859, 0.851, 0.855, 0.591, 0.508, 0.670),
            "S1+AEF(t-1) swap [10m]": (0.886, 0.949, 0.917, 0.732, 0.662, 0.797),
            "S1+AEF(t-1) trained [10m]": (0.889, 0.948, 0.918, 0.734, 0.663, 0.800),
        }
        for method, values in expected.items():
            row = self.table_i[method]
            got = (row["precision"], row["recall"], row["dice"], row["per_scene_mean"], *row["per_scene_ci95"])
            with self.subTest(method=method):
                self.assertEqual(tuple(round(v, 3) for v in got), values)
                self.assertEqual(row["valid_pixels"], 51897488)

    def test_table_ii_k16(self):
        configs = load("table_II/ablation_10m_summary.json")["configs"]
        (k16,) = [c for c in configs
                  if c["convention"] == "thr030_common" and c["ablation"] == "aef_width" and c["value"] == 16]
        seeds = k16["per_scene_mean_iou_seeds"]
        self.assertEqual(len(seeds), 3)
        self.assertEqual(round(statistics.fmean(seeds), 3), 0.741)
        self.assertEqual(round(statistics.stdev(seeds), 3), 0.001)


if __name__ == "__main__":
    unittest.main()
