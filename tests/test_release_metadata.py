from __future__ import annotations

import csv
import json
import random
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTest(unittest.TestCase):
    def test_training_index_counts(self) -> None:
        with (ROOT / "metadata" / "training_samples.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4678)
        self.assertEqual(Counter(row["split"] for row in rows), {"train": 3742, "valid": 936})
        self.assertEqual(len({row["tile_name"] for row in rows}), len(rows))
        self.assertEqual(sum(row["s1_date"][:4] != row["aef_year"] for row in rows), 98)

        tile_names = sorted(row["tile_name"] for row in rows)
        random.Random(42).shuffle(tile_names)
        expected_valid = set(tile_names[:936])
        indexed_valid = {row["tile_name"] for row in rows if row["split"] == "valid"}
        self.assertEqual(indexed_valid, expected_valid)

    def test_evaluation_index_count(self) -> None:
        with (ROOT / "metadata" / "evaluation_samples.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 53)
        self.assertEqual(len({row["sample_id"] for row in rows}), 53)

    def test_model_registry(self) -> None:
        registry = json.loads((ROOT / "models" / "model_registry.json").read_text())
        model = registry["s1aef_resnet34"]
        self.assertEqual(len(model["checkpoint_sha256"]), 64)
        self.assertEqual(model["paper_threshold"], 0.3)


if __name__ == "__main__":
    unittest.main()
