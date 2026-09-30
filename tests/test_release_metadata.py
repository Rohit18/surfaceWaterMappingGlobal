from __future__ import annotations

import ast
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

    def test_training_centroids(self) -> None:
        collection = json.loads((ROOT / "metadata" / "training_sample_centroids.geojson").read_text())
        features = collection["features"]
        props = [feature["properties"] for feature in features]
        self.assertEqual(len(props), 5278)
        self.assertEqual(len({p["tile_name"] for p in props}), 5278)
        self.assertEqual(Counter(p["split"] for p in props), {"train": 4222, "valid": 1056})
        self.assertEqual(
            Counter((p["sample_set"], p["split"]) for p in props),
            {
                ("sword_river_node", "train"): 3742,
                ("sword_river_node", "valid"): 936,
                ("open_water_supplement", "train"): 480,
                ("open_water_supplement", "valid"): 120,
            },
        )
        self.assertEqual(sum(p["in_previous_year_model"] for p in props), 5140)
        with (ROOT / "metadata" / "training_samples.csv").open(newline="") as handle:
            index = {row["tile_name"]: row for row in csv.DictReader(handle)}
        for feature in features:
            row = index.get(feature["properties"]["tile_name"])
            if row is None:
                continue
            self.assertEqual(feature["properties"]["split"], row["split"])
            lon, lat = feature["geometry"]["coordinates"]
            self.assertAlmostEqual(lon, float(row["centroid_lon"]), delta=1e-6)
            self.assertAlmostEqual(lat, float(row["centroid_lat"]), delta=1e-6)

    def test_registry_matches_downloader(self) -> None:
        registry = json.loads((ROOT / "models" / "model_registry.json").read_text())
        tree = ast.parse((ROOT / "scripts" / "download_model.py").read_text())
        variants = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "VARIANTS"
        )
        pairs = {
            "fused_current": ("s1aef_resnet34", "58321212", "width_k16_seed42"),
            "s1_only": ("s1_only_resnet34", "58321212", "width_k0_seed42"),
            "fused_previous_year": ("s1aef_previous_year_resnet34", "58321217", "width_k16_seed42"),
        }
        self.assertEqual(set(variants), set(pairs))
        for variant, (key, job, task) in pairs.items():
            entry = registry[key]
            self.assertEqual(entry["hugging_face_revision"], "v2-labelclass")
            self.assertEqual(
                variants[variant],
                {
                    entry["checkpoint"]: entry["checkpoint_sha256"],
                    entry["normalization"]: entry["normalization_sha256"],
                },
            )
            self.assertEqual(entry["training_run"]["slurm_job"], job)
            self.assertEqual(entry["training_run"]["task"], task)


if __name__ == "__main__":
    unittest.main()
