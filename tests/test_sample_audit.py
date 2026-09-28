from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

from audit_s1_dw_timing import paired_row, summarize  # noqa: E402
from build_sample_footprints import bounds, grid_ring  # noqa: E402
from inventory_training_samples import water_context  # noqa: E402
from annotate_coastline import threshold_distance_km  # noqa: E402
from annotate_pld_wfs import annotate_batch, cql_bbox_filter  # noqa: E402
from recover_s1_raster_provenance import choose_point_match  # noqa: E402
from annotate_dw_label_manifests import summarize as summarize_labels  # noqa: E402
from assess_sample_balance import assess  # noqa: E402
from sample_water_feature_supplement import select_rows  # noqa: E402
from materialize_supplement_samples import matching_aef_urls, parse_ids  # noqa: E402
try:  # src/train.py imports torch and fastai, which the metadata CI job does not install
    from train import TripletRecord, split_triplets_from_manifest  # noqa: E402
    HAVE_TRAINING_STACK = True
except ModuleNotFoundError:
    HAVE_TRAINING_STACK = False
from shapely.geometry import LineString, Polygon, mapping  # noqa: E402
from shapely.strtree import STRtree  # noqa: E402


class SampleAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        with (ROOT / "metadata" / "training_samples.csv").open(newline="") as handle:
            self.first = next(csv.DictReader(handle))

    def test_grid_ring_contains_recorded_centroid(self) -> None:
        ring = grid_ring(self.first)
        self.assertEqual(ring[0], ring[-1])
        min_lon, min_lat, max_lon, max_lat = bounds(ring)
        self.assertLess(min_lon, float(self.first["centroid_lon"]))
        self.assertGreater(max_lon, float(self.first["centroid_lon"]))
        self.assertLess(min_lat, float(self.first["centroid_lat"]))
        self.assertGreater(max_lat, float(self.first["centroid_lat"]))

    def test_pair_sign_and_summary(self) -> None:
        provenance = {
            "pair_source": "test",
            "s1_ids": ["s1"],
            "s1_times": [36_000_000],
            "dw_ids": ["dw"],
            "dw_times": [28_800_000],
            "s1_passes": ["ASCENDING"],
            "s1_orbits": [12],
        }
        row = paired_row(self.first, provenance)
        self.assertEqual(row["pair_status"], "resolved")
        self.assertEqual(float(row["s1_minus_dw_hours"]), 2.0)
        self.assertEqual(row["within_6_hours"], 1)
        report = summarize([row])
        self.assertEqual(report["resolved_samples"], 1)
        self.assertEqual(report["absolute_gap_hours"]["median"], 2.0)

    def test_water_context_is_multilabel(self) -> None:
        row = {
            "pld_overlap_fraction": "0.25",
            "pld_reservoir_flag": "true",
            "coastline_distance_km": "2.5",
            "jrc_permanent_fraction": "0.3",
            "jrc_seasonal_fraction": "0.1",
        }
        self.assertEqual(
            water_context(row, lake_threshold=0.01, coast_km=5),
            [
                "river_targeted", "lake_or_reservoir", "reservoir", "coastal",
                "persistent_water", "seasonal_water",
            ],
        )

    def test_water_context_applies_pld_fraction_threshold(self) -> None:
        row = {"pld_overlap_fraction": "0.0001", "pld_feature_count": "3"}
        self.assertEqual(
            water_context(row, lake_threshold=0.01, coast_km=5),
            ["river_targeted"],
        )

    def test_coastline_threshold_uses_footprint(self) -> None:
        boundary = LineString([(0.0, -1.0), (0.0, 1.0)])
        boundaries = [boundary]
        tree = STRtree(boundaries)
        footprint = Polygon([(0.01, -0.01), (0.02, -0.01), (0.02, 0.01), (0.01, 0.01)])
        distance = threshold_distance_km(footprint, 0.0, boundaries, tree, threshold_km=5.0)
        self.assertIsNotNone(distance)
        self.assertAlmostEqual(distance, 1.11195, places=4)

    def test_pld_annotation_unions_overlaps(self) -> None:
        footprints = [{"tile_name": "sample", "geometry": Polygon.from_bounds(0, 0, 2, 2)}]
        features = [
            {
                "id": "lake.1",
                "geometry": mapping(Polygon.from_bounds(0, 0, 1.5, 2)),
                "properties": {"fid": "1", "p_ref_area": 3.0},
            },
            {
                "id": "lake.2",
                "geometry": mapping(Polygon.from_bounds(1, 0, 2, 2)),
                "properties": {"fid": "2", "p_ref_area": 2.0},
            },
        ]
        row = annotate_batch(footprints, features, "test")[0]
        self.assertEqual(float(row["pld_overlap_fraction"]), 1.0)
        self.assertEqual(row["pld_feature_count"], 2)
        self.assertEqual(row["pld_lake_ids"], "1;2")
        self.assertEqual(row["pld_max_reference_area_km2"], "3.000000")

    def test_pld_cql_uses_wfs_lat_lon_axis_order(self) -> None:
        footprint = {"tile_name": "sample", "geometry": Polygon.from_bounds(10, 20, 11, 21)}
        self.assertEqual(cql_bbox_filter([footprint]), "BBOX(geom,20.0,10.0,21.0,11.0)")

    def test_raster_fingerprint_prefers_exact_scene(self) -> None:
        fingerprint = {"vv": -10.0, "vh": -20.0, "angle": 35.0}
        candidates = [
            {"candidate_index": 0, "vv": -11.0, "vh": -19.0, "angle": 42.0},
            {"candidate_index": 1, "vv": -10.0, "vh": -20.0, "angle": 35.00001},
        ]
        match = choose_point_match(fingerprint, candidates)
        self.assertIsNotNone(match)
        self.assertEqual(match["candidate_index"], 1)
        self.assertEqual(match["match_kind"], "exact")

    def test_dw_label_summary_reports_pixel_exposure(self) -> None:
        rows = [
            {"dw_label_water_fraction": "0.0"},
            {"dw_label_water_fraction": "0.2"},
        ]
        report = summarize_labels(rows)
        self.assertEqual(report["threshold_counts"]["below_0.01"], 1)
        self.assertEqual(report["threshold_counts"]["at_least_0.10"], 1)
        self.assertEqual(report["water_fraction"]["equivalent_full_water_tiles"], 0.2)

    def test_balance_assessment_preserves_overlapping_contexts(self) -> None:
        rows = [
            {
                "split": "valid", "hemisphere": "south", "sample_year": "2022",
                "sample_month": "10", "dw_label_water_fraction": "0.2",
                "water_contexts": "river_targeted;lake_or_reservoir;coastal;seasonal_water",
            }
        ]
        report = assess(rows)
        self.assertEqual(report["cross_strata"]["lake_coastal"]["samples"], 1)
        self.assertEqual(report["context_counts"]["seasonal_water"]["samples"], 1)
        self.assertTrue(report["mild_supplement_plan"]["quota_tags_are_overlapping"])

    def test_supplement_selection_requires_both_seasonal_signals(self) -> None:
        base = {
            "primary_target": "high_water_seasonal_or_ephemeral",
            "pair_status": "pair_found", "absolute_time_delta_hours": "2",
            "dw_water_fraction": "0.2", "jrc_seasonal_fraction": "0.2",
            "hemisphere": "south", "water_fraction_bin": "10_to_40pct",
            "centroid_lon": "0", "centroid_lat": "0",
        }
        valid = {**base, "candidate_id": "valid"}
        low_jrc = {
            **base, "candidate_id": "low_jrc", "centroid_lon": "1",
            "jrc_seasonal_fraction": "0.09",
        }
        low_dw = {
            **base, "candidate_id": "low_dw", "centroid_lon": "2",
            "dw_water_fraction": "0.09",
        }
        selected = select_rows(
            [low_jrc, low_dw, valid],
            {"high_water_seasonal_or_ephemeral": 1},
            minimum_km=20,
            seed=42,
        )
        self.assertEqual([row["candidate_id"] for row in selected], ["valid"])

    def test_materializer_normalizes_scene_ids_and_prefers_matching_crs(self) -> None:
        self.assertEqual(
            parse_ids('["scene_a"]', "COLLECTION"),
            ["COLLECTION/scene_a"],
        )
        records = {
            2022: [
                ("EPSG:32610", "preferred", (-1, -1, 1, 1)),
                ("EPSG:32611", "fallback", (-1, -1, 1, 1)),
            ]
        }
        self.assertEqual(
            matching_aef_urls(records, 2022, (-0.5, -0.5, 0.5, 0.5), "EPSG:32610"),
            ["preferred"],
        )

    @unittest.skipUnless(HAVE_TRAINING_STACK, "torch and fastai are not installed")
    def test_fixed_split_manifest_is_immutable(self) -> None:
        records = [
            TripletRecord("a.tif", Path("a"), None, Path("la")),
            TripletRecord("b.tif", Path("b"), None, Path("lb")),
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "split.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["tile_name", "split"])
                writer.writeheader()
                writer.writerows([
                    {"tile_name": "a.tif", "split": "valid"},
                    {"tile_name": "b.tif", "split": "train"},
                ])
            train, valid = split_triplets_from_manifest(records, manifest)
        self.assertEqual([record.name for record in train], ["b.tif"])
        self.assertEqual([record.name for record in valid], ["a.tif"])


if __name__ == "__main__":
    unittest.main()
