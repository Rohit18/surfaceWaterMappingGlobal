#!/usr/bin/env python3
"""Recover Sentinel-1 scene provenance by matching catalog pixels to stored rasters.

The date embedded in legacy sample names is a search anchor, not always the sensing
date.  For every target raster, this script searches nearby Earth Engine S1 GRD scenes
and compares VV, VH, and incidence-angle values at five output-pixel centers. Exact
matches identify the scene(s) that actually contributed pixels to the stored raster.

The output is resumable and contains only metadata and match diagnostics; imagery is
not downloaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rasterio as rio
import numpy as np
from rasterio.warp import transform

from build_sample_footprints import grid_ring


S1_COLLECTION = "COPERNICUS/S1_GRD"
POINT_FRACTIONS = ((0.5, 0.5), (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75))
OUTPUT_FIELDS = (
    "tile_name", "indexed_date", "match_status", "provenance_quality",
    "candidate_window_days", "candidate_scene_count", "matched_s1_scene_count",
    "matched_s1_image_ids", "matched_s1_datetimes", "matched_s1_offsets_days",
    "matched_s1_passes", "matched_s1_relative_orbits", "sample_point_count",
    "matched_point_count", "exact_point_match_count", "angle_point_match_count",
    "max_matched_vv_error", "max_matched_vh_error", "max_matched_angle_error", "error",
)


def initialize_ee(project: str | None):
    try:
        import ee

        ee.deprecation.deprecated_assets = {"__skip_optional_catalog__": None}
        ee.Initialize(project=project)
        return ee
    except Exception as exc:
        raise SystemExit(
            "Earth Engine authentication is required. On a remote shell without gcloud, run "
            "`earthengine authenticate --auth_mode=notebook --force`: {}".format(exc)
        ) from exc


def iso_utc(timestamp_ms: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def chunks(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def raster_fingerprints(path: Path) -> list[dict[str, Any]]:
    """Read VV/VH/angle at stable interior output-pixel centers."""
    with rio.open(path) as dataset:
        if dataset.count < 3:
            raise ValueError("Expected at least three S1 bands: {}".format(path))
        # These legacy TIFFs are commonly stored as one large compressed block. Read
        # that block once; five tiny window reads can otherwise decompress it five times.
        data = dataset.read([1, 2, 3])
        valid = (
            np.isfinite(data).all(axis=0)
            & (np.abs(data[0]) > 1e-12)
            & (np.abs(data[1]) > 1e-12)
            & (data[2] > 0)
        )
        valid_rows, valid_columns = np.nonzero(valid)
        if len(valid_rows) == 0:
            raise ValueError("No valid dual-polarization fingerprint pixels: {}".format(path))
        result = []
        used_pixels = set()
        for point_index, (row_fraction, column_fraction) in enumerate(POINT_FRACTIONS):
            row = min(int(round((dataset.height - 1) * row_fraction)), dataset.height - 1)
            column = min(int(round((dataset.width - 1) * column_fraction)), dataset.width - 1)
            if not valid[row, column] or (row, column) in used_pixels:
                distances = (valid_rows - row) ** 2 + (valid_columns - column) ** 2
                for candidate in np.argsort(distances):
                    pixel = (int(valid_rows[candidate]), int(valid_columns[candidate]))
                    if pixel not in used_pixels:
                        row, column = pixel
                        break
            used_pixels.add((row, column))
            values = data[:, row, column]
            x, y = dataset.xy(row, column)
            lon, lat = transform(dataset.crs, "EPSG:4326", [x], [y])
            result.append(
                {
                    "point_index": point_index,
                    "lon": float(lon[0]),
                    "lat": float(lat[0]),
                    "vv": float(values[0]),
                    "vh": float(values[1]),
                    "angle": float(values[2]),
                }
            )
    return result


def candidate_metadata(ee, rows: list[dict[str, str]], window_days: int):
    features = []
    for row in rows:
        region = ee.Geometry.Polygon([grid_ring(row)], proj="EPSG:4326", geodesic=False)
        target = ee.Date(row["s1_date"])
        collection = (
            ee.ImageCollection(S1_COLLECTION)
            .filterBounds(region)
            .filterDate(target.advance(-window_days, "day"), target.advance(window_days + 1, "day"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .sort("system:time_start")
        )
        features.append(
            ee.Feature(
                None,
                {
                    "tile_name": row["tile_name"],
                    "ids": collection.aggregate_array("system:index"),
                    "times": collection.aggregate_array("system:time_start"),
                    "passes": collection.aggregate_array("orbitProperties_pass"),
                    "orbits": collection.aggregate_array("relativeOrbitNumber_start"),
                },
            )
        )
    response = ee.FeatureCollection(features).getInfo()
    result = {}
    for feature in response["features"]:
        properties = feature["properties"]
        result[properties["tile_name"]] = {
            "ids": list(properties.get("ids", [])),
            "times": [int(value) for value in properties.get("times", [])],
            "passes": list(properties.get("passes", [])),
            "orbits": list(properties.get("orbits", [])),
        }
    return result


def sample_candidates(ee, metadata, fingerprints):
    features = []
    for tile_name, candidates in metadata.items():
        for candidate_index, image_id in enumerate(candidates["ids"]):
            image = ee.Image("{}/{}".format(S1_COLLECTION, image_id))
            for fingerprint in fingerprints[tile_name]:
                point = ee.Geometry.Point([fingerprint["lon"], fingerprint["lat"]])
                values = image.select(["VV", "VH", "angle"]).reduceRegion(
                    reducer=ee.Reducer.first(), geometry=point, scale=10
                )
                features.append(
                    ee.Feature(
                        None,
                        values.combine(
                            {
                                "tile_name": tile_name,
                                "candidate_index": candidate_index,
                                "point_index": fingerprint["point_index"],
                            }
                        ),
                    )
                )
    if not features:
        return {}
    response = ee.FeatureCollection(features).getInfo()
    result: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for feature in response["features"]:
        properties = feature["properties"]
        record = {
            "candidate_index": int(properties["candidate_index"]),
            "vv": properties.get("VV"),
            "vh": properties.get("VH"),
            "angle": properties.get("angle"),
        }
        result.setdefault(properties["tile_name"], {}).setdefault(
            int(properties["point_index"]), []
        ).append(record)
    return result


def choose_point_match(
    fingerprint: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any] | None:
    scored = []
    for candidate in candidates:
        if not all(isinstance(candidate.get(field), (int, float)) for field in ("vv", "vh", "angle")):
            continue
        vv_error = abs(float(candidate["vv"]) - fingerprint["vv"])
        vh_error = abs(float(candidate["vh"]) - fingerprint["vh"])
        angle_error = abs(float(candidate["angle"]) - fingerprint["angle"])
        # VV/VH equality is decisive. The angle band is float32 and can differ by a
        # few 1e-4 degrees after reprojection to the output grid.
        exact = vv_error <= 1e-7 and vh_error <= 1e-7 and angle_error <= 5e-4
        scored.append(
            {
                **candidate,
                "vv_error": vv_error,
                "vh_error": vh_error,
                "angle_error": angle_error,
                "match_kind": "exact" if exact else "candidate",
            }
        )
    exact = [candidate for candidate in scored if candidate["match_kind"] == "exact"]
    if exact:
        return min(exact, key=lambda item: (item["angle_error"], item["vv_error"] + item["vh_error"]))

    # Angle is a stable geometric fingerprint even where an output-grid reprojection
    # makes the point sample land in a neighboring speckle pixel. Require a unique
    # near-identical angle within the short date window.
    angle_matches = [candidate for candidate in scored if candidate["angle_error"] <= 0.003]
    if len(angle_matches) == 1:
        match = angle_matches[0]
        match["match_kind"] = "angle"
        return match
    return None


def result_row(
    row: dict[str, str],
    candidates: dict[str, Any],
    fingerprints: list[dict[str, Any]],
    samples: dict[int, list[dict[str, Any]]],
    window_days: int,
) -> dict[str, Any]:
    output = {field: "" for field in OUTPUT_FIELDS}
    output.update(
        {
            "tile_name": row["tile_name"],
            "indexed_date": row["s1_date"],
            "candidate_window_days": window_days,
            "candidate_scene_count": len(candidates["ids"]),
            "sample_point_count": len(fingerprints),
        }
    )
    point_matches = []
    for fingerprint in fingerprints:
        match = choose_point_match(fingerprint, samples.get(fingerprint["point_index"], []))
        if match is not None:
            point_matches.append(match)
    if not point_matches:
        output.update(
            {
                "match_status": "unresolved",
                "provenance_quality": "missing",
                "matched_point_count": 0,
                "exact_point_match_count": 0,
                "angle_point_match_count": 0,
                "error": "No stored-raster fingerprint matched a nearby catalog scene",
            }
        )
        return output

    matched_indices = sorted({int(match["candidate_index"]) for match in point_matches})
    exact_count = sum(match["match_kind"] == "exact" for match in point_matches)
    angle_count = sum(match["match_kind"] == "angle" for match in point_matches)
    times = [candidates["times"][index] for index in matched_indices]
    target_ms = int(datetime.fromisoformat(row["s1_date"]).replace(tzinfo=timezone.utc).timestamp() * 1000)
    quality = "raster_verified" if exact_count else "raster_angle_matched"
    output.update(
        {
            "match_status": "matched",
            "provenance_quality": quality,
            "matched_s1_scene_count": len(matched_indices),
            "matched_s1_image_ids": json.dumps(
                ["{}/{}".format(S1_COLLECTION, candidates["ids"][index]) for index in matched_indices]
            ),
            "matched_s1_datetimes": json.dumps([iso_utc(timestamp) for timestamp in times]),
            "matched_s1_offsets_days": json.dumps(
                [round((timestamp - target_ms) / 86_400_000, 6) for timestamp in times]
            ),
            "matched_s1_passes": json.dumps([candidates["passes"][index] for index in matched_indices]),
            "matched_s1_relative_orbits": json.dumps(
                [candidates["orbits"][index] for index in matched_indices]
            ),
            "matched_point_count": len(point_matches),
            "exact_point_match_count": exact_count,
            "angle_point_match_count": angle_count,
            "max_matched_vv_error": "{:.9f}".format(max(match["vv_error"] for match in point_matches)),
            "max_matched_vh_error": "{:.9f}".format(max(match["vh_error"] for match in point_matches)),
            "max_matched_angle_error": "{:.9f}".format(
                max(match["angle_error"] for match in point_matches)
            ),
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("metadata/training_samples.csv"))
    parser.add_argument("--s1-raster-dir", type=Path, required=True)
    parser.add_argument(
        "--existing-timing", type=Path, default=Path("outputs/audit/s1_dw_timing.csv"),
        help="When present, skip rows already marked resolved.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/audit/s1_raster_provenance.csv")
    )
    parser.add_argument("--ee-project", default=None)
    parser.add_argument("--window-days", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="Seed completed rows from another recovery CSV while writing to --output.",
    )
    parser.add_argument(
        "--retry-unresolved", action="store_true",
        help="When resuming, process prior unresolved rows again and retain matched rows.",
    )
    parser.add_argument(
        "--retry-quality", action="append", default=[],
        help="When resuming, reprocess rows having this provenance_quality value.",
    )
    args = parser.parse_args()
    if args.window_days < 0 or args.batch_size < 1:
        parser.error("window days must be non-negative and batch size must be positive")

    with args.index.expanduser().resolve().open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    timing_path = args.existing_timing.expanduser().resolve()
    if timing_path.exists():
        with timing_path.open(newline="") as handle:
            resolved = {
                row["tile_name"] for row in csv.DictReader(handle) if row.get("pair_status") == "resolved"
            }
        rows = [row for row in rows if row["tile_name"] not in resolved]

    output = args.output.expanduser().resolve()
    completed: dict[str, dict[str, str]] = {}
    resume_path = output if output.exists() else (
        args.resume_from.expanduser().resolve() if args.resume_from else output
    )
    if args.resume and resume_path.exists():
        with resume_path.open(newline="") as handle:
            completed = {row["tile_name"]: row for row in csv.DictReader(handle)}
        if args.retry_unresolved:
            completed = {
                tile_name: row
                for tile_name, row in completed.items()
                if row.get("match_status") == "matched"
            }
        if args.retry_quality:
            completed = {
                tile_name: row
                for tile_name, row in completed.items()
                if row.get("provenance_quality") not in set(args.retry_quality)
            }
        rows = [row for row in rows if row["tile_name"] not in completed]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    raster_root = args.s1_raster_dir.expanduser().resolve()
    ee = initialize_ee(args.ee_project)
    new_rows = []
    for batch_number, batch in enumerate(chunks(rows, args.batch_size), start=1):
        fingerprints = {
            row["tile_name"]: raster_fingerprints(raster_root / row["tile_name"])
            for row in batch
        }
        metadata = candidate_metadata(ee, batch, args.window_days)
        sampled = sample_candidates(
            ee, metadata, fingerprints
        )
        for row in batch:
            tile_name = row["tile_name"]
            new_rows.append(
                result_row(
                    row,
                    metadata[tile_name],
                    fingerprints[tile_name],
                    sampled.get(tile_name, {}),
                    args.window_days,
                )
            )
        print(
            "Processed batch {} ({} new samples; {} matched)".format(
                batch_number,
                len(new_rows),
                sum(row["match_status"] == "matched" for row in new_rows),
            ),
            flush=True,
        )
        # Persist after every batch so --resume survives scheduler/network interruptions.
        merged = list(completed.values()) + new_rows
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(merged)

    merged = list(completed.values()) + new_rows
    print(
        json.dumps(
            {
                "rows": len(merged),
                "matched": sum(row["match_status"] == "matched" for row in merged),
                "raster_verified": sum(
                    row["provenance_quality"] == "raster_verified" for row in merged
                ),
                "raster_angle_matched": sum(
                    row["provenance_quality"] == "raster_angle_matched" for row in merged
                ),
                "unresolved": sum(row["match_status"] != "matched" for row in merged),
                "output_csv": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
