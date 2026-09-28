#!/usr/bin/env python3
"""Materialize exact S1/Dynamic World/AEF triplets for the supplement manifest.

The sampler records every Earth Engine scene ID and a projected 10 m output grid.
This command exports those exact S1 and Dynamic World components, retrieves the
acquisition-year AlphaEarth embedding, validates the aligned triplet, and writes a
resumable per-shard provenance manifest. It never performs a date-only scene lookup.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform as transform_coordinates, transform_bounds

from reconstruct_sample import (
    aef_url,
    download_aef_index,
    download_ee,
    initialize_ee,
    sha256,
    write_aef,
)


S1_COLLECTION = "COPERNICUS/S1_GRD"
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
S1_BANDS = ("VV", "VH", "angle")
OUTPUT_FIELDS = (
    "candidate_id", "tile_name", "primary_target", "split", "group_id",
    "s1_path", "aef_path", "label_path", "s1_image_ids", "dw_image_ids",
    "s1_datetimes_utc", "dw_datetimes_utc", "maximum_time_delta_hours",
    "s1_valid_fraction", "dw_valid_fraction", "water_fraction", "aef_year",
    "aef_source_urls", "s1_sha256", "aef_sha256", "label_sha256", "status",
    "error", "updated_at",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("outputs/audit/supplement_samples.csv"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/pscratch/sd/r/rohit9/S1ML/training_supplement/v1"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--ee-project", default="rohit-global-water")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.90)
    parser.add_argument("--maximum-gap-hours", type=float, default=24.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if not 0 <= args.minimum_valid_fraction <= 1:
        parser.error("--minimum-valid-fraction must be in [0, 1]")
    args.manifest = args.manifest.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir else args.output_root / "cache"
    )
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "candidate_id", "primary_target", "split", "group_id", "s1_image_ids",
        "dw_image_ids", "crs", "width", "height", "transform_a", "transform_b",
        "transform_c", "transform_d", "transform_e", "transform_f",
    }
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError("Supplement manifest is missing columns: {}".format(", ".join(sorted(missing))))
    return rows


def parse_ids(value: str, collection: str) -> list[str]:
    values = json.loads(value)
    if not isinstance(values, list) or not values:
        raise ValueError("Expected a non-empty JSON image-ID list")
    return [item if str(item).startswith(collection + "/") else collection + "/" + str(item) for item in values]


def load_aef_records(index_path: Path, years: set[int]) -> dict[int, list[tuple[str, str, tuple[float, float, float, float]]]]:
    """Read the 762 MB source index once per shard, not once per sample."""
    records = {year: [] for year in years}
    with index_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                year = int(row["year"])
            except (KeyError, ValueError):
                continue
            if year not in records:
                continue
            try:
                bounds = tuple(float(row[key]) for key in (
                    "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"
                ))
            except (KeyError, ValueError):
                continue
            crs = row.get("crs", "")
            if crs.isdigit():
                crs = "EPSG:" + crs
            records[year].append((crs.upper(), aef_url(row.get("path", "")), bounds))
    return records


def matching_aef_urls(
    records: dict[int, list[tuple[str, str, tuple[float, float, float, float]]]],
    year: int,
    bbox: tuple[float, float, float, float],
    crs: str,
) -> list[str]:
    candidates = [
        (source_crs == crs.upper(), url)
        for source_crs, url, tile in records.get(year, [])
        if tile[0] < bbox[2] and tile[2] > bbox[0] and tile[1] < bbox[3] and tile[3] > bbox[1]
    ]
    preferred = [url for same_crs, url in candidates if same_crs]
    return sorted(set(preferred or [url for _, url in candidates]))


def grid(row: dict[str, str]) -> tuple[str, Affine, int, int]:
    affine = Affine(*(float(row["transform_{}".format(key)]) for key in "abcdef"))
    return row["crs"], affine, int(row["width"]), int(row["height"])


def exact_region(ee: Any, crs: str, affine: Affine, width: int, height: int):
    left, bottom, right, top = array_bounds(height, width, affine)
    xs = [left, right, right, left, left]
    ys = [bottom, bottom, top, top, bottom]
    lon, lat = transform_coordinates(crs, "EPSG:4326", xs, ys)
    ring = list(map(list, zip(lon, lat)))
    return ee.Geometry.Polygon([ring], geodesic=False), (left, bottom, right, top)


def collection_from_ids(ee: Any, image_ids: list[str], bands: tuple[str, ...]):
    return ee.ImageCollection.fromImages([ee.Image(image_id).select(list(bands)) for image_id in image_ids])


def image_times(ee: Any, image_ids: list[str]) -> list[int]:
    collection = ee.ImageCollection.fromImages([ee.Image(image_id) for image_id in image_ids])
    return [int(value) for value in collection.aggregate_array("system:time_start").getInfo()]


def iso_utc(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def server_metrics(ee: Any, s1: Any, dw: Any, region: Any) -> dict[str, float]:
    metrics = ee.Image.cat(
        [
            s1.mask().reduce(ee.Reducer.min()).unmask(0).rename("s1_valid"),
            dw.mask().unmask(0).rename("dw_valid"),
            dw.eq(0).unmask(0).rename("water"),
        ]
    ).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=10,
        # This is only a validity/coverage diagnostic.  Earth Engine can count
        # source-projection pixels here (especially near the poles), so the
        # limit must comfortably exceed the 1.21M pixels in the 10 m output
        # grid.  Keep the requested scale exact; bestEffort would make the
        # coverage threshold inconsistent between samples.
        maxPixels=50_000_000, tileScale=4,
    ).getInfo()
    return {key: float(metrics.get(key) or 0) for key in ("s1_valid", "dw_valid", "water")}


def validate_triplet(s1_path: Path, aef_path: Path, label_path: Path) -> None:
    with rio.open(s1_path) as s1, rio.open(aef_path) as aef, rio.open(label_path) as label:
        if s1.count != 3 or aef.count != 64 or label.count != 1:
            raise ValueError("Expected 3/64/1 S1/AEF/label bands")
        signature = lambda src: (src.crs, src.transform, src.width, src.height)
        if signature(s1) != signature(aef) or signature(s1) != signature(label):
            raise ValueError("S1, AEF, and label grids are not identical")
        values = set(np.unique(label.read(1)).tolist())
        if not values.issubset({0, 1}):
            raise ValueError("Label values are not binary: {}".format(sorted(values)))


def atomic_write_aef(
    output: Path, urls: list[str], crs: str, affine: Affine, width: int, height: int
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".tif", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_aef(temporary, urls, crs, affine, width, height)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def base_result(row: dict[str, str], root: Path) -> dict[str, Any]:
    tile_name = row["candidate_id"] + ".tif"
    return {
        "candidate_id": row["candidate_id"], "tile_name": tile_name,
        "primary_target": row["primary_target"], "split": row["split"],
        "group_id": row["group_id"], "s1_path": str(root / "s1" / tile_name),
        "aef_path": str(root / "aef" / tile_name),
        "label_path": str(root / "labels" / tile_name), "status": "", "error": "",
        "updated_at": utc_now(),
    }


def materialize(
    ee: Any, row: dict[str, str], args: argparse.Namespace,
    aef_records: dict[int, list[tuple[str, str, tuple[float, float, float, float]]]],
) -> dict[str, Any]:
    result = base_result(row, args.output_root)
    s1_path, aef_path, label_path = (Path(result[key]) for key in ("s1_path", "aef_path", "label_path"))
    try:
        s1_ids = parse_ids(row["s1_image_ids"], S1_COLLECTION)
        dw_ids = parse_ids(row["dw_image_ids"], DW_COLLECTION)
        crs, affine, width, height = grid(row)
        region, projected_bounds = exact_region(ee, crs, affine, width, height)
        s1_collection = collection_from_ids(ee, s1_ids, S1_BANDS)
        dw_collection = collection_from_ids(ee, dw_ids, ("label",))
        s1 = s1_collection.mosaic().select(list(S1_BANDS)).float().clip(region)
        dw = dw_collection.mosaic().select("label").clip(region)
        metrics = server_metrics(ee, s1, dw, region)
        if metrics["s1_valid"] < args.minimum_valid_fraction or metrics["dw_valid"] < args.minimum_valid_fraction:
            raise ValueError("Exact-grid valid coverage is below threshold: {}".format(metrics))

        s1_times, dw_times = image_times(ee, s1_ids), image_times(ee, dw_ids)
        maximum_gap = max(abs(s1_time - dw_time) / 3_600_000 for s1_time in s1_times for dw_time in dw_times)
        if maximum_gap > args.maximum_gap_hours:
            raise ValueError("A scene-component gap exceeds {} hours: {}".format(args.maximum_gap_hours, maximum_gap))
        aef_year = max(2017, datetime.fromtimestamp(min(s1_times) / 1000, tz=timezone.utc).year)
        bbox_wgs84 = transform_bounds(crs, "EPSG:4326", *projected_bounds, densify_pts=21)
        aef_urls = matching_aef_urls(aef_records, aef_year, bbox_wgs84, crs)
        if not aef_urls:
            raise ValueError("No acquisition-year AEF tile overlaps the output grid")

        if args.overwrite or not s1_path.exists():
            download_ee(s1, s1_path, crs, affine, width, height, args.timeout)
        if args.overwrite or not label_path.exists():
            water = dw.eq(0).unmask(0, sameFootprint=False).uint8().rename("water").clip(region)
            download_ee(water, label_path, crs, affine, width, height, args.timeout)
        if args.overwrite or not aef_path.exists():
            atomic_write_aef(aef_path, aef_urls, crs, affine, width, height)
        validate_triplet(s1_path, aef_path, label_path)
        result.update(
            {
                "s1_image_ids": json.dumps(s1_ids), "dw_image_ids": json.dumps(dw_ids),
                "s1_datetimes_utc": json.dumps([iso_utc(value) for value in s1_times]),
                "dw_datetimes_utc": json.dumps([iso_utc(value) for value in dw_times]),
                "maximum_time_delta_hours": "{:.6f}".format(maximum_gap),
                "s1_valid_fraction": "{:.6f}".format(metrics["s1_valid"]),
                "dw_valid_fraction": "{:.6f}".format(metrics["dw_valid"]),
                "water_fraction": "{:.6f}".format(metrics["water"]),
                "aef_year": aef_year, "aef_source_urls": json.dumps(aef_urls),
                "s1_sha256": sha256(s1_path), "aef_sha256": sha256(aef_path),
                "label_sha256": sha256(label_path), "status": "ok",
            }
        )
    except Exception as exc:
        result.update({"status": "error", "error": "{}: {}".format(type(exc).__name__, exc)})
    result["updated_at"] = utc_now()
    return result


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    rows = [row for index, row in enumerate(read_rows(args.manifest)) if index % args.num_shards == args.shard_index]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifests" / "materialize_{:03d}_of_{:03d}.csv".format(
        args.shard_index, args.num_shards
    )
    previous: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        with manifest_path.open(newline="") as handle:
            previous = {row["candidate_id"]: row for row in csv.DictReader(handle)}
    ee = initialize_ee(args.ee_project)
    aef_index = download_aef_index(args.cache_dir, args.timeout)
    years = {
        max(2017, datetime.fromisoformat(row["s1_datetime_utc"].replace("Z", "+00:00")).year)
        for row in rows
    }
    aef_records = load_aef_records(aef_index, years)
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        prior = previous.get(row["candidate_id"])
        if prior and prior.get("status") == "ok" and not args.overwrite:
            try:
                validate_triplet(Path(prior["s1_path"]), Path(prior["aef_path"]), Path(prior["label_path"]))
                result = prior
            except Exception:
                result = materialize(ee, row, args, aef_records)
        else:
            result = materialize(ee, row, args, aef_records)
        results.append(result)
        write_manifest(manifest_path, results)
        print("[{}/{}] {} {}{}".format(
            index, len(rows), row["candidate_id"], result["status"],
            ": " + result["error"] if result.get("error") else "",
        ), flush=True)
    counts = Counter(row["status"] for row in results)
    print(json.dumps({"manifest": str(manifest_path), "status_counts": dict(counts)}, indent=2))
    if counts.get("error", 0):
        raise SystemExit("Shard contains {} failed samples; rerun after resolving them".format(counts["error"]))


if __name__ == "__main__":
    main()
