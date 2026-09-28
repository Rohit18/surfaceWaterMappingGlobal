#!/usr/bin/env python3
"""Audit Sentinel-1 versus Dynamic World acquisition timing for training samples.

The command performs metadata-only Earth Engine queries; it does not export imagery.
Existing generation manifests can be supplied to reuse recorded provenance. Dynamic
World time is the acquisition time of its corresponding Sentinel-2 L1C observation.
The signed gap is defined as S1 time minus Dynamic World time, so positive values mean
that Sentinel-1 was acquired later.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from build_sample_footprints import grid_ring


S1_COLLECTION = "COPERNICUS/S1_GRD"
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
HOUR_MS = 60 * 60 * 1000
OUTPUT_FIELDS = (
    "tile_name", "sword_node_id", "split", "s1_date", "centroid_lon", "centroid_lat",
    "pair_status", "provenance_quality", "pair_source", "s1_scene_count", "dw_scene_count",
    "nearest_s1_image_id", "nearest_s1_datetime_utc", "nearest_dw_image_id",
    "nearest_dw_datetime_utc", "s1_minus_dw_hours", "absolute_time_delta_hours",
    "same_utc_day", "within_6_hours", "within_12_hours", "within_24_hours",
    "within_48_hours", "s1_acquisition_span_minutes", "dw_acquisition_span_minutes",
    "s1_orbit_pass", "s1_relative_orbit", "s1_image_ids", "s1_datetimes_utc",
    "dw_image_ids", "dw_datetimes_utc", "error",
)


def iso_utc(timestamp_ms: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_utc(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return int(parsed.timestamp() * 1000)


def json_list(value: str) -> list[Any]:
    if not value:
        return []
    parsed = json.loads(value)
    return parsed if isinstance(parsed, list) else [parsed]


def full_id(collection: str, value: str) -> str:
    return value if value.startswith(collection + "/") else "{}/{}".format(collection, value)


def load_manifest_provenance(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Read original-DW and replacement-pair manifests into one tile lookup."""
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.expanduser().resolve().open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if "matched_s1_image_ids" in row:
                if row.get("match_status") != "matched":
                    continue
                key = row.get("tile_name", "")
                if not key:
                    continue
                current = result.setdefault(
                    key,
                    {
                        "pair_source": "raster_pixel_match",
                        "s1_ids": [], "s1_times": [], "dw_ids": [], "dw_times": [],
                        "s1_passes": [], "s1_orbits": [],
                    },
                )
                current["s1_ids"] = json_list(row.get("matched_s1_image_ids", ""))
                current["s1_times"] = [
                    parse_utc(value) for value in json_list(row.get("matched_s1_datetimes", ""))
                ]
                current["s1_passes"] = json_list(row.get("matched_s1_passes", ""))
                current["s1_orbits"] = json_list(row.get("matched_s1_relative_orbits", ""))
                current["raster_match_quality"] = row.get("provenance_quality", "")
                current["pair_source"] = (
                    "raster_pixel_match_plus_original_dw_manifest"
                    if current.get("dw_times")
                    else "raster_pixel_match"
                )
                continue
            if row.get("status") not in ("ok", "exists"):
                continue
            if "replacement_tile_name" in row:
                key = row.get("replacement_tile_name", "")
                if not key:
                    continue
                result[key] = {
                    "pair_source": "recovery_manifest",
                    "s1_ids": json_list(row.get("s1_scene_ids", "")),
                    "s1_times": [parse_utc(value) for value in json_list(row.get("s1_datetimes", ""))],
                    "dw_ids": json_list(row.get("dw_image_ids", "")),
                    "dw_times": [parse_utc(value) for value in json_list(row.get("dw_datetimes", ""))],
                    "s1_passes": [],
                    "s1_orbits": [],
                }
            elif "dw_image_id" in row:
                key = row.get("tile_name", "")
                if not key or not row.get("dw_datetime"):
                    continue
                current = result.setdefault(
                    key,
                    {
                        "pair_source": "original_dw_manifest",
                        "s1_ids": [], "s1_times": [], "dw_ids": [], "dw_times": [],
                        "s1_passes": [], "s1_orbits": [],
                    },
                )
                current["dw_ids"] = [row.get("dw_image_id", "")]
                current["dw_times"] = [parse_utc(row["dw_datetime"])]
    return result


def initialize_ee(project: str | None):
    try:
        import ee
        ee.deprecation.deprecated_assets = {"__skip_optional_catalog__": None}
        ee.Initialize(project=project)
        return ee
    except Exception as exc:
        raise SystemExit(
            "Earth Engine authentication is required for unresolved samples. Run "
            "`earthengine authenticate --auth_mode=notebook --force` on a remote shell "
            "without gcloud, then rerun this command. Original error: {}".format(exc)
        ) from exc


def query_batch(ee, rows: list[dict[str, str]], window_days: int) -> dict[str, dict[str, Any]]:
    features = []
    for row in rows:
        geometry = ee.Geometry.Polygon([grid_ring(row)], proj="EPSG:4326", geodesic=False)
        features.append(ee.Feature(geometry, {"tile_name": row["tile_name"], "s1_date": row["s1_date"]}))

    def resolve(item):
        item = ee.Feature(item)
        region = item.geometry()
        target = ee.Date(item.get("s1_date"))
        s1 = (
            ee.ImageCollection(S1_COLLECTION)
            .filterBounds(region)
            .filterDate(target, target.advance(1, "day"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .sort("system:time_start")
        )
        dw = (
            ee.ImageCollection(DW_COLLECTION)
            .filterBounds(region)
            .filterDate(target.advance(-window_days, "day"), target.advance(window_days + 1, "day"))
            .map(lambda image: image.set(
                "_midnight_gap_ms",
                ee.Number(image.get("system:time_start")).subtract(target.millis()).abs(),
            ))
            .sort("_midnight_gap_ms")
        )
        dw_count = dw.size()
        selected_dw = ee.Image(dw.first())
        return ee.Feature(None, {
            "tile_name": item.get("tile_name"),
            "s1_indices": s1.aggregate_array("system:index"),
            "s1_times": s1.aggregate_array("system:time_start"),
            "s1_passes": s1.aggregate_array("orbitProperties_pass"),
            "s1_orbits": s1.aggregate_array("relativeOrbitNumber_start"),
            "dw_count": dw_count,
            "dw_index": ee.Algorithms.If(dw_count.gt(0), selected_dw.get("system:index"), ""),
            "dw_time": ee.Algorithms.If(dw_count.gt(0), selected_dw.get("system:time_start"), None),
        })

    response = ee.FeatureCollection(features).map(resolve).getInfo()
    result: dict[str, dict[str, Any]] = {}
    for feature in response["features"]:
        properties = feature["properties"]
        dw_index, dw_time = properties.get("dw_index"), properties.get("dw_time")
        result[properties["tile_name"]] = {
            "pair_source": "earth_engine_metadata",
            "s1_ids": [full_id(S1_COLLECTION, str(value)) for value in properties.get("s1_indices", [])],
            "s1_times": [int(value) for value in properties.get("s1_times", [])],
            "s1_passes": list(properties.get("s1_passes", [])),
            "s1_orbits": list(properties.get("s1_orbits", [])),
            "dw_ids": [full_id(DW_COLLECTION, str(dw_index))] if dw_index else [],
            "dw_times": [int(dw_time)] if dw_time is not None else [],
        }
    return result


def load_stac_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[row["tile_name"]] = row
    return result


def query_s1_stac(
    row: dict[str, str],
    endpoint: str,
    timeout: int,
    attempts: int = 4,
) -> dict[str, Any]:
    """Resolve same-day IW VV/VH scenes through the public Earth Search STAC API."""
    ring = grid_ring(row)
    xs, ys = zip(*ring)
    payload = {
        "collections": ["sentinel-1-grd"],
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
        "datetime": "{}T00:00:00Z/{}T23:59:59Z".format(row["s1_date"], row["s1_date"]),
        "limit": 100,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "surface-water-sample-audit/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 == attempts:
                return {"tile_name": row["tile_name"], "error": "{}: {}".format(type(exc).__name__, exc)}
            time.sleep(2 ** attempt)
    else:  # pragma: no cover - loop always returns or breaks
        raise RuntimeError(last_error)

    records = []
    for item in body.get("features", []):
        properties = item.get("properties", {})
        mode = str(properties.get("sar:instrument_mode", "")).upper()
        polarizations = {str(value).upper() for value in properties.get("sar:polarizations", [])}
        if mode != "IW" or not {"VV", "VH"}.issubset(polarizations):
            continue
        sensing_start = properties.get("start_datetime") or properties.get("datetime")
        if not sensing_start:
            continue
        records.append(
            {
                "id": "EARTH-SEARCH/sentinel-1-grd/{}".format(item["id"]),
                "time": parse_utc(sensing_start),
                "pass": str(properties.get("sat:orbit_state", "")).upper(),
                "orbit": properties.get("sat:relative_orbit", ""),
            }
        )
    records.sort(key=lambda value: (value["time"], value["id"]))
    return {
        "tile_name": row["tile_name"],
        "s1_ids": [value["id"] for value in records],
        "s1_times": [value["time"] for value in records],
        "s1_passes": [value["pass"] for value in records],
        "s1_orbits": [value["orbit"] for value in records],
        "error": "" if records else "No same-day IW VV/VH Sentinel-1 STAC item",
    }


def query_s1_stac_many(
    rows: list[dict[str, str]],
    endpoint: str,
    workers: int,
    timeout: int,
    cache_path: Path,
) -> dict[str, dict[str, Any]]:
    cache_path = cache_path.expanduser().resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = load_stac_cache(cache_path)
    pending = [row for row in rows if row["tile_name"] not in cached]
    if pending:
        with cache_path.open("a") as cache_handle:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(query_s1_stac, row, endpoint, timeout): row["tile_name"]
                    for row in pending
                }
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    cache_handle.write(json.dumps(result) + "\n")
                    cache_handle.flush()
                    cached[result["tile_name"]] = result
                    completed += 1
                    if completed % 100 == 0 or completed == len(pending):
                        print("Resolved public STAC metadata {}/{}".format(completed, len(pending)), flush=True)
    return cached


def nearest_pair(s1_times: list[int], dw_times: list[int]) -> tuple[int, int, int, int]:
    candidates = [
        (abs(s1_time - dw_time), s1_time, dw_time, s1_index, dw_index)
        for s1_index, s1_time in enumerate(s1_times)
        for dw_index, dw_time in enumerate(dw_times)
    ]
    _, s1_time, dw_time, s1_index, dw_index = min(candidates)
    return s1_index, dw_index, s1_time, dw_time


def paired_row(index_row: dict[str, str], provenance: dict[str, Any] | None) -> dict[str, Any]:
    output: dict[str, Any] = {field: "" for field in OUTPUT_FIELDS}
    for field in ("tile_name", "sword_node_id", "split", "s1_date", "centroid_lon", "centroid_lat"):
        output[field] = index_row[field]
    if provenance is None:
        output.update(
            {
                "pair_status": "unresolved",
                "provenance_quality": "missing",
                "error": "No provenance metadata",
            }
        )
        return output

    s1_ids = list(provenance.get("s1_ids", []))
    s1_times = [int(value) for value in provenance.get("s1_times", [])]
    dw_ids = list(provenance.get("dw_ids", []))
    dw_times = [int(value) for value in provenance.get("dw_times", [])]
    s1_passes = list(provenance.get("s1_passes", []))
    s1_orbits = list(provenance.get("s1_orbits", []))
    output.update(
        {
            "pair_source": provenance.get("pair_source", ""),
            "provenance_quality": (
                provenance.get("raster_match_quality")
                if provenance.get("raster_match_quality")
                else
                "verified"
                if provenance.get("pair_source") == "recovery_manifest"
                else "catalog_inferred"
                if provenance.get("pair_source") in (
                    "manifest_dw_plus_earth_search_s1",
                    "manifest_dw_plus_earth_engine_s1",
                    "earth_engine_metadata",
                )
                else "incomplete"
            ),
            "s1_scene_count": len(s1_times),
            "dw_scene_count": len(dw_times),
            "s1_image_ids": json.dumps(s1_ids),
            "s1_datetimes_utc": json.dumps([iso_utc(value) for value in s1_times]),
            "dw_image_ids": json.dumps(dw_ids),
            "dw_datetimes_utc": json.dumps([iso_utc(value) for value in dw_times]),
            "s1_acquisition_span_minutes": (
                "{:.3f}".format((max(s1_times) - min(s1_times)) / 60000) if s1_times else ""
            ),
            "dw_acquisition_span_minutes": (
                "{:.3f}".format((max(dw_times) - min(dw_times)) / 60000) if dw_times else ""
            ),
        }
    )
    if not s1_times or not dw_times:
        missing = "S1" if not s1_times else "Dynamic World"
        output.update(
            {
                "pair_status": "unresolved",
                "error": provenance.get("error") or "Missing {} acquisition metadata".format(missing),
            }
        )
        return output

    s1_index, dw_index, s1_time, dw_time = nearest_pair(s1_times, dw_times)
    signed_hours = (s1_time - dw_time) / HOUR_MS
    absolute_hours = abs(signed_hours)
    output.update(
        {
            "pair_status": "resolved",
            "nearest_s1_image_id": s1_ids[s1_index] if s1_index < len(s1_ids) else "",
            "nearest_s1_datetime_utc": iso_utc(s1_time),
            "nearest_dw_image_id": dw_ids[dw_index] if dw_index < len(dw_ids) else "",
            "nearest_dw_datetime_utc": iso_utc(dw_time),
            "s1_minus_dw_hours": "{:.6f}".format(signed_hours),
            "absolute_time_delta_hours": "{:.6f}".format(absolute_hours),
            "same_utc_day": int(iso_utc(s1_time)[:10] == iso_utc(dw_time)[:10]),
            "within_6_hours": int(absolute_hours <= 6),
            "within_12_hours": int(absolute_hours <= 12),
            "within_24_hours": int(absolute_hours <= 24),
            "within_48_hours": int(absolute_hours <= 48),
            "s1_orbit_pass": s1_passes[s1_index] if s1_index < len(s1_passes) else "",
            "s1_relative_orbit": s1_orbits[s1_index] if s1_index < len(s1_orbits) else "",
        }
    )
    return output


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = [row for row in rows if row["pair_status"] == "resolved"]
    verified = [
        row for row in resolved
        if row["provenance_quality"] in {"verified", "raster_verified"}
    ]
    gaps = [float(row["absolute_time_delta_hours"]) for row in resolved]
    signed = [float(row["s1_minus_dw_hours"]) for row in resolved]
    total = len(rows)
    summary: dict[str, Any] = {
        "samples": total,
        "resolved_samples": len(resolved),
        "unresolved_samples": total - len(resolved),
        "resolved_percent": round(100 * len(resolved) / total, 3) if total else 0,
        "verified_samples": len(verified),
        "verified_percent": round(100 * len(verified) / total, 3) if total else 0,
        "verified_by_method": {
            "generation_manifest": sum(
                row["provenance_quality"] == "verified" for row in resolved
            ),
            "stored_raster_pixel_match": sum(
                row["provenance_quality"] == "raster_verified" for row in resolved
            ),
        },
        "provenance_quality_counts": {
            value: sum(row["provenance_quality"] == value for row in rows)
            for value in sorted({row["provenance_quality"] for row in rows})
        },
        "status_by_provenance_quality": {
            quality: {
                status: sum(
                    row["provenance_quality"] == quality and row["pair_status"] == status
                    for row in rows
                )
                for status in ("resolved", "unresolved")
            }
            for quality in sorted({row["provenance_quality"] for row in rows})
        },
    }
    if not resolved:
        return summary
    summary.update(
        {
            "absolute_gap_hours": {
                "mean": round(mean(gaps), 6),
                "median": round(median(gaps), 6),
                "p25": round(percentile(gaps, 0.25), 6),
                "p75": round(percentile(gaps, 0.75), 6),
                "p90": round(percentile(gaps, 0.90), 6),
                "p95": round(percentile(gaps, 0.95), 6),
                "max": round(max(gaps), 6),
            },
            "thresholds": {
                "within_6_hours": sum(value <= 6 for value in gaps),
                "within_12_hours": sum(value <= 12 for value in gaps),
                "within_24_hours": sum(value <= 24 for value in gaps),
                "within_48_hours": sum(value <= 48 for value in gaps),
            },
            "ordering": {
                "s1_before_dw": sum(value < 0 for value in signed),
                "same_time": sum(value == 0 for value in signed),
                "s1_after_dw": sum(value > 0 for value in signed),
            },
            "same_utc_day": sum(int(row["same_utc_day"]) for row in resolved),
            "multi_scene_s1_samples": sum(int(row["s1_scene_count"]) > 1 for row in resolved),
            "multi_scene_dw_samples": sum(int(row["dw_scene_count"]) > 1 for row in resolved),
        }
    )
    summary["threshold_percent_of_resolved"] = {
        key: round(100 * value / len(resolved), 3) for key, value in summary["thresholds"].items()
    }
    summary["timing_by_provenance_quality"] = {}
    for quality in sorted({row["provenance_quality"] for row in resolved}):
        subset = [row for row in resolved if row["provenance_quality"] == quality]
        subset_gaps = [float(row["absolute_time_delta_hours"]) for row in subset]
        summary["timing_by_provenance_quality"][quality] = {
            "samples": len(subset),
            "absolute_gap_hours_mean": round(mean(subset_gaps), 6),
            "absolute_gap_hours_median": round(median(subset_gaps), 6),
            "within_6_hours": sum(value <= 6 for value in subset_gaps),
            "within_12_hours": sum(value <= 12 for value in subset_gaps),
            "within_24_hours": sum(value <= 24 for value in subset_gaps),
            "within_48_hours": sum(value <= 48 for value in subset_gaps),
            "same_utc_day": sum(int(row["same_utc_day"]) for row in subset),
        }
    index_offsets = [
        (
            parse_utc(row["nearest_s1_datetime_utc"])
            - parse_utc(row["s1_date"] + "T00:00:00Z")
        )
        / HOUR_MS
        for row in resolved
    ]
    summary["s1_vs_indexed_date"] = {
        "same_utc_date": sum(
            row["nearest_s1_datetime_utc"][:10] == row["s1_date"] for row in resolved
        ),
        "different_utc_date": sum(
            row["nearest_s1_datetime_utc"][:10] != row["s1_date"] for row in resolved
        ),
        "offset_hours_from_indexed_midnight": {
            "min": round(min(index_offsets), 6),
            "median": round(median(index_offsets), 6),
            "max": round(max(index_offsets), 6),
        },
    }
    if verified:
        verified_gaps = [float(row["absolute_time_delta_hours"]) for row in verified]
        verified_signed = [float(row["s1_minus_dw_hours"]) for row in verified]
        summary["verified_only"] = {
            "absolute_gap_hours": {
                "mean": round(mean(verified_gaps), 6),
                "median": round(median(verified_gaps), 6),
                "p25": round(percentile(verified_gaps, 0.25), 6),
                "p75": round(percentile(verified_gaps, 0.75), 6),
                "p90": round(percentile(verified_gaps, 0.90), 6),
                "p95": round(percentile(verified_gaps, 0.95), 6),
                "max": round(max(verified_gaps), 6),
            },
            "thresholds": {
                "within_6_hours": sum(value <= 6 for value in verified_gaps),
                "within_12_hours": sum(value <= 12 for value in verified_gaps),
                "within_24_hours": sum(value <= 24 for value in verified_gaps),
                "within_48_hours": sum(value <= 48 for value in verified_gaps),
            },
            "ordering": {
                "s1_before_dw": sum(value < 0 for value in verified_signed),
                "same_time": sum(value == 0 for value in verified_signed),
                "s1_after_dw": sum(value > 0 for value in verified_signed),
            },
        }
    return summary


def chunks(values: list[dict[str, str]], size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("metadata/training_samples.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/audit/s1_dw_timing.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/audit/s1_dw_timing_summary.json"))
    parser.add_argument("--provenance-manifest", type=Path, action="append", default=[])
    parser.add_argument("--offline-only", action="store_true", help="Use manifests only; do not query Earth Engine.")
    parser.add_argument(
        "--s1-stac-fallback",
        action="store_true",
        help="Resolve missing S1 times from public Earth Search metadata; exact DW manifest times are still required.",
    )
    parser.add_argument("--stac-url", default="https://earth-search.aws.element84.com/v1/search")
    parser.add_argument("--stac-workers", type=int, default=8)
    parser.add_argument("--stac-timeout", type=int, default=60)
    parser.add_argument("--stac-cache", type=Path, default=Path("outputs/audit/s1_stac_cache.jsonl"))
    parser.add_argument("--window-days", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--ee-project", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if args.window_days < 0 or args.batch_size < 1 or args.stac_workers < 1:
        parser.error("window/batch values must be non-negative and worker counts must be >= 1")
    if args.offline_only and args.s1_stac_fallback:
        parser.error("--offline-only and --s1-stac-fallback are mutually exclusive")

    with args.index.expanduser().resolve().open(newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    if args.max_samples is not None:
        index_rows = index_rows[: args.max_samples]
    provenance = load_manifest_provenance(args.provenance_manifest)

    if args.s1_stac_fallback:
        targets = [
            row for row in index_rows
            if provenance.get(row["tile_name"], {}).get("dw_times")
            and not provenance.get(row["tile_name"], {}).get("s1_times")
        ]
        resolved = query_s1_stac_many(
            targets, args.stac_url, args.stac_workers, args.stac_timeout, args.stac_cache
        )
        for row in targets:
            tile_name = row["tile_name"]
            stac = resolved.get(tile_name, {})
            existing = provenance[tile_name]
            existing["s1_ids"] = stac.get("s1_ids", [])
            existing["s1_times"] = stac.get("s1_times", [])
            existing["s1_passes"] = stac.get("s1_passes", [])
            existing["s1_orbits"] = stac.get("s1_orbits", [])
            existing["pair_source"] = "manifest_dw_plus_earth_search_s1"
            if stac.get("error"):
                existing["error"] = stac["error"]
    elif not args.offline_only:
        ee = initialize_ee(args.ee_project)
        for number, batch in enumerate(chunks(index_rows, args.batch_size), start=1):
            queried = query_batch(ee, batch, args.window_days)
            for tile_name, values in queried.items():
                # Preserve the exact DW label provenance when available, but fill
                # its missing S1 scene metadata from the live collection query.
                existing = provenance.get(tile_name, {})
                if existing.get("dw_times"):
                    values["dw_ids"] = existing["dw_ids"]
                    values["dw_times"] = existing["dw_times"]
                    values["pair_source"] = "manifest_dw_plus_earth_engine_s1"
                provenance[tile_name] = values
            print("Resolved metadata batch {} ({} samples)".format(number, len(batch)), flush=True)

    output_rows = [paired_row(row, provenance.get(row["tile_name"])) for row in index_rows]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    summary = summarize(output_rows)
    summary.update(
        {
            "signed_gap_definition": "Sentinel-1 acquisition time minus Dynamic World/Sentinel-2 acquisition time",
            "dynamic_world_selection": (
                "Recorded Dynamic World image(s) from generation manifests; timing uses the "
                "nearest S1-Dynamic World acquisition pair when a mosaic has multiple scenes."
                if args.offline_only and args.provenance_manifest
                else "Nearest image to 00:00 UTC of indexed S1 date within +/-{} day(s)".format(
                    args.window_days
                )
            ),
            "publication_guidance": (
                "Use verified_only statistics for scientific reporting. Verified rows have either "
                "generation-recorded provenance or an exact VV/VH/angle match to the stored S1 raster. "
                "Date-only catalog candidates remain diagnostic rather than verified."
            ),
            "output_csv": str(output),
        }
    )
    summary_path = args.summary.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
