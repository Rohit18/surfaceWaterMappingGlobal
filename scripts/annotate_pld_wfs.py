#!/usr/bin/env python3
"""Annotate sample footprints with targeted SWOT Prior Lake Database polygons.

HydroWeb publishes the PLD as the ``REF_DATA:swot_prior_lake_db`` WFS layer.
This script batches indexed bounding-box filters, then calculates exact polygon
intersections locally.  It avoids downloading the multi-gigabyte global PLD files.

WFS 2.0 applies EPSG:4326 axis order to CQL BBOX filters, so CQL coordinates are
sent as latitude,longitude even though returned GeoJSON is longitude,latitude.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.strtree import STRtree


DEFAULT_ENDPOINT = "https://hydroweb.next.theia-land.fr/geoserver/REF_DATA/ows"
DEFAULT_LAYER = "REF_DATA:swot_prior_lake_db"
DEFAULT_SOURCE = "SWOT_PLD_2.02_HydroWeb_WFS"


class WFSQueryError(RuntimeError):
    pass


def chunks(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def cql_bbox_filter(footprints: list[dict[str, Any]]) -> str:
    clauses = []
    for item in footprints:
        min_lon, min_lat, max_lon, max_lat = item["geometry"].bounds
        clauses.append(
            "BBOX(geom,{},{},{},{})".format(min_lat, min_lon, max_lat, max_lon)
        )
    return " OR ".join(clauses)


def cache_path(cache_dir: Path | None, footprints: list[dict[str, Any]]) -> Path | None:
    if cache_dir is None:
        return None
    names = "\n".join(item["tile_name"] for item in footprints)
    digest = hashlib.sha256(names.encode()).hexdigest()[:20]
    return cache_dir / "{}.geojson".format(digest)


def request_features(
    footprints: list[dict[str, Any]],
    endpoint: str,
    layer: str,
    timeout_seconds: float,
    retries: int,
    cache_dir: Path | None,
) -> list[dict[str, Any]]:
    cached = cache_path(cache_dir, footprints)
    if cached is not None and cached.exists():
        return json.loads(cached.read_text())["features"]

    features: list[dict[str, Any]] = []
    start_index = 0
    count = 10_000
    while True:
        data = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": layer,
            "propertyName": "fid,geom,p_ref_area",
            "outputFormat": "application/json",
            "count": str(count),
            "startIndex": str(start_index),
            "sortBy": "fid",
            "cql_filter": cql_bbox_filter(footprints),
        }
        response = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(endpoint, data=data, timeout=timeout_seconds)
                if response.ok and "json" in (response.headers.get("content-type") or ""):
                    break
            except requests.RequestException:
                response = None
            if attempt < retries:
                time.sleep(2**attempt)
        if response is None or not response.ok or "json" not in (
            response.headers.get("content-type") or ""
        ):
            status = response.status_code if response is not None else "request-error"
            raise WFSQueryError("WFS query failed with status {}".format(status))

        payload = response.json()
        page = payload.get("features", [])
        features.extend(page)
        number_matched = payload.get("numberMatched")
        if len(page) < count or (
            isinstance(number_matched, int) and len(features) >= number_matched
        ):
            break
        start_index += len(page)

    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return features


def fetch_with_split(
    footprints: list[dict[str, Any]],
    endpoint: str,
    layer: str,
    timeout_seconds: float,
    retries: int,
    cache_dir: Path | None,
) -> list[dict[str, Any]]:
    """Retry a failed batch as two smaller indexed queries."""
    try:
        return request_features(
            footprints, endpoint, layer, timeout_seconds, retries, cache_dir
        )
    except WFSQueryError:
        if len(footprints) == 1:
            raise
        middle = len(footprints) // 2
        return fetch_with_split(
            footprints[:middle], endpoint, layer, timeout_seconds, retries, cache_dir
        ) + fetch_with_split(
            footprints[middle:], endpoint, layer, timeout_seconds, retries, cache_dir
        )


def annotate_batch(
    footprints: list[dict[str, Any]], features: list[dict[str, Any]], source: str
) -> list[dict[str, Any]]:
    sample_geometries = [item["geometry"] for item in footprints]
    sample_tree = STRtree(sample_geometries)
    intersections: list[list[Any]] = [[] for _ in footprints]
    ids: list[list[str]] = [[] for _ in footprints]
    reference_areas: list[list[float]] = [[] for _ in footprints]

    for feature in features:
        if not feature.get("geometry"):
            continue
        lake = shape(feature["geometry"])
        if lake.is_empty:
            continue
        if not lake.is_valid:
            lake = lake.buffer(0)
        properties = feature.get("properties", {})
        lake_id = str(properties.get("fid") or feature.get("id") or "")
        reference_area = properties.get("p_ref_area")
        for index in sample_tree.query(lake):
            sample_index = int(index)
            intersection = lake.intersection(sample_geometries[sample_index])
            if intersection.is_empty or intersection.area <= 0:
                continue
            intersections[sample_index].append(intersection)
            ids[sample_index].append(lake_id)
            if isinstance(reference_area, (int, float)) and math.isfinite(reference_area):
                reference_areas[sample_index].append(float(reference_area))

    rows = []
    for index, item in enumerate(footprints):
        covered = unary_union(intersections[index]).area if intersections[index] else 0.0
        fraction = min(covered / sample_geometries[index].area, 1.0)
        unique_ids = sorted(set(ids[index]))
        rows.append(
            {
                "tile_name": item["tile_name"],
                "pld_source": source,
                "pld_overlap_fraction": "{:.8f}".format(fraction),
                "pld_feature_count": len(unique_ids),
                "pld_lake_ids": ";".join(unique_ids),
                "pld_max_reference_area_km2": (
                    "{:.6f}".format(max(reference_areas[index]))
                    if reference_areas[index]
                    else ""
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--footprints", type=Path, default=Path("outputs/audit/sample_footprints.geojson")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/audit/pld_annotations.csv"))
    parser.add_argument(
        "--summary", type=Path, default=Path("outputs/audit/pld_annotations_summary.json")
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--source-label", default=DEFAULT_SOURCE)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=90)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/audit/pld_wfs_cache"))
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers <= 0 or args.timeout_seconds <= 0 or args.retries < 0:
        parser.error("batch size, workers, and timeout must be positive; retries cannot be negative")

    collection = json.loads(args.footprints.expanduser().resolve().read_text())
    footprints = [
        {
            "tile_name": feature["properties"]["tile_name"],
            "geometry": shape(feature["geometry"]),
        }
        for feature in collection["features"]
    ]
    batches = list(chunks(footprints, args.batch_size))
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else None

    def process(batch):
        features = fetch_with_split(
            batch,
            args.endpoint,
            args.layer,
            args.timeout_seconds,
            args.retries,
            cache_dir,
        )
        return annotate_batch(batch, features, args.source_label)

    output_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for completed, rows in enumerate(executor.map(process, batches), start=1):
            output_rows.extend(rows)
            print(
                "Processed {}/{} batches ({} footprints)".format(
                    completed, len(batches), len(output_rows)
                ),
                flush=True,
            )

    by_name = {row["tile_name"]: row for row in output_rows}
    ordered_rows = [by_name[item["tile_name"]] for item in footprints]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered_rows)

    fractions = [float(row["pld_overlap_fraction"]) for row in ordered_rows]
    counts = [int(row["pld_feature_count"]) for row in ordered_rows]
    summary = {
        "samples": len(ordered_rows),
        "samples_intersecting_pld": sum(count > 0 for count in counts),
        "unique_pld_lake_ids": len(
            {
                lake_id
                for row in ordered_rows
                for lake_id in row["pld_lake_ids"].split(";")
                if lake_id
            }
        ),
        "overlap_threshold_counts": {
            "greater_than_0": sum(value > 0 for value in fractions),
            "at_least_0.001": sum(value >= 0.001 for value in fractions),
            "at_least_0.01": sum(value >= 0.01 for value in fractions),
            "at_least_0.05": sum(value >= 0.05 for value in fractions),
        },
        "source": args.source_label,
        "layer": args.layer,
        "method": "batched WFS BBOX retrieval; exact local polygon intersection and union",
        "reservoir_note": (
            "The public WFS layer exposes lake ID, geometry, and reference area but no "
            "reservoir provenance; reservoir status is not inferred here."
        ),
        "output_csv": str(output),
    }
    summary_path = args.summary.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
