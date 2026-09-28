#!/usr/bin/env python3
"""Build a stratified 600-sample open-water supplement with <=24 h S1/DW gaps.

The command is deliberately staged and resumable:

* ``pool`` builds a large, source-traceable centroid pool from SWOT PLD, Global Dam
  Watch reservoirs, GSHHG shorelines, and JRC Global Surface Water seasonality.
* ``pairs`` searches exact Sentinel-1 and Dynamic World acquisition timestamps in a
  deterministic non-October target month, then measures water/valid coverage.
* ``select`` applies the 24-hour rule, spatial de-duplication, water-exposure bins,
  quotas, and a grouped train/validation assignment.

This creates sample centroids, exact footprints, source IDs, and scene provenance. It
does not export the large S1/DW/AEF raster triplets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
import shapefile
from rasterio.warp import transform as transform_coordinates
from shapely.geometry import Point, box, mapping, shape
from shapely.strtree import STRtree


PLD_ENDPOINT = "https://hydroweb.next.theia-land.fr/geoserver/REF_DATA/ows"
PLD_LAYER = "REF_DATA:swot_prior_lake_db"
SWORD_LAYER = "REF_DATA:swot_prior_river_db"
S1_COLLECTION = "COPERNICUS/S1_GRD"
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
JRC_COLLECTION = "JRC/GSW1_4/GlobalSurfaceWater"
NON_OCTOBER_MONTHS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12)
PAIR_FIELDS = (
    "candidate_id", "primary_target", "source_dataset", "source_feature_id",
    "centroid_lon", "centroid_lat", "hemisphere", "group_id", "target_year",
    "target_month", "pair_attempt", "pair_status", "s1_image_id", "s1_image_ids", "s1_datetime_utc",
    "dw_image_id", "dw_image_ids", "dw_datetime_utc", "dw_datetimes_utc",
    "s1_minus_dw_hours", "absolute_time_delta_hours", "maximum_component_time_delta_hours",
    "s1_valid_fraction", "dw_valid_fraction", "dw_water_fraction",
    "jrc_max_extent_fraction", "jrc_permanent_fraction", "jrc_seasonal_fraction",
    "water_fraction_bin", "selection_tags", "error",
)


def stable_score(seed: int, *values: Any) -> int:
    text = "|".join([str(seed), *(str(value) for value in values)])
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    value = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(value))


def valid_location(lon: float, lat: float) -> bool:
    return -180 <= lon <= 180 and -70 <= lat <= 75


def read_existing(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="") as handle:
        return [(float(row["centroid_lon"]), float(row["centroid_lat"])) for row in csv.DictReader(handle)]


def far_from_existing(
    lon: float, lat: float, existing: list[tuple[float, float]], minimum_km: float
) -> bool:
    lat_radius = minimum_km / 110.574
    lon_radius = minimum_km / max(20.0, 111.320 * math.cos(math.radians(lat)))
    return not any(
        abs(other_lat - lat) <= lat_radius
        and abs(other_lon - lon) <= lon_radius
        and haversine_km(lon, lat, other_lon, other_lat) < minimum_km
        for other_lon, other_lat in existing
    )


def candidate(
    target: str,
    source: str,
    source_id: str,
    lon: float,
    lat: float,
    seed: int,
    **extra: Any,
) -> dict[str, Any]:
    digest = hashlib.sha256("{}|{}|{}".format(source, source_id, seed).encode()).hexdigest()[:12]
    year = 2020 + stable_score(seed, source_id, "year") % 4
    month = NON_OCTOBER_MONTHS[stable_score(seed, source_id, "month") % len(NON_OCTOBER_MONTHS)]
    row: dict[str, Any] = {
        "candidate_id": "SUP_{}_{}".format(target.upper(), digest),
        "primary_target": target,
        "source_dataset": source,
        "source_feature_id": source_id,
        "centroid_lon": "{:.8f}".format(lon),
        "centroid_lat": "{:.8f}".format(lat),
        "hemisphere": "north" if lat >= 0 else "south",
        "group_id": "{}:{}".format(source, source_id),
        "target_year": year,
        "target_month": month,
    }
    row.update(extra)
    return row


def read_reservoir_candidates(
    path: Path,
    existing: list[tuple[float, float]],
    seed: int,
    limit: int,
    existing_distance_km: float,
) -> tuple[list[dict[str, Any]], list[Any]]:
    reader = shapefile.Reader(str(path))
    fields = [field[0] for field in reader.fields[1:]]
    result = []
    polygons = []
    for item in reader.iterShapeRecords():
        record = dict(zip(fields, item.record))
        geometry = shape(item.shape.__geo_interface__)
        if geometry.is_empty:
            continue
        polygons.append(geometry)
        point = geometry.representative_point()
        lon, lat = point.x, point.y
        if not valid_location(lon, lat) or not far_from_existing(
            lon, lat, existing, existing_distance_km
        ):
            continue
        source_id = str(record.get("GDW_ID") or record.get("GRAND_ID") or len(result))
        area = record.get("AREA_POLY") or record.get("AREA_SKM") or ""
        result.append(
            candidate(
                "confirmed_reservoir", "GDW_v1.0", source_id, lon, lat, seed,
                reference_area_km2=area, reservoir_name=record.get("RES_NAME", ""),
            )
        )
    result.sort(
        key=lambda row: (
            0 if row["hemisphere"] == "south" else 1,
            stable_score(seed, row["source_feature_id"], "reservoir"),
        )
    )
    return result[:limit], polygons


def coastline_points(path: Path, spacing_km: float) -> Iterable[tuple[str, float, float]]:
    reader = shapefile.Reader(str(path))
    for shape_record in reader.iterShapeRecords():
        source_id = str(shape_record.record[0])
        points = shape_record.shape.points
        parts = list(shape_record.shape.parts) + [len(points)]
        for part_index, (start, end) in enumerate(zip(parts, parts[1:])):
            previous = None
            distance = spacing_km
            for lon, lat in points[start:end]:
                if not valid_location(lon, lat):
                    continue
                if previous is not None:
                    distance += haversine_km(previous[0], previous[1], lon, lat)
                previous = (lon, lat)
                if distance >= spacing_km:
                    yield "{}:{}".format(source_id, part_index), lon, lat
                    distance = 0.0


def read_coastal_candidates(
    paths: list[Path],
    existing: list[tuple[float, float]],
    seed: int,
    limit: int,
    existing_distance_km: float,
) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        for sequence, (source_id, lon, lat) in enumerate(coastline_points(path, 35.0)):
            if not far_from_existing(lon, lat, existing, existing_distance_km):
                continue
            unique_id = "{}:{}".format(source_id, sequence)
            result.append(candidate("coastal_or_estuarine", "GSHHG_2.3.7", unique_id, lon, lat, seed))
    result.sort(
        key=lambda row: (
            0 if row["hemisphere"] == "south" else 1,
            stable_score(seed, row["source_feature_id"], "coast"),
        )
    )
    return result[:limit]


def pld_cell_features(
    cell: tuple[int, int, int, int], count: int, timeout: int, retries: int
) -> list[dict[str, Any]]:
    min_lon, min_lat, max_lon, max_lat = cell
    data = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeNames": PLD_LAYER, "propertyName": "fid,geom,p_ref_area",
        "outputFormat": "application/json", "count": str(count), "sortBy": "fid",
        "cql_filter": "BBOX(geom,{},{},{},{}) AND p_ref_area >= 1".format(
            min_lat, min_lon, max_lat, max_lon
        ),
    }
    for attempt in range(retries + 1):
        try:
            response = requests.post(PLD_ENDPOINT, data=data, timeout=timeout)
            if response.ok and "json" in (response.headers.get("content-type") or ""):
                return response.json().get("features", [])
        except requests.RequestException:
            pass
        if attempt < retries:
            time.sleep(2**attempt)
    return []


def query_sword_presence(
    candidates: list[dict[str, Any]], batch_size: int, timeout: int
) -> set[str]:
    connected: set[str] = set()
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        clauses = []
        boxes = []
        for row in batch:
            lon, lat = float(row["centroid_lon"]), float(row["centroid_lat"])
            lat_radius = 10.0 / 110.574
            lon_radius = 10.0 / max(20.0, 111.320 * math.cos(math.radians(lat)))
            boxes.append(box(lon - lon_radius, lat - lat_radius, lon + lon_radius, lat + lat_radius))
            clauses.append(
                "BBOX(geom,{},{},{},{})".format(
                    lat - lat_radius, lon - lon_radius, lat + lat_radius, lon + lon_radius
                )
            )
        data = {
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": SWORD_LAYER, "propertyName": "fid,geom",
            "outputFormat": "application/json", "count": "10000",
            "cql_filter": " OR ".join(clauses),
        }
        response = requests.post(PLD_ENDPOINT, data=data, timeout=timeout)
        response.raise_for_status()
        lines = [shape(feature["geometry"]) for feature in response.json().get("features", []) if feature.get("geometry")]
        if not lines:
            continue
        tree = STRtree(lines)
        for row, search_box in zip(batch, boxes):
            if len(tree.query(search_box)):
                connected.add(row["candidate_id"])
    return connected


def read_lake_candidates(
    existing: list[tuple[float, float]],
    reservoir_polygons: list[Any],
    seed: int,
    limit: int,
    per_cell: int,
    workers: int,
    existing_distance_km: float,
    timeout: int,
) -> list[dict[str, Any]]:
    cells = [
        (lon, lat, lon + 30, lat + 15)
        for lat in range(-60, 75, 15)
        for lon in range(-180, 180, 30)
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pages = list(executor.map(lambda cell: pld_cell_features(cell, per_cell, timeout, 2), cells))
    reservoir_tree = STRtree(reservoir_polygons)
    result = []
    seen = set()
    for feature in (feature for page in pages for feature in page):
        source_id = str(feature.get("properties", {}).get("fid") or feature.get("id", ""))
        if not source_id or source_id in seen or not feature.get("geometry"):
            continue
        seen.add(source_id)
        geometry = shape(feature["geometry"])
        point = geometry.representative_point()
        lon, lat = point.x, point.y
        if not valid_location(lon, lat) or not far_from_existing(
            lon, lat, existing, existing_distance_km
        ):
            continue
        if len(reservoir_tree.query(point, predicate="intersects")):
            continue
        result.append(
            candidate(
                "non_river_natural_lake", "SWOT_PLD_2.02", source_id, lon, lat, seed,
                reference_area_km2=feature.get("properties", {}).get("p_ref_area", ""),
            )
        )
    result.sort(
        key=lambda row: (
            0 if row["hemisphere"] == "south" else 1,
            stable_score(seed, row["source_feature_id"], "lake"),
        )
    )
    result = result[: max(limit * 3, limit)]
    connected = query_sword_presence(result, batch_size=25, timeout=timeout)
    return [row for row in result if row["candidate_id"] not in connected][:limit]


def initialize_ee(project: str | None):
    import ee

    ee.deprecation.deprecated_assets = {"__skip_optional_catalog__": None}
    ee.Initialize(project=project)
    return ee


def seasonal_candidates(
    ee, existing: list[tuple[float, float]], seed: int, limit: int, existing_distance_km: float
) -> list[dict[str, Any]]:
    seasonality = ee.Image(JRC_COLLECTION).select("seasonality")
    masked = seasonality.updateMask(seasonality.gte(1).And(seasonality.lte(9)))
    sampled = masked.stratifiedSample(
        numPoints=max(100, math.ceil(limit / 9)),
        classBand="seasonality",
        region=ee.Geometry.Rectangle([-180, -60, 180, 75], geodesic=False),
        scale=1000,
        seed=seed,
        geometries=True,
        tileScale=4,
    ).getInfo()["features"]
    result = []
    for feature in sampled:
        lon, lat = feature["geometry"]["coordinates"]
        if not far_from_existing(lon, lat, existing, existing_distance_km):
            continue
        source_id = "JRC_{}_{}".format(round(lon, 5), round(lat, 5))
        result.append(
            candidate(
                "high_water_seasonal_or_ephemeral", "JRC_GSW1.4", source_id,
                lon, lat, seed, reference_seasonality_months=feature["properties"].get("seasonality", ""),
            )
        )
    result.sort(
        key=lambda row: (
            0 if row["hemisphere"] == "south" else 1,
            stable_score(seed, row["source_feature_id"], "seasonal"),
        )
    )
    return result[:limit]


def seasonal_fraction_stats(ee, rows: list[dict[str, Any]]) -> dict[str, float]:
    features = ee.FeatureCollection(
        [
            ee.Feature(
                footprint_geometry(ee, float(row["centroid_lon"]), float(row["centroid_lat"])),
                {"candidate_id": row["candidate_id"]},
            )
            for row in rows
        ]
    )
    seasonality = ee.Image(JRC_COLLECTION).select("seasonality")
    seasonal = seasonality.gte(1).And(seasonality.lte(11)).unmask(0).rename("seasonal")
    reduced = seasonal.reduceRegions(
        collection=features, reducer=ee.Reducer.mean(), scale=30, tileScale=4
    ).getInfo()
    return {
        feature["properties"]["candidate_id"]: float(feature["properties"].get("mean") or 0)
        for feature in reduced["features"]
    }


def prefilter_seasonal_candidates(
    ee,
    rows: list[dict[str, Any]],
    minimum_fraction: float,
    batch_size: int,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    accepted = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        stats = seasonal_fraction_stats(ee, batch)
        for row in batch:
            fraction = stats.get(row["candidate_id"], 0.0)
            if fraction >= minimum_fraction:
                item = dict(row)
                item["reference_seasonal_fraction"] = "{:.6f}".format(fraction)
                accepted.append(item)
        print(
            "Seasonal prefilter {}/{}: {} accepted".format(
                min(start + len(batch), len(rows)), len(rows), len(accepted)
            ),
            flush=True,
        )
    accepted.sort(
        key=lambda row: (
            0 if row["hemisphere"] == "south" else 1,
            stable_score(seed, row["candidate_id"], "seasonal_prefilter"),
        )
    )
    return accepted[:limit]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def iso_utc(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def query_month_metadata(ee, rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    features = []
    for row in rows:
        start = "{:04d}-{:02d}-01".format(int(row["target_year"]), int(row["target_month"]))
        month = int(row["target_month"])
        end = "{:04d}-{:02d}-01".format(
            int(row["target_year"]) + (1 if month == 12 else 0), 1 if month == 12 else month + 1
        )
        features.append(
            ee.Feature(
                ee.Geometry.Point([float(row["centroid_lon"]), float(row["centroid_lat"])]),
                {"candidate_id": row["candidate_id"], "start": start, "end": end},
            )
        )

    def attach(feature):
        geometry = feature.geometry()
        start, end = ee.String(feature.get("start")), ee.String(feature.get("end"))
        s1 = (
            ee.ImageCollection(S1_COLLECTION).filterBounds(geometry).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .sort("system:time_start")
        )
        dw = ee.ImageCollection(DW_COLLECTION).filterBounds(geometry).filterDate(start, end).sort("system:time_start")
        return feature.set(
            {
                "s1_ids": s1.aggregate_array("system:index"),
                "s1_times": s1.aggregate_array("system:time_start"),
                "dw_ids": dw.aggregate_array("system:index"),
                "dw_times": dw.aggregate_array("system:time_start"),
            }
        )

    result = ee.FeatureCollection(features).map(attach).getInfo()
    return {feature["properties"]["candidate_id"]: feature["properties"] for feature in result["features"]}


def nearest_pair(metadata: dict[str, Any], max_gap_hours: float) -> tuple[str, int, str, int] | None:
    pairs = []
    for s1_id, s1_time in zip(metadata.get("s1_ids", []), metadata.get("s1_times", [])):
        for dw_id, dw_time in zip(metadata.get("dw_ids", []), metadata.get("dw_times", [])):
            gap = abs(int(s1_time) - int(dw_time)) / 3_600_000
            if gap <= max_gap_hours:
                pairs.append((gap, int(s1_time), str(s1_id), int(dw_time), str(dw_id)))
    if not pairs:
        return None
    _, s1_time, s1_id, dw_time, dw_id = min(pairs)
    return s1_id, s1_time, dw_id, dw_time


def footprint_geometry(ee, lon: float, lat: float):
    return ee.Geometry.Point([lon, lat]).buffer(5500).bounds(maxError=10)


def query_pair_stats(ee, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    features = []
    for row in rows:
        region = footprint_geometry(ee, float(row["centroid_lon"]), float(row["centroid_lat"]))
        s1_time = datetime.fromisoformat(row["s1_datetime_utc"].replace("Z", "+00:00"))
        s1_collection = (
            ee.ImageCollection(S1_COLLECTION).filterBounds(region)
            .filterDate((s1_time - timedelta(minutes=1)).isoformat(), (s1_time + timedelta(minutes=1)).isoformat())
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )
        s1_time_ms = int(s1_time.timestamp() * 1000)
        dw_collection = (
            ee.ImageCollection(DW_COLLECTION).filterBounds(region)
            .filterDate((s1_time - timedelta(hours=24)).isoformat(), (s1_time + timedelta(hours=24)).isoformat())
            .map(lambda image: image.set("pair_distance", ee.Number(image.get("system:time_start")).subtract(s1_time_ms).abs()))
            .sort("pair_distance", False)
        )
        s1 = s1_collection.select(["VV", "VH"]).mosaic()
        dw = dw_collection.select("label").mosaic()
        jrc = ee.Image(JRC_COLLECTION)
        reducer = ee.Reducer.mean()
        s1_valid = s1.mask().reduce(ee.Reducer.min()).unmask(0).rename("s1_valid")
        metrics = ee.Image.cat(
            [
                s1_valid,
                dw.mask().unmask(0).rename("dw_valid"),
                dw.eq(0).unmask(0).rename("dw_water"),
                jrc.select("max_extent").unmask(0).rename("jrc_max"),
                jrc.select("seasonality").eq(12).unmask(0).rename("jrc_permanent"),
                jrc.select("seasonality").gte(1).And(jrc.select("seasonality").lte(11)).unmask(0).rename("jrc_seasonal"),
            ]
        )
        stats = metrics.reduceRegion(reducer, region, 30, maxPixels=5_000_000, tileScale=2)
        features.append(
            ee.Feature(
                region,
                {
                    "candidate_id": row["candidate_id"],
                    "s1_image_ids": s1_collection.aggregate_array("system:index"),
                    "dw_image_ids": dw_collection.aggregate_array("system:index"),
                    "dw_image_times": dw_collection.aggregate_array("system:time_start"),
                },
            ).set(stats)
        )

    result = ee.FeatureCollection(features).getInfo()
    return {feature["properties"]["candidate_id"]: feature["properties"] for feature in result["features"]}


def resilient_pair_stats(ee, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    try:
        return query_pair_stats(ee, rows)
    except Exception:
        if len(rows) <= 1:
            raise
        middle = len(rows) // 2
        result = resilient_pair_stats(ee, rows[:middle])
        result.update(resilient_pair_stats(ee, rows[middle:]))
        return result


def selection_tags(row: dict[str, Any]) -> list[str]:
    tags = [row["primary_target"]]
    if float(row.get("jrc_seasonal_fraction") or 0) >= 0.10 and float(row.get("dw_water_fraction") or 0) >= 0.10:
        tags.append("high_water_seasonal_or_ephemeral")
    if row["source_dataset"] == "GDW_v1.0":
        tags.append("confirmed_reservoir")
    if row["source_dataset"] == "GSHHG_2.3.7":
        tags.append("coastal_or_estuarine")
    return sorted(set(tags))


def resolve_pair_attempt(
    ee,
    rows: list[dict[str, str]],
    batch_size: int,
    max_gap_hours: float,
    min_valid_fraction: float,
) -> list[dict[str, Any]]:
    paired = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        metadata = query_month_metadata(ee, batch)
        for row in batch:
            item = dict(row)
            pair = nearest_pair(metadata.get(row["candidate_id"], {}), max_gap_hours)
            if pair is None:
                item.update({"pair_status": "no_pair_within_limit", "error": ""})
            else:
                s1_id, s1_time, dw_id, dw_time = pair
                signed = (s1_time - dw_time) / 3_600_000
                item.update(
                    {
                        "pair_status": "pair_found",
                        "s1_image_id": s1_id, "s1_datetime_utc": iso_utc(s1_time),
                        "dw_image_id": dw_id, "dw_datetime_utc": iso_utc(dw_time),
                        "s1_minus_dw_hours": "{:.6f}".format(signed),
                        "absolute_time_delta_hours": "{:.6f}".format(abs(signed)), "error": "",
                    }
                )
            paired.append(item)
        print("Resolved metadata for {}/{} candidates".format(min(start + len(batch), len(rows)), len(rows)), flush=True)

    found = [row for row in paired if row["pair_status"] == "pair_found"]
    for start in range(0, len(found), batch_size):
        batch = found[start : start + batch_size]
        stats = resilient_pair_stats(ee, batch)
        for row in batch:
            values = stats.get(row["candidate_id"], {})
            row["s1_valid_fraction"] = "{:.6f}".format(float(values.get("s1_valid") or 0))
            row["dw_valid_fraction"] = "{:.6f}".format(float(values.get("dw_valid") or 0))
            row["s1_image_ids"] = json.dumps(values.get("s1_image_ids", []))
            row["dw_image_ids"] = json.dumps(values.get("dw_image_ids", []))
            dw_times = [int(value) for value in values.get("dw_image_times", [])]
            row["dw_datetimes_utc"] = json.dumps([iso_utc(value) for value in dw_times])
            s1_ms = int(
                datetime.fromisoformat(row["s1_datetime_utc"].replace("Z", "+00:00")).timestamp()
                * 1000
            )
            row["maximum_component_time_delta_hours"] = "{:.6f}".format(
                max((abs(s1_ms - value) / 3_600_000 for value in dw_times), default=0.0)
            )
            row["dw_water_fraction"] = "{:.6f}".format(float(values.get("dw_water") or 0))
            row["jrc_max_extent_fraction"] = "{:.6f}".format(float(values.get("jrc_max") or 0))
            row["jrc_permanent_fraction"] = "{:.6f}".format(float(values.get("jrc_permanent") or 0))
            row["jrc_seasonal_fraction"] = "{:.6f}".format(float(values.get("jrc_seasonal") or 0))
            water = float(row["dw_water_fraction"])
            row["water_fraction_bin"] = "above_40pct" if water > 0.40 else "10_to_40pct" if water >= 0.10 else "1_to_10pct" if water >= 0.01 else "below_1pct"
            row["selection_tags"] = ";".join(selection_tags(row))
            if float(row["s1_valid_fraction"]) < min_valid_fraction or float(row["dw_valid_fraction"]) < min_valid_fraction:
                row["pair_status"] = "insufficient_valid_coverage"
        print("Measured coverage for {}/{} pairs".format(min(start + len(batch), len(found)), len(found)), flush=True)
    return paired


def resolve_pairs(
    ee,
    rows: list[dict[str, str]],
    batch_size: int,
    max_gap_hours: float,
    min_valid_fraction: float,
    attempts: int,
    seed: int = 42,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    final: dict[str, dict[str, Any]] = {}
    pending = [dict(row) for row in rows]
    months = list(NON_OCTOBER_MONTHS)
    for attempt in range(attempts):
        trial_rows = []
        for row in pending:
            item = dict(row)
            ordinal = (stable_score(seed, row["candidate_id"], "pair_window") + attempt * 13) % 44
            item["target_year"] = str(2020 + ordinal // len(months))
            item["target_month"] = str(months[ordinal % len(months)])
            item["pair_attempt"] = attempt + 1
            trial_rows.append(item)
        resolved = resolve_pair_attempt(
            ee, trial_rows, batch_size, max_gap_hours, min_valid_fraction
        )
        next_pending = []
        for row in resolved:
            water = float(row.get("dw_water_fraction") or 0)
            required_water = (
                0.10
                if row["primary_target"] == "high_water_seasonal_or_ephemeral"
                else 0.01
            )
            seasonal_ok = (
                row["primary_target"] != "high_water_seasonal_or_ephemeral"
                or float(row.get("jrc_seasonal_fraction") or 0) >= 0.10
            )
            accepted = (
                row["pair_status"] == "pair_found"
                and water >= required_water
                and seasonal_ok
            )
            if accepted:
                final[row["candidate_id"]] = row
            else:
                if row["pair_status"] == "pair_found":
                    row["pair_status"] = "insufficient_water_exposure"
                previous = final.get(row["candidate_id"])
                if previous is None or water > float(previous.get("dw_water_fraction") or 0):
                    final[row["candidate_id"]] = row
                next_pending.append(next(item for item in pending if item["candidate_id"] == row["candidate_id"]))
        pending = next_pending
        print(
            "Pair attempt {}/{}: {} accepted, {} still pending".format(
                attempt + 1, attempts, len(rows) - len(pending), len(pending)
            ),
            flush=True,
        )
        if checkpoint_path is not None:
            checkpoint_rows = [final[row["candidate_id"]] for row in rows if row["candidate_id"] in final]
            write_csv(checkpoint_path, checkpoint_rows, PAIR_FIELDS)
        if not pending:
            break
    return [final[row["candidate_id"]] for row in rows]


def spatially_distinct(row: dict[str, str], selected: list[dict[str, str]], minimum_km: float) -> bool:
    lon, lat = float(row["centroid_lon"]), float(row["centroid_lat"])
    return all(
        haversine_km(lon, lat, float(other["centroid_lon"]), float(other["centroid_lat"])) >= minimum_km
        for other in selected
    )


def select_rows(
    rows: list[dict[str, str]], quotas: dict[str, int], minimum_km: float, seed: int
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    eligible = [
        row for row in rows
        if row["pair_status"] == "pair_found"
        and float(row["absolute_time_delta_hours"]) <= 24
        and float(row.get("dw_water_fraction") or 0) >= 0.01
        and (
            row["primary_target"] != "high_water_seasonal_or_ephemeral"
            or (
                float(row.get("dw_water_fraction") or 0) >= 0.10
                and float(row.get("jrc_seasonal_fraction") or 0) >= 0.10
            )
        )
    ]
    for target, quota in quotas.items():
        candidates = [row for row in eligible if row["primary_target"] == target]
        candidates.sort(
            key=lambda row: (
                0 if row["hemisphere"] == "south" else 1,
                0 if row["water_fraction_bin"] == "10_to_40pct" else 1 if row["water_fraction_bin"] == "above_40pct" else 2,
                stable_score(seed, row["candidate_id"], "select"),
            )
        )
        chosen = []
        for row in candidates:
            if spatially_distinct(row, selected, minimum_km):
                selected.append(row)
                chosen.append(row)
            if len(chosen) == quota:
                break
        if len(chosen) != quota:
            raise RuntimeError("Only selected {} of {} required {} samples".format(len(chosen), quota, target))
    return selected


def grid_properties(row: dict[str, str]) -> dict[str, Any]:
    lon, lat = float(row["centroid_lon"]), float(row["centroid_lat"])
    zone = min(60, max(1, int((lon + 180) // 6) + 1))
    epsg = (32600 if lat >= 0 else 32700) + zone
    crs = "EPSG:{}".format(epsg)
    xs, ys = transform_coordinates("EPSG:4326", crs, [lon], [lat])
    width = height = 1100
    origin_x, origin_y = xs[0] - 5500, ys[0] + 5500
    corners_x = [origin_x, origin_x + 11000, origin_x + 11000, origin_x, origin_x]
    corners_y = [origin_y, origin_y, origin_y - 11000, origin_y - 11000, origin_y]
    corner_lon, corner_lat = transform_coordinates(crs, "EPSG:4326", corners_x, corners_y)
    return {
        "crs": crs, "width": width, "height": height,
        "transform_a": 10, "transform_b": 0, "transform_c": round(origin_x, 6),
        "transform_d": 0, "transform_e": -10, "transform_f": round(origin_y, 6),
        "footprint": {"type": "Polygon", "coordinates": [list(map(list, zip(corner_lon, corner_lat)))]},
    }


def assign_splits(rows: list[dict[str, str]], seed: int) -> None:
    by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_target[row["primary_target"]].append(row)
    valid_ids = set()
    for target, group in by_target.items():
        group.sort(key=lambda row: stable_score(seed, target, row["group_id"], "split"))
        valid_ids.update(row["candidate_id"] for row in group[: math.ceil(len(group) * 0.20)])
    for row in rows:
        row["split"] = "valid" if row["candidate_id"] in valid_ids else "train"


def write_selection(
    rows: list[dict[str, str]], output: Path, footprints: Path, summary: Path,
    seed: int, preserve_splits: bool = False,
) -> None:
    if not preserve_splits:
        assign_splits(rows, seed)
    elif any(row.get("split") not in {"train", "valid"} for row in rows):
        raise ValueError("Every row must have a train/valid split when preserving splits")
    output_rows = []
    features = []
    for row in rows:
        properties = grid_properties(row)
        item = {key: value for key, value in row.items() if key != "error"}
        item.update({key: value for key, value in properties.items() if key != "footprint"})
        output_rows.append(item)
        feature_properties = dict(item)
        features.append({"type": "Feature", "geometry": properties["footprint"], "properties": feature_properties})
    write_csv(output, output_rows)
    footprints.parent.mkdir(parents=True, exist_ok=True)
    footprints.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    absolute_gaps = [float(row["absolute_time_delta_hours"]) for row in rows]
    component_gaps = [float(row["maximum_component_time_delta_hours"]) for row in rows]
    signed_gaps = [float(row["s1_minus_dw_hours"]) for row in rows]
    pairwise_distances = [
        haversine_km(
            float(row["centroid_lon"]), float(row["centroid_lat"]),
            float(other["centroid_lon"]), float(other["centroid_lat"]),
        )
        for index, row in enumerate(rows)
        for other in rows[:index]
    ]
    seasonal_rows = [
        row for row in rows
        if row["primary_target"] == "high_water_seasonal_or_ephemeral"
    ]
    split_by_target = {
        target: dict(Counter(row["split"] for row in rows if row["primary_target"] == target))
        for target in sorted(set(row["primary_target"] for row in rows))
    }
    report = {
        "selected_samples": len(rows),
        "unique_candidate_ids": len(set(row["candidate_id"] for row in rows)),
        "unique_group_ids": len(set(row["group_id"] for row in rows)),
        "primary_target_counts": dict(Counter(row["primary_target"] for row in rows)),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "split_counts_by_primary_target": split_by_target,
        "hemisphere_counts": dict(Counter(row["hemisphere"] for row in rows)),
        "month_counts": dict(sorted(Counter(row["target_month"] for row in rows).items(), key=lambda item: int(item[0]))),
        "water_fraction_bin_counts": dict(Counter(row["water_fraction_bin"] for row in rows)),
        "s1_dw_timing_hours": {
            "median_signed_s1_minus_dw": statistics.median(signed_gaps),
            "median_absolute": statistics.median(absolute_gaps),
            "maximum_absolute_nearest_pair": max(absolute_gaps),
            "maximum_absolute_mosaic_component": max(component_gaps),
            "s1_earlier": sum(gap < 0 for gap in signed_gaps),
            "same_timestamp": sum(gap == 0 for gap in signed_gaps),
            "s1_later": sum(gap > 0 for gap in signed_gaps),
            "same_utc_date": sum(
                row["s1_datetime_utc"][:10] == row["dw_datetime_utc"][:10]
                for row in rows
            ),
            "within_6_hours": sum(gap <= 6 for gap in component_gaps),
            "within_12_hours": sum(gap <= 12 for gap in component_gaps),
            "within_24_hours": sum(gap <= 24 for gap in component_gaps),
        },
        "coverage": {
            "minimum_s1_valid_fraction": min(float(row["s1_valid_fraction"]) for row in rows),
            "minimum_dw_valid_fraction": min(float(row["dw_valid_fraction"]) for row in rows),
        },
        "minimum_pairwise_centroid_distance_km": min(pairwise_distances),
        "seasonal_primary_minima": {
            "jrc_seasonal_fraction": min(float(row["jrc_seasonal_fraction"]) for row in seasonal_rows),
            "dw_water_fraction": min(float(row["dw_water_fraction"]) for row in seasonal_rows),
        },
        "sources": sorted(set(row["source_dataset"] for row in rows)),
        "source_counts": dict(Counter(row["source_dataset"] for row in rows)),
        "output_csv": str(output.resolve()), "footprints_geojson": str(footprints.resolve()),
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    pool = subparsers.add_parser("pool")
    pool.add_argument("--index", type=Path, default=Path("metadata/training_samples.csv"))
    pool.add_argument("--reservoir-shapefile", type=Path, default=Path("data/reference/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp"))
    pool.add_argument("--coastline-shapefile", type=Path, action="append", default=[])
    pool.add_argument("--output", type=Path, default=Path("outputs/audit/supplement_candidate_pool.csv"))
    pool.add_argument("--ee-project", default="rohit-global-water")
    pool.add_argument("--seed", type=int, default=42)
    pool.add_argument("--existing-distance-km", type=float, default=20)
    pool.add_argument("--lake-pool", type=int, default=700)
    pool.add_argument("--reservoir-pool", type=int, default=450)
    pool.add_argument("--coastal-pool", type=int, default=450)
    pool.add_argument("--seasonal-pool", type=int, default=500)
    pool.add_argument("--pld-per-cell", type=int, default=40)
    pool.add_argument("--workers", type=int, default=8)
    pool.add_argument("--timeout", type=int, default=180)

    seasonal_pool = subparsers.add_parser("seasonal-pool")
    seasonal_pool.add_argument("--index", type=Path, default=Path("metadata/training_samples.csv"))
    seasonal_pool.add_argument(
        "--output", type=Path,
        default=Path("outputs/audit/supplement_seasonal_candidate_pool.csv"),
    )
    seasonal_pool.add_argument("--ee-project", default="rohit-global-water")
    seasonal_pool.add_argument("--seed", type=int, default=142)
    seasonal_pool.add_argument("--existing-distance-km", type=float, default=20)
    seasonal_pool.add_argument("--raw-pool", type=int, default=8000)
    seasonal_pool.add_argument("--limit", type=int, default=800)
    seasonal_pool.add_argument("--batch-size", type=int, default=100)
    seasonal_pool.add_argument("--minimum-seasonal-fraction", type=float, default=0.10)

    pairs = subparsers.add_parser("pairs")
    pairs.add_argument("--pool", type=Path, default=Path("outputs/audit/supplement_candidate_pool.csv"))
    pairs.add_argument("--output", type=Path, default=Path("outputs/audit/supplement_candidate_pairs.csv"))
    pairs.add_argument("--ee-project", default="rohit-global-water")
    pairs.add_argument("--batch-size", type=int, default=75)
    pairs.add_argument("--limit", type=int, default=None)
    pairs.add_argument("--max-gap-hours", type=float, default=24)
    pairs.add_argument("--min-valid-fraction", type=float, default=0.90)
    pairs.add_argument("--attempts", type=int, default=8)
    pairs.add_argument("--seed", type=int, default=42)
    pairs.add_argument("--primary-target", default=None)
    pairs.add_argument(
        "--checkpoint", type=Path,
        default=Path("outputs/audit/supplement_candidate_pairs_checkpoint.csv"),
    )

    select = subparsers.add_parser("select")
    select.add_argument("--pairs", type=Path, action="append", default=[])
    select.add_argument("--output", type=Path, default=Path("outputs/audit/supplement_samples.csv"))
    select.add_argument("--footprints", type=Path, default=Path("outputs/audit/supplement_sample_footprints.geojson"))
    select.add_argument("--summary", type=Path, default=Path("outputs/audit/supplement_samples_summary.json"))
    select.add_argument("--minimum-separation-km", type=float, default=20)
    select.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.stage == "pool":
        existing = read_existing(args.index)
        reservoirs, reservoir_polygons = read_reservoir_candidates(
            args.reservoir_shapefile, existing, args.seed, args.reservoir_pool, args.existing_distance_km
        )
        coast_paths = args.coastline_shapefile or [
            Path("data/reference/GSHHS_shp/i/GSHHS_i_L1.shp"),
            Path("data/reference/GSHHS_shp/i/GSHHS_i_L5.shp"),
        ]
        coasts = read_coastal_candidates(
            coast_paths, existing, args.seed, args.coastal_pool, args.existing_distance_km
        )
        lakes = read_lake_candidates(
            existing, reservoir_polygons, args.seed, args.lake_pool, args.pld_per_cell,
            args.workers, args.existing_distance_km, args.timeout,
        )
        ee = initialize_ee(args.ee_project)
        seasonal = seasonal_candidates(
            ee, existing, args.seed, args.seasonal_pool, args.existing_distance_km
        )
        rows = lakes + reservoirs + coasts + seasonal
        write_csv(args.output, rows)
        print(json.dumps({"candidate_pool": len(rows), "primary_target_counts": dict(Counter(row["primary_target"] for row in rows)), "output_csv": str(args.output.resolve())}, indent=2))
    elif args.stage == "seasonal-pool":
        ee = initialize_ee(args.ee_project)
        existing = read_existing(args.index)
        raw = seasonal_candidates(
            ee, existing, args.seed, args.raw_pool, args.existing_distance_km
        )
        rows = prefilter_seasonal_candidates(
            ee, raw, args.minimum_seasonal_fraction, args.batch_size, args.limit, args.seed
        )
        write_csv(args.output, rows)
        print(
            json.dumps(
                {"raw_candidates": len(raw), "prefiltered_candidates": len(rows), "output_csv": str(args.output.resolve())},
                indent=2,
            )
        )
    elif args.stage == "pairs":
        ee = initialize_ee(args.ee_project)
        pool_rows = read_csv(args.pool)
        if args.primary_target:
            pool_rows = [
                row for row in pool_rows if row["primary_target"] == args.primary_target
            ]
        if args.limit is not None:
            pool_rows = pool_rows[: args.limit]
        rows = resolve_pairs(
            ee, pool_rows, args.batch_size, args.max_gap_hours, args.min_valid_fraction,
            args.attempts, seed=args.seed, checkpoint_path=args.checkpoint,
        )
        write_csv(args.output, rows, PAIR_FIELDS)
        print(json.dumps({"candidates": len(rows), "status_counts": dict(Counter(row["pair_status"] for row in rows)), "output_csv": str(args.output.resolve())}, indent=2))
    else:
        quotas = {
            "non_river_natural_lake": 200,
            "confirmed_reservoir": 125,
            "coastal_or_estuarine": 125,
            "high_water_seasonal_or_ephemeral": 150,
        }
        pair_paths = args.pairs or [Path("outputs/audit/supplement_candidate_pairs.csv")]
        combined: dict[str, dict[str, str]] = {}
        for path in pair_paths:
            for row in read_csv(path):
                current = combined.get(row["candidate_id"])
                if current is None or row["pair_status"] == "pair_found":
                    combined[row["candidate_id"]] = row
        rows = select_rows(list(combined.values()), quotas, args.minimum_separation_km, args.seed)
        write_selection(rows, args.output, args.footprints, args.summary, args.seed)


if __name__ == "__main__":
    main()
