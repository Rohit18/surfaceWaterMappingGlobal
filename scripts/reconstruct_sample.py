#!/usr/bin/env python3
"""Reconstruct one indexed S1/AEF/Dynamic World training triplet.

Requires authenticated Google Earth Engine access and network access to Source
Cooperative. The command records resolved source IDs/URLs beside the outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio as rio
import requests
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds
from rasterio.warp import reproject, transform_bounds


S1_COLLECTION = "COPERNICUS/S1_GRD"
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
AEF_INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.csv"
AEF_PREFIX = "https://data.source.coop/tge-labs/aef/"


def read_sample(path: Path, tile_name: str) -> dict[str, str]:
    with path.open(newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if row["tile_name"] == tile_name]
    if len(matches) != 1:
        raise ValueError("Expected one index row for {}, found {}".format(tile_name, len(matches)))
    return matches[0]


def grid(row: dict[str, str]) -> tuple[str, Affine, int, int, tuple[float, float, float, float]]:
    affine = Affine(*(float(row["transform_{}".format(key)]) for key in "abcdef"))
    width, height = int(row["width"]), int(row["height"])
    bounds = array_bounds(height, width, affine)
    return row["crs"], affine, width, height, bounds


def polygon(bounds: tuple[float, float, float, float]) -> dict:
    left, bottom, right, top = bounds
    return {
        "type": "Polygon",
        "coordinates": [[[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]],
    }


def initialize_ee(project: str | None):
    try:
        import ee
        ee.Initialize(project=project)
        return ee
    except Exception as exc:
        raise SystemExit("Earth Engine is not ready. Run `earthengine authenticate`: {}".format(exc)) from exc


def collection_ids(collection, collection_name: str) -> list[tuple[int, str]]:
    indices = collection.aggregate_array("system:index").getInfo()
    times = collection.aggregate_array("system:time_start").getInfo()
    return sorted((int(timestamp), "{}/{}".format(collection_name, index)) for timestamp, index in zip(times, indices))


def resolve_ee_images(ee, row: dict[str, str], region, min_dw_valid_fraction: float):
    date = datetime.strptime(row["s1_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    next_day = date + timedelta(days=1)
    s1 = (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(region)
        .filterDate(date.date().isoformat(), next_day.date().isoformat())
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH", "angle"])
        .sort("system:time_start")
    )
    s1_ids = collection_ids(s1, S1_COLLECTION)
    if not s1_ids:
        raise RuntimeError("No dual-polarization IW S1 scene found on {}".format(row["s1_date"]))

    dw_start = date - timedelta(days=1)
    dw_end = date + timedelta(days=2)
    dw = (
        ee.ImageCollection(DW_COLLECTION)
        .filterBounds(region)
        .filterDate(dw_start.date().isoformat(), dw_end.date().isoformat())
        .select(["label"])
        .sort("system:time_start")
    )
    dw_ids = collection_ids(dw, DW_COLLECTION)
    if not dw_ids:
        raise RuntimeError("No Dynamic World image within +/-1 day of {}".format(row["s1_date"]))
    target_ms = int(date.timestamp() * 1000)
    _, selected_dw = min(dw_ids, key=lambda item: (abs(item[0] - target_ms), item[0], item[1]))

    s1_image = s1.mosaic().select(["VV", "VH", "angle"]).float().clip(region)
    dw_label = ee.Image(selected_dw).select("label").clip(region)
    valid_fraction_result = (
        dw_label.mask().rename("valid").unmask(0)
        .reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1_000_000_000)
        .getInfo()
    )
    valid_fraction = float(valid_fraction_result.get("valid") or 0.0)
    if valid_fraction < min_dw_valid_fraction:
        raise RuntimeError(
            "Dynamic World valid fraction {:.6f} is below {:.6f}".format(
                valid_fraction, min_dw_valid_fraction
            )
        )
    water = dw_label.eq(0).unmask(0, sameFootprint=False).uint8().rename("water").clip(region)
    return s1_image, water, [image_id for _, image_id in s1_ids], selected_dw, valid_fraction


def download_ee(image, output: Path, crs: str, affine: Affine, width: int, height: int, timeout: int) -> None:
    params = {
        "name": output.stem,
        "crs": crs,
        "crs_transform": list(affine)[:6],
        "dimensions": [width, height],
        "format": "ZIPPED_GEO_TIFF",
        "filePerBand": False,
    }
    response = requests.get(image.getDownloadURL(params), timeout=timeout)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith((".tif", ".tiff"))]
        if not members:
            raise RuntimeError("Earth Engine response contained no GeoTIFF")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", dir=output.parent, delete=False) as temporary:
            temporary.write(archive.read(members[0]))
            temporary_path = Path(temporary.name)
        temporary_path.replace(output)


def download_aef_index(cache_dir: Path, timeout: int) -> Path:
    path = cache_dir / "aef_index.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(AEF_INDEX_URL, timeout=timeout)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def first(row: dict[str, str], keys: tuple[str, ...]) -> str:
    return next((str(row[key]) for key in keys if row.get(key)), "")


def aef_url(value: str) -> str:
    if value.startswith("s3://us-west-2.opendata.source.coop/tge-labs/aef/"):
        value = value.replace("s3://us-west-2.opendata.source.coop/tge-labs/aef/", AEF_PREFIX, 1)
    elif value.startswith("v1/"):
        value = AEF_PREFIX + value
    elif not value.startswith("http"):
        value = AEF_PREFIX + value.lstrip("/")
    return value[:-4] + ".tiff" if value.endswith(".vrt") else value


def overlapping_aef_urls(
    index_path: Path,
    year: int,
    bbox_wgs84: tuple[float, float, float, float],
    crs: str,
) -> list[str]:
    candidates = []
    with index_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("year", "").strip() != str(year):
                continue
            row_crs = first(row, ("crs", "proj:epsg", "epsg"))
            if row_crs.isdigit():
                row_crs = "EPSG:" + row_crs
            try:
                tile = tuple(
                    float(first(row, keys))
                    for keys in (
                        ("wgs84_west", "west", "xmin"), ("wgs84_south", "south", "ymin"),
                        ("wgs84_east", "east", "xmax"), ("wgs84_north", "north", "ymax"),
                    )
                )
            except ValueError:
                continue
            overlaps = tile[0] < bbox_wgs84[2] and tile[2] > bbox_wgs84[0] and tile[1] < bbox_wgs84[3] and tile[3] > bbox_wgs84[1]
            location = first(row, ("location", "path", "href", "url", "filename"))
            if overlaps and location:
                candidates.append((row_crs.upper() == crs.upper(), aef_url(location)))
    preferred = [url for same_crs, url in candidates if same_crs]
    return sorted(set(preferred or [url for _, url in candidates]))


def write_aef(
    output: Path,
    urls: list[str],
    crs: str,
    affine: Affine,
    width: int,
    height: int,
) -> None:
    if not urls:
        raise RuntimeError("No AEF source tiles overlap the indexed grid")
    data = np.full((64, height, width), -128, dtype=np.int8)
    for url in urls:
        with rio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif"):
            with rio.open(url) as source:
                for band in range(1, min(source.count, 64) + 1):
                    reproject(
                        rio.band(source, band), data[band - 1],
                        src_transform=source.transform, src_crs=source.crs, src_nodata=-128,
                        dst_transform=affine, dst_crs=crs, dst_nodata=-128,
                        resampling=Resampling.nearest, init_dest_nodata=False,
                    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(
        output, "w", driver="GTiff", height=height, width=width, count=64,
        crs=crs, transform=affine, dtype="int8", nodata=-128, compress="deflate",
        tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER",
    ) as destination:
        destination.write(data)
        destination.descriptions = tuple("A{:02d}".format(index) for index in range(64))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("metadata/training_samples.csv"))
    parser.add_argument("--tile-name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--ee-project", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--min-dw-valid-fraction", type=float, default=0.90)
    args = parser.parse_args()
    if not 0.0 <= args.min_dw_valid_fraction <= 1.0:
        parser.error("--min-dw-valid-fraction must be in [0, 1]")

    row = read_sample(args.index, args.tile_name)
    crs, affine, width, height, bounds = grid(row)
    bbox_wgs84 = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    ee = initialize_ee(args.ee_project)
    region = ee.Geometry(polygon(bbox_wgs84))
    s1_image, water_image, s1_ids, dw_id, dw_valid_fraction = resolve_ee_images(
        ee, row, region, args.min_dw_valid_fraction
    )

    root = args.output_root.expanduser().resolve()
    s1_path, label_path, aef_path = root / "s1" / args.tile_name, root / "labels" / args.tile_name, root / "aef" / args.tile_name
    download_ee(s1_image, s1_path, crs, affine, width, height, args.timeout)
    download_ee(water_image, label_path, crs, affine, width, height, args.timeout)
    urls = overlapping_aef_urls(
        download_aef_index(args.cache_dir.expanduser().resolve(), args.timeout),
        int(row["aef_year"]), bbox_wgs84, crs,
    )
    write_aef(aef_path, urls, crs, affine, width, height)

    provenance = {
        "tile_name": args.tile_name,
        "s1_image_ids": s1_ids,
        "dynamic_world_image_id": dw_id,
        "dynamic_world_valid_fraction": dw_valid_fraction,
        "aef_year": int(row["aef_year"]),
        "aef_source_urls": urls,
        "sha256": {"s1": sha256(s1_path), "label": sha256(label_path), "aef": sha256(aef_path)},
    }
    provenance_path = root / "provenance" / (Path(args.tile_name).stem + ".json")
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
