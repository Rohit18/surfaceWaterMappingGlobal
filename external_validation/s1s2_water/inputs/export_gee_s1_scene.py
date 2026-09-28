#!/usr/bin/env python3
"""Export one scene's Sentinel-1 model input from Earth Engine COPERNICUS/S1_GRD.

This reproduces the paper's input pipeline (surfaceWaterMappingGlobal/scripts/reconstruct_sample.py):
VV, VH and angle from COPERNICUS/S1_GRD, mosaicked, cast to float, requested on a fixed 10 m grid
with Earth Engine's default (nearest-neighbour) resampling. Differences from the paper pipeline:
the exact benchmark GRD product IDs are used instead of a same-day date filter, because several
S1S2-Water scenes are mosaics of two dates, and they are mosaicked in the benchmark's recorded
source order (later sources on top). Pixels without S1 data are written as NaN, never 0.

The 10 m grid covers the benchmark S1 raster footprint in its CRS, with the same upper-left corner.
"""

import argparse
import concurrent.futures
import hashlib
import io
import json
import math
import pathlib
import time

import numpy as np
import rasterio
import requests
from rasterio.transform import Affine
from rasterio.windows import Window


ROOT = pathlib.Path(__file__).resolve().parents[1]
GRD_MANIFEST = ROOT / "data/model/grd_manifest.json"
EE_PROJECT = "163583997904"
PIXEL = 10.0
TILE = 1830  # 1830 x 1830 x 3 bands x float32 = 40.2 MB, below Earth Engine's 48 MiB request limit
FILL = -9999.0


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_tile(image, crs, transform, row_off, col_off, height, width, retries=6):
    tile_transform = transform * Affine.translation(col_off, row_off)
    params = {"crs": crs, "crs_transform": list(tile_transform)[:6],
              "dimensions": [width, height], "format": "GEO_TIFF"}
    for attempt in range(retries):
        try:
            response = requests.get(image.getDownloadURL(params), timeout=600)
            response.raise_for_status()
            with rasterio.open(io.BytesIO(response.content)) as source:
                data = source.read().astype("float32")
                if data.shape != (3, height, width):
                    raise RuntimeError(f"tile shape {data.shape} != {(3, height, width)}")
                if not np.allclose(list(source.transform)[:6], list(tile_transform)[:6]):
                    raise RuntimeError(f"tile transform {source.transform} != {tile_transform}")
            return row_off, col_off, data
        except Exception as exc:  # network or Earth Engine throttling
            if attempt == retries - 1:
                raise
            wait = 10 * 2 ** attempt
            print(f"tile ({row_off},{col_off}) attempt {attempt + 1} failed: {exc}; retry in {wait}s", flush=True)
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import ee
    ee.Initialize(project=EE_PROJECT)

    records = sorted((r for r in json.loads(GRD_MANIFEST.read_text())["records"]
                      if r["scene_id"] == args.scene), key=lambda r: r["source_order"])
    if not records:
        raise RuntimeError(f"Scene {args.scene} has no resolved GRD lineage")
    metadata = json.loads((ROOT / f"data/s1s2_water/metadata/{args.scene}/sentinel12_{args.scene}_meta.json").read_text())
    date = metadata["properties"]["date_s1"]
    date_iso = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    benchmark = ROOT / f"data/s1s2_water/test/{args.scene}/sentinel12_s1_{args.scene}_img.tif"
    output_dir = ROOT / f"work/model/scene_{args.scene}/inputs_gee10m"
    output = output_dir / f"s1_{date_iso}.tif"
    manifest_path = output_dir / "s1_gee_manifest.json"
    if output.exists() and manifest_path.exists() and not args.overwrite:
        print(f"verified output already exists: {output}", flush=True)
        return

    with rasterio.open(benchmark) as reference:
        crs = reference.crs.to_string()
        left, top = reference.transform.c, reference.transform.f
        width = int(round(reference.width * reference.transform.a / PIXEL))
        height = int(round(reference.height * -reference.transform.e / PIXEL))
    transform = Affine(PIXEL, 0.0, left, 0.0, -PIXEL, top)

    image_ids = [f"COPERNICUS/S1_GRD/{r['product_id']}" for r in records]
    images = [ee.Image(image_id).select(["VV", "VH", "angle"]) for image_id in image_ids]
    image = (ee.ImageCollection(images).mosaic().select(["VV", "VH", "angle"]).float()
             .unmask(FILL, False))

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tif.part")
    profile = {"driver": "GTiff", "width": width, "height": height, "count": 3, "dtype": "float32",
               "crs": crs, "transform": transform, "nodata": float("nan"), "compress": "deflate",
               "predictor": 3, "tiled": True, "blockxsize": 512, "blockysize": 512, "BIGTIFF": "IF_SAFER"}
    windows = [(r, c, min(TILE, height - r), min(TILE, width - c))
               for r in range(0, height, TILE) for c in range(0, width, TILE)]
    started = time.time()
    valid = 0
    with rasterio.open(temporary, "w", **profile) as destination:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(fetch_tile, image, crs, transform, *w) for w in windows]
            for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                row_off, col_off, data = future.result()
                missing = np.any(data == FILL, axis=0) | ~np.all(np.isfinite(data), axis=0)
                data[:, missing] = np.nan
                valid += int((~missing).sum())
                destination.write(data, window=Window(col_off, row_off, data.shape[2], data.shape[1]))
                print(f"scene {args.scene}: tile {done}/{len(windows)} ({time.time() - started:.0f}s)", flush=True)
        destination.descriptions = ("VV", "VH", "angle")
        destination.update_tags(source="Earth Engine COPERNICUS/S1_GRD", image_ids=json.dumps(image_ids),
                                mosaic_rule="ee.ImageCollection(images in source_order).mosaic(); later on top",
                                resampling="Earth Engine default (nearest neighbour)",
                                units="VV/VH dB as stored in COPERNICUS/S1_GRD; angle degrees")
    temporary.replace(output)

    with rasterio.open(output) as check:
        stats = {}
        for band, name in enumerate(("VV", "VH", "angle"), start=1):
            values = check.read(band, out_shape=(check.height // 10, check.width // 10))
            values = values[np.isfinite(values)]
            stats[name] = {"p01": float(np.percentile(values, 1)), "median": float(np.median(values)),
                           "p99": float(np.percentile(values, 99))} if values.size else None
    report = {"scene_id": args.scene, "date_s1": date_iso, "benchmark": str(benchmark),
              "image_ids": image_ids, "source_order": [r["source_order"] for r in records],
              "output": str(output), "sha256": sha256(output),
              "grid": {"crs": crs, "transform": list(transform)[:6], "width": width, "height": height},
              "valid_pixels": valid, "valid_fraction": valid / (width * height),
              "band_stats_decimated": stats, "tiles": len(windows),
              "elapsed_seconds": round(time.time() - started, 1)}
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
