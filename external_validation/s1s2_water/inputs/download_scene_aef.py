#!/usr/bin/env python3
"""Stream one scene's annual AlphaEarth tiles onto its authoritative S1 grid."""

import argparse
import hashlib
import json
import pathlib

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENE_MANIFEST = ROOT / "data/alphaearth/aef_scene_manifest.json"
INDEX = ROOT / "data/alphaearth/cache/aef_index.csv"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--reference", type=pathlib.Path, default=None,
                        help="raster defining the output grid (default: the benchmark S1 raster)")
    parser.add_argument("--output-dir", type=pathlib.Path, default=None,
                        help="default: work/model/scene_<id>/inputs")
    parser.add_argument("--pixel-size", type=float, default=None,
                        help="resample the reference grid to this pixel size, keeping its CRS, "
                             "upper-left corner and extent (same rule as export_gee_s1_scene.py)")
    args = parser.parse_args()
    manifest = json.loads(SCENE_MANIFEST.read_text())["scenes"][str(args.scene)]
    reference_path = args.reference or pathlib.Path(manifest["reference"])
    output_dir = args.output_dir or ROOT / f"work/model/scene_{args.scene}/inputs"
    output = output_dir / f"alphaearth_{manifest['year']}.tif"
    report_path = output_dir / f"alphaearth_{manifest['year']}_manifest.json"
    if output.exists() and report_path.exists():
        print(f"verified output already exists: {output}", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tif.part")
    urls = [tile["url"] for tile in manifest["tiles"]]

    with rasterio.open(reference_path) as reference:
        profile = reference.profile.copy()
        if args.pixel_size:
            profile.update(
                transform=rasterio.Affine(args.pixel_size, 0.0, reference.transform.c,
                                          0.0, -args.pixel_size, reference.transform.f),
                width=int(round(reference.width * reference.transform.a / args.pixel_size)),
                height=int(round(reference.height * -reference.transform.e / args.pixel_size)))
        profile.update(count=64, dtype="int8", nodata=-128, compress="zstd", predictor=1,
                       interleave="band", tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")
        with rasterio.open(temporary, "w", **profile) as destination:
            fill = np.full((64, 512, 512), -128, dtype="int8")
            for _, window in destination.block_windows(1):
                destination.write(fill[:, : int(window.height), : int(window.width)], window=window)
            for tile_number, url in enumerate(urls, start=1):
                print(f"scene {args.scene} tile {tile_number}/{len(urls)}: {url}", flush=True)
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                                  CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif",
                                  GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3"):
                    with rasterio.open(url) as source:
                        if source.count != 64 or source.dtypes[0] != "int8":
                            raise RuntimeError(f"Unexpected AlphaEarth source layout: {source.profile}")
                        for band in range(1, 65):
                            reproject(source=rasterio.band(source, band),
                                      destination=rasterio.band(destination, band),
                                      src_transform=source.transform, src_crs=source.crs, src_nodata=-128,
                                      dst_transform=destination.transform, dst_crs=destination.crs, dst_nodata=-128,
                                      resampling=Resampling.nearest, init_dest_nodata=False, num_threads=4)
            destination.descriptions = tuple(f"A{index:02d}" for index in range(64))
            destination.update_tags(source="AlphaEarth Foundations annual embeddings v1",
                                    year=str(manifest["year"]), resampling="nearest",
                                    quantization="raw signed int8; -128 nodata")
    temporary.replace(output)

    ranges = []
    valid_count = None
    with rasterio.open(output) as source:
        for band in range(1, 65):
            minimum, maximum, count = 127, -127, 0
            for _, window in source.block_windows(band):
                data = source.read(band, window=window)
                valid = data != -128
                count += int(valid.sum())
                if valid.any():
                    minimum = min(minimum, int(data[valid].min()))
                    maximum = max(maximum, int(data[valid].max()))
            ranges.append([minimum, maximum])
            if valid_count is None:
                valid_count = count
        report = {
            "scene_id": args.scene, "year": manifest["year"], "output": str(output),
            "sha256": sha256(output), "index_sha256": sha256(INDEX), "sources": urls,
            "grid_reference": str(reference_path), "pixel_size_override": args.pixel_size,
            "grid": {"crs": source.crs.to_string(), "transform": list(source.transform),
                     "width": source.width, "height": source.height, "bands": source.count,
                     "dtype": source.dtypes[0], "nodata": source.nodata},
            "pixel_counts_band_1": {"valid": valid_count,
                                    "nodata": source.width * source.height - valid_count},
            "band_ranges": ranges,
        }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
