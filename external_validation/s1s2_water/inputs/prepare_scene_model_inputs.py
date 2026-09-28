#!/usr/bin/env python3
"""Build one S1 model stack from benchmark VV/VH and exact GRD annotations."""

import argparse
import hashlib
import json
import pathlib
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window
from scipy.interpolate import LinearNDInterpolator


ROOT = pathlib.Path(__file__).resolve().parents[1]
GRD_MANIFEST = ROOT / "data/model/grd_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_points(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist()
                   if name.count("/annotation/") == 1
                   and "/annotation/calibration/" not in name
                   and "-vv-" in name.lower() and name.endswith("-001.xml")]
        if len(members) != 1:
            raise RuntimeError(f"Expected one VV product annotation in {zip_path}, found {members}")
        root = ET.fromstring(archive.read(members[0]))
    points = root.findall(".//geolocationGridPoint")
    return {
        "annotation": members[0],
        "longitude": np.array([float(point.findtext("longitude")) for point in points]),
        "latitude": np.array([float(point.findtext("latitude")) for point in points]),
        "incidence": np.array([float(point.findtext("incidenceAngle")) for point in points]),
        "point_count": len(points),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    records = [record for record in json.loads(GRD_MANIFEST.read_text())["records"]
               if record["scene_id"] == args.scene]
    records.sort(key=lambda record: record["source_order"])
    if not records:
        raise RuntimeError(f"Scene {args.scene} has no resolved GRD lineage")
    metadata_path = ROOT / f"data/s1s2_water/metadata/{args.scene}/sentinel12_{args.scene}_meta.json"
    metadata = json.loads(metadata_path.read_text())
    date = metadata["properties"]["date_s1"]
    date_iso = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    benchmark = ROOT / f"data/s1s2_water/test/{args.scene}/sentinel12_s1_{args.scene}_img.tif"
    output_dir = ROOT / f"work/model/scene_{args.scene}/inputs"
    angle_path = output_dir / "angle_from_grd_annotation.tif"
    stack_path = output_dir / f"s1_{date_iso}.tif"
    manifest_path = output_dir / "input_manifest.json"
    if not args.overwrite and angle_path.exists() and stack_path.exists() and manifest_path.exists():
        print(f"scene {args.scene} inputs already exist", flush=True)
        return

    sources = []
    with rasterio.open(benchmark) as reference:
        for record in records:
            zip_path = ROOT / f"data/model/scene_{args.scene}/{record['file_name']}"
            if not zip_path.exists():
                raise FileNotFoundError(zip_path)
            points = annotation_points(zip_path)
            point_x, point_y = transform("EPSG:4326", reference.crs,
                                         points["longitude"], points["latitude"])
            interpolator = LinearNDInterpolator(np.column_stack([point_x, point_y]),
                                                points["incidence"], fill_value=np.nan)
            sources.append({"record": record, "zip_path": zip_path, "points": points,
                            "interpolator": interpolator})
        profile = reference.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate", tiled=True,
                       blockxsize=512, blockysize=512)
        output_dir.mkdir(parents=True, exist_ok=True)
        valid_pixels = 0
        with rasterio.open(angle_path, "w", **profile) as destination:
            for row_off in range(0, reference.height, 256):
                height = min(256, reference.height - row_off)
                rows = np.arange(row_off, row_off + height, dtype="float64") + 0.5
                cols = np.arange(reference.width, dtype="float64") + 0.5
                col_grid, row_grid = np.meshgrid(cols, rows)
                x, y = reference.transform * (col_grid, row_grid)
                values = np.full(x.shape, np.nan, dtype="float64")
                for source in sources:
                    candidate = source["interpolator"](x, y)
                    covered = np.isfinite(candidate)
                    values[covered] = candidate[covered]
                valid_pixels += int(np.isfinite(values).sum())
                destination.write(values.astype("float32"), 1,
                                  window=Window(0, row_off, reference.width, height))
            destination.set_band_description(1, "angle")
            destination.update_tags(method="linear interpolation inside exact SAFE geolocation grids; later recorded sources overwrite overlap",
                                    source_order=json.dumps([record["product_id"] for record in records]))

    with rasterio.open(benchmark) as source, rasterio.open(angle_path) as angle_source:
        profile = source.profile.copy()
        profile.update(count=3, dtype="float32", nodata=np.nan, compress="deflate", tiled=True,
                       blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        with rasterio.open(stack_path, "w", **profile) as destination:
            for _, window in source.block_windows(1):
                destination.write(source.read((1, 2), window=window).astype("float32") / 100.0,
                                  indexes=(1, 2), window=window)
                destination.write(angle_source.read(1, window=window), indexes=3, window=window)
            destination.descriptions = ("VV", "VH", "angle")
            destination.update_tags(radar_source=benchmark.name,
                                    radar_scale="stored signed integer divided by 100 to restore dB",
                                    angle_source=angle_path.name)

    total_pixels = profile["width"] * profile["height"]
    report = {
        "scene_id": args.scene, "date_s1": date_iso, "benchmark": str(benchmark),
        "angle": str(angle_path), "angle_sha256": sha256(angle_path),
        "s1_stack": str(stack_path), "s1_stack_sha256": sha256(stack_path),
        "angle_valid_pixels": valid_pixels, "angle_total_pixels": total_pixels,
        "angle_coverage": valid_pixels / total_pixels,
        "mosaic_rule": "recorded s1_srcids order; later source overwrites finite overlap",
        "sources": [{"product_id": source["record"]["product_id"],
                     "source_order": source["record"]["source_order"],
                     "zip": str(source["zip_path"]), "zip_sha256": sha256(source["zip_path"]),
                     "zip_md5_expected": source["record"]["md5"],
                     "annotation": source["points"]["annotation"],
                     "annotation_point_count": source["points"]["point_count"]}
                    for source in sources],
    }
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
