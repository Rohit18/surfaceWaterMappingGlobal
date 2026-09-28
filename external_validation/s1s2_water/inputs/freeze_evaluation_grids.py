#!/usr/bin/env python3
"""Create the method-independent 30 m grid manifest for all test scenes."""

import csv
import math
import pathlib

import rasterio


ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSET_TABLE = ROOT / "evaluation" / "benchmark_assets.csv"
OUTPUT_TABLE = ROOT / "evaluation" / "grids_30m.csv"
RESOLUTION = 30.0


def snap_down(value):
    quotient = value / RESOLUTION
    nearest = round(quotient)
    if abs(quotient - nearest) < 1e-7:
        return nearest * RESOLUTION
    return math.floor(quotient) * RESOLUTION


def snap_up(value):
    quotient = value / RESOLUTION
    nearest = round(quotient)
    if abs(quotient - nearest) < 1e-7:
        return nearest * RESOLUTION
    return math.ceil(quotient) * RESOLUTION


def main():
    with ASSET_TABLE.open(newline="", encoding="utf-8") as source:
        masks = {
            int(row["scene_id"]): pathlib.Path(row["path"])
            for row in csv.DictReader(source)
            if row["asset"] == "msk"
        }

    rows = []
    for scene_id, path in sorted(masks.items()):
        with rasterio.open(path) as dataset:
            if dataset.crs is None or not dataset.crs.is_projected:
                raise ValueError(f"Scene {scene_id} does not have a projected CRS")
            left = snap_down(dataset.bounds.left)
            bottom = snap_down(dataset.bounds.bottom)
            right = snap_up(dataset.bounds.right)
            top = snap_up(dataset.bounds.top)
            width = int(round((right - left) / RESOLUTION))
            height = int(round((top - bottom) / RESOLUTION))
            rows.append(
                {
                    "scene_id": scene_id,
                    "crs": dataset.crs.to_string(),
                    "resolution_m": int(RESOLUTION),
                    "left": left,
                    "bottom": bottom,
                    "right": right,
                    "top": top,
                    "width": width,
                    "height": height,
                    "reference_mask": str(path),
                    "reference_valid": str(path).replace("_msk.tif", "_valid.tif"),
                    "reference_valid_fraction_min": 0.5,
                    "reference_water_fraction_primary": 0.5,
                }
            )

    with OUTPUT_TABLE.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} frozen grids to {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
