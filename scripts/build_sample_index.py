#!/usr/bin/env python3
"""Build a public reconstruction index from the local paper training tiles."""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import rasterio as rio
from rasterio.warp import transform


NAME_RE = re.compile(r"^S1_(\d{8})_(\d+)\.tif$")
FIELDS = (
    "tile_name", "s1_date", "sword_node_id", "aef_year", "split",
    "centroid_lon", "centroid_lat", "crs", "width", "height",
    "transform_a", "transform_b", "transform_c", "transform_d", "transform_e", "transform_f",
)


def aef_index(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*.tif")):
        if path.name in result:
            raise RuntimeError("Duplicate AEF filename: {}".format(path.name))
        result[path.name] = path
    return result


def paper_split(names: list[str], valid_pct: float, seed: int) -> dict[str, str]:
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    valid_count = max(1, int(round(len(shuffled) * valid_pct)))
    valid = set(shuffled[:valid_count])
    return {name: ("valid" if name in valid else "train") for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-dir", type=Path, required=True)
    parser.add_argument("--aef-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("metadata/training_samples.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-pct", type=float, default=0.2)
    parser.add_argument(
        "--validation-metrics",
        type=Path,
        default=None,
        help="Optional tile-metrics CSV used to verify the reconstructed validation split.",
    )
    args = parser.parse_args()

    s1_paths = sorted(args.s1_dir.expanduser().resolve().glob("*.tif"))
    if not s1_paths:
        raise FileNotFoundError("No S1 GeoTIFFs found in {}".format(args.s1_dir))
    names = [path.name for path in s1_paths]
    splits = paper_split(names, args.valid_pct, args.seed)
    aef_paths = aef_index(args.aef_dir.expanduser().resolve())

    if args.validation_metrics is not None:
        with args.validation_metrics.open(newline="") as handle:
            recorded = {row["tile_name"] for row in csv.DictReader(handle)}
        reconstructed = {name for name, split in splits.items() if split == "valid"}
        if recorded != reconstructed:
            raise RuntimeError(
                "Validation split mismatch: reconstructed-only={}, recorded-only={}".format(
                    len(reconstructed - recorded), len(recorded - reconstructed)
                )
            )

    rows = []
    for path in s1_paths:
        match = NAME_RE.match(path.name)
        if not match:
            raise ValueError("Unexpected tile name: {}".format(path.name))
        aef_path = aef_paths.get(path.name)
        if aef_path is None:
            raise FileNotFoundError("Missing AEF tile for {}".format(path.name))
        with rio.open(path) as src:
            if src.crs is None:
                raise ValueError("Missing CRS: {}".format(path))
            center_x = (src.bounds.left + src.bounds.right) / 2.0
            center_y = (src.bounds.bottom + src.bounds.top) / 2.0
            lon, lat = transform(src.crs, "EPSG:4326", [center_x], [center_y])
            affine = src.transform
            rows.append(
                {
                    "tile_name": path.name,
                    "s1_date": "{}-{}-{}".format(match.group(1)[:4], match.group(1)[4:6], match.group(1)[6:]),
                    "sword_node_id": match.group(2),
                    "aef_year": aef_path.parent.name,
                    "split": splits[path.name],
                    "centroid_lon": "{:.8f}".format(lon[0]),
                    "centroid_lat": "{:.8f}".format(lat[0]),
                    "crs": src.crs.to_string(),
                    "width": src.width,
                    "height": src.height,
                    "transform_a": "{:.12g}".format(affine.a),
                    "transform_b": "{:.12g}".format(affine.b),
                    "transform_c": "{:.12g}".format(affine.c),
                    "transform_d": "{:.12g}".format(affine.d),
                    "transform_e": "{:.12g}".format(affine.e),
                    "transform_f": "{:.12g}".format(affine.f),
                }
            )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "valid")}
    print({"output": str(output), "samples": len(rows), **counts})


if __name__ == "__main__":
    main()
