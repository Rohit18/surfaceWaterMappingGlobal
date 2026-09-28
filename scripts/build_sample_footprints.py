#!/usr/bin/env python3
"""Build exact WGS84 sample footprints from the released raster-grid index."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from affine import Affine
from rasterio.warp import transform_geom


def grid_ring(row: dict[str, str]) -> list[list[float]]:
    """Return the four raster-grid corners as a closed WGS84 ring."""
    affine = Affine(*(float(row["transform_{}".format(key)]) for key in "abcdef"))
    width, height = int(row["width"]), int(row["height"])
    corners = [
        affine * (0, 0),
        affine * (width, 0),
        affine * (width, height),
        affine * (0, height),
        affine * (0, 0),
    ]
    source = {"type": "Polygon", "coordinates": [[list(point) for point in corners]]}
    transformed = transform_geom(row["crs"], "EPSG:4326", source, precision=10)
    return [[float(x), float(y)] for x, y in transformed["coordinates"][0]]


def bounds(ring: list[list[float]]) -> tuple[float, float, float, float]:
    xs, ys = zip(*ring)
    return min(xs), min(ys), max(xs), max(ys)


def feature(row: dict[str, str]) -> dict[str, Any]:
    ring = grid_ring(row)
    properties: dict[str, Any] = dict(row)
    for field in ("centroid_lon", "centroid_lat"):
        properties[field] = float(properties[field])
    for field in ("width", "height", "aef_year"):
        properties[field] = int(properties[field])
    for key in "abcdef":
        field = "transform_{}".format(key)
        properties[field] = float(properties[field])
    min_lon, min_lat, max_lon, max_lat = bounds(ring)
    properties.update(
        {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        }
    )
    return {
        "type": "Feature",
        "id": row["tile_name"],
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": properties,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("metadata/training_samples.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/audit/sample_footprints.geojson"))
    parser.add_argument(
        "--bounds-csv",
        type=Path,
        default=Path("outputs/audit/sample_footprint_bounds.csv"),
        help="Flat companion table convenient for spatial databases and QA.",
    )
    args = parser.parse_args()

    with args.index.expanduser().resolve().open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Sample index is empty: {}".format(args.index))

    features = [feature(row) for row in rows]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"type": "FeatureCollection", "features": features}) + "\n")

    bounds_output = args.bounds_csv.expanduser().resolve()
    bounds_output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "tile_name", "sword_node_id", "s1_date", "split", "centroid_lon", "centroid_lat",
        "min_lon", "min_lat", "max_lon", "max_lat", "crs", "width", "height",
    ]
    with bounds_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in features:
            writer.writerow({field: item["properties"][field] for field in fields})

    print(json.dumps({"samples": len(features), "geojson": str(output), "bounds_csv": str(bounds_output)}))


if __name__ == "__main__":
    main()
