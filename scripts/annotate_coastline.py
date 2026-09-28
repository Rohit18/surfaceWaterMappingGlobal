#!/usr/bin/env python3
"""Flag sample footprints within a specified distance of a GSHHG coastline.

The audit uses GSHHG L1 (land/ocean) plus L5 (Antarctic ice/ocean), not its lake
boundaries. Distances are evaluated only inside the requested threshold because the
coverage question is whether a sample is coastal, not its global nearest-coast rank.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import shapefile
from shapely.affinity import scale
from shapely.geometry import Polygon, box, shape
from shapely.strtree import STRtree

from build_sample_footprints import grid_ring


KM_PER_DEGREE = 111.195


def load_boundaries(paths: list[Path]):
    boundaries = []
    for path in paths:
        with shapefile.Reader(str(path.expanduser().resolve())) as reader:
            for item in reader.iterShapes():
                geometry = shape(item.__geo_interface__)
                if not geometry.is_empty:
                    boundaries.append(geometry.boundary)
    if not boundaries:
        raise ValueError("No coastline geometry found")
    return boundaries


def threshold_distance_km(
    footprint: Polygon,
    latitude: float,
    boundaries,
    tree: STRtree,
    threshold_km: float,
) -> float | None:
    """Return distance when within threshold, otherwise None.

    Longitude is scaled by cos(latitude) before planar distance is evaluated. At the
    small threshold used here this local equirectangular approximation is adequate.
    """
    min_lon, min_lat, max_lon, max_lat = footprint.bounds
    latitude_pad = threshold_km / 110.574
    cosine = max(abs(math.cos(math.radians(latitude))), 0.01)
    longitude_pad = threshold_km / (111.320 * cosine)
    search = box(
        min_lon - longitude_pad,
        min_lat - latitude_pad,
        max_lon + longitude_pad,
        max_lat + latitude_pad,
    )
    candidate_indices = tree.query(search)
    if len(candidate_indices) == 0:
        return None
    scaled_footprint = scale(footprint, xfact=cosine, yfact=1.0, origin=(0.0, 0.0))
    minimum = min(
        scaled_footprint.distance(
            scale(boundaries[int(index)], xfact=cosine, yfact=1.0, origin=(0.0, 0.0))
        )
        * KM_PER_DEGREE
        for index in candidate_indices
    )
    return minimum if minimum <= threshold_km else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("metadata/training_samples.csv"))
    parser.add_argument("--coastline-shapefile", type=Path, action="append", required=True)
    parser.add_argument("--threshold-km", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/audit/coastline_annotations.csv"))
    parser.add_argument(
        "--summary", type=Path, default=Path("outputs/audit/coastline_annotations_summary.json")
    )
    parser.add_argument("--source-label", default="GSHHG_2.3.7_intermediate_L1_L5")
    args = parser.parse_args()
    if args.threshold_km <= 0:
        parser.error("--threshold-km must be positive")

    with args.index.expanduser().resolve().open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    boundaries = load_boundaries(args.coastline_shapefile)
    tree = STRtree(boundaries)
    output_rows = []
    for number, row in enumerate(rows, start=1):
        footprint = Polygon(grid_ring(row))
        distance = threshold_distance_km(
            footprint, float(row["centroid_lat"]), boundaries, tree, args.threshold_km
        )
        output_rows.append(
            {
                "tile_name": row["tile_name"],
                "coastline_source": args.source_label,
                "coastline_distance_km": "{:.6f}".format(distance) if distance is not None else "",
                "coastline_distance_lower_bound_km": (
                    "" if distance is not None else "{:.6f}".format(args.threshold_km)
                ),
                "coastal_within_threshold": int(distance is not None),
                "coastal_threshold_km": "{:.6f}".format(args.threshold_km),
            }
        )
        if number % 500 == 0:
            print("Processed {}/{} footprints".format(number, len(rows)), flush=True)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0])
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    coastal = sum(int(row["coastal_within_threshold"]) for row in output_rows)
    summary = {
        "samples": len(output_rows),
        "coastal_within_threshold": coastal,
        "noncoastal_beyond_threshold": len(output_rows) - coastal,
        "threshold_km_from_footprint": args.threshold_km,
        "source": args.source_label,
        "source_shapefiles": [str(path.expanduser().resolve()) for path in args.coastline_shapefile],
        "method": "GSHHG boundary spatial index; local equirectangular distance",
        "output_csv": str(output),
    }
    summary_path = args.summary.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
