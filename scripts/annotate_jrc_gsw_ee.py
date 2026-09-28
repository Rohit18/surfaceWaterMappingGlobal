#!/usr/bin/env python3
"""Annotate sample footprints with JRC Global Surface Water summary fractions.

This is a metadata/statistics operation in Earth Engine. It does not export imagery.
Fractions are computed over complete sample footprints at 30 m.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from build_sample_footprints import grid_ring


COLLECTION = "JRC/GSW1_4/GlobalSurfaceWater"
OUTPUT_FIELDS = (
    "tile_name", "jrc_gsw_source", "jrc_max_extent_fraction",
    "jrc_permanent_fraction", "jrc_seasonal_fraction",
)


def initialize_ee(project: str | None):
    try:
        import ee
        # The optional deprecated-assets STAC file is hosted separately on Google
        # Storage and can hang on restricted HPC nodes. It is not needed here.
        ee.deprecation.deprecated_assets = {"__skip_optional_catalog__": None}
        ee.Initialize(project=project)
        return ee
    except Exception as exc:
        raise SystemExit(
            "Earth Engine is not ready. On a remote shell without gcloud, run "
            "`earthengine authenticate --auth_mode=notebook --force`: {}".format(exc)
        ) from exc


def chunks(values, size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def reduce_batch(ee, rows: list[dict[str, str]], scale: int, tile_scale: int):
    seasonality = ee.Image(COLLECTION).select("seasonality").unmask(0)
    image = ee.Image.cat(
        ee.Image(COLLECTION).select("max_extent").eq(1).unmask(0).rename("jrc_max_extent_fraction"),
        seasonality.eq(12).rename("jrc_permanent_fraction"),
        seasonality.gt(0).And(seasonality.lt(12)).rename("jrc_seasonal_fraction"),
    )
    features = [
        ee.Feature(
            ee.Geometry.Polygon([grid_ring(row)], proj="EPSG:4326", geodesic=False),
            {"tile_name": row["tile_name"]},
        )
        for row in rows
    ]
    response = image.reduceRegions(
        collection=ee.FeatureCollection(features),
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=tile_scale,
    ).getInfo()
    output = []
    for feature in response["features"]:
        properties = feature["properties"]
        row = {field: "" for field in OUTPUT_FIELDS}
        row.update(
            {
                "tile_name": properties["tile_name"],
                "jrc_gsw_source": COLLECTION,
            }
        )
        for field in OUTPUT_FIELDS[2:]:
            value = properties.get(field)
            row[field] = "{:.8f}".format(float(value)) if value is not None else ""
        output.append(row)
    return output


def reduce_batch_with_retry(ee, rows, scale: int, tile_scale: int, attempts: int = 4):
    for attempt in range(attempts):
        try:
            return reduce_batch(ee, rows, scale, tile_scale)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("metadata/training_samples.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/audit/jrc_gsw_annotations.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/audit/jrc_gsw_annotations_summary.json"))
    parser.add_argument("--ee-project", default=None)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--scale", type=int, default=30)
    parser.add_argument("--tile-scale", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if min(args.batch_size, args.scale, args.tile_scale) < 1:
        parser.error("batch size, scale, and tile scale must be positive")

    with args.index.expanduser().resolve().open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    annotations = []
    if args.resume and output.exists():
        with output.open(newline="") as handle:
            annotations = list(csv.DictReader(handle))
        completed = {row["tile_name"] for row in annotations}
    pending = [row for row in rows if row["tile_name"] not in completed]
    if not args.resume or not output.exists():
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
            writer.writeheader()
    ee = initialize_ee(args.ee_project)
    for number, batch in enumerate(chunks(pending, args.batch_size), start=1):
        batch_annotations = reduce_batch_with_retry(ee, batch, args.scale, args.tile_scale)
        with output.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
            writer.writerows(batch_annotations)
        annotations.extend(batch_annotations)
        print(
            "Reduced batch {} ({} of {} samples total)".format(number, len(annotations), len(rows)),
            flush=True,
        )
    summary = {
        "samples": len(annotations),
        "source": COLLECTION,
        "scale_m": args.scale,
        "method": "Earth Engine reduceRegions mean over exact sample footprints",
        "resumable": True,
        "output_csv": str(output),
    }
    summary_path = args.summary.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
