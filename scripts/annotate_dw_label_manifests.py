#!/usr/bin/env python3
"""Extract exact Dynamic World water-pixel exposure from generation manifests.

The original and replacement generation manifests use different tile-name columns.
This command normalizes both forms into one annotation table that can be merged into
``sample_inventory.csv`` with ``inventory_training_samples.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


OUTPUT_FIELDS = (
    "tile_name",
    "dw_label_water_pixels",
    "dw_label_nonwater_pixels",
    "dw_label_total_pixels",
    "dw_label_water_fraction",
    "dw_label_generation_source",
)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def manifest_annotations(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Return the best successful pixel-count row for every normalized tile name."""
    annotations: dict[str, dict[str, Any]] = {}
    row_quality: dict[str, int] = {}
    for path in paths:
        resolved = path.expanduser().resolve()
        with resolved.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row.get("status") not in {"ok", "exists"}:
                continue
            key = row.get("replacement_tile_name") or row.get("tile_name", "")
            if not key or not row.get("water_pixels") or not row.get("nonwater_pixels"):
                continue
            water = int(row["water_pixels"])
            nonwater = int(row["nonwater_pixels"])
            total = water + nonwater
            if total <= 0:
                continue
            quality = 2 if row.get("status") == "ok" else 1
            if quality < row_quality.get(key, -1):
                continue
            annotations[key] = {
                "tile_name": key,
                "dw_label_water_pixels": water,
                "dw_label_nonwater_pixels": nonwater,
                "dw_label_total_pixels": total,
                "dw_label_water_fraction": "{:.8f}".format(water / total),
                "dw_label_generation_source": (
                    "replacement_generation_manifest"
                    if row.get("replacement_tile_name")
                    else "original_generation_manifest"
                ),
            }
            row_quality[key] = quality
    return annotations


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fractions = [float(row["dw_label_water_fraction"]) for row in rows]
    return {
        "annotated_samples": len(rows),
        "water_fraction": {
            "min": round(min(fractions), 8),
            "p25": round(percentile(fractions, 0.25), 8),
            "median": round(median(fractions), 8),
            "mean": round(mean(fractions), 8),
            "p75": round(percentile(fractions, 0.75), 8),
            "max": round(max(fractions), 8),
            "equivalent_full_water_tiles": round(sum(fractions), 3),
        },
        "threshold_counts": {
            "below_0.01": sum(value < 0.01 for value in fractions),
            "at_least_0.01": sum(value >= 0.01 for value in fractions),
            "at_least_0.10": sum(value >= 0.10 for value in fractions),
            "at_least_0.25": sum(value >= 0.25 for value in fractions),
            "at_least_0.50": sum(value >= 0.50 for value in fractions),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--index", type=Path, default=Path("metadata/training_samples.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/audit/dw_label_annotations.csv")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("outputs/audit/dw_label_annotations_summary.json")
    )
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    with args.index.expanduser().resolve().open(newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    index_names = [row["tile_name"] for row in index_rows]
    annotations = manifest_annotations(args.manifest)
    missing = [name for name in index_names if name not in annotations]
    if missing and not args.allow_missing:
        raise SystemExit(
            "Missing exact Dynamic World pixel counts for {} indexed samples; examples: {}. "
            "Pass all original and replacement manifests, or use --allow-missing."
            .format(len(missing), ", ".join(missing[:5]))
        )

    rows = [annotations[name] for name in index_names if name in annotations]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    report = summarize(rows)
    report.update(
        {
            "indexed_samples": len(index_names),
            "missing_samples": len(missing),
            "output_csv": str(output),
        }
    )
    summary_path = args.summary.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
