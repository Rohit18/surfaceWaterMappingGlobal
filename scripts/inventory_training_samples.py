#!/usr/bin/env python3
"""Create a compact, extensible inventory of the released training samples.

Every sample is marked as river-targeted because the sampling frame was SWORD v16.
Optional annotation CSVs can add PLD, JRC Global Surface Water, or coastline results.
The resulting water contexts are deliberately multi-label: a footprint can be both a
river and a reservoir, for example.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path
from statistics import mean, median
from typing import Any


CORE_FIELDS = (
    "tile_name", "sword_node_id", "split", "s1_date", "aef_year",
    "centroid_lon", "centroid_lat", "crs", "width", "height",
)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 6),
        "p25": round(percentile(values, 0.25), 6),
        "median": round(median(values), 6),
        "mean": round(mean(values), 6),
        "p75": round(percentile(values, 0.75), 6),
        "max": round(max(values), 6),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open(newline="") as handle:
        return list(csv.DictReader(handle))


def annotation_lookup(paths: list[Path]) -> tuple[dict[str, dict[str, str]], list[str]]:
    lookup: dict[str, dict[str, str]] = {}
    fields: list[str] = []
    for path in paths:
        rows = read_csv(path)
        if not rows or "tile_name" not in rows[0]:
            raise ValueError("Annotation CSV requires tile_name: {}".format(path))
        for row in rows:
            target = lookup.setdefault(row["tile_name"], {})
            for key, value in row.items():
                if key != "tile_name" and key not in CORE_FIELDS:
                    target[key] = value
                    if key not in fields:
                        fields.append(key)
    return lookup, fields


def water_context(
    row: dict[str, Any],
    lake_threshold: float,
    coast_km: float,
    jrc_threshold: float = 0.01,
) -> list[str]:
    contexts = ["river_targeted"]
    pld_fraction = number(row.get("pld_overlap_fraction"))
    pld_count = number(row.get("pld_feature_count")) or 0.0
    if (
        pld_fraction is not None and pld_fraction >= lake_threshold
    ) or (pld_fraction is None and pld_count > 0):
        contexts.append("lake_or_reservoir")
    reservoir_count = number(row.get("pld_reservoir_count")) or 0.0
    if truthy(row.get("pld_reservoir_flag")) or reservoir_count > 0:
        contexts.append("reservoir")
    coast_distance = number(row.get("coastline_distance_km"))
    if coast_distance is not None and coast_distance <= coast_km:
        contexts.append("coastal")
    maximum_extent = number(row.get("jrc_max_extent_fraction")) or 0.0
    permanent = number(row.get("jrc_permanent_fraction")) or 0.0
    seasonal = number(row.get("jrc_seasonal_fraction")) or 0.0
    if maximum_extent >= jrc_threshold:
        contexts.append("observed_open_water")
    if permanent >= jrc_threshold:
        contexts.append("persistent_water")
    if seasonal >= jrc_threshold:
        contexts.append("seasonal_water")
    return contexts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("metadata/training_samples.csv"))
    parser.add_argument("--annotation-csv", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("outputs/audit/sample_inventory.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/audit/sample_inventory_summary.json"))
    parser.add_argument("--lake-overlap-threshold", type=float, default=0.01)
    parser.add_argument("--coastal-distance-km", type=float, default=5.0)
    parser.add_argument("--jrc-context-fraction-threshold", type=float, default=0.01)
    args = parser.parse_args()
    if (
        not 0 <= args.lake_overlap_threshold <= 1
        or not 0 <= args.jrc_context_fraction_threshold <= 1
        or args.coastal_distance_km < 0
    ):
        parser.error("fraction thresholds must be in [0,1] and coastal distance must be non-negative")

    rows = read_csv(args.index)
    annotations, annotation_fields = annotation_lookup(args.annotation_csv)
    node_counts = Counter(row["sword_node_id"] for row in rows)
    inventory: list[dict[str, Any]] = []
    for row in rows:
        affine = [float(row["transform_{}".format(key)]) for key in "abcdef"]
        pixel_area_m2 = abs(affine[0] * affine[4] - affine[1] * affine[3])
        parsed_date = date.fromisoformat(row["s1_date"])
        item: dict[str, Any] = {field: row[field] for field in CORE_FIELDS}
        item.update(
            {
                "sample_year": parsed_date.year,
                "sample_month": parsed_date.month,
                "hemisphere": "north" if float(row["centroid_lat"]) >= 0 else "south",
                "tile_area_km2": "{:.6f}".format(
                    pixel_area_m2 * int(row["width"]) * int(row["height"]) / 1_000_000
                ),
                "node_sample_count": node_counts[row["sword_node_id"]],
                "repeated_node": int(node_counts[row["sword_node_id"]] > 1),
                "sampling_frame": "SWORD_v16_river_node",
                "indexed_date_quality": "nominal_search_anchor_not_acquisition_timestamp",
            }
        )
        item.update(annotations.get(row["tile_name"], {}))
        item["water_contexts"] = ";".join(
            water_context(
                item,
                args.lake_overlap_threshold,
                args.coastal_distance_km,
                args.jrc_context_fraction_threshold,
            )
        )
        inventory.append(item)

    output_fields = list(CORE_FIELDS) + [
        "sample_year", "sample_month", "hemisphere", "tile_area_km2",
        "node_sample_count", "repeated_node", "sampling_frame", "indexed_date_quality",
    ] + annotation_fields + ["water_contexts"]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(inventory)

    contexts = Counter(
        context for row in inventory for context in row["water_contexts"].split(";") if context
    )
    context_combinations = Counter(row["water_contexts"] for row in inventory)
    years = Counter(str(row["sample_year"]) for row in inventory)
    months = Counter(str(row["sample_month"]) for row in inventory)
    splits = Counter(row["split"] for row in inventory)
    hemispheres = Counter(row["hemisphere"] for row in inventory)
    areas = [float(row["tile_area_km2"]) for row in inventory]
    assessed = {
        "lake_or_reservoir_overlap": any(field.startswith("pld_") for field in annotation_fields),
        "reservoir_status": any(
            field in {"pld_reservoir_count", "pld_reservoir_flag"}
            for field in annotation_fields
        ),
        "coastal": "coastline_distance_km" in annotation_fields,
        "water_persistence": any(field.startswith("jrc_") for field in annotation_fields),
        "dynamic_world_label_exposure": "dw_label_water_fraction" in annotation_fields,
    }
    summary = {
        "samples": len(inventory),
        "unique_sword_nodes": len(node_counts),
        "repeated_node_samples": sum(int(row["repeated_node"]) for row in inventory),
        "sampling_frame": "SWORD v16 river nodes; all samples are river-targeted by design",
        "split_counts": dict(sorted(splits.items())),
        "year_counts": dict(sorted(years.items())),
        "month_counts": dict(sorted(months.items(), key=lambda pair: int(pair[0]))),
        "hemisphere_counts": dict(sorted(hemispheres.items())),
        "tile_area_km2": distribution(areas),
        "water_context_counts_multilabel": dict(sorted(contexts.items())),
        "water_context_combination_counts": dict(
            sorted(context_combinations.items(), key=lambda pair: (-pair[1], pair[0]))
        ),
        "water_context_thresholds": {
            "pld_overlap_fraction": args.lake_overlap_threshold,
            "jrc_footprint_fraction": args.jrc_context_fraction_threshold,
            "coastline_distance_km": args.coastal_distance_km,
        },
        "annotation_assessment": assessed,
        "not_yet_assessed": [name for name, complete in assessed.items() if not complete],
        "classification_note": (
            "Water contexts are multi-label. Do not force river/lake/reservoir/coastal classes "
            "to be mutually exclusive."
        ),
        "output_csv": str(output),
    }
    jrc_fields = (
        "jrc_max_extent_fraction", "jrc_permanent_fraction", "jrc_seasonal_fraction"
    )
    if all(field in annotation_fields for field in jrc_fields):
        summary["jrc_gsw_fraction_distributions"] = {
            field: distribution([float(row[field]) for row in inventory]) for field in jrc_fields
        }
        summary["jrc_gsw_threshold_counts"] = {
            field: {
                "greater_than_0": sum(float(row[field]) > 0 for row in inventory),
                "at_least_0.001": sum(float(row[field]) >= 0.001 for row in inventory),
                "at_least_0.01": sum(float(row[field]) >= 0.01 for row in inventory),
                "at_least_0.05": sum(float(row[field]) >= 0.05 for row in inventory),
            }
            for field in jrc_fields
        }
    if "pld_overlap_fraction" in annotation_fields:
        pld_fractions = [float(row["pld_overlap_fraction"]) for row in inventory]
        summary["pld_overlap_fraction_distribution"] = distribution(pld_fractions)
        summary["pld_overlap_threshold_counts"] = {
            "greater_than_0": sum(value > 0 for value in pld_fractions),
            "at_least_0.001": sum(value >= 0.001 for value in pld_fractions),
            "at_least_0.01": sum(value >= 0.01 for value in pld_fractions),
            "at_least_0.05": sum(value >= 0.05 for value in pld_fractions),
        }
    if "dw_label_water_fraction" in annotation_fields:
        label_fractions = [float(row["dw_label_water_fraction"]) for row in inventory]
        summary["dw_label_water_fraction_distribution"] = distribution(label_fractions)
        summary["dw_label_water_fraction_threshold_counts"] = {
            "below_0.01": sum(value < 0.01 for value in label_fractions),
            "at_least_0.01": sum(value >= 0.01 for value in label_fractions),
            "at_least_0.10": sum(value >= 0.10 for value in label_fractions),
            "at_least_0.25": sum(value >= 0.25 for value in label_fractions),
            "at_least_0.50": sum(value >= 0.50 for value in label_fractions),
        }
        summary["dw_label_equivalent_full_water_tiles"] = round(sum(label_fractions), 3)
    summary_path = args.summary.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
