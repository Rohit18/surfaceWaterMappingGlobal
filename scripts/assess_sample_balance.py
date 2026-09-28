#!/usr/bin/env python3
"""Assess water-context representation and emit a mild supplementation plan."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable


Predicate = Callable[[dict[str, str]], bool]


def contexts(row: dict[str, str]) -> set[str]:
    return {value for value in row.get("water_contexts", "").split(";") if value}


def has(name: str) -> Predicate:
    return lambda row: name in contexts(row)


def stratum_predicates() -> dict[str, Predicate]:
    lake = has("lake_or_reservoir")
    coast = has("coastal")
    persistent = has("persistent_water")
    seasonal = has("seasonal_water")
    observed = has("observed_open_water")
    return {
        "lake_inland": lambda row: lake(row) and not coast(row),
        "lake_coastal": lambda row: lake(row) and coast(row),
        "coastal_nonlake": lambda row: coast(row) and not lake(row),
        "neither_lake_nor_coastal": lambda row: not lake(row) and not coast(row),
        "persistent_only": lambda row: persistent(row) and not seasonal(row),
        "seasonal_only": lambda row: seasonal(row) and not persistent(row),
        "persistent_and_seasonal": lambda row: persistent(row) and seasonal(row),
        "observed_not_persistent_or_seasonal": (
            lambda row: observed(row) and not persistent(row) and not seasonal(row)
        ),
        "no_observed_open_water_at_1pct": lambda row: not observed(row),
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in rows)
    result: dict[str, Any] = {
        "samples": len(rows),
        "train": split_counts.get("train", 0),
        "valid": split_counts.get("valid", 0),
    }
    if rows and all(row.get("dw_label_water_fraction", "") for row in rows):
        values = [float(row["dw_label_water_fraction"]) for row in rows]
        result["dw_label_water_fraction"] = {
            "mean": round(mean(values), 6),
            "median": round(median(values), 6),
            "p75": round(percentile(values, 0.75), 6),
            "equivalent_full_water_tiles": round(sum(values), 1),
            "below_0.01": sum(value < 0.01 for value in values),
            "at_least_0.10": sum(value >= 0.10 for value in values),
            "at_least_0.25": sum(value >= 0.25 for value in values),
        }
    return result


def timing_strata(
    rows: list[dict[str, str]], timing_rows: list[dict[str, str]] | None
) -> dict[str, Any] | None:
    if not timing_rows:
        return None
    gaps = {
        row["tile_name"]: float(row["absolute_time_delta_hours"])
        for row in timing_rows
        if row.get("pair_status") == "resolved" and row.get("absolute_time_delta_hours", "")
    }
    bins = {
        "0_to_6_hours": lambda value: value <= 6,
        "over_6_to_12_hours": lambda value: 6 < value <= 12,
        "over_12_to_24_hours": lambda value: 12 < value <= 24,
        "over_24_to_48_hours": lambda value: 24 < value <= 48,
        "over_48_hours": lambda value: value > 48,
    }
    result = {
        name: summarize_rows(
            [row for row in rows if row["tile_name"] in gaps and predicate(gaps[row["tile_name"]])]
        )
        for name, predicate in bins.items()
    }
    result["recommendation"] = (
        "Do not add samples merely to equalize acquisition-gap bins. Evaluate metrics by "
        "these bins first; every current bin has hundreds of samples."
    )
    return result


def assess(
    rows: list[dict[str, str]], timing_rows: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    context_names = (
        "river_targeted", "lake_or_reservoir", "coastal", "observed_open_water",
        "persistent_water", "seasonal_water",
    )
    context_report = {
        name: summarize_rows([row for row in rows if name in contexts(row)])
        for name in context_names
    }
    strata = {
        name: summarize_rows([row for row in rows if predicate(row)])
        for name, predicate in stratum_predicates().items()
    }
    hemisphere_report = {
        hemisphere: summarize_rows([row for row in rows if row["hemisphere"] == hemisphere])
        for hemisphere in ("north", "south")
    }
    temporal = {
        "year_counts": dict(sorted(Counter(row["sample_year"] for row in rows).items())),
        "month_counts": dict(
            sorted(Counter(row["sample_month"] for row in rows).items(), key=lambda item: int(item[0]))
        ),
    }
    report = {
        "samples": len(rows),
        "decision": (
            "Do not equalize the existing contexts. Broad river-connected lake, coastal, "
            "persistent, and seasonal contexts have usable counts, but the SWORD-only "
            "sampling frame cannot establish independent representation of all open water."
        ),
        "context_counts": context_report,
        "cross_strata": strata,
        "hemisphere": hemisphere_report,
        "temporal": temporal,
        "representation_findings": {
            "adequate_for_current_river_connected_scope": [
                "lake_or_reservoir_context", "coastal_context", "persistent_water",
                "seasonal_water",
            ],
            "gaps": [
                "non_river_lakes", "confirmed_reservoirs", "independently_sampled_coast_or_estuary",
                "high_water_seasonal_or_ephemeral_scenes", "southern_hemisphere",
                "non_October_local_hydrologic_seasons",
            ],
            "evaluation_warning": (
                "Lake/coastal overlap has too few validation samples for a stable standalone "
                "subgroup estimate, and reservoir status has not been annotated."
            ),
        },
        "mild_supplement_plan": {
            "unique_new_samples_target": "500-600 (about 11-13% of the current dataset)",
            "quota_tags_are_overlapping": True,
            "candidate_quotas": {
                "non_river_natural_lake": 200,
                "confirmed_reservoir": 125,
                "coastal_or_estuarine": 125,
                "high_water_seasonal_or_ephemeral": 150,
            },
            "selection_constraints": [
                "Exclude the SWORD sampling corridor for non-river lake and reservoir targets.",
                "Prefer southern/tropical candidates and local wet/dry seasons underrepresented by October-heavy sampling.",
                "For each target, retain roughly 50% tiles with 10-40% water, 25% above 40%, and 25% boundary/hard examples with 1-10% water.",
                "Group by lake/reservoir ID, basin, or coastal segment before splitting to prevent spatial leakage.",
                "Reserve at least 30 validation samples for each key new stratum; samples may satisfy more than one stratum.",
            ],
            "do_not_remove": (
                "Retain current low-water and hard-negative tiles; address pixel imbalance with "
                "water-aware crop sampling and modest batch weighting rather than deleting them."
            ),
            "stopping_rule": (
                "Stop adding a stratum when its held-out precision, recall, and IoU are stable "
                "across water-fraction bins; equal tile counts are not the objective."
            ),
        },
    }
    timing = timing_strata(rows, timing_rows)
    if timing is not None:
        report["s1_dynamic_world_gap_strata"] = timing
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("outputs/audit/sample_inventory.csv"))
    parser.add_argument("--timing", type=Path, default=Path("outputs/audit/s1_dw_timing.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/audit/sample_balance_summary.json"))
    args = parser.parse_args()
    with args.inventory.expanduser().resolve().open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    timing_path = args.timing.expanduser().resolve()
    timing_rows = None
    if timing_path.exists():
        with timing_path.open(newline="") as handle:
            timing_rows = list(csv.DictReader(handle))
    report = assess(rows, timing_rows)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
