#!/usr/bin/env python3
"""Replace supplement rows that fail the exact 10 m export-grid coverage check.

The initial candidate search measures a geodesic bounding footprint.  The final
rasters use an 11 km UTM-aligned square, so a small number of edge-of-swath rows
can fall below the final coverage threshold.  This command reads the materializer
manifests, keeps every successful row, and substitutes eligible candidates of the
same water-feature type after checking their exact export grids in Earth Engine.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

from materialize_supplement_samples import (
    DW_COLLECTION,
    S1_BANDS,
    S1_COLLECTION,
    collection_from_ids,
    exact_region,
    grid,
    parse_ids,
    server_metrics,
)
from sample_water_feature_supplement import (
    grid_properties,
    read_csv,
    spatially_distinct,
    stable_score,
    write_selection,
)
from reconstruct_sample import initialize_ee


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/audit/supplement_samples.csv"))
    parser.add_argument(
        "--materialization-manifests", type=Path,
        default=Path("/pscratch/sd/r/rohit9/S1ML/training_supplement/v1/manifests"),
    )
    parser.add_argument(
        "--candidate-pairs", type=Path, action="append",
        default=[
            Path("outputs/audit/supplement_candidate_pairs_checkpoint.csv"),
            Path("outputs/audit/supplement_coastal_candidate_pairs.csv"),
            Path("outputs/audit/supplement_seasonal_candidate_pairs.csv"),
        ],
    )
    parser.add_argument("--footprints", type=Path, default=Path("outputs/audit/supplement_sample_footprints.geojson"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/audit/supplement_samples_summary.json"))
    parser.add_argument("--backup", type=Path, default=Path("outputs/audit/supplement_samples.before_exact_coverage_repair.csv"))
    parser.add_argument("--ee-project", default="rohit-global-water")
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.90)
    parser.add_argument("--minimum-distance-km", type=float, default=20.0)
    parser.add_argument("--maximum-gap-hours", type=float, default=24.0)
    parser.add_argument("--expected-samples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def materialization_statuses(directory: Path) -> dict[str, dict[str, str]]:
    paths = sorted(directory.glob("materialize_*_of_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No materialization manifests found in {directory}")
    statuses: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                statuses[row["candidate_id"]] = row
    return statuses


def eligible(row: dict[str, str], maximum_gap_hours: float) -> bool:
    if row.get("pair_status") != "pair_found":
        return False
    if float(row.get("absolute_time_delta_hours") or 1e9) > maximum_gap_hours:
        return False
    if float(row.get("maximum_component_time_delta_hours") or 1e9) > maximum_gap_hours:
        return False
    if float(row.get("dw_water_fraction") or 0) < 0.01:
        return False
    if row.get("primary_target") == "high_water_seasonal_or_ephemeral":
        return (
            float(row.get("dw_water_fraction") or 0) >= 0.10
            and float(row.get("jrc_seasonal_fraction") or 0) >= 0.10
        )
    return True


def candidate_sort_key(row: dict[str, str], seed: int) -> tuple[Any, ...]:
    water_bin = row.get("water_fraction_bin")
    bin_rank = 0 if water_bin == "10_to_40pct" else 1 if water_bin == "above_40pct" else 2
    return (
        0 if row.get("hemisphere") == "south" else 1,
        bin_rank,
        stable_score(seed, row["candidate_id"], "select"),
    )


def exact_metrics(ee: Any, row: dict[str, str]) -> dict[str, float]:
    crs, affine, width, height = grid(row)
    region, _ = exact_region(ee, crs, affine, width, height)
    s1_ids = parse_ids(row["s1_image_ids"], S1_COLLECTION)
    dw_ids = parse_ids(row["dw_image_ids"], DW_COLLECTION)
    s1 = collection_from_ids(ee, s1_ids, S1_BANDS).mosaic().select(list(S1_BANDS)).float().clip(region)
    dw = collection_from_ids(ee, dw_ids, ("label",)).mosaic().select("label").clip(region)
    return server_metrics(ee, s1, dw, region)


def main() -> None:
    args = parse_args()
    selected = read_csv(args.manifest)
    if len(selected) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} selected rows, found {len(selected)}")
    statuses = materialization_statuses(args.materialization_manifests)
    missing = {row["candidate_id"] for row in selected}.difference(statuses)
    if missing:
        raise RuntimeError(f"Materialization status is missing {len(missing)} selected candidates")

    failures = {
        candidate_id: row for candidate_id, row in statuses.items()
        if candidate_id in {item["candidate_id"] for item in selected} and row.get("status") != "ok"
    }
    unexpected = {
        candidate_id: row.get("error", "") for candidate_id, row in failures.items()
        if "Exact-grid valid coverage is below threshold" not in row.get("error", "")
    }
    if unexpected:
        examples = list(unexpected.items())[:5]
        raise RuntimeError(f"Refusing to replace non-coverage failures ({len(unexpected)}): {examples}")
    if not failures:
        print("All selected samples already pass exact-grid coverage; no repair needed.")
        return

    by_id: dict[str, dict[str, str]] = {}
    for path in args.candidate_pairs:
        for row in read_csv(path):
            by_id[row["candidate_id"]] = row
    selected_ids = {row["candidate_id"] for row in selected}
    alternatives = [
        row for row in by_id.values()
        if row["candidate_id"] not in selected_ids and eligible(row, args.maximum_gap_hours)
    ]
    for row in alternatives:
        # Construct the exact UTM grid used by the materializer.
        properties = grid_properties(row)
        row.update({key: value for key, value in properties.items() if key != "footprint"})

    retained = [row for row in selected if row["candidate_id"] not in failures]
    ee = initialize_ee(args.ee_project)
    replacements: dict[str, dict[str, str]] = {}
    checked = 0
    for rejected in (row for row in selected if row["candidate_id"] in failures):
        candidates = [row for row in alternatives if row["primary_target"] == rejected["primary_target"]]
        candidates.sort(key=lambda row: candidate_sort_key(row, args.seed))
        replacement = None
        for row in candidates:
            if row["candidate_id"] in {item["candidate_id"] for item in replacements.values()}:
                continue
            if not spatially_distinct(row, retained + list(replacements.values()), args.minimum_distance_km):
                continue
            checked += 1
            try:
                metrics = exact_metrics(ee, row)
            except Exception as exc:
                print(f"Rejected {row['candidate_id']}: exact-grid query failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            print(f"Checked {row['candidate_id']}: {metrics}", flush=True)
            if (
                metrics["s1_valid"] >= args.minimum_valid_fraction
                and metrics["dw_valid"] >= args.minimum_valid_fraction
            ):
                replacement = dict(row)
                replacement["split"] = rejected["split"]
                replacement["s1_valid_fraction"] = f"{metrics['s1_valid']:.6f}"
                replacement["dw_valid_fraction"] = f"{metrics['dw_valid']:.6f}"
                break
        if replacement is None:
            raise RuntimeError(f"No exact-grid replacement found for {rejected['candidate_id']}")
        replacements[rejected["candidate_id"]] = replacement
        print(
            f"Replacing {rejected['candidate_id']} with {replacement['candidate_id']} "
            f"({rejected['primary_target']}, split={rejected['split']})",
            flush=True,
        )

    revised = [replacements.get(row["candidate_id"], row) for row in selected]
    if len({row["candidate_id"] for row in revised}) != args.expected_samples:
        raise RuntimeError("Repair did not produce the expected number of unique candidates")
    if not args.backup.exists():
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.manifest, args.backup)
    write_selection(
        revised, args.manifest, args.footprints, args.summary, args.seed,
        preserve_splits=True,
    )
    print(f"Replaced {len(replacements)} rows after checking {checked} alternatives.")


if __name__ == "__main__":
    main()
