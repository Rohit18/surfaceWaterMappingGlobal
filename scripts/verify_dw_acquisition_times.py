#!/usr/bin/env python3
"""Resolve Dynamic World granule acquisition times from Earth Engine and check them.

The S1 side of ``s1_dw_timing.csv`` was verified against stored raster pixels, but the
Dynamic World timestamps were only ever copied from a generation manifest. This script
asks Earth Engine for the authoritative ``system:time_start`` of every DW granule the
dataset references, then recomputes the S1-DW gap from those times.

Granules are resolved once each and cached, so the per-sample cost is a dictionary
lookup. The output is resumable: re-running only queries granules not already cached.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

COLLECTION = "GOOGLE/DYNAMICWORLD/V1"


def initialize_ee(project: str | None):
    try:
        import ee

        ee.Initialize(project=project)
        return ee
    except Exception as exc:
        raise SystemExit("Earth Engine initialization failed: {}".format(exc)) from exc


def iso_utc(timestamp_ms: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def granule(image_id: str) -> str:
    """Strip the collection prefix; the two sets store ids differently."""
    return image_id.rsplit("/", 1)[-1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def json_list(value: str) -> list[str]:
    if not value:
        return []
    parsed = json.loads(value)
    return parsed if isinstance(parsed, list) else [parsed]


def load_cache(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    return {row["granule_id"]: int(row["time_start_ms"]) for row in read_csv(path)}


def save_cache(path: Path, cache: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["granule_id", "time_start_ms", "time_start_utc"])
        for key in sorted(cache):
            writer.writerow([key, cache[key], iso_utc(cache[key])])


def resolve(ee, granules: list[str], batch: int, cache: dict[str, int]) -> list[str]:
    """Query system:time_start in batches; return granules Earth Engine did not return."""
    pending = [key for key in granules if key not in cache]
    for start in range(0, len(pending), batch):
        chunk = pending[start : start + batch]
        collection = ee.ImageCollection(COLLECTION).filter(
            ee.Filter.inList("system:index", chunk)
        )
        features = ee.FeatureCollection(
            collection.map(
                lambda image: ee.Feature(
                    None,
                    {"id": image.get("system:index"), "t": image.get("system:time_start")},
                )
            )
        ).getInfo()
        for feature in features["features"]:
            properties = feature["properties"]
            if properties.get("t") is not None:
                cache[granule(properties["id"])] = int(properties["t"])
        print(
            "  resolved {}/{} granules".format(len(cache), len(granules)),
            flush=True,
        )
    return [key for key in granules if key not in cache]


def sample_rows(audit: Path) -> list[dict[str, Any]]:
    """One record per sample across both sets, with its DW granules and stored times."""
    rows: list[dict[str, Any]] = []
    for record in read_csv(audit / "s1_dw_timing.csv"):
        rows.append(
            {
                "sample_id": record["tile_name"],
                "sample_set": "main",
                "s1_datetime_utc": record["nearest_s1_datetime_utc"],
                "stored_dw_datetime_utc": record["nearest_dw_datetime_utc"],
                "stored_gap_hours": record["absolute_time_delta_hours"],
                "nearest_dw_granule": granule(record["nearest_dw_image_id"]),
                "dw_granules": [granule(value) for value in json_list(record["dw_image_ids"])],
                "s1_datetimes": json_list(record["s1_datetimes_utc"]),
            }
        )
    for record in read_csv(audit / "supplement_samples.csv"):
        rows.append(
            {
                "sample_id": record["candidate_id"],
                "sample_set": "supplement",
                "s1_datetime_utc": record["s1_datetime_utc"],
                "stored_dw_datetime_utc": record["dw_datetime_utc"],
                "stored_gap_hours": record["absolute_time_delta_hours"],
                "nearest_dw_granule": granule(record["dw_image_id"]),
                "dw_granules": [granule(value) for value in json_list(record["dw_image_ids"])],
                "s1_datetimes": [record["s1_datetime_utc"]],
            }
        )
    return rows


def parse_utc(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


HOUR_MS = 3_600_000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_audit = Path(__file__).resolve().parent.parent / "outputs" / "audit"
    parser.add_argument("--audit-dir", type=Path, default=default_audit)
    parser.add_argument("--cache", type=Path, default=None, help="granule time cache CSV")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--ee-project", default=None)
    args = parser.parse_args()

    audit = args.audit_dir
    cache_path = args.cache or audit / "dw_granule_times.csv"
    output_path = args.output or audit / "dw_acquisition_verification.csv"
    summary_path = args.summary or audit / "dw_acquisition_verification_summary.json"

    rows = sample_rows(audit)
    granules = sorted({key for row in rows for key in row["dw_granules"]} | {row["nearest_dw_granule"] for row in rows})
    print("samples: {} | unique DW granules: {}".format(len(rows), len(granules)), flush=True)

    cache = load_cache(cache_path)
    print("cached: {}".format(len(cache)), flush=True)
    ee = initialize_ee(args.ee_project)
    missing = resolve(ee, granules, args.batch, cache)
    save_cache(cache_path, cache)
    if missing:
        print("WARNING: {} granules unresolved".format(len(missing)))

    records: list[dict[str, Any]] = []
    for row in rows:
        stored_ms = parse_utc(row["stored_dw_datetime_utc"])
        actual_ms = cache.get(row["nearest_dw_granule"])
        s1_ms = parse_utc(row["s1_datetime_utc"])
        # Recompute the gap using every resolved DW granule, as the audit does.
        resolved = [cache[key] for key in row["dw_granules"] if key in cache]
        s1_times = [parse_utc(value) for value in row["s1_datetimes"]] or [s1_ms]
        best = min(
            (abs(s - d), s, d) for s in s1_times for d in resolved
        ) if resolved else None
        records.append(
            {
                "sample_id": row["sample_id"],
                "sample_set": row["sample_set"],
                "nearest_dw_granule": row["nearest_dw_granule"],
                "stored_dw_datetime_utc": row["stored_dw_datetime_utc"],
                "actual_dw_datetime_utc": iso_utc(actual_ms) if actual_ms is not None else "",
                "dw_time_error_seconds": (
                    "{:.0f}".format((stored_ms - actual_ms) / 1000) if actual_ms is not None else ""
                ),
                "s1_datetime_utc": row["s1_datetime_utc"],
                "stored_gap_hours": row["stored_gap_hours"],
                "verified_gap_hours": "{:.6f}".format(best[0] / HOUR_MS) if best else "",
                "gap_error_hours": (
                    "{:.6f}".format(float(row["stored_gap_hours"]) - best[0] / HOUR_MS)
                    if best and row["stored_gap_hours"]
                    else ""
                ),
                "dw_granules_resolved": len(resolved),
                "dw_granules_referenced": len(row["dw_granules"]),
            }
        )

    fields = list(records[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    errors = [abs(int(r["dw_time_error_seconds"])) for r in records if r["dw_time_error_seconds"]]
    gap_errors = [abs(float(r["gap_error_hours"])) for r in records if r["gap_error_hours"]]
    summary = {
        "samples": len(records),
        "unique_granules": len(granules),
        "granules_resolved": len(cache),
        "granules_unresolved": len(missing),
        "samples_with_actual_dw_time": len(errors),
        "dw_time_error_seconds": {
            "exact": sum(1 for value in errors if value == 0),
            "within_1s": sum(1 for value in errors if value <= 1),
            "within_60s": sum(1 for value in errors if value <= 60),
            "max": max(errors) if errors else None,
        },
        "gap_error_hours": {
            "exact": sum(1 for value in gap_errors if value < 1e-6),
            "within_0.01h": sum(1 for value in gap_errors if value <= 0.01),
            "max": max(gap_errors) if gap_errors else None,
        },
        "output_csv": str(output_path.resolve()),
        "cache_csv": str(cache_path.resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
