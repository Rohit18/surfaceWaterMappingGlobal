#!/usr/bin/env python3
"""Join the audit annotation tables onto the sample footprint polygons.

Produces two standalone layers:

``enriched_sample_footprints``
    The 4,678 SWORD river-node samples: footprint geometry plus every column
    from ``sample_inventory.csv`` and ``s1_dw_timing.csv`` (which between them
    already carry the DW-label, JRC, PLD and coastline annotation blocks).

``unified_sample_footprints``
    Those 4,678 plus the 600 water-feature supplement samples, reconciled onto
    a shared core schema. Fields native to only one set are emitted as null on
    the other; derived fields are named in ``DERIVED_FIELDS`` and explained in
    the join report.

``clean_sample_footprints``
    The unified layer minus samples whose Dynamic World label mosaic spans more
    than one acquisition date. Earth Engine mosaics last-on-top, so those tiles
    carry labels from two different days in one raster and have no single
    observation time. Excluded samples are listed in ``excluded_samples.csv``.

Every layer is written as both GeoJSON and a geometry-free CSV twin.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Columns kept as text even when they parse as numbers: identifiers, codes,
# dates, and the JSON-array-in-a-string list fields.
STRING_FIELDS = frozenset(
    {
        "sample_id",
        "sample_set",
        "tile_name",
        "candidate_id",
        "sword_node_id",
        "source_dataset",
        "source_feature_id",
        "group_id",
        "split",
        "target_class",
        "primary_target",
        "sampling_frame",
        "indexed_date_quality",
        "hemisphere",
        "crs",
        "water_contexts",
        "selection_tags",
        "water_fraction_bin",
        "dw_water_fraction_bin",
        "pld_source",
        "pld_lake_ids",
        "jrc_gsw_source",
        "coastline_source",
        "dw_label_generation_source",
        "pair_status",
        "provenance_quality",
        "pair_source",
        "s1_orbit_pass",
        "s1_relative_orbit",
        "s1_date",
        "s1_scene_id",
        "dw_scene_id",
        "s1_image_id",
        "s1_image_ids",
        "s1_datetime_utc",
        "s1_datetimes_utc",
        "dw_image_id",
        "dw_image_ids",
        "dw_datetime_utc",
        "dw_datetimes_utc",
        "nearest_s1_image_id",
        "nearest_s1_datetime_utc",
        "nearest_dw_image_id",
        "nearest_dw_datetime_utc",
        "error",
    }
)

# Fields this script computes rather than copies; reported alongside the join.
DERIVED_FIELDS = {
    "sample_id": "tile_name (main) or candidate_id (supplement)",
    "sample_set": "'main' for the SWORD river-node audit, 'supplement' for the water-feature supplement",
    "target_class": "'river_node' for main rows, the native primary_target for supplement rows",
    "dw_water_fraction": "main dw_label_water_fraction / supplement dw_water_fraction",
    "dw_water_fraction_bin": "binned dw_water_fraction on the supplement's own edges (0.01/0.10/0.40); reproduces water_fraction_bin exactly on all 600 supplement rows",
    "s1_image_id": "main nearest_s1_image_id / supplement s1_image_id",
    "s1_datetime_utc": "main nearest_s1_datetime_utc / supplement s1_datetime_utc",
    "dw_image_id": "main nearest_dw_image_id / supplement dw_image_id",
    "dw_datetime_utc": "main nearest_dw_datetime_utc / supplement dw_datetime_utc",
    "dw_datetime_utc_verified": "Earth Engine system:time_start for the nearest DW granule (see verify_dw_acquisition_times.py)",
    "dw_gap_hours_verified": "|S1 - DW| recomputed from the Earth Engine granule times",
    "dw_granule_count": "number of distinct DW granules the label mosaic references",
    "dw_spread_hours": "time span across those granules; ~0 when they come from one Sentinel-2 datatake",
    "dw_date_count": "distinct UTC dates among those granules",
    "dw_multi_date": "1 when the label mosaic spans more than one date, so the tile has no single observation time",
    "s1_scene_id": "s1_image_id with any Earth Engine collection prefix stripped, so the two sets compare",
    "dw_scene_id": "dw_image_id with any Earth Engine collection prefix stripped",
    "acquisition_year": "year of the resolved S1 acquisition timestamp, not the indexed search date",
    "acquisition_month": "month of the resolved S1 acquisition timestamp",
    "min_lon/min_lat/max_lon/max_lat": "carried from the main footprints; recomputed from the polygon ring for supplement rows",
}

# Column order for the unified layer: shared core first, then each set's natives.
UNIFIED_CORE = [
    "sample_id",
    "sample_set",
    "split",
    "target_class",
    "centroid_lon",
    "centroid_lat",
    "hemisphere",
    "acquisition_year",
    "acquisition_month",
    "dw_water_fraction",
    "dw_water_fraction_bin",
    "jrc_max_extent_fraction",
    "jrc_permanent_fraction",
    "jrc_seasonal_fraction",
    "pair_status",
    "s1_scene_id",
    "dw_scene_id",
    "s1_image_id",
    "s1_datetime_utc",
    "dw_image_id",
    "dw_datetime_utc",
    "s1_minus_dw_hours",
    "absolute_time_delta_hours",
    "dw_gap_hours_verified",
    "dw_granule_count",
    "dw_spread_hours",
    "dw_date_count",
    "dw_multi_date",
    "s1_image_ids",
    "s1_datetimes_utc",
    "dw_image_ids",
    "dw_datetimes_utc",
    "crs",
    "width",
    "height",
    "transform_a",
    "transform_b",
    "transform_c",
    "transform_d",
    "transform_e",
    "transform_f",
    "min_lon",
    "min_lat",
    "max_lon",
    "max_lat",
]

DW_STAT_FIELDS = (
    "dw_granule_count",
    "dw_spread_hours",
    "dw_date_count",
    "dw_multi_date",
    "dw_datetime_utc_verified",
    "dw_gap_hours_verified",
)

WATER_FRACTION_EDGES = ((0.01, "below_1pct"), (0.10, "1_to_10pct"), (0.40, "10_to_40pct"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def index_by(rows: Iterable[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def water_fraction_bin(value: str | None) -> str | None:
    """Bin a water fraction on the supplement's published edges."""
    if value in (None, ""):
        return None
    fraction = float(value)
    for edge, label in WATER_FRACTION_EDGES:
        if fraction < edge:
            return label
    return "above_40pct"


HOUR_MS = 3_600_000.0


def json_list(value: str | None) -> list[str]:
    if not value:
        return []
    parsed = json.loads(value)
    return parsed if isinstance(parsed, list) else [parsed]


def parse_utc(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def utc_datetime(timestamp_ms: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def scene_id(image_id: str | None) -> str | None:
    """Strip an Earth Engine collection prefix so scene ids compare across sets."""
    if image_id in (None, ""):
        return None
    return image_id.rsplit("/", 1)[-1]


def load_granule_times(path: Path) -> dict[str, int]:
    """Earth Engine system:time_start per DW granule, from verify_dw_acquisition_times.py."""
    if not path.exists():
        return {}
    return {row["granule_id"]: int(row["time_start_ms"]) for row in read_csv(path)}


def dw_granule_stats(image_ids: str, s1_datetime: str, times: dict[str, int]) -> dict[str, Any]:
    """Timing facts about the DW granules behind one label mosaic."""
    granules = [scene_id(value) for value in json_list(image_ids)]
    resolved = sorted(times[key] for key in granules if key in times)
    stats: dict[str, Any] = {
        "dw_granule_count": len(granules),
        "dw_spread_hours": "",
        "dw_date_count": "",
        "dw_multi_date": "",
        "dw_datetime_utc_verified": "",
        "dw_gap_hours_verified": "",
    }
    if not resolved:
        return stats
    dates = {utc_datetime(value)[:10] for value in resolved}
    stats["dw_spread_hours"] = "{:.6f}".format((resolved[-1] - resolved[0]) / HOUR_MS)
    stats["dw_date_count"] = len(dates)
    stats["dw_multi_date"] = int(len(dates) > 1)
    if s1_datetime:
        s1_ms = parse_utc(s1_datetime)
        nearest = min(resolved, key=lambda value: abs(value - s1_ms))
        stats["dw_datetime_utc_verified"] = utc_datetime(nearest)
        stats["dw_gap_hours_verified"] = "{:.6f}".format(abs(nearest - s1_ms) / HOUR_MS)
    return stats


def ring_bounds(ring: list[list[float]]) -> dict[str, float]:
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    return {"min_lon": min(xs), "min_lat": min(ys), "max_lon": max(xs), "max_lat": max(ys)}


def column_types(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, str]:
    """Infer one type per column from all of its values, so QGIS sees a stable schema."""
    types: dict[str, str] = {}
    for field in fields:
        if field in STRING_FIELDS:
            types[field] = "str"
            continue
        values = [row[field] for row in rows if row.get(field) not in (None, "")]
        if not values:
            types[field] = "str"
            continue
        kind = "int"
        for value in values:
            text = str(value)
            try:
                int(text)
            except ValueError:
                kind = "float"
                break
        if kind == "float":
            for value in values:
                try:
                    float(str(value))
                except ValueError:
                    kind = "str"
                    break
        types[field] = kind
    return types


def cast(value: Any, kind: str) -> Any:
    if value in (None, ""):
        return None
    if kind == "int":
        return int(str(value))
    if kind == "float":
        return float(str(value))
    return str(value)


def typed_properties(
    rows: list[dict[str, Any]], fields: list[str], types: dict[str, str]
) -> list[dict[str, Any]]:
    return [{field: cast(row.get(field), types[field]) for field in fields} for row in rows]


def write_geojson(path: Path, records: list[dict[str, Any]], geometries: list[dict[str, Any]], id_field: str) -> None:
    features = [
        {
            "type": "Feature",
            "id": record[id_field],
            "geometry": geometry,
            "properties": record,
        }
        for record, geometry in zip(records, geometries)
    ]
    payload = {"type": "FeatureCollection", "features": features}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")


def write_csv(path: Path, records: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: "" if record[field] is None else record[field] for field in fields})


def build_main(audit: Path, report: dict[str, Any], granule_times: dict[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Join the six annotation tables onto the 4,678 river-node footprints."""
    footprints = json.loads((audit / "sample_footprints.geojson").read_text())["features"]
    tables = {
        "sample_inventory": read_csv(audit / "sample_inventory.csv"),
        "s1_dw_timing": read_csv(audit / "s1_dw_timing.csv"),
        "dw_label_annotations": read_csv(audit / "dw_label_annotations.csv"),
        "jrc_gsw_annotations": read_csv(audit / "jrc_gsw_annotations.csv"),
        "pld_annotations": read_csv(audit / "pld_annotations.csv"),
        "coastline_annotations": read_csv(audit / "coastline_annotations.csv"),
    }

    keys = [feature["properties"]["tile_name"] for feature in footprints]
    key_set = set(keys)
    report["main"] = {
        "footprint_features": len(footprints),
        "footprint_keys_unique": len(key_set) == len(keys),
        "tables": {},
        "conflicts": {},
    }

    indexed = {}
    for name, rows in tables.items():
        table_keys = [row["tile_name"] for row in rows]
        indexed[name] = index_by(rows, "tile_name")
        report["main"]["tables"][name] = {
            "rows": len(rows),
            "keys_unique": len(set(table_keys)) == len(table_keys),
            "missing_from_table": sorted(key_set - set(table_keys))[:20],
            "extra_in_table": sorted(set(table_keys) - key_set)[:20],
            "missing_count": len(key_set - set(table_keys)),
            "extra_count": len(set(table_keys) - key_set),
        }

    # Where two sources carry the same column, keep the first and record any
    # value that disagrees rather than silently letting one win.
    ordered_sources = [
        ("footprints", None),
        ("sample_inventory", indexed["sample_inventory"]),
        ("s1_dw_timing", indexed["s1_dw_timing"]),
        ("dw_label_annotations", indexed["dw_label_annotations"]),
        ("jrc_gsw_annotations", indexed["jrc_gsw_annotations"]),
        ("pld_annotations", indexed["pld_annotations"]),
        ("coastline_annotations", indexed["coastline_annotations"]),
    ]

    fields: list[str] = []
    field_owner: dict[str, str] = {}
    for name, table in ordered_sources:
        columns = (
            list(footprints[0]["properties"].keys())
            if table is None
            else list(next(iter(table.values())).keys())
        )
        for column in columns:
            if column not in field_owner:
                field_owner[column] = name
                fields.append(column)

    conflicts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    geometries: list[dict[str, Any]] = []
    for feature in footprints:
        key = feature["properties"]["tile_name"]
        record: dict[str, Any] = {}
        for name, table in ordered_sources:
            row = feature["properties"] if table is None else table.get(key, {})
            for column, value in row.items():
                text = "" if value is None else str(value)
                if column not in record or record[column] in (None, ""):
                    record[column] = text
                elif text and not _same_value(record[column], text):
                    conflicts["{}:{}".format(name, column)] += 1
        record.update(
            dw_granule_stats(
                record.get("dw_image_ids", ""), record.get("nearest_s1_datetime_utc", ""), granule_times
            )
        )
        records.append(record)
        geometries.append(feature["geometry"])

    for field in DW_STAT_FIELDS:
        if field not in fields:
            fields.append(field)
    report["main"]["conflicts"] = dict(conflicts)
    report["main"]["attribute_count"] = len(fields)
    return records, geometries, fields


def _same_value(left: str, right: str) -> bool:
    """Compare as numbers when both sides are numeric, else as text."""
    if left == right:
        return True
    try:
        return float(left) == float(right)
    except ValueError:
        return False


def build_unified(
    main_records: list[dict[str, Any]],
    main_geometries: list[dict[str, Any]],
    main_fields: list[str],
    audit: Path,
    report: dict[str, Any],
    granule_times: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Stack the supplement onto the main set over a shared core schema."""
    supplement = json.loads((audit / "supplement_sample_footprints.geojson").read_text())["features"]
    supplement_fields = list(supplement[0]["properties"].keys())

    main_ids = {record["tile_name"] for record in main_records}
    supplement_ids = {feature["properties"]["candidate_id"] for feature in supplement}
    report["unified"] = {
        "main_rows": len(main_records),
        "supplement_rows": len(supplement),
        "total_rows": len(main_records) + len(supplement),
        "id_collisions": sorted(main_ids & supplement_ids),
        "shared_native_fields": sorted(set(main_fields) & set(supplement_fields)),
        "main_only_fields": sorted(set(main_fields) - set(supplement_fields)),
        "supplement_only_fields": sorted(set(supplement_fields) - set(main_fields)),
    }

    records: list[dict[str, Any]] = []
    geometries: list[dict[str, Any]] = []

    for record, geometry in zip(main_records, main_geometries):
        row = dict(record)
        timestamp = row.get("nearest_s1_datetime_utc") or ""
        row.update(
            {
                "sample_id": row["tile_name"],
                "sample_set": "main",
                "target_class": "river_node",
                "dw_water_fraction": row.get("dw_label_water_fraction"),
                "dw_water_fraction_bin": water_fraction_bin(row.get("dw_label_water_fraction")),
                "s1_image_id": row.get("nearest_s1_image_id"),
                "s1_datetime_utc": row.get("nearest_s1_datetime_utc"),
                "dw_image_id": row.get("nearest_dw_image_id"),
                "dw_datetime_utc": row.get("nearest_dw_datetime_utc"),
                "s1_scene_id": scene_id(row.get("nearest_s1_image_id")),
                "dw_scene_id": scene_id(row.get("nearest_dw_image_id")),
                "acquisition_year": timestamp[:4],
                "acquisition_month": timestamp[5:7].lstrip("0"),
            }
        )
        records.append(row)
        geometries.append(geometry)

    for feature in supplement:
        source = feature["properties"]
        row: dict[str, Any] = {key: "" if value is None else str(value) for key, value in source.items()}
        timestamp = row.get("s1_datetime_utc") or ""
        row.update(
            {
                "sample_id": row["candidate_id"],
                "sample_set": "supplement",
                "target_class": row.get("primary_target"),
                "dw_water_fraction_bin": water_fraction_bin(row.get("dw_water_fraction")),
                "s1_scene_id": scene_id(row.get("s1_image_id")),
                "dw_scene_id": scene_id(row.get("dw_image_id")),
                "acquisition_year": timestamp[:4],
                "acquisition_month": timestamp[5:7].lstrip("0"),
            }
        )
        row.update({key: str(value) for key, value in ring_bounds(feature["geometry"]["coordinates"][0]).items()})
        row.update(dw_granule_stats(row.get("dw_image_ids", ""), row.get("s1_datetime_utc", ""), granule_times))
        records.append(row)
        geometries.append(feature["geometry"])

    fields = list(UNIFIED_CORE)
    for field in main_fields + supplement_fields + sorted(DERIVED_FIELDS):
        if field not in fields and any(field in record for record in records):
            fields.append(field)
    for record in records:
        for field in fields:
            record.setdefault(field, "")

    # A native bin label the supplement publishes; recomputed for both sets above.
    mismatched = sum(
        1
        for record in records
        if record.get("water_fraction_bin") and record["water_fraction_bin"] != record["dw_water_fraction_bin"]
    )
    report["unified"]["native_bin_mismatches"] = mismatched
    report["unified"]["attribute_count"] = len(fields)
    report["unified"]["pair_status_vocabulary"] = dict(Counter(record["pair_status"] for record in records))
    report["unified"]["dw_multi_date_samples"] = sum(1 for record in records if str(record.get("dw_multi_date")) == "1")
    return records, geometries, fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "audit",
        help="directory holding the audit CSVs and footprint GeoJSONs",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to --audit-dir")
    parser.add_argument(
        "--granule-times",
        type=Path,
        default=None,
        help="DW granule time cache from verify_dw_acquisition_times.py (defaults to the audit dir copy)",
    )
    args = parser.parse_args()

    audit = args.audit_dir
    output = args.output_dir or audit
    output.mkdir(parents=True, exist_ok=True)

    granule_path = args.granule_times or audit / "dw_granule_times.csv"
    granule_times = load_granule_times(granule_path)
    report: dict[str, Any] = {
        "derived_fields": DERIVED_FIELDS,
        "dw_granule_times": {"source": str(granule_path), "granules": len(granule_times)},
    }
    if not granule_times:
        print("WARNING: {} not found; DW timing fields will be empty".format(granule_path))

    main_records, main_geometries, main_fields = build_main(audit, report, granule_times)
    main_types = column_types(main_records, main_fields)
    main_typed = typed_properties(main_records, main_fields, main_types)
    write_geojson(output / "enriched_sample_footprints.geojson", main_typed, main_geometries, "tile_name")
    write_csv(output / "enriched_sample_footprints.csv", main_typed, main_fields)

    unified_records, unified_geometries, unified_fields = build_unified(
        main_records, main_geometries, main_fields, audit, report, granule_times
    )
    unified_types = column_types(unified_records, unified_fields)
    unified_typed = typed_properties(unified_records, unified_fields, unified_types)
    write_geojson(output / "unified_sample_footprints.geojson", unified_typed, unified_geometries, "sample_id")
    write_csv(output / "unified_sample_footprints.csv", unified_typed, unified_fields)

    keep = [
        (record, geometry)
        for record, geometry in zip(unified_typed, unified_geometries)
        if str(record.get("dw_multi_date")) != "1"
    ]
    dropped = [record for record in unified_typed if str(record.get("dw_multi_date")) == "1"]
    write_geojson(
        output / "clean_sample_footprints.geojson",
        [record for record, _ in keep],
        [geometry for _, geometry in keep],
        "sample_id",
    )
    write_csv(output / "clean_sample_footprints.csv", [record for record, _ in keep], unified_fields)
    write_geojson(
        output / "clean_sample_centroids.geojson",
        [record for record, _ in keep],
        [
            {"type": "Point", "coordinates": [record["centroid_lon"], record["centroid_lat"]]}
            for record, _ in keep
        ],
        "sample_id",
    )
    exclusion_fields = [
        "sample_id", "sample_set", "split", "target_class", "dw_granule_count",
        "dw_date_count", "dw_spread_hours", "dw_gap_hours_verified", "exclusion_reason",
    ]
    write_csv(
        output / "excluded_samples.csv",
        [
            dict(
                {field: record.get(field) for field in exclusion_fields},
                exclusion_reason="dw_label_mosaic_spans_multiple_acquisition_dates",
            )
            for record in dropped
        ],
        exclusion_fields,
    )
    report["clean"] = {
        "rows": len(keep),
        "excluded": len(dropped),
        "excluded_ids": sorted(record["sample_id"] for record in dropped),
        "excluded_by_target_class": dict(Counter(record["target_class"] for record in dropped)),
        "excluded_by_split": dict(Counter(record["split"] for record in dropped)),
        "criterion": "dw_multi_date == 1",
    }

    report["main"]["fields"] = main_fields
    report["main"]["field_types"] = main_types
    report["unified"]["fields"] = unified_fields
    report["unified"]["field_types"] = unified_types
    (output / "enriched_footprints_join_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print("enriched_sample_footprints: {} features, {} attributes".format(len(main_typed), len(main_fields)))
    print("unified_sample_footprints:  {} features, {} attributes".format(len(unified_typed), len(unified_fields)))
    print("clean_sample_footprints:    {} features ({} excluded)".format(len(keep), len(dropped)))
    print("clean_sample_centroids:    {} points".format(len(keep)))
    print("join conflicts: {}".format(report["main"]["conflicts"] or "none"))


if __name__ == "__main__":
    main()
