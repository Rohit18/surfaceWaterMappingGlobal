#!/usr/bin/env python3
"""Build the immutable 4,678 + 600 paper-retraining triplet and split manifests."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


FIELDS = (
    "sample_id", "tile_name", "s1_path", "aef_path", "label_path", "group_id",
    "source_id", "split", "sampling_frame", "primary_target",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve(name: str, roots: list[Path], kind: str) -> Path:
    matches = [root / name for root in roots if (root / name).exists()]
    if not matches:
        raise FileNotFoundError("Missing {} raster for {} in {}".format(kind, name, roots))
    # Roots are ordered overlays: the primary dataset wins when present and
    # later recovery roots only fill gaps.  Recovery trees can contain symlinks
    # or regenerated versions of a primary raster, so multiple matches are not
    # themselves an ambiguity.
    return matches[0].resolve()


def aef_index(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("*.tif"):
        if path.name in result:
            raise ValueError("Duplicate AEF filename: {}".format(path.name))
        result[path.name] = path.resolve()
    return result


def existing_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    aef = aef_index(args.existing_aef_root)
    result = []
    for row in read_csv(args.existing_index):
        name = row["tile_name"]
        if name not in aef:
            raise FileNotFoundError("Missing existing AEF raster: {}".format(name))
        result.append(
            {
                "sample_id": args.sample_id, "tile_name": name,
                "s1_path": str(resolve(name, args.existing_s1_root, "S1")),
                "aef_path": str(aef[name]),
                "label_path": str(resolve(name, args.existing_label_root, "label")),
                "group_id": "SWORD_v16:{}".format(row["sword_node_id"]),
                "source_id": row["sword_node_id"], "split": row["split"],
                "sampling_frame": "SWORD_v16_river_node", "primary_target": "river_targeted",
            }
        )
    return result


def supplement_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_paths = sorted((args.supplement_root / "manifests").glob("materialize_*_of_*.csv"))
    if not manifest_paths:
        raise FileNotFoundError("No supplement materialization manifests found")
    source = {row["candidate_id"]: row for row in read_csv(args.supplement_index)}
    materialized: dict[str, dict[str, str]] = {}
    for path in manifest_paths:
        for row in read_csv(path):
            if row["candidate_id"] in materialized:
                raise ValueError("Duplicate materialized candidate: {}".format(row["candidate_id"]))
            materialized[row["candidate_id"]] = row
    missing = sorted(set(source).difference(materialized))
    errors = sorted(key for key, row in materialized.items() if row.get("status") != "ok")
    extras = sorted(set(materialized).difference(source))
    if missing or errors or extras:
        raise RuntimeError(
            "Supplement is incomplete: missing={}, errors={}, extras={}; examples={}".format(
                len(missing), len(errors), len(extras), (missing + errors + extras)[:8]
            )
        )
    result = []
    for candidate_id, selected in source.items():
        row = materialized[candidate_id]
        paths = [Path(row[key]) for key in ("s1_path", "aef_path", "label_path")]
        if not all(path.exists() for path in paths):
            raise FileNotFoundError("Materialized files are missing for {}".format(candidate_id))
        result.append(
            {
                "sample_id": args.sample_id, "tile_name": row["tile_name"],
                "s1_path": str(paths[0].resolve()), "aef_path": str(paths[1].resolve()),
                "label_path": str(paths[2].resolve()), "group_id": selected["group_id"],
                "source_id": selected["source_feature_id"], "split": selected["split"],
                "sampling_frame": selected["source_dataset"],
                "primary_target": selected["primary_target"],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-index", type=Path, default=Path("metadata/training_samples.csv"))
    parser.add_argument(
        "--supplement-index", type=Path,
        default=Path("outputs/audit/supplement_samples.csv"),
    )
    parser.add_argument(
        "--existing-s1-root", type=Path, action="append",
        default=[
            Path("/pscratch/sd/r/rohit9/S1ML/training_mosaics/s1_multiband"),
            Path("/pscratch/sd/r/rohit9/S1ML/training_mosaics/s1_multiband_dwlabel_recovered"),
        ],
    )
    parser.add_argument(
        "--existing-label-root", type=Path, action="append",
        default=[
            Path("/pscratch/sd/r/rohit9/S1ML/training_mosaics/dw_binary"),
            Path("/pscratch/sd/r/rohit9/S1ML/training_mosaics/dw_label_water_pm1d_recovered"),
        ],
    )
    parser.add_argument(
        "--existing-aef-root", type=Path,
        default=Path("/pscratch/sd/r/rohit9/S1ML/training_embeddings/alphaearth_v1_annual_int8"),
    )
    parser.add_argument(
        "--supplement-root", type=Path,
        default=Path("/pscratch/sd/r/rohit9/S1ML/training_supplement/v1"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/pscratch/sd/r/rohit9/S1ML/training_supplement/paper_retrain_v1"),
    )
    parser.add_argument("--sample-id", default="paper_sword4678_plus_openwater600_v1")
    args = parser.parse_args()

    original = existing_rows(args)
    supplement = supplement_rows(args)
    rows = original + supplement
    names = [row["tile_name"] for row in rows]
    if len(rows) != 5278 or len(set(names)) != 5278:
        raise RuntimeError("Expected 5,278 unique triplets, got {} rows and {} names".format(len(rows), len(set(names))))
    splits = Counter(row["split"] for row in rows)
    if splits != {"train": 4222, "valid": 1056}:
        raise RuntimeError("Unexpected fixed split counts: {}".format(dict(splits)))
    output = args.output_root / "triplets.csv"
    fixed = args.output_root / "fixed_split.csv"
    summary = args.output_root / "manifest_summary.json"
    write_csv(output, FIELDS, rows)
    write_csv(fixed, ("tile_name", "split"), rows)
    report = {
        "triplets": len(rows), "original_triplets": len(original),
        "supplement_triplets": len(supplement), "split_counts": dict(splits),
        "primary_target_counts": dict(Counter(row["primary_target"] for row in rows)),
        "triplet_manifest": str(output.resolve()), "fixed_split_manifest": str(fixed.resolve()),
    }
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
