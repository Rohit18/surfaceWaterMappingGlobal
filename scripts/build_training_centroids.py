#!/usr/bin/env python3
"""Write the centroids of the 5,278 training tiles of the released models as GeoJSON points.

Inputs: the 4,678-tile index (`metadata/training_samples.csv`), the 600-tile open-water
supplement (`outputs/audit/supplement_samples.csv`), the fixed split of the acquisition-year
runs (`fixed_split.csv`) and the previous-year training manifest (`triplets.csv`).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-index", type=Path, default=Path("metadata/training_samples.csv"))
    parser.add_argument("--supplement", type=Path, default=Path("outputs/audit/supplement_samples.csv"))
    parser.add_argument("--fixed-split", type=Path, required=True)
    parser.add_argument("--previous-year-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("metadata/training_sample_centroids.geojson"))
    args = parser.parse_args()

    split = {row["tile_name"]: row["split"] for row in read_csv(args.fixed_split)}
    previous_year = {row["tile_name"] for row in read_csv(args.previous_year_manifest)}

    samples = []
    for row in read_csv(args.training_index):
        samples.append({
            "tile_name": row["tile_name"],
            "sample_set": "sword_river_node",
            "sampling_frame": "SWORD_v16",
            "primary_target": "river_targeted",
            "source_feature_id": row["sword_node_id"],
            "s1_date": row["s1_date"],
            "s1_date_basis": "legacy_indexed_date",
            "aef_year": int(row["aef_year"]),
            "split": row["split"],
            "lon": float(row["centroid_lon"]),
            "lat": float(row["centroid_lat"]),
        })
    for row in read_csv(args.supplement):
        samples.append({
            "tile_name": row["candidate_id"] + ".tif",
            "sample_set": "open_water_supplement",
            "sampling_frame": row["source_dataset"],
            "primary_target": row["primary_target"],
            "source_feature_id": row["source_feature_id"],
            "s1_date": row["s1_datetime_utc"][:10],
            "s1_date_basis": "s1_acquisition_utc",
            "aef_year": int(row["s1_datetime_utc"][:4]),
            "split": row["split"],
            "lon": float(row["centroid_lon"]),
            "lat": float(row["centroid_lat"]),
        })

    names = [sample["tile_name"] for sample in samples]
    if len(set(names)) != len(names) or set(names) != set(split):
        raise RuntimeError("The index and supplement do not match the fixed split")
    if not previous_year <= set(names):
        raise RuntimeError("The previous-year manifest has tiles outside the fixed split")
    for sample in samples:
        if sample["split"] != split[sample["tile_name"]]:
            raise RuntimeError("Split mismatch for {}".format(sample["tile_name"]))

    features = []
    for sample in samples:
        lon, lat = sample.pop("lon"), sample.pop("lat")
        sample["in_previous_year_model"] = sample["tile_name"] in previous_year
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
            "properties": sample,
        })
    # One feature per line keeps the file diffable.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        handle.write('{"type":"FeatureCollection","features":[\n')
        handle.write(",\n".join(json.dumps(feature, separators=(",", ":")) for feature in features))
        handle.write("\n]}\n")
    print("wrote {} features to {}".format(len(features), args.output))


if __name__ == "__main__":
    main()
