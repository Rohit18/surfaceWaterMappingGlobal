#!/usr/bin/env python3
"""Extract the 53 paper evaluation scenes from the public GSWD scene table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = (
    "sample_id", "reference_date", "sentinel1_grd_scene", "sentinel1_slc_scene",
    "centroid_lat", "centroid_lon", "reference_datetime", "s1_acquisition_datetime",
    "time_delta_hours", "reference_sensor", "aef_year_t", "aef_year_t_minus_1",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-table", type=Path, required=True)
    parser.add_argument("--paper-per-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("metadata/evaluation_samples.csv"))
    args = parser.parse_args()

    with args.paper_per_scene.open(newline="") as handle:
        selected = {row["sid"] for row in csv.DictReader(handle)}
    with args.scene_table.open(newline="") as handle:
        source = {row["SampleID_clean"]: row for row in csv.DictReader(handle)}
    missing = selected - set(source)
    if missing:
        raise ValueError("Paper samples missing from source table: {}".format(sorted(missing)))

    rows = []
    for sample_id in sorted(selected):
        row = source[sample_id]
        acquisition_year = int(row["acquisition_datetime"][:4])
        rows.append(
            {
                "sample_id": sample_id,
                "reference_date": row["Date"],
                "sentinel1_grd_scene": row["s1_grd_scene"],
                "sentinel1_slc_scene": row["s1_slc_scene"],
                "centroid_lat": row["Latitude"],
                "centroid_lon": row["Longitude"],
                "reference_datetime": row["reference_datetime"],
                "s1_acquisition_datetime": row["acquisition_datetime"],
                "time_delta_hours": row["time_delta_hours"],
                "reference_sensor": row["reference_sensor"],
                "aef_year_t": acquisition_year,
                "aef_year_t_minus_1": acquisition_year - 1,
            }
        )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print({"output": str(output), "scenes": len(rows)})


if __name__ == "__main__":
    main()
