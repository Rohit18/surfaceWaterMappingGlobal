#!/usr/bin/env python3
"""Control input for the 4 scenes whose existing 3 m S1 input is a same-day mosaic of more than one pass.

Reproduces the existing 3 m input's acquisition choice at 10 m by calling the unchanged functions of
download_intercomparison_s1aef_label_products.py: s1_collection_for_day (IW, VV+VH, same UTC day, sorted by time),
s1_image_from_collection (mosaic(), last image on top, float, clip) and download_ee_label. Only the grid differs:
the 10 m grid of build_10m_inputs.py, and the clip/filter region is that grid's WGS84 bbox (the original used the
label bbox, which would leave the new margin empty).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from rasterio.transform import Affine

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_intercomparison_s1aef_label_products as dl  # noqa: E402
from paths import env_root  # noqa: E402  (environment variables; see README.md)

MAN = env_root("SWM_S1ML") / "intercomparison_s1aef/manifests/inference_inputs_primary_protocol_permanent.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-root", type=Path, required=True)
    ap.add_argument("--samples", nargs="+", default=["SID05", "SID21", "SID61", "SID63"])
    ap.add_argument("--ee-project", default="ee-mukherjeerishi")
    args = ap.parse_args()
    import ee
    import google.auth
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/earthengine",
                                           "https://www.googleapis.com/auth/cloud-platform"])
    ee.Initialize(credentials=creds, project=args.ee_project)
    prim = {r["sample_id"]: r for r in csv.DictReader(MAN.open())}
    for sid in args.samples:
        g = json.loads((args.inputs_root / sid / "grid.json").read_text())
        target = datetime.strptime(prim[sid]["date"], "%Y-%m-%d").date()
        grid = dl.LabelGrid(path=Path(prim[sid]["label_path"]), sample_id=sid, crs=g["crs"],
                            transform=Affine(*g["transform"]), width=g["width"], height=g["height"],
                            bounds=tuple(g["bounds"]), bbox_wgs84=tuple(g["bbox_wgs84"]),
                            region_wgs84=dl.polygon_from_bounds(tuple(g["bbox_wgs84"])))
        region = dl.make_region(ee, grid)
        collection = dl.s1_collection_for_day(ee, region, target)
        count, scene_ids, datetimes = dl.collection_metadata(collection)
        out = args.inputs_root / sid / f"{sid}_s1_daymosaic_10m.tif"
        dl.download_ee_label(dl.s1_image_from_collection(collection, region), grid, f"{sid}_{target}_s1", out, 300)
        rec = {"sample_id": sid, "date": str(target), "image_count": count, "scene_ids": scene_ids,
               "datetimes": datetimes, "manifest_scene_id": prim[sid]["scene_id"], "output": str(out)}
        (args.inputs_root / sid / "daymosaic_control.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
