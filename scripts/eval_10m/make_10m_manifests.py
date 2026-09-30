#!/usr/bin/env python3
"""Inference manifests for the 10 m runs (same columns as the existing manifests; only s1_path / aef_path change).

Writes to <out>/manifests/:
  inputs_10m_t.csv                 exact scene_id S1 + acquisition-year AEF, 768 x 768 px grid (53 scenes)
  inputs_10m_tminus1.csv           same S1 + previous-year AEF
  inputs_10m_daymosaic_t.csv       control, 4 scenes: same-day mosaic S1 (the 3 m input's acquisition) + t AEF
  inputs_10m_daymosaic_tminus1.csv control, 4 scenes: same, previous-year AEF
  inputs_10m_nomargin_t.csv        control: inputs cropped to the label footprint (no context margin)
  inputs_10m_nomargin_tminus1.csv  control: same, previous-year AEF
The no-margin inputs are pixel-exact crops of the 768 x 768 rasters (the footprint window in grid.json).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import rasterio
from rasterio.windows import Window, transform as window_transform

from paths import env_root  # noqa: E402  (environment variables; see README.md)

MAN = env_root("SWM_S1ML") / "intercomparison_s1aef/manifests"
DAYMOSAIC = ["SID05", "SID21", "SID61", "SID63"]


def crop(src_path: Path, dst_path: Path, fw: dict) -> None:
    win = Window(fw["col_off"], fw["row_off"], fw["width"], fw["height"])
    with rasterio.open(src_path) as src:
        data = src.read(window=win)
        prof = src.profile.copy()
        prof.update(width=fw["width"], height=fw["height"], transform=window_transform(win, src.transform))
        prof.pop("blockxsize", None); prof.pop("blockysize", None); prof["tiled"] = False
        descr = src.descriptions
    with rasterio.open(dst_path, "w", **prof) as dst:
        dst.write(data)
        dst.descriptions = descr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    (args.out / "manifests").mkdir(parents=True, exist_ok=True)
    src = {"t": list(csv.DictReader((MAN / "inference_inputs_primary_protocol_permanent.csv").open())),
           "tminus1": list(csv.DictReader((MAN / "inference_inputs_nearest_SID_tminus1.csv").open()))}
    fields = list(src["t"][0].keys())
    for which, rows in src.items():
        out = {"": [], "daymosaic_": [], "nomargin_": []}
        for r in rows:
            sid = r["sample_id"]
            g = json.loads((args.inputs_root / sid / "grid.json").read_text())
            s1 = Path(g["paths"]["s1"]); aef = Path(g["paths"]["aef_t" if which == "t" else "aef_tminus1"])
            assert s1.exists() and aef.exists(), sid
            out[""].append({**r, "s1_path": str(s1), "aef_path": str(aef), "aef_resolution": "10m_grid"})
            if sid in DAYMOSAIC:
                dm = args.inputs_root / sid / f"{sid}_s1_daymosaic_10m.tif"
                assert dm.exists(), dm
                out["daymosaic_"].append({**r, "s1_path": str(dm), "aef_path": str(aef), "aef_resolution": "10m_grid"})
            fw = g["footprint_window"]
            nm_dir = args.inputs_root / sid / "nomargin"
            nm_dir.mkdir(exist_ok=True)
            nm_s1, nm_aef = nm_dir / s1.name.replace("_10m", "_10m_nomargin"), nm_dir / aef.name.replace("_10m", "_10m_nomargin")
            for a, b in ((s1, nm_s1), (aef, nm_aef)):
                if not b.exists():
                    crop(a, b, fw)
            out["nomargin_"].append({**r, "s1_path": str(nm_s1), "aef_path": str(nm_aef), "aef_resolution": "10m_nomargin"})
        for prefix, rr in out.items():
            path = args.out / "manifests" / f"inputs_10m_{prefix}{which}.csv"
            with path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rr)
            print(path, len(rr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
