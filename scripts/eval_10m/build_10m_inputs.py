#!/usr/bin/env python3
"""Build 10 m Sentinel-1 and AlphaEarth inputs for the 53 GSWD scenes (GRSL revision, Task 1).

No new export logic. The two existing export functions are called unchanged; only the output grid differs:
- S1:  select_intercomparison_nearest_s1.download_scene(ee, sample, scene_id, out_path, timeout)
       -> ee.Image("COPERNICUS/S1_GRD/<scene_id>").select(["VV","VH","angle"]).float(), getDownloadURL with
          crs / crs_transform / dimensions of the grid (Earth Engine default nearest resampling).
- AEF: fetch_alphaearth_embeddings.write_embedding(bbox_wgs84, out, year, crs, transform, w, h, cache_dir)
       (Source Cooperative annual embeddings v1, 64 int8 bands, nearest, nodata -128), called exactly as
       download_intercomparison_s1aef_label_products.download_alphaearth does.

Grid per scene: label CRS; 10 m pixels; edges at integer multiples of 10 m in that CRS; centred on the label
footprint; at least --min-size px (768 = 7.68 km) on each side and always covering the footprint.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_alphaearth_embeddings as alphaearth  # noqa: E402
import select_intercomparison_nearest_s1 as nearest  # noqa: E402
from paths import env_root  # noqa: E402  (environment variables; see README.md)

MANIFESTS = env_root("SWM_S1ML") / "intercomparison_s1aef/manifests"
PRIMARY = MANIFESTS / "inference_inputs_primary_protocol_permanent.csv"
TMINUS1 = MANIFESTS / "inference_inputs_nearest_SID_tminus1.csv"
AEF_CACHE = env_root("SWM_S1ML") / "intercomparison_s1aef/cache"  # index only (tiles are streamed)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def grid_10m(label_path: Path, pixel: float, min_size: int) -> dict:
    with rasterio.open(label_path) as src:
        crs = src.crs.to_string()
        left, bottom, right, top = src.bounds
    cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
    fp_w = math.ceil((right - left) / pixel) + 2
    fp_h = math.ceil((top - bottom) / pixel) + 2
    width, height = max(min_size, fp_w), max(min_size, fp_h)
    x0 = round((cx - width * pixel / 2.0) / pixel) * pixel
    y1 = round((cy + height * pixel / 2.0) / pixel) * pixel
    # grow if rounding left the footprint uncovered
    while x0 > left:
        x0 -= pixel; width += 1
    while x0 + width * pixel < right:
        width += 1
    while y1 < top:
        y1 += pixel; height += 1
    while y1 - height * pixel > bottom:
        height += 1
    transform = Affine(pixel, 0.0, x0, 0.0, -pixel, y1)
    g_bounds = (x0, y1 - height * pixel, x0 + width * pixel, y1)
    bbox_wgs84 = tuple(float(v) for v in transform_bounds(crs, "EPSG:4326", *g_bounds, densify_pts=21))
    # label footprint as a pixel window of the 10 m grid (outward snapped) - used for the no-margin control
    col0 = int(math.floor((left - x0) / pixel)); col1 = int(math.ceil((right - x0) / pixel))
    row0 = int(math.floor((y1 - top) / pixel)); row1 = int(math.ceil((y1 - bottom) / pixel))
    return {"crs": crs, "transform": transform, "width": width, "height": height, "bounds": g_bounds,
            "bbox_wgs84": bbox_wgs84, "label_bounds": (left, bottom, right, top),
            "footprint_window": {"col_off": col0, "row_off": row0, "width": col1 - col0, "height": row1 - row0},
            "margin_m": {"left": left - g_bounds[0], "right": g_bounds[2] - right,
                         "top": g_bounds[3] - top, "bottom": bottom - g_bounds[1]}}


def valid_fraction(path: Path, window=None) -> dict:
    with rasterio.open(path) as src:
        data = src.read(window=window)
        nod = src.nodata
        if data.dtype.kind == "f":
            bad = ~np.isfinite(data)
            if nod is not None and np.isfinite(nod):
                bad |= data == nod
        else:
            bad = data == nod if nod is not None else np.zeros(data.shape, bool)
        return {"count": int(src.count), "dtype": str(src.dtypes[0]), "nodata": None if nod is None else float(nod),
                "crs": src.crs.to_string(), "transform": list(src.transform)[:6], "width": src.width,
                "height": src.height, "valid_fraction_all_bands": float((~bad.any(axis=0)).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--products", nargs="+", choices=("grid", "s1", "aef_t", "aef_tminus1"), required=True)
    ap.add_argument("--pixel-size", type=float, default=10.0)
    ap.add_argument("--min-size", type=int, default=768)
    ap.add_argument("--only-sample", nargs="+", default=None)
    ap.add_argument("--ee-project", default="ee-mukherjeerishi")
    ap.add_argument("--ee-adc", action="store_true", help="initialise Earth Engine with application-default credentials")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    prim = {r["sample_id"]: r for r in csv.DictReader(PRIMARY.open())}
    tm1 = {r["sample_id"]: r for r in csv.DictReader(TMINUS1.open())}
    sids = sorted(prim) if not args.only_sample else args.only_sample
    ee = None
    report_rows = []
    for sid in sids:
        row = prim[sid]
        out_dir = args.out_root / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        g = grid_10m(Path(row["label_path"]), args.pixel_size, args.min_size)
        year_t = int(row["date"][:4])
        year_t1 = int(Path(tm1[sid]["aef_path"]).stem.split("_")[0])
        assert year_t1 == year_t - 1, (sid, year_t, year_t1)
        paths = {"s1": out_dir / f"{sid}_s1_10m.tif",
                 "aef_t": out_dir / f"{sid}_aef_{year_t}_10m.tif",
                 "aef_tminus1": out_dir / f"{sid}_aef_{year_t1}_10m.tif"}
        grid_json = out_dir / "grid.json"
        info = {"sample_id": sid, "label_path": row["label_path"], "scene_id": row["scene_id"],
                "acquisition_datetime": row["acquisition_datetime"], "aef_year_t": year_t,
                "aef_year_tminus1": year_t1, "pixel_size": args.pixel_size,
                "crs": g["crs"], "transform": list(g["transform"])[:6], "width": g["width"], "height": g["height"],
                "bounds": g["bounds"], "bbox_wgs84": g["bbox_wgs84"], "label_bounds": g["label_bounds"],
                "footprint_window": g["footprint_window"], "margin_m": g["margin_m"],
                "paths": {k: str(v) for k, v in paths.items()}}
        if grid_json.exists():
            info = {**json.loads(grid_json.read_text()), **{k: v for k, v in info.items() if k != "exports"}}
        info.setdefault("exports", {})

        if "s1" in args.products and (args.overwrite or not paths["s1"].exists()):
            if ee is None:
                if args.ee_adc:  # gcloud application-default credentials (stored OAuth token rejected on 2026-09-28)
                    import ee as ee_module
                    import google.auth
                    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/earthengine",
                                                           "https://www.googleapis.com/auth/cloud-platform"])
                    ee_module.Initialize(credentials=creds, project=args.ee_project)
                    ee = ee_module
                else:
                    ee = nearest.init_ee(args.ee_project)
            image = ee.Image("COPERNICUS/S1_GRD/{}".format(row["scene_id"]))
            props = image.toDictionary(["system:index", "instrumentMode", "orbitProperties_pass",
                                        "transmitterReceiverPolarisation", "system:time_start"]).getInfo()
            label_grid = nearest.LabelGrid(path=paths["s1"], crs=g["crs"], transform=g["transform"],
                                           width=g["width"], height=g["height"])
            sample = nearest.Sample(sample_id=sid, row={}, label=label_grid)
            nearest.download_scene(ee, sample, row["scene_id"], paths["s1"], args.timeout)
            info["exports"]["s1"] = {"ee_asset": "COPERNICUS/S1_GRD/{}".format(row["scene_id"]),
                                     "ee_properties": props, "updated_at": utc_now(),
                                     "function": "select_intercomparison_nearest_s1.download_scene"}
        for key, year in (("aef_t", year_t), ("aef_tminus1", year_t1)):
            if key in args.products and (args.overwrite or not paths[key].exists()):
                sources = alphaearth.write_embedding(
                    bbox_wgs84=g["bbox_wgs84"], output_path=paths[key], year=year, dst_crs=g["crs"],
                    dst_transform=g["transform"], width=g["width"], height=g["height"], cache_dir=AEF_CACHE,
                    output_mode=alphaearth.DEFAULT_OUTPUT_MODE, compression=alphaearth.DEFAULT_COMPRESSION,
                    timeout=args.timeout)
                info["exports"][key] = {"year": year, "sources": sources, "updated_at": utc_now(),
                                        "function": "fetch_alphaearth_embeddings.write_embedding"}

        # verification
        fw = g["footprint_window"]
        win = ((fw["row_off"], fw["row_off"] + fw["height"]), (fw["col_off"], fw["col_off"] + fw["width"]))
        checks = {}
        for key, p in paths.items():
            if p.exists():
                v = valid_fraction(p)
                v["valid_fraction_footprint"] = valid_fraction(p, window=win)["valid_fraction_all_bands"]
                v["grid_ok"] = (v["crs"] == g["crs"] and v["width"] == g["width"] and v["height"] == g["height"]
                                and np.allclose(v["transform"], list(g["transform"])[:6]))
                v["band_count_ok"] = v["count"] == (3 if key == "s1" else 64)
                checks[key] = v
        if "s1" in info["exports"]:
            checks.setdefault("s1", {})["product_id_ok"] = (
                info["exports"]["s1"]["ee_properties"].get("system:index") == row["scene_id"])
        info["checks"] = checks
        grid_json.write_text(json.dumps(info, indent=2, default=str) + "\n")
        rr = {"sample_id": sid, "crs": g["crs"], "x0": g["transform"].c, "y0": g["transform"].f,
              "pixel_size": args.pixel_size, "width": g["width"], "height": g["height"],
              "label_left": g["label_bounds"][0], "label_bottom": g["label_bounds"][1],
              "label_right": g["label_bounds"][2], "label_top": g["label_bounds"][3],
              "margin_left_m": g["margin_m"]["left"], "margin_right_m": g["margin_m"]["right"],
              "margin_top_m": g["margin_m"]["top"], "margin_bottom_m": g["margin_m"]["bottom"],
              "fp_col_off": fw["col_off"], "fp_row_off": fw["row_off"], "fp_width": fw["width"],
              "fp_height": fw["height"], "scene_id": row["scene_id"], "aef_year_t": year_t,
              "aef_year_tminus1": year_t1}
        for key in ("s1", "aef_t", "aef_tminus1"):
            c = checks.get(key, {})
            rr[f"{key}_grid_ok"] = c.get("grid_ok", "")
            rr[f"{key}_bands_ok"] = c.get("band_count_ok", "")
            rr[f"{key}_valid_all"] = c.get("valid_fraction_all_bands", "")
            rr[f"{key}_valid_footprint"] = c.get("valid_fraction_footprint", "")
        rr["s1_product_id_ok"] = checks.get("s1", {}).get("product_id_ok", "")
        report_rows.append(rr)
        print(json.dumps({"sample_id": sid, **{k: rr[k] for k in rr if k.endswith(("_ok", "_valid_footprint"))}}),
              flush=True)

    if report_rows and not args.only_sample:
        with (args.out_root / "grids_10m.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(report_rows[0]))
            w.writeheader(); w.writerows(report_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
