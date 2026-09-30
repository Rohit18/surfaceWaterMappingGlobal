#!/usr/bin/env python3
"""Export PNG layers for the 53 PlanetScope reference scenes from the 10 m-inference probabilities (Task A).

Modified copy of the bundle's 6_processing/code/e_bundle_exports/export_53_scene_layers.py. Unchanged: grid (the
label grid, ~3 m, 1024 x 1024), PNG encodings, overlay colour, stretch, error-map colours, six-panel layout, threshold
0.30, `load_scene` reading PlanetScope / label / OPERA with the same helpers (make_permanent_water_comparison_figures).
Changed:
  - probabilities: 10 m-inference rasters already resampled bilinearly to the label grid in
    revision_checks_20260929/predictions_10m_on_3m/ (S1-only k0, S1+AEF(t) k16, t-1 swap, t-1 trained; seed 42);
  - Sentinel-1 background and s1_vv_db / s1_vh_db: the S1 product the 10 m run used
    (revision_checks_20260929/inputs_10m/<SID>/<SID>_s1_10m.tif), resampled bilinearly to the label grid for display;
  - panel IoUs in layers.json and on the panels are computed on the paper's common valid mask (51,897,488 px over the
    53 scenes), which is also written as common_valid.png (255 = scored);
  - the scene table (scene_table.csv) with location, continent, label water fraction and common-valid fraction,
    used by make_figure1b_10m.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from PIL import Image
from rasterio.enums import Resampling
from shapely.geometry import Point, shape

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_permanent_water_comparison_figures import (  # noqa: E402
    LABEL_DIR, OPERA_DIR, PS_DIR, overlay, ps_rgb, read_on_grid, s1_background,
)
import score_ablations as SA  # noqa: E402  (common_mask: the paper's common valid mask)
import scoring as S  # noqa: E402
from paths import env_root  # noqa: E402  (environment variables; see README.md)

PREV = env_root("SWM_EVAL10M")
OUT = env_root("SWM_ABLATION10M")
R3 = PREV / "predictions_10m_on_3m"
MODELS = {
    "s1only": R3 / "openwater/width_k0_seed42_current_tta/probabilities",
    "s1aef_t": R3 / "openwater/width_k16_seed42_current_tta/probabilities",
    "s1aef_tminus1_swap": R3 / "openwater/width_k16_seed42_tminus1_tta/probabilities",
    "s1aef_tminus1_trained": R3 / "tminus1/width_k16_seed42_tminus1_tta/probabilities",
}
S1_10M = PREV / "inputs_10m/{0}/{0}_s1_10m.tif"
SCENES_CSV = env_root("SWM_GSWD_DIR") / "S1_Intercomparison_All_Scenes.csv"
NATURAL_EARTH = OUT / "fig1B/ancillary/ne_50m_admin_0_countries.geojson"
THRESHOLD = 0.30
ERROR_COLOURS = {"tn": (0.88, 0.88, 0.88), "tp": (0.0, 0.45, 0.80), "fp": (0.84, 0.15, 0.16),
                 "fn": (1.0, 0.60, 0.0), "nodata": (1.0, 1.0, 1.0)}


def find_prob(root: Path, sid: str) -> Path:
    matches = sorted(root.glob(f"*/{sid}_prob.tif"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{sid}: expected one probability under {root}, found {matches}")
    return matches[0]


def valid_of(array, nodata):
    valid = np.isfinite(array)
    if nodata is not None:
        valid &= array != nodata
    return valid


def iou(pred, label, valid):
    tp = int((pred & label & valid).sum())
    fp = int((pred & ~label & valid).sum())
    fn = int((~pred & label & valid).sum())
    return tp / max(tp + fp + fn, 1)


def save_gray(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="L").save(path, optimize=True)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8), mode="RGB").save(path, optimize=True)


def encode_prob(prob, valid):
    out = np.full(prob.shape, 255, np.uint8)
    out[valid] = np.round(250 * np.clip(prob[valid], 0, 1)).astype(np.uint8)
    return out


def encode_water(water, valid):
    out = np.full(water.shape, 128, np.uint8)
    out[valid] = np.where(water[valid], 255, 0)
    return out


def encode_db(db, lo, hi):
    out = np.full(db.shape, 255, np.uint8)
    ok = np.isfinite(db)
    out[ok] = np.round(250 * np.clip((db[ok] - lo) / (hi - lo), 0, 1)).astype(np.uint8)
    return out


def error_rgb(pred, label, valid):
    rgb = np.ones(pred.shape + (3,), np.float32)
    for name, mask in (("tn", ~pred & ~label), ("tp", pred & label), ("fp", pred & ~label), ("fn", ~pred & label)):
        rgb[mask & valid] = ERROR_COLOURS[name]
    return rgb


def load_scene(sid: str) -> dict:
    ref_path = find_prob(MODELS["s1aef_t"], sid)
    with rio.open(ref_path) as ref:
        grid = {"crs": ref.crs.to_string(), "transform": list(ref.transform)[:6],
                "width": ref.width, "height": ref.height, "date_folder": ref_path.parent.name}
        probs = {}
        for name, root in MODELS.items():
            array, nodata = read_on_grid(find_prob(root, sid), ref, Resampling.bilinear)
            probs[name] = (array, valid_of(array, nodata))
        label, _ = read_on_grid(LABEL_DIR / f"{sid}.tif", ref, Resampling.nearest)
        opera, opera_nodata = read_on_grid(OPERA_DIR / f"{sid}_opera_bwtr_binary.tif", ref, Resampling.nearest)
        s1_path = Path(str(S1_10M).format(sid))
        s1_db, _ = read_on_grid(s1_path, ref, Resampling.bilinear, indexes=[1, 2])
        display = s1_background(s1_path, ref)
        planet = ps_rgb(PS_DIR / f"{sid}.tif", ref)
    # The label grid is the probability grid, so read_on_grid returned the raw label values.
    lvalid = (label == 0) | (label == 1)
    common = SA.common_mask(sid, grid["date_folder"], lvalid)
    # Label files declare 0 as nodata although 0 is the evaluated non-water class (as in the published figure).
    label_water = label > 0.5
    return {"grid": grid, "probs": probs, "label": label_water, "label_valid": lvalid, "common": common,
            "opera": (opera == 1, valid_of(opera, opera_nodata)), "s1_db": s1_db, "display": display,
            "planet": planet, "s1_path": str(s1_path)}


def six_panel_images(scene: dict):
    s1, label, common = scene["display"], scene["label"], scene["common"]
    s1only, s1only_valid = scene["probs"]["s1only"]
    s1aef, s1aef_valid = scene["probs"]["s1aef_t"]
    opera, opera_valid = scene["opera"]
    s1only_water, s1aef_water = s1only >= THRESHOLD, s1aef >= THRESHOLD
    ious = {"s1only": iou(s1only_water, label, common), "s1aef_t": iou(s1aef_water, label, common),
            "opera": iou(opera, label, common)}
    panels = [
        (scene["planet"], "PlanetScope"),
        (np.repeat(s1[..., None], 3, axis=2), "Sentinel-1 (VV/VH)"),
        (overlay(s1, label), "Hand label"),
        (overlay(s1, s1only_water & s1only_valid), "S1-only\nIoU={:.2f}".format(ious["s1only"])),
        (overlay(s1, s1aef_water & s1aef_valid), "S1+AEF (ours)\nIoU={:.2f}".format(ious["s1aef_t"])),
        (overlay(s1, opera & opera_valid), "OPERA\nIoU={:.2f}".format(ious["opera"])),
    ]
    return panels, ious


def write_scene(sid: str, scene: dict, out: Path, dpi: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    label, common = scene["label"], scene["common"]
    save_rgb(out / "planetscope_rgb.png", scene["planet"])
    save_gray(out / "s1_display.png", np.round(255 * scene["display"]))
    save_gray(out / "s1_vv_db.png", encode_db(scene["s1_db"][0], -30.0, 5.0))
    save_gray(out / "s1_vh_db.png", encode_db(scene["s1_db"][1], -40.0, -5.0))
    save_gray(out / "label_water.png", np.where(label, 255, 0))
    save_gray(out / "common_valid.png", np.where(common, 255, 0))
    opera, opera_valid = scene["opera"]
    save_gray(out / "opera_water.png", encode_water(opera, opera_valid))
    save_rgb(out / "opera_error.png", error_rgb(opera, label, opera_valid))
    for name, (prob, valid) in scene["probs"].items():
        water = prob >= THRESHOLD
        save_gray(out / f"{name}_prob.png", encode_prob(prob, valid))
        save_gray(out / f"{name}_water_t030.png", encode_water(water, valid))
        save_rgb(out / f"{name}_error_t030.png", error_rgb(water, label, valid))
    panels, ious = six_panel_images(scene)
    fig, axes = plt.subplots(1, 6, figsize=(13.5, 2.5))
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image, interpolation="nearest")
        axis.set_title(title, fontsize=11, pad=3)
        axis.set_xticks([]); axis.set_yticks([])
    fig.subplots_adjust(left=0.008, right=0.992, bottom=0.02, top=0.82, wspace=0.18)
    fig.savefig(out / f"{sid}_six_panel.png", dpi=dpi, facecolor="white")
    plt.close(fig)
    all_ious = {name: iou(prob >= THRESHOLD, label, common) for name, (prob, _) in scene["probs"].items()}
    all_ious["opera"] = ious["opera"]
    meta = {"sample_id": sid, **scene["grid"], "threshold": THRESHOLD, "inference": "10 m, bilinear to label grid",
            "iou_mask": "paper common valid mask (common_valid.png)",
            "panel_water_iou": {k: round(v, 4) for k, v in all_ious.items()},
            "panel_water_iou_full": all_ious,
            "label_water_fraction": round(float(label[scene["label_valid"]].mean()), 4),
            "common_valid_fraction": round(float(common.mean()), 4),
            "common_valid_pixels": int(common.sum()),
            "s1_display_source": scene["s1_path"],
            "probability_sources": {k: str(find_prob(v, sid)) for k, v in MODELS.items()}}
    (out / "layers.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def locate(sids) -> dict:
    """Latitude/longitude from the scene CSV; country and continent from Natural Earth 1:50m admin-0 (point in
    polygon; if the point is in no polygon, the nearest polygon in degrees)."""
    rows = {r["SampleID_clean"]: r for r in csv.DictReader(SCENES_CSV.open())}
    feats = [(shape(f["geometry"]), f["properties"]) for f in json.loads(NATURAL_EARTH.read_text())["features"]]
    out = {}
    for sid in sids:
        lat, lon = float(rows[sid]["Latitude"]), float(rows[sid]["Longitude"])
        pt = Point(lon, lat)
        hit = [p for g, p in feats if g.contains(pt)]
        how = "contains"
        if not hit:
            hit = [min(feats, key=lambda gp: gp[0].distance(pt))[1]]
            how = "nearest"
        out[sid] = {"latitude": lat, "longitude": lon, "country": hit[0]["NAME"], "continent": hit[0]["CONTINENT"],
                    "subregion": hit[0]["SUBREGION"], "ne_match": how}
    return out


def one(args):
    sid, images_dir, dpi = args
    meta = write_scene(sid, load_scene(sid), images_dir / sid, dpi)
    print(sid, meta["panel_water_iou"], flush=True)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--procs", type=int, default=8)
    args = parser.parse_args()
    sids = sorted((p.name.replace("_prob.tif", "") for p in MODELS["s1aef_t"].glob("*/*_prob.tif")),
                  key=lambda s: int(s[3:]))
    assert len(sids) == 53
    args.images_dir.mkdir(parents=True, exist_ok=True)
    with Pool(args.procs) as pool:
        metas = pool.map(one, [(sid, args.images_dir, args.dpi) for sid in sids])
    loc = locate(sids)
    rows = []
    for meta in metas:
        sid = meta["sample_id"]
        rows.append({"sample_id": sid, "date_folder": meta["date_folder"], **loc[sid],
                     "label_water_fraction": meta["label_water_fraction"],
                     "common_valid_fraction": meta["common_valid_fraction"],
                     "common_valid_pixels": meta["common_valid_pixels"],
                     **{f"iou_{k}": v for k, v in meta["panel_water_iou"].items()}})
    pd.DataFrame(rows).to_csv(args.images_dir / "panel_iou_all_scenes.csv", index=False)
    full = [{"sample_id": m["sample_id"], **{f"iou_{k}": v for k, v in m["panel_water_iou_full"].items()},
             "label_water_fraction_full": float(m["label_water_fraction"])} for m in metas]
    pd.DataFrame(rows).merge(pd.DataFrame(full)[["sample_id"] + [f"iou_{k}" for k in m_keys(metas)]],
                             on="sample_id", suffixes=("", "_full")).to_csv(
        OUT / "fig1B/scene_table.csv", index=False)
    print("common valid pixels:", sum(m["common_valid_pixels"] for m in metas), flush=True)
    return 0


def m_keys(metas):
    return list(metas[0]["panel_water_iou_full"])


if __name__ == "__main__":
    raise SystemExit(main())
