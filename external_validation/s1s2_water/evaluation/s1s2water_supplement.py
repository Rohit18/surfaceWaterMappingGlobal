#!/usr/bin/env python3
"""S1S2-Water supplementary evaluation, tables, figures and PNG layers.

Scores corrected OPERA DSWx-S1 and the label-class S1-only (k0) and S1+AEF (k16) models, seeds 42/43/44, on
Earth Engine COPERNICUS/S1_GRD 10 m inputs, on ONE common support per scene: 30 m reference-valid pixels
(frozen grid, evaluation/DESIGN.md) where OPERA and all six model runs are valid. Reference water = area
fraction >= 0.5. Model probabilities are warped bilinearly to 30 m and thresholded at 0.50 (the
validation-selected threshold of every run); 0.30, the manuscript's main-table threshold, is reported as
a sensitivity (methods with suffix `_t030`). OPERA = DSWx-S1 v1.2 with the official post-tag boundary fix
(commit e7f20a), B02 BWTR layer, nearest resampling, classes 0/1 valid.

Reuses the reference aggregation and OPERA mosaicking of code/evaluate_intercomparison.py, so seed-42 values
reproduce evaluation/intercomparison_labelclass/ exactly.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from scipy.stats import wilcoxon

ROOT = Path("/pscratch/sd/r/rohit9/surface_water_validation")
sys.path.insert(0, str(ROOT / "code"))
import evaluate_intercomparison as EI  # noqa: E402

SCENES = EI.SCENES
SEEDS = (42, 43, 44)
THRESHOLD = 0.50
SENSITIVITY_THRESHOLD = 0.30
MODELS = {f"{name}_seed{seed}": (f"lc_k{width}_seed{seed}_tta", name)
          for seed in SEEDS for width, name in ((0, "s1_only"), (16, "s1_aef"))}
LABELS = {"opera": "OPERA DSWx-S1", "s1_only": "S1-only", "s1_aef": "S1+AEF"}
COLOURS = {"opera": "#c08a5a", "s1_only": "#8d99a6", "s1_aef": "#14556e"}
WATER_COLOR = np.array([0.0, 0.70, 1.0], dtype=np.float32)
ERROR_COLOURS = {"tn": (0.88, 0.88, 0.88), "tp": (0.0, 0.45, 0.80), "fp": (0.84, 0.15, 0.16),
                 "fn": (1.0, 0.60, 0.0)}
EXAMPLES = (23, 25, 78, 35)


def ratio(a, b):
    return None if b == 0 else a / b


def metrics(tp, fp, fn, tn):
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "evaluated": tp + fp + fn + tn,
            "water_iou": ratio(tp, tp + fp + fn), "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn)}


def confusion(pred, ref, common):
    return (int(np.count_nonzero(common & pred & ref)), int(np.count_nonzero(common & pred & ~ref)),
            int(np.count_nonzero(common & ~pred & ref)), int(np.count_nonzero(common & ~pred & ~ref)))


def probability(scene, tag, shape, transform, crs):
    paths = list((ROOT / f"work/model/scene_{scene}/inference_inputs_gee10m" / tag / "probabilities").glob("*/*_prob.tif"))
    if len(paths) != 1:
        raise RuntimeError(f"scene {scene} {tag}: {paths}")
    prob = np.full(shape, -9999.0, dtype="float32")
    with rasterio.open(paths[0]) as source:
        reproject(rasterio.band(source, 1), prob, src_transform=source.transform, src_crs=source.crs,
                  src_nodata=source.nodata, dst_transform=transform, dst_crs=crs, dst_nodata=-9999.0,
                  resampling=Resampling.bilinear)
    valid = np.isfinite(prob) & (prob >= 0) & (prob <= 1)
    return prob, valid, str(paths[0])


def s1_db_30m(scene, shape, transform, crs):
    """Earth Engine VV/VH (dB, 10 m) averaged in linear power onto the 30 m grid."""
    path = next((ROOT / f"work/model/scene_{scene}/inputs_gee10m").glob("s1_*.tif"))
    out = []
    with rasterio.open(path) as source:
        for band in (1, 2):
            linear = 10.0 ** (source.read(band).astype("float32") / 10.0)
            dest = np.full(shape, np.nan, dtype="float32")
            reproject(linear, dest, src_transform=source.transform, src_crs=source.crs, src_nodata=np.nan,
                      dst_transform=transform, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.average)
            with np.errstate(divide="ignore", invalid="ignore"):
                out.append(10.0 * np.log10(dest))
    return out, str(path)


def stretch(array):
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return np.zeros(array.shape, np.float32)
    lo, hi = np.percentile(valid, (2, 98))
    return np.nan_to_num(np.clip((array - lo) / max(float(hi - lo), 1e-6), 0, 1), nan=0.0).astype(np.float32)


def overlay(background, water):
    rgb = np.repeat(background[..., None], 3, axis=2).astype(np.float32)
    rgb[water] = 0.35 * rgb[water] + 0.65 * WATER_COLOR
    return rgb


def error_rgb(pred, ref, common):
    rgb = np.ones(pred.shape + (3,), np.float32)
    for name, mask in (("tn", ~pred & ~ref), ("tp", pred & ref), ("fp", pred & ~ref), ("fn", ~pred & ref)):
        rgb[mask & common] = ERROR_COLOURS[name]
    return rgb


def save_gray(path, array):
    Image.fromarray(array.astype(np.uint8), mode="L").save(path, optimize=True)


def save_rgb(path, rgb):
    Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8), mode="RGB").save(path, optimize=True)


def encode_water(water, valid):
    out = np.full(water.shape, 128, np.uint8)
    out[valid] = np.where(water[valid], 255, 0)
    return out


def encode_prob(prob, valid):
    out = np.full(prob.shape, 255, np.uint8)
    out[valid] = np.round(250 * np.clip(prob[valid], 0, 1)).astype(np.uint8)
    return out


def encode_db(db, lo, hi):
    out = np.full(db.shape, 255, np.uint8)
    ok = np.isfinite(db)
    out[ok] = np.round(250 * np.clip((db[ok] - lo) / (hi - lo), 0, 1)).astype(np.uint8)
    return out


def half(array, reducer):
    """2 x 2 block reduction for 60 m display layers (odd edge row/column dropped)."""
    h, w = (array.shape[0] // 2) * 2, (array.shape[1] // 2) * 2
    blocks = array[:h, :w].reshape(h // 2, 2, w // 2, 2)
    return reducer(blocks, axis=(1, 3))


def process_scene(scene, grid, images_dir, meta_dir):
    shape = (int(grid["height"]), int(grid["width"]))
    transform = from_origin(float(grid["left"]), float(grid["top"]), 30.0, 30.0)
    crs = grid["crs"]
    valid_fraction = np.zeros(shape, dtype="float32")
    water_valid_fraction = np.zeros(shape, dtype="float32")
    with rasterio.open(grid["reference_valid"]) as source:
        native_valid = source.read(1).astype("float32")
        reproject(native_valid, valid_fraction, src_transform=source.transform, src_crs=source.crs,
                  dst_transform=transform, dst_crs=crs, resampling=Resampling.average)
    with rasterio.open(grid["reference_mask"]) as source:
        reproject(source.read(1).astype("float32") * native_valid, water_valid_fraction,
                  src_transform=source.transform, src_crs=source.crs, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.average)
    del native_valid
    reference_valid = valid_fraction >= float(grid["reference_valid_fraction_min"])
    reference_fraction = np.divide(water_valid_fraction, valid_fraction, out=np.zeros_like(valid_fraction),
                                   where=valid_fraction > 0)
    reference = reference_fraction >= 0.5
    opera, opera_valid, opera_sources = EI.opera_prediction(scene, shape, transform, crs)
    probs, sources = {}, {"opera": opera_sources}
    common = reference_valid & opera_valid
    for name, (tag, _) in MODELS.items():
        prob, valid, path = probability(scene, tag, shape, transform, crs)
        probs[name] = (prob, valid)
        sources[name] = path
        common &= valid
    preds = {"opera": opera, **{name: prob >= THRESHOLD for name, (prob, _) in probs.items()}}
    rows = []
    for name, pred in preds.items():
        rows.append({"scene_id": scene, "method": name, "model_threshold": "" if name == "opera" else THRESHOLD,
                     **metrics(*confusion(pred, reference, common))})
    for name, (prob, _) in probs.items():
        rows.append({"scene_id": scene, "method": f"{name}_t030", "model_threshold": SENSITIVITY_THRESHOLD,
                     **metrics(*confusion(prob >= SENSITIVITY_THRESHOLD, reference, common))})

    # --- metadata
    meta = json.loads((ROOT / f"data/s1s2_water/metadata/{scene}/sentinel12_{scene}_meta.json").read_text())["properties"]
    west, south, east, north = transform_bounds(crs, "EPSG:4326", float(grid["left"]), float(grid["bottom"]),
                                                float(grid["right"]), float(grid["top"]))
    info = {"scene_id": scene, "date_s1": meta["date_s1"], "landcover": meta.get("landcover", ""),
            "centroid_lon": round((west + east) / 2, 4), "centroid_lat": round((south + north) / 2, 4),
            "reference_valid_pixels_30m": int(reference_valid.sum()), "common_pixels_30m": int(common.sum()),
            "common_fraction_of_reference_valid": round(int(common.sum()) / max(int(reference_valid.sum()), 1), 4),
            "reference_water_fraction_on_common": round(float(reference[common].mean()), 4)}

    # --- PNG layers (30 m categorical, 60 m continuous)
    out = images_dir / f"scene_{scene}"
    out.mkdir(parents=True, exist_ok=True)
    (vv, vh), s1_path = s1_db_30m(scene, shape, transform, crs)
    display = 0.55 * stretch(vv) + 0.45 * stretch(vh)
    save_gray(out / "s1_vv_db_60m.png", encode_db(half(vv, np.nanmean), -30.0, 5.0))
    save_gray(out / "s1_vh_db_60m.png", encode_db(half(vh, np.nanmean), -40.0, -5.0))
    save_gray(out / "s1_display_60m.png", np.round(255 * half(display, np.mean)))
    ref_png = np.full(shape, 128, np.uint8)
    ref_png[reference_valid] = np.where(reference[reference_valid], 255, 0)
    save_gray(out / "reference_water_30m.png", ref_png)
    save_gray(out / "reference_water_fraction_60m.png",
              np.where(half(reference_valid, np.all), np.round(250 * half(reference_fraction, np.mean)), 255))
    save_gray(out / "common_support_30m.png", np.where(common, 255, 0))
    save_gray(out / "opera_water_30m.png", encode_water(opera, opera_valid))
    save_rgb(out / "opera_error_30m.png", error_rgb(opera, reference, common))
    for name in ("s1_only_seed42", "s1_aef_seed42"):
        prob, valid = probs[name]
        save_gray(out / f"{name}_water_t050_30m.png", encode_water(prob >= THRESHOLD, valid))
        save_gray(out / f"{name}_prob_60m.png", encode_prob(half(np.where(valid, prob, 0), np.mean), half(valid, np.all)))
        save_rgb(out / f"{name}_error_t050_30m.png", error_rgb(prob >= THRESHOLD, reference, common))
    ious = {r["method"]: r["water_iou"] for r in rows}
    panels = [
        (np.repeat(display[..., None], 3, axis=2), "Sentinel-1 (VV/VH)"),
        (overlay(display, reference & reference_valid), "S1S2-Water label"),
        (overlay(display, preds["s1_only_seed42"] & probs["s1_only_seed42"][1]), "S1-only\nIoU={:.2f}".format(ious["s1_only_seed42"])),
        (overlay(display, preds["s1_aef_seed42"] & probs["s1_aef_seed42"][1]), "S1+AEF (ours)\nIoU={:.2f}".format(ious["s1_aef_seed42"])),
        (overlay(display, opera & opera_valid), "OPERA\nIoU={:.2f}".format(ious["opera"])),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.6))
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image, interpolation="nearest")
        axis.set_title(title, fontsize=10, pad=3)
        axis.set_xticks([]); axis.set_yticks([])
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.02, top=0.80, wspace=0.08)
    fig.savefig(out / f"scene_{scene}_five_panel.png", dpi=130, facecolor="white")
    plt.close(fig)
    (out / "layers.json").write_text(json.dumps({**info, "crs": crs, "transform_30m": list(transform)[:6],
                                                 "width_30m": shape[1], "height_30m": shape[0],
                                                 "water_iou_on_common": {k: round(v, 4) for k, v in ious.items()},
                                                 "s1_input": s1_path, "sources": sources}, indent=2) + "\n")
    return rows, info, panels


def short_landcover(label):
    """'Tree cover, broadleaved, deciduous, closed to open (>15%)' -> 'Tree cover (broadleaved)'."""
    parts = [p.strip() for p in label.split(",")]
    if parts[0] == "Tree cover" and len(parts) > 1:
        return f"Tree cover ({parts[1]})"
    return parts[0]


def pooled(rows, method):
    sel = [r for r in rows if r["method"] == method]
    total = metrics(*(sum(r[k] for r in sel) for k in ("tp", "fp", "fn", "tn")))
    ious = np.array([r["water_iou"] for r in sel], dtype=float)
    total.update(scene_mean_iou=float(ious.mean()), scene_median_iou=float(np.median(ious)))
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    args = parser.parse_args()
    for d in (args.results_dir, args.images_dir, args.figures_dir):
        d.mkdir(parents=True, exist_ok=True)
    with (ROOT / "evaluation/grids_30m.csv").open(newline="") as source:
        grids = {int(row["scene_id"]): row for row in csv.DictReader(source)}
    rows, infos, example_panels = [], [], {}
    for scene in SCENES:
        scene_rows, info, panels = process_scene(scene, grids[scene], args.images_dir, args.results_dir)
        rows += scene_rows
        infos.append(info)
        if scene in EXAMPLES:
            example_panels[scene] = panels
        print(scene, {r["method"]: round(r["water_iou"], 3) for r in scene_rows}, flush=True)

    methods = ["opera"] + list(MODELS) + [f"{m}_t030" for m in MODELS]
    with (args.results_dir / "s1s2water_per_scene_all_methods.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    pooled_rows = [{"method": m, **pooled(rows, m)} for m in methods]
    with (args.results_dir / "s1s2water_pooled_all_methods.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pooled_rows[0]))
        writer.writeheader(); writer.writerows(pooled_rows)

    # Seed summary and paired tests against OPERA
    lookup = {(r["scene_id"], r["method"]): r["water_iou"] for r in rows}
    summary = {"threshold": THRESHOLD, "sensitivity_threshold_suffix_t030": SENSITIVITY_THRESHOLD,
               "reference_water_fraction": 0.5, "grid": "30 m frozen grid",
               "support": "reference-valid AND OPERA-valid AND all six model runs valid", "scenes": list(SCENES),
               "opera": {k: pooled_rows[0][k] for k in ("water_iou", "precision", "recall", "scene_mean_iou", "scene_median_iou")},
               "models": {}, "paired_vs_opera": {}}
    for name in ("s1_only", "s1_aef", "s1_only_t030", "s1_aef_t030"):
        base, suffix = name.replace("_t030", ""), ("_t030" if name.endswith("_t030") else "")
        per_seed = {s: next(p for p in pooled_rows if p["method"] == f"{base}_seed{s}{suffix}") for s in SEEDS}
        summary["models"][name] = {
            key: {"seeds": [per_seed[s][key] for s in SEEDS], "mean": float(np.mean([per_seed[s][key] for s in SEEDS])),
                  "sd": float(np.std([per_seed[s][key] for s in SEEDS], ddof=1))}
            for key in ("water_iou", "precision", "recall", "scene_mean_iou", "scene_median_iou")}
        for s in SEEDS:
            method = f"{base}_seed{s}{suffix}"
            diff = np.array([lookup[(sc, method)] - lookup[(sc, "opera")] for sc in SCENES])
            summary["paired_vs_opera"][method] = {
                "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()), "median_diff": float(np.median(diff)),
                "mean_diff": float(diff.mean()),
                "wilcoxon_p": float(wilcoxon([lookup[(sc, method)] for sc in SCENES],
                                             [lookup[(sc, 'opera')] for sc in SCENES]).pvalue)}
    (args.results_dir / "s1s2water_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Supplementary table: one row per scene
    table = []
    for info in infos:
        sc = info["scene_id"]
        row = dict(info)
        row["opera_iou"] = round(lookup[(sc, "opera")], 4)
        for name in ("s1_only", "s1_aef"):
            vals = [lookup[(sc, f"{name}_seed{s}")] for s in SEEDS]
            row[f"{name}_iou_seed42"] = round(vals[0], 4)
            row[f"{name}_iou_seed_mean"] = round(float(np.mean(vals)), 4)
            row[f"{name}_iou_seed_sd"] = round(float(np.std(vals, ddof=1)), 4)
            vals030 = [lookup[(sc, f"{name}_seed{s}_t030")] for s in SEEDS]
            row[f"{name}_iou_t030_seed_mean"] = round(float(np.mean(vals030)), 4)
        table.append(row)
    with (args.results_dir / "s1s2water_supplementary_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)

    # Figure: per-scene IoU, OPERA and three seeds per model
    x = np.arange(len(SCENES))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(x, [lookup[(sc, "opera")] for sc in SCENES], "s", color=COLOURS["opera"], ms=5, label=LABELS["opera"])
    for offset, name, marker in ((-0.15, "s1_only", "o"), (0.15, "s1_aef", "D")):
        vals = np.array([[lookup[(sc, f"{name}_seed{s}")] for s in SEEDS] for sc in SCENES])
        ax.vlines(x + offset, vals.min(axis=1), vals.max(axis=1), color=COLOURS[name], lw=1.2)
        ax.plot(x + offset, vals.mean(axis=1), marker, color=COLOURS[name], ms=4.5,
                label=f"{LABELS[name]} (mean, range of 3 seeds)")
    ax.set_xticks(x, [str(sc) for sc in SCENES], fontsize=8)
    ax.set_xlabel("S1S2-Water test scene"); ax.set_ylabel("Water IoU (30 m)")
    ax.set_ylim(0, 1.02); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.figures_dir / f"suppl_s1s2water_per_scene_iou.{ext}", dpi=300)
    plt.close(fig)

    # Figure: example scenes, five panels each (seed 42)
    fig, axes = plt.subplots(len(EXAMPLES), 5, figsize=(11.5, 2.45 * len(EXAMPLES)))
    for r, sc in enumerate(EXAMPLES):
        for c, (image, title) in enumerate(example_panels[sc]):
            axes[r, c].imshow(image, interpolation="nearest")
            axes[r, c].set_title(title, fontsize=10, pad=3)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        lc = next(i["landcover"] for i in infos if i["scene_id"] == sc)
        axes[r, 0].set_ylabel(f"Scene {sc}\n{short_landcover(lc)}", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.01, top=0.95, wspace=0.08, hspace=0.35)
    for ext in ("png", "pdf"):
        fig.savefig(args.figures_dir / f"suppl_s1s2water_examples.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print(json.dumps(summary["models"], indent=1)[:1500])
    print(json.dumps(summary["paired_vs_opera"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
