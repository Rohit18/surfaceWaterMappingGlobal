#!/usr/bin/env python3
"""Evaluate corrected OPERA, S1-only, and S1+AEF together.

--models released   : released checkpoints on the benchmark S1 raster (default; evaluation/intercomparison)
--models labelclass : label-class paper reruns (openwater 58321212, k0/k16 seed 42) on Earth Engine
                      COPERNICUS/S1_GRD inputs at 10 m (evaluation/intercomparison_labelclass)
"""

import argparse
import csv
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENES = (23, 25, 29, 33, 35, 47, 53, 57, 75, 77, 78, 80, 82, 88, 89)
REFERENCE_THRESHOLDS = (0.25, 0.50, 0.75)
METHODS = ("opera_corrected", "s1_only", "s1_aef")
MODEL_SETS = {
    "released": {
        "thresholds": {"s1_only": 0.50, "s1_aef": 0.75},
        "directory": "inference",
        "tags": {"s1_only": ("s1_only_tta", "scene57_s1_only_tta"),
                 "s1_aef": ("s1_aef_tta", "scene57_s1_aef_tta")},
        "labels": {"s1_only": "Released S1-only", "s1_aef": "Released S1+AEF"},
        "output": "evaluation/intercomparison",
    },
    "labelclass": {
        "thresholds": {"s1_only": 0.50, "s1_aef": 0.50},
        "directory": "inference_inputs_gee10m",
        "tags": {"s1_only": ("lc_k0_seed42_tta",), "s1_aef": ("lc_k16_seed42_tta",)},
        "labels": {"s1_only": "S1-only (label class, GEE S1)", "s1_aef": "S1+AEF (label class, GEE S1)"},
        "output": "evaluation/intercomparison_labelclass",
    },
}
MODEL_SET = MODEL_SETS["released"]
MODEL_THRESHOLDS = MODEL_SET["thresholds"]
OUTPUT = ROOT / MODEL_SET["output"]


def ratio(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def probability_path(scene, method):
    tags = MODEL_SET["tags"][method]
    base = ROOT / f"work/model/scene_{scene}" / MODEL_SET["directory"]
    matches = []
    for tag in tags:
        matches.extend((base / tag / "probabilities").glob("*/*_prob.tif"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {method} probability for scene {scene}: {matches}")
    return matches[0]


def opera_paths(scene):
    directory = ROOT / "work/opera" / (
        "scene_57_boundary_fix" if scene == 57 else f"scene_{scene}"
    ) / "dswx_output"
    paths = sorted(directory.glob("*_B02_BWTR.tif"))
    if not paths:
        raise RuntimeError(f"No corrected OPERA BWTR products for scene {scene} in {directory}")
    return paths


def add_metrics(row):
    tp, fp, fn = row["tp"], row["fp"], row["fn"]
    row.update(
        water_iou=ratio(tp, tp + fp + fn),
        precision=ratio(tp, tp + fp),
        recall=ratio(tp, tp + fn),
        f1=ratio(2 * tp, 2 * tp + fp + fn),
    )
    return row


def model_prediction(scene, method, shape, transform, crs):
    path = probability_path(scene, method)
    probability = np.full(shape, -9999.0, dtype="float32")
    with rasterio.open(path) as source:
        reproject(
            rasterio.band(source, 1), probability,
            src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
            dst_transform=transform, dst_crs=crs, dst_nodata=-9999.0,
            resampling=Resampling.bilinear,
        )
    valid = np.isfinite(probability) & (probability >= 0) & (probability <= 1)
    return probability >= MODEL_THRESHOLDS[method], valid, [str(path)]


def opera_prediction(scene, shape, transform, crs):
    mosaic = np.full(shape, 255, dtype="uint8")
    paths = opera_paths(scene)
    for path in paths:
        warped = np.full(shape, 255, dtype="uint8")
        with rasterio.open(path) as source:
            reproject(
                rasterio.band(source, 1), warped,
                src_transform=source.transform, src_crs=source.crs,
                src_nodata=source.nodata, dst_transform=transform, dst_crs=crs,
                dst_nodata=255, resampling=Resampling.nearest,
            )
        covered = np.isin(warped, (0, 1))
        mosaic[covered] = warped[covered]
    return mosaic == 1, np.isin(mosaic, (0, 1)), [str(path) for path in paths]


def main():
    global MODEL_SET, MODEL_THRESHOLDS, OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", choices=sorted(MODEL_SETS), default="released")
    args = parser.parse_args()
    MODEL_SET = MODEL_SETS[args.models]
    MODEL_THRESHOLDS = MODEL_SET["thresholds"]
    OUTPUT = ROOT / MODEL_SET["output"]
    with (ROOT / "evaluation/grids_30m.csv").open(newline="") as source:
        grids = {int(row["scene_id"]): row for row in csv.DictReader(source)}
    rows = []
    scene_support = []
    for scene in SCENES:
        grid = grids[scene]
        shape = (int(grid["height"]), int(grid["width"]))
        transform = from_origin(float(grid["left"]), float(grid["top"]), 30.0, 30.0)
        crs = grid["crs"]
        valid_fraction = np.zeros(shape, dtype="float32")
        water_valid_fraction = np.zeros(shape, dtype="float32")
        with rasterio.open(grid["reference_valid"]) as source:
            native_valid = source.read(1).astype("float32")
            reproject(native_valid, valid_fraction, src_transform=source.transform,
                      src_crs=source.crs, dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.average)
        with rasterio.open(grid["reference_mask"]) as source:
            reproject(source.read(1).astype("float32") * native_valid,
                      water_valid_fraction, src_transform=source.transform,
                      src_crs=source.crs, dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.average)
        reference_valid = valid_fraction >= float(grid["reference_valid_fraction_min"])
        reference_fraction = np.divide(
            water_valid_fraction, valid_fraction, out=np.zeros_like(valid_fraction),
            where=valid_fraction > 0,
        )
        predictions = {}
        valid = {}
        sources = {}
        predictions["opera_corrected"], valid["opera_corrected"], sources["opera_corrected"] = \
            opera_prediction(scene, shape, transform, crs)
        for method in ("s1_only", "s1_aef"):
            predictions[method], valid[method], sources[method] = \
                model_prediction(scene, method, shape, transform, crs)
        common = reference_valid.copy()
        for method in METHODS:
            common &= valid[method]
        scene_support.append({
            "scene_id": scene,
            "reference_valid": int(reference_valid.sum()),
            "common_evaluated": int(common.sum()),
            "common_coverage_of_reference": ratio(int(common.sum()), int(reference_valid.sum())),
            **{f"{method}_valid": int((reference_valid & valid[method]).sum()) for method in METHODS},
        })
        for threshold in REFERENCE_THRESHOLDS:
            reference = reference_fraction >= threshold
            for method in METHODS:
                prediction = predictions[method]
                tp = int(np.count_nonzero(common & prediction & reference))
                fp = int(np.count_nonzero(common & prediction & ~reference))
                fn = int(np.count_nonzero(common & ~prediction & reference))
                tn = int(np.count_nonzero(common & ~prediction & ~reference))
                rows.append(add_metrics({
                    "scene_id": scene, "method": method,
                    "model_threshold": MODEL_THRESHOLDS.get(method),
                    "reference_water_fraction": threshold,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "evaluated": int(common.sum()),
                    "common_coverage_of_reference": ratio(int(common.sum()), int(reference_valid.sum())),
                    "source_paths": json.dumps(sources[method]),
                }))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "per_scene_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (OUTPUT / "scene_support.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scene_support[0])); writer.writeheader(); writer.writerows(scene_support)
    pooled = []
    for threshold in REFERENCE_THRESHOLDS:
        for method in METHODS:
            selected = [row for row in rows if row["method"] == method and row["reference_water_fraction"] == threshold]
            counts = {key: sum(row[key] for row in selected) for key in ("tp", "fp", "fn", "tn", "evaluated")}
            ious = np.array([row["water_iou"] for row in selected])
            pooled.append(add_metrics({
                "method": method, "model_threshold": MODEL_THRESHOLDS.get(method),
                "reference_water_fraction": threshold, **counts,
                "scene_count": len(selected), "median_scene_iou": float(np.median(ious)),
                "q1_scene_iou": float(np.percentile(ious, 25)),
                "q3_scene_iou": float(np.percentile(ious, 75)),
            }))
    with (OUTPUT / "pooled_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pooled[0])); writer.writeheader(); writer.writerows(pooled)

    primary = {(row["scene_id"], row["method"]): row for row in rows
               if row["reference_water_fraction"] == 0.5}
    paired = []
    for scene in SCENES:
        values = {method: primary[(scene, method)]["water_iou"] for method in METHODS}
        paired.append({
            "scene_id": scene, **{f"{method}_iou": value for method, value in values.items()},
            "s1_only_minus_opera": values["s1_only"] - values["opera_corrected"],
            "s1_aef_minus_opera": values["s1_aef"] - values["opera_corrected"],
            "s1_aef_minus_s1_only": values["s1_aef"] - values["s1_only"],
        })
    with (OUTPUT / "paired_primary_iou.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    x = np.arange(len(SCENES))
    figure, axis = plt.subplots(figsize=(11, 4.8))
    labels = {"opera_corrected": "OPERA DSWx-S1 corrected", **MODEL_SET["labels"]}
    for method in METHODS:
        axis.plot(x, [row[f"{method}_iou"] for row in paired], "o-", label=labels[method])
    axis.set_xticks(x, [str(scene) for scene in SCENES]); axis.set_ylim(0, 1)
    axis.set_xlabel("S1S2-Water test scene"); axis.set_ylabel("Water IoU at 30 m")
    axis.grid(axis="y", alpha=0.3); axis.legend(); figure.tight_layout()
    figure.savefig(OUTPUT / "paired_primary_iou.png", dpi=200); plt.close(figure)
    summary = {
        "included_scenes": list(SCENES),
        "model_set": args.models,
        "model_thresholds": MODEL_THRESHOLDS,
        "excluded_scene_31": "unresolved acquisition lineage and anomalous source encoding",
        "primary_reference_water_fraction": 0.5,
        "support_rule": "intersection of reference-valid, corrected OPERA-valid, S1-only-valid, and S1+AEF-valid pixels",
        "opera_implementation": {
            "base_tag": "DSWX-SAR v1.2",
            "official_post_tag_fix_commit": "e7f20a",
            "label": "opera_corrected",
        },
        "pooled_primary": [row for row in pooled if row["reference_water_fraction"] == 0.5],
        "median_paired_iou_differences": {
            key: float(np.median([row[key] for row in paired]))
            for key in ("s1_only_minus_opera", "s1_aef_minus_opera", "s1_aef_minus_s1_only")
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
