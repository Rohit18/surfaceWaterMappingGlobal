#!/usr/bin/env python3
"""Evaluate S1-only and S1+AEF probabilities over all acquisition-valid scenes."""

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
METHODS = {
    "s1_only": {"threshold": 0.50, "tags": ("s1_only_tta", "scene57_s1_only_tta")},
    "s1_aef": {"threshold": 0.75, "tags": ("s1_aef_tta", "scene57_s1_aef_tta")},
}
REFERENCE_THRESHOLDS = (0.25, 0.50, 0.75)
OUTPUT_DIR = ROOT / "evaluation/models"


def ratio(a, b):
    return None if b == 0 else a / b


def probability_path(scene, method):
    base = ROOT / f"work/model/scene_{scene}/inference"
    matches = []
    for tag in METHODS[method]["tags"]:
        matches.extend((base / tag / "probabilities").glob("*/*_prob.tif"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {method} probability for scene {scene}, found {matches}")
    return matches[0]


def confusion(prediction, reference, evaluated):
    return {
        "tp": int(np.count_nonzero(evaluated & prediction & reference)),
        "fp": int(np.count_nonzero(evaluated & prediction & ~reference)),
        "fn": int(np.count_nonzero(evaluated & ~prediction & reference)),
        "tn": int(np.count_nonzero(evaluated & ~prediction & ~reference)),
        "evaluated": int(evaluated.sum()),
    }


def add_metrics(row):
    tp, fp, fn = row["tp"], row["fp"], row["fn"]
    row.update(water_iou=ratio(tp, tp + fp + fn), precision=ratio(tp, tp + fp),
               recall=ratio(tp, tp + fn), f1=ratio(2 * tp, 2 * tp + fp + fn))
    return row


def main():
    with (ROOT / "evaluation/grids_30m.csv").open(newline="") as source:
        grids = {int(row["scene_id"]): row for row in csv.DictReader(source)}
    rows = []
    for scene in SCENES:
        grid = grids[scene]
        width, height = int(grid["width"]), int(grid["height"])
        transform = from_origin(float(grid["left"]), float(grid["top"]), 30.0, 30.0)
        valid_fraction = np.zeros((height, width), dtype="float32")
        water_valid_fraction = np.zeros((height, width), dtype="float32")
        with rasterio.open(grid["reference_valid"]) as valid_source:
            valid_native = valid_source.read(1).astype("float32")
            reproject(valid_native, valid_fraction, src_transform=valid_source.transform,
                      src_crs=valid_source.crs, dst_transform=transform, dst_crs=grid["crs"],
                      resampling=Resampling.average)
        with rasterio.open(grid["reference_mask"]) as mask_source:
            water_valid = mask_source.read(1).astype("float32") * valid_native
            reproject(water_valid, water_valid_fraction, src_transform=mask_source.transform,
                      src_crs=mask_source.crs, dst_transform=transform, dst_crs=grid["crs"],
                      resampling=Resampling.average)
        reference_valid = valid_fraction >= float(grid["reference_valid_fraction_min"])
        reference_fraction = np.divide(water_valid_fraction, valid_fraction,
                                       out=np.zeros_like(valid_fraction), where=valid_fraction > 0)
        probabilities, method_valid, predictions = {}, {}, {}
        for method, specification in METHODS.items():
            path = probability_path(scene, method)
            probability = np.full((height, width), -9999.0, dtype="float32")
            with rasterio.open(path) as source:
                reproject(rasterio.band(source, 1), probability, src_transform=source.transform,
                          src_crs=source.crs, src_nodata=source.nodata, dst_transform=transform,
                          dst_crs=grid["crs"], dst_nodata=-9999.0, resampling=Resampling.bilinear)
            probabilities[method] = probability
            method_valid[method] = np.isfinite(probability) & (probability >= 0) & (probability <= 1)
            predictions[method] = probability >= specification["threshold"]
        common = reference_valid.copy()
        for valid in method_valid.values():
            common &= valid
        reference_valid_count = int(reference_valid.sum())
        for reference_threshold in REFERENCE_THRESHOLDS:
            reference = reference_fraction >= reference_threshold
            for method in METHODS:
                counts = confusion(predictions[method], reference, common)
                row = {"scene_id": scene, "method": method,
                       "model_threshold": METHODS[method]["threshold"],
                       "reference_water_fraction": reference_threshold,
                       **counts,
                       "valid_coverage_of_reference": ratio(counts["evaluated"], reference_valid_count),
                       "probability_path": str(probability_path(scene, method))}
                rows.append(add_metrics(row))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (OUTPUT_DIR / "per_scene_model_metrics.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    pooled = []
    for reference_threshold in REFERENCE_THRESHOLDS:
        for method in METHODS:
            selected = [row for row in rows if row["method"] == method
                        and row["reference_water_fraction"] == reference_threshold]
            counts = {name: sum(row[name] for row in selected) for name in ("tp", "fp", "fn", "tn", "evaluated")}
            metrics = add_metrics({"method": method, "model_threshold": METHODS[method]["threshold"],
                                   "reference_water_fraction": reference_threshold, **counts})
            ious = np.array([row["water_iou"] for row in selected], dtype="float64")
            metrics.update(scene_count=len(selected), median_scene_iou=float(np.median(ious)),
                           q1_scene_iou=float(np.percentile(ious, 25)),
                           q3_scene_iou=float(np.percentile(ious, 75)))
            pooled.append(metrics)
    with (OUTPUT_DIR / "pooled_model_metrics.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(pooled[0]))
        writer.writeheader(); writer.writerows(pooled)

    primary = [row for row in rows if row["reference_water_fraction"] == 0.5]
    lookup = {(row["scene_id"], row["method"]): row for row in primary}
    differences = [{"scene_id": scene,
                    "s1_only_iou": lookup[(scene, "s1_only")]["water_iou"],
                    "s1_aef_iou": lookup[(scene, "s1_aef")]["water_iou"],
                    "s1_aef_minus_s1_only": lookup[(scene, "s1_aef")]["water_iou"] - lookup[(scene, "s1_only")]["water_iou"]}
                   for scene in SCENES]
    with (OUTPUT_DIR / "paired_model_iou.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(differences[0]))
        writer.writeheader(); writer.writerows(differences)
    x = np.arange(len(SCENES))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, [row["s1_only_iou"] for row in differences], "o-", label="S1-only")
    ax.plot(x, [row["s1_aef_iou"] for row in differences], "o-", label="S1+AEF")
    ax.set_xticks(x, [str(scene) for scene in SCENES]); ax.set_ylim(0, 1)
    ax.set_xlabel("S1S2-Water test scene"); ax.set_ylabel("Water IoU (30 m)")
    ax.grid(axis="y", alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "paired_model_iou.png", dpi=200); plt.close(fig)
    summary = {"included_scenes": list(SCENES), "excluded_scene_31": "unresolved acquisition lineage and anomalous source encoding",
               "primary_reference_water_fraction": 0.5, "model_thresholds": {k: v["threshold"] for k, v in METHODS.items()},
               "pooled_primary": [row for row in pooled if row["reference_water_fraction"] == 0.5]}
    (OUTPUT_DIR / "model_evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
