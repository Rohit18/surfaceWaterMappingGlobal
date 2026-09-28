#!/usr/bin/env python3
"""Evaluate the label-class paper models (and the released models) on S1S2-Water.

Primary: frozen 30 m grid and reference aggregation from evaluation/DESIGN.md (reference water
fraction 0.5; sensitivity 0.25/0.75). Secondary: the benchmark's native S1 grid with the
hand-labelled mask and valid mask as supplied, which is the protocol of Wieland et al. (2024,
IEEE JSTARS 17:1084) for comparison with their published S1 results.

Every method is scored on one common support per scene: reference-valid pixels where every
method in METHODS has a valid probability. Probabilities are warped bilinearly before the
threshold. Model thresholds are the validation-selected thresholds recorded in each training
run's eval summary (0.50 for all six label-class runs; 0.75/0.50 for the released models);
0.30, the paper's main-table threshold, is reported as a sensitivity.
"""

import argparse
import csv
import json
import multiprocessing
import pathlib

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import Window


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENES = (23, 25, 29, 33, 35, 47, 53, 57, 75, 77, 78, 80, 82, 88, 89)
OUTPUT_DIR = ROOT / "evaluation/models_labelclass"
REFERENCE_THRESHOLDS = (0.25, 0.50, 0.75)
NATIVE_BLOCK_ROWS = 1024


def build_methods():
    methods = {
        "released_s1_only": {"dirs": ("inference",), "tags": ("s1_only_tta", "scene57_s1_only_tta"),
                             "thresholds": (0.50,), "input": "benchmark", "model": "released S1-only seed43"},
        "released_s1_aef": {"dirs": ("inference",), "tags": ("s1_aef_tta", "scene57_s1_aef_tta"),
                            "thresholds": (0.75,), "input": "benchmark", "model": "released S1+AEF (HF)"},
    }
    for seed in (42, 43, 44):
        for width, name in ((0, "s1_only"), (16, "s1_aef")):
            methods[f"lc_{name}_gee_seed{seed}"] = {
                "dirs": ("inference_inputs_gee10m",), "tags": (f"lc_k{width}_seed{seed}_tta",),
                "thresholds": (0.50, 0.30), "input": "gee10m", "model": f"label-class k{width} seed{seed}"}
    for width, name in ((0, "s1_only"), (16, "s1_aef")):
        methods[f"lc_{name}_bench_seed42"] = {
            "dirs": ("inference_inputs",), "tags": (f"lc_k{width}_seed42_tta",),
            "thresholds": (0.50, 0.30), "input": "benchmark", "model": f"label-class k{width} seed42"}
    return methods


METHODS = build_methods()


def ratio(a, b):
    return None if b == 0 else a / b


def probability_path(scene, method):
    matches = []
    for directory in METHODS[method]["dirs"]:
        for tag in METHODS[method]["tags"]:
            matches.extend((ROOT / f"work/model/scene_{scene}" / directory / tag / "probabilities").glob("*/*_prob.tif"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {method} probability for scene {scene}, found {matches}")
    return matches[0]


def warp_probability(path, shape, transform, crs):
    probability = np.full(shape, -9999.0, dtype="float32")
    with rasterio.open(path) as source:
        reproject(rasterio.band(source, 1), probability, src_transform=source.transform,
                  src_crs=source.crs, src_nodata=source.nodata, dst_transform=transform,
                  dst_crs=crs, dst_nodata=-9999.0, resampling=Resampling.bilinear)
    valid = np.isfinite(probability) & (probability >= 0) & (probability <= 1)
    return probability, valid


def confusion(prediction, reference, evaluated):
    return np.array([np.count_nonzero(evaluated & prediction & reference),
                     np.count_nonzero(evaluated & prediction & ~reference),
                     np.count_nonzero(evaluated & ~prediction & reference),
                     np.count_nonzero(evaluated & ~prediction & ~reference)], dtype="int64")


def metric_row(counts, **keys):
    tp, fp, fn, tn = (int(v) for v in counts)
    return {**keys, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "evaluated": tp + fp + fn + tn,
            "water_iou": ratio(tp, tp + fp + fn), "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn)}


def evaluate_30m(scene, grid):
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
    del valid_native, water_valid
    reference_valid = valid_fraction >= float(grid["reference_valid_fraction_min"])
    reference_fraction = np.divide(water_valid_fraction, valid_fraction,
                                   out=np.zeros_like(valid_fraction), where=valid_fraction > 0)
    probabilities, common = {}, reference_valid.copy()
    for method in METHODS:
        probabilities[method], valid = warp_probability(probability_path(scene, method), (height, width),
                                                        transform, grid["crs"])
        common &= valid
    rows = []
    for reference_threshold in REFERENCE_THRESHOLDS:
        reference = reference_fraction >= reference_threshold
        for method, spec in METHODS.items():
            for threshold in spec["thresholds"]:
                counts = confusion(probabilities[method] >= threshold, reference, common)
                rows.append(metric_row(counts, grid="30m", scene_id=scene, method=method,
                                       model_threshold=threshold, reference_water_fraction=reference_threshold,
                                       common_coverage_of_reference=ratio(int(common.sum()), int(reference_valid.sum()))))
    return rows


def evaluate_native(scene, grid):
    """Benchmark-native grid: mask as supplied, valid == 1, processed in row blocks."""
    counts = {(m, t): np.zeros(4, dtype="int64") for m, s in METHODS.items() for t in s["thresholds"]}
    reference_valid_total = common_total = 0
    paths = {method: probability_path(scene, method) for method in METHODS}
    with rasterio.open(grid["reference_mask"]) as mask_source, rasterio.open(grid["reference_valid"]) as valid_source:
        for row_off in range(0, mask_source.height, NATIVE_BLOCK_ROWS):
            window = Window(0, row_off, mask_source.width, min(NATIVE_BLOCK_ROWS, mask_source.height - row_off))
            reference = mask_source.read(1, window=window) == 1
            common = valid_source.read(1, window=window) == 1
            reference_valid_total += int(common.sum())
            block_transform = mask_source.window_transform(window)
            probabilities = {}
            for method, path in paths.items():
                probabilities[method], valid = warp_probability(path, reference.shape, block_transform, mask_source.crs)
                common &= valid
            common_total += int(common.sum())
            for (method, threshold), total in counts.items():
                total += confusion(probabilities[method] >= threshold, reference, common)
    return [metric_row(total, grid="native", scene_id=scene, method=method, model_threshold=threshold,
                       reference_water_fraction=None,
                       common_coverage_of_reference=ratio(common_total, reference_valid_total))
            for (method, threshold), total in counts.items()]


def summarize(rows):
    pooled = []
    keys = sorted({(r["grid"], r["method"], r["model_threshold"], r["reference_water_fraction"]) for r in rows},
                  key=lambda k: (k[0], k[1], k[2], -1 if k[3] is None else k[3]))
    for grid, method, threshold, reference_threshold in keys:
        selected = [r for r in rows if (r["grid"], r["method"], r["model_threshold"], r["reference_water_fraction"])
                    == (grid, method, threshold, reference_threshold)]
        totals = np.sum([[r["tp"], r["fp"], r["fn"], r["tn"]] for r in selected], axis=0)
        row = metric_row(totals, grid=grid, method=method, model_threshold=threshold,
                         reference_water_fraction=reference_threshold)
        ious = np.array([r["water_iou"] for r in selected if r["water_iou"] is not None], dtype="float64")
        weights = np.array([r["evaluated"] for r in selected if r["water_iou"] is not None], dtype="float64")
        row.update(input=METHODS[method]["input"], model=METHODS[method]["model"], scene_count=len(selected),
                   median_scene_iou=float(np.median(ious)), q1_scene_iou=float(np.percentile(ious, 25)),
                   q3_scene_iou=float(np.percentile(ious, 75)), mean_scene_iou=float(ious.mean()),
                   pixel_weighted_mean_scene_iou=float(np.average(ious, weights=weights)))
        pooled.append(row)
    return pooled


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_scene(scene):
    with (ROOT / "evaluation/grids_30m.csv").open(newline="") as source:
        grid = next(row for row in csv.DictReader(source) if int(row["scene_id"]) == scene)
    rows = evaluate_30m(scene, grid) + evaluate_native(scene, grid)
    print(f"scene {scene} done", flush=True)
    return rows


def main():
    global METHODS, OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=None, help="subset of method names (default: all)")
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--processes", type=int, default=1, help="scenes evaluated in parallel")
    args = parser.parse_args()
    if args.methods:
        unknown = set(args.methods) - set(METHODS)
        if unknown:
            raise SystemExit(f"unknown methods: {sorted(unknown)}")
        METHODS = {name: spec for name, spec in METHODS.items() if name in args.methods}
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    # fork start method: workers inherit the filtered METHODS
    with multiprocessing.get_context("fork").Pool(args.processes) as pool:
        rows = [row for scene_rows in pool.map(evaluate_scene, SCENES) for row in scene_rows]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "per_scene_metrics.csv", rows)
    pooled = summarize(rows)
    write_csv(OUTPUT_DIR / "pooled_metrics.csv", pooled)
    provenance = {method: {**spec, "probability_paths": {scene: str(probability_path(scene, method)) for scene in SCENES}}
                  for method, spec in METHODS.items()}
    summary = {"scenes": list(SCENES), "excluded_scene_31": "unresolved acquisition lineage and anomalous source encoding",
               "primary": "30m grid, reference water fraction 0.5, validation-selected model threshold",
               "native_grid_note": "benchmark mask/valid as supplied; comparable in protocol to Wieland et al. 2024",
               "methods": provenance,
               "pooled_primary": [r for r in pooled if r["grid"] == "30m" and r["reference_water_fraction"] == 0.5],
               "pooled_native": [r for r in pooled if r["grid"] == "native"]}
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["pooled_primary"], indent=1))


if __name__ == "__main__":
    main()
