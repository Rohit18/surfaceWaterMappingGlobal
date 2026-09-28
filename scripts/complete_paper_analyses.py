#!/usr/bin/env python
"""Regenerate the two analyses the manuscript needed but had no surviving script.

1. ``sweep``     - full probability-threshold sweep per configuration, computed
                   from the saved probability rasters. Supplies the S1-only
                   reference-set-optimal threshold quoted in the abstract and
                   Section III-A, which the paper_retrain evaluation did not
                   cover (it swept only 0.30 and 0.50).
2. ``proximity`` - train/evaluation centroid proximity screen and the
                   leave-out-the-close-scenes recomputation for Section II-D.

Both reproduce the published 0.30/0.50 confusion counts exactly, so they sit on
the same footing as the evaluation pipeline.

Usage:
    python scripts/complete_paper_analyses.py sweep     --out results/.../sweep.json
    python scripts/complete_paper_analyses.py proximity --out results/.../proximity.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RUN_SETS = {
    # Label-class reruns (paper_labelclass_v1): the models the manuscript now reports.
    "labelclass_v1": {
        "openwater_pred": "/pscratch/sd/r/rohit9/S1ML/intercomparison_s1aef/predictions/paper_labelclass_v1/openwater/58321212",
        "tminus1_pred": "/pscratch/sd/r/rohit9/S1ML/intercomparison_s1aef/predictions/paper_labelclass_v1/tminus1/58321217",
        "openwater_eval": "/pscratch/sd/r/rohit9/S1ML/training_runs/paper_labelclass_v1/openwater/reports/58321212/evaluation",
        "tminus1_eval": "/pscratch/sd/r/rohit9/S1ML/training_runs/paper_labelclass_v1/tminus1/reports/58321217/evaluation",
    },
    # Original runs trained on mixed binary50 + label-class labels (results of 2026-09-09).
    "mixed_v1": {
        "openwater_pred": "/pscratch/sd/r/rohit9/S1ML/intercomparison_s1aef/predictions/paper_retrain_openwater_v1/57725872",
        "tminus1_pred": "/pscratch/sd/r/rohit9/S1ML/intercomparison_s1aef/predictions/paper_tminus1_v1/57952761",
        "openwater_eval": "/global/u2/r/rohit9/Workspace/reports/paper_retrain_openwater_v1/57725872/evaluation",
        "tminus1_eval": "/pscratch/sd/r/rohit9/S1ML/training_runs/paper_tminus1_v1/reports/57952761/evaluation",
    },
}
DEFAULT_RUN_SET = "labelclass_v1"
OPENWATER_PRED = Path(RUN_SETS[DEFAULT_RUN_SET]["openwater_pred"])
TMINUS1_PRED = Path(RUN_SETS[DEFAULT_RUN_SET]["tminus1_pred"])
OPENWATER_EVAL = Path(RUN_SETS[DEFAULT_RUN_SET]["openwater_eval"])
TMINUS1_EVAL = Path(RUN_SETS[DEFAULT_RUN_SET]["tminus1_eval"])
OPERA_CSV = Path(
    "/pscratch/sd/r/rohit9/S1ML/intercomparison_opera/grouped/"
    "sid_permanent_water_per_sample_metrics.csv"
)
LABELS = "/global/u2/r/rohit9/Workspace/data/S1_Intercomparison_All_Scenes_subset/labels/{}.tif"
EVAL_META = Path("/global/u2/r/rohit9/surfaceWaterMappingGlobal/metadata/evaluation_samples.csv")
FOOTPRINTS = Path(
    "/global/u2/r/rohit9/surfaceWaterMappingGlobal/outputs/audit/unified_sample_footprints.geojson"
)

SWEEP_RUNS = {
    "S1-only (k0, seed42)": OPENWATER_PRED / "width_k0_seed42_current_tta",
    "S1+AEF(t) (k16, seed42)": OPENWATER_PRED / "width_k16_seed42_current_tta",
    "S1+AEF(t-1) swap (k16, seed42)": OPENWATER_PRED / "width_k16_seed42_tminus1_tta",
    "S1+AEF(t-1) trained (k16, seed42)": TMINUS1_PRED / "width_k16_seed42_tminus1_tta",
}
EVAL_RUNS = [
    ("S1-only", OPENWATER_EVAL, "width_k0_seed42_current_tta_paper030"),
    ("S1+AEF(t)", OPENWATER_EVAL, "width_k16_seed42_current_tta_paper030"),
    ("S1+AEF(t-1) swap", OPENWATER_EVAL, "width_k16_seed42_tminus1_tta"),
    ("S1+AEF(t-1) trained", TMINUS1_EVAL, "width_k16_seed42_tminus1_tta"),
]

NBINS = 100_000                      # edge spacing 1e-5; 0.01/0.30/0.50 are exact edges
EDGES = np.linspace(0.0, 1.0, NBINS + 1)
THRESHOLD = 0.30
EARTH_RADIUS_KM = 6371.0088


# --------------------------------------------------------------------------- sweep
def accumulate(root: Path):
    """Histogram P(water) separately over water- and non-water-labelled pixels."""
    import rasterio

    hist_water = np.zeros(NBINS, np.int64)
    hist_other = np.zeros(NBINS, np.int64)
    per_scene = {}
    for prob_file in sorted(glob.glob(str(root / "probabilities" / "*" / "*_prob.tif"))):
        sid = os.path.basename(prob_file).replace("_prob.tif", "")
        with rasterio.open(prob_file) as src:
            prob, nodata = src.read(1), src.nodata
        with rasterio.open(LABELS.format(sid)) as src:
            label = src.read(1)
        valid = np.isfinite(prob)
        if nodata is not None:
            valid &= prob != nodata
        pv, lv = prob[valid], label[valid]
        is_water = lv == 1
        hw = np.histogram(pv[is_water], bins=EDGES)[0]
        ho = np.histogram(pv[~is_water], bins=EDGES)[0]
        hist_water += hw
        hist_other += ho
        per_scene[sid] = (hw, ho)
    return hist_water, hist_other, per_scene


def confusion(hist_water, hist_other, threshold):
    idx = int(round(threshold * NBINS))
    tp, fn = hist_water[idx:].sum(), hist_water[:idx].sum()
    fp, tn = hist_other[idx:].sum(), hist_other[:idx].sum()
    return int(tp), int(fp), int(tn), int(fn)


def metrics(tp, fp, tn, fn):
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "water_iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
    }


GRID = np.concatenate([
    np.arange(0.00001, 0.01, 0.00001),
    np.arange(0.01, 0.10, 0.0005),
    np.arange(0.10, 0.95, 0.005),
])


def run_sweep(out_path: Path) -> int:
    results = {}
    for name, root in SWEEP_RUNS.items():
        hw, ho, per_scene = accumulate(root)
        entry = {"scenes": len(per_scene), "valid_pixels": int(hw.sum() + ho.sum())}

        for check in (0.30, 0.50):
            entry[f"at_{check:.2f}"] = metrics(*confusion(hw, ho, check))
        entry["at_0.01"] = metrics(*confusion(hw, ho, 0.01))

        best_thr, best = None, None
        for thr in GRID:
            m = metrics(*confusion(hw, ho, thr))
            if best is None or m["water_iou"] > best["water_iou"]:
                best_thr, best = float(thr), m
        entry["optimal"] = {"threshold": best_thr, **best}

        idx = int(round(best_thr * NBINS))
        scene_iou = []
        for hw_s, ho_s in per_scene.values():
            tp, fn, fp = hw_s[idx:].sum(), hw_s[:idx].sum(), ho_s[idx:].sum()
            scene_iou.append(tp / (tp + fp + fn) if tp + fp + fn else 0.0)
        entry["per_scene_mean_at_optimal"] = float(np.mean(scene_iou))

        results[name] = entry
        print(f"{name}\n  optimal thr={best_thr:.5f} IoU={best['water_iou']:.6f}  "
              f"at 0.30 IoU={entry['at_0.30']['water_iou']:.6f}  "
              f"at 0.01 IoU={entry['at_0.01']['water_iou']:.6f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


# ----------------------------------------------------------------------- proximity
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1)[:, None], np.radians(lat2)[None, :]
    dphi = p2 - p1
    dlam = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def run_proximity(out_path: Path) -> int:
    evaluation = pd.read_csv(EVAL_META)
    features = json.loads(FOOTPRINTS.read_text())["features"]
    training = pd.DataFrame([f["properties"] for f in features])
    is_supp = training["sample_set"].astype(str).str.lower().eq("supplement").to_numpy()

    dist = haversine_km(evaluation.centroid_lat.to_numpy(), evaluation.centroid_lon.to_numpy(),
                        training.centroid_lat.to_numpy(), training.centroid_lon.to_numpy())
    nearest = dist.min(axis=1)
    nearest_supp = dist[:, is_supp].min(axis=1)

    counts = {}
    for km in (5, 11, 20):
        counts[f"within_{km}km"] = {
            "total": int((nearest <= km).sum()),
            "legacy": int((dist[:, ~is_supp].min(axis=1) <= km).sum()),
            "supplement": int((nearest_supp <= km).sum()),
        }
        print(f"within {km:>2} km: {counts[f'within_{km}km']}")

    near11 = sorted(evaluation.loc[nearest <= 11, "sample_id"])
    print(f"\nscenes within 11 km ({len(near11)}): {near11}")
    print(f"closest supplement sample to any evaluation scene: {nearest_supp.min():.1f} km")

    excluded = {}
    print(f"\n{'configuration':<24}{'pooled/53':>11}{'pooled/45':>11}{'per-scene/53':>14}{'per-scene/45':>14}")
    rows = list(EVAL_RUNS) + [("OPERA DSWx-S1", None, None)]
    for name, base, tag in rows:
        if tag is None:
            frame = pd.read_csv(OPERA_CSV)
        else:
            frame = pd.read_csv(base / tag / f"{tag}_per_sample_metrics.csv")
            frame = frame[np.isclose(frame["threshold"], THRESHOLD)]
        keep = frame[~frame.sample_id.isin(near11)]
        pooled = lambda d: d.tp.sum() / (d.tp.sum() + d.fp.sum() + d.fn.sum())
        excluded[name] = {
            "pooled_53": float(pooled(frame)), "pooled_45": float(pooled(keep)),
            "per_scene_53": float(frame.water_iou.mean()),
            "per_scene_45": float(keep.water_iou.mean()),
        }
        e = excluded[name]
        print(f"{name:<24}{e['pooled_53']:>11.4f}{e['pooled_45']:>11.4f}"
              f"{e['per_scene_53']:>14.4f}{e['per_scene_45']:>14.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "counts": counts,
        "scenes_within_11km": near11,
        "closest_supplement_km": float(nearest_supp.min()),
        "exclusion_recomputation": excluded,
    }, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


# --------------------------------------------------------------------- matched mask
OPERA_MASK_DIR = Path("/pscratch/sd/r/rohit9/S1ML/intercomparison_opera/binary_masks")
MATCHED_CONFIGS = [
    ("S1-only", OPENWATER_PRED / "width_k0_seed42_current_tta"),
    ("S1+AEF(t)", OPENWATER_PRED / "width_k16_seed42_current_tta"),
    ("S1+AEF(t-1) swap", OPENWATER_PRED / "width_k16_seed42_tminus1_tta"),
    ("S1+AEF(t-1) trained", TMINUS1_PRED / "width_k16_seed42_tminus1_tta"),
]


def run_matched_mask(out_path: Path) -> int:
    """Score every method on the intersection of all products' valid masks.

    Table I as published lets each row use its own no-data mask, so the rows do
    not cover identical pixel sets (S1-only 55.1M, S1+AEF 54.4M, OPERA 52.7M).
    This recomputes all five rows on the common mask, which is the only fully
    like-for-like comparison available.
    """
    import rasterio
    from scipy.stats import wilcoxon

    opera_rows = pd.read_csv(OPERA_CSV).set_index("sample_id")
    prob_index = {}
    for name, root in MATCHED_CONFIGS:
        prob_index[name] = {
            os.path.basename(f).replace("_prob.tif", ""): f
            for f in glob.glob(str(root / "probabilities" / "*" / "*_prob.tif"))
        }
    scenes = sorted(prob_index[MATCHED_CONFIGS[0][0]])

    totals = {name: np.zeros(3, np.int64) for name, _ in MATCHED_CONFIGS}
    totals["OPERA DSWx-S1"] = np.zeros(3, np.int64)
    scene_iou = {name: [] for name in totals}
    native_pixels, matched_pixels = 0, 0

    for sid in scenes:
        with rasterio.open(LABELS.format(sid)) as src:
            label = src.read(1)
        with rasterio.open(opera_rows.loc[sid, "binary_mask_path"]) as src:
            opera, opera_nodata = src.read(1), src.nodata
        valid = np.ones(label.shape, bool)
        if opera_nodata is not None:
            valid &= opera != opera_nodata
        probs = {}
        for name, _ in MATCHED_CONFIGS:
            with rasterio.open(prob_index[name][sid]) as src:
                arr, nodata = src.read(1), src.nodata
            good = np.isfinite(arr)
            if nodata is not None:
                good &= arr != nodata
            valid &= good
            probs[name] = arr
        native_pixels += label.size
        matched_pixels += int(valid.sum())

        truth = label[valid] == 1
        preds = {name: probs[name][valid] >= THRESHOLD for name, _ in MATCHED_CONFIGS}
        preds["OPERA DSWx-S1"] = opera[valid] == 1
        for name, pred in preds.items():
            tp = int(np.count_nonzero(pred & truth))
            fp = int(np.count_nonzero(pred & ~truth))
            fn = int(np.count_nonzero(~pred & truth))
            totals[name] += (tp, fp, fn)
            scene_iou[name].append(tp / (tp + fp + fn) if tp + fp + fn else 0.0)

    print(f"common valid pixels: {matched_pixels:,} of {native_pixels:,} "
          f"({100 * matched_pixels / native_pixels:.2f}%)\n")
    print(f"{'method':<22}{'pooled IoU':>12}{'per-scene':>12}")
    results = {}
    for name in totals:
        tp, fp, fn = totals[name]
        pooled = tp / (tp + fp + fn)
        per_scene = float(np.mean(scene_iou[name]))
        results[name] = {"pooled_water_iou": float(pooled), "per_scene_mean": per_scene,
                         "tp": int(tp), "fp": int(fp), "fn": int(fn)}
        print(f"{name:<22}{pooled:>12.4f}{per_scene:>12.4f}")

    print(f"\n{'paired comparison':<34}{'wins':>9}{'median':>10}{'p':>12}")
    pairs = {}
    ref = np.asarray(scene_iou["OPERA DSWx-S1"])
    for name in ("S1+AEF(t)", "S1+AEF(t-1) swap", "S1+AEF(t-1) trained", "S1-only"):
        arr = np.asarray(scene_iou[name])
        diff = arr - ref
        p = float(wilcoxon(arr, ref).pvalue)
        pairs[f"{name} vs OPERA"] = {"wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()),
                                     "ties": int((diff == 0).sum()), "median_diff": float(np.median(diff)),
                                     "mean_diff": float(diff.mean()), "wilcoxon_p": p}
        print(f"{name + ' vs OPERA':<34}{(diff > 0).sum():>6}/{len(diff)}"
              f"{np.median(diff):>+10.4f}{p:>12.3g}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "threshold": THRESHOLD,
        "common_valid_pixels": matched_pixels,
        "total_pixels": native_pixels,
        "methods": results,
        "paired_vs_opera": pairs,
    }, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


# ------------------------------------------------------------------ paired vs OPERA
def run_paired(out_path: Path) -> int:
    """Per-scene paired comparison against OPERA at THRESHOLD, each product on its own valid mask
    (the Table I / Fig. 3 convention); matched-mask gives the same test on the common mask."""
    from scipy.stats import wilcoxon

    opera = pd.read_csv(OPERA_CSV).set_index("sample_id")["water_iou"]
    pairs = {}
    print(f"{'comparison':<34}{'wins':>8}{'losses':>8}{'ties':>6}{'median':>10}{'mean':>10}{'p':>11}")
    for name, base, tag in [EVAL_RUNS[1], EVAL_RUNS[2], EVAL_RUNS[3], EVAL_RUNS[0]]:
        frame = pd.read_csv(base / tag / f"{tag}_per_sample_metrics.csv")
        frame = frame[np.isclose(frame["threshold"], THRESHOLD)].set_index("sample_id")["water_iou"]
        joined = pd.concat([frame.rename("model"), opera.rename("opera")], axis=1, join="inner")
        diff = joined["model"] - joined["opera"]
        p = float(wilcoxon(joined["model"], joined["opera"]).pvalue)
        pairs[f"{name} vs OPERA"] = {"scenes": int(len(diff)), "wins": int((diff > 0).sum()),
                                     "losses": int((diff < 0).sum()), "ties": int((diff == 0).sum()),
                                     "median_diff": float(diff.median()), "mean_diff": float(diff.mean()),
                                     "wilcoxon_p": p}
        e = pairs[f"{name} vs OPERA"]
        print(f"{name + ' vs OPERA':<34}{e['wins']:>5}/{e['scenes']}{e['losses']:>8}{e['ties']:>6}"
              f"{e['median_diff']:>+10.4f}{e['mean_diff']:>+10.4f}{p:>11.3g}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"threshold": THRESHOLD, "mask": "each product's own valid mask",
                                    "paired_vs_opera": pairs}, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


def use_run_set(name: str) -> None:
    """Point every run table at one run set (module-level tables are built from the default)."""
    global OPENWATER_PRED, TMINUS1_PRED, OPENWATER_EVAL, TMINUS1_EVAL, SWEEP_RUNS, EVAL_RUNS, MATCHED_CONFIGS
    paths = RUN_SETS[name]
    OPENWATER_PRED, TMINUS1_PRED = Path(paths["openwater_pred"]), Path(paths["tminus1_pred"])
    OPENWATER_EVAL, TMINUS1_EVAL = Path(paths["openwater_eval"]), Path(paths["tminus1_eval"])
    SWEEP_RUNS = {
        "S1-only (k0, seed42)": OPENWATER_PRED / "width_k0_seed42_current_tta",
        "S1+AEF(t) (k16, seed42)": OPENWATER_PRED / "width_k16_seed42_current_tta",
        "S1+AEF(t-1) swap (k16, seed42)": OPENWATER_PRED / "width_k16_seed42_tminus1_tta",
        "S1+AEF(t-1) trained (k16, seed42)": TMINUS1_PRED / "width_k16_seed42_tminus1_tta",
    }
    EVAL_RUNS = [
        ("S1-only", OPENWATER_EVAL, "width_k0_seed42_current_tta_paper030"),
        ("S1+AEF(t)", OPENWATER_EVAL, "width_k16_seed42_current_tta_paper030"),
        ("S1+AEF(t-1) swap", OPENWATER_EVAL, "width_k16_seed42_tminus1_tta"),
        ("S1+AEF(t-1) trained", TMINUS1_EVAL, "width_k16_seed42_tminus1_tta"),
    ]
    MATCHED_CONFIGS = [
        ("S1-only", OPENWATER_PRED / "width_k0_seed42_current_tta"),
        ("S1+AEF(t)", OPENWATER_PRED / "width_k16_seed42_current_tta"),
        ("S1+AEF(t-1) swap", OPENWATER_PRED / "width_k16_seed42_tminus1_tta"),
        ("S1+AEF(t-1) trained", TMINUS1_PRED / "width_k16_seed42_tminus1_tta"),
    ]
    print(f"run set: {name}\n  {OPENWATER_PRED}\n  {TMINUS1_PRED}\n", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=("sweep", "proximity", "matched-mask", "paired"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-set", choices=sorted(RUN_SETS), default=DEFAULT_RUN_SET)
    args = parser.parse_args()
    use_run_set(args.run_set)
    if args.analysis == "sweep":
        return run_sweep(args.out)
    if args.analysis == "proximity":
        return run_proximity(args.out)
    if args.analysis == "paired":
        return run_paired(args.out)
    return run_matched_mask(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
