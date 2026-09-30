#!/usr/bin/env python3
"""Common-mask scoring for the 53 PlanetScope (GSWD) scenes, as used in the GRSL revision.

Conventions (identical to scripts/complete_paper_analyses.py matched-mask, plus the label check):
- label pixels scored: value 0 or 1 (1 = water); the label's declared nodata 0 is ignored;
- a probability pixel is valid where it is finite and != its nodata value;
- predicted water: probability >= threshold; OPERA water: binary mask == 1, valid where != nodata;
- pooled metrics sum TP/FP/FN over scenes;
- per-scene mean with bootstrap 95% CI: rng = np.random.default_rng(42), 10,000 resamples of scene
  indices (rng.integers(0, n, size=(10000, n))), np.quantile(means, [0.025, 0.975]);
- paired tests: scipy.stats.wilcoxon(a, b) with defaults; wins/losses/ties by sign of the per-scene
  IoU difference.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import rasterio

from paths import env_root  # noqa: E402  (environment variables; see README.md)

LABELS = str(env_root("SWM_GSWD_DIR") / "labels" / "{}.tif")
OPERA_CSV = env_root("SWM_S1ML") / "intercomparison_opera/grouped/sid_permanent_water_per_sample_metrics.csv"
OPERA_MASK = str(env_root("SWM_S1ML") / "intercomparison_opera/binary_masks" / "{}_opera_bwtr_binary.tif")
PRED = env_root("SWM_S1ML") / "intercomparison_s1aef/predictions/paper_labelclass_v1"
OW = PRED / "openwater/58321212"
TM = PRED / "tminus1/58321217"

# The five methods of the paper's common mask (3 m inference, existing rasters).
PAPER_3M = {
    "S1-only": OW / "width_k0_seed42_current_tta",
    "S1+AEF(t)": OW / "width_k16_seed42_current_tta",
    "S1+AEF(t-1) swap": OW / "width_k16_seed42_tminus1_tta",
    "S1+AEF(t-1) trained": TM / "width_k16_seed42_tminus1_tta",
}
OPERA_NAME = "OPERA DSWx-S1"


def prob_index(root: Path) -> Dict[str, str]:
    return {os.path.basename(f).replace("_prob.tif", ""): f
            for f in glob.glob(str(Path(root) / "probabilities" / "*" / "*_prob.tif"))}


def read_prob(path: str) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        arr, nodata = src.read(1), src.nodata
    good = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        good &= arr != nodata
    return arr, good


def read_label(sid: str) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(LABELS.format(sid)) as src:
        label = src.read(1)
    return label, (label == 0) | (label == 1)


def read_opera(sid: str) -> tuple[np.ndarray, np.ndarray]:
    path = OPERA_MASK.format(sid)
    with rasterio.open(path) as src:
        arr, nodata = src.read(1), src.nodata
    good = np.ones(arr.shape, bool) if nodata is None else arr != nodata
    return arr, good


def counts(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(pred.size - tp - fp - fn)
    return np.array([tp, fp, fn, tn], np.int64)


def iou(c) -> float:
    tp, fp, fn = c[0], c[1], c[2]
    return float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0


def pooled_metrics(c) -> dict:
    tp, fp, fn, tn = (int(v) for v in c)
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "pooled_water_iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def bootstrap_ci(values: Sequence[float]) -> list:
    values = np.asarray(values, float)
    n = len(values)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(10000, n))
    means = values[idx].mean(axis=1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def paired(a: Sequence[float], b: Sequence[float]) -> dict:
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError as exc:  # all differences zero
        p = float("nan")
    return {"n": int(len(d)), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
            "ties": int((d == 0).sum()), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
            "wilcoxon_p": p}


def summarize(per_scene_counts: Dict[str, Dict[str, np.ndarray]], scenes: Sequence[str]) -> dict:
    """per_scene_counts[method][sid] = [tp, fp, fn, tn]."""
    out = {}
    for name, by_sid in per_scene_counts.items():
        total = np.sum([by_sid[s] for s in scenes], axis=0)
        scene_iou = [iou(by_sid[s]) for s in scenes]
        m = pooled_metrics(total)
        m["per_scene_mean"] = float(np.mean(scene_iou))
        m["per_scene_ci95"] = bootstrap_ci(scene_iou)
        m["valid_pixels"] = int(total.sum())
        out[name] = m
    return out


def scene_ious(per_scene_counts, name, scenes):
    return [iou(per_scene_counts[name][s]) for s in scenes]
