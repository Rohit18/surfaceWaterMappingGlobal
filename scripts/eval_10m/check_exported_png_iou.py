#!/usr/bin/env python3
"""Check (Task A): IoUs recomputed from the exported PNGs alone reproduce the 29 Sep scoring.

For every scene: water = <model>_water_t030.png == 255, truth = label_water.png == 255, scored pixels =
common_valid.png == 255. TP/FP/FN/TN must equal the rows mask = common, threshold = 0.30, method "S1-only [10m]" /
"S1+AEF(t) [10m]" of revision_checks_20260929/task1_10m/results/per_scene_counts_long.csv exactly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from paths import env_root  # noqa: E402  (environment variables; see README.md)

OUT = env_root("SWM_ABLATION10M")
IMG = OUT / "images_10m/planetscope_53_scenes"
LONG = env_root("SWM_EVAL10M") / "task1_10m/results/per_scene_counts_long.csv"
PAIRS = {"s1only": "S1-only [10m]", "s1aef_t": "S1+AEF(t) [10m]"}


def png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def main() -> int:
    ref = pd.read_csv(LONG)
    ref = ref[(ref["mask"] == "common") & np.isclose(ref.threshold, 0.30)].set_index(["method", "sample_id"])
    rows, ok = [], True
    for d in sorted(p for p in IMG.iterdir() if p.is_dir()):
        sid = d.name
        truth, common = png(d / "label_water.png") == 255, png(d / "common_valid.png") == 255
        meta = json.loads((d / "layers.json").read_text())
        for model, method in PAIRS.items():
            w = png(d / f"{model}_water_t030.png")
            assert not (common & (w == 128)).any(), (sid, model, "common pixel marked no data")
            pred = w == 255
            tp = int((pred & truth & common).sum()); fp = int((pred & ~truth & common).sum())
            fn = int((~pred & truth & common).sum()); tn = int((~pred & ~truth & common).sum())
            r = ref.loc[(method, sid)]
            same = (tp, fp, fn, tn) == (int(r.tp), int(r.fp), int(r.fn), int(r.tn))
            iou = tp / max(tp + fp + fn, 1)
            same_json = abs(meta["panel_water_iou_full"][model] - iou) < 1e-12
            ok &= same and same_json
            rows.append({"sample_id": sid, "model": model, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "iou_png": iou,
                         "iou_29sep": float(r.water_iou), "counts_equal": same, "layers_json_equal": same_json})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fig1B/check_png_iou_vs_29sep.csv", index=False)
    print(f"{len(df)} scene-model pairs; counts equal: {int(df.counts_equal.sum())}; layers.json equal: "
          f"{int(df.layers_json_equal.sum())}; max |IoU diff| = {float((df.iou_png - df.iou_29sep).abs().max()):.3g}")
    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
