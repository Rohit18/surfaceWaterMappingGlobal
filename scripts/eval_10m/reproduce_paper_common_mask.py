#!/usr/bin/env python3
"""Step 0: reproduce the paper's common-mask Table I values (3 m inference, seed 42) before any new run.

Writes per-scene confusion counts at 0.30 and 0.50 and the pooled / per-scene / paired summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring as S  # noqa: E402

THRESHOLDS = (0.30, 0.50)
PAIRS = [
    ("S1+AEF(t)", "S1-only"),
    ("S1+AEF(t)", S.OPERA_NAME),
    ("S1+AEF(t-1) swap", "S1-only"),
    ("S1+AEF(t)", "S1+AEF(t-1) swap"),
    ("S1+AEF(t-1) swap", S.OPERA_NAME),
    ("S1+AEF(t-1) trained", S.OPERA_NAME),
    ("S1-only", S.OPERA_NAME),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    idx = {name: S.prob_index(root) for name, root in S.PAPER_3M.items()}
    scenes = sorted(idx["S1-only"])
    assert len(scenes) == 53 and all(sorted(v) == scenes for v in idx.values()), "scene sets differ"

    per = {t: {n: {} for n in list(S.PAPER_3M) + [S.OPERA_NAME]} for t in THRESHOLDS}
    rows = []
    total_px = common_px = 0
    for sid in scenes:
        label, lvalid = S.read_label(sid)
        opera, ovalid = S.read_opera(sid)
        valid = lvalid & ovalid
        probs = {}
        for name in S.PAPER_3M:
            arr, good = S.read_prob(idx[name][sid])
            probs[name] = arr
            valid &= good
        total_px += label.size
        common_px += int(valid.sum())
        truth = label[valid] == 1
        for t in THRESHOLDS:
            for name, arr in probs.items():
                per[t][name][sid] = S.counts(arr[valid] >= t, truth)
            per[t][S.OPERA_NAME][sid] = S.counts(opera[valid] == 1, truth)
            for name in per[t]:
                c = per[t][name][sid]
                rows.append({"sample_id": sid, "threshold": t, "method": name, "tp": c[0], "fp": c[1],
                             "fn": c[2], "tn": c[3], "water_iou": S.iou(c)})
    pd.DataFrame(rows).to_csv(args.out_dir / "paper3m_common_mask_per_scene.csv", index=False)

    result = {"common_valid_pixels": common_px, "total_pixels": total_px, "scenes": len(scenes)}
    for t in THRESHOLDS:
        summ = S.summarize(per[t], scenes)
        pairs = {f"{a} vs {b}": S.paired(S.scene_ious(per[t], a, scenes), S.scene_ious(per[t], b, scenes))
                 for a, b in PAIRS}
        result[f"thr_{t:.2f}"] = {"methods": summ, "paired": pairs}
    (args.out_dir / "paper3m_common_mask_summary.json").write_text(json.dumps(result, indent=2) + "\n")

    print(f"common valid pixels {common_px:,} of {total_px:,} ({100 * common_px / total_px:.2f}%)")
    for t in THRESHOLDS:
        print(f"\nthreshold {t:.2f}")
        for n, m in result[f"thr_{t:.2f}"]["methods"].items():
            lo, hi = m["per_scene_ci95"]
            print(f"  {n:<22} P {m['precision']:.3f} R {m['recall']:.3f} Dice {m['dice']:.3f} "
                  f"IoU {m['pooled_water_iou']:.4f} per-scene {m['per_scene_mean']:.3f} [{lo:.3f}, {hi:.3f}]")
        for k, v in result[f"thr_{t:.2f}"]["paired"].items():
            print(f"  {k:<40} {v['wins']}/{v['losses']}/{v['ties']} mean {v['mean_diff']:+.4f} "
                  f"median {v['median_diff']:+.4f} p {v['wilcoxon_p']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
