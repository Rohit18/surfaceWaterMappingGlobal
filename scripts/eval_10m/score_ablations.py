#!/usr/bin/env python3
"""Score the label-class ablation runs of array 58321212 (AEF width, training-set size) on the 53 GSWD scenes.

--res 3m : existing 3 m-inference probabilities (paper, predictions/paper_labelclass_v1/openwater/58321212/).
--res 10m: 10 m-inference probabilities resampled bilinearly to the 3 m label grid. k0 and k16 (seeds 42/43/44) come
           from $PREV/predictions_10m_on_3m/ (read only); the other 24 runs are resampled here from
           $OUT/predictions_10m/ into $OUT/predictions_10m_on_3m/ with score_10m.resample_to_label (unchanged code).

Masks: own = label in {0,1} AND the run's own valid pixels (original Table II convention);
       common = the paper's common mask (label in {0,1}, OPERA valid, the four 3 m paper rasters valid, the four
       10 m seed-42 rasters valid; 51,897,488 px). Thresholds 0.30 and 0.50 (prob >= thr).
Output: long CSV of per-scene counts (run, res, mask, threshold, sample_id, tp, fp, fn, tn).
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring as S  # noqa: E402
from paths import env_root  # noqa: E402  (environment variables; see README.md)
import score_10m as T  # noqa: E402  (resample_to_label, paths of the 29 Sep run)

OUT = env_root("SWM_ABLATION10M")
PREV = env_root("SWM_EVAL10M")
SEEDS = (42, 43, 44)
WIDTHS = (0, 1, 2, 3, 4, 8, 16)
SIZES = (1000, 2000, 3000)
TAGS = [f"width_k{k}_seed{s}_current_tta" for k in WIDTHS for s in SEEDS] + \
       [f"train{n}_k16_seed{s}_current_tta" for n in SIZES for s in SEEDS]
PREV_10M = {f"width_k{k}_seed{s}_current_tta" for k in (0, 16) for s in SEEDS}
THRESHOLDS = (0.30, 0.50)
PAPER10 = {"S1-only": "openwater/width_k0_seed42_current_tta", "S1+AEF(t)": "openwater/width_k16_seed42_current_tta",
           "S1+AEF(t-1) swap": "openwater/width_k16_seed42_tminus1_tta",
           "S1+AEF(t-1) trained": "tminus1/width_k16_seed42_tminus1_tta"}

IDX3 = {t: S.prob_index(S.OW / t) for t in TAGS}
IDXP = {n: S.prob_index(r) for n, r in S.PAPER_3M.items()}


def prob10_on3(tag: str, sid: str, date: str) -> tuple[np.ndarray, np.ndarray]:
    if tag in PREV_10M:
        return S.read_prob(str(PREV / "predictions_10m_on_3m/openwater" / tag / "probabilities" / date / f"{sid}_prob.tif"))
    src = OUT / "predictions_10m/openwater" / tag / "probabilities" / date / f"{sid}_prob.tif"
    dst = OUT / "predictions_10m_on_3m/openwater" / tag / "probabilities" / date / f"{sid}_prob.tif"
    if dst.exists():
        return S.read_prob(str(dst))
    arr = T.resample_to_label(src, sid, dst)
    return arr, np.isfinite(arr) & (arr != T.NODATA)


def common_mask(sid: str, date: str, lvalid: np.ndarray) -> np.ndarray:
    _, ovalid = S.read_opera(sid)
    m = lvalid & ovalid
    for n in S.PAPER_3M:
        m &= S.read_prob(IDXP[n][sid])[1]
        m &= S.read_prob(str(PREV / "predictions_10m_on_3m" / PAPER10[n] / "probabilities" / date / f"{sid}_prob.tif"))[1]
    return m


def score_scene(args) -> list[dict]:
    sid, res = args
    date = Path(IDX3[TAGS[0]][sid]).parent.name
    label, lvalid = S.read_label(sid)
    truth = label == 1
    common = common_mask(sid, date, lvalid)
    rows = []
    for tag in TAGS:
        if res == "3m":
            assert Path(IDX3[tag][sid]).parent.name == date, (tag, sid)
            prob, valid = S.read_prob(IDX3[tag][sid])
        else:
            prob, valid = prob10_on3(tag, sid, date)
        own = lvalid & valid
        not_covered = int((common & ~valid).sum())
        for t in THRESHOLDS:
            for mname, m in (("own", own), ("common", common)):
                c = S.counts(prob[m] >= t, truth[m])
                rows.append({"run": tag, "res": res, "mask": mname, "threshold": t, "sample_id": sid, "date": date,
                             "tp": c[0], "fp": c[1], "fn": c[2], "tn": c[3], "common_px_not_valid_in_run": not_covered})
    print(sid, res, int(common.sum()), flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", choices=("3m", "10m"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    scenes = sorted(IDX3[TAGS[0]], key=lambda s: int(s[3:]))
    assert len(scenes) == 53 and all(sorted(IDX3[t]) == sorted(scenes) for t in TAGS)
    with Pool(a.procs) as pool:
        rows = [r for part in pool.map(score_scene, [(s, a.res) for s in scenes]) for r in part]
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    cm = df[(df["mask"] == "common") & (df.threshold == 0.30) & (df.run == TAGS[0])]
    print("common mask pixels:", int(cm[["tp", "fp", "fn", "tn"]].sum().sum()))
    print("max common px not valid in a run:", int(df.common_px_not_valid_in_run.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
