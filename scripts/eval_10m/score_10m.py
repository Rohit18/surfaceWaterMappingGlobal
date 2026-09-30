#!/usr/bin/env python3
"""Task 1 scoring: resample 10 m probabilities to the 3 m label grid (bilinear), then score as in the paper.

Resampling: rasterio.warp.reproject, Resampling.bilinear, src_nodata = -9999 (the probability nodata), destination
= the label's CRS/transform/shape, dst_nodata = -9999. Probabilities are resampled, then thresholded. A 3 m pixel is
valid where the resampled probability is finite and != -9999. Resampled rasters are written under
$OUT/predictions_10m_on_3m/ (float32, nodata -9999).

Masks:
  (a) common: paper common mask (label in {0,1}; valid in the four 3 m model rasters and OPERA) AND valid in the four
      10 m seed-42 model rasters; 3 m and 10 m methods and OPERA are all scored on these pixels.
  (b) own: label in {0,1} AND the method's own valid pixels (reference only).
Thresholds 0.30 and 0.50 (prob >= thr). Statistics as in scoring.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring as S  # noqa: E402
from paths import env_root  # noqa: E402  (environment variables; see README.md)

OUT = env_root("SWM_EVAL10M")
P10 = OUT / "predictions_10m"
R3 = OUT / "predictions_10m_on_3m"
NODATA = -9999.0
THRESHOLDS = (0.30, 0.50)
OW3, TM3 = S.OW, S.TM

# name -> (3 m root, 10 m group/tag)
MAIN = {
    "S1-only": (OW3 / "width_k0_seed42_current_tta", "openwater/width_k0_seed42_current_tta"),
    "S1+AEF(t)": (OW3 / "width_k16_seed42_current_tta", "openwater/width_k16_seed42_current_tta"),
    "S1+AEF(t-1) swap": (OW3 / "width_k16_seed42_tminus1_tta", "openwater/width_k16_seed42_tminus1_tta"),
    "S1+AEF(t-1) trained": (TM3 / "width_k16_seed42_tminus1_tta", "tminus1/width_k16_seed42_tminus1_tta"),
}
SEEDS = {f"{m} seed {s}": (OW3 / f"width_k{k}_seed{s}_current_tta", f"openwater/width_k{k}_seed{s}_current_tta")
         for m, k in (("S1-only", 0), ("S1+AEF(t)", 16)) for s in (42, 43, 44)}
CONTROLS = {
    "daymosaic": {n: g.replace("openwater/", "openwater_daymosaic/").replace("tminus1/", "tminus1_daymosaic/")
                  for n, (_, g) in MAIN.items()},
    "nomargin": {n: g.replace("openwater/", "openwater_nomargin/").replace("tminus1/", "tminus1_nomargin/")
                 for n, (_, g) in MAIN.items()},
}
DAYMOSAIC_SIDS = ["SID05", "SID21", "SID61", "SID63"]
PAIRS = [("S1+AEF(t)", "S1-only"), ("S1+AEF(t)", S.OPERA_NAME), ("S1+AEF(t-1) swap", "S1-only"),
         ("S1+AEF(t)", "S1+AEF(t-1) swap"), ("S1+AEF(t-1) swap", S.OPERA_NAME),
         ("S1+AEF(t-1) trained", S.OPERA_NAME), ("S1-only", S.OPERA_NAME)]


def resample_to_label(prob10: Path, sid: str, dst: Path) -> np.ndarray:
    with rasterio.open(S.LABELS.format(sid)) as lab:
        crs, transform, shape = lab.crs, lab.transform, (lab.height, lab.width)
    out = np.full(shape, NODATA, np.float32)
    with rasterio.open(prob10) as src:
        assert src.nodata == NODATA, (prob10, src.nodata)
        reproject(source=rasterio.band(src, 1), destination=out, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=NODATA, dst_transform=transform, dst_crs=crs, dst_nodata=NODATA,
                  resampling=Resampling.bilinear)
    dst.parent.mkdir(parents=True, exist_ok=True)
    prof = {"driver": "GTiff", "dtype": "float32", "count": 1, "width": shape[1], "height": shape[0], "crs": crs,
            "transform": transform, "nodata": NODATA, "compress": "lzw", "tiled": True, "blockxsize": 256,
            "blockysize": 256}
    with rasterio.open(dst, "w", **prof) as d:
        d.write(out, 1)
        d.update_tags(source=str(prob10), resampling="bilinear", src_nodata=str(NODATA))
    return out


def load10(group_tag: str, sid: str, date: str) -> tuple[np.ndarray, np.ndarray]:
    src = P10 / group_tag / "probabilities" / date / f"{sid}_prob.tif"
    dst = R3 / group_tag / "probabilities" / date / f"{sid}_prob.tif"
    arr = resample_to_label(src, sid, dst) if not dst.exists() else rasterio.open(dst).read(1)
    return arr, np.isfinite(arr) & (arr != NODATA)


def table(per, scenes, names):
    return S.summarize({n: per[n] for n in names}, scenes)


def pairs(per, scenes, suffix=""):
    return {f"{a}{suffix} vs {b}{suffix if b != S.OPERA_NAME else ''}":
            S.paired(S.scene_ious(per, a + suffix, scenes),
                     S.scene_ious(per, b + (suffix if b != S.OPERA_NAME else ""), scenes))
            for a, b in PAIRS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT / "task1_10m" / "results")
    ap.add_argument("--skip-controls", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    idx3 = {n: S.prob_index(r) for n, (r, _) in {**MAIN, **SEEDS}.items()}
    scenes = sorted(idx3["S1-only"])
    dates = {sid: Path(idx3["S1-only"][sid]).parent.name for sid in scenes}
    labels_meta = {}

    # per[mask][thr][method][sid] = counts
    methods_common = [f"{n} [3m]" for n in MAIN] + [f"{n} [10m]" for n in MAIN] + [S.OPERA_NAME] + \
        [f"{n} [3m]" for n in SEEDS] + [f"{n} [10m]" for n in SEEDS]
    ctrl_names = {c: [f"{n} [10m {c}]" for n in MAIN] for c in CONTROLS}
    per = {"common": {t: {m: {} for m in methods_common} for t in THRESHOLDS},
           "paper_common": {t: {m: {} for m in [f"{n} [3m]" for n in MAIN] + [S.OPERA_NAME]} for t in THRESHOLDS},
           "own": {t: {m: {} for m in [f"{n} [3m]" for n in MAIN] + [f"{n} [10m]" for n in MAIN] +
                       [f"{n} [3m]" for n in SEEDS] + [f"{n} [10m]" for n in SEEDS] + [S.OPERA_NAME]}
                   for t in THRESHOLDS},
           "worldcover_pairwise": {t: {m: {} for m in ("S1-only [3m]", "S1+AEF(t) [3m]", "S1-only [10m]",
                                                        "S1+AEF(t) [10m]")} for t in THRESHOLDS}}
    for c in CONTROLS:
        per[f"ctrl_{c}"] = {t: {m: {} for m in [f"{n} [3m]" for n in MAIN] + [f"{n} [10m]" for n in MAIN] +
                                ctrl_names[c] + [S.OPERA_NAME]} for t in THRESHOLDS}
    pix = {"paper_common": 0, "common": 0, "total": 0}
    ctrl_pix = {c: 0 for c in CONTROLS}

    for sid in scenes:
        label, lvalid = S.read_label(sid)
        opera, ovalid = S.read_opera(sid)
        truth = label == 1
        p3, v3, p10, v10 = {}, {}, {}, {}
        for n, (root, g) in {**MAIN, **SEEDS}.items():
            p3[n], v3[n] = S.read_prob(idx3[n][sid])
            p10[n], v10[n] = load10(g, sid, dates[sid])
        paper_common = lvalid & ovalid & v3["S1-only"] & v3["S1+AEF(t)"] & v3["S1+AEF(t-1) swap"] & v3["S1+AEF(t-1) trained"]
        common = paper_common & v10["S1-only"] & v10["S1+AEF(t)"] & v10["S1+AEF(t-1) swap"] & v10["S1+AEF(t-1) trained"]
        pix["paper_common"] += int(paper_common.sum()); pix["common"] += int(common.sum()); pix["total"] += label.size
        # seeds share their width's input validity; check
        for n in SEEDS:
            base = "S1-only" if n.startswith("S1-only") else "S1+AEF(t)"
            assert np.array_equal(v10[n], v10[base]) and np.array_equal(v3[n], v3[base]), (sid, n)
        labels_meta[sid] = {"label_water_fraction": float(truth[lvalid].mean()),
                            "common_valid_fraction": float(common.mean()),
                            "paper_common_valid_fraction": float(paper_common.mean()),
                            "valid10_s1aef_t_fraction": float(v10["S1+AEF(t)"].mean())}
        ctrl = {}
        if not args.skip_controls:
            for c, groups in CONTROLS.items():
                if c == "daymosaic" and sid not in DAYMOSAIC_SIDS:
                    continue
                ctrl[c] = {n: load10(g, sid, dates[sid]) for n, g in groups.items()}
        for t in THRESHOLDS:
            tv = truth[common]
            for n in list(MAIN) + list(SEEDS):
                per["common"][t][f"{n} [3m]"][sid] = S.counts(p3[n][common] >= t, tv)
                per["common"][t][f"{n} [10m]"][sid] = S.counts(p10[n][common] >= t, tv)
                own3 = lvalid & v3[n]; own10 = lvalid & v10[n]
                per["own"][t][f"{n} [3m]"][sid] = S.counts(p3[n][own3] >= t, truth[own3])
                per["own"][t][f"{n} [10m]"][sid] = S.counts(p10[n][own10] >= t, truth[own10])
            per["common"][t][S.OPERA_NAME][sid] = S.counts(opera[common] == 1, tv)
            oo = lvalid & ovalid
            per["own"][t][S.OPERA_NAME][sid] = S.counts(opera[oo] == 1, truth[oo])
            tp = truth[paper_common]
            for n in MAIN:
                per["paper_common"][t][f"{n} [3m]"][sid] = S.counts(p3[n][paper_common] >= t, tp)
            per["paper_common"][t][S.OPERA_NAME][sid] = S.counts(opera[paper_common] == 1, tp)
            # WorldCover-style pairwise mask (as analyze_s1_vs_s1aef_worldcover_errors.py): S1-only & S1+AEF(t) valid
            for res, pp, vv in (("3m", p3, v3), ("10m", p10, v10)):
                pw = vv["S1-only"] & vv["S1+AEF(t)"]
                per["worldcover_pairwise"][t][f"S1-only [{res}]"][sid] = S.counts(pp["S1-only"][pw] >= t, truth[pw])
                per["worldcover_pairwise"][t][f"S1+AEF(t) [{res}]"][sid] = S.counts(pp["S1+AEF(t)"][pw] >= t, truth[pw])
            for c, cc in ctrl.items():
                cm = common & np.logical_and.reduce([v for _, v in cc.values()])
                if t == THRESHOLDS[0]:
                    ctrl_pix[c] += int(cm.sum())
                tc = truth[cm]
                for n in MAIN:
                    per[f"ctrl_{c}"][t][f"{n} [3m]"][sid] = S.counts(p3[n][cm] >= t, tc)
                    per[f"ctrl_{c}"][t][f"{n} [10m]"][sid] = S.counts(p10[n][cm] >= t, tc)
                    per[f"ctrl_{c}"][t][f"{n} [10m {c}]"][sid] = S.counts(cc[n][0][cm] >= t, tc)
                per[f"ctrl_{c}"][t][S.OPERA_NAME][sid] = S.counts(opera[cm] == 1, tc)
        print(sid, {k: round(v, 4) for k, v in labels_meta[sid].items()}, flush=True)

    res = {"pixels": pix, "control_pixels": ctrl_pix, "scenes": len(scenes)}
    rows = []
    for t in THRESHOLDS:
        key = f"thr_{t:.2f}"
        res[key] = {}
        # (a) common mask
        c = per["common"][t]
        res[key]["common_mask"] = {
            "methods": table(c, scenes, list(c)),
            "paired_10m": pairs(c, scenes, " [10m]"),
            "paired_3m": pairs(c, scenes, " [3m]"),
            "paired_10m_vs_3m": {n: S.paired(S.scene_ious(c, f"{n} [10m]", scenes), S.scene_ious(c, f"{n} [3m]", scenes))
                                 for n in list(MAIN) + list(SEEDS)},
        }
        for s in (43, 44):
            res[key]["common_mask"][f"paired_seed{s}"] = {
                res_: S.paired(S.scene_ious(c, f"S1+AEF(t) seed {s} [{res_}]", scenes),
                               S.scene_ious(c, f"S1-only seed {s} [{res_}]", scenes)) for res_ in ("3m", "10m")}
        # paper common (3 m) as a re-check
        res[key]["paper_common_mask_3m"] = {"methods": table(per["paper_common"][t], scenes, list(per["paper_common"][t])),
                                             "paired": pairs(per["paper_common"][t], scenes, " [3m]")}
        # (b) own masks
        res[key]["own_mask"] = {"methods": table(per["own"][t], scenes, list(per["own"][t])),
                                "paired_10m": pairs(per["own"][t], scenes, " [10m]"),
                                "paired_3m": pairs(per["own"][t], scenes, " [3m]")}
        # WorldCover-style scene counts, S1+AEF(t) vs S1-only
        wc = {}
        for name, src in (("pairwise_mask", per["worldcover_pairwise"][t]), ("common_mask", c)):
            for r in ("3m", "10m"):
                d = np.array(S.scene_ious(src, f"S1+AEF(t) [{r}]", scenes)) - np.array(S.scene_ious(src, f"S1-only [{r}]", scenes))
                wc[f"{name}_{r}"] = {"gain_gt_0.05": int((d > 0.05).sum()), "abs_le_0.05": int((np.abs(d) <= 0.05).sum()),
                                     "loss_gt_0.05": int((d < -0.05).sum())}
        res[key]["worldcover_style_counts"] = wc
        # controls
        for cname in CONTROLS:
            cc = per[f"ctrl_{cname}"][t]
            sc = sorted(next(iter(cc.values())).keys())
            if not sc:
                continue
            res[key][f"control_{cname}"] = {"scenes": sc, "methods": table(cc, sc, list(cc)),
                                            "paired_10m_ctrl": pairs(cc, sc, f" [10m {cname}]"),
                                            "paired_10m": pairs(cc, sc, " [10m]"), "paired_3m": pairs(cc, sc, " [3m]")}
        for sid in scenes:
            row = {"sample_id": sid, "threshold": t, **labels_meta[sid]}
            for n in list(MAIN) + list(SEEDS):
                i3, i10 = S.iou(c[f"{n} [3m]"][sid]), S.iou(c[f"{n} [10m]"][sid])
                row[f"{n} iou_3m"], row[f"{n} iou_10m"], row[f"{n} delta"] = i3, i10, i10 - i3
            row["OPERA iou"] = S.iou(c[S.OPERA_NAME][sid])
            for cname in CONTROLS:
                cc = per[f"ctrl_{cname}"][t]
                for n in MAIN:
                    if sid in cc[f"{n} [10m {cname}]"]:
                        row[f"{n} iou_10m_{cname}"] = S.iou(cc[f"{n} [10m {cname}]"][sid])
            rows.append(row)
    pd.DataFrame(rows).to_csv(args.out_dir / "per_scene_3m_vs_10m.csv", index=False)
    long = []
    for mask in per:
        for t in per[mask]:
            for m, by in per[mask][t].items():
                for sid, cts in by.items():
                    long.append({"mask": mask, "threshold": t, "method": m, "sample_id": sid, "tp": cts[0], "fp": cts[1],
                                 "fn": cts[2], "tn": cts[3], "water_iou": S.iou(cts)})
    pd.DataFrame(long).to_csv(args.out_dir / "per_scene_counts_long.csv", index=False)
    for t in THRESHOLDS:
        (args.out_dir / f"summary_thr{t:.2f}.json").write_text(json.dumps(
            {"pixels": pix, "control_pixels": ctrl_pix, **res[f"thr_{t:.2f}"]}, indent=2) + "\n")
    print(json.dumps(pix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
