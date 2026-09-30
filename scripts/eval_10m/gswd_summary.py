#!/usr/bin/env python3
"""GSWD summaries per variant and threshold from gswd/per_scene_counts_long.csv (statistics: scripts/orig/scoring.py,
unchanged: pooled metrics, per-scene mean with bootstrap CI (rng 42, 10,000), Wilcoxon with defaults).

Writes gswd/summary_thr030.json, gswd/summary_thr050.json and gswd/reproduction_check.json (G0).
Seed statistics: mean and sample SD (ddof = 1) over seeds 42/43/44.
Variant "g1_average_on_g4_scenes" (and "paper_on_g4_scenes") restricts to the G4 scenes for the G4 comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # scoring, score_10m and evaluate_intercomparison_opera_binary
import scoring as S  # noqa: E402
from paths import env_root  # noqa: E402  (environment variables; see README.md)

OUT = env_root("SWM_RESOLUTION") / "gswd"
PREV_LONG = env_root("SWM_EVAL10M") / "task1_10m/results/per_scene_counts_long.csv"
MODELS = ("S1-only", "S1+AEF(t)")
SEEDS = (42, 43, 44)
EXPECTED = {  # thr 0.30, paper common mask, seed 42 (brief / manuscript)
    "S1-only": {"pooled_water_iou": 0.768, "precision": 0.883, "recall": 0.854, "per_scene_mean": 0.617, "ci": [0.527, 0.702]},
    "S1+AEF(t)": {"pooled_water_iou": 0.851, "precision": 0.891, "recall": 0.950, "per_scene_mean": 0.741, "ci": [0.669, 0.806]},
    "OPERA": {"pooled_water_iou": 0.747, "precision": 0.859, "recall": 0.851, "per_scene_mean": 0.591, "ci": [0.508, 0.670]},
}
EXPECTED_WINS = {"S1+AEF(t) vs OPERA": 48, "S1+AEF(t) vs S1-only": 44, "S1-only vs OPERA": 36}


def by_scene(df, method, seed):
    sel = df[(df.method == method) & (df.seed.astype(str) == str(seed))]
    return {r.sample_id: np.array([r.tp, r.fp, r.fn, r.tn], np.int64) for r in sel.itertuples()}


def variant_summary(df: pd.DataFrame, support: pd.DataFrame, variant: str) -> dict:
    # Scene order = plain string sort (SID03, ..., SID100, SID14, ...), as in the 29 Sep scoring. The bootstrap draws
    # scene indices, so the CI depends on this order (numeric order moves the endpoints by up to 0.003).
    scenes = sorted(df.sample_id.unique())
    per = {f"{m} seed {s}": by_scene(df, m, s) for m in MODELS for s in SEEDS}
    per["OPERA"] = by_scene(df, "OPERA", "")
    summ = S.summarize(per, scenes)
    out = {"variant": variant, "scenes": len(scenes), "scene_ids": scenes,
           "common_pixels": summ["OPERA"]["valid_pixels"],
           "common_water_pixels": int(summ["OPERA"]["tp"] + summ["OPERA"]["fn"]), "methods": {}, "seeds": {},
           "vs_opera": {}, "s1aef_vs_s1only": {}, "empty_union_scenes": {}}
    out["methods"] = {"S1-only": summ["S1-only seed 42"], "S1+AEF(t)": summ["S1+AEF(t) seed 42"], "OPERA": summ["OPERA"],
                      **{k: v for k, v in summ.items() if "seed" in k}}
    for m in MODELS:
        pooled = [summ[f"{m} seed {s}"]["pooled_water_iou"] for s in SEEDS]
        scene_mean = [summ[f"{m} seed {s}"]["per_scene_mean"] for s in SEEDS]
        out["seeds"][m] = {"pooled_iou": {"values": pooled, "mean": float(np.mean(pooled)), "sd": float(np.std(pooled, ddof=1))},
                           "per_scene_mean": {"values": scene_mean, "mean": float(np.mean(scene_mean)),
                                              "sd": float(np.std(scene_mean, ddof=1))}}
    o = S.scene_ious(per, "OPERA", scenes)
    for m in MODELS:
        for s in SEEDS:
            a = S.scene_ious(per, f"{m} seed {s}", scenes)
            out["vs_opera"][f"{m} seed {s}"] = {
                "pooled_iou_diff": summ[f"{m} seed {s}"]["pooled_water_iou"] - summ["OPERA"]["pooled_water_iou"],
                "per_scene_mean_diff": summ[f"{m} seed {s}"]["per_scene_mean"] - summ["OPERA"]["per_scene_mean"],
                **S.paired(a, o)}
        diffs = [out["vs_opera"][f"{m} seed {s}"]["pooled_iou_diff"] for s in SEEDS]
        out["vs_opera"][f"{m} seed mean"] = {"pooled_iou_diff_mean": float(np.mean(diffs)),
                                             "pooled_iou_diff_sd": float(np.std(diffs, ddof=1)),
                                             "wins_range": [min(out["vs_opera"][f"{m} seed {s}"]["wins"] for s in SEEDS),
                                                            max(out["vs_opera"][f"{m} seed {s}"]["wins"] for s in SEEDS)],
                                             "wilcoxon_p_worst": max(out["vs_opera"][f"{m} seed {s}"]["wilcoxon_p"] for s in SEEDS)}
    for s in SEEDS:
        out["s1aef_vs_s1only"][f"seed {s}"] = S.paired(S.scene_ious(per, f"S1+AEF(t) seed {s}", scenes),
                                                        S.scene_ious(per, f"S1-only seed {s}", scenes))
    for k, by in per.items():
        out["empty_union_scenes"][k] = [sid for sid in scenes if by[sid][:3].sum() == 0]
    return out


def reference_valid(support: pd.DataFrame, variant: str, scenes) -> int:
    sup = support[support.sample_id.isin(scenes)]
    col = {"paper": "label_valid_3m", "g2": "label_valid_3m", "g3_3m": "label_valid_3m", "g1_average": "ref_valid_blocks",
           "g1_bilinear": "ref_valid_blocks", "g3_30m": "ref_valid_blocks", "g4": "g4_ref_valid"}[variant.split("_on_")[0]]
    return int(sup[col].sum())


def reproduction(df: pd.DataFrame, summ30: dict, summ50: dict) -> dict:
    res = {"expected_vs_computed": {}, "wins": {}, "exact_counts_vs_29sep": {}}
    ok = True
    m = summ30["paper"]["methods"]
    for name, e in EXPECTED.items():
        g = m[name]
        comp = {"pooled_water_iou": g["pooled_water_iou"], "precision": g["precision"], "recall": g["recall"],
                "per_scene_mean": g["per_scene_mean"], "ci": g["per_scene_ci95"]}
        match = all(round(comp[k], 3) == e[k] for k in ("pooled_water_iou", "precision", "recall", "per_scene_mean")) and \
            [round(x, 3) for x in comp["ci"]] == e["ci"]
        ok &= match
        res["expected_vs_computed"][name] = {"computed": comp, "expected": e, "match_3dp": match}
    v = summ30["paper"]
    got = {"S1+AEF(t) vs OPERA": v["vs_opera"]["S1+AEF(t) seed 42"], "S1+AEF(t) vs S1-only": v["s1aef_vs_s1only"]["seed 42"],
           "S1-only vs OPERA": v["vs_opera"]["S1-only seed 42"]}
    for k, e in EXPECTED_WINS.items():
        res["wins"][k] = {"wins": got[k]["wins"], "losses": got[k]["losses"], "ties": got[k]["ties"],
                          "wilcoxon_p": got[k]["wilcoxon_p"], "expected_wins": e, "match": got[k]["wins"] == e}
        ok &= got[k]["wins"] == e
    res["wins"]["S1-only vs OPERA"]["expected_p"] = 0.015
    ok &= round(got["S1-only vs OPERA"]["wilcoxon_p"], 3) == 0.015
    # exact per-scene counts vs the 29 Sep scoring (common mask; seed rows and OPERA; both thresholds)
    prev = pd.read_csv(PREV_LONG)
    prev = prev[prev["mask"] == "common"]
    n, same = 0, 0
    for thr in (0.30, 0.50):
        p = prev[np.isclose(prev.threshold, thr)].set_index(["method", "sample_id"])
        d = df[(df.variant == "paper") & np.isclose(df.threshold, thr)]
        for r in d.itertuples():
            key = "OPERA DSWx-S1" if r.method == "OPERA" else f"{r.method} seed {r.seed} [10m]"
            e = p.loc[(key, r.sample_id)]
            n += 1
            same += int((r.tp, r.fp, r.fn, r.tn) == (e.tp, e.fp, e.fn, e.tn))
    res["exact_counts_vs_29sep"] = {"rows_compared": n, "rows_identical": same}
    ok &= n == same == 2 * 53 * 7
    res["passed"] = bool(ok)
    return res


def main() -> int:
    df = pd.read_csv(OUT / "per_scene_counts_long.csv", dtype={"seed": str}).fillna({"seed": ""})
    df["seed"] = df.seed.str.replace(".0", "", regex=False)
    support = pd.read_csv(OUT / "per_scene_support.csv")
    g4_scenes = sorted(df[df.variant == "g4"].sample_id.unique())
    out = {}
    for thr in (0.30, 0.50):
        d = df[np.isclose(df.threshold, thr)]
        res = {}
        for v in ("paper", "g1_average", "g1_bilinear", "g2", "g3_3m", "g3_30m", "g4"):
            res[v] = variant_summary(d[d.variant == v], support, v)
        for v in ("g1_average", "paper"):
            res[f"{v}_on_g4_scenes"] = variant_summary(d[(d.variant == v) & d.sample_id.isin(g4_scenes)], support,
                                                       f"{v}_on_g4_scenes")
        for v, r in res.items():
            r["reference_valid_pixels"] = reference_valid(support, v, r["scene_ids"])
            r["common_fraction_of_reference_valid"] = r["common_pixels"] / r["reference_valid_pixels"]
        sup = support
        res["g3_retention"] = {
            "g3_3m_common_over_paper_common": int(sup.g3_3m_common.sum()) / int(sup.paper_common.sum()),
            "g3_3m_water_over_paper_common_water": int(sup.g3_3m_common_water.sum()) / int(sup.paper_common_water.sum()),
            "g3_30m_common_over_g1_average_common": int(sup.g3_30m_common.sum()) / int(sup.g1_average_common.sum()),
            "g3_30m_water_over_g1_average_common_water": int(sup.g3_30m_common_water.sum()) / int(sup.g1_average_common_water.sum()),
            "pure_blocks_over_all_blocks": int(sup.pure_blocks.sum()) / int(sup.blocks.sum())}
        res["trimming"] = {"label_pixels_dropped": int(sup.trimmed_label_pixels.sum()),
                           "label_valid_pixels_dropped": int(sup.trimmed_label_valid.sum()),
                           "label_pixels_total": int(sup.label_valid_3m.sum()) if False else 53 * 1024 * 1024,
                           "per_scene": "1024 x 1024 -> 1020 x 1020 (102 x 102 blocks); 8,176 label pixels per scene"}
        res["paper_rule_common_pixels"] = int(sup.paper_rule_common.sum())
        res["g4_scenes"] = g4_scenes
        out[thr] = res
        (OUT / f"summary_thr{int(round(thr * 100)):03d}.json").write_text(json.dumps(res, indent=2) + "\n")
    rep = reproduction(df, out[0.30], out[0.50])
    (OUT / "reproduction_check.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
