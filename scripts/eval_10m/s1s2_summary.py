#!/usr/bin/env python3
"""S1S2-Water summaries per variant and threshold from s1s2water/per_scene_counts_long.csv.

Statistics as for GSWD (scripts/orig/scoring.py: pooled metrics, per-scene mean with bootstrap CI over scenes in
numeric order, rng 42, 10,000 resamples; Wilcoxon with defaults); seeds: mean and sample SD (ddof = 1), wins vs OPERA
as a range over seeds, worst-case (largest) Wilcoxon p over seeds.
S0 check: the "paper" variant must equal s1s2water_per_scene_all_methods.csv row for row (TP/FP/FN/TN; opera, all six
runs at 0.50 and *_t030 at 0.30) and reproduce Table III at 0.30.
S2 (exploratory): units are (scene, offset class); "s2_native_nonzero"/"s2_frozen_nonzero" pool the classes with a
nonzero offset, "s2_*_zero" the aligned class (0, 0), which is a control (both must be identical).
Writes s1s2water/summary_thr030.json, summary_thr050.json, reproduction_check.json.
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

OUT = env_root("SWM_RESOLUTION") / "s1s2water"
# identical (md5) to the 22 Sep bundle's 4_results_s1s2water/supplement/ copy that the check run read
SUPP = HERE.parents[1] / "external_validation/s1s2_water/results/s1s2water_per_scene_all_methods.csv"
MODELS = ("S1-only", "S1+AEF")
SEEDS = (42, 43, 44)
VARIANTS = ("paper", "s1_average", "opera_union", "s3_pure_paper", "s3_pure_average", "refcut025_paper",
            "refcut075_paper", "refcut025_average", "refcut075_average")


def by_unit(df, method, seed):
    sel = df[(df.method == method) & (df.seed == str(seed))]
    return {r.unit: np.array([r.tp, r.fp, r.fn, r.tn], np.int64) for r in sel.itertuples()}


def variant_summary(df, name):
    units = sorted(df.unit.unique(), key=lambda u: tuple(int(x) if x.isdigit() else x for x in u.replace("@", "_").split("_")))
    per = {f"{m} seed {s}": by_unit(df, m, s) for m in MODELS for s in SEEDS}
    per["OPERA"] = by_unit(df, "OPERA", "")
    summ = S.summarize(per, units)
    out = {"variant": name, "units": len(units), "unit_ids": units, "common_pixels": summ["OPERA"]["valid_pixels"],
           "common_water_pixels": int(summ["OPERA"]["tp"] + summ["OPERA"]["fn"]),
           "methods": {"S1-only": summ["S1-only seed 42"], "S1+AEF": summ["S1+AEF seed 42"], "OPERA": summ["OPERA"],
                       **{k: v for k, v in summ.items() if "seed" in k}},
           "seeds": {}, "vs_opera": {}, "s1aef_vs_s1only": {}, "empty_union_units": {}}
    for m in MODELS:
        out["seeds"][m] = {}
        for key, field in (("pooled_iou", "pooled_water_iou"), ("per_scene_mean", "per_scene_mean"),
                           ("precision", "precision"), ("recall", "recall")):
            vals = [summ[f"{m} seed {s}"][field] for s in SEEDS]
            out["seeds"][m][key] = {"values": vals, "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
    o = S.scene_ious(per, "OPERA", units)
    for m in MODELS:
        for s in SEEDS:
            out["vs_opera"][f"{m} seed {s}"] = {
                "pooled_iou_diff": summ[f"{m} seed {s}"]["pooled_water_iou"] - summ["OPERA"]["pooled_water_iou"],
                "per_scene_mean_diff": summ[f"{m} seed {s}"]["per_scene_mean"] - summ["OPERA"]["per_scene_mean"],
                **S.paired(S.scene_ious(per, f"{m} seed {s}", units), o)}
        d = [out["vs_opera"][f"{m} seed {s}"]["pooled_iou_diff"] for s in SEEDS]
        dm = [out["vs_opera"][f"{m} seed {s}"]["per_scene_mean_diff"] for s in SEEDS]
        out["vs_opera"][f"{m} seed mean"] = {
            "pooled_iou_diff_mean": float(np.mean(d)), "pooled_iou_diff_sd": float(np.std(d, ddof=1)),
            "per_scene_mean_diff_mean": float(np.mean(dm)), "per_scene_mean_diff_sd": float(np.std(dm, ddof=1)),
            "wins_range": [min(out["vs_opera"][f"{m} seed {s}"]["wins"] for s in SEEDS),
                           max(out["vs_opera"][f"{m} seed {s}"]["wins"] for s in SEEDS)],
            "wilcoxon_p_worst": max(out["vs_opera"][f"{m} seed {s}"]["wilcoxon_p"] for s in SEEDS)}
    for s in SEEDS:
        out["s1aef_vs_s1only"][f"seed {s}"] = S.paired(S.scene_ious(per, f"S1+AEF seed {s}", units),
                                                        S.scene_ious(per, f"S1-only seed {s}", units))
    for k, by in per.items():
        out["empty_union_units"][k] = [u for u in units if by[u][:3].sum() == 0]
    return out


def s0_check(df, summ30):
    supp = pd.read_csv(SUPP)
    n = same = 0
    for r in supp.itertuples():
        meth = r.method
        thr = 0.30 if meth.endswith("_t030") else 0.50
        base = meth.replace("_t030", "")
        if base == "opera":
            key = ("OPERA", "")
        else:
            name, seed = base.rsplit("_seed", 1)
            key = ({"s1_only": "S1-only", "s1_aef": "S1+AEF"}[name], seed)
        e = df[(df.variant == "paper") & np.isclose(df.threshold, thr) & (df.method == key[0]) & (df.seed == key[1])
               & (df.sample_id == r.scene_id)].iloc[0]
        n += 1
        same += int((e.tp, e.fp, e.fn, e.tn) == (r.tp, r.fp, r.fn, r.tn))
    p = summ30["paper"]
    comp = {"pooled_iou S1-only": (p["seeds"]["S1-only"]["pooled_iou"]["mean"], p["seeds"]["S1-only"]["pooled_iou"]["sd"]),
            "pooled_iou S1+AEF": (p["seeds"]["S1+AEF"]["pooled_iou"]["mean"], p["seeds"]["S1+AEF"]["pooled_iou"]["sd"]),
            "pooled_iou OPERA": (p["methods"]["OPERA"]["pooled_water_iou"],),
            "per_scene_mean S1-only": (p["seeds"]["S1-only"]["per_scene_mean"]["mean"], p["seeds"]["S1-only"]["per_scene_mean"]["sd"]),
            "per_scene_mean S1+AEF": (p["seeds"]["S1+AEF"]["per_scene_mean"]["mean"], p["seeds"]["S1+AEF"]["per_scene_mean"]["sd"]),
            "per_scene_mean OPERA": (p["methods"]["OPERA"]["per_scene_mean"],),
            "S1+AEF > OPERA wins per seed": tuple(p["vs_opera"][f"S1+AEF seed {s}"]["wins"] for s in SEEDS)}
    exp = {"pooled_iou S1-only": (0.794, 0.026), "pooled_iou S1+AEF": (0.941, 0.001), "pooled_iou OPERA": (0.869,),
           "per_scene_mean S1-only": (0.723, 0.019), "per_scene_mean S1+AEF": (0.862, 0.002),
           "per_scene_mean OPERA": (0.763,), "S1+AEF > OPERA wins per seed": (12, 12, 12)}
    rows = {k: {"computed": comp[k], "expected": exp[k],
                "match": tuple(round(x, 3) if isinstance(x, float) else x for x in comp[k]) == exp[k]} for k in exp}
    ok = all(r["match"] for r in rows.values()) and n == same == len(supp)
    return {"supplement_csv": str(SUPP), "rows_compared": n, "rows_identical": same, "table_iii": rows, "passed": ok}


def main() -> int:
    df = pd.read_csv(OUT / "per_scene_counts_long.csv", dtype={"seed": str}).fillna({"seed": ""})
    df["seed"] = df.seed.str.replace(".0", "", regex=False)
    df["unit"] = df.sample_id.astype(str)
    support = pd.read_csv(OUT / "per_scene_support.csv")
    out = {}
    for thr in (0.30, 0.50):
        d = df[np.isclose(df.threshold, thr)]
        res = {v: variant_summary(d[d.variant == v], v) for v in VARIANTS}
        for v in VARIANTS:
            res[v]["reference_valid_pixels"] = int(support.reference_valid.sum())
            res[v]["common_fraction_of_reference_valid"] = res[v]["common_pixels"] / res[v]["reference_valid_pixels"]
        s2 = d[d.variant.str.startswith("s2_")].copy()
        s2["cls"] = s2.variant.str.split("@").str[1]
        s2["kind"] = s2.variant.str.split("@").str[0]
        s2["unit"] = s2.sample_id.astype(str) + "@" + s2.cls
        for kind in ("s2_native", "s2_frozen"):
            for label, sel in (("nonzero", s2.cls != "0_0"), ("zero", s2.cls == "0_0")):
                sub = s2[(s2.kind == kind) & sel]
                res[f"{kind}_{label}"] = variant_summary(sub, f"{kind}_{label}")
        res["pure_retention"] = {
            "paper_common_retained": int(support.pure_paper_common.sum()) / int(support.paper_common.sum()),
            "paper_common_water_retained": int(support.pure_paper_common_water.sum()) / int(support.paper_common_water.sum()),
            "reference_valid_pure_fraction": int(support.reference_pure.sum()) / int(support.reference_valid.sum())}
        res["opera_mosaic_rule"] = {
            "paper_valid_pixels": int(support.opera_valid.sum()), "union_valid_pixels": int(support.opera_union_valid.sum()),
            "valid_differs": int(support.opera_union_vs_paper_valid_differs.sum()),
            "water_differs_where_both_valid": int(support.opera_union_vs_paper_water_differs_where_both_valid.sum())}
        out[thr] = res
        (OUT / f"summary_thr{int(round(thr * 100)):03d}.json").write_text(json.dumps(res, indent=2) + "\n")
    rep = s0_check(df, out[0.30])
    (OUT / "reproduction_check.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
