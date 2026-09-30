#!/usr/bin/env python3
"""Aggregate per-scene ablation counts into Table II values (mean +/- population SD over seeds 42/43/44).

Per run: per-scene mean water IoU (53 scenes), pooled recall and pooled IoU (TP/FP/FN summed over scenes).
Across seeds: statistics.fmean and statistics.pstdev, as aggregate_paper_retrain.aggregate_runs.
Conventions: (i) threshold 0.50, each run's own valid pixels (original Table II); (ii) threshold 0.30, common mask.
--check-3m compares convention (i) at 3 m with ablation_metrics.csv of 58321212 and the per-sample CSVs of the
evaluation reports (exact TP/FP/FN/TN per scene).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

from paths import env_root  # noqa: E402  (environment variables; see README.md)

OW_REPORTS = env_root("SWM_S1ML") / "training_runs/paper_labelclass_v1/openwater/reports/58321212/evaluation"
ABLATION_CSV = env_root("SWM_S1ML") / "training_runs/paper_labelclass_v1/openwater/results/58321212/ablation_metrics.csv"
SEEDS = (42, 43, 44)
CONFIGS = [("aef_width", k, [f"width_k{k}_seed{s}_current_tta" for s in SEEDS]) for k in (0, 1, 2, 3, 4, 8, 16)] + \
          [("training_tiles", n, [f"train{n}_k16_seed{s}_current_tta" for s in SEEDS]) for n in (1000, 2000, 3000)] + \
          [("training_tiles", 4222, [f"width_k16_seed{s}_current_tta" for s in SEEDS])]
CONVENTIONS = {"thr050_own": (0.50, "own"), "thr030_common": (0.30, "common")}


def iou(tp, fp, fn):
    return tp / (tp + fp + fn) if tp + fp + fn else 0.0


def run_stats(df: pd.DataFrame) -> dict:
    assert len(df) == 53, len(df)
    scene = [iou(r.tp, r.fp, r.fn) for r in df.itertuples()]
    tp, fp, fn = int(df.tp.sum()), int(df.fp.sum()), int(df.fn.sum())
    return {"per_scene_mean_iou": statistics.fmean(scene), "pooled_recall": tp / max(tp + fn, 1),
            "pooled_iou": iou(tp, fp, fn), "pooled_precision": tp / max(tp + fp, 1), "tp": tp, "fp": fp, "fn": fn,
            "valid_pixels": int(df[["tp", "fp", "fn", "tn"]].sum().sum())}


def aggregate(counts: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    per_run, rows = [], []
    for conv, (thr, mask) in CONVENTIONS.items():
        sub = counts[(counts.threshold == thr) & (counts["mask"] == mask)]
        for ablation, value, tags in CONFIGS:
            st = []
            for t in tags:
                s = run_stats(sub[sub.run == t])
                st.append(s)
                per_run.append({"convention": conv, "ablation": ablation, "value": value, "run": t, **s})
            row = {"convention": conv, "ablation": ablation, "value": value, "seeds": len(st)}
            for key in ("per_scene_mean_iou", "pooled_recall", "pooled_iou"):
                vals = [s[key] for s in st]
                row[f"{key}_mean"], row[f"{key}_std"] = statistics.fmean(vals), statistics.pstdev(vals)
                row[f"{key}_seeds"] = vals
            rows.append(row)
    return per_run, rows


def check_3m(counts: pd.DataFrame, rows: list[dict]) -> dict:
    ref = pd.read_csv(ABLATION_CSV)
    out = {"ablation_metrics_csv": str(ABLATION_CSV), "rows": [], "per_sample_exact": {}}
    ok = True
    for r in rows:
        if r["convention"] != "thr050_own":
            continue
        e = ref[(ref.ablation == r["ablation"]) & (ref.value == r["value"])].iloc[0]
        got = {"per_scene_water_iou_mean": round(r["per_scene_mean_iou_mean"], 6),
               "per_scene_water_iou_std": round(r["per_scene_mean_iou_std"], 6),
               "recall_mean": round(r["pooled_recall_mean"], 6), "recall_std": round(r["pooled_recall_std"], 6)}
        exp = {k: float(e[k]) for k in got}
        # ablation_metrics.csv averages the per-sample CSV's water_iou, which is rounded to 6 decimals; this can move
        # the 6th decimal of the SD by 1 (k0: 0.0665145053 from rounded vs 0.0665144724 from exact counts).
        match6 = all(abs(got[k] - exp[k]) < 5e-7 for k in got)
        match = all(abs(got[k] - exp[k]) < 1.0000001e-6 for k in got)
        ok &= match
        out["rows"].append({"ablation": r["ablation"], "value": int(r["value"]), "computed": got, "csv": exp,
                            "match_6dp": match6, "match_within_1e-6": match,
                            "match_3dp": all(round(got[k], 3) == round(exp[k], 3) for k in got)})
    sub = counts[(counts.threshold == 0.50) & (counts["mask"] == "own")]
    for run in sorted(sub.run.unique()):
        f = OW_REPORTS / run / f"{run}_per_sample_metrics.csv"
        e = pd.read_csv(f).set_index("sample_id")
        g = sub[sub.run == run].set_index("sample_id")
        same = all(int(g.loc[s, c]) == int(e.loc[s, c]) for s in e.index for c in ("tp", "fp", "fn", "tn"))
        out["per_sample_exact"][run] = same
        ok &= same
    out["passed"] = bool(ok)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--check-3m", action="store_true")
    a = ap.parse_args()
    counts = pd.read_csv(a.counts)
    per_run, rows = aggregate(counts)
    pd.DataFrame(per_run).to_csv(f"{a.out_prefix}_per_run.csv", index=False)
    res = {"counts": str(a.counts), "sd": "population (statistics.pstdev), as aggregate_paper_retrain.py",
           "configs": rows}
    if a.check_3m:
        res["check_3m"] = check_3m(counts, rows)
        print(json.dumps(res["check_3m"], indent=1))
    Path(f"{a.out_prefix}_summary.json").write_text(json.dumps(res, indent=2) + "\n")
    for r in rows:
        print(f"{r['convention']:14s} {r['ablation']:15s} {r['value']:5d}  IoU {r['per_scene_mean_iou_mean']:.3f} "
              f"+/- {r['per_scene_mean_iou_std']:.3f}  recall {r['pooled_recall_mean']:.3f} +/- "
              f"{r['pooled_recall_std']:.3f}  pooled {r['pooled_iou_mean']:.3f} +/- {r['pooled_iou_std']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
