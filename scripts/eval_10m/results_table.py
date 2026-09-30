#!/usr/bin/env python3
"""Summary tables (Markdown) from gswd/ or s1s2water/ summary_thr0{30,50}.json: rows = variant; pooled IoU of S1-only,
S1+AEF and OPERA (seed 42), S1+AEF - OPERA, S1-only - OPERA, per-scene mean IoU, wins vs OPERA (Wilcoxon p), common
pixels; plus a seed line (mean +/- sample SD over seeds 42-44)."""
import json, sys
from pathlib import Path
D = Path(sys.argv[1]); thr = sys.argv[2]; variants = sys.argv[3].split(",")
s = json.loads((D / f"summary_thr{thr}.json").read_text())
A, O, M = ("S1+AEF(t)" if "gswd" in str(D) else "S1+AEF"), "OPERA", "S1-only"
lines = ["| Variant | Common px (% of ref-valid) | Pooled IoU S1-only / S1+AEF / OPERA | S1+AEF - OPERA | S1-only - OPERA | "
         "Per-scene mean S1-only / S1+AEF / OPERA | S1+AEF > OPERA (W/L/T, p) | S1-only > OPERA (W/L/T, p) |",
         "|---|---|---|---|---|---|---|---|"]
seed_lines = []
for v in variants:
    r = s[v]; m = r["methods"]; vs = r["vs_opera"]
    frac = f"{100*r['common_fraction_of_reference_valid']:.1f}%" if 'common_fraction_of_reference_valid' in r else 'n/a'
    a42, s42 = vs[f"{A} seed 42"], vs[f"{M} seed 42"]
    lines.append(f"| {v} | {r['common_pixels']:,} ({frac}) | "
                 f"{m[M]['pooled_water_iou']:.3f} / {m[A]['pooled_water_iou']:.3f} / {m[O]['pooled_water_iou']:.3f} | "
                 f"{a42['pooled_iou_diff']:+.3f} | {s42['pooled_iou_diff']:+.3f} | "
                 f"{m[M]['per_scene_mean']:.3f} / {m[A]['per_scene_mean']:.3f} / {m[O]['per_scene_mean']:.3f} | "
                 f"{a42['wins']}/{a42['losses']}/{a42['ties']}, {a42['wilcoxon_p']:.2g} | "
                 f"{s42['wins']}/{s42['losses']}/{s42['ties']}, {s42['wilcoxon_p']:.2g} |")
    sd = r["seeds"]
    seed_lines.append(f"| {v} | {sd[M]['pooled_iou']['mean']:.3f} +/- {sd[M]['pooled_iou']['sd']:.3f} | "
                      f"{sd[A]['pooled_iou']['mean']:.3f} +/- {sd[A]['pooled_iou']['sd']:.3f} | "
                      f"{sd[M]['per_scene_mean']['mean']:.3f} +/- {sd[M]['per_scene_mean']['sd']:.3f} | "
                      f"{sd[A]['per_scene_mean']['mean']:.3f} +/- {sd[A]['per_scene_mean']['sd']:.3f} | "
                      f"{vs[f'{A} seed mean']['wins_range'][0]}-{vs[f'{A} seed mean']['wins_range'][1]}, {vs[f'{A} seed mean']['wilcoxon_p_worst']:.2g} | "
                      f"{vs[f'{M} seed mean']['wins_range'][0]}-{vs[f'{M} seed mean']['wins_range'][1]}, {vs[f'{M} seed mean']['wilcoxon_p_worst']:.2g} |")
print("\n".join(lines))
print()
print("| Variant | Pooled IoU S1-only, seeds 42-44 | Pooled IoU S1+AEF, seeds | Per-scene mean S1-only, seeds | Per-scene mean S1+AEF, seeds | S1+AEF > OPERA wins (range), worst p | S1-only > OPERA wins (range), worst p |")
print("|---|---|---|---|---|---|---|")
print("\n".join(seed_lines))
