#!/usr/bin/env python3
"""Markdown tables comparing the WorldCover class shares at 3 m and 10 m inference (from class_shares.json)."""
import json
from pathlib import Path
from paths import env_root  # noqa: E402  (environment variables; see README.md)
D = env_root("SWM_ABLATION10M") / "worldcover_10m"
a, b = (json.loads((D / d / "class_shares.json").read_text()) for d in ("reproduction_3m", "inference_10m"))
out = ["| | 3 m inference | 10 m inference |", "|---|---:|---:|",
       "| Scene groups gain > 0.05 / abs(delta) <= 0.05 / loss > 0.05 | " +
       " | ".join(f"{x['chip_groups'].get('aef_helps',0)} / {x['chip_groups'].get('neutral',0)} / {x['chip_groups'].get('aef_hurts',0)}" for x in (a, b)) + " |",
       "| Mean per-scene IoU S1-only / S1+AEF(t) (pair-valid pixels) | " +
       " | ".join(f"{x['chip_iou']['mean_s1_water_iou']:.4f} / {x['chip_iou']['mean_s1aef_water_iou']:.4f}" for x in (a, b)) + " |"]
for cat in ("aef_fixes_s1_fn", "aef_fixes_s1_fp", "aef_adds_fn", "aef_adds_fp"):
    out.append(f"| **{a['categories'][cat]['name']}**, pixels | {a['categories'][cat]['total_pixels']:,} | {b['categories'][cat]['total_pixels']:,} |")
    classes = []
    for x in (a, b):
        for c in x["categories"][cat]["classes"][:5]:
            if c["class"] not in classes:
                classes.append(c["class"])
    for cl in classes:
        cells = []
        for x in (a, b):
            m = [c for c in x["categories"][cat]["classes"] if c["class"] == cl]
            cells.append(f"{m[0]['share_pct']:.1f}% ({m[0]['pixels']:,})" if m else "0")
        out.append(f"| &nbsp;&nbsp;{cl} | {cells[0]} | {cells[1]} |")
(D / "worldcover_3m_vs_10m.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
