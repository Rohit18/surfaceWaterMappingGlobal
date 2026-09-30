#!/usr/bin/env python3
"""Table II at 3 m and 10 m inference from ablation_{3m,10m}_summary.json (aggregate_ablations.py).

Writes ablations_10m/table_II_3m_vs_10m.md (both conventions, per-scene IoU, pooled recall, pooled IoU, mean +/- SD
over seeds 42/43/44), table_II_10m.tex (threshold 0.50, own valid pixels: the original Table II convention) and
table_II_10m_thr030_common.tex (threshold 0.30, common valid mask: the main-text convention).
"""
from __future__ import annotations

import json
from pathlib import Path

from paths import env_root  # noqa: E402  (environment variables; see README.md)

D = env_root("SWM_ABLATION10M") / "ablations_10m"
CONV = {"thr050_own": "threshold 0.50, each run's own valid pixels (original Table II convention)",
        "thr030_common": "threshold 0.30, common valid mask, 51,897,488 px (main-text convention)"}


def load(res):
    rows = json.loads((D / f"ablation_{res}_summary.json").read_text())["configs"]
    return {(r["convention"], r["ablation"], r["value"]): r for r in rows}


def label(ablation, value, tex=False):
    if ablation == "aef_width":
        if value == 0:
            return "S1 only ($k=0$)" if tex else "S1 only (k=0)"
        return (f"$k={value}$" if tex else f"k={value}") + (" (ours)" if value == 16 else "")
    return f"{value:,}".replace(",", "{,}" if tex else ",") + (" (full training split)" if value == 4222 else " tiles")


ORDER = [("aef_width", k) for k in (0, 1, 2, 3, 4, 8, 16)] + [("training_tiles", n) for n in (1000, 2000, 3000, 4222)]


def ms(r, key, tex=False):
    sep = r"\,$\pm$\," if tex else " +/- "
    return f"{r[key + '_mean']:.3f}{sep}{r[key + '_std']:.3f}"


def markdown(t3, t10):
    out = ["# Table II at 3 m and 10 m inference", "",
           "Per run: per-scene mean water IoU over the 53 scenes, pooled recall and pooled IoU (TP, FP, FN summed over "
           "scenes). Mean +/- population SD over seeds 42/43/44 (statistics.pstdev, as aggregate_paper_retrain.py). "
           "Panel (b) 4,222 tiles = the k = 16 runs of panel (a).", ""]
    for conv, desc in CONV.items():
        out += [f"## {desc}", "",
                "| Configuration | Per-scene IoU, 3 m | Per-scene IoU, 10 m | Recall, 3 m | Recall, 10 m | Pooled IoU, 3 m | Pooled IoU, 10 m |",
                "|---|---|---|---|---|---|---|"]
        for i, (ab, v) in enumerate(ORDER):
            if i == 7:
                out.append("| **(b) training tiles, k = 16** | | | | | | |")
            if i == 0:
                out.append("| **(a) AEF width, full training set** | | | | | | |")
            a, b = t3[(conv, ab, v)], t10[(conv, ab, v)]
            out.append(f"| {label(ab, v)} | {ms(a, 'per_scene_mean_iou')} | {ms(b, 'per_scene_mean_iou')} | "
                       f"{ms(a, 'pooled_recall')} | {ms(b, 'pooled_recall')} | {ms(a, 'pooled_iou')} | {ms(b, 'pooled_iou')} |")
        out.append("")
        out += ["Seed values (per-scene IoU, 10 m): " + "; ".join(
            f"{label(ab, v)} " + "/".join(f"{x:.3f}" for x in t10[(conv, ab, v)]["per_scene_mean_iou_seeds"])
            for ab, v in ORDER), ""]
    return "\n".join(out) + "\n"


def tex(t10, conv, caption, lab):
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{" + caption + "}", r"\label{" + lab + "}", r"\footnotesize",
             r"\begin{tabular}{lcc}", r"\toprule", r"Configuration & Water IoU & Recall \\", r"\midrule",
             r"\multicolumn{3}{l}{\textit{(a) AEF width $k$, full training set}} \\"]
    for i, (ab, v) in enumerate(ORDER):
        if i == 7:
            lines += [r"\midrule", r"\multicolumn{3}{l}{\textit{(b) Training tiles, $k=16$}} \\"]
        r = t10[(conv, ab, v)]
        row = f"{label(ab, v, True)} & {ms(r, 'per_scene_mean_iou', True)} & {ms(r, 'pooled_recall', True)} \\\\"
        if ab == "aef_width" and v == 16:
            row = (r"\textbf{$k=16$ (ours)} & \textbf{" + ms(r, 'per_scene_mean_iou', True) + r"} & \textbf{"
                   + ms(r, 'pooled_recall', True) + r"} \\")
        lines.append(row)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    t3, t10 = load("3m"), load("10m")
    (D / "table_II_3m_vs_10m.md").write_text(markdown(t3, t10))
    (D / "table_II_10m.tex").write_text(tex(
        t10, "thr050_own",
        r"Ablations on the 53 PlanetScope reference scenes with inference at 10\,m: per-scene water IoU and pooled "
        r"recall, mean\,$\pm$\,SD over three seeds, threshold 0.5.", "tab:ablations"))
    (D / "table_II_10m_thr030_common.tex").write_text(tex(
        t10, "thr030_common",
        r"Ablations on the 53 PlanetScope reference scenes with inference at 10\,m: per-scene water IoU and pooled "
        r"recall on the common valid pixels, mean\,$\pm$\,SD over three seeds, threshold 0.30.", "tab:ablations030"))
    print((D / "table_II_3m_vs_10m.md").read_text())


if __name__ == "__main__":
    main()
