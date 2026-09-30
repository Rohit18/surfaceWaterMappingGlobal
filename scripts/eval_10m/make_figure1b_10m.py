#!/usr/bin/env python3
"""Figure 1B from the 10 m-inference predictions (Task A): the current scenes and two alternatives chosen by rule.

Rendering is the bundle's figure1b() (export_53_scene_layers.py): 3 x 6 panels, figsize 13.5 x 7.1, same titles,
stretch, overlay colour, "B  Surface water predictions" header, dpi 250, PNG and PDF. Scene loading and the panel
IoUs (threshold 0.30, paper common valid mask) come from export_53_scene_layers_10m.py.

Selection rule for the alternatives (applied to scene_table.csv; all values at 10 m inference, threshold 0.30,
common mask; gain = IoU(S1+AEF(t)) - IoU(S1-only)):
  eligible: label water fraction in [0.05, 0.60], common-valid fraction >= 0.95, not one of the current scenes
            (SID23, SID46, SID17);
  role C (large gain):   gain > 0.20;                                ranked by gain, largest first;
  role B (median gain):  any eligible scene;                         ranked by |gain - median gain of all 53 scenes|;
  role A (S1-only good): IoU(S1-only) >= 0.80 and |gain| <= 0.05;    ranked by |gain|, smallest first;
  a set is filled greedily in the order C, B, A: each role takes its best-ranked candidate whose continent (Natural
  Earth CONTINENT) differs from the roles already filled and which is not already used; ties by SID number.
  alt1 = the first set; alt2 = the same procedure with the alt1 scenes also excluded.
Rows are drawn in the order C, B, A (largest gain first, as the current figure).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_53_scene_layers_10m as X  # noqa: E402

OUT = X.OUT / "fig1B"
CURRENT = [("SID23", "Europe"), ("SID46", "SE Asia"), ("SID17", "Spain")]  # FIG1B_ROWS of the bundle exporter
SHORT = {"United States of America": "USA"}  # row label only
LWF, CVF, C_GAIN, A_S1, A_GAIN = (0.05, 0.60), 0.95, 0.20, 0.80, 0.05
PREV_3M_VS_10M = X.PREV / "task1_10m/results/per_scene_3m_vs_10m.csv"


def table() -> pd.DataFrame:
    d = pd.read_csv(OUT / "scene_table.csv")
    d["s1only"], d["s1aef"], d["opera"] = d.iou_s1only_full, d.iou_s1aef_t_full, d.iou_opera_full
    d["gain"] = d.s1aef - d.s1only
    d["sid_num"] = d.sample_id.str[3:].astype(int)
    p = pd.read_csv(PREV_3M_VS_10M)
    p = p[p.threshold.round(2) == 0.30].set_index("sample_id")
    d["s1only_3m"] = d.sample_id.map(p["S1-only iou_3m"])
    d["s1aef_3m"] = d.sample_id.map(p["S1+AEF(t) iou_3m"])
    # cross-check with the 29 Sep scoring (same mask and threshold)
    assert (d.sample_id.map(p["S1-only iou_10m"]) - d.s1only).abs().max() < 1e-12
    assert (d.sample_id.map(p["S1+AEF(t) iou_10m"]) - d.s1aef).abs().max() < 1e-12
    return d


def candidates(d: pd.DataFrame, median: float) -> dict:
    e = d[d.label_water_fraction.between(*LWF) & (d.common_valid_fraction >= CVF)
          & ~d.sample_id.isin([s for s, _ in CURRENT])].copy()
    e["dist_median"] = (e.gain - median).abs()
    e["abs_gain"] = e.gain.abs()
    return {"C": e[e.gain > C_GAIN].sort_values(["gain", "sid_num"], ascending=[False, True]),
            "B": e.sort_values(["dist_median", "sid_num"]),
            "A": e[(e.s1only >= A_S1) & (e.abs_gain <= A_GAIN)].sort_values(["abs_gain", "sid_num"])}


def pick(cands: dict, exclude: set) -> dict:
    chosen, used_cont = {}, set()
    for role in ("C", "B", "A"):
        for r in cands[role].itertuples():
            if r.sample_id in exclude or r.sample_id in {c.sample_id for c in chosen.values()} or r.continent in used_cont:
                continue
            chosen[role] = r
            used_cont.add(r.continent)
            break
        else:
            raise RuntimeError(f"no candidate for role {role}")
    return chosen


def render(rows, stem: str) -> None:
    """figure1b() of the bundle exporter, with the rows passed in."""
    fig, axes = plt.subplots(3, 6, figsize=(13.5, 7.1))
    for row, (sid, region) in enumerate(rows):
        panels, _ = X.six_panel_images(X.load_scene(sid))
        for col, (image, title) in enumerate(panels):
            axis = axes[row, col]
            axis.imshow(image, interpolation="nearest")
            axis.set_title(title, fontsize=11, pad=3)
            axis.set_xticks([]); axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(1.2)
        axes[row, 0].set_ylabel(f"{sid}\n{region}", fontsize=12, fontweight="bold", labelpad=8)
    fig.text(0.008, 0.985, "B", fontsize=15, fontweight="bold", color="#1b2a3a", va="top")
    fig.text(0.035, 0.985, "Surface water predictions", fontsize=15, fontweight="bold", color="#1b2a3a", va="top")
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.01, top=0.90, wspace=0.18, hspace=0.32)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=250, facecolor="white")
    plt.close(fig)


def fmt_cands(df: pd.DataFrame, extra: str) -> list[str]:
    lines = ["| Scene | Country | Continent | Label water fraction | Common-valid fraction | S1-only | S1+AEF(t) | Gain | "
             + extra + " |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    col = {"rank: gain": "gain", "rank: abs(gain - median)": "dist_median", "rank: abs(gain)": "abs_gain"}[extra]
    for r in df.itertuples():
        lines.append(f"| {r.sample_id} | {r.country} | {r.continent} | {r.label_water_fraction:.4f} | "
                     f"{r.common_valid_fraction:.4f} | {r.s1only:.4f} | {r.s1aef:.4f} | {r.gain:+.4f} | "
                     f"{getattr(r, col):.4f} |")
    return lines


def main() -> int:
    d = table()
    median = float(d.gain.median())
    cands = candidates(d, median)
    current_ids = {s for s, _ in CURRENT}
    alt1 = pick(cands, current_ids)
    alt2 = pick(cands, current_ids | {r.sample_id for r in alt1.values()})
    by = d.set_index("sample_id")
    versions = {
        "current": [(s, reg, "current") for s, reg in CURRENT],
        "alt1": [(alt1[k].sample_id, SHORT.get(alt1[k].country, alt1[k].country), k) for k in ("C", "B", "A")],
        "alt2": [(alt2[k].sample_id, SHORT.get(alt2[k].country, alt2[k].country), k) for k in ("C", "B", "A")],
    }
    role_name = {"C": "large gain (> 0.20)", "B": "near the median gain", "A": "S1-only >= 0.80, |gain| <= 0.05",
                 "current": "current Fig. 1B"}
    rows = []
    for v, items in versions.items():
        render([(s, reg) for s, reg, _ in items], f"figure1B_10m_{v}")
        for i, (s, reg, role) in enumerate(items, 1):
            r = by.loc[s]
            rows.append({"version": v, "row": i, "sample_id": s, "row_label": reg, "role": role_name[role],
                         "country": r.country, "continent": r.continent,
                         "label_water_fraction": r.label_water_fraction, "common_valid_fraction": r.common_valid_fraction,
                         "iou_s1only": round(r.s1only, 4), "iou_s1aef_t": round(r.s1aef, 4),
                         "iou_opera": round(r.opera, 4), "gain": round(r.gain, 4),
                         "s1only_beats_s1aef": bool(r.s1only > r.s1aef),
                         "iou_s1only_3m_inference": round(r.s1only_3m, 4), "iou_s1aef_t_3m_inference": round(r.s1aef_3m, 4),
                         "panel_title_s1only": f"{r.s1only:.2f}", "panel_title_s1aef": f"{r.s1aef:.2f}",
                         "panel_title_opera": f"{r.opera:.2f}"})
        print(v, [s for s, _, _ in items], flush=True)
    pi = pd.DataFrame(rows)
    pi.to_csv(OUT / "panel_iou.csv", index=False)

    md = ["# Figure 1B scene selection (10 m inference)", "",
          "Values: 10 m inference, probabilities resampled bilinearly to the 3 m label grid, threshold 0.30, IoU on the "
          "paper's common valid mask; gain = IoU(S1+AEF(t)) - IoU(S1-only). Source: `fig1B/scene_table.csv` (written by "
          "`scripts/export_53_scene_layers_10m.py`). Continent: Natural Earth 1:50m admin-0 `CONTINENT` at the scene "
          "centre from `S1_Intercomparison_All_Scenes.csv` (point in polygon; SID15, SID46 and SID55 lie in no polygon "
          "and take the nearest one).", "",
          "## Rule", "",
          f"1. Eligible scenes: label water fraction in [{LWF[0]:.2f}, {LWF[1]:.2f}], common-valid fraction >= {CVF:.2f}, "
          "and not one of the current Fig. 1B scenes (SID23, SID46, SID17).",
          f"2. Role C, large AEF gain: gain > {C_GAIN:.2f}; ranked by gain, largest first.",
          f"3. Role B, near the median gain: the median gain over all 53 scenes is {median:+.4f}; ranked by "
          "|gain - median|, smallest first.",
          f"4. Role A, S1-only already good and AEF changes little: S1-only IoU >= {A_S1:.2f} and |gain| <= {A_GAIN:.2f}; "
          "ranked by |gain|, smallest first.",
          "5. A set is filled in the order C, B, A. Each role takes its best-ranked candidate that is not already in the "
          "set and whose continent differs from those already in the set. Ties are broken by SID number.",
          "6. alt1 is the first set. alt2 is built by the same procedure after also excluding the alt1 scenes.",
          "7. Rows are drawn in the order C, B, A. Row labels: SID and country (United States of America shortened to USA).", "",
          "This is the suggested rule of the brief, with two additions needed to make it deterministic: the "
          "|gain| <= 0.05 bound for \"AEF changes little\" (the same band as the WorldCover no-change group) and the "
          "fixed filling order and ranking above.", "",
          f"Eligible scenes: {len(cands['B'])} of 53.", ""]
    for role, extra in (("C", "rank: gain"), ("B", "rank: abs(gain - median)"), ("A", "rank: abs(gain)")):
        md += [f"### Candidates for role {role} ({role_name[role]}), in rank order", ""] + fmt_cands(cands[role], extra) + [""]
    md += ["Scenes excluded by the eligibility filters: " + ", ".join(
        f"{r.sample_id} ({'water fraction %.4f' % r.label_water_fraction if not (LWF[0] <= r.label_water_fraction <= LWF[1]) else ''}"
        f"{'; ' if not (LWF[0] <= r.label_water_fraction <= LWF[1]) and r.common_valid_fraction < CVF else ''}"
        f"{'common-valid %.4f' % r.common_valid_fraction if r.common_valid_fraction < CVF else ''})"
        for r in d.sort_values("sid_num").itertuples()
        if not (LWF[0] <= r.label_water_fraction <= LWF[1]) or r.common_valid_fraction < CVF) + ".", ""]
    md += ["## Chosen sets and panel IoUs", ""]
    for v in versions:
        md += [f"### {v}", "", "| Row | Scene | Role | Country / continent | S1-only | S1+AEF(t) | OPERA | Gain | "
               "3 m inference: S1-only / S1+AEF(t) |", "|---:|---|---|---|---:|---:|---:|---:|---|"]
        for r in pi[pi.version == v].itertuples():
            md.append(f"| {r.row} | {r.sample_id} | {r.role} | {r.country} / {r.continent} | {r.iou_s1only:.4f} | "
                      f"{r.iou_s1aef_t:.4f} | {r.iou_opera:.4f} | {r.gain:+.4f} | {r.iou_s1only_3m_inference:.4f} / "
                      f"{r.iou_s1aef_t_3m_inference:.4f} |")
        beats = pi[(pi.version == v) & pi.s1only_beats_s1aef]
        md += ["", "S1-only beats S1+AEF(t): " + (", ".join(f"{r.sample_id} ({r.iou_s1only:.2f} vs {r.iou_s1aef_t:.2f})"
                                                          for r in beats.itertuples()) or "none") + ".", ""]
    (OUT / "scene_selection_rule.md").write_text("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
