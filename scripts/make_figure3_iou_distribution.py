#!/usr/bin/env python
"""Regenerate Figure 3: per-scene water IoU distribution over the evaluation scenes.

The original figure script was not kept with the manuscript package, so this
rebuilds it from the per-scene metrics emitted by the evaluation runs. Layout
follows the published caption: a smoothed distribution per method, one point per
scene, a diamond at the mean, and a bootstrap 95% CI bar.

Usage:
    python scripts/make_figure3_iou_distribution.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

RUN_SETS = {
    # Label-class reruns (paper_labelclass_v1): the models the manuscript now reports.
    "labelclass_v1": (
        "/pscratch/sd/r/rohit9/S1ML/training_runs/paper_labelclass_v1/openwater/reports/58321212/evaluation",
        "/pscratch/sd/r/rohit9/S1ML/training_runs/paper_labelclass_v1/tminus1/reports/58321217/evaluation",
    ),
    # Original runs trained on mixed binary50 + label-class labels (figure of 2026-09-09).
    "mixed_v1": (
        "/global/u2/r/rohit9/Workspace/reports/paper_retrain_openwater_v1/57725872/evaluation",
        "/pscratch/sd/r/rohit9/S1ML/training_runs/paper_tminus1_v1/reports/57952761/evaluation",
    ),
}
DEFAULT_RUN_SET = "labelclass_v1"
OPENWATER, TMINUS1 = (Path(p) for p in RUN_SETS[DEFAULT_RUN_SET])
OPERA = Path(
    "/pscratch/sd/r/rohit9/S1ML/intercomparison_opera/grouped/"
    "sid_permanent_water_per_sample_metrics.csv"
)

THRESHOLD = 0.30
BOOTSTRAP = 10_000
SEED = 42

# Ordered bottom-to-top, weakest first, so the AEF configurations read as a group.
def build_methods(openwater: Path, tminus1: Path) -> list:
    return [
        ("S1 only", openwater, "width_k0_seed42_current_tta_paper030", "#8d99a6"),
        ("OPERA DSWx-S1", openwater, None, "#c08a5a"),
        ("S1 + AEF ($t-1$), swap", openwater, "width_k16_seed42_tminus1_tta", "#6fa8bd"),
        ("S1 + AEF ($t-1$), trained", tminus1, "width_k16_seed42_tminus1_tta", "#3d7f9c"),
        ("S1 + AEF ($t$)", openwater, "width_k16_seed42_current_tta_paper030", "#14556e"),
    ]


METHODS = build_methods(OPENWATER, TMINUS1)


def load_scene_iou(base: Path, tag: str | None) -> pd.Series:
    if tag is None:
        frame = pd.read_csv(OPERA)
        return frame.set_index("sample_id")["water_iou"]
    frame = pd.read_csv(base / tag / f"{tag}_per_sample_metrics.csv")
    frame = frame[np.isclose(frame["threshold"], THRESHOLD)]
    return frame.set_index("sample_id")["water_iou"]


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    """Match aggregate_paper_retrain.bootstrap_ci exactly.

    Same RNG construction, same integer-index resampling and same quantile call,
    so the intervals drawn in this figure are the intervals printed in Table I.
    Uses its own generator seeded per call rather than the shared jitter RNG.
    """
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    means = array[rng.integers(0, len(array), size=(BOOTSTRAP, len(array)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stem", default="figure3_iou_distribution")
    parser.add_argument("--run-set", choices=sorted(RUN_SETS), default=DEFAULT_RUN_SET)
    args = parser.parse_args()
    methods = build_methods(*(Path(p) for p in RUN_SETS[args.run_set]))
    print(f"run set: {args.run_set}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    grid = np.linspace(0.0, 1.0, 512)

    fig, ax = plt.subplots(figsize=(3.5, 4.0), dpi=400)
    row_height = 1.30
    summary = []

    for row, (label, base, tag, color) in enumerate(methods):
        iou = load_scene_iou(base, tag).dropna()
        values = iou.to_numpy(dtype=float)
        base_y = row * row_height

        # Smoothed distribution, clipped to the unit interval by reflection so the
        # density does not leak past IoU 0 or 1.
        kde = gaussian_kde(values, bw_method=0.35)
        density = kde(grid) + kde(-grid) + kde(2.0 - grid)
        density = density / density.max() * 0.62

        ax.fill_between(grid, base_y, base_y + density, color=color, alpha=0.30, linewidth=0)
        ax.plot(grid, base_y + density, color=color, linewidth=1.0)

        jitter = rng.uniform(0.06, 0.20, size=values.size)
        ax.scatter(
            values, base_y + jitter, s=3.2, color=color, alpha=0.55,
            linewidths=0, zorder=3,
        )

        mean = float(values.mean())
        low, high = bootstrap_ci(values)
        # CI bar and mean sit on their own baseline below the density, with the
        # numeric label set to the right of the bar so it cannot collide with the
        # next row's curve.
        bar_y = base_y - 0.17
        ax.plot([low, high], [bar_y] * 2, color=color, linewidth=1.5,
                solid_capstyle="butt", zorder=4)
        ax.plot(mean, bar_y, marker="D", markersize=4.0, color=color,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        ax.text(high + 0.018, bar_y, f"{mean:.3f}", ha="left", va="center",
                fontsize=6.2, color=color, zorder=5)

        summary.append({
            "method": label.replace("$", ""), "scenes": int(values.size),
            "per_scene_mean_water_iou": round(mean, 6),
            "ci95_low": round(low, 6), "ci95_high": round(high, 6),
        })

    ax.set_yticks([r * row_height for r in range(len(methods))])
    ax.set_yticklabels([m[0] for m in methods], fontsize=7)
    ax.set_ylim(-0.45, (len(methods) - 1) * row_height + 0.85)
    ax.set_xlim(0.0, 1.10)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.spines["bottom"].set_bounds(0.0, 1.0)
    ax.set_xlabel("Water IoU (per scene)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(axis="x", color="#d8dee3", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"{args.stem}.{ext}", bbox_inches="tight")
    plt.close(fig)

    (args.out_dir / f"{args.stem}_values.json").write_text(
        json.dumps({"threshold": THRESHOLD, "bootstrap_replicates": BOOTSTRAP,
                    "seed": SEED, "methods": summary}, indent=2) + "\n"
    )
    for entry in summary:
        print(f"{entry['method']:<28} n={entry['scenes']:>3}  "
              f"mean={entry['per_scene_mean_water_iou']:.4f}  "
              f"CI95=[{entry['ci95_low']:.4f}, {entry['ci95_high']:.4f}]")
    print(f"\nwrote {args.out_dir}/{args.stem}.pdf / .png / _values.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
