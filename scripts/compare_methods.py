#!/usr/bin/env python3
"""Compute paired per-scene water-IoU comparisons used in the paper."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def read_iou(path: Path) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["sample_id"]: float(row["water_iou"]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-a", type=Path, required=True)
    parser.add_argument("--method-b", type=Path, required=True)
    parser.add_argument("--name-a", default="method_a")
    parser.add_argument("--name-b", default="method_b")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()

    a, b = read_iou(args.method_a), read_iou(args.method_b)
    if set(a) != set(b):
        raise ValueError("Sample IDs differ between methods")
    sample_ids = sorted(a)
    differences = np.asarray([a[sample] - b[sample] for sample in sample_ids])
    rng = np.random.default_rng(args.seed)
    bootstrap = rng.choice(differences, size=(args.bootstrap_replicates, len(differences)), replace=True).mean(axis=1)
    statistic, p_value = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
    result = {
        "method_a": args.name_a,
        "method_b": args.name_b,
        "scenes": len(sample_ids),
        "a_wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "b_wins": int((differences < 0).sum()),
        "mean_iou_difference": float(differences.mean()),
        "median_iou_difference": float(np.median(differences)),
        "mean_difference_ci95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_two_sided_p": float(p_value),
        "bootstrap_seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
