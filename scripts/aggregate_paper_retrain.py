#!/usr/bin/env python3
"""Aggregate completed supplement retrains into paper replacement tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (42, 43, 44)
WIDTHS = (0, 1, 2, 3, 4, 8, 16)

# OPERA DSWx-S1 comparator. Earlier revisions copied this row forward from the
# previous paper table, and its per-scene mean (0.527) could not be reproduced
# from any surviving artifact while its pooled values matched these files
# exactly. The row is now recomputed here so every row of the main table is
# produced by one code path.
OPERA_ROOT = Path("/pscratch/sd/r/rohit9/S1ML/intercomparison_opera/grouped")
OPERA_SUMMARY = OPERA_ROOT / "sid_permanent_water_summary.json"
OPERA_PER_SAMPLE = OPERA_ROOT / "sid_permanent_water_per_sample_metrics.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluation_paths(report_root: Path, tag: str, prefix: str | None = None) -> tuple[Path, Path]:
    prefix = prefix or tag
    root = report_root / "evaluation" / tag
    return root / (prefix + "_confusion_summary.json"), root / (prefix + "_per_sample_metrics.csv")


def load_evaluation(report_root: Path, tag: str, prefix: str | None = None) -> dict[str, Any]:
    summary_path, per_sample_path = evaluation_paths(report_root, tag, prefix)
    summary = json.loads(summary_path.read_text())
    rows = read_csv(per_sample_path)
    if len(rows) != 53:
        raise RuntimeError("Expected 53 evaluation scenes in {}, found {}".format(per_sample_path, len(rows)))
    thresholds = {round(float(row["threshold"]), 8) for row in rows}
    if len(thresholds) != 1:
        raise RuntimeError("Per-scene metrics contain multiple thresholds: {}".format(thresholds))
    return {"summary": summary, "rows": rows, "summary_path": str(summary_path), "per_sample_path": str(per_sample_path)}


def load_opera_evaluation(summary_path: Path, per_sample_path: Path) -> dict[str, Any]:
    """Load the OPERA comparator into the same shape as load_evaluation().

    The OPERA per-sample file carries no ``threshold`` column (the product ships
    a binary water layer, so there is nothing to threshold) and reports a
    ``status`` column instead. Everything else -- the ``metrics`` block and the
    per-scene ``water_iou`` column -- matches the model evaluations, so the row
    can go through paper_row() unchanged.
    """
    summary = json.loads(summary_path.read_text())
    rows = read_csv(per_sample_path)
    if len(rows) != 53:
        raise RuntimeError("Expected 53 OPERA scenes in {}, found {}".format(per_sample_path, len(rows)))
    bad = [row["sample_id"] for row in rows if row.get("status") not in (None, "", "ok")]
    if bad:
        raise RuntimeError("OPERA scenes without a usable comparison: {}".format(bad))
    return {"summary": summary, "rows": rows, "summary_path": str(summary_path), "per_sample_path": str(per_sample_path)}


def bootstrap_ci(values: list[float], seed: int = 42, replicates: int = 10_000) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(replicates, len(array)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paper_row(method: str, inputs: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluation["summary"]["metrics"]
    scene_ious = [float(row["water_iou"]) for row in evaluation["rows"]]
    low, high = bootstrap_ci(scene_ious)
    return {
        "method": method, "input": inputs, "threshold": 0.30,
        "precision": round(float(metrics["precision"]), 6),
        "recall": round(float(metrics["recall"]), 6),
        "dice": round(float(metrics["dice"]), 6),
        "pooled_water_iou": round(float(metrics["water_iou"]), 6),
        "per_scene_mean_water_iou": round(statistics.fmean(scene_ious), 6),
        "ci95_low": round(low, 6), "ci95_high": round(high, 6),
    }


def run_metric(evaluation: dict[str, Any]) -> tuple[float, float]:
    rows = evaluation["rows"]
    scene_iou = statistics.fmean(float(row["water_iou"]) for row in rows)
    tp = sum(int(row["tp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    return scene_iou, tp / max(tp + fn, 1)


def aggregate_runs(ablation: str, value: int, runs: list[dict[str, Any]]) -> dict[str, Any]:
    ious, recalls = zip(*(run_metric(run) for run in runs))
    return {
        "ablation": ablation, "value": value, "seeds": len(runs), "threshold": 0.5,
        "per_scene_water_iou_mean": round(statistics.fmean(ious), 6),
        "per_scene_water_iou_std": round(statistics.pstdev(ious), 6),
        "recall_mean": round(statistics.fmean(recalls), 6),
        "recall_std": round(statistics.pstdev(recalls), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--opera-summary", type=Path, default=OPERA_SUMMARY)
    parser.add_argument("--opera-per-sample", type=Path, default=OPERA_PER_SAMPLE)
    args = parser.parse_args()
    report_root, output_root = args.report_root.resolve(), args.output_root.resolve()

    s1_tag = "width_k0_seed42_current_tta"
    aef_tag = "width_k16_seed42_current_tta"
    previous_tag = "width_k16_seed42_tminus1_tta"
    main_evaluations = {
        "s1": load_evaluation(report_root, s1_tag + "_paper030", s1_tag + "_paper030"),
        "current": load_evaluation(report_root, aef_tag + "_paper030", aef_tag + "_paper030"),
        "previous": load_evaluation(report_root, previous_tag),
    }
    paper_rows = [
        paper_row("S1 only", "S1", main_evaluations["s1"]),
        paper_row("S1+AEF (t-1)", "S1 + previous-year AEF", main_evaluations["previous"]),
        paper_row("S1+AEF (t)", "S1 + acquisition-year AEF", main_evaluations["current"]),
    ]
    opera_evaluation = load_opera_evaluation(args.opera_summary, args.opera_per_sample)
    paper_rows.append(paper_row("OPERA DSWx-S1", "S1 binary 30 m", opera_evaluation))

    ablation_rows = []
    audit_runs: dict[str, list[str]] = {}
    for width in WIDTHS:
        runs = []
        for seed in SEEDS:
            tag = "width_k{}_seed{}_current_tta".format(width, seed)
            evaluation = load_evaluation(report_root, tag)
            runs.append(evaluation)
        ablation_rows.append(aggregate_runs("aef_width", width, runs))
        audit_runs["width_{}".format(width)] = [run["summary_path"] for run in runs]
    for size in (1000, 2000, 3000):
        runs = []
        for seed in SEEDS:
            tag = "train{}_k16_seed{}_current_tta".format(size, seed)
            runs.append(load_evaluation(report_root, tag))
        ablation_rows.append(aggregate_runs("training_tiles", size, runs))
        audit_runs["training_{}".format(size)] = [run["summary_path"] for run in runs]
    full_runs = [load_evaluation(report_root, "width_k16_seed{}_current_tta".format(seed)) for seed in SEEDS]
    ablation_rows.append(aggregate_runs("training_tiles", 4222, full_runs))

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "paper_metrics.csv", paper_rows)
    write_csv(output_root / "ablation_metrics.csv", ablation_rows)
    audit = {
        "dataset": {"total": 5278, "train": 4222, "valid": 1056, "new": 600},
        "thresholds": {"main_table": 0.30, "ablations": 0.50},
        "bootstrap": {"replicates": 10000, "seed": 42, "unit": "evaluation scene"},
        "main_evaluation_sources": {key: value["summary_path"] for key, value in main_evaluations.items()},
        "ablation_evaluation_sources": audit_runs,
        "opera_comparator": {
            "status": "recomputed from source; no longer copied from the previous paper table",
            "summary_path": opera_evaluation["summary_path"],
            "per_sample_path": opera_evaluation["per_sample_path"],
            "note": (
                "OPERA DSWx-S1 ships a binary B01_WTR water layer, so the 0.30 threshold "
                "column does not apply to this row. Valid-pixel coverage also differs from "
                "the model rows because each product carries its own no-data mask; see "
                "opera_valid_pixels and the matched-mask check in completion_results.md."
            ),
            "opera_valid_pixels": int(opera_evaluation["summary"]["counts"]["valid_pixels"]),
        },
    }
    (output_root / "aggregation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = [
        "# Paper values after open-water supplementation", "",
        "Dataset: 5,278 tiles (4,222 train / 1,056 validation), including 600 new open-water samples.", "",
        "## Main independent evaluation (53 scenes, threshold 0.30)", "",
        "| Method | Precision | Recall | Dice | Pooled water IoU | Mean scene IoU | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paper_rows:
        lines.append(
            "| {method} | {precision:.3f} | {recall:.3f} | {dice:.3f} | {pooled_water_iou:.3f} | "
            "{per_scene_mean_water_iou:.3f} | {ci95_low:.3f}–{ci95_high:.3f} |".format(
                **{key: float(value) if key not in {"method", "input"} else value for key, value in row.items()}
            )
        )
    lines.extend(["", "See `paper_metrics.csv`, `ablation_metrics.csv`, and `aggregation_audit.json` for exact values and provenance."])
    (output_root / "paper_number_replacements.md").write_text("\n".join(lines) + "\n")
    print((output_root / "paper_number_replacements.md").read_text())


if __name__ == "__main__":
    main()
