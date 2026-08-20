#!/usr/bin/env python3
"""Evaluate probability GeoTIFFs against binary reference masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


def counts(label: np.ndarray, probability: np.ndarray, valid: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    predicted = probability >= threshold
    water = label > 0.5
    return (
        int((predicted & water & valid).sum()),
        int((predicted & ~water & valid).sum()),
        int((~predicted & ~water & valid).sum()),
        int((~predicted & water & valid).sum()),
    )


def metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "dice": (2 * tp) / max(2 * tp + fp + fn, 1),
        "water_iou": tp / max(tp + fp + fn, 1),
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
    }


def label_on_grid(path: Path, probability_ds: rio.DatasetReader) -> np.ndarray:
    with rio.open(path) as label_ds:
        if (
            label_ds.crs == probability_ds.crs
            and label_ds.transform == probability_ds.transform
            and label_ds.width == probability_ds.width
            and label_ds.height == probability_ds.height
        ):
            return label_ds.read(1)
        with WarpedVRT(
            label_ds,
            crs=probability_ds.crs,
            transform=probability_ds.transform,
            width=probability_ds.width,
            height=probability_ds.height,
            resampling=Resampling.nearest,
        ) as vrt:
            return vrt.read(1)


def bootstrap_mean(values: np.ndarray, seed: int, replicates: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with sample_id,label_path,prob_path")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.manifest.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    required = {"sample_id", "label_path", "prob_path"}
    missing = required - set(source_rows[0] if source_rows else {})
    if missing:
        raise ValueError("Manifest missing columns: {}".format(sorted(missing)))

    per_scene = []
    pooled = np.zeros(4, dtype=np.int64)
    for row in source_rows:
        probability_path = Path(row["prob_path"]).expanduser()
        with rio.open(probability_path) as probability_ds:
            probability = probability_ds.read(1).astype(np.float32)
            label = label_on_grid(Path(row["label_path"]).expanduser(), probability_ds)
            valid = np.isfinite(probability)
            if probability_ds.nodata is not None:
                valid &= probability != probability_ds.nodata
        scene_counts = counts(label, probability, valid, args.threshold)
        pooled += np.asarray(scene_counts)
        per_scene.append({"sample_id": row["sample_id"], "threshold": args.threshold, **metrics(*scene_counts)})

    values = np.asarray([row["water_iou"] for row in per_scene], dtype=np.float64)
    ci_low, ci_high = bootstrap_mean(values, args.seed, args.bootstrap_replicates)
    summary = {
        "threshold": args.threshold,
        "scenes": len(per_scene),
        "pooled": metrics(*pooled.tolist()),
        "per_scene_mean_water_iou": float(values.mean()),
        "per_scene_water_iou_ci95": [ci_low, ci_high],
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_scene_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_scene[0].keys())
        writer.writeheader()
        writer.writerows(per_scene)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
