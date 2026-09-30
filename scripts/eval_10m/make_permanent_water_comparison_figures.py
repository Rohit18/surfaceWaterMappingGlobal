#!/usr/bin/env python3
"""Render permanent-water comparison panels for all SID reference labels.

Each figure contains the original Sentinel-1 scene, PlanetScope imagery, the
hand label, the best S1+AEF permanent-water prediction, and OPERA BWTR.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from paths import env_root  # (environment variables; see README.md)


DATA_ROOT = env_root("SWM_S1ML") / "intercomparison_s1aef"
LABEL_DIR = env_root("SWM_GSWD_DIR") / "labels"
PS_DIR = env_root("SWM_GSWD_DIR") / "PS"
S1_ROOT = DATA_ROOT / "s1"
S1_NEAREST_ROOT = DATA_ROOT / "s1_nearest"
PROB_ROOT = DATA_ROOT / "predictions/s1_aef_proj4_seed44_tta_retry_pinmem_exclude_nid001300/probabilities"
OPERA_DIR = env_root("SWM_S1ML") / "intercomparison_opera/binary_masks"
DEFAULT_OUTPUT_DIR = Path("outputs/permanent_water_s1_ps_label_s1aef_opera_figures")

# From the permanent-water subset of the saved S1+AEF evaluation.
DEFAULT_THRESHOLD = 0.30
WATER_COLOR = np.array([0.0, 0.70, 1.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--only-sample", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    return args


def stretch(array: np.ndarray) -> np.ndarray:
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return np.zeros(array.shape, dtype=np.float32)
    lo, hi = np.percentile(valid, (2, 98))
    return np.clip((array - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(np.float32)


def read_on_grid(path: Path, ref: rio.DatasetReader, resampling: Resampling, indexes: int | Sequence[int] = 1) -> Tuple[np.ndarray, float | int | None]:
    with rio.open(path) as src:
        same_grid = src.crs == ref.crs and src.transform == ref.transform and src.width == ref.width and src.height == ref.height
        if same_grid:
            return src.read(indexes).astype(np.float32), src.nodata
        with WarpedVRT(src, crs=ref.crs, transform=ref.transform, width=ref.width, height=ref.height, resampling=resampling) as vrt:
            return vrt.read(indexes).astype(np.float32), src.nodata


def probability_paths() -> Dict[str, Path]:
    return {path.name.removesuffix("_prob.tif"): path for path in PROB_ROOT.glob("*/*_prob.tif")}


def original_s1_path(sample_id: str, probability_path: Path) -> Path:
    nearest = S1_NEAREST_ROOT / sample_id / "{}_nearest_s1.tif".format(sample_id)
    if nearest.exists():
        return nearest
    date = probability_path.parent.name
    candidate = S1_ROOT / sample_id / "{}_s1.tif".format(date)
    if candidate.exists():
        return candidate
    scenes = sorted((S1_ROOT / sample_id).glob("*_s1.tif"))
    if len(scenes) == 1:
        return scenes[0]
    raise FileNotFoundError("No original Sentinel-1 scene found for {}".format(sample_id))


def ps_rgb(path: Path, ref: rio.DatasetReader) -> np.ndarray:
    with rio.open(path) as src:
        indexes = [band for band in (3, 2, 1) if band <= src.count]
    if len(indexes) < 3:
        array, _ = read_on_grid(path, ref, Resampling.bilinear)
        return np.repeat(stretch(array)[..., None], 3, axis=2)
    array, _ = read_on_grid(path, ref, Resampling.bilinear, indexes=indexes)
    return np.moveaxis(np.stack([stretch(band) for band in array]), 0, -1)


def s1_background(path: Path, ref: rio.DatasetReader) -> np.ndarray:
    array, _ = read_on_grid(path, ref, Resampling.bilinear, indexes=[1, 2])
    if array.ndim == 2:
        return stretch(array)
    return 0.55 * stretch(array[0]) + 0.45 * stretch(array[1])


def overlay(background: np.ndarray, water: np.ndarray) -> np.ndarray:
    rgb = np.repeat(background[..., None], 3, axis=2).astype(np.float32)
    rgb[water] = 0.35 * rgb[water] + 0.65 * WATER_COLOR
    return rgb


def iou(pred: np.ndarray, label: np.ndarray, valid: np.ndarray) -> float:
    tp = int((pred & label & valid).sum())
    fp = int((pred & ~label & valid).sum())
    fn = int((~pred & label & valid).sum())
    return tp / max(tp + fp + fn, 1)


def render(sample_id: str, probability_path: Path, output_dir: Path, threshold: float, dpi: int) -> Dict[str, object]:
    label_path = LABEL_DIR / "{}.tif".format(sample_id)
    ps_path = PS_DIR / "{}.tif".format(sample_id)
    opera_path = OPERA_DIR / "{}_opera_bwtr_binary.tif".format(sample_id)
    s1_path = original_s1_path(sample_id, probability_path)
    required = [label_path, ps_path, opera_path, s1_path, probability_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input(s) for {}: {}".format(sample_id, "; ".join(missing)))

    with rio.open(probability_path) as ref:
        probability = ref.read(1).astype(np.float32)
        prob_nodata = ref.nodata
        label, label_nodata = read_on_grid(label_path, ref, Resampling.nearest)
        opera, opera_nodata = read_on_grid(opera_path, ref, Resampling.nearest)
        s1 = s1_background(s1_path, ref)
        ps = ps_rgb(ps_path, ref)

    valid = np.isfinite(probability) & np.isfinite(label) & np.isfinite(opera)
    for array, nodata in ((probability, prob_nodata), (label, label_nodata), (opera, opera_nodata)):
        if nodata is not None:
            valid &= array != nodata
    label_water = label > 0.5
    s1aef_water = probability >= threshold
    opera_water = opera > 0.5
    s1aef_iou = iou(s1aef_water, label_water, valid)
    opera_iou = iou(opera_water, label_water, valid)

    panels = [
        (np.repeat(s1[..., None], 3, axis=2), "Original S1 (VV/VH)"),
        (ps, "PlanetScope"),
        (overlay(s1, label_water & valid), "Hand label"),
        (overlay(s1, s1aef_water & valid), "Best permanent S1+AEF\nT={:.2f} | IoU={:.3f}".format(threshold, s1aef_iou)),
        (overlay(s1, opera_water & valid), "OPERA BWTR\nIoU={:.3f}".format(opera_iou)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(17.5, 4.1), constrained_layout=True)
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image, interpolation="nearest")
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("{} | permanent-water scene".format(sample_id), fontsize=13)
    output_path = output_dir / "{}.png".format(sample_id)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"sample_id": sample_id, "s1_path": str(s1_path), "ps_path": str(ps_path), "label_path": str(label_path), "s1aef_probability_path": str(probability_path), "s1aef_threshold": threshold, "opera_mask_path": str(opera_path), "s1aef_iou": s1aef_iou, "opera_iou": opera_iou, "valid_pixels": int(valid.sum()), "output_path": str(output_path)}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit("Output directory is not empty; use --overwrite: {}".format(output_dir))
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = probability_paths()
    samples = sorted(path.stem for path in LABEL_DIR.glob("SID*.tif"))
    if args.only_sample:
        samples = [sample for sample in samples if sample in set(args.only_sample)]
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    rows = []
    for sample_id in samples:
        if sample_id not in paths:
            raise FileNotFoundError("No S1+AEF probability raster for {}".format(sample_id))
        rows.append(render(sample_id, paths[sample_id], output_dir, args.threshold, args.dpi))
        print("wrote {}".format(rows[-1]["output_path"]))
    with (output_dir / "figure_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.zip:
        archive = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name)
        print("zipped {}".format(archive))
    print("rendered {} permanent-water figures to {}".format(len(rows), output_dir))


if __name__ == "__main__":
    main()
