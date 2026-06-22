#!/usr/bin/env python
"""Probability-only inference for S1-only and S1+AEF flood models."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import torch
from torch.utils.data import Dataset


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data" / "inference"
DEFAULT_SCENES_ROOT = DEFAULT_DATA_ROOT / "s1"
DEFAULT_AEF_PATH = DEFAULT_DATA_ROOT / "alphaearth_2025.tif"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs"
DEFAULT_TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"
DEFAULT_S1_RUN_DIR = REPOSITORY_ROOT / "models" / "s1_resnet34"
DEFAULT_S1AEF_RUN_DIR = REPOSITORY_ROOT / "models" / "s1aef_resnet34"
PROB_NODATA = -9999.0
TTA_OPS: Tuple[Tuple[int, ...], ...] = (
    (),
    (-1,),
    (-2,),
    (-2, -1),
)


@dataclass(frozen=True)
class SceneRecord:
    date: str
    year: int
    s1_path: Path
    prob_path: Path


class DummySegmentationDataset(Dataset):
    """Single-item dataset used only to instantiate the training-time model."""

    def __init__(self, channels: int, crop_size: int):
        self.image = torch.zeros((channels, crop_size, crop_size), dtype=torch.float32)
        self.mask = torch.zeros((crop_size, crop_size), dtype=torch.int64)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.image, self.mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-kind", choices=("s1", "s1aef"), required=True)
    parser.add_argument("--scenes-root", type=Path, default=DEFAULT_SCENES_ROOT)
    parser.add_argument("--aef-path", type=Path, default=DEFAULT_AEF_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--region-name", default="midwest")
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--tta", dest="tta", action="store_true", default=True)
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--n-proj-bands", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    args.scenes_root = args.scenes_root.expanduser().resolve()
    args.aef_path = args.aef_path.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.train_script = args.train_script.expanduser().resolve()
    if args.run_dir is None:
        args.run_dir = DEFAULT_S1_RUN_DIR if args.model_kind == "s1" else DEFAULT_S1AEF_RUN_DIR
    args.run_dir = args.run_dir.expanduser().resolve()
    if args.weights_path is not None:
        args.weights_path = args.weights_path.expanduser().resolve()

    if args.tile < 32:
        parser.error("--tile must be >= 32")
    if args.overlap < 0 or args.overlap >= args.tile:
        parser.error("--overlap must be >= 0 and < --tile")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.crop_size < 32:
        parser.error("--crop-size must be >= 32")
    if args.max_scenes is not None and args.max_scenes < 1:
        parser.error("--max-scenes must be >= 1")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_training_module(train_script: Path):
    if not train_script.exists():
        raise FileNotFoundError("Training script not found: {}".format(train_script))
    spec = importlib.util.spec_from_file_location("train_s1aef_resnet34_bottleneck", train_script)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load module spec from {}".format(train_script))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_device(device_name: str, device_id: int) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda:{}".format(device_id))
    if torch.cuda.is_available():
        return torch.device("cuda:{}".format(device_id))
    return torch.device("cpu")


def eval_summary_path(run_dir: Path, model_kind: str) -> Path:
    name = "s1dw_resnet34_eval_summary.json" if model_kind == "s1" else "s1aef_bottleneck_resnet34_eval_summary.json"
    return run_dir / name


def load_eval_summary(run_dir: Path, model_kind: str) -> dict:
    path = eval_summary_path(run_dir, model_kind)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def infer_loss(summary: dict) -> str:
    return str(summary.get("loss") or "ce_dice")


def infer_n_proj_bands(run_dir: Path, weights_path: Path, explicit: Optional[int]) -> int:
    if explicit is not None:
        return int(explicit)

    summary_path = eval_summary_path(run_dir, "s1aef")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        shape = str(summary.get("bottleneck_proj_shape", ""))
        match = re.search(r"64->(\d+)", shape)
        if match:
            return int(match.group(1))

    if weights_path.exists():
        state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if isinstance(state, dict):
            for key, value in state.items():
                if key.endswith("aef_proj.weight"):
                    return int(value.shape[0])
    return 16


def resolve_weights_path(run_dir: Path, model_kind: str, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    if model_kind == "s1":
        return run_dir / "models" / "s1dw_resnet34_best.pth"
    return run_dir / "models" / "s1aef_bottleneck_resnet34_best.pth"


def default_output_tag(model_kind: str, use_tta: bool) -> str:
    if model_kind == "s1":
        return "s1_only_tta" if use_tta else "s1_only_plain"
    return "s1_aef_tta" if use_tta else "s1_aef_plain"


def load_band_stats(run_dir: Path, expected_bands: int) -> Tuple[np.ndarray, np.ndarray]:
    stats_path = run_dir / "band_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError("Missing band_stats.npz at {}".format(stats_path))
    stats = np.load(stats_path)
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)
    if mean.shape != (expected_bands,) or std.shape != (expected_bands,):
        raise ValueError(
            "Expected {n}-band stats, got mean={m}, std={s}".format(
                n=expected_bands, m=mean.shape, s=std.shape
            )
        )
    std = np.where(std <= 0, 1.0, std).astype(np.float32)
    return mean, std


def discover_scenes(
    scenes_root: Path,
    output_dir: Path,
    years: Optional[Sequence[int]],
    max_scenes: Optional[int],
) -> List[SceneRecord]:
    if not scenes_root.exists():
        raise FileNotFoundError("Scenes root does not exist: {}".format(scenes_root))
    year_filter = None if years is None else {int(year) for year in years}
    records: List[SceneRecord] = []
    for path in sorted(scenes_root.rglob("s1_*.tif")):
        try:
            date_obj = datetime.strptime(path.stem.replace("s1_", ""), "%Y-%m-%d")
        except ValueError:
            continue
        year = int(date_obj.year)
        if year_filter is not None and year not in year_filter:
            continue
        records.append(
            SceneRecord(
                date=date_obj.date().isoformat(),
                year=year,
                s1_path=path,
                prob_path=output_dir / "probabilities" / str(year) / "{}_prob.tif".format(path.stem),
            )
        )
    if max_scenes is not None:
        records = records[:max_scenes]
    if not records:
        raise FileNotFoundError("No Sentinel-1 scenes matched the requested filters.")
    return records


def validate_inputs(
    records: Sequence[SceneRecord],
    model_kind: str,
    aef_path: Path,
    run_dir: Path,
    weights_path: Path,
    expected_bands: int,
) -> dict:
    if not run_dir.exists():
        raise FileNotFoundError("Run dir does not exist: {}".format(run_dir))
    if not weights_path.exists():
        raise FileNotFoundError("Weights do not exist: {}".format(weights_path))
    load_band_stats(run_dir, expected_bands=expected_bands)

    reference = None
    for record in records:
        with rio.open(str(record.s1_path)) as src:
            if src.count != 3:
                raise ValueError("Expected 3 S1 bands in {}, found {}".format(record.s1_path, src.count))
            current = (src.crs.to_string() if src.crs else None, src.transform, src.width, src.height)
            if reference is None:
                reference = current
            elif current != reference:
                raise ValueError("S1 grid mismatch for {}".format(record.s1_path))

    aef_summary = None
    if model_kind == "s1aef":
        if not aef_path.exists():
            raise FileNotFoundError("AEF path does not exist: {}".format(aef_path))
        with rio.open(str(aef_path)) as src:
            if src.count != 64:
                raise ValueError("Expected 64 AEF bands in {}, found {}".format(aef_path, src.count))
            aef_summary = {
                "path": str(aef_path),
                "bands": src.count,
                "crs": src.crs.to_string() if src.crs else None,
                "width": src.width,
                "height": src.height,
                "nodata": src.nodata,
            }

    return {
        "scene_count": len(records),
        "reference_grid": None
        if reference is None
        else {
            "crs": reference[0],
            "transform": list(reference[1])[:6],
            "width": reference[2],
            "height": reference[3],
        },
        "aef": aef_summary,
    }


def build_dummy_dls(train_mod, channels: int, crop_size: int, artifact_dir: Path, device: torch.device):
    dataset = DummySegmentationDataset(channels=channels, crop_size=crop_size)
    train_dl = train_mod.DataLoader(dataset, bs=1, shuffle=False, num_workers=0, device=None)
    valid_dl = train_mod.DataLoader(dataset, bs=1, shuffle=False, num_workers=0, device=None)
    dls = train_mod.DataLoaders(train_dl, valid_dl, path=artifact_dir, device=device)
    dls.c = 2
    dls.vocab = ["other", "water"]
    return dls


def build_model_args(
    model_kind: str,
    run_dir: Path,
    loss: str,
    n_proj_bands: int,
) -> SimpleNamespace:
    n_aef_bands = 0 if model_kind == "s1" else 64
    best_name = "s1dw_resnet34_best" if model_kind == "s1" else "s1aef_bottleneck_resnet34_best"
    return SimpleNamespace(
        n_s1_bands=3,
        n_aef_bands=n_aef_bands,
        n_proj_bands=0 if model_kind == "s1" else n_proj_bands,
        loss=loss,
        model_family="resnet34_unet",
        segformer_variant="segformer_b4",
        codes=["other", "water"],
        hf_cache_dir=run_dir,
        pretrained=False,
        hf_local_files_only=True,
        use_fp16=False,
        selection_metric="water_iou",
        tuning_strategy="inference",
        grad_accum_steps=1,
        artifact_dir=run_dir,
        best_model_name=best_name,
    )


def load_model(
    train_mod,
    model_kind: str,
    run_dir: Path,
    weights_path: Path,
    device: torch.device,
    crop_size: int,
    loss: str,
    n_proj_bands: int,
) -> torch.nn.Module:
    channels = 3 if model_kind == "s1" else 67
    args = build_model_args(model_kind=model_kind, run_dir=run_dir, loss=loss, n_proj_bands=n_proj_bands)
    dls = build_dummy_dls(train_mod=train_mod, channels=channels, crop_size=crop_size, artifact_dir=run_dir, device=device)
    learn = train_mod.build_learner(args, dls)
    state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    learn.model.load_state_dict(state, strict=True)
    model = learn.model.to(device)
    model.eval()
    return model


def plan_tiles(height: int, width: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    step = max(1, tile_size - overlap)

    def starts(length: int) -> List[int]:
        if length <= tile_size:
            return [0]
        out = list(range(0, length - tile_size, step))
        if out[-1] + tile_size < length:
            out.append(length - tile_size)
        return out

    windows: List[Tuple[int, int, int, int]] = []
    for top in starts(height):
        tile_h = min(tile_size, height - top)
        for left in starts(width):
            tile_w = min(tile_size, width - left)
            windows.append((top, left, tile_h, tile_w))
    return windows


def read_window(ds: rio.DatasetReader, top: int, left: int, height: int, width: int) -> np.ndarray:
    return ds.read(window=((top, top + height), (left, left + width))).astype(np.float32)


def read_window_mask(ds: rio.DatasetReader, top: int, left: int, height: int, width: int) -> np.ndarray:
    return ds.dataset_mask(window=((top, top + height), (left, left + width))) == 0


def forward_logits(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    fp16: bool,
    use_tta: bool,
) -> torch.Tensor:
    batch = batch.to(device, non_blocking=True)
    with torch.inference_mode():
        if not use_tta:
            if fp16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return model(batch).float().cpu()
            return model(batch).float().cpu()

        logits_sum = None
        for dims in TTA_OPS:
            augmented = torch.flip(batch, dims=list(dims)) if dims else batch
            if fp16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(augmented).float()
            else:
                logits = model(augmented).float()
            restored = torch.flip(logits, dims=list(dims)) if dims else logits
            logits_sum = restored if logits_sum is None else logits_sum + restored
        return (logits_sum / len(TTA_OPS)).cpu()


def softmax_water_probability(mean_logits: np.ndarray) -> np.ndarray:
    shifted = mean_logits - mean_logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted).astype(np.float32, copy=False)
    return exp[1] / exp.sum(axis=0)


def write_probability(prob: np.ndarray, src_profile: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = src_profile.copy()
    profile.update(
        count=1,
        dtype=rio.float32,
        nodata=PROB_NODATA,
        compress="lzw",
        BIGTIFF="IF_SAFER",
    )
    with rio.open(str(out_path), "w", **profile) as dst:
        dst.write(prob.astype(np.float32), 1)
        dst.set_band_description(1, "P(water)")


def summarize_scene(
    record: SceneRecord,
    valid_mask: np.ndarray,
    water_prob: np.ndarray,
    elapsed_seconds: float,
    status: str,
) -> dict:
    valid_pixels = int(valid_mask.sum())
    total_pixels = int(valid_mask.size)
    nodata_fraction = 1.0 - (valid_pixels / max(total_pixels, 1))
    row = {
        "date": record.date,
        "year": record.year,
        "scene": record.s1_path.stem,
        "s1_path": str(record.s1_path),
        "prob_path": str(record.prob_path),
        "status": status,
        "seconds": round(float(elapsed_seconds), 3),
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "nodata_fraction": round(float(nodata_fraction), 6),
    }
    if valid_pixels == 0:
        row.update({"mean_probability": 0.0, "p95_probability": 0.0, "max_probability": 0.0})
        return row
    valid_probs = water_prob[valid_mask]
    row.update(
        {
            "mean_probability": round(float(valid_probs.mean()), 6),
            "p95_probability": round(float(np.percentile(valid_probs, 95)), 6),
            "max_probability": round(float(valid_probs.max()), 6),
        }
    )
    return row


def summarize_scene_failure(record: SceneRecord, elapsed_seconds: float, error: str) -> dict:
    return {
        "date": record.date,
        "year": record.year,
        "scene": record.s1_path.stem,
        "s1_path": str(record.s1_path),
        "prob_path": str(record.prob_path),
        "status": "error",
        "seconds": round(float(elapsed_seconds), 3),
        "total_pixels": "",
        "valid_pixels": "",
        "nodata_fraction": "",
        "mean_probability": "",
        "p95_probability": "",
        "max_probability": "",
        "error": error,
    }


def infer_scene(
    record: SceneRecord,
    model_kind: str,
    aef_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    band_mean: np.ndarray,
    band_std: np.ndarray,
) -> dict:
    scene_start = time.perf_counter()
    with rio.open(str(record.s1_path)) as s1_ds:
        if s1_ds.count != 3:
            raise ValueError("Expected 3-band S1 input, found {} in {}".format(s1_ds.count, record.s1_path))

        aef_ds = rio.open(str(aef_path)) if model_kind == "s1aef" else None
        try:
            aef_vrt = None
            if aef_ds is not None:
                if aef_ds.count != 64:
                    raise ValueError("Expected 64-band AEF input, found {} in {}".format(aef_ds.count, aef_path))
                aef_vrt = WarpedVRT(
                    aef_ds,
                    crs=s1_ds.crs,
                    transform=s1_ds.transform,
                    width=s1_ds.width,
                    height=s1_ds.height,
                    resampling=Resampling.nearest,
                    nodata=aef_ds.nodata,
                )

            height, width = s1_ds.height, s1_ds.width
            src_profile = s1_ds.profile.copy()
            mean_3d = band_mean.reshape(-1, 1, 1)
            std_3d = band_std.reshape(-1, 1, 1)
            s1_fill = band_mean[:3].reshape(3, 1, 1)
            aef_fill = band_mean[3:].reshape(64, 1, 1) if model_kind == "s1aef" else None
            aef_nodata = None if aef_vrt is None else aef_vrt.nodata

            logit_sum = np.zeros((2, height, width), dtype=np.float32)
            weight = np.zeros((height, width), dtype=np.float32)
            invalid_mask = np.zeros((height, width), dtype=bool)
            windows = plan_tiles(height=height, width=width, tile_size=args.tile, overlap=args.overlap)
            batch_tensors: List[torch.Tensor] = []
            batch_windows: List[Tuple[int, int, int, int]] = []

            def flush() -> None:
                if not batch_tensors:
                    return
                batch = torch.stack(batch_tensors, dim=0)
                logits = forward_logits(
                    model=model,
                    batch=batch,
                    device=device,
                    fp16=args.fp16,
                    use_tta=args.tta,
                ).numpy()
                for idx, (top, left, tile_h, tile_w) in enumerate(batch_windows):
                    logit_sum[:, top : top + tile_h, left : left + tile_w] += logits[idx, :, :tile_h, :tile_w]
                    weight[top : top + tile_h, left : left + tile_w] += 1.0
                batch_tensors.clear()
                batch_windows.clear()

            for top, left, tile_h, tile_w in windows:
                s1_tile = read_window(s1_ds, top=top, left=left, height=tile_h, width=tile_w)
                s1_invalid = read_window_mask(s1_ds, top=top, left=left, height=tile_h, width=tile_w)
                s1_invalid |= ~np.isfinite(s1_tile).all(axis=0)

                if model_kind == "s1aef":
                    assert aef_vrt is not None
                    assert aef_fill is not None
                    aef_tile = read_window(aef_vrt, top=top, left=left, height=tile_h, width=tile_w)
                    aef_invalid = read_window_mask(aef_vrt, top=top, left=left, height=tile_h, width=tile_w)
                    aef_invalid |= ~np.isfinite(aef_tile).all(axis=0)
                    if aef_nodata is not None:
                        aef_invalid |= np.any(aef_tile == aef_nodata, axis=0)
                    tile_invalid = s1_invalid | aef_invalid
                else:
                    aef_tile = None
                    tile_invalid = s1_invalid

                invalid_mask[top : top + tile_h, left : left + tile_w] |= tile_invalid
                s1_tile = np.nan_to_num(s1_tile, nan=0.0, posinf=0.0, neginf=0.0)
                if tile_invalid.any():
                    s1_tile = np.where(tile_invalid[None, :, :], s1_fill, s1_tile)

                if model_kind == "s1aef":
                    assert aef_tile is not None
                    assert aef_fill is not None
                    aef_tile = np.nan_to_num(aef_tile, nan=0.0, posinf=0.0, neginf=0.0)
                    if tile_invalid.any():
                        aef_tile = np.where(tile_invalid[None, :, :], aef_fill, aef_tile)
                    stacked = np.concatenate([s1_tile, aef_tile], axis=0)
                else:
                    stacked = s1_tile

                stacked = (stacked - mean_3d) / std_3d
                if tile_h != args.tile or tile_w != args.tile:
                    padded = np.zeros((stacked.shape[0], args.tile, args.tile), dtype=np.float32)
                    padded[:, :tile_h, :tile_w] = stacked
                    stacked = padded

                batch_tensors.append(torch.from_numpy(stacked.astype(np.float32, copy=False)))
                batch_windows.append((top, left, tile_h, tile_w))
                if len(batch_tensors) >= args.batch_size:
                    flush()
            flush()
        finally:
            if aef_vrt is not None:
                aef_vrt.close()
            if aef_ds is not None:
                aef_ds.close()

    weight = np.maximum(weight, 1e-6)
    mean_logits = logit_sum / weight[None, :, :]
    water_prob = softmax_water_probability(mean_logits)
    valid_mask = ~invalid_mask
    water_prob[invalid_mask] = PROB_NODATA
    write_probability(prob=water_prob, src_profile=src_profile, out_path=record.prob_path)
    return summarize_scene(
        record=record,
        valid_mask=valid_mask,
        water_prob=water_prob,
        elapsed_seconds=time.perf_counter() - scene_start,
        status="ok",
    )


def write_scene_manifest(manifest_path: Path, records: Sequence[SceneRecord], model_kind: str, aef_path: Optional[Path]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["date", "year", "model_kind", "s1_path", "aef_path", "prob_path"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "date": record.date,
                    "year": record.year,
                    "model_kind": model_kind,
                    "s1_path": str(record.s1_path),
                    "aef_path": "" if aef_path is None else str(aef_path),
                    "prob_path": str(record.prob_path),
                }
            )


def write_scene_summary(summary_path: Path, rows: Sequence[dict]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "year",
        "scene",
        "s1_path",
        "prob_path",
        "status",
        "seconds",
        "total_pixels",
        "valid_pixels",
        "nodata_fraction",
        "mean_probability",
        "p95_probability",
        "max_probability",
        "error",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    args = parse_args()
    weights_path = resolve_weights_path(args.run_dir, args.model_kind, args.weights_path)
    summary = load_eval_summary(args.run_dir, args.model_kind)
    loss = infer_loss(summary)
    n_proj_bands = 0 if args.model_kind == "s1" else infer_n_proj_bands(args.run_dir, weights_path, args.n_proj_bands)
    expected_bands = 3 if args.model_kind == "s1" else 67
    if args.output_tag is None:
        args.output_tag = default_output_tag(args.model_kind, args.tta)
    output_dir = args.output_root / args.output_tag

    records = discover_scenes(
        scenes_root=args.scenes_root,
        output_dir=output_dir,
        years=args.years,
        max_scenes=args.max_scenes,
    )
    validation = validate_inputs(
        records=records,
        model_kind=args.model_kind,
        aef_path=args.aef_path,
        run_dir=args.run_dir,
        weights_path=weights_path,
        expected_bands=expected_bands,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_scene_manifest(
        manifest_path=output_dir / "scene_manifest.csv",
        records=records,
        model_kind=args.model_kind,
        aef_path=args.aef_path if args.model_kind == "s1aef" else None,
    )

    device = resolve_device(device_name=args.device, device_id=args.device_id)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    band_mean, band_std = load_band_stats(args.run_dir, expected_bands=expected_bands)
    year_counts = Counter(record.year for record in records)
    run_config = {
        "created_at": utc_now(),
        "region_name": args.region_name,
        "model_kind": args.model_kind,
        "scenes_root": str(args.scenes_root),
        "aef_path": str(args.aef_path) if args.model_kind == "s1aef" else None,
        "run_dir": str(args.run_dir),
        "weights_path": str(weights_path),
        "train_script": str(args.train_script),
        "output_dir": str(output_dir),
        "device": str(device),
        "tile": args.tile,
        "overlap": args.overlap,
        "batch_size": args.batch_size,
        "tta": bool(args.tta),
        "fp16": bool(args.fp16 and device.type == "cuda"),
        "probability_nodata": PROB_NODATA,
        "loss": loss,
        "n_proj_bands": n_proj_bands,
        "scene_count": len(records),
        "year_counts": {str(year): count for year, count in sorted(year_counts.items())},
        "validation": validation,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(json.dumps(run_config, indent=2))
    if args.validate_only:
        print(json.dumps({"status": "validate-only-ok", "output_dir": str(output_dir)}))
        return

    train_mod = load_training_module(args.train_script)
    model = load_model(
        train_mod=train_mod,
        model_kind=args.model_kind,
        run_dir=args.run_dir,
        weights_path=weights_path,
        device=device,
        crop_size=args.crop_size,
        loss=loss,
        n_proj_bands=n_proj_bands,
    )
    print(json.dumps({"loaded_weights": str(weights_path), "model_kind": args.model_kind}))

    summary_rows: List[dict] = []
    summary_path = output_dir / "scene_summary.csv"
    for index, record in enumerate(records, start=1):
        if args.skip_existing and record.prob_path.exists():
            row = {
                "date": record.date,
                "year": record.year,
                "scene": record.s1_path.stem,
                "s1_path": str(record.s1_path),
                "prob_path": str(record.prob_path),
                "status": "skipped-existing",
                "seconds": 0.0,
            }
            summary_rows.append(row)
            write_scene_summary(summary_path=summary_path, rows=summary_rows)
            print(json.dumps({"scene_index": index, "scene": record.s1_path.name, "status": "skipped-existing"}))
            continue

        scene_start = time.perf_counter()
        try:
            row = infer_scene(
                record=record,
                model_kind=args.model_kind,
                aef_path=args.aef_path,
                model=model,
                device=device,
                args=args,
                band_mean=band_mean,
                band_std=band_std,
            )
            row["error"] = ""
        except Exception as exc:
            error_text = "{}: {}".format(type(exc).__name__, exc)
            row = summarize_scene_failure(
                record=record,
                elapsed_seconds=time.perf_counter() - scene_start,
                error=error_text,
            )
            print(json.dumps({"scene_index": index, "scene": record.s1_path.name, "status": "error", "error": error_text}))
            traceback.print_exc()
        summary_rows.append(row)
        write_scene_summary(summary_path=summary_path, rows=summary_rows)
        print(json.dumps({"scene_index": index, **row}))


if __name__ == "__main__":
    main()
