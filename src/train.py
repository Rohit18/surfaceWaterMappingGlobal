#!/usr/bin/env python

"""Train S1 segmentation models with optional raw-AEF fusion.

This script is the NERSC-ready training entrypoint for the crop-based wetland
segmentation workflow. It keeps the current loss, metrics, and schedule shape
while supporting two comparable input modes:

- S1 only: a 3-band Sentinel-1 baseline for direct DW-target comparisons.
- S1 + raw AEF: a 67-band tensor where a learned 1x1 convolution projects the
  64 raw AlphaEarth bands before fusion with S1 inside the model.
- Training uses fixed 512x512 random crops to support batching across the
  variable tile sizes in the dataset.
- Validation keeps full tiles (bs=1) so the reported metrics stay tied to the
  original segmentation task rather than crop-only evaluation.
"""

from __future__ import annotations

import argparse
import copy as copy_module
import csv
import gc
import importlib
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio as rio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

from fastai.imports import *
from fastai.torch_imports import *
from fastai.vision.all import *


DEFAULT_INPUT_DIR = Path("/pscratch/sd/r/rohit9/S1ML/training_mosaics/s1_multiband")
DEFAULT_AEF_DIR = Path("/pscratch/sd/r/rohit9/S1ML/training_embeddings/alphaearth_v1_annual_int8")
DEFAULT_LABEL_DIR = Path("/pscratch/sd/r/rohit9/S1ML/training_mosaics/dw_binary")
DEFAULT_ARTIFACT_DIR = Path(
    "/pscratch/sd/r/rohit9/S1ML/training_runs/s1aef_bottleneck_resnet34_crop512"
)
DEFAULT_HF_CACHE_DIR = Path("/pscratch/sd/r/rohit9/UFO/model_cache/huggingface")

SEGFORMER_MODEL_MAP = {
    "segformer_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer_b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "segformer_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "segformer_b3": "nvidia/segformer-b3-finetuned-ade-512-512",
    "segformer_b4": "nvidia/segformer-b4-finetuned-ade-512-512",
    "segformer_b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}


@dataclass(frozen=True)
class TripletRecord:
    name: str
    s1_path: Path
    aef_path: Optional[Path]
    label_path: Path
    sample_id: str = ""


class CombinedLoss(nn.Module):
    """Focal + Dice, carried over from the working baseline."""

    def __init__(self, axis=1, smooth=1.0, alpha=1.0):
        super().__init__()
        self.axis = axis
        self.alpha = alpha
        self.focal_loss = FocalLossFlat(axis=axis)
        self.dice_loss = DiceLoss(axis, smooth)

    def forward(self, pred, targ):
        pred = pred.float()
        targ = targ.long()
        return self.focal_loss(pred, targ) + (self.alpha * self.dice_loss(pred, targ))

    def decodes(self, x):
        return x.argmax(dim=self.axis)

    def activation(self, x):
        return F.softmax(x, dim=self.axis)


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_softmax_flat(
    probas: torch.Tensor, labels: torch.Tensor, classes: str = "present"
) -> torch.Tensor:
    if probas.numel() == 0:
        return probas * 0.0
    num_classes = probas.size(1)
    losses = []
    class_to_sum = list(range(num_classes)) if classes in ("all", "present") else classes
    for class_idx in class_to_sum:
        foreground = (labels == class_idx).float()
        if classes == "present" and foreground.sum() == 0:
            continue
        errors = (foreground - probas[:, class_idx]).abs()
        errors_sorted, permutation = torch.sort(errors, 0, descending=True)
        foreground_sorted = foreground[permutation.data]
        grad = _lovasz_grad(foreground_sorted)
        losses.append((errors_sorted * grad).sum())
    if not losses:
        return torch.tensor(0.0, device=probas.device, requires_grad=True)
    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    """Lovasz-Softmax loss for IoU-oriented segmentation tuning."""

    def __init__(self, axis: int = 1):
        super().__init__()
        self.axis = axis

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        probas = F.softmax(pred.float(), dim=self.axis)
        _, num_classes, _, _ = probas.shape
        probas_flat = probas.permute(0, 2, 3, 1).reshape(-1, num_classes)
        targ_flat = targ.long().reshape(-1)
        return _lovasz_softmax_flat(probas_flat, targ_flat, classes="present")

    def decodes(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmax(dim=self.axis)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.axis)


class CrossEntropyDiceLoss(nn.Module):
    """Cross-entropy + Dice baseline for a less recall-heavy alternative."""

    def __init__(self, axis: int = 1, smooth: float = 1.0, alpha: float = 1.0):
        super().__init__()
        self.axis = axis
        self.alpha = alpha
        self.cross_entropy = CrossEntropyLossFlat(axis=axis)
        self.dice_loss = DiceLoss(axis, smooth)

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        targ = targ.long()
        return self.cross_entropy(pred, targ) + (self.alpha * self.dice_loss(pred, targ))

    def decodes(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmax(dim=self.axis)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.axis)


class WaterIoU(Metric):
    """Per-class IoU for the water class (index 1)."""

    def __init__(self):
        self.inter = 0
        self.union = 0

    def reset(self):
        self.inter = 0
        self.union = 0

    def accumulate(self, learn):
        pred = learn.pred.argmax(dim=1)
        targ = learn.y
        water_pred = pred == 1
        water_targ = targ == 1
        self.inter += (water_pred & water_targ).float().sum().item()
        self.union += (water_pred | water_targ).float().sum().item()

    @property
    def value(self):
        return self.inter / self.union if self.union > 0 else 0.0

    @property
    def name(self):
        return "water_iou"


class NonFiniteLossCallback(Callback):
    """Stop loudly on NaN/Inf instead of silently cancelling the fit."""

    order = TerminateOnNaNCallback.order

    def after_batch(self):
        if not self.training or self.loss is None:
            return

        loss = self.loss.detach().float()
        if torch.isfinite(loss).all():
            return

        lr = None
        if getattr(self.learn, "opt", None) is not None and len(self.opt.hypers) > 0:
            lr = self.opt.hypers[-1].get("lr")

        debug_summary = {
            "phase": getattr(self.learn, "training_phase", "unknown"),
            "epoch": int(getattr(self.learn, "epoch", -1)),
            "iter": int(getattr(self.learn, "iter", -1)),
            "loss": float(loss.item()),
            "lr": None if lr is None else float(lr),
        }
        print({"non_finite_loss": debug_summary})
        raise RuntimeError("Non-finite loss detected during {phase}.".format(**debug_summary))


class EpochTimerCallback(Callback):
    """Track wall-clock epoch durations for benchmark summaries."""

    def before_fit(self):
        self.epoch_times = []
        self._epoch_start = None

    def before_epoch(self):
        self._epoch_start = time.perf_counter()

    def after_epoch(self):
        if self._epoch_start is None:
            return
        self.epoch_times.append(time.perf_counter() - self._epoch_start)
        self._epoch_start = None


def load_segformer_components():
    try:
        transformers = importlib.import_module("transformers")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "SegFormer support requires the 'transformers' package. "
            "Use a SegFormer-capable environment or keep --model_family=resnet34_unet."
        ) from error
    return transformers.SegformerConfig, transformers.SegformerForSemanticSegmentation


class AEFBottleneckUNet(nn.Module):
    """Wrap a U-Net with a learned raw-AEF bottleneck projection."""

    def __init__(self, unet_model: nn.Module, n_s1: int = 3, n_aef: int = 64, n_proj: int = 4):
        super().__init__()
        self.n_s1 = n_s1
        self.n_aef = n_aef
        self.n_proj = n_proj
        self.aef_proj = nn.Conv2d(n_aef, n_proj, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.aef_proj.weight, gain=0.1)
        self.unet = unet_model

    def forward(self, x):
        s1 = x[:, : self.n_s1]
        aef = x[:, self.n_s1 :]
        aef_projected = self.aef_proj(aef)
        fused = torch.cat([s1, aef_projected], dim=1)
        return self.unet(fused)


class AEFBottleneckSegFormer(nn.Module):
    """Wrap a SegFormer with the same learned AEF projection used by the U-Net."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        n_s1: int = 3,
        n_aef: int = 64,
        n_proj: int = 4,
        hf_cache_dir: Optional[Path] = None,
        pretrained: bool = True,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.n_s1 = n_s1
        self.n_aef = n_aef
        self.n_proj = n_proj
        self.aef_proj = nn.Conv2d(n_aef, n_proj, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.aef_proj.weight, gain=0.1)

        SegformerConfig, SegformerForSemanticSegmentation = load_segformer_components()
        cache_dir = str(hf_cache_dir) if hf_cache_dir is not None else None
        if pretrained:
            config = SegformerConfig.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            config.num_labels = num_labels
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        else:
            config = SegformerConfig.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            config.num_labels = num_labels
            self.segformer = SegformerForSemanticSegmentation(config)

        self._adapt_patch_embedding(in_channels=n_s1 + n_proj)

    def _adapt_patch_embedding(self, in_channels: int) -> None:
        old_proj = self.segformer.segformer.encoder.patch_embeddings[0].proj
        new_proj = nn.Conv2d(
            in_channels,
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )
        with torch.no_grad():
            new_proj.weight.zero_()
            copied_channels = min(old_proj.in_channels, in_channels)
            if copied_channels > 0:
                new_proj.weight[:, :copied_channels] = old_proj.weight[:, :copied_channels]
            if in_channels > copied_channels:
                mean_weight = old_proj.weight.mean(dim=1, keepdim=True)
                new_proj.weight[:, copied_channels:] = mean_weight
            if old_proj.bias is not None and new_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        self.segformer.segformer.encoder.patch_embeddings[0].proj = new_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = x[:, : self.n_s1]
        aef = x[:, self.n_s1 :]
        aef_projected = self.aef_proj(aef)
        fused = torch.cat([s1, aef_projected], dim=1)
        logits = self.segformer(pixel_values=fused).logits
        if logits.shape[-2:] != fused.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=fused.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits


class S1AEFCropDataset(Dataset):
    """Training dataset that samples aligned random crops from triplet tiles."""

    def __init__(
        self,
        records: Sequence[TripletRecord],
        crop_size: int,
        band_mean: np.ndarray,
        band_std: np.ndarray,
        n_s1_bands: int,
        rotate_max_deg: float,
        rotate_p: float,
        scale_jitter_min: float,
        scale_jitter_max: float,
        s1_gain_jitter: float,
        s1_bias_jitter: float,
        gaussian_noise_std: float,
        gaussian_noise_p: float,
    ):
        self.records = list(records)
        self.crop_size = crop_size
        self.band_mean = torch.from_numpy(band_mean.astype(np.float32)).view(-1, 1, 1)
        self.band_std = torch.from_numpy(band_std.astype(np.float32)).view(-1, 1, 1)
        self.n_s1_bands = n_s1_bands
        self.rotate_max_deg = rotate_max_deg
        self.rotate_p = rotate_p
        self.scale_jitter_min = scale_jitter_min
        self.scale_jitter_max = scale_jitter_max
        self.s1_gain_jitter = s1_gain_jitter
        self.s1_bias_jitter = s1_bias_jitter
        self.gaussian_noise_std = gaussian_noise_std
        self.gaussian_noise_p = gaussian_noise_p

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        image, mask = load_triplet_tensors(self.records[idx])
        image, mask = random_crop(image, mask, self.crop_size)
        image, mask = apply_train_augmentations(
            image=image,
            mask=mask,
            crop_size=self.crop_size,
            n_s1_bands=self.n_s1_bands,
            rotate_max_deg=self.rotate_max_deg,
            rotate_p=self.rotate_p,
            scale_jitter_min=self.scale_jitter_min,
            scale_jitter_max=self.scale_jitter_max,
            s1_gain_jitter=self.s1_gain_jitter,
            s1_bias_jitter=self.s1_bias_jitter,
            gaussian_noise_std=self.gaussian_noise_std,
            gaussian_noise_p=self.gaussian_noise_p,
        )
        image = normalize_image(image, self.band_mean, self.band_std)
        return image, mask.long()


class S1AEFFullTileDataset(Dataset):
    """Validation dataset that keeps full aligned tiles (bs=1)."""

    def __init__(self, records: Sequence[TripletRecord], band_mean: np.ndarray, band_std: np.ndarray):
        self.records = list(records)
        self.band_mean = torch.from_numpy(band_mean.astype(np.float32)).view(-1, 1, 1)
        self.band_std = torch.from_numpy(band_std.astype(np.float32)).view(-1, 1, 1)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        image, mask = load_triplet_tensors(self.records[idx])
        image = normalize_image(image, self.band_mean, self.band_std)
        return image, mask.long()


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument("--{name}".format(name=name), dest=name, action="store_true", help=help_text)
    parser.add_argument(
        "--no-{name}".format(name=name),
        dest=name,
        action="store_false",
        help="Disable {name}".format(name=name.replace("_", " ")),
    )
    parser.set_defaults(**{name: default})


def using_aef(args: argparse.Namespace) -> bool:
    return args.n_aef_bands > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train S1 segmentation models with optional raw-AEF fusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--aef_dir", type=Path, default=DEFAULT_AEF_DIR)
    parser.add_argument("--label_dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument(
        "--triplet_manifest",
        type=Path,
        default=None,
        help="CSV with explicit tile_name/name, s1_path, label_path, and optional aef_path rows.",
    )
    parser.add_argument(
        "--triplet_sample_id",
        default=None,
        help="Optional sample_id filter for --triplet_manifest. Required when it contains multiple labels.",
    )
    parser.add_argument("--artifact_dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--warm_start_run_dir",
        type=Path,
        default=None,
        help="Parent run directory with a compatible best checkpoint and band_stats.npz.",
    )
    parser.add_argument(
        "--warm_start_checkpoint",
        type=Path,
        default=None,
        help="Explicit checkpoint for strict warm-start loading.",
    )
    parser.add_argument(
        "--band_stats_path",
        type=Path,
        default=None,
        help="Precomputed band_stats.npz to copy into the new artifact directory.",
    )
    parser.add_argument(
        "--model_family",
        choices=["resnet34_unet", "segformer"],
        default="resnet34_unet",
    )
    parser.add_argument(
        "--segformer_variant",
        choices=list(SEGFORMER_MODEL_MAP.keys()),
        default="segformer_b4",
    )
    parser.add_argument("--hf_cache_dir", type=Path, default=DEFAULT_HF_CACHE_DIR)
    parser.add_argument("--benchmark_summary_path", type=Path, default=None)
    parser.add_argument("--mode", choices=["benchmark", "train"], default="train")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--tile_overlap", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--disable_pin_memory",
        action="store_true",
        help="Disable DataLoader pinned memory for clusters where CUDA pinning is unreliable.",
    )
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Seed for the train/validation split. Defaults to --seed.",
    )
    parser.add_argument(
        "--train_subset_seed",
        type=int,
        default=None,
        help="Seed for deterministic train-record subsampling. Defaults to --seed.",
    )
    parser.add_argument("--valid_pct", type=float, default=0.2)
    parser.add_argument(
        "--max_train_records",
        type=int,
        default=None,
        help="Cap the number of training records after the validation split.",
    )
    parser.add_argument("--stats_sample_size", type=int, default=256)
    parser.add_argument("--freeze_epochs", type=int, default=2)
    parser.add_argument("--partial_unfreeze_epochs", type=int, default=2)
    parser.add_argument("--finetune_epochs", type=int, default=12)
    parser.add_argument("--finetune_tail_epochs", type=int, default=10)
    parser.add_argument("--warmup_lr", type=float, default=1e-4)
    parser.add_argument("--partial_unfreeze_lr", type=float, default=1e-5)
    parser.add_argument("--unfreeze_lr", type=float, default=3e-6)
    parser.add_argument("--finetune_tail_lr", type=float, default=1.5e-6)
    parser.add_argument(
        "--tuning_strategy",
        choices=["staged", "head_only", "decoder_aef", "last_encoder_decoder_aef", "full_low_lr"],
        default="staged",
        help="Explicit trainable scope for transfer-learning strategy comparisons.",
    )
    parser.add_argument("--strategy_epochs", type=int, default=6)
    parser.add_argument("--strategy_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--selection_metric", choices=["dice", "water_iou"], default="water_iou")
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--loss", choices=["focal_dice", "ce_dice", "lovasz"], default="focal_dice")
    parser.add_argument("--rotate_max_deg", type=float, default=5.0)
    parser.add_argument("--rotate_p", type=float, default=0.5)
    parser.add_argument("--scale_jitter_min", type=float, default=1.0)
    parser.add_argument("--scale_jitter_max", type=float, default=1.0)
    parser.add_argument("--s1_gain_jitter", type=float, default=0.0)
    parser.add_argument("--s1_bias_jitter", type=float, default=0.0)
    parser.add_argument("--gaussian_noise_std", type=float, default=0.0)
    parser.add_argument("--gaussian_noise_p", type=float, default=0.0)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_valid_batches", type=int, default=None)
    parser.add_argument("--probe_batch_sizes", type=int, nargs="+", default=[12, 8, 4, 2])
    parser.add_argument("--max_benchmark_batch_size", type=int, default=None)
    parser.add_argument("--eval_thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--worst_tile_count", type=int, default=25)
    parser.add_argument("--n_s1_bands", type=int, default=3)
    parser.add_argument("--n_aef_bands", type=int, default=64)
    parser.add_argument("--n_proj_bands", type=int, default=4)
    add_bool_arg(parser, "use_fp16", True, "Enable mixed precision on GPU.")
    add_bool_arg(parser, "pretrained", True, "Use ImageNet-pretrained encoder weights.")
    add_bool_arg(parser, "hf_local_files_only", False, "Restrict SegFormer loading to the local Hugging Face cache.")
    add_bool_arg(parser, "force_tiled_validation", False, "Force tiled inference during validation and threshold sweeps.")
    args = parser.parse_args()

    args.codes = ["other", "water"]
    if args.n_aef_bands < 0:
        parser.error("--n_aef_bands must be >= 0")
    if args.n_aef_bands == 0 and args.n_proj_bands != 0:
        parser.error("--n_proj_bands must be 0 when --n_aef_bands=0")
    if args.n_aef_bands == 0 and args.model_family == "segformer":
        parser.error("--model_family=segformer is not yet supported with --n_aef_bands=0")

    if args.model_family == "segformer":
        variant_name = args.segformer_variant.replace("segformer_", "")
        if using_aef(args):
            args.best_model_name = "s1aef_bottleneck_segformer_{variant}_best".format(
                variant=variant_name
            )
            args.history_csv = "s1aef_bottleneck_segformer_{variant}_history.csv".format(
                variant=variant_name
            )
            args.threshold_sweep_csv = "s1aef_bottleneck_segformer_{variant}_threshold_sweep.csv".format(
                variant=variant_name
            )
            args.eval_summary_json = "s1aef_bottleneck_segformer_{variant}_eval_summary.json".format(
                variant=variant_name
            )
            args.tile_metrics_csv = "s1aef_bottleneck_segformer_{variant}_tile_metrics.csv".format(
                variant=variant_name
            )
            args.worst_tiles_csv = "s1aef_bottleneck_segformer_{variant}_worst_tiles.csv".format(
                variant=variant_name
            )
    else:
        if using_aef(args):
            args.best_model_name = "s1aef_bottleneck_resnet34_best"
            args.history_csv = "s1aef_bottleneck_resnet34_history.csv"
            args.threshold_sweep_csv = "s1aef_bottleneck_resnet34_threshold_sweep.csv"
            args.eval_summary_json = "s1aef_bottleneck_resnet34_eval_summary.json"
            args.tile_metrics_csv = "s1aef_bottleneck_resnet34_tile_metrics.csv"
            args.worst_tiles_csv = "s1aef_bottleneck_resnet34_worst_tiles.csv"
        else:
            args.best_model_name = "s1dw_resnet34_best"
            args.history_csv = "s1dw_resnet34_history.csv"
            args.threshold_sweep_csv = "s1dw_resnet34_threshold_sweep.csv"
            args.eval_summary_json = "s1dw_resnet34_eval_summary.json"
            args.tile_metrics_csv = "s1dw_resnet34_tile_metrics.csv"
            args.worst_tiles_csv = "s1dw_resnet34_worst_tiles.csv"
    args.benchmark_summary_json = "benchmark_summary.json"
    if args.eval_thresholds is None:
        coarse_low = np.arange(0.30, 0.55, 0.05)
        focused_mid = np.arange(0.55, 0.71, 0.01)
        coarse_high = np.arange(0.75, 0.81, 0.05)
        args.eval_thresholds = sorted(
            {round(float(v), 2) for v in np.concatenate([coarse_low, focused_mid, coarse_high])}
        )
    args.use_tta_eval = True
    args.tta_eval_mode_count = 4
    args.input_dir = args.input_dir.expanduser().resolve()
    args.aef_dir = args.aef_dir.expanduser().resolve()
    args.label_dir = args.label_dir.expanduser().resolve()
    if args.triplet_manifest is not None:
        args.triplet_manifest = args.triplet_manifest.expanduser().resolve()
    args.artifact_dir = args.artifact_dir.expanduser().resolve()
    if args.warm_start_run_dir is not None:
        args.warm_start_run_dir = args.warm_start_run_dir.expanduser().resolve()
    if args.warm_start_checkpoint is not None:
        args.warm_start_checkpoint = args.warm_start_checkpoint.expanduser().resolve()
    if args.band_stats_path is not None:
        args.band_stats_path = args.band_stats_path.expanduser().resolve()
    args.hf_cache_dir = args.hf_cache_dir.expanduser().resolve()
    if args.benchmark_summary_path is not None:
        args.benchmark_summary_path = args.benchmark_summary_path.expanduser().resolve()
    if args.crop_size < 1:
        parser.error("--crop_size must be >= 1")
    if args.tile_overlap < 0 or args.tile_overlap >= args.crop_size:
        parser.error("--tile_overlap must be >= 0 and < --crop_size")
    if args.scale_jitter_min <= 0 or args.scale_jitter_max <= 0:
        parser.error("--scale_jitter_min and --scale_jitter_max must be > 0")
    if args.scale_jitter_min > args.scale_jitter_max:
        parser.error("--scale_jitter_min must be <= --scale_jitter_max")
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.grad_accum_steps < 1:
        parser.error("--grad_accum_steps must be >= 1")
    if args.strategy_epochs < 1:
        parser.error("--strategy_epochs must be >= 1")
    if args.strategy_lr <= 0:
        parser.error("--strategy_lr must be > 0")
    if args.max_benchmark_batch_size is not None and args.max_benchmark_batch_size < 1:
        parser.error("--max_benchmark_batch_size must be >= 1")
    if args.worst_tile_count < 1:
        parser.error("--worst_tile_count must be >= 1")
    if args.triplet_sample_id and args.triplet_manifest is None:
        parser.error("--triplet_sample_id requires --triplet_manifest")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: Optional[int] = None) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def setup_environment(args: argparse.Namespace) -> torch.device:
    seed_everything(args.seed)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "models").mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        actual_device = args.device_id if args.device_id < device_count else 0
        torch.cuda.set_device(actual_device)
        device = torch.device("cuda:{idx}".format(idx=actual_device))
        torch.backends.cudnn.benchmark = True
        print(
            {
                "device": str(device),
                "device_name": torch.cuda.get_device_name(actual_device),
                "device_count": device_count,
            }
        )
    else:
        device = torch.device("cpu")
        print({"device": "cpu"})

    n_input = args.n_s1_bands + args.n_aef_bands
    n_unet_input = args.n_s1_bands + (args.n_proj_bands if using_aef(args) else 0)
    if using_aef(args):
        dataloader_bands = "{s1} S1 + {aef} AEF = {n}".format(
            s1=args.n_s1_bands, aef=args.n_aef_bands, n=n_input
        )
        model_input_bands = "{s1} S1 + {proj} projected = {n}".format(
            s1=args.n_s1_bands, proj=args.n_proj_bands, n=n_unet_input
        )
    else:
        dataloader_bands = "{s1} S1 = {n}".format(s1=args.n_s1_bands, n=n_input)
        model_input_bands = "{s1} S1 = {n}".format(s1=args.n_s1_bands, n=n_unet_input)
    print(
        {
            "artifact_dir": str(args.artifact_dir),
            "mode": args.mode,
            "model_family": args.model_family,
            "segformer_variant": args.segformer_variant if args.model_family == "segformer" else None,
            "dataloader_bands": dataloader_bands,
            "model_input_bands": model_input_bands,
            "crop_size": args.crop_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "loss": args.loss,
            "tuning_strategy": args.tuning_strategy,
            "grad_accum_steps": args.grad_accum_steps,
            "selection_metric": args.selection_metric,
            "eval_thresholds": args.eval_thresholds,
            "force_tiled_validation": args.force_tiled_validation,
            "use_fp16": args.use_fp16,
            "pretrained": args.pretrained,
        }
    )
    return device


def build_aef_index(aef_dir: Path) -> Dict[str, Path]:
    if not aef_dir.exists():
        raise FileNotFoundError("AEF directory does not exist: {path}".format(path=aef_dir))
    index: Dict[str, Path] = {}
    duplicates: List[str] = []
    for path in sorted(aef_dir.rglob("*.tif")):
        rel_parts = path.relative_to(aef_dir).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if path.name in index:
            duplicates.append(path.name)
        else:
            index[path.name] = path
    if duplicates:
        raise RuntimeError(
            "Found duplicate AEF filenames across recursive folders. Examples: {examples}".format(
                examples=duplicates[:5]
            )
        )
    if not index:
        raise FileNotFoundError("No AEF GeoTIFFs found under {path}".format(path=aef_dir))
    return index


def collect_triplets(
    input_dir: Path,
    aef_dir: Optional[Path],
    label_dir: Path,
) -> List[TripletRecord]:
    if not input_dir.exists():
        raise FileNotFoundError("Input directory does not exist: {path}".format(path=input_dir))
    if not label_dir.exists():
        raise FileNotFoundError("Label directory does not exist: {path}".format(path=label_dir))

    use_aef = aef_dir is not None
    aef_index = build_aef_index(aef_dir) if use_aef else {}
    s1_paths = sorted(input_dir.glob("*.tif"))
    triplets: List[TripletRecord] = []
    missing_aef: List[str] = []
    missing_labels: List[str] = []

    for s1_path in s1_paths:
        aef_path = aef_index.get(s1_path.name) if use_aef else None
        label_path = label_dir / s1_path.name
        if use_aef and aef_path is None:
            missing_aef.append(s1_path.name)
            continue
        if not label_path.exists():
            missing_labels.append(label_path.name)
            continue
        triplets.append(
            TripletRecord(
                name=s1_path.name,
                s1_path=s1_path,
                aef_path=aef_path,
                label_path=label_path,
            )
        )

    if not triplets:
        raise FileNotFoundError(
            "No aligned S1/AEF/label triplets found. Missing AEF: {ma}, missing labels: {ml}".format(
                ma=len(missing_aef), ml=len(missing_labels)
            )
        )

    print(
        {
            "triplet_count": len(triplets),
            "missing_aef_count": len(missing_aef) if use_aef else 0,
            "missing_label_count": len(missing_labels),
            "missing_aef_examples": missing_aef[:3] if use_aef else [],
            "missing_label_examples": missing_labels[:3],
        }
    )
    return triplets


def collect_triplets_from_manifest(
    manifest_path: Path,
    sample_id: Optional[str],
    expect_aef: bool,
) -> List[TripletRecord]:
    if not manifest_path.exists():
        raise FileNotFoundError("Triplet manifest does not exist: {path}".format(path=manifest_path))
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"s1_path", "label_path"}
    if expect_aef:
        required.add("aef_path")
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError("Triplet manifest is missing columns: {cols}".format(cols=", ".join(sorted(missing))))

    manifest_sample_ids = sorted({row.get("sample_id", "") for row in rows if row.get("sample_id", "")})
    if sample_id is None and len(manifest_sample_ids) > 1:
        raise ValueError(
            "Triplet manifest has multiple sample_id values; pass --triplet_sample_id. Examples: {ids}".format(
                ids=", ".join(manifest_sample_ids[:5])
            )
        )
    selected_rows = [row for row in rows if sample_id is None or row.get("sample_id", "") == sample_id]
    if not selected_rows:
        raise ValueError("Triplet manifest has no rows for sample_id {sample}".format(sample=sample_id or "<all>"))

    records: List[TripletRecord] = []
    missing_paths: List[str] = []
    for row in selected_rows:
        name = row.get("tile_name") or row.get("name") or Path(row["s1_path"]).name
        s1_path = Path(row["s1_path"]).expanduser()
        label_path = Path(row["label_path"]).expanduser()
        aef_value = row.get("aef_path", "")
        aef_path = Path(aef_value).expanduser() if expect_aef and aef_value else None
        path_checks = [("s1", s1_path), ("label", label_path)]
        if expect_aef:
            if aef_path is None:
                missing_paths.append("{name}: empty aef_path".format(name=name))
                continue
            path_checks.append(("aef", aef_path))
        missing_for_row = ["{kind}={path}".format(kind=kind, path=path) for kind, path in path_checks if not path.exists()]
        if missing_for_row:
            missing_paths.append("{name}: {paths}".format(name=name, paths=", ".join(missing_for_row)))
            continue
        records.append(
            TripletRecord(
                name=name,
                s1_path=s1_path.resolve(),
                aef_path=aef_path.resolve() if aef_path is not None else None,
                label_path=label_path.resolve(),
                sample_id=row.get("sample_id", sample_id or ""),
            )
        )

    if missing_paths:
        raise FileNotFoundError(
            "Triplet manifest has missing or incomplete paths for {n} row(s). Examples: {examples}".format(
                n=len(missing_paths),
                examples="; ".join(missing_paths[:5]),
            )
        )
    if not records:
        raise FileNotFoundError("Triplet manifest did not produce any usable records: {}".format(manifest_path))
    print(
        {
            "triplet_manifest": str(manifest_path),
            "triplet_sample_id": sample_id or (manifest_sample_ids[0] if len(manifest_sample_ids) == 1 else None),
            "triplet_count": len(records),
        }
    )
    return sorted(records, key=lambda record: record.name)


def split_triplets(records: Sequence[TripletRecord], valid_pct: float, seed: int) -> Tuple[List[TripletRecord], List[TripletRecord]]:
    if not records:
        raise ValueError("Cannot split an empty record list.")
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    valid_count = max(1, int(round(len(shuffled) * valid_pct)))
    valid_records = sorted(shuffled[:valid_count], key=lambda r: r.name)
    train_records = sorted(shuffled[valid_count:], key=lambda r: r.name)
    if not train_records:
        raise ValueError("Train split is empty; reduce --valid_pct.")
    return train_records, valid_records


def limit_train_records(
    records: Sequence[TripletRecord],
    max_records: Optional[int],
    subset_seed: int,
) -> List[TripletRecord]:
    selected = list(records)
    if max_records is None:
        return selected
    if max_records <= 0:
        raise ValueError("--max_train_records must be positive when set.")
    rng = random.Random(subset_seed)
    shuffled = list(selected)
    rng.shuffle(shuffled)
    return sorted(shuffled[: min(max_records, len(shuffled))], key=lambda r: r.name)


def write_split_manifest(
    artifact_dir: Path,
    train_records: Sequence[TripletRecord],
    valid_records: Sequence[TripletRecord],
) -> Path:
    split_path = artifact_dir / "triplet_split_manifest.csv"
    fields = ("split", "sample_id", "name", "s1_path", "aef_path", "label_path")
    rows = [("train", record) for record in train_records] + [("valid", record) for record in valid_records]
    with split_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, record in rows:
            writer.writerow(
                {
                    "split": split,
                    "sample_id": record.sample_id,
                    "name": record.name,
                    "s1_path": str(record.s1_path),
                    "aef_path": "" if record.aef_path is None else str(record.aef_path),
                    "label_path": str(record.label_path),
                }
            )
    print({"triplet_split_manifest": str(split_path), "train_tiles": len(train_records), "valid_tiles": len(valid_records)})
    return split_path


def subset_records(records: Sequence[TripletRecord], max_batches: Optional[int], batch_size: int) -> List[TripletRecord]:
    selected = list(records)
    if max_batches is None:
        return selected
    max_items = max(batch_size * max_batches, batch_size)
    return selected[: min(len(selected), max_items)]


def open_geotiff_float(path: Path) -> torch.Tensor:
    with rio.open(str(path)) as src:
        data = src.read().astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(data)


def open_mask_binary(path: Path) -> torch.Tensor:
    with rio.open(str(path)) as src:
        data = src.read(1).astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    mask = (data > 0.5).astype(np.int64)
    return torch.from_numpy(mask)


def load_triplet_tensors(record: TripletRecord) -> Tuple[torch.Tensor, torch.Tensor]:
    s1 = open_geotiff_float(record.s1_path)
    mask = open_mask_binary(record.label_path)
    if record.aef_path is None:
        if s1.shape[1:] != mask.shape:
            raise ValueError(
                "Triplet {name} has mismatched shapes: s1={s1}, mask={mask}".format(
                    name=record.name, s1=tuple(s1.shape), mask=tuple(mask.shape)
                )
            )
        image = s1
    else:
        aef = open_geotiff_float(record.aef_path)
        if s1.shape[1:] != aef.shape[1:] or s1.shape[1:] != mask.shape:
            raise ValueError(
                "Triplet {name} has mismatched shapes: s1={s1}, aef={aef}, mask={mask}".format(
                    name=record.name, s1=tuple(s1.shape), aef=tuple(aef.shape), mask=tuple(mask.shape)
                )
            )
        image = torch.cat([s1, aef], dim=0)
    return image.float(), mask.long()


def compute_band_stats(records: Sequence[TripletRecord], max_files: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    selected = list(records)[: min(len(records), max_files)] if max_files is not None else list(records)
    if not selected:
        raise ValueError("Cannot compute stats without at least one training record.")

    channel_sum = None
    channel_sumsq = None
    pixel_count = 0

    for record in selected:
        image, _ = load_triplet_tensors(record)
        flat = image.numpy().reshape(image.shape[0], -1)
        if channel_sum is None:
            channel_sum = flat.sum(axis=1)
            channel_sumsq = np.square(flat).sum(axis=1)
        else:
            channel_sum += flat.sum(axis=1)
            channel_sumsq += np.square(flat).sum(axis=1)
        pixel_count += flat.shape[1]

    mean = channel_sum / pixel_count
    variance = np.maximum((channel_sumsq / pixel_count) - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32)


def get_band_stats(
    train_records: Sequence[TripletRecord],
    artifact_dir: Path,
    stats_sample_size: int,
    source_stats_path: Optional[Path] = None,
    expected_bands: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    stats_cache = artifact_dir / "band_stats.npz"
    if source_stats_path is not None:
        if not source_stats_path.exists():
            raise FileNotFoundError("Band stats source does not exist: {path}".format(path=source_stats_path))
        if source_stats_path.resolve() != stats_cache.resolve():
            shutil.copy2(source_stats_path, stats_cache)
        save_json(
            artifact_dir / "band_stats_source.json",
            {"source_band_stats": str(source_stats_path.resolve()), "artifact_band_stats": str(stats_cache)},
        )
        print("Copied band stats from {source} to {dest}".format(source=source_stats_path, dest=stats_cache))
    if stats_cache.exists():
        cached = np.load(stats_cache)
        validate_band_stats(cached["mean"], cached["std"], expected_bands, stats_cache)
        print("Loaded cached band stats from {path}".format(path=stats_cache))
        return cached["mean"], cached["std"]
    band_mean, band_std = compute_band_stats(train_records, max_files=stats_sample_size)
    validate_band_stats(band_mean, band_std, expected_bands, stats_cache)
    np.savez(stats_cache, mean=band_mean, std=band_std)
    print("Saved band stats to {path}".format(path=stats_cache))
    return band_mean, band_std


def validate_band_stats(mean: np.ndarray, std: np.ndarray, expected_bands: Optional[int], path: Path) -> None:
    if mean.shape != std.shape:
        raise ValueError("Band stats mean/std shape mismatch at {path}: {mean} vs {std}".format(path=path, mean=mean.shape, std=std.shape))
    if expected_bands is not None and len(mean) != expected_bands:
        raise ValueError(
            "Band stats at {path} have {actual} bands; expected {expected}".format(
                path=path,
                actual=len(mean),
                expected=expected_bands,
            )
        )


def normalize_image(image: torch.Tensor, band_mean: torch.Tensor, band_std: torch.Tensor) -> torch.Tensor:
    return (image - band_mean) / band_std


def random_crop(image: torch.Tensor, mask: torch.Tensor, crop_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    _, height, width = image.shape
    if height < crop_size or width < crop_size:
        raise ValueError(
            "Crop size {crop} exceeds tile shape ({h}, {w})".format(
                crop=crop_size, h=height, w=width
            )
        )
    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)
    return image[:, top : top + crop_size, left : left + crop_size], mask[top : top + crop_size, left : left + crop_size]


def apply_train_augmentations(
    image: torch.Tensor,
    mask: torch.Tensor,
    crop_size: int,
    n_s1_bands: int,
    rotate_max_deg: float,
    rotate_p: float,
    scale_jitter_min: float,
    scale_jitter_max: float,
    s1_gain_jitter: float,
    s1_bias_jitter: float,
    gaussian_noise_std: float,
    gaussian_noise_p: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        image = torch.flip(image, dims=(-1,))
        mask = torch.flip(mask, dims=(-1,))

    rotations = random.randint(0, 3)
    if rotations:
        image = torch.rot90(image, rotations, dims=(-2, -1))
        mask = torch.rot90(mask, rotations, dims=(-2, -1))

    if scale_jitter_min != 1.0 or scale_jitter_max != 1.0:
        scale = random.uniform(scale_jitter_min, scale_jitter_max)
        scaled_size = max(32, int(round(crop_size * scale)))
        if scaled_size != crop_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(scaled_size, scaled_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(scaled_size, scaled_size),
                mode="nearest",
            ).squeeze(0).squeeze(0)
            if scaled_size > crop_size:
                image, mask = random_crop(image, mask, crop_size)
            else:
                pad_h = crop_size - scaled_size
                pad_w = crop_size - scaled_size
                top = random.randint(0, pad_h)
                left = random.randint(0, pad_w)
                bottom = pad_h - top
                right = pad_w - left
                image = F.pad(image, (left, right, top, bottom), mode="reflect")
                mask = F.pad(mask, (left, right, top, bottom), mode="constant", value=0)

    if rotate_max_deg > 0 and random.random() < rotate_p:
        angle = random.uniform(-rotate_max_deg, rotate_max_deg)
        image = TVF.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR)
        mask = TVF.rotate(
            mask.unsqueeze(0).float(),
            angle=angle,
            interpolation=InterpolationMode.NEAREST,
        ).squeeze(0)

    if s1_gain_jitter > 0 or s1_bias_jitter > 0:
        gain = 1.0 + random.uniform(-s1_gain_jitter, s1_gain_jitter)
        bias = random.uniform(-s1_bias_jitter, s1_bias_jitter)
        image[:n_s1_bands] = (image[:n_s1_bands] * gain) + bias

    if gaussian_noise_std > 0 and random.random() < gaussian_noise_p:
        image = image + torch.randn_like(image) * gaussian_noise_std

    return image.float(), mask.long()


def build_dataloaders(
    args: argparse.Namespace,
    train_records: Sequence[TripletRecord],
    valid_records: Sequence[TripletRecord],
    band_mean: np.ndarray,
    band_std: np.ndarray,
    device: torch.device,
) -> DataLoaders:
    train_dataset = S1AEFCropDataset(
        records=train_records,
        crop_size=args.crop_size,
        band_mean=band_mean,
        band_std=band_std,
        n_s1_bands=args.n_s1_bands,
        rotate_max_deg=args.rotate_max_deg,
        rotate_p=args.rotate_p,
        scale_jitter_min=args.scale_jitter_min,
        scale_jitter_max=args.scale_jitter_max,
        s1_gain_jitter=args.s1_gain_jitter,
        s1_bias_jitter=args.s1_bias_jitter,
        gaussian_noise_std=args.gaussian_noise_std,
        gaussian_noise_p=args.gaussian_noise_p,
    )
    valid_dataset = S1AEFFullTileDataset(valid_records, band_mean=band_mean, band_std=band_std)

    pin_memory = device.type == "cuda" and not args.disable_pin_memory
    train_dl = DataLoader(
        train_dataset,
        bs=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        device=None,
        wif=seed_worker,
    )
    valid_dl = DataLoader(
        valid_dataset,
        bs=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        device=None,
        wif=seed_worker,
    )
    dls = DataLoaders(train_dl, valid_dl, path=args.artifact_dir, device=device)
    dls.c = len(args.codes)
    dls.vocab = args.codes
    return dls


def format_lr(lr_max):
    if isinstance(lr_max, slice):
        return {"low": lr_max.start, "high": lr_max.stop}
    return lr_max


def build_callbacks(args: argparse.Namespace, epoch_timer: Optional[EpochTimerCallback] = None) -> List[Callback]:
    callbacks: List[Callback] = [
        GradientClip(args.grad_clip),
        NonFiniteLossCallback(),
        SaveModelCallback(
            monitor=args.selection_metric,
            comp=np.greater,
            min_delta=args.min_delta,
            fname=args.best_model_name,
        ),
        EarlyStoppingCallback(
            monitor=args.selection_metric,
            comp=np.greater,
            min_delta=args.min_delta,
            patience=args.patience,
        ),
    ]
    if args.grad_accum_steps > 1:
        callbacks.append(GradientAccumulation(n_acc=args.grad_accum_steps))
    if epoch_timer is not None:
        callbacks.append(epoch_timer)
    return callbacks


def recorder_history_to_df(learn: Learner, phase: str) -> pd.DataFrame:
    values = list(learn.recorder.values)
    if not values:
        return pd.DataFrame(columns=["phase"])

    column_count = len(values[0])
    metric_names = list(learn.recorder.metric_names[1:])
    if metric_names and metric_names[-1] == "time" and len(metric_names) == column_count + 1:
        metric_names = metric_names[:-1]
    if len(metric_names) != column_count:
        metric_names = ["value_{idx}".format(idx=idx) for idx in range(column_count)]
    return pd.DataFrame(values, columns=metric_names).assign(phase=phase)


def run_phase(
    learn: Learner,
    args: argparse.Namespace,
    phase: str,
    epochs: int,
    lr_max,
    epoch_timer: Optional[EpochTimerCallback] = None,
) -> pd.DataFrame:
    if epochs <= 0:
        return pd.DataFrame(columns=["phase"])

    print({"phase": phase, "epochs": epochs, "lr_max": format_lr(lr_max)})
    learn.training_phase = phase
    try:
        learn.fit_one_cycle(
            epochs,
            lr_max=lr_max,
            wd=args.weight_decay,
            cbs=build_callbacks(args, epoch_timer=epoch_timer),
            reset_opt=True,
        )
    finally:
        learn.training_phase = None
    return recorder_history_to_df(learn, phase=phase)


def build_schedule(args: argparse.Namespace) -> List[Tuple[str, int, object]]:
    if args.mode == "benchmark":
        return [("freeze", 2, args.warmup_lr)]
    if args.tuning_strategy != "staged":
        return [(args.tuning_strategy, args.strategy_epochs, args.strategy_lr)]
    schedule: List[Tuple[str, int, object]] = [
        ("freeze", args.freeze_epochs, args.warmup_lr),
    ]
    bridge_epochs = min(args.partial_unfreeze_epochs, args.finetune_epochs)
    remaining_finetune_epochs = max(args.finetune_epochs - bridge_epochs, 0)
    if bridge_epochs > 0:
        schedule.append(
            (
                "partial_unfreeze",
                bridge_epochs,
                slice(args.partial_unfreeze_lr / 20, args.partial_unfreeze_lr),
            )
        )
    if remaining_finetune_epochs > 0:
        schedule.append(
            (
                "finetune",
                remaining_finetune_epochs,
                slice(args.unfreeze_lr / 50, args.unfreeze_lr),
            )
        )
    if args.finetune_tail_epochs > 0:
        schedule.append(
            (
                "finetune_tail",
                args.finetune_tail_epochs,
                slice(args.finetune_tail_lr / 10, args.finetune_tail_lr),
            )
        )
    return schedule


def set_requires_grad(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def trainable_parameter_summary(model: nn.Module) -> Dict[str, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"trainable_parameters": int(trainable), "total_parameters": int(total)}


def resnet_unet_children(model: nn.Module) -> List[nn.Module]:
    if not isinstance(model, AEFBottleneckUNet):
        raise ValueError("Explicit tuning strategies currently require AEFBottleneckUNet.")
    children = list(model.unet.layers) if hasattr(model.unet, "layers") else list(model.unet.children())
    if len(children) < 3:
        raise ValueError("Unexpected DynamicUnet layout for tuning strategy.")
    return children


def enable_head_only(model: nn.Module) -> None:
    children = resnet_unet_children(model)
    parameterized_children = [module for module in children if any(True for _ in module.parameters())]
    if not parameterized_children:
        raise ValueError("DynamicUnet head did not expose trainable parameters.")
    head = parameterized_children[-1]
    set_requires_grad(head, True)


def enable_decoder(model: nn.Module) -> None:
    children = resnet_unet_children(model)
    for module in children[1:]:
        set_requires_grad(module, True)


def enable_aef_projection(model: nn.Module) -> None:
    if not isinstance(model, AEFBottleneckUNet):
        raise ValueError("AEF projection strategy requires AEFBottleneckUNet.")
    set_requires_grad(model.aef_proj, True)


def enable_last_encoder_stage(model: nn.Module) -> None:
    encoder = resnet_unet_children(model)[0]
    encoder_children = list(encoder.children())
    if not encoder_children:
        raise ValueError("DynamicUnet encoder did not expose child stages.")
    set_requires_grad(encoder_children[-1], True)


def apply_explicit_tuning_strategy(learn: Learner, strategy: str) -> None:
    set_requires_grad(learn.model, False)
    if strategy == "head_only":
        enable_head_only(learn.model)
    elif strategy == "decoder_aef":
        enable_decoder(learn.model)
        enable_aef_projection(learn.model)
    elif strategy == "last_encoder_decoder_aef":
        enable_decoder(learn.model)
        enable_aef_projection(learn.model)
        enable_last_encoder_stage(learn.model)
    elif strategy == "full_low_lr":
        set_requires_grad(learn.model, True)
    else:
        raise ValueError("Unknown explicit tuning strategy: {}".format(strategy))
    summary = trainable_parameter_summary(learn.model)
    if summary["trainable_parameters"] == 0:
        raise ValueError("Tuning strategy {} left no trainable parameters.".format(strategy))
    print({"tuning_strategy_scope": strategy, **summary})


def apply_phase_freeze(learn: Learner, phase: str) -> None:
    if phase in {"head_only", "decoder_aef", "last_encoder_decoder_aef", "full_low_lr"}:
        apply_explicit_tuning_strategy(learn, phase)
        return
    if phase == "freeze":
        learn.freeze()
    elif phase == "partial_unfreeze":
        learn.freeze_to(-2)
    else:
        learn.unfreeze()


def build_wrapped_splitter(base_splitter):
    def wrapped_splitter(model: AEFBottleneckUNet):
        groups = list(base_splitter(model.unet))
        if not groups:
            return L([L(model.parameters())])
        final_group = L(groups[-1]) + L(model.aef_proj.parameters())
        return L(groups[:-1]) + L([final_group])

    return wrapped_splitter


def build_segformer_splitter():
    def splitter(model: AEFBottleneckSegFormer):
        encoder = model.segformer.segformer.encoder
        groups: List[L] = []
        for patch_embedding, blocks, norm in zip(
            encoder.patch_embeddings,
            encoder.block,
            encoder.layer_norm,
        ):
            groups.append(
                L(patch_embedding.parameters()) + L(blocks.parameters()) + L(norm.parameters())
            )
        groups.append(L(model.segformer.decode_head.parameters()) + L(model.aef_proj.parameters()))
        return L(groups)

    return splitter


def build_loss(loss_name: str) -> nn.Module:
    if loss_name == "focal_dice":
        return CombinedLoss()
    if loss_name == "ce_dice":
        return CrossEntropyDiceLoss()
    if loss_name == "lovasz":
        return LovaszSoftmaxLoss()
    raise ValueError("Unknown loss: {name}".format(name=loss_name))


def build_learner(args: argparse.Namespace, dls: DataLoaders) -> Learner:
    n_total_in = args.n_s1_bands + args.n_aef_bands
    n_model_in = args.n_s1_bands + (args.n_proj_bands if using_aef(args) else 0)
    loss_func = build_loss(args.loss)
    if args.model_family == "segformer":
        wrapped_model = AEFBottleneckSegFormer(
            model_name=SEGFORMER_MODEL_MAP[args.segformer_variant],
            num_labels=len(args.codes),
            n_s1=args.n_s1_bands,
            n_aef=args.n_aef_bands,
            n_proj=args.n_proj_bands,
            hf_cache_dir=args.hf_cache_dir,
            pretrained=args.pretrained,
            local_files_only=args.hf_local_files_only,
        )
        wrapped_model = wrapped_model.to(dls.device)
        learn = Learner(
            dls,
            wrapped_model,
            loss_func=loss_func,
            metrics=[JaccardCoeff(), Dice(), WaterIoU()],
            opt_func=ranger,
            splitter=build_segformer_splitter(),
            path=args.artifact_dir,
            model_dir="models",
        )
        model_name = "AEFBottleneckSegFormer({variant})".format(
            variant=args.segformer_variant
        )
    else:
        learn = unet_learner(
            dls,
            resnet34,
            normalize=False,
            n_in=n_model_in,
            n_out=len(args.codes),
            metrics=[JaccardCoeff(), Dice(), WaterIoU()],
            loss_func=loss_func,
            opt_func=ranger,
            act_cls=Mish,
            path=args.artifact_dir,
            model_dir="models",
            pretrained=args.pretrained,
        )
        if using_aef(args):
            base_splitter = learn.splitter
            wrapped_model = AEFBottleneckUNet(
                unet_model=learn.model,
                n_s1=args.n_s1_bands,
                n_aef=args.n_aef_bands,
                n_proj=args.n_proj_bands,
            )
            learn.model = wrapped_model.to(dls.device)
            learn.splitter = build_wrapped_splitter(base_splitter)
            model_name = "AEFBottleneckUNet(resnet34_unet)"
        else:
            model_name = "S1UNet(resnet34_unet)"
        learn.model = learn.model.to(dls.device)
    if args.use_fp16 and torch.cuda.is_available():
        learn = learn.to_fp16()

    model_summary = {
        "model": model_name,
        "dataloader_input_bands": n_total_in,
        "backbone_input_bands": n_model_in,
        "loss": args.loss,
        "selection_metric": args.selection_metric,
        "tuning_strategy": args.tuning_strategy,
        "grad_accum_steps": args.grad_accum_steps,
        "precision": "fp16" if args.use_fp16 and torch.cuda.is_available() else "fp32",
        "best_model_path": str(args.artifact_dir / "models" / "{name}.pth".format(name=args.best_model_name)),
    }
    if using_aef(args):
        model_summary["bottleneck"] = "{aef}->{proj}".format(
            aef=args.n_aef_bands, proj=args.n_proj_bands
        )
    print(
        model_summary
    )
    return learn


def resolve_warm_start_checkpoint(args: argparse.Namespace) -> Optional[Path]:
    if args.warm_start_checkpoint is not None:
        return args.warm_start_checkpoint
    if args.warm_start_run_dir is None:
        return None
    return args.warm_start_run_dir / "models" / "{name}.pth".format(name=args.best_model_name)


def resolve_band_stats_source(args: argparse.Namespace) -> Optional[Path]:
    if args.band_stats_path is not None:
        return args.band_stats_path
    if args.warm_start_run_dir is not None:
        return args.warm_start_run_dir / "band_stats.npz"
    return None


def unwrap_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Warm-start checkpoint must contain a state dict mapping.")
    for key in ("model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    if not checkpoint or not all(isinstance(key, str) for key in checkpoint.keys()):
        raise ValueError("Warm-start checkpoint does not expose string state-dict keys.")
    return checkpoint


def validate_warm_start_projection(
    state_dict: Dict[str, torch.Tensor],
    model: nn.Module,
    args: argparse.Namespace,
    checkpoint_path: Path,
) -> None:
    if not using_aef(args):
        return
    key = "aef_proj.weight"
    if key not in state_dict:
        raise ValueError("Warm-start checkpoint is missing {key}: {path}".format(key=key, path=checkpoint_path))
    checkpoint_shape = tuple(state_dict[key].shape)
    model_shape = tuple(model.state_dict()[key].shape)
    expected_shape = (args.n_proj_bands, args.n_aef_bands, 1, 1)
    if checkpoint_shape != expected_shape or model_shape != expected_shape:
        raise ValueError(
            "Warm-start AEF projection shape mismatch at {path}: checkpoint={ckpt}, model={model}, expected={expected}".format(
                path=checkpoint_path,
                ckpt=checkpoint_shape,
                model=model_shape,
                expected=expected_shape,
            )
        )


def maybe_warm_start(learn: Learner, args: argparse.Namespace) -> Optional[Path]:
    checkpoint_path = resolve_warm_start_checkpoint(args)
    if checkpoint_path is None:
        return None
    if not checkpoint_path.exists():
        raise FileNotFoundError("Warm-start checkpoint does not exist: {path}".format(path=checkpoint_path))
    state_dict = unwrap_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    validate_warm_start_projection(state_dict, learn.model, args, checkpoint_path)
    try:
        learn.model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError("Warm-start checkpoint is incompatible with this model: {path}\n{error}".format(path=checkpoint_path, error=exc)) from exc
    save_json(
        args.artifact_dir / "warm_start.json",
        {
            "checkpoint": str(checkpoint_path.resolve()),
            "parent_run_dir": "" if args.warm_start_run_dir is None else str(args.warm_start_run_dir),
            "n_aef_bands": args.n_aef_bands,
            "n_proj_bands": args.n_proj_bands,
        },
    )
    print({"warm_start_checkpoint": str(checkpoint_path), "warm_start_strict": True})
    return checkpoint_path


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def inspect_bottleneck(model: nn.Module, artifact_dir: Path) -> dict:
    weight = model.aef_proj.weight.detach().cpu().squeeze(-1).squeeze(-1)
    abs_weight = weight.abs()
    top_k = 5
    report = {
        "projection_shape": list(weight.shape),
        "weight_norm_per_proj_band": [float(v) for v in weight.norm(dim=1).tolist()],
        "weight_norm_per_aef_band": [float(v) for v in weight.norm(dim=0).tolist()],
    }
    for idx in range(weight.shape[0]):
        top_vals, top_idx = abs_weight[idx].topk(top_k)
        report["proj_band_{idx}_top{top}_aef_bands".format(idx=idx, top=top_k)] = top_idx.tolist()
        report["proj_band_{idx}_top{top}_weights".format(idx=idx, top=top_k)] = [
            round(v, 6) for v in top_vals.tolist()
        ]

    np.save(artifact_dir / "aef_bottleneck_weights.npy", weight.numpy())
    save_json(artifact_dir / "aef_bottleneck_report.json", report)
    print(
        {
            "bottleneck_weight_file": str(artifact_dir / "aef_bottleneck_weights.npy"),
            "bottleneck_report_file": str(artifact_dir / "aef_bottleneck_report.json"),
        }
    )
    return report


def get_tta_transforms(mode_count: int):
    transforms = [
        ("identity", lambda x: x),
        ("flip_horizontal", lambda x: torch.flip(x, dims=(-1,))),
        ("flip_vertical", lambda x: torch.flip(x, dims=(-2,))),
        ("flip_both", lambda x: torch.flip(x, dims=(-2, -1))),
    ]
    return transforms[: max(1, min(mode_count, len(transforms)))]


def is_cuda_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def sliding_positions(size: int, tile_size: int, overlap: int) -> List[int]:
    if size <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    positions = list(range(0, size - tile_size + 1, stride))
    if positions[-1] != size - tile_size:
        positions.append(size - tile_size)
    return positions


def predict_logits_tiled(model: nn.Module, xb: torch.Tensor, tile_size: int, overlap: int) -> torch.Tensor:
    batch_size, _, height, width = xb.shape
    if batch_size != 1:
        raise ValueError("Tiled inference expects batch size 1, got {bs}".format(bs=batch_size))

    y_positions = sliding_positions(height, tile_size, overlap)
    x_positions = sliding_positions(width, tile_size, overlap)
    logits_sum = None
    weight_sum = torch.zeros((1, 1, height, width), device=xb.device, dtype=torch.float32)

    for top in y_positions:
        for left in x_positions:
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            tile = xb[:, :, top:bottom, left:right]
            logits = model(tile).float()
            if logits_sum is None:
                logits_sum = torch.zeros(
                    (1, logits.shape[1], height, width), device=xb.device, dtype=torch.float32
                )
            logits_sum[:, :, top:bottom, left:right] += logits
            weight_sum[:, :, top:bottom, left:right] += 1.0

    return logits_sum / weight_sum.clamp_min(1.0)


def predict_logits_with_fallback(
    model: nn.Module,
    xb: torch.Tensor,
    tile_size: int,
    overlap: int,
    force_tiled: bool = False,
) -> Tuple[torch.Tensor, bool]:
    if force_tiled:
        return predict_logits_tiled(model, xb, tile_size=tile_size, overlap=overlap), True
    try:
        return model(xb).float(), False
    except RuntimeError as error:
        if not is_cuda_oom(error):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print({"eval_fallback": "switching to tiled inference"})
        return predict_logits_tiled(model, xb, tile_size=tile_size, overlap=overlap), True


def predict_water_probabilities(
    model: nn.Module,
    xb: torch.Tensor,
    tile_size: int,
    overlap: int,
    use_tta: bool = False,
    tta_mode_count: int = 4,
    force_tiled: bool = False,
) -> Tuple[torch.Tensor, bool]:
    if not use_tta:
        logits, used_tiled = predict_logits_with_fallback(
            model, xb, tile_size=tile_size, overlap=overlap, force_tiled=force_tiled
        )
        return F.softmax(logits, dim=1)[:, 1], used_tiled

    transforms = get_tta_transforms(tta_mode_count)
    probs_sum = None
    used_tiled_any = force_tiled
    for _, transform in transforms:
        augmented_x = transform(xb)
        logits, used_tiled = predict_logits_with_fallback(
            model, augmented_x, tile_size=tile_size, overlap=overlap, force_tiled=used_tiled_any
        )
        augmented_probs = F.softmax(logits, dim=1)[:, 1:2]
        restored_probs = transform(augmented_probs)
        probs_sum = restored_probs if probs_sum is None else probs_sum + restored_probs
        used_tiled_any = used_tiled_any or used_tiled
    return (probs_sum / len(transforms))[:, 0], used_tiled_any


def metrics_from_binary_confusion(tp: int, fp: int, tn: int, fn: int) -> dict:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    water_iou = tp / max(tp + fp + fn, 1)
    dice = (2 * tp) / max((2 * tp) + fp + fn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    other_iou = tn / max(tn + fp + fn, 1)
    other_dice = (2 * tn) / max((2 * tn) + fp + fn, 1)
    mean_iou = (other_iou + water_iou) / 2.0
    mean_dice = (other_dice + dice) / 2.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "water_iou": float(water_iou),
        "dice": float(mean_dice),
        "accuracy": float(accuracy),
        "jaccard": float(mean_iou),
    }


def validate_model(
    learn: Learner,
    tile_size: int,
    overlap: int,
    force_tiled: bool = False,
    max_valid_batches: Optional[int] = None,
) -> dict:
    device = next(learn.model.parameters()).device
    loss_func = learn.loss_func
    total_loss = 0.0
    batch_count = 0
    tp = fp = tn = fn = 0
    use_tiled = False

    was_training = learn.model.training
    learn.model.eval()
    with torch.inference_mode():
        for batch_idx, (xb, yb) in enumerate(learn.dls.valid):
            if max_valid_batches is not None and batch_idx >= max_valid_batches:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits, used_tiled = predict_logits_with_fallback(
                learn.model,
                xb,
                tile_size=tile_size,
                overlap=overlap,
                force_tiled=force_tiled or use_tiled,
            )
            use_tiled = use_tiled or used_tiled
            batch_loss = loss_func(logits, yb).detach().float().item()
            pred = logits.argmax(dim=1)
            total_loss += batch_loss
            batch_count += 1
            tp += int(((pred == 1) & (yb == 1)).sum().item())
            fp += int(((pred == 1) & (yb != 1)).sum().item())
            tn += int(((pred != 1) & (yb != 1)).sum().item())
            fn += int(((pred != 1) & (yb == 1)).sum().item())
    learn.model.train(was_training)

    metric_summary = metrics_from_binary_confusion(tp=tp, fp=fp, tn=tn, fn=fn)
    metric_summary.update(
        {
            "loss": float(total_loss / max(batch_count, 1)),
            "used_tiled_inference": use_tiled,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
    )
    return metric_summary


def evaluate_threshold_sweep(
    learn: Learner,
    thresholds: Sequence[float],
    tile_size: int,
    overlap: int,
    use_tta: bool = False,
    tta_mode_count: int = 4,
    force_tiled: bool = False,
    max_valid_batches: Optional[int] = None,
) -> pd.DataFrame:
    device = next(learn.model.parameters()).device
    threshold_tensor = torch.tensor(thresholds, device=device, dtype=torch.float32).view(-1, 1, 1, 1)
    counts = {name: torch.zeros(len(thresholds), dtype=torch.long) for name in ["tp", "fp", "tn", "fn"]}
    eval_mode = "tta" if use_tta else "plain"
    if force_tiled:
        eval_mode = "{mode}_tiled".format(mode=eval_mode)
    use_tiled = force_tiled

    was_training = learn.model.training
    learn.model.eval()
    with torch.inference_mode():
        for batch_idx, (xb, yb) in enumerate(learn.dls.valid):
            if max_valid_batches is not None and batch_idx >= max_valid_batches:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            probs, used_tiled_now = predict_water_probabilities(
                learn.model,
                xb,
                tile_size=tile_size,
                overlap=overlap,
                use_tta=use_tta,
                tta_mode_count=tta_mode_count,
                force_tiled=force_tiled or use_tiled,
            )
            use_tiled = use_tiled or used_tiled_now
            pred = probs.unsqueeze(0) >= threshold_tensor
            targ = (yb == 1).unsqueeze(0)
            counts["tp"] += (pred & targ).sum(dim=(1, 2, 3)).cpu()
            counts["fp"] += (pred & ~targ).sum(dim=(1, 2, 3)).cpu()
            counts["tn"] += ((~pred) & ~targ).sum(dim=(1, 2, 3)).cpu()
            counts["fn"] += ((~pred) & targ).sum(dim=(1, 2, 3)).cpu()
    learn.model.train(was_training)

    records: List[dict] = []
    for idx, threshold in enumerate(thresholds):
        tp = int(counts["tp"][idx].item())
        fp = int(counts["fp"][idx].item())
        tn = int(counts["tn"][idx].item())
        fn = int(counts["fn"][idx].item())
        metrics = metrics_from_binary_confusion(tp=tp, fp=fp, tn=tn, fn=fn)
        records.append(
            {
                "eval_mode": eval_mode,
                "threshold": float(threshold),
                "used_tiled_inference": use_tiled,
                **metrics,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    return pd.DataFrame(records)


def evaluate_tiles_at_threshold(
    learn: Learner,
    valid_records: Sequence[TripletRecord],
    threshold: float,
    tile_size: int,
    overlap: int,
    use_tta: bool = False,
    tta_mode_count: int = 4,
    force_tiled: bool = False,
    max_valid_batches: Optional[int] = None,
) -> pd.DataFrame:
    device = next(learn.model.parameters()).device
    rows: List[dict] = []
    eval_mode = "tta" if use_tta else "plain"
    if force_tiled:
        eval_mode = "{mode}_tiled".format(mode=eval_mode)

    was_training = learn.model.training
    learn.model.eval()
    with torch.inference_mode():
        for batch_idx, ((xb, yb), record) in enumerate(zip(learn.dls.valid, valid_records)):
            if max_valid_batches is not None and batch_idx >= max_valid_batches:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            probs, used_tiled = predict_water_probabilities(
                learn.model,
                xb,
                tile_size=tile_size,
                overlap=overlap,
                use_tta=use_tta,
                tta_mode_count=tta_mode_count,
                force_tiled=force_tiled,
            )
            pred = probs >= threshold
            targ = yb == 1
            tp = int((pred & targ).sum().item())
            fp = int((pred & ~targ).sum().item())
            tn = int(((~pred) & ~targ).sum().item())
            fn = int(((~pred) & targ).sum().item())
            metrics = metrics_from_binary_confusion(tp=tp, fp=fp, tn=tn, fn=fn)
            rows.append(
                {
                    "tile_name": record.name,
                    "eval_mode": eval_mode,
                    "threshold": float(threshold),
                    "used_tiled_inference": bool(used_tiled),
                    "water_pixels": int(targ.sum().item()),
                    **metrics,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                }
            )
    learn.model.train(was_training)
    return pd.DataFrame(rows)


def summarize_threshold_results(threshold_df: pd.DataFrame) -> dict:
    summary: dict = {}
    if threshold_df.empty:
        return summary

    for eval_mode in threshold_df["eval_mode"].unique():
        mode_df = threshold_df[threshold_df["eval_mode"] == eval_mode].sort_values(
            ["water_iou", "dice", "threshold"],
            ascending=[False, False, True],
        )
        best = mode_df.iloc[0]
        summary.update(
            {
                "{mode}_best_threshold".format(mode=eval_mode): round(float(best["threshold"]), 4),
                "{mode}_best_water_iou".format(mode=eval_mode): round(float(best["water_iou"]), 6),
                "{mode}_best_dice".format(mode=eval_mode): round(float(best["dice"]), 6),
                "{mode}_best_precision".format(mode=eval_mode): round(float(best["precision"]), 6),
                "{mode}_best_recall".format(mode=eval_mode): round(float(best["recall"]), 6),
            }
        )

    best_row = threshold_df.sort_values(["water_iou", "dice", "threshold"], ascending=[False, False, True]).iloc[0]
    summary.update(
        {
            "best_eval_mode": str(best_row["eval_mode"]),
            "best_eval_threshold": round(float(best_row["threshold"]), 4),
            "best_eval_water_iou": round(float(best_row["water_iou"]), 6),
            "best_eval_dice": round(float(best_row["dice"]), 6),
            "best_eval_precision": round(float(best_row["precision"]), 6),
            "best_eval_recall": round(float(best_row["recall"]), 6),
        }
    )
    return summary


def summarize_worst_tiles(tile_df: pd.DataFrame, worst_tile_count: int) -> pd.DataFrame:
    if tile_df.empty:
        return tile_df
    return tile_df.sort_values(
        ["water_iou", "precision", "recall", "tile_name"],
        ascending=[True, True, True, True],
    ).head(worst_tile_count)


def run_training_schedule(
    learn: Learner,
    args: argparse.Namespace,
    schedule: Sequence[Tuple[str, int, object]],
    epoch_timer: Optional[EpochTimerCallback] = None,
) -> pd.DataFrame:
    training_history = []
    for phase_name, epochs, lr_max in schedule:
        apply_phase_freeze(learn, phase_name)
        training_history.append(
            run_phase(
                learn=learn,
                args=args,
                phase=phase_name,
                epochs=epochs,
                lr_max=lr_max,
                epoch_timer=epoch_timer,
            )
        )
        if epochs > 0:
            learn.load(args.best_model_name, with_opt=False)
    return pd.concat(training_history, ignore_index=True) if training_history else pd.DataFrame(columns=["phase"])


def effective_epoch_count(args: argparse.Namespace) -> int:
    return int(args.freeze_epochs + args.partial_unfreeze_epochs + args.finetune_epochs + args.finetune_tail_epochs)


def probe_batch_size(
    args: argparse.Namespace,
    train_records: Sequence[TripletRecord],
    valid_records: Sequence[TripletRecord],
    band_mean: np.ndarray,
    band_std: np.ndarray,
    device: torch.device,
) -> dict:
    if device.type != "cuda":
        return {
            "selected_batch_size": args.batch_size,
            "probe_results": [],
            "probe_skipped": "CUDA unavailable",
        }

    results: List[dict] = []
    selected_batch_size = None
    train_probe_records = list(train_records[: max(max(args.probe_batch_sizes), 16)])
    valid_probe_records = list(valid_records[:1])

    for candidate in args.probe_batch_sizes:
        candidate_args = copy_module.deepcopy(args)
        candidate_args.batch_size = candidate
        candidate_args.num_workers = min(candidate_args.num_workers, 2)
        try:
            dls = build_dataloaders(
                candidate_args,
                train_records=train_probe_records,
                valid_records=valid_probe_records,
                band_mean=band_mean,
                band_std=band_std,
                device=device,
            )
            learn = build_learner(candidate_args, dls)
            learn.model.train()
            optimizer = torch.optim.AdamW(learn.model.parameters(), lr=args.warmup_lr)
            xb, yb = dls.one_batch()
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            if args.use_fp16:
                scaler = torch.cuda.amp.GradScaler(enabled=True)
                with torch.cuda.amp.autocast(enabled=True):
                    pred = learn.model(xb)
                    loss = learn.loss_func(pred, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = learn.model(xb)
                loss = learn.loss_func(pred, yb)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            result = {
                "batch_size": candidate,
                "success": True,
                "loss": float(loss.detach().float().item()),
                "step_seconds": round(elapsed, 4),
                "max_memory_gb": round(peak_mem_gb, 4),
            }
            results.append(result)
            selected_batch_size = candidate
            del pred, loss, xb, yb, learn, dls, optimizer
            gc.collect()
            torch.cuda.empty_cache()
            break
        except RuntimeError as error:
            if not is_cuda_oom(error):
                raise
            results.append({"batch_size": candidate, "success": False, "error": "cuda_oom"})
            gc.collect()
            torch.cuda.empty_cache()

    if selected_batch_size is None:
        raise RuntimeError("Batch size probe failed for all candidates: {candidates}".format(candidates=args.probe_batch_sizes))

    probe_summary = {"selected_batch_size": selected_batch_size, "probe_results": results}
    print(probe_summary)
    return probe_summary


def maybe_override_batch_size_from_summary(args: argparse.Namespace) -> None:
    if args.benchmark_summary_path is None:
        return
    if not args.benchmark_summary_path.exists():
        raise FileNotFoundError(
            "Benchmark summary path does not exist: {path}".format(path=args.benchmark_summary_path)
        )
    summary = json.loads(args.benchmark_summary_path.read_text())
    selected = summary.get("selected_batch_size")
    if not selected:
        raise ValueError(
            "Benchmark summary at {path} is missing selected_batch_size".format(path=args.benchmark_summary_path)
        )
    args.batch_size = int(selected)
    print(
        {
            "batch_size_override": args.batch_size,
            "benchmark_summary_path": str(args.benchmark_summary_path),
        }
    )


def build_audit_summary(
    args: argparse.Namespace,
    train_records: Sequence[TripletRecord],
    valid_records: Sequence[TripletRecord],
    band_mean: np.ndarray,
    band_std: np.ndarray,
    dls: DataLoaders,
) -> dict:
    sample_image, sample_mask = load_triplet_tensors(train_records[0])
    batch_images, batch_masks = dls.one_batch()
    return {
        "paired_tiles": len(train_records) + len(valid_records),
        "train_tiles": len(train_records),
        "valid_tiles": len(valid_records),
        "sample_image_shape": tuple(sample_image.shape),
        "sample_mask_shape": tuple(sample_mask.shape),
        "sample_mask_values": sorted(torch.unique(sample_mask).tolist()),
        "band_mean_s1": band_mean[:3].round(6).tolist(),
        "band_std_s1": band_std[:3].round(6).tolist(),
        "band_mean_aef_first5": band_mean[3:8].round(6).tolist() if using_aef(args) else [],
        "band_std_aef_first5": band_std[3:8].round(6).tolist() if using_aef(args) else [],
        "train_batch_images": tuple(batch_images.shape),
        "train_batch_masks": tuple(batch_masks.shape),
        "model_family": args.model_family,
        "segformer_variant": args.segformer_variant if args.model_family == "segformer" else None,
        "loss": args.loss,
        "tuning_strategy": args.tuning_strategy,
        "augmentation_rotate_max_deg": getattr(dls.train.dataset, "rotate_max_deg", None),
        "augmentation_scale_jitter": [
            getattr(dls.train.dataset, "scale_jitter_min", None),
            getattr(dls.train.dataset, "scale_jitter_max", None),
        ],
        "augmentation_s1_gain_jitter": getattr(dls.train.dataset, "s1_gain_jitter", None),
        "augmentation_s1_bias_jitter": getattr(dls.train.dataset, "s1_bias_jitter", None),
        "augmentation_gaussian_noise_std": getattr(dls.train.dataset, "gaussian_noise_std", None),
    }


def run_benchmark_mode(
    args: argparse.Namespace,
    device: torch.device,
    train_records: Sequence[TripletRecord],
    valid_records: Sequence[TripletRecord],
    band_mean: np.ndarray,
    band_std: np.ndarray,
) -> dict:
    planned_full_epochs = effective_epoch_count(args)
    probe_summary = probe_batch_size(args, train_records, valid_records, band_mean, band_std, device)
    args.batch_size = int(probe_summary["selected_batch_size"])
    if args.max_benchmark_batch_size is not None:
        args.batch_size = min(args.batch_size, args.max_benchmark_batch_size)
        probe_summary["selected_batch_size"] = args.batch_size
    train_records = subset_records(train_records, args.max_train_batches, args.batch_size)
    valid_records = subset_records(valid_records, args.max_valid_batches, 1)
    dls = build_dataloaders(args, train_records, valid_records, band_mean, band_std, device)
    print(build_audit_summary(args, train_records, valid_records, band_mean, band_std, dls))

    learn = build_learner(args, dls)
    maybe_warm_start(learn, args)
    epoch_timer = EpochTimerCallback()
    history_df = run_training_schedule(
        learn=learn,
        args=args,
        schedule=build_schedule(args),
        epoch_timer=epoch_timer,
    )
    history_path = args.artifact_dir / args.history_csv
    history_df.to_csv(history_path, index=False)

    validation = validate_model(
        learn,
        tile_size=args.crop_size,
        overlap=args.tile_overlap,
        force_tiled=args.force_tiled_validation,
        max_valid_batches=args.max_valid_batches,
    )
    seconds_per_epoch = sum(epoch_timer.epoch_times) / max(len(epoch_timer.epoch_times), 1)
    samples_per_second = (
        (len(train_records) / seconds_per_epoch) if seconds_per_epoch > 0 else None
    )
    benchmark_summary = {
        "mode": "benchmark",
        "model_family": args.model_family,
        "segformer_variant": args.segformer_variant if args.model_family == "segformer" else None,
        "selected_batch_size": args.batch_size,
        "probe_results": probe_summary["probe_results"],
        "epoch_times_seconds": [round(v, 4) for v in epoch_timer.epoch_times],
        "avg_seconds_per_epoch": round(seconds_per_epoch, 4),
        "samples_per_second": None if samples_per_second is None else round(samples_per_second, 4),
        "projected_full_train_seconds": round(seconds_per_epoch * planned_full_epochs, 1),
        "projected_full_train_hours": round((seconds_per_epoch * planned_full_epochs) / 3600.0, 3),
        "train_records_used": len(train_records),
        "valid_records_used": len(valid_records),
        "validation": validation,
        "history_csv": str(history_path),
        "loss": args.loss,
        "selection_metric": args.selection_metric,
        "tuning_strategy": args.tuning_strategy,
        "force_tiled_validation": bool(args.force_tiled_validation),
        "split_seed": args.seed if args.split_seed is None else args.split_seed,
        "train_subset_seed": args.seed if args.train_subset_seed is None else args.train_subset_seed,
        "max_train_records": args.max_train_records,
    }
    save_json(args.artifact_dir / args.benchmark_summary_json, benchmark_summary)
    print(benchmark_summary)
    return benchmark_summary


def run_train_mode(
    args: argparse.Namespace,
    device: torch.device,
    train_records: Sequence[TripletRecord],
    valid_records: Sequence[TripletRecord],
    band_mean: np.ndarray,
    band_std: np.ndarray,
) -> dict:
    maybe_override_batch_size_from_summary(args)
    train_records = subset_records(train_records, args.max_train_batches, args.batch_size)
    valid_records = subset_records(valid_records, args.max_valid_batches, 1)
    dls = build_dataloaders(args, train_records, valid_records, band_mean, band_std, device)
    print(build_audit_summary(args, train_records, valid_records, band_mean, band_std, dls))

    learn = build_learner(args, dls)
    maybe_warm_start(learn, args)
    history_df = run_training_schedule(learn=learn, args=args, schedule=build_schedule(args))
    history_path = args.artifact_dir / args.history_csv
    history_df.to_csv(history_path, index=False)
    print(history_df.tail())

    if using_aef(args):
        bottleneck_report = inspect_bottleneck(learn.model, args.artifact_dir)
        print({"bottleneck_weight_norms": bottleneck_report["weight_norm_per_proj_band"]})

    validation = validate_model(
        learn,
        tile_size=args.crop_size,
        overlap=args.tile_overlap,
        force_tiled=args.force_tiled_validation,
        max_valid_batches=args.max_valid_batches,
    )
    threshold_results = [
        evaluate_threshold_sweep(
            learn,
            thresholds=args.eval_thresholds,
            tile_size=args.crop_size,
            overlap=args.tile_overlap,
            use_tta=False,
            force_tiled=args.force_tiled_validation,
            max_valid_batches=args.max_valid_batches,
        )
    ]
    if args.use_tta_eval:
        threshold_results.append(
            evaluate_threshold_sweep(
                learn,
                thresholds=args.eval_thresholds,
                tile_size=args.crop_size,
                overlap=args.tile_overlap,
                use_tta=True,
                tta_mode_count=args.tta_eval_mode_count,
                force_tiled=args.force_tiled_validation,
                max_valid_batches=args.max_valid_batches,
            )
        )
    threshold_df = pd.concat(threshold_results, ignore_index=True)
    threshold_path = args.artifact_dir / args.threshold_sweep_csv
    threshold_df.to_csv(threshold_path, index=False)
    print(
        threshold_df.sort_values(
            ["eval_mode", "water_iou", "precision", "threshold"],
            ascending=[True, False, False, True],
        ).groupby("eval_mode").head(5)
    )

    best_eval_row = threshold_df.sort_values(
        ["water_iou", "dice", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    best_eval_mode = str(best_eval_row["eval_mode"])
    tile_metrics_df = evaluate_tiles_at_threshold(
        learn,
        valid_records=valid_records,
        threshold=float(best_eval_row["threshold"]),
        tile_size=args.crop_size,
        overlap=args.tile_overlap,
        use_tta=best_eval_mode.startswith("tta"),
        tta_mode_count=args.tta_eval_mode_count,
        force_tiled=best_eval_mode.endswith("_tiled"),
        max_valid_batches=args.max_valid_batches,
    )
    tile_metrics_path = args.artifact_dir / args.tile_metrics_csv
    tile_metrics_df.to_csv(tile_metrics_path, index=False)
    worst_tiles_df = summarize_worst_tiles(tile_metrics_df, worst_tile_count=args.worst_tile_count)
    worst_tiles_path = args.artifact_dir / args.worst_tiles_csv
    worst_tiles_df.to_csv(worst_tiles_path, index=False)

    summary = {
        "model_family": args.model_family,
        "segformer_variant": args.segformer_variant if args.model_family == "segformer" else None,
        "final_valid_loss": round(float(validation["loss"]), 6),
        "final_valid_jaccard": round(float(validation["jaccard"]), 6),
        "final_valid_dice": round(float(validation["dice"]), 6),
        "final_valid_water_iou": round(float(validation["water_iou"]), 6),
        "final_valid_precision": round(float(validation["precision"]), 6),
        "final_valid_recall": round(float(validation["recall"]), 6),
        "used_tiled_validation": bool(validation["used_tiled_inference"]),
        "best_model_path": str(args.artifact_dir / "models" / "{name}.pth".format(name=args.best_model_name)),
        "history_csv": str(history_path),
        "threshold_sweep_csv": str(threshold_path),
        "tile_metrics_csv": str(tile_metrics_path),
        "worst_tiles_csv": str(worst_tiles_path),
        "loss": args.loss,
        "selection_metric": args.selection_metric,
        "tuning_strategy": args.tuning_strategy,
        "force_tiled_validation": bool(args.force_tiled_validation),
        "split_seed": args.seed if args.split_seed is None else args.split_seed,
        "train_subset_seed": args.seed if args.train_subset_seed is None else args.train_subset_seed,
        "max_train_records": args.max_train_records,
        "train_records_used": len(train_records),
        "valid_records_used": len(valid_records),
    }
    if using_aef(args):
        summary["bottleneck_proj_shape"] = "{aef}->{proj}".format(
            aef=args.n_aef_bands, proj=args.n_proj_bands
        )
    else:
        summary["input_bands"] = args.n_s1_bands
    summary.update(summarize_threshold_results(threshold_df))
    if not worst_tiles_df.empty:
        summary["worst_tile_examples"] = worst_tiles_df[
            ["tile_name", "water_iou", "precision", "recall"]
        ].round(6).to_dict(orient="records")
    save_json(args.artifact_dir / args.eval_summary_json, summary)
    print(summary)
    return summary


def main() -> None:
    args = parse_args()
    device = setup_environment(args)

    if args.triplet_manifest is not None:
        triplets = collect_triplets_from_manifest(
            args.triplet_manifest,
            args.triplet_sample_id,
            expect_aef=using_aef(args),
        )
    else:
        triplets = collect_triplets(
            args.input_dir,
            args.aef_dir if using_aef(args) else None,
            args.label_dir,
        )
    split_seed = args.seed if args.split_seed is None else args.split_seed
    train_subset_seed = args.seed if args.train_subset_seed is None else args.train_subset_seed
    full_train_records, valid_records = split_triplets(triplets, valid_pct=args.valid_pct, seed=split_seed)
    train_records = limit_train_records(
        full_train_records,
        max_records=args.max_train_records,
        subset_seed=train_subset_seed,
    )
    print(
        {
            "split_seed": split_seed,
            "train_subset_seed": train_subset_seed,
            "full_train_tiles": len(full_train_records),
            "train_tiles_after_record_cap": len(train_records),
            "valid_tiles": len(valid_records),
            "max_train_records": args.max_train_records,
        }
    )
    write_split_manifest(args.artifact_dir, train_records, valid_records)
    band_mean, band_std = get_band_stats(
        train_records,
        args.artifact_dir,
        args.stats_sample_size,
        source_stats_path=resolve_band_stats_source(args),
        expected_bands=args.n_s1_bands + args.n_aef_bands,
    )

    if args.mode == "benchmark":
        run_benchmark_mode(args, device, train_records, valid_records, band_mean, band_std)
    else:
        run_train_mode(args, device, train_records, valid_records, band_mean, band_std)


if __name__ == "__main__":
    main()
