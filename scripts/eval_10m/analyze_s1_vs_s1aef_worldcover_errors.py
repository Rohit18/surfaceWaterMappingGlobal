#!/usr/bin/env python3
"""Analyze S1-only vs S1+AEF validation errors by ESA WorldCover class."""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.request import urlopen

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds

from paths import env_root  # noqa: E402  (environment variables; see README.md)


DEFAULT_WORLDCOVER_DIR = env_root("SWM_S1ML") / "intercomparison_s1aef/worldcover/esa_worldcover_2021_v200"

PROB_NODATA = -9999.0
EXPECTED_WORLDCOVER_CODES = {10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100}
WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}
ERROR_CATEGORIES = (
    "aef_fixes_s1_fn",
    "aef_fixes_s1_fp",
    "aef_adds_fn",
    "aef_adds_fp",
)
VALIDATION_GROUPS = ("permanent_water", "not_permanent_water")
CHIP_FIELDS = (
    "sample_id",
    "date",
    "group",
    "validation_group",
    "chip_group",
    "s1_threshold",
    "s1aef_threshold",
    "s1_water_iou",
    "s1aef_water_iou",
    "delta_iou",
    "valid_pixels",
    "label_water_pixels",
    "s1_pred_water_pixels",
    "s1aef_pred_water_pixels",
    "aef_fixes_s1_fn_pixels",
    "aef_fixes_s1_fp_pixels",
    "aef_adds_fn_pixels",
    "aef_adds_fp_pixels",
    "changed_error_pixels",
    "label_path",
    "s1_prob_path",
    "s1aef_prob_path",
    "worldcover_tiles",
)
CLASS_FIELDS = (
    "scope",
    "category",
    "validation_group",
    "chip_group",
    "worldcover_code",
    "worldcover_class",
    "pixel_count",
    "pixel_fraction",
    "chip_count",
    "chip_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--s1-csv", type=Path, required=True)
    parser.add_argument("--s1aef-csv", type=Path, required=True)
    parser.add_argument("--s1-summary", type=Path, required=True)
    parser.add_argument("--s1aef-summary", type=Path, required=True)
    parser.add_argument("--worldcover-dir", type=Path, default=DEFAULT_WORLDCOVER_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="fixed decision threshold for both models (default: each summary's best_threshold_by_water_iou)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be >= 1")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.expanduser().open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_best_threshold(path: Path) -> float:
    summary = json.loads(path.expanduser().read_text())
    try:
        return float(summary["best_threshold_by_water_iou"]["threshold"])
    except KeyError as exc:
        raise KeyError("Missing best_threshold_by_water_iou.threshold in {}".format(path)) from exc


def join_rows(s1_rows: Sequence[Dict[str, str]], s1aef_rows: Sequence[Dict[str, str]]) -> List[Tuple[Dict[str, str], Dict[str, str]]]:
    s1_by_id = {row["sample_id"]: row for row in s1_rows}
    s1aef_by_id = {row["sample_id"]: row for row in s1aef_rows}
    missing_from_aef = sorted(set(s1_by_id).difference(s1aef_by_id))
    missing_from_s1 = sorted(set(s1aef_by_id).difference(s1_by_id))
    if missing_from_aef or missing_from_s1:
        raise ValueError(
            "Sample ID mismatch. Missing from AEF: {}; missing from S1: {}".format(
                missing_from_aef[:10],
                missing_from_s1[:10],
            )
        )
    return [(s1_by_id[sample_id], s1aef_by_id[sample_id]) for sample_id in sorted(s1_by_id)]


def metrics_from_counts(tp: int, fp: int, fn: int) -> float:
    return float(tp / max(tp + fp + fn, 1))


def read_label_on_grid(label_path: Path, ref_ds: DatasetReader) -> np.ndarray:
    with rio.open(label_path) as label_ds:
        if (
            label_ds.crs == ref_ds.crs
            and label_ds.transform == ref_ds.transform
            and label_ds.width == ref_ds.width
            and label_ds.height == ref_ds.height
        ):
            return label_ds.read(1).astype(np.float32)
        with WarpedVRT(
            label_ds,
            crs=ref_ds.crs,
            transform=ref_ds.transform,
            width=ref_ds.width,
            height=ref_ds.height,
            resampling=Resampling.nearest,
        ) as vrt:
            return vrt.read(1).astype(np.float32)


def read_probability_pair(s1_path: Path, s1aef_path: Path, label_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    with rio.open(s1_path) as s1_ds, rio.open(s1aef_path) as aef_ds:
        if (
            s1_ds.crs != aef_ds.crs
            or s1_ds.transform != aef_ds.transform
            or s1_ds.width != aef_ds.width
            or s1_ds.height != aef_ds.height
        ):
            raise ValueError("Probability grids differ: {} vs {}".format(s1_path, s1aef_path))
        s1 = s1_ds.read(1).astype(np.float32)
        aef = aef_ds.read(1).astype(np.float32)
        label = read_label_on_grid(label_path, s1_ds)
        grid = {
            "crs": s1_ds.crs,
            "transform": s1_ds.transform,
            "width": s1_ds.width,
            "height": s1_ds.height,
            "bounds": s1_ds.bounds,
            "s1_nodata": s1_ds.nodata if s1_ds.nodata is not None else PROB_NODATA,
            "aef_nodata": aef_ds.nodata if aef_ds.nodata is not None else PROB_NODATA,
        }
    return s1, aef, label, grid


def tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return "ESA_WorldCover_10m_2021_v200_{}{:02d}{}{:03d}_Map.tif".format(ns, abs(lat), ew, abs(lon))


def worldcover_tile_names(bounds_wgs84: Tuple[float, float, float, float]) -> List[str]:
    west, south, east, north = bounds_wgs84
    lon_starts = range(int(np.floor(west / 3.0) * 3), int(np.floor(east / 3.0) * 3) + 1, 3)
    lat_starts = range(int(np.floor(south / 3.0) * 3), int(np.floor(north / 3.0) * 3) + 1, 3)
    return [tile_name(lat, lon) for lat in lat_starts for lon in lon_starts]


def download_worldcover_tile(name: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s3_uri = "s3://esa-worldcover/v200/2021/map/{}".format(name)
    try:
        subprocess.run(
            ["aws", "s3", "cp", "--no-sign-request", s3_uri, str(out_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    url = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{}".format(name)
    with urlopen(url) as response, tempfile.NamedTemporaryFile(dir=str(out_path.parent), delete=False) as tmp:
        tmp.write(response.read())
        tmp_path = Path(tmp.name)
    tmp_path.replace(out_path)


def ensure_worldcover_tiles(names: Sequence[str], worldcover_dir: Path, skip_download: bool) -> List[Path]:
    paths: List[Path] = []
    missing: List[Path] = []
    for name in names:
        path = worldcover_dir / name
        if path.exists():
            paths.append(path)
        else:
            missing.append(path)
    if missing and skip_download:
        raise FileNotFoundError("Missing WorldCover tiles with --skip-download: {}".format(", ".join(str(p) for p in missing)))
    for path in missing:
        print("Downloading {}".format(path.name), file=sys.stderr)
        download_worldcover_tile(path.name, path)
        paths.append(path)
    return paths


def reproject_worldcover_on_grid(tile_paths: Sequence[Path], grid: Dict[str, Any]) -> np.ndarray:
    worldcover = np.zeros((grid["height"], grid["width"]), dtype=np.uint8)
    tmp = np.zeros_like(worldcover)
    for path in tile_paths:
        tmp.fill(0)
        with rio.open(path) as src:
            reproject(
                source=rio.band(src, 1),
                destination=tmp,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=grid["transform"],
                dst_crs=grid["crs"],
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
        valid = tmp != 0
        worldcover[valid] = tmp[valid]
    return worldcover


def confusion_masks(label_water: np.ndarray, s1_pred: np.ndarray, aef_pred: np.ndarray, valid: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "aef_fixes_s1_fn": label_water & ~s1_pred & aef_pred & valid,
        "aef_fixes_s1_fp": ~label_water & s1_pred & ~aef_pred & valid,
        "aef_adds_fn": label_water & s1_pred & ~aef_pred & valid,
        "aef_adds_fp": ~label_water & ~s1_pred & aef_pred & valid,
    }


def chip_group(delta_iou: float) -> str:
    if delta_iou > 0.05:
        return "aef_helps"
    if delta_iou < -0.05:
        return "aef_hurts"
    return "neutral"


def validation_group(row_group: str) -> str:
    if row_group == "sid_permanent_water":
        return "permanent_water"
    return "not_permanent_water"


def add_class_counts(
    totals: Dict[Tuple[str, str, str, int], Counter],
    category: str,
    val_group: str,
    group: str,
    wc: np.ndarray,
    mask: np.ndarray,
) -> None:
    values, counts = np.unique(wc[mask], return_counts=True)
    for value, count in zip(values.tolist(), counts.tolist()):
        code = int(value)
        if code == 0:
            continue
        key = (category, val_group, group, code)
        totals[key]["pixels"] += int(count)
        totals[key]["chips"] += 1


def class_rows_from_totals(totals: Dict[Tuple[str, str, str, int], Counter], scope: str) -> List[Dict[str, Any]]:
    category_pixel_totals = Counter()
    category_chip_totals = Counter()
    for (category, val_group, group, _code), counts in totals.items():
        category_pixel_totals[(category, val_group, group)] += int(counts["pixels"])
        category_chip_totals[(category, val_group, group)] += int(counts["chips"])

    rows: List[Dict[str, Any]] = []
    for (category, val_group, group, code), counts in sorted(totals.items()):
        pixel_total = category_pixel_totals[(category, val_group, group)]
        chip_total = category_chip_totals[(category, val_group, group)]
        rows.append(
            {
                "scope": scope,
                "category": category,
                "validation_group": val_group,
                "chip_group": group,
                "worldcover_code": code,
                "worldcover_class": WORLDCOVER_CLASSES.get(code, "Unknown"),
                "pixel_count": int(counts["pixels"]),
                "pixel_fraction": int(counts["pixels"]) / max(pixel_total, 1),
                "chip_count": int(counts["chips"]),
                "chip_fraction": int(counts["chips"]) / max(chip_total, 1),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: float) -> str:
    return "{:.4f}".format(float(value))


def top_classes(class_rows: Sequence[Dict[str, Any]], category: str, group: str = "all", n: int = 5) -> List[Dict[str, Any]]:
    rows = [row for row in class_rows if row["category"] == category and row["chip_group"] == group and row["validation_group"] == "all"]
    return sorted(rows, key=lambda row: (int(row["pixel_count"]), int(row["chip_count"])), reverse=True)[:n]


def top_classes_for_validation_group(
    class_rows: Sequence[Dict[str, Any]],
    category: str,
    val_group: str,
    group: str = "all",
    n: int = 5,
) -> List[Dict[str, Any]]:
    rows = [
        row
        for row in class_rows
        if row["category"] == category and row["validation_group"] == val_group and row["chip_group"] == group
    ]
    return sorted(rows, key=lambda row: (int(row["pixel_count"]), int(row["chip_count"])), reverse=True)[:n]


def summarize_chip_rows(chip_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not chip_rows:
        return {
            "sample_count": 0,
            "mean_s1_water_iou": 0.0,
            "mean_s1aef_water_iou": 0.0,
            "mean_delta_iou": 0.0,
            "median_delta_iou": 0.0,
            "chip_groups": {},
            "changed_error_totals": {},
        }
    return {
        "sample_count": len(chip_rows),
        "mean_s1_water_iou": float(np.mean([float(row["s1_water_iou"]) for row in chip_rows])),
        "mean_s1aef_water_iou": float(np.mean([float(row["s1aef_water_iou"]) for row in chip_rows])),
        "mean_delta_iou": float(np.mean([float(row["delta_iou"]) for row in chip_rows])),
        "median_delta_iou": float(np.median([float(row["delta_iou"]) for row in chip_rows])),
        "chip_groups": dict(Counter(row["chip_group"] for row in chip_rows)),
        "changed_error_totals": {
            category: int(sum(int(row["{}_pixels".format(category)]) for row in chip_rows))
            for category in ERROR_CATEGORIES
        },
    }


def render_report(summary: Dict[str, Any], chip_rows: Sequence[Dict[str, Any]], class_rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# S1-Only vs S1+AEF WorldCover Error Analysis",
        "",
        "Created: `{}`".format(summary["created_at"]),
        "",
        "## Setup",
        "",
        "- S1-only: `{}` at threshold `{}`".format(summary["inputs"]["s1_csv"], summary["thresholds"]["s1"]),
        "- S1+AEF: `{}` at threshold `{}`".format(summary["inputs"]["s1aef_csv"], summary["thresholds"]["s1aef"]),
        "- Processed chips: `{}`".format(summary["sample_count"]),
        "- ESA WorldCover cache: `{}`".format(summary["inputs"]["worldcover_dir"]),
        "- Validation groups: `permanent_water` = SID labels, `not_permanent_water` = non-SID post-flood labels",
        "",
        "## Chip Delta IoU",
        "",
        "- Mean S1 Water IoU: `{}`".format(fmt(summary["chip_iou"]["mean_s1_water_iou"])),
        "- Mean S1+AEF Water IoU: `{}`".format(fmt(summary["chip_iou"]["mean_s1aef_water_iou"])),
        "- Mean delta IoU: `{}`".format(fmt(summary["chip_iou"]["mean_delta_iou"])),
        "- Groups: `{}` helps, `{}` neutral, `{}` hurts".format(
            summary["chip_groups"].get("aef_helps", 0),
            summary["chip_groups"].get("neutral", 0),
            summary["chip_groups"].get("aef_hurts", 0),
        ),
        "",
        "## Validation Group Summary",
        "",
        "| Validation group | Chips | S1 IoU | S1+AEF IoU | Mean delta | Helps | Neutral | Hurts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for val_group in VALIDATION_GROUPS:
        item = summary["validation_groups"][val_group]
        counts = item["chip_groups"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                val_group,
                item["sample_count"],
                fmt(item["mean_s1_water_iou"]),
                fmt(item["mean_s1aef_water_iou"]),
                fmt(item["mean_delta_iou"]),
                counts.get("aef_helps", 0),
                counts.get("neutral", 0),
                counts.get("aef_hurts", 0),
            )
        )
    lines.extend(
        [
            "",
        "## WorldCover Classes",
        "",
        ]
    )
    labels = {
        "aef_fixes_s1_fn": "AEF fixes S1 false negatives",
        "aef_fixes_s1_fp": "AEF fixes S1 false positives",
        "aef_adds_fn": "AEF adds false negatives",
        "aef_adds_fp": "AEF adds false positives",
    }
    for val_group in VALIDATION_GROUPS:
        lines.append("### {}".format(val_group))
        val_summary = summary["validation_groups"][val_group]
        lines.extend(
            [
                "",
                "- Chips: `{}`".format(val_summary["sample_count"]),
                "- Mean delta IoU: `{}`".format(fmt(val_summary["mean_delta_iou"])),
                "- Groups: `{}` helps, `{}` neutral, `{}` hurts".format(
                    val_summary["chip_groups"].get("aef_helps", 0),
                    val_summary["chip_groups"].get("neutral", 0),
                    val_summary["chip_groups"].get("aef_hurts", 0),
                ),
                "",
            ]
        )
        for category in ERROR_CATEGORIES:
            lines.append("#### {}".format(labels[category]))
            rows = top_classes_for_validation_group(class_rows, category, val_group)
            if not rows:
                lines.append("")
                lines.append("No pixels in this category.")
                lines.append("")
                continue
            lines.extend(["", "| Class | Pixels | Pixel fraction | Chips |", "| --- | ---: | ---: | ---: |"])
            for row in rows:
                lines.append(
                    "| {} | {} | {} | {} |".format(
                        row["worldcover_class"],
                        row["pixel_count"],
                        fmt(row["pixel_fraction"]),
                        row["chip_count"],
                    )
                )
            lines.append("")

    def chip_table(title: str, rows: Sequence[Dict[str, Any]]) -> None:
        lines.append("## {}".format(title))
        lines.extend(["", "| Sample | Group | S1 IoU | S1+AEF IoU | Delta |", "| --- | --- | ---: | ---: | ---: |"])
        for row in rows:
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    row["sample_id"],
                    row["chip_group"],
                    fmt(row["s1_water_iou"]),
                    fmt(row["s1aef_water_iou"]),
                    fmt(row["delta_iou"]),
                )
            )
        lines.append("")

    chip_table("Top 10 AEF Improvements", summary["top_10_delta_iou"])
    chip_table("Bottom 10 AEF Regressions", summary["bottom_10_delta_iou"])
    for val_group in VALIDATION_GROUPS:
        chip_table(
            "Top AEF Improvements: {}".format(val_group),
            sorted(
                [row for row in chip_rows if row["validation_group"] == val_group],
                key=lambda row: float(row["delta_iou"]),
                reverse=True,
            )[:5],
        )
        chip_table(
            "Worst AEF Regressions: {}".format(val_group),
            sorted(
                [row for row in chip_rows if row["validation_group"] == val_group],
                key=lambda row: float(row["delta_iou"]),
            )[:5],
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError("{} exists and is not empty; pass --overwrite".format(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_threshold = read_best_threshold(args.s1_summary) if args.threshold is None else args.threshold
    aef_threshold = read_best_threshold(args.s1aef_summary) if args.threshold is None else args.threshold
    joined = join_rows(read_csv_rows(args.s1_csv), read_csv_rows(args.s1aef_csv))
    if args.max_samples is not None:
        joined = joined[: args.max_samples]

    chip_rows: List[Dict[str, Any]] = []
    overall_class_totals: Dict[Tuple[str, str, str, int], Counter] = defaultdict(Counter)
    valid_worldcover_codes = Counter()
    changed_error_totals = Counter()
    missing_or_unexpected_worldcover = Counter()

    for index, (s1_row, aef_row) in enumerate(joined, start=1):
        sample_id = s1_row["sample_id"]
        label_path = Path(s1_row["label_path"]).expanduser()
        s1_prob_path = Path(s1_row["prob_path"]).expanduser()
        aef_prob_path = Path(aef_row["prob_path"]).expanduser()
        s1_prob, aef_prob, label, grid = read_probability_pair(s1_prob_path, aef_prob_path, label_path)

        valid = (
            np.isfinite(s1_prob)
            & np.isfinite(aef_prob)
            & (s1_prob != grid["s1_nodata"])
            & (aef_prob != grid["aef_nodata"])
        )
        label_water = label > 0.5
        s1_pred = s1_prob >= s1_threshold
        aef_pred = aef_prob >= aef_threshold

        s1_tp = int((s1_pred & label_water & valid).sum())
        s1_fp = int((s1_pred & ~label_water & valid).sum())
        s1_fn = int((~s1_pred & label_water & valid).sum())
        aef_tp = int((aef_pred & label_water & valid).sum())
        aef_fp = int((aef_pred & ~label_water & valid).sum())
        aef_fn = int((~aef_pred & label_water & valid).sum())
        s1_iou = metrics_from_counts(s1_tp, s1_fp, s1_fn)
        aef_iou = metrics_from_counts(aef_tp, aef_fp, aef_fn)
        delta = aef_iou - s1_iou
        group = chip_group(delta)
        val_group = validation_group(s1_row["group"])

        bounds_wgs84 = transform_bounds(grid["crs"], "EPSG:4326", *grid["bounds"], densify_pts=21)
        tile_names = worldcover_tile_names(bounds_wgs84)
        tile_paths = ensure_worldcover_tiles(tile_names, args.worldcover_dir.expanduser(), args.skip_download)
        wc = reproject_worldcover_on_grid(tile_paths, grid)
        if wc.shape != s1_prob.shape:
            raise ValueError("WorldCover shape mismatch for {}: {} vs {}".format(sample_id, wc.shape, s1_prob.shape))
        unexpected = set(np.unique(wc[valid]).astype(int).tolist()).difference(EXPECTED_WORLDCOVER_CODES | {0})
        for code in unexpected:
            missing_or_unexpected_worldcover[int(code)] += int((wc == code).sum())
        for code, count in zip(*np.unique(wc[valid], return_counts=True)):
            valid_worldcover_codes[int(code)] += int(count)

        masks = confusion_masks(label_water, s1_pred, aef_pred, valid)
        changed_stack = np.zeros_like(valid, dtype=np.uint8)
        for category, mask in masks.items():
            changed_stack += mask.astype(np.uint8)
            pixel_count = int(mask.sum())
            changed_error_totals[category] += pixel_count
            add_class_counts(overall_class_totals, category, "all", "all", wc, mask)
            add_class_counts(overall_class_totals, category, "all", group, wc, mask)
            add_class_counts(overall_class_totals, category, val_group, "all", wc, mask)
            add_class_counts(overall_class_totals, category, val_group, group, wc, mask)
        if changed_stack.size and int(changed_stack.max()) > 1:
            raise ValueError("Changed-error masks are not mutually exclusive for {}".format(sample_id))
        changed_error_pixels = int(changed_stack.sum())
        if changed_error_pixels > int(valid.sum()):
            raise ValueError("Changed-error pixels exceed valid pixels for {}".format(sample_id))

        chip_rows.append(
            {
                "sample_id": sample_id,
                "date": s1_row["date"],
                "group": s1_row["group"],
                "validation_group": val_group,
                "chip_group": group,
                "s1_threshold": s1_threshold,
                "s1aef_threshold": aef_threshold,
                "s1_water_iou": s1_iou,
                "s1aef_water_iou": aef_iou,
                "delta_iou": delta,
                "valid_pixels": int(valid.sum()),
                "label_water_pixels": int((label_water & valid).sum()),
                "s1_pred_water_pixels": int((s1_pred & valid).sum()),
                "s1aef_pred_water_pixels": int((aef_pred & valid).sum()),
                "aef_fixes_s1_fn_pixels": int(masks["aef_fixes_s1_fn"].sum()),
                "aef_fixes_s1_fp_pixels": int(masks["aef_fixes_s1_fp"].sum()),
                "aef_adds_fn_pixels": int(masks["aef_adds_fn"].sum()),
                "aef_adds_fp_pixels": int(masks["aef_adds_fp"].sum()),
                "changed_error_pixels": changed_error_pixels,
                "label_path": str(label_path),
                "s1_prob_path": str(s1_prob_path),
                "s1aef_prob_path": str(aef_prob_path),
                "worldcover_tiles": ";".join(tile_names),
            }
        )
        print("Processed {}/{} {}".format(index, len(joined), sample_id), file=sys.stderr)

    class_rows = class_rows_from_totals(overall_class_totals, "changed_errors")
    chip_rows_sorted = sorted(chip_rows, key=lambda row: (str(row["date"]), str(row["sample_id"])))
    top_10 = sorted(chip_rows, key=lambda row: float(row["delta_iou"]), reverse=True)[:10]
    bottom_10 = sorted(chip_rows, key=lambda row: float(row["delta_iou"]))[:10]
    group_counts = Counter(row["chip_group"] for row in chip_rows)
    validation_group_summaries = {
        val_group: summarize_chip_rows([row for row in chip_rows if row["validation_group"] == val_group])
        for val_group in VALIDATION_GROUPS
    }
    summary = {
        "created_at": utc_now(),
        "sample_count": len(chip_rows),
        "inputs": {
            "s1_csv": str(args.s1_csv),
            "s1aef_csv": str(args.s1aef_csv),
            "s1_summary": str(args.s1_summary),
            "s1aef_summary": str(args.s1aef_summary),
            "worldcover_dir": str(args.worldcover_dir),
        },
        "thresholds": {
            "s1": s1_threshold,
            "s1aef": aef_threshold,
        },
        "chip_iou": {
            "mean_s1_water_iou": float(np.mean([float(row["s1_water_iou"]) for row in chip_rows])) if chip_rows else 0.0,
            "mean_s1aef_water_iou": float(np.mean([float(row["s1aef_water_iou"]) for row in chip_rows])) if chip_rows else 0.0,
            "mean_delta_iou": float(np.mean([float(row["delta_iou"]) for row in chip_rows])) if chip_rows else 0.0,
            "median_delta_iou": float(np.median([float(row["delta_iou"]) for row in chip_rows])) if chip_rows else 0.0,
        },
        "chip_groups": dict(group_counts),
        "validation_groups": validation_group_summaries,
        "changed_error_totals": dict(changed_error_totals),
        "worldcover_valid_pixel_codes": dict(sorted(valid_worldcover_codes.items())),
        "unexpected_worldcover_codes": dict(sorted(missing_or_unexpected_worldcover.items())),
        "top_10_delta_iou": top_10,
        "bottom_10_delta_iou": bottom_10,
        "outputs": {
            "summary": str(out_dir / "summary.json"),
            "chip_delta_iou": str(out_dir / "chip_delta_iou.csv"),
            "worldcover_error_classes": str(out_dir / "worldcover_error_classes.csv"),
            "report": str(out_dir / "report.md"),
        },
    }

    write_csv(out_dir / "chip_delta_iou.csv", chip_rows_sorted, CHIP_FIELDS)
    write_csv(out_dir / "worldcover_error_classes.csv", class_rows, CLASS_FIELDS)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "report.md").write_text(render_report(summary, chip_rows_sorted, class_rows))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
