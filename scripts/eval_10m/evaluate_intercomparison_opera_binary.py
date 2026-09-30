#!/usr/bin/env python3
"""Compare intercomparison labels with binary OPERA DSWx-S1 water masks.

The OPERA archive is expected to contain paired ``B01_WTR`` and
``B02_BWTR`` rasters. This script uses ``B02_BWTR`` as the binary OPERA
water product and warps it onto each label grid before computing confusion
metrics. OPERA values are interpreted as:

* 1: water
* 0: observed non-water
* 250, 251, 255, nodata: ignored
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds


DEFAULT_CSV = Path("data/S1_Intercomparison_All_Scenes.csv")
DEFAULT_LABEL_DIR = Path("data/S1_Intercomparison_All_Scenes_subset/labels")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPERA_ROOT = REPOSITORY_ROOT / "data" / "OPERA_WTR_BWTR"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "intercomparison_opera"
DEFAULT_TEMPORAL_TOLERANCE_HOURS = 24.0
DEFAULT_SAME_PASS_SECONDS = 180.0
OPERA_NODATA = 255
OPERA_VALID_VALUES = (0, 1)
OPERA_WATER_VALUES = (1,)
OPERA_BINARY_FLOAT_NODATA = -9999.0
OPERA_RE = re.compile(
    r"^OPERA_L3_DSWx-S1_(?P<tile>T[^_]+)_(?P<acq>\d{8}T\d{6}Z)_"
    r"(?P<proc>\d{8}T\d{6}Z)_(?P<sat>S1[AB])_30_v(?P<version>[^_]+)_B02_BWTR\.tif$"
)

INDEX_FIELDS = (
    "path",
    "filename",
    "tile",
    "acquisition_time",
    "processing_time",
    "satellite",
    "version",
    "crs",
    "width",
    "height",
    "nodata",
    "left",
    "bottom",
    "right",
    "top",
    "wgs84_left",
    "wgs84_bottom",
    "wgs84_right",
    "wgs84_top",
)
MATCH_FIELDS = (
    "sample_id",
    "status",
    "label_path",
    "reference_date",
    "s1_acquisition_time",
    "opera_match_count",
    "opera_time_diff_seconds",
    "opera_files",
    "binary_mask_path",
    "valid_pixels",
    "label_water_pixels",
    "opera_water_pixels",
    "tp",
    "fp",
    "tn",
    "fn",
    "precision",
    "recall",
    "specificity",
    "water_iou",
    "dice",
    "accuracy",
    "balanced_accuracy",
    "error",
)


@dataclass(frozen=True)
class Label:
    sample_id: str
    path: Path
    reference_date: str
    s1_acquisition_time: datetime
    crs: Any
    transform: Any
    width: int
    height: int
    bounds: Tuple[float, float, float, float]
    wgs84_bounds: Tuple[float, float, float, float]


@dataclass(frozen=True)
class OperaFile:
    path: Path
    filename: str
    tile: str
    acquisition_time: datetime
    processing_time: str
    satellite: str
    version: str
    crs: str
    width: int
    height: int
    nodata: Optional[float]
    bounds: Tuple[float, float, float, float]
    wgs84_bounds: Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--opera-root", type=Path, default=DEFAULT_OPERA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-csv", type=Path, default=None, help="Defaults to OUTPUT_DIR/manifests/opera_b02_index.csv.")
    parser.add_argument("--temporal-tolerance-hours", type=float, default=DEFAULT_TEMPORAL_TOLERANCE_HOURS)
    parser.add_argument("--same-pass-seconds", type=float, default=DEFAULT_SAME_PASS_SECONDS)
    parser.add_argument("--resampling", choices=("nearest", "bilinear"), default="nearest")
    parser.add_argument(
        "--water-threshold",
        type=float,
        default=0.5,
        help="Threshold after resampling a binary OPERA water score to the label grid.",
    )
    parser.add_argument("--only-sample", action="append", default=None, help="Sample ID to process; may be repeated.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--write-masks", action="store_true", default=True)
    parser.add_argument("--no-write-masks", dest="write_masks", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.temporal_tolerance_hours < 0:
        parser.error("--temporal-tolerance-hours must be >= 0")
    if args.same_pass_seconds < 0:
        parser.error("--same-pass-seconds must be >= 0")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be >= 1")
    if args.water_threshold < 0 or args.water_threshold > 1:
        parser.error("--water-threshold must be in [0, 1]")

    args.csv = args.csv.expanduser().resolve()
    args.label_dir = args.label_dir.expanduser().resolve()
    args.opera_root = args.opera_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.index_csv = (args.index_csv or args.output_dir / "manifests" / "opera_b02_index.csv").expanduser().resolve()
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_opera_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def parse_csv_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def bbox_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError("CSV does not exist: {}".format(path))
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"SampleID", "SampleID_clean", "Date", "acquisition_datetime"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError("CSV is missing required columns: {}".format(", ".join(sorted(missing))))
    return rows


def load_labels(csv_path: Path, label_dir: Path, only_sample: Optional[Sequence[str]], max_samples: Optional[int]) -> List[Label]:
    lookup: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(csv_path):
        for key in (row.get("SampleID_clean", ""), row.get("SampleID", "")):
            if key and key not in lookup:
                lookup[key] = row

    selected_paths = sorted(label_dir.glob("*.tif"))
    if only_sample:
        wanted = set(only_sample)
        selected_paths = [path for path in selected_paths if path.stem in wanted]
        missing = sorted(wanted.difference(path.stem for path in selected_paths))
        if missing:
            raise ValueError("--only-sample not found in label directory: {}".format(", ".join(missing)))
    if max_samples is not None:
        selected_paths = selected_paths[:max_samples]

    labels: List[Label] = []
    missing_rows: List[str] = []
    for path in selected_paths:
        row = lookup.get(path.stem)
        if row is None:
            missing_rows.append(path.stem)
            continue
        with rio.open(path) as src:
            if src.crs is None:
                raise ValueError("Label is missing CRS: {}".format(path))
            bounds = tuple(float(value) for value in src.bounds)
            wgs84 = tuple(float(value) for value in transform_bounds(src.crs, "EPSG:4326", *bounds, densify_pts=21))
            labels.append(
                Label(
                    sample_id=path.stem,
                    path=path,
                    reference_date=row["Date"],
                    s1_acquisition_time=parse_csv_time(row["acquisition_datetime"]),
                    crs=src.crs,
                    transform=src.transform,
                    width=int(src.width),
                    height=int(src.height),
                    bounds=bounds,  # type: ignore[arg-type]
                    wgs84_bounds=wgs84,  # type: ignore[arg-type]
                )
            )
    if missing_rows:
        raise ValueError("Labels missing from CSV: {}".format(", ".join(missing_rows)))
    return labels


def opera_from_path(path: Path) -> OperaFile:
    match = OPERA_RE.match(path.name)
    if not match:
        raise ValueError("Unexpected OPERA B02 filename: {}".format(path.name))
    with rio.open(path) as src:
        if src.crs is None:
            raise ValueError("OPERA file is missing CRS: {}".format(path))
        bounds = tuple(float(value) for value in src.bounds)
        wgs84 = tuple(float(value) for value in transform_bounds(src.crs, "EPSG:4326", *bounds, densify_pts=21))
        return OperaFile(
            path=path,
            filename=path.name,
            tile=match.group("tile"),
            acquisition_time=parse_opera_time(match.group("acq")),
            processing_time=match.group("proc"),
            satellite=match.group("sat"),
            version=match.group("version"),
            crs=src.crs.to_string(),
            width=int(src.width),
            height=int(src.height),
            nodata=src.nodata,
            bounds=bounds,  # type: ignore[arg-type]
            wgs84_bounds=wgs84,  # type: ignore[arg-type]
        )


def opera_to_row(opera: OperaFile) -> Dict[str, Any]:
    return {
        "path": str(opera.path),
        "filename": opera.filename,
        "tile": opera.tile,
        "acquisition_time": opera.acquisition_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "processing_time": opera.processing_time,
        "satellite": opera.satellite,
        "version": opera.version,
        "crs": opera.crs,
        "width": opera.width,
        "height": opera.height,
        "nodata": "" if opera.nodata is None else opera.nodata,
        "left": opera.bounds[0],
        "bottom": opera.bounds[1],
        "right": opera.bounds[2],
        "top": opera.bounds[3],
        "wgs84_left": opera.wgs84_bounds[0],
        "wgs84_bottom": opera.wgs84_bounds[1],
        "wgs84_right": opera.wgs84_bounds[2],
        "wgs84_top": opera.wgs84_bounds[3],
    }


def opera_from_row(row: Dict[str, str]) -> OperaFile:
    nodata = row.get("nodata", "")
    return OperaFile(
        path=Path(row["path"]),
        filename=row["filename"],
        tile=row["tile"],
        acquisition_time=datetime.strptime(row["acquisition_time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc),
        processing_time=row["processing_time"],
        satellite=row["satellite"],
        version=row["version"],
        crs=row["crs"],
        width=int(row["width"]),
        height=int(row["height"]),
        nodata=float(nodata) if nodata != "" else None,
        bounds=(float(row["left"]), float(row["bottom"]), float(row["right"]), float(row["top"])),
        wgs84_bounds=(
            float(row["wgs84_left"]),
            float(row["wgs84_bottom"]),
            float(row["wgs84_right"]),
            float(row["wgs84_top"]),
        ),
    )


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_or_build_opera_index(opera_root: Path, index_csv: Path, rebuild: bool) -> List[OperaFile]:
    if index_csv.exists() and not rebuild:
        with index_csv.open(newline="") as handle:
            return [opera_from_row(row) for row in csv.DictReader(handle)]

    paths = sorted(opera_root.glob("*_B02_BWTR.tif"))
    if not paths:
        raise FileNotFoundError("No *_B02_BWTR.tif files found in {}".format(opera_root))
    files = [opera_from_path(path) for path in paths]
    write_csv(index_csv, [opera_to_row(file) for file in files], INDEX_FIELDS)
    return files


def keep_latest_processing_version(opera_files: Iterable[OperaFile]) -> List[OperaFile]:
    latest: Dict[Tuple[str, datetime, str], OperaFile] = {}
    for opera in opera_files:
        key = (opera.tile, opera.acquisition_time, opera.satellite)
        current = latest.get(key)
        if current is None or opera.processing_time > current.processing_time:
            latest[key] = opera
    return sorted(latest.values(), key=lambda opera: (opera.acquisition_time, opera.tile, opera.satellite, opera.filename))


def select_opera_matches(label: Label, opera_files: Sequence[OperaFile], tolerance_seconds: float, same_pass_seconds: float) -> List[OperaFile]:
    spatial = [opera for opera in opera_files if bbox_intersects(label.wgs84_bounds, opera.wgs84_bounds)]
    temporal = [
        opera
        for opera in spatial
        if abs((opera.acquisition_time - label.s1_acquisition_time).total_seconds()) <= tolerance_seconds
    ]
    if not temporal:
        return []
    best_diff = min(abs((opera.acquisition_time - label.s1_acquisition_time).total_seconds()) for opera in temporal)
    return [
        opera
        for opera in temporal
        if abs((opera.acquisition_time - label.s1_acquisition_time).total_seconds()) <= best_diff + same_pass_seconds
    ]


def read_label_water(label: Label) -> np.ndarray:
    with rio.open(label.path) as src:
        arr = src.read(1)
        nodata = src.nodata
    valid = np.ones(arr.shape, dtype=bool)
    if nodata is not None:
        valid &= arr != nodata
    return (arr > 0) & valid


def read_opera_on_label_grid(opera: OperaFile, label: Label) -> np.ndarray:
    with rio.open(opera.path) as src:
        with WarpedVRT(
            src,
            crs=label.crs,
            transform=label.transform,
            width=label.width,
            height=label.height,
            resampling=Resampling.nearest,
            nodata=OPERA_NODATA,
        ) as vrt:
            return vrt.read(1)


def read_opera_binary_score_on_label_grid(opera: OperaFile, label: Label, resampling: Resampling) -> np.ndarray:
    with rio.open(opera.path) as src:
        arr = src.read(1)
        valid = np.isin(arr, OPERA_VALID_VALUES)
        source = np.full(arr.shape, OPERA_BINARY_FLOAT_NODATA, dtype=np.float32)
        source[valid] = 0.0
        source[np.isin(arr, OPERA_WATER_VALUES) & valid] = 1.0
        dst = np.full((label.height, label.width), OPERA_BINARY_FLOAT_NODATA, dtype=np.float32)
        reproject(
            source=source,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=OPERA_BINARY_FLOAT_NODATA,
            dst_transform=label.transform,
            dst_crs=label.crs,
            dst_nodata=OPERA_BINARY_FLOAT_NODATA,
            resampling=resampling,
        )
        return dst


def merged_binary_opera(label: Label, matches: Sequence[OperaFile], resampling: str, water_threshold: float) -> np.ndarray:
    if resampling == "bilinear":
        return merged_resampled_binary_opera(label, matches, Resampling.bilinear, water_threshold)

    water = np.zeros((label.height, label.width), dtype=bool)
    valid = np.zeros((label.height, label.width), dtype=bool)
    for opera in matches:
        arr = read_opera_on_label_grid(opera, label)
        this_valid = np.isin(arr, OPERA_VALID_VALUES)
        valid |= this_valid
        water |= np.isin(arr, OPERA_WATER_VALUES) & this_valid
    out = np.full((label.height, label.width), OPERA_NODATA, dtype=np.uint8)
    out[valid] = 0
    out[water] = 1
    return out


def merged_resampled_binary_opera(
    label: Label,
    matches: Sequence[OperaFile],
    resampling: Resampling,
    water_threshold: float,
) -> np.ndarray:
    water = np.zeros((label.height, label.width), dtype=bool)
    valid = np.zeros((label.height, label.width), dtype=bool)
    for opera in matches:
        score = read_opera_binary_score_on_label_grid(opera, label, resampling)
        this_valid = score != OPERA_BINARY_FLOAT_NODATA
        valid |= this_valid
        water |= (score >= water_threshold) & this_valid
    out = np.full((label.height, label.width), OPERA_NODATA, dtype=np.uint8)
    out[valid] = 0
    out[water] = 1
    return out


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    water_iou = tp / max(tp + fp + fn, 1)
    dice = (2 * tp) / max((2 * tp) + fp + fn, 1)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    balanced_accuracy = (recall + specificity) / 2.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "water_iou": float(water_iou),
        "dice": float(dice),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
    }


def compare(label: Label, opera_binary: np.ndarray) -> Dict[str, Any]:
    label_water = read_label_water(label)
    valid = opera_binary != OPERA_NODATA
    opera_water = opera_binary == 1
    tp = int((opera_water & label_water & valid).sum())
    fp = int((opera_water & ~label_water & valid).sum())
    tn = int((~opera_water & ~label_water & valid).sum())
    fn = int((~opera_water & label_water & valid).sum())
    row: Dict[str, Any] = {
        "valid_pixels": int(valid.sum()),
        "label_water_pixels": int((label_water & valid).sum()),
        "opera_water_pixels": int((opera_water & valid).sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    row.update(metrics_from_counts(tp, fp, tn, fn))
    return row


def write_binary_mask(path: Path, label: Label, arr: np.ndarray, overwrite: bool, resampling: str) -> None:
    if path.exists() and not overwrite:
        return
    profile = {
        "driver": "GTiff",
        "height": label.height,
        "width": label.width,
        "count": 1,
        "dtype": "uint8",
        "crs": label.crs,
        "transform": label.transform,
        "nodata": OPERA_NODATA,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.update_tags(
            1,
            description="Binary OPERA DSWx-S1 B02_BWTR {}-resampled to label grid; 1=water, 0=observed non-water, 255=nodata/ignored.".format(
                resampling
            ),
        )


def round_floats(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for key, value in list(out.items()):
        if isinstance(value, float):
            out[key] = round(value, 6)
    return out


def summarize(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    matched_rows = [row for row in rows if row.get("status") == "ok"]
    count_rows = [row for row in rows if row.get("status") == "ok" and all(key in row for key in ("tp", "fp", "tn", "fn"))]
    counts = Counter({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    for row in count_rows:
        counts.update({key: int(row[key]) for key in ("tp", "fp", "tn", "fn")})
    tp, fp, tn, fn = (int(counts[key]) for key in ("tp", "fp", "tn", "fn"))
    metrics = metrics_from_counts(tp, fp, tn, fn)
    return {
        "created_at": utc_now(),
        "csv": str(args.csv),
        "label_dir": str(args.label_dir),
        "opera_root": str(args.opera_root),
        "output_dir": str(args.output_dir),
        "temporal_tolerance_hours": args.temporal_tolerance_hours,
        "same_pass_seconds": args.same_pass_seconds,
        "resampling": args.resampling,
        "water_threshold": args.water_threshold,
        "label_count": len(rows),
        "opera_index_file_count": getattr(args, "opera_index_file_count", None),
        "opera_latest_version_file_count": getattr(args, "opera_latest_version_file_count", None),
        "matched_label_count": len(matched_rows),
        "metric_label_count": len(count_rows),
        "unmatched_label_count": sum(row.get("status") == "no_match" for row in rows),
        "error_count": sum(row.get("status") == "error" for row in rows),
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "valid_pixels": tp + fp + tn + fn},
        "metrics": {key: round(value, 6) for key, value in metrics.items()},
        "confusion_matrix": {
            "rows": "label",
            "columns": "opera_binary",
            "labels": ["other", "water"],
            "matrix": [[tn, fp], [fn, tp]],
        },
        "outputs": {
            "matches": str(args.output_dir / "manifests" / "opera_label_matches.csv"),
            "summary": str(args.output_dir / "manifests" / "opera_label_comparison_summary.json"),
            "binary_masks": str(args.output_dir / "binary_masks"),
            "opera_index": str(args.index_csv),
        },
    }


def main() -> None:
    args = parse_args()
    labels = load_labels(args.csv, args.label_dir, args.only_sample, args.max_samples)
    opera_index_files = load_or_build_opera_index(args.opera_root, args.index_csv, args.rebuild_index)
    opera_files = keep_latest_processing_version(opera_index_files)
    args.opera_index_file_count = len(opera_index_files)
    args.opera_latest_version_file_count = len(opera_files)
    tolerance_seconds = args.temporal_tolerance_hours * 3600.0
    rows: List[Dict[str, Any]] = []

    for label in labels:
        row: Dict[str, Any] = {
            "sample_id": label.sample_id,
            "label_path": str(label.path),
            "reference_date": label.reference_date,
            "s1_acquisition_time": label.s1_acquisition_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "ok",
            "error": "",
        }
        try:
            matches = select_opera_matches(label, opera_files, tolerance_seconds, args.same_pass_seconds)
            row["opera_match_count"] = len(matches)
            row["opera_files"] = ";".join(str(opera.path) for opera in matches)
            if matches:
                row["opera_time_diff_seconds"] = min(
                    abs((opera.acquisition_time - label.s1_acquisition_time).total_seconds()) for opera in matches
                )
            if not matches:
                row["status"] = "no_match"
                rows.append(row)
                continue
            if args.dry_run:
                rows.append(row)
                continue
            opera_binary = merged_binary_opera(label, matches, args.resampling, args.water_threshold)
            mask_path = args.output_dir / "binary_masks" / "{}_opera_bwtr_binary.tif".format(label.sample_id)
            if args.write_masks:
                write_binary_mask(mask_path, label, opera_binary, args.overwrite, args.resampling)
            row["binary_mask_path"] = str(mask_path)
            row.update(compare(label, opera_binary))
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        rows.append(round_floats(row))

    match_csv = args.output_dir / "manifests" / "opera_label_matches.csv"
    write_csv(match_csv, rows, MATCH_FIELDS)
    summary = summarize(rows, args)
    summary_path = args.output_dir / "manifests" / "opera_label_comparison_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
