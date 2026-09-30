#!/usr/bin/env python3
"""Download label-aligned Sentinel-1 and AlphaEarth products.

Each label GeoTIFF defines the output grid. For every matched sample in the
intercomparison CSV, this script downloads products for the reference date and
the requested date offsets.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fetch_alphaearth_embeddings as alphaearth


DEFAULT_CSV = Path("data/S1_Intercomparison_All_Scenes.csv")
DEFAULT_LABEL_DIR = Path("data/S1_Intercomparison_All_Scenes_subset/labels")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "intercomparison_s1aef"
DEFAULT_OFFSET_DAYS = (-1, 0, 1)
S1_BANDS = ("VV", "VH", "angle")
PRODUCTS = ("s1", "alphaearth")

DOWNLOAD_FIELDS = (
    "product",
    "sample_id",
    "label_path",
    "reference_date",
    "offset_days",
    "target_date",
    "year",
    "image_count",
    "source_count",
    "matched_scene_ids",
    "matched_datetimes",
    "output_path",
    "status",
    "error",
    "updated_at",
)
INPUT_FIELDS = (
    "sample_id",
    "reference_date",
    "offset_days",
    "target_date",
    "year",
    "label_path",
    "s1_path",
    "aef_path",
    "s1_exists",
    "aef_exists",
)
SAMPLE_FIELDS = (
    "sample_id",
    "source",
    "sample_id_raw",
    "sample_id_clean",
    "reference_date",
    "csv_s1_grd_scene",
    "label_path",
    "crs",
    "width",
    "height",
    "transform",
    "bounds",
    "bbox_wgs84",
)


@dataclass(frozen=True)
class LabelGrid:
    path: Path
    sample_id: str
    crs: str
    transform: Any
    width: int
    height: int
    bounds: Tuple[float, float, float, float]
    bbox_wgs84: Tuple[float, float, float, float]
    region_wgs84: Dict[str, Any]


@dataclass(frozen=True)
class Sample:
    sample_id: str
    row: Dict[str, str]
    label: LabelGrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--offset-days", type=int, nargs="+", default=list(DEFAULT_OFFSET_DAYS))
    parser.add_argument("--products", nargs="+", choices=PRODUCTS, default=list(PRODUCTS))
    parser.add_argument("--only-sample", action="append", default=None, help="Sample ID to process; may be repeated.")
    parser.add_argument("--sample-offset", type=int, default=0, help="Skip this many matched samples after filtering.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--manifest-tag", default=None, help="Optional suffix for per-job manifest shards.")
    parser.add_argument("--ee-project", default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--alphaearth-output-mode", choices=alphaearth.OUTPUT_MODES, default=alphaearth.DEFAULT_OUTPUT_MODE)
    parser.add_argument("--alphaearth-compression", choices=alphaearth.COMPRESSIONS, default=alphaearth.DEFAULT_COMPRESSION)
    args = parser.parse_args()

    if args.sample_offset < 0:
        parser.error("--sample-offset must be >= 0")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be >= 1")
    if not args.offset_days:
        parser.error("--offset-days must include at least one value")

    args.csv = args.csv.expanduser().resolve()
    args.label_dir = args.label_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def require_module(module_name: str, package_name: Optional[str] = None) -> Any:
    try:
        return __import__(module_name, fromlist=["_marker"])
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency '{}'. Use the S1+AEF environment (requirements.txt)."
            "".format(package_name or module_name)
        ) from exc


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def polygon_from_bounds(bounds: Tuple[float, float, float, float]) -> Dict[str, Any]:
    left, bottom, right, top = bounds
    return {
        "type": "Polygon",
        "coordinates": [[[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]],
    }


def init_ee(project: Optional[str]) -> Any:
    ee = require_module("ee", "earthengine-api")
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise SystemExit(
            "Earth Engine credentials are not configured. Run authenticate_earthengine.py once, then rerun with --resume."
        ) from exc
    return ee


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError("CSV does not exist: {}".format(path))
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"SampleID", "SampleID_clean", "Date", "s1_grd_scene"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError("CSV is missing required columns: {}".format(", ".join(sorted(missing))))
    return rows


def row_lookup(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    for row in rows:
        for key in (row.get("SampleID_clean", ""), row.get("SampleID", "")):
            if key and key not in lookup:
                lookup[key] = row
    return lookup


def read_label(path: Path) -> LabelGrid:
    rio = require_module("rasterio")
    warp = require_module("rasterio.warp")
    with rio.open(path) as src:
        if src.crs is None:
            raise ValueError("Label is missing CRS: {}".format(path))
        crs = src.crs.to_string()
        bounds = tuple(float(value) for value in src.bounds)
        bbox_wgs84 = tuple(float(value) for value in warp.transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21))
        return LabelGrid(
            path=path,
            sample_id=path.stem,
            crs=crs,
            transform=src.transform,
            width=int(src.width),
            height=int(src.height),
            bounds=bounds,  # type: ignore[arg-type]
            bbox_wgs84=bbox_wgs84,  # type: ignore[arg-type]
            region_wgs84=polygon_from_bounds(bbox_wgs84),  # type: ignore[arg-type]
        )


def load_samples(csv_path: Path, label_dir: Path) -> List[Sample]:
    rows = read_csv_rows(csv_path)
    lookup = row_lookup(rows)
    labels = sorted(label_dir.glob("*.tif"))
    if not labels:
        raise FileNotFoundError("No .tif labels found in {}".format(label_dir))

    samples: List[Sample] = []
    missing: List[str] = []
    for label_path in labels:
        row = lookup.get(label_path.stem)
        if row is None:
            missing.append(label_path.stem)
            continue
        samples.append(Sample(sample_id=label_path.stem, row=row, label=read_label(label_path)))
    if missing:
        raise ValueError("Labels missing from CSV: {}".format(", ".join(missing)))
    return samples


def select_samples(samples: Sequence[Sample], only_sample: Optional[Sequence[str]], sample_offset: int, max_samples: Optional[int]) -> List[Sample]:
    selected = list(samples)
    if only_sample:
        wanted = set(only_sample)
        selected = [sample for sample in selected if sample.sample_id in wanted]
        missing = sorted(wanted.difference(sample.sample_id for sample in selected))
        if missing:
            raise ValueError("--only-sample not found: {}".format(", ".join(missing)))
    if sample_offset:
        selected = selected[sample_offset:]
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def tagged_path(directory: Path, stem: str, suffix: str, tag: Optional[str]) -> Path:
    if tag:
        return directory / "{}_{}{}".format(stem, tag, suffix)
    return directory / "{}{}".format(stem, suffix)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def sample_rows(samples: Sequence[Sample]) -> List[Dict[str, Any]]:
    return [
        {
            "sample_id": sample.sample_id,
            "source": sample.row.get("Source", ""),
            "sample_id_raw": sample.row.get("SampleID", ""),
            "sample_id_clean": sample.row.get("SampleID_clean", ""),
            "reference_date": sample.row["Date"],
            "csv_s1_grd_scene": sample.row.get("s1_grd_scene", ""),
            "label_path": str(sample.label.path),
            "crs": sample.label.crs,
            "width": sample.label.width,
            "height": sample.label.height,
            "transform": json.dumps(list(sample.label.transform)[:6]),
            "bounds": json.dumps(sample.label.bounds),
            "bbox_wgs84": json.dumps(sample.label.bbox_wgs84),
        }
        for sample in samples
    ]


def make_region(ee: Any, label: LabelGrid) -> Any:
    return ee.Geometry(label.region_wgs84)


def s1_collection_for_day(ee: Any, region: Any, target: date) -> Any:
    end = target + timedelta(days=1)
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(target.isoformat(), end.isoformat())
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(list(S1_BANDS))
        .sort("system:time_start")
    )


def collection_metadata(collection: Any) -> Tuple[int, List[str], List[str]]:
    count = int(collection.size().getInfo())
    if count == 0:
        return 0, [], []
    scene_ids = [str(value) for value in collection.aggregate_array("system:index").getInfo()]
    times = collection.aggregate_array("system:time_start").getInfo()
    datetimes = [datetime.utcfromtimestamp(value / 1000).strftime("%Y-%m-%dT%H:%M:%SZ") for value in times]
    return count, scene_ids, datetimes


def s1_image_from_collection(collection: Any, region: Any) -> Any:
    return collection.mosaic().select(list(S1_BANDS)).float().clip(region)


def download_ee_label(image: Any, label: LabelGrid, name: str, out_path: Path, timeout: int) -> None:
    requests = require_module("requests")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "name": name,
        "crs": label.crs,
        "crs_transform": list(label.transform)[:6],
        "dimensions": [label.width, label.height],
        "format": "ZIPPED_GEO_TIFF",
        "filePerBand": False,
    }
    url = image.getDownloadURL(params)
    with requests.get(url, timeout=timeout) as response:
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            tif_names = [item for item in zf.namelist() if item.lower().endswith((".tif", ".tiff"))]
            if not tif_names:
                raise RuntimeError("Earth Engine zip did not contain a GeoTIFF")
            with tempfile.NamedTemporaryFile("wb", dir=out_path.parent, delete=False) as tmp:
                tmp_path = Path(tmp.name)
                with zf.open(tif_names[0]) as src:
                    tmp.write(src.read())
    tmp_path.replace(out_path)


def download_s1(ee: Any, sample: Sample, target: date, offset_days: int, out_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    status = "exists"
    error = ""
    image_count = ""
    scene_ids: List[str] = []
    datetimes: List[str] = []
    if args.overwrite or not (args.resume and out_path.exists()):
        try:
            region = make_region(ee, sample.label)
            collection = s1_collection_for_day(ee, region, target)
            count, scene_ids, datetimes = collection_metadata(collection)
            image_count = count
            if count == 0:
                status = "no_match"
            else:
                image = s1_image_from_collection(collection, region)
                download_ee_label(
                    image=image,
                    label=sample.label,
                    name="{}_{}_s1".format(sample.sample_id, target.isoformat()),
                    out_path=out_path,
                    timeout=args.timeout,
                )
                status = "ok"
        except Exception as exc:
            status = "error"
            error = "{}: {}".format(type(exc).__name__, exc)
    return {
        "product": "s1",
        "sample_id": sample.sample_id,
        "label_path": str(sample.label.path),
        "reference_date": sample.row["Date"],
        "offset_days": offset_days,
        "target_date": target.isoformat(),
        "year": target.year,
        "image_count": image_count,
        "source_count": "",
        "matched_scene_ids": json.dumps(scene_ids),
        "matched_datetimes": json.dumps(datetimes),
        "output_path": str(out_path),
        "status": status,
        "error": error,
        "updated_at": utc_now(),
    }


def download_alphaearth(sample: Sample, target: date, offset_days: int, out_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    status = "exists"
    error = ""
    source_count = ""
    if args.overwrite or not (args.resume and out_path.exists()):
        try:
            sources = alphaearth.write_embedding(
                bbox_wgs84=sample.label.bbox_wgs84,
                output_path=out_path,
                year=target.year,
                dst_crs=sample.label.crs,
                dst_transform=sample.label.transform,
                width=sample.label.width,
                height=sample.label.height,
                cache_dir=args.output_dir / "cache",
                output_mode=args.alphaearth_output_mode,
                compression=args.alphaearth_compression,
                timeout=args.timeout,
            )
            status = "ok"
            source_count = len(sources)
        except Exception as exc:
            status = "error"
            error = "{}: {}".format(type(exc).__name__, exc)
    return {
        "product": "alphaearth",
        "sample_id": sample.sample_id,
        "label_path": str(sample.label.path),
        "reference_date": sample.row["Date"],
        "offset_days": offset_days,
        "target_date": target.isoformat(),
        "year": target.year,
        "image_count": "",
        "source_count": source_count,
        "matched_scene_ids": "",
        "matched_datetimes": "",
        "output_path": str(out_path),
        "status": status,
        "error": error,
        "updated_at": utc_now(),
    }


def input_row(sample: Sample, target: date, offset_days: int, s1_path: Path, aef_path: Path) -> Dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "reference_date": sample.row["Date"],
        "offset_days": offset_days,
        "target_date": target.isoformat(),
        "year": target.year,
        "label_path": str(sample.label.path),
        "s1_path": str(s1_path),
        "aef_path": str(aef_path),
        "s1_exists": str(s1_path.exists()).lower(),
        "aef_exists": str(aef_path.exists()).lower(),
    }


def main() -> None:
    args = parse_args()
    samples = load_samples(args.csv, args.label_dir)
    selected = select_samples(samples, args.only_sample, args.sample_offset, args.max_samples)
    if not selected:
        raise SystemExit("No samples selected.")

    manifest_dir = args.output_dir / "manifests"
    downloads_path = tagged_path(manifest_dir, "downloads", ".csv", args.manifest_tag)
    inputs_path = tagged_path(manifest_dir, "inference_inputs", ".csv", args.manifest_tag)
    samples_path = tagged_path(manifest_dir, "samples", ".csv", args.manifest_tag)
    write_csv(samples_path, sample_rows(selected), SAMPLE_FIELDS)

    expanded = len(selected) * len(args.offset_days)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "matched_samples_total": len(samples),
                    "selected_samples": len(selected),
                    "offset_days": args.offset_days,
                    "sample_date_rows": expanded,
                    "products": args.products,
                    "samples_manifest": str(samples_path),
                },
                indent=2,
            )
        )
        return

    ee = init_ee(args.ee_project) if "s1" in args.products else None
    download_rows: List[Dict[str, Any]] = []
    input_rows: List[Dict[str, Any]] = []

    for sample in selected:
        reference = parse_date(sample.row["Date"])
        for offset_days in args.offset_days:
            target = reference + timedelta(days=offset_days)
            s1_path = args.output_dir / "s1" / sample.sample_id / "{}_s1.tif".format(target.isoformat())
            aef_path = args.output_dir / "alphaearth" / sample.sample_id / "{}_aef.tif".format(target.isoformat())

            if "alphaearth" in args.products:
                download_rows.append(download_alphaearth(sample, target, offset_days, aef_path, args))
                write_csv(downloads_path, download_rows, DOWNLOAD_FIELDS)

            if "s1" in args.products:
                assert ee is not None
                download_rows.append(download_s1(ee, sample, target, offset_days, s1_path, args))
                write_csv(downloads_path, download_rows, DOWNLOAD_FIELDS)

            input_rows.append(input_row(sample, target, offset_days, s1_path, aef_path))
            write_csv(inputs_path, input_rows, INPUT_FIELDS)

    write_csv(downloads_path, download_rows, DOWNLOAD_FIELDS)
    write_csv(inputs_path, input_rows, INPUT_FIELDS)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir),
                "selected_samples": len(selected),
                "sample_date_rows": expanded,
                "download_rows": len(download_rows),
                "downloads_manifest": str(downloads_path),
                "inference_inputs_manifest": str(inputs_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
