#!/usr/bin/env python3
"""Create one nearest Sentinel-1 product per intercomparison label.

The intercomparison CSV already records the nearest Sentinel-1 GRD scene for
each label in ``s1_grd_scene``. This script writes a canonical nearest-S1
layout, reusing an existing exact-day download when it matches that CSV scene,
and downloading the CSV scene directly otherwise.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_CSV = Path("data/S1_Intercomparison_All_Scenes.csv")
DEFAULT_LABEL_DIR = Path("data/S1_Intercomparison_All_Scenes_subset/labels")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data" / "intercomparison_s1aef"
S1_BANDS = ("VV", "VH", "angle")
MANIFEST_FIELDS = (
    "sample_id",
    "reference_date",
    "acquisition_datetime",
    "date_delta_days",
    "time_delta_hours",
    "scene_id",
    "label_path",
    "output_path",
    "source_path",
    "status",
    "error",
    "updated_at",
)


@dataclass(frozen=True)
class LabelGrid:
    path: Path
    crs: str
    transform: Any
    width: int
    height: int


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
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to DATA_ROOT/s1_nearest.")
    parser.add_argument("--manifest", type=Path, default=None, help="Defaults to DATA_ROOT/manifests/nearest_s1.csv.")
    parser.add_argument("--downloads-glob", default="downloads_sample_*.csv")
    parser.add_argument("--only-sample", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ee-project", default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be >= 1")
    args.csv = args.csv.expanduser().resolve()
    args.label_dir = args.label_dir.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = (args.output_dir or args.data_root / "s1_nearest").expanduser().resolve()
    args.manifest = (args.manifest or args.data_root / "manifests" / "nearest_s1.csv").expanduser().resolve()
    return args


def require_module(module_name: str, package_name: Optional[str] = None) -> Any:
    try:
        return __import__(module_name, fromlist=["_marker"])
    except ImportError as exc:
        raise SystemExit("Missing dependency '{}'. Use the S1+AEF environment.".format(package_name or module_name)) from exc


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_label(path: Path) -> LabelGrid:
    rio = require_module("rasterio")
    with rio.open(path) as src:
        if src.crs is None:
            raise ValueError("Label is missing CRS: {}".format(path))
        return LabelGrid(
            path=path,
            crs=src.crs.to_string(),
            transform=src.transform,
            width=int(src.width),
            height=int(src.height),
        )


def read_samples(csv_path: Path, label_dir: Path) -> List[Sample]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup: Dict[str, Dict[str, str]] = {}
    for row in rows:
        for key in (row.get("SampleID_clean", ""), row.get("SampleID", "")):
            if key and key not in lookup:
                lookup[key] = row
    samples: List[Sample] = []
    missing: List[str] = []
    for label_path in sorted(label_dir.glob("*.tif")):
        row = lookup.get(label_path.stem)
        if row is None:
            missing.append(label_path.stem)
            continue
        samples.append(Sample(label_path.stem, row, read_label(label_path)))
    if missing:
        raise ValueError("Labels missing from CSV: {}".format(", ".join(missing)))
    return samples


def select_samples(samples: Sequence[Sample], only_sample: Optional[Sequence[str]], max_samples: Optional[int]) -> List[Sample]:
    selected = list(samples)
    if only_sample:
        wanted = set(only_sample)
        selected = [sample for sample in selected if sample.sample_id in wanted]
        missing = sorted(wanted.difference(sample.sample_id for sample in selected))
        if missing:
            raise ValueError("--only-sample not found: {}".format(", ".join(missing)))
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def load_existing_s1(data_root: Path, downloads_glob: str) -> Dict[Tuple[str, str], Path]:
    matches: Dict[Tuple[str, str], Path] = {}
    for manifest in sorted((data_root / "manifests").glob(downloads_glob)):
        with manifest.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("product") != "s1" or row.get("status") not in {"ok", "exists"}:
                    continue
                path = Path(row.get("output_path", ""))
                if not path.exists():
                    continue
                scene_ids = set()
                try:
                    scene_ids.update(json.loads(row.get("matched_scene_ids") or "[]"))
                except json.JSONDecodeError:
                    pass
                for scene_id in scene_ids:
                    matches[(row["sample_id"], scene_id)] = path
    return matches


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def link_or_copy(source: Path, dest: Path, mode: str, overwrite: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if not overwrite:
            return
        dest.unlink()
    if mode == "symlink":
        dest.symlink_to(source)
    elif mode == "hardlink":
        os.link(source, dest)
    else:
        import shutil

        shutil.copy2(source, dest)


def init_ee(project: Optional[str]) -> Any:
    ee = require_module("ee", "earthengine-api")
    ee.Initialize(project=project)
    return ee


def download_scene(ee: Any, sample: Sample, scene_id: str, out_path: Path, timeout: int) -> None:
    requests = require_module("requests")
    image = ee.Image("COPERNICUS/S1_GRD/{}".format(scene_id)).select(list(S1_BANDS)).float()
    params = {
        "name": "{}_nearest_s1".format(sample.sample_id),
        "crs": sample.label.crs,
        "crs_transform": list(sample.label.transform)[:6],
        "dimensions": [sample.label.width, sample.label.height],
        "format": "ZIPPED_GEO_TIFF",
        "filePerBand": False,
    }
    url = image.getDownloadURL(params)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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


def date_delta_days(row: Dict[str, str]) -> str:
    try:
        reference = datetime.strptime(row["Date"], "%Y-%m-%d").date()
        acquisition = datetime.strptime(row["acquisition_datetime"], "%Y-%m-%d %H:%M:%S").date()
        return str((acquisition - reference).days)
    except Exception:
        return ""


def output_path(output_dir: Path, sample: Sample) -> Path:
    return output_dir / sample.sample_id / "{}_nearest_s1.tif".format(sample.sample_id)


def main() -> None:
    args = parse_args()
    samples = select_samples(read_samples(args.csv, args.label_dir), args.only_sample, args.max_samples)
    existing = load_existing_s1(args.data_root, args.downloads_glob)
    rows: List[Dict[str, Any]] = []

    if args.dry_run:
        reusable = sum((sample.sample_id, sample.row["s1_grd_scene"]) in existing for sample in samples)
        print(json.dumps({"status": "dry-run", "samples": len(samples), "reusable_existing": reusable, "needs_download": len(samples) - reusable}, indent=2))
        return

    ee = None
    for sample in samples:
        scene_id = sample.row["s1_grd_scene"]
        out_path = output_path(args.output_dir, sample)
        source_path = existing.get((sample.sample_id, scene_id))
        status = "exists" if out_path.exists() and args.resume and not args.overwrite else "ok"
        error = ""
        try:
            if status != "exists":
                if source_path is not None:
                    link_or_copy(source_path, out_path, args.copy_mode, overwrite=args.overwrite)
                    status = "linked" if args.copy_mode == "symlink" else args.copy_mode
                else:
                    if ee is None:
                        ee = init_ee(args.ee_project)
                    download_scene(ee, sample, scene_id, out_path, args.timeout)
                    status = "downloaded"
        except Exception as exc:
            status = "error"
            error = "{}: {}".format(type(exc).__name__, exc)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "reference_date": sample.row["Date"],
                "acquisition_datetime": sample.row.get("acquisition_datetime", ""),
                "date_delta_days": date_delta_days(sample.row),
                "time_delta_hours": sample.row.get("time_delta_hours", ""),
                "scene_id": scene_id,
                "label_path": str(sample.label.path),
                "output_path": str(out_path),
                "source_path": str(source_path or ""),
                "status": status,
                "error": error,
                "updated_at": utc_now(),
            }
        )
        write_csv(args.manifest, rows, MANIFEST_FIELDS)

    write_csv(args.manifest, rows, MANIFEST_FIELDS)
    status_counts: Dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    print(json.dumps({"status": "complete", "samples": len(samples), "status_counts": status_counts, "manifest": str(args.manifest)}, indent=2))


if __name__ == "__main__":
    main()
