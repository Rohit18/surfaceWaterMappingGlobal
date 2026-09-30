#!/usr/bin/env python3
"""Run S1+AEF inference over PNW tiled inputs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import rasterio as rio
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data" / "inference"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "predictions"
DEFAULT_RUN_DIR = REPOSITORY_ROOT / "models" / "s1aef_resnet34"
DEFAULT_INFER_CORE = Path(__file__).resolve().parent / "infer_s1_probability_models.py"
DEFAULT_TRAIN_SCRIPT = Path(__file__).resolve().parent / "train-s1aef-resnet34-bottleneck.py"


@dataclass(frozen=True)
class TileScene:
    date: str
    year: int
    tile_id: str
    s1_path: Path
    aef_path: Path
    prob_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--s1-root", type=Path, default=None)
    parser.add_argument("--aef-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-tag", default="s1_aef_tta")
    parser.add_argument("--manifest-tag", default=None)
    parser.add_argument("--input-manifest", type=Path, default=None)
    parser.add_argument("--sample-id", default=None, help="Optional input-manifest sample_id filter.")
    parser.add_argument("--source-id", default=None, help="Optional input-manifest source_id filter.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--infer-core-script", type=Path, default=DEFAULT_INFER_CORE)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--aef-year", type=int, default=2025)
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--dates", nargs="+", default=None)
    parser.add_argument("--tiles", nargs="+", default=None)
    parser.add_argument("--tile-offset", type=int, default=0)
    parser.add_argument("--tile-count", type=int, default=None)
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
    parser.add_argument(
        "--aef-ablation",
        "--aef_ablation",
        choices=("none", "mean"),
        default="none",
        help="Replace AEF channels with training-set band means during inference.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    args.data_root = args.data_root.expanduser().resolve()
    if args.s1_root is None:
        args.s1_root = args.data_root / "s1"
    if args.aef_root is None:
        args.aef_root = args.data_root / "alphaearth" / str(args.aef_year)
    args.s1_root = args.s1_root.expanduser().resolve()
    args.aef_root = args.aef_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.infer_core_script = args.infer_core_script.expanduser().resolve()
    args.train_script = args.train_script.expanduser().resolve()
    if args.weights_path is not None:
        args.weights_path = args.weights_path.expanduser().resolve()
    if args.input_manifest is not None:
        args.input_manifest = args.input_manifest.expanduser().resolve()
    if args.max_scenes is not None and args.max_scenes < 1:
        parser.error("--max-scenes must be >= 1")
    if args.tile_offset < 0:
        parser.error("--tile-offset must be >= 0")
    if args.tile_count is not None and args.tile_count < 1:
        parser.error("--tile-count must be >= 1")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_core(path: Path):
    if not path.exists():
        raise FileNotFoundError("Inference core script not found: {}".format(path))
    spec = importlib.util.spec_from_file_location("infer_s1_probability_models_core", path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load module spec from {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discover_tile_scenes(args: argparse.Namespace, output_dir: Path) -> List[TileScene]:
    if args.input_manifest is not None:
        if not args.input_manifest.exists():
            if args.allow_empty:
                return []
            raise FileNotFoundError("Input manifest does not exist: {}".format(args.input_manifest))
        records: List[TileScene] = []
        with args.input_manifest.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if args.sample_id is not None and row.get("sample_id") != args.sample_id:
                    continue
                if args.source_id is not None and row.get("source_id") != args.source_id:
                    continue
                s1_path = Path(row["s1_path"]).expanduser().resolve()
                aef_path = Path(row["aef_path"]).expanduser().resolve()
                tile_id = row.get("tile_id") or row.get("name") or (
                    s1_path.stem[:-3] if s1_path.stem.endswith("_s1") else s1_path.stem
                )
                tile_id = Path(tile_id).stem
                date = row.get("scene_key") or row.get("date")
                if not date:
                    date_match = re.search(r"\d{4}-\d{2}-\d{2}", row.get("name") or s1_path.stem)
                    date = date_match.group(0) if date_match else "undated"
                year_text = row.get("year") or (str(date)[:4] if str(date)[:4].isdigit() else "0")
                year = int(year_text)
                records.append(
                    TileScene(
                        date=str(date),
                        year=year,
                        tile_id=tile_id,
                        s1_path=s1_path,
                        aef_path=aef_path,
                        prob_path=output_dir / "probabilities" / str(date) / "{}_prob.tif".format(tile_id),
                    )
                )
        if not records and args.allow_empty:
            return []
        if not records:
            raise FileNotFoundError("No rows found in input manifest {}".format(args.input_manifest))
        records = sorted(records, key=lambda record: (record.date, record.tile_id))
        if args.tile_offset:
            records = records[args.tile_offset :]
        if args.tile_count is not None:
            records = records[: args.tile_count]
        if args.max_scenes is not None:
            records = records[: args.max_scenes]
        if not records and args.allow_empty:
            return []
        if not records:
            raise FileNotFoundError("No input-manifest rows remain after filtering and slicing.")
        return records

    if not args.s1_root.exists():
        raise FileNotFoundError("S1 root does not exist: {}".format(args.s1_root))
    if not args.aef_root.exists():
        raise FileNotFoundError("AEF root does not exist: {}".format(args.aef_root))

    year_filter = None if args.years is None else {int(year) for year in args.years}
    date_filter = None if args.dates is None else set(args.dates)
    tile_filter = None if args.tiles is None else set(args.tiles)
    all_records: List[TileScene] = []
    for s1_path in sorted(args.s1_root.rglob("*_s1.tif")):
        date = s1_path.parent.name
        if date_filter is not None and date not in date_filter:
            continue
        try:
            year = int(date[:4])
        except ValueError:
            continue
        if year_filter is not None and year not in year_filter:
            continue
        tile_id = s1_path.stem[:-3] if s1_path.stem.endswith("_s1") else s1_path.stem
        aef_path = args.aef_root / "{}_aef.tif".format(tile_id)
        all_records.append(
            TileScene(
                date=date,
                year=year,
                tile_id=tile_id,
                s1_path=s1_path,
                aef_path=aef_path,
                prob_path=output_dir / "probabilities" / date / "{}_prob.tif".format(tile_id),
            )
        )

    tile_ids = sorted({record.tile_id for record in all_records})
    if tile_filter is not None:
        tile_ids = [tile_id for tile_id in tile_ids if tile_id in tile_filter]
    if args.tile_offset:
        tile_ids = tile_ids[args.tile_offset:]
    if args.tile_count is not None:
        tile_ids = tile_ids[: args.tile_count]
    wanted_tiles = set(tile_ids)
    records = [record for record in all_records if record.tile_id in wanted_tiles]
    if args.max_scenes is not None:
        records = records[: args.max_scenes]
    if not records and args.allow_empty:
        return []
    if not records:
        raise FileNotFoundError("No tiled S1 scenes found under {}".format(args.s1_root))
    return records


def validate_records(records: Sequence[TileScene], run_dir: Path, weights_path: Path, core) -> dict:
    if not run_dir.exists():
        raise FileNotFoundError("Run dir does not exist: {}".format(run_dir))
    if not weights_path.exists():
        raise FileNotFoundError("Weights do not exist: {}".format(weights_path))
    core.load_band_stats(run_dir, expected_bands=67)

    missing_s1 = [str(record.s1_path) for record in records if not record.s1_path.exists()]
    missing_aef = [str(record.aef_path) for record in records if not record.aef_path.exists()]
    if missing_s1:
        raise FileNotFoundError("Missing {} S1 tiles; first: {}".format(len(missing_s1), missing_s1[0]))
    if missing_aef:
        raise FileNotFoundError("Missing {} AEF tiles; first: {}".format(len(missing_aef), missing_aef[0]))

    first = records[0]
    with rio.open(str(first.s1_path)) as s1, rio.open(str(first.aef_path)) as aef:
        if s1.count != 3:
            raise ValueError("Expected 3 S1 bands in {}, found {}".format(first.s1_path, s1.count))
        if aef.count != 64:
            raise ValueError("Expected 64 AEF bands in {}, found {}".format(first.aef_path, aef.count))
        sample_grid = {
            "s1_crs": s1.crs.to_string() if s1.crs else None,
            "s1_width": s1.width,
            "s1_height": s1.height,
            "aef_crs": aef.crs.to_string() if aef.crs else None,
            "aef_width": aef.width,
            "aef_height": aef.height,
        }

    return {
        "scene_tile_count": len(records),
        "date_counts": dict(sorted(Counter(record.date for record in records).items())),
        "tile_count": len({record.tile_id for record in records}),
        "sample_grid": sample_grid,
    }


def write_manifest(path: Path, records: Sequence[TileScene]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["date", "year", "tile_id", "s1_path", "aef_path", "prob_path"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "date": record.date,
                    "year": record.year,
                    "tile_id": record.tile_id,
                    "s1_path": str(record.s1_path),
                    "aef_path": str(record.aef_path),
                    "prob_path": str(record.prob_path),
                }
            )


def write_summary(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "date",
        "year",
        "tile_id",
        "scene",
        "s1_path",
        "aef_path",
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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def tagged_path(directory: Path, stem: str, suffix: str, tag: Optional[str]) -> Path:
    if tag:
        return directory / "{}_{}{}".format(stem, tag, suffix)
    return directory / "{}{}".format(stem, suffix)


def main() -> None:
    args = parse_args()
    core = load_core(args.infer_core_script)
    output_dir = args.output_root / args.output_tag
    records = discover_tile_scenes(args, output_dir=output_dir)
    if not records and args.allow_empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        empty_config = {
            "created_at": utc_now(),
            "status": "empty",
            "data_root": str(args.data_root),
            "s1_root": str(args.s1_root),
            "output_dir": str(output_dir),
            "dates": args.dates,
            "tiles": args.tiles,
            "tile_offset": args.tile_offset,
            "tile_count": args.tile_count,
        }
        run_config_path = output_dir / ("run_config_{}.json".format(args.manifest_tag) if args.manifest_tag else "run_config.json")
        run_config_path.write_text(json.dumps(empty_config, indent=2))
        print(json.dumps(empty_config, indent=2))
        return
    weights_path = core.resolve_weights_path(args.run_dir, "s1aef", args.weights_path)
    validation = validate_records(records, run_dir=args.run_dir, weights_path=weights_path, core=core)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(tagged_path(output_dir, "scene_tile_manifest", ".csv", args.manifest_tag), records)

    summary = core.load_eval_summary(args.run_dir, "s1aef")
    loss = core.infer_loss(summary)
    n_proj_bands = core.infer_n_proj_bands(args.run_dir, weights_path, args.n_proj_bands)
    device = core.resolve_device(args.device, args.device_id)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    band_mean, band_std = core.load_band_stats(args.run_dir, expected_bands=67)

    run_config = {
        "created_at": utc_now(),
        "model_kind": "s1aef",
        "data_root": str(args.data_root),
        "s1_root": str(args.s1_root),
        "input_manifest": str(args.input_manifest) if args.input_manifest else None,
        "sample_id": args.sample_id,
        "source_id": args.source_id,
        "aef_root": str(args.aef_root),
        "output_dir": str(output_dir),
        "run_dir": str(args.run_dir),
        "weights_path": str(weights_path),
        "train_script": str(args.train_script),
        "infer_core_script": str(args.infer_core_script),
        "device": str(device),
        "tile": args.tile,
        "overlap": args.overlap,
        "batch_size": args.batch_size,
        "tta": bool(args.tta),
        "fp16": bool(args.fp16 and device.type == "cuda"),
        "loss": loss,
        "n_proj_bands": n_proj_bands,
        "aef_ablation": args.aef_ablation,
        "validation": validation,
    }
    run_config_path = output_dir / ("run_config_{}.json".format(args.manifest_tag) if args.manifest_tag else "run_config.json")
    run_config_path.write_text(json.dumps(run_config, indent=2))
    print(json.dumps(run_config, indent=2))
    if args.validate_only:
        print(json.dumps({"status": "validate-only-ok", "output_dir": str(output_dir)}))
        return

    train_mod = core.load_training_module(args.train_script)
    model = core.load_model(
        train_mod=train_mod,
        model_kind="s1aef",
        run_dir=args.run_dir,
        weights_path=weights_path,
        device=device,
        crop_size=args.crop_size,
        loss=loss,
        n_proj_bands=n_proj_bands,
    )
    print(json.dumps({"loaded_weights": str(weights_path), "model_kind": "s1aef"}))

    rows: List[dict] = []
    summary_path = tagged_path(output_dir, "scene_tile_summary", ".csv", args.manifest_tag)
    for index, item in enumerate(records, start=1):
        if args.skip_existing and item.prob_path.exists():
            row = {
                "date": item.date,
                "year": item.year,
                "tile_id": item.tile_id,
                "scene": item.s1_path.stem,
                "s1_path": str(item.s1_path),
                "aef_path": str(item.aef_path),
                "prob_path": str(item.prob_path),
                "status": "skipped-existing",
                "seconds": 0.0,
            }
            rows.append(row)
            write_summary(summary_path, rows)
            print(json.dumps({"scene_index": index, "date": item.date, "tile_id": item.tile_id, "status": "skipped-existing"}))
            continue

        start = time.perf_counter()
        record = core.SceneRecord(date=item.date, year=item.year, s1_path=item.s1_path, prob_path=item.prob_path)
        try:
            row = core.infer_scene(
                record=record,
                model_kind="s1aef",
                aef_path=item.aef_path,
                model=model,
                device=device,
                args=args,
                band_mean=band_mean,
                band_std=band_std,
            )
            row["tile_id"] = item.tile_id
            row["aef_path"] = str(item.aef_path)
            row["error"] = ""
        except Exception as exc:
            error = "{}: {}".format(type(exc).__name__, exc)
            row = core.summarize_scene_failure(record=record, elapsed_seconds=time.perf_counter() - start, error=error)
            row["tile_id"] = item.tile_id
            row["aef_path"] = str(item.aef_path)
            print(json.dumps({"scene_index": index, "date": item.date, "tile_id": item.tile_id, "status": "error", "error": error}))
            traceback.print_exc()
        rows.append(row)
        write_summary(summary_path, rows)
        print(json.dumps({"scene_index": index, "date": item.date, "tile_id": item.tile_id, "status": row.get("status")}))


if __name__ == "__main__":
    main()
