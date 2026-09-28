#!/usr/bin/env python3
"""Prepare strict prior-acquisition-year AEF inputs, preserving S1, labels and splits."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

SAMPLE_ID = "paper_openwater_strict_tminus1_v1"


def covering_aef_urls(records, year, bbox, crs):
    """Include adjacent projections; apply native-projection pixels last."""
    candidates = [(source_crs == crs.upper(), url)
                  for source_crs, url, bounds in records.get(year, [])
                  if bounds[0] < bbox[2] and bounds[2] > bbox[0]
                  and bounds[1] < bbox[3] and bounds[3] > bbox[1]]
    return list(dict.fromkeys(url for _, url in sorted(candidates)))


def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def eligible_rows(root, rows):
    path = root / "coverage_exclusions.csv"
    excluded = {r["tile_name"] for r in read(path)} if path.exists() else set()
    return [r for r in rows if r["tile_name"] not in excluded]


def prepare(args):
    original = {r["tile_name"]: r["s1_date"] for r in read(args.repo / "metadata/training_samples.csv")}
    supplement = {r["candidate_id"] + ".tif": r["s1_datetime_utc"] for r in read(args.repo / "outputs/audit/supplement_samples.csv")}
    dates = {**original, **supplement}
    rows, excluded = [], []
    for row in read(args.source):
        year = int(dates[row["tile_name"]][:4]) - 1
        row = dict(row, acquisition_date=dates[row["tile_name"]], aef_year=str(year))
        if year < 2017:
            excluded.append(dict(row, reason="prior-year AEF unavailable"))
            continue
        row["sample_id"] = SAMPLE_ID
        row["aef_path"] = str(args.root / "aef" / str(year) / row["tile_name"])
        rows.append(row)
    assert len(rows) + len(excluded) == 5278
    assert len({r["tile_name"] for r in rows}) == len(rows)
    args.root.mkdir(parents=True, exist_ok=True)
    write(args.root / "planned_triplets.csv", rows)
    if excluded:
        write(args.root / "excluded.csv", excluded)
    report = dict(total=len(rows), splits=dict(Counter(r["split"] for r in rows)),
                  excluded=len(excluded), aef_years=dict(Counter(r["aef_year"] for r in rows)),
                  source=str(args.source), policy="strict acquisition year minus one; no fallback")
    (args.root / "plan.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def materialize(args):
    import numpy as np
    import rasterio as rio
    from rasterio.warp import transform_bounds
    from materialize_supplement_samples import load_aef_records, atomic_write_aef
    from reconstruct_sample import sha256
    rows = read(args.root / "planned_triplets.csv")[args.shard::args.shards]
    # Filter after sharding so a recorded exclusion cannot reassign unfinished
    # tiles to shards that have already completed.
    rows = eligible_rows(args.root, rows)
    records = load_aef_records(args.index, {int(r["aef_year"]) for r in rows})
    for row in rows:
        out = Path(row["aef_path"])
        sidecar = out.with_suffix(".json")
        with rio.open(row["s1_path"]) as s1:
            bbox = transform_bounds(s1.crs, "EPSG:4326", *s1.bounds, densify_pts=21)
            urls = covering_aef_urls(records, int(row["aef_year"]), bbox, str(s1.crs))
            if not urls:
                raise ValueError(f"No prior-year source for {row['tile_name']}")
            if not (out.exists() and sidecar.exists()):
                out.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_aef(out, urls, str(s1.crs), s1.transform, s1.width, s1.height)
            else:
                # Preserve provenance for validated files reused from earlier attempts.
                prior = json.loads(sidecar.read_text())
                if prior["aef_year"] != int(row["aef_year"]):
                    raise ValueError(f"Cached AEF year mismatch: {out}")
                urls = prior["source_urls"]
            with rio.open(out) as aef:
                assert (aef.crs, aef.transform, aef.width, aef.height, aef.count) == (s1.crs, s1.transform, s1.width, s1.height, 64)
                valid = np.all(aef.read() != -128, axis=0)
                fraction = float(valid.mean())
                if fraction < 0.90:
                    raise ValueError(f"Insufficient AEF coverage {fraction}: {out}")
        record = dict(tile_name=row["tile_name"], aef_year=int(row["aef_year"]),
                      acquisition_date=row["acquisition_date"], source_urls=urls,
                      aef_sha256=sha256(out), valid_fraction=fraction)
        sidecar.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


def finalize(args):
    rows = eligible_rows(args.root, read(args.root / "planned_triplets.csv"))
    for r in rows:
        p = Path(r["aef_path"])
        info = json.loads(p.with_suffix(".json").read_text())
        assert p.is_file() and p.stat().st_size > 0
        assert info["tile_name"] == r["tile_name"]
        assert info["aef_year"] == int(r["acquisition_date"][:4]) - 1 == int(r["aef_year"])
        assert info["valid_fraction"] >= 0.90
    write(args.root / "triplets.csv", rows)
    write(args.root / "fixed_split.csv", [{"tile_name": r["tile_name"], "split": r["split"]} for r in rows])
    report = dict(total=len(rows), splits=dict(Counter(r["split"] for r in rows)),
                  coverage_exclusions=read(args.root / "coverage_exclusions.csv")
                  if (args.root / "coverage_exclusions.csv").exists() else [],
                  policy="strict acquisition year minus one; minimum AEF coverage 0.90")
    (args.root / "manifest_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated {len(rows)} strict t-1 triplets")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "materialize", "finalize"])
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path("/pscratch/sd/r/rohit9/S1ML/training_supplement/paper_retrain_v1/triplets.csv"))
    p.add_argument("--index", type=Path, default=Path("/pscratch/sd/r/rohit9/S1ML/training_supplement/v1/cache/aef_index.csv"))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=12)
    a = p.parse_args()
    if not 0 <= a.shard < a.shards:
        p.error("shard must be in [0, shards)")
    globals()[a.mode](a)
