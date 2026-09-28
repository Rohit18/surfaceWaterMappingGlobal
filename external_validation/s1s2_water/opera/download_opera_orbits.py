#!/usr/bin/env python3
"""Retrieve precise Sentinel-1 orbit files for every exact benchmark SLC."""

import hashlib
import json
import pathlib

from s1reader.s1_orbit import retrieve_orbit_file


ROOT = pathlib.Path(__file__).resolve().parents[1]
SLC_MANIFEST = ROOT / "data/rtc/slc_manifest.json"
OUTPUT = ROOT / "data/rtc/orbit_manifest.json"


def sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    records = []
    for slc in json.loads(SLC_MANIFEST.read_text())["records"]:
        directory = ROOT / "data/rtc" / f"scene_{slc['scene_id']}" / "orbits"
        safe = ROOT / "data/rtc" / f"scene_{slc['scene_id']}" / slc["file_name"]
        orbit = retrieve_orbit_file(str(safe), str(directory), orbit_type_preference="precise")
        if not isinstance(orbit, str) or "POEORB" not in orbit:
            raise RuntimeError(f"Precise orbit unavailable for {slc['slc_product_id']}: {orbit}")
        path = pathlib.Path(orbit)
        records.append({
            "scene_id": slc["scene_id"],
            "source_order": slc["source_order"],
            "slc_product_id": slc["slc_product_id"],
            "orbit_file": path.name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
        print(f"scene {slc['scene_id']} {slc['slc_product_id']} -> {path.name}", flush=True)
    OUTPUT.write_text(json.dumps({
        "source_manifest": str(SLC_MANIFEST),
        "selection": "s1-reader 0.2.5 precise-orbit preference from ASF public s1-orbits bucket",
        "record_count": len(records),
        "records": records,
    }, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
