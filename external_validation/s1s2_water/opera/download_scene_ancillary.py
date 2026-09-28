#!/usr/bin/env python3
"""Download, verify, checksum, and mosaic OPERA ancillary data for one scene."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import re
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/ancillary/opera_ancillary_manifest.json"
URL_AUDIT = ROOT / "data/ancillary/url_audit.json"
TYPE_DIR = {
    "dem": "dem",
    "hand": "hand",
    "worldcover": "worldcover",
    "reference_water": "reference_water",
}


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def ocean_zero_urls():
    if not URL_AUDIT.is_file():
        return set()
    unavailable = {item["url"]: item for item in json.loads(URL_AUDIT.read_text())
                   if item["status"] == 404 and item["kind"] in ("dem", "hand")}
    paired = set()
    for url, item in unavailable.items():
        counterpart = item["name"].replace("_DEM.tif", "_HAND.tif").replace("_HAND.tif", "_DEM.tif")
        if any(other["name"] == counterpart for other in unavailable.values()):
            paired.add(url)
    return paired


def absent_worldcover_urls():
    """WorldCover tiles with an audited HTTP 404. ESA publishes no tile for open ocean;
    DSWx reads uncovered pixels as WorldCover No_data (0), so the tile is omitted."""
    if not URL_AUDIT.is_file():
        return set()
    return {item["url"] for item in json.loads(URL_AUDIT.read_text())
            if item["status"] == 404 and item["kind"] == "worldcover"}


def create_ocean_zero(destination):
    match = re.search(r"_([NS])(\d{2})_00_([EW])(\d{3})_00_(?:DEM|HAND)\.tif$", destination.name)
    if not match:
        raise RuntimeError(f"Cannot parse one-degree tile origin: {destination.name}")
    latitude = int(match.group(2)) * (1 if match.group(1) == "N" else -1)
    longitude = int(match.group(4)) * (1 if match.group(3) == "E" else -1)
    half_pixel = 1.0 / 7200.0
    wgs84_wkt = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
        '298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    completed = subprocess.run([
        str(ROOT / "envs/dswx-sar-1.2/bin/gdal_create"), "-q", "-of", "GTiff",
        "-outsize", "3600", "3600", "-bands", "1", "-burn", "0", "-ot", "Float32",
        "-a_srs", wgs84_wkt, "-a_ullr",
        str(longitude - half_pixel), str(latitude + 1 + half_pixel),
        str(longitude + 1 - half_pixel), str(latitude + half_pixel),
        "-co", "TILED=YES", "-co", "COMPRESS=DEFLATE", str(temporary),
    ], check=False)
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(
            f"gdal_create did not produce ocean-fill tile {temporary} "
            f"(exit {completed.returncode})"
        )
    os.replace(temporary, destination)


def fetch(item):
    url, destination, synthesize_ocean = item
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    if synthesize_ocean:
        create_ocean_zero(destination)
        print(f"OCEAN ZERO {destination}", flush=True)
        return destination
    subprocess.run([
        "curl", "--fail", "--show-error", "--location", "--retry", "8",
        "--retry-all-errors", "--continue-at", "-", "--output", str(temporary), url,
    ], check=True)
    os.replace(temporary, destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    ocean_urls = ocean_zero_urls()
    absent_urls = absent_worldcover_urls()
    absent = []
    scene = next(item for item in manifest["scenes"] if item["scene_id"] == args.scene)
    downloads = []
    by_kind = {}
    for kind, directory_name in TYPE_DIR.items():
        directory = ROOT / "data/ancillary" / directory_name / f"scene_{args.scene}"
        if kind == "dem":
            directory = directory / "glo30_aws"
        paths = []
        for item in scene[kind]:
            if item["url"] in absent_urls:
                absent.append({"kind": kind, "name": item["name"], "url": item["url"]})
                print(f"ABSENT UPSTREAM {item['url']}", flush=True)
                continue
            path = directory / item["name"]
            paths.append(path)
            downloads.append((item["url"], path, item["url"] in ocean_urls))
        by_kind[kind] = paths
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(fetch, downloads))

    gdalbuildvrt = ROOT / "envs/dswx-sar-1.2/bin/gdalbuildvrt"
    for kind, paths in by_kind.items():
        directory = paths[0].parent
        with (directory / "sha256sum.txt").open("w", encoding="utf-8") as output:
            for path in sorted(paths):
                output.write(f"{digest(path)}  {path.name}\n")
        vrt = directory / f"scene{args.scene}_{kind}.vrt"
        subprocess.run([str(gdalbuildvrt), str(vrt), *map(str, paths)], check=True)
    receipt = {
        "scene_id": args.scene,
        "source_manifest": str(MANIFEST),
        "files": {
            kind: [{"path": str(path), "bytes": path.stat().st_size,
                    "sha256": digest(path),
                    "synthetic_ocean_zero": any(
                        url in ocean_urls and candidate == path
                        for url, candidate, _ in downloads
                    )} for path in paths]
            for kind, paths in by_kind.items()
        },
        "absent_upstream_404": absent,
        "synthetic_ocean_policy": (
            "Zero elevation and zero HAND only where both authoritative 1-degree URLs "
            "returned HTTP 404 in data/ancillary/url_audit.json. WorldCover tiles with an "
            "audited HTTP 404 are omitted from the mosaic (no synthetic tile); DSWx reads the "
            "uncovered area as WorldCover No_data (0)."
        ),
    }
    receipt_path = ROOT / "data/ancillary" / f"scene_{args.scene}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(receipt_path), "files": len(downloads)}, indent=2))


if __name__ == "__main__":
    main()
