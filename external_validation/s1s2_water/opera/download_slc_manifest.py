#!/usr/bin/env python3
"""Download and verify exact SLC lineage, optionally for one benchmark scene."""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/rtc/slc_manifest.json"


def md5(path):
    checksum = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, required=True)
    args = parser.parse_args()
    records = [record for record in json.loads(MANIFEST.read_text())["records"]
               if record["scene_id"] == args.scene]
    if not records:
        raise RuntimeError(f"No SLC records for scene {args.scene}")
    cookie_jar = ROOT / f"work/.asf_cookies_scene_{args.scene}"
    os.umask(0o077)
    for index, record in enumerate(records, start=1):
        directory = ROOT / f"data/rtc/scene_{args.scene}"
        destination = directory / record["file_name"]
        partial = destination.with_suffix(destination.suffix + ".part")
        directory.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size == record["bytes"] and md5(destination) == record["md5"]:
            print(f"{index}/{len(records)} verified existing {destination}", flush=True); continue
        print(f"{index}/{len(records)} downloading {record['slc_product_id']}", flush=True)
        subprocess.run(["curl", "--netrc", "--location", "--cookie-jar", str(cookie_jar),
                        "--cookie", str(cookie_jar), "--fail", "--show-error", "--retry", "8",
                        "--retry-all-errors", "--retry-delay", "5", "--continue-at", "-",
                        "--output", str(partial), record["url"]], check=True)
        if partial.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Size mismatch: {partial.stat().st_size} != {record['bytes']}")
        actual = md5(partial)
        if actual != record["md5"]:
            raise RuntimeError(f"MD5 mismatch: {actual} != {record['md5']}")
        partial.replace(destination)
        print(f"verified {destination}", flush=True)


if __name__ == "__main__":
    main()
