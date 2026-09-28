#!/usr/bin/env python3
"""Audit HTTP availability for ancillary files that are not already local."""

import concurrent.futures
import json
import pathlib

import requests


ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/ancillary/opera_ancillary_manifest.json"


def destination(scene_id, kind, name):
    path = ROOT / "data/ancillary" / kind / f"scene_{scene_id}"
    if kind == "dem":
        path = path / "glo30_aws"
    return path / name


def check(item):
    scene_id, kind, record = item
    path = destination(scene_id, kind, record["name"])
    if path.is_file() and path.stat().st_size > 0:
        return {"scene_id": scene_id, "kind": kind, "name": record["name"],
                "url": record["url"], "status": "local"}
    response = requests.head(record["url"], timeout=60, allow_redirects=True)
    return {"scene_id": scene_id, "kind": kind, "name": record["name"],
            "url": record["url"], "status": response.status_code}


def main():
    manifest = json.loads(MANIFEST.read_text())
    items = [(scene["scene_id"], kind, record)
             for scene in manifest["scenes"]
             for kind in ("dem", "hand") for record in scene[kind]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(check, items))
    output = ROOT / "data/ancillary/url_audit.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    missing = [item for item in results if item["status"] != "local" and item["status"] != 200]
    print(json.dumps({"checked": len(results), "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
