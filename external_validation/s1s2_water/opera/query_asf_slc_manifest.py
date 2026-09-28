#!/usr/bin/env python3
"""Resolve each exact benchmark GRD to its same-start, same-orbit ASF SLC."""

import json
import pathlib
import time

import requests


ROOT = pathlib.Path(__file__).resolve().parents[1]
GRD_MANIFEST = ROOT / "data/model/grd_manifest.json"
OUTPUT = ROOT / "data/rtc/slc_manifest.json"
SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"


def query(record):
    platform = "Sentinel-1A" if record["product_id"].startswith("S1A_") else "Sentinel-1B"
    params = {"platform": platform, "processingLevel": "SLC", "beamMode": "IW",
              "start": record["start_time"], "end": record["stop_time"], "output": "geojson"}
    for attempt in range(6):
        response = requests.get(SEARCH_URL, params=params, timeout=120)
        if response.status_code == 429 and attempt < 5:
            time.sleep(3 * (attempt + 1)); continue
        response.raise_for_status()
        features = response.json().get("features", [])
        matches = [feature for feature in features
                   if feature["properties"].get("startTime", "") <= record["start_time"]
                   and feature["properties"].get("stopTime", "") >= record["stop_time"]
                   and int(feature["properties"].get("orbit", -1)) == record["orbit"]
                   and "_IW_SLC__1SDV_" in feature["properties"].get("sceneName", "")]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one exact SLC for {record['product_id']}, found "
                               f"{[feature['properties'].get('sceneName') for feature in matches]}")
        return matches[0]
    raise RuntimeError(f"Retries exhausted for {record['product_id']}")


def main():
    records = []
    for grd in json.loads(GRD_MANIFEST.read_text())["records"]:
        print(f"scene {grd['scene_id']}: {grd['product_id']}", flush=True)
        feature = query(grd)
        properties = feature["properties"]
        records.append({
            "scene_id": grd["scene_id"], "source_order": grd["source_order"],
            "grd_product_id": grd["product_id"], "slc_product_id": properties["sceneName"],
            "url": properties["url"], "file_name": properties["fileName"],
            "bytes": int(properties["bytes"]), "md5": properties["md5sum"],
            "start_time": properties["startTime"], "stop_time": properties["stopTime"],
            "orbit": int(properties["orbit"]), "path_number": int(properties["pathNumber"]),
            "flight_direction": properties["flightDirection"], "geometry": feature["geometry"],
        })
    result = {"search_url": SEARCH_URL,
              "mapping_rule": "same absolute orbit and SLC sensing interval contains GRD sensing interval",
              "record_count": len(records), "records": records}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "record_count": len(records),
                      "total_bytes": sum(record["bytes"] for record in records)}, indent=2))


if __name__ == "__main__":
    main()
