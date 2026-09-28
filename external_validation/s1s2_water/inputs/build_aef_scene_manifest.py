#!/usr/bin/env python3
"""Map each acquisition-valid S1S2-Water test scene to annual AlphaEarth tiles."""

import csv
import json
import pathlib

import rasterio
from rasterio.warp import transform_bounds


ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/alphaearth/cache/aef_index.csv"
OUTPUT = ROOT / "data/alphaearth/aef_scene_manifest.json"
SCENES = (23, 25, 29, 33, 35, 47, 53, 57, 75, 77, 78, 80, 82, 88, 89)
SOURCE_PREFIX = "https://data.source.coop/tge-labs/aef/"


def overlaps(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def to_url(path):
    path = path.strip()
    if path.startswith("s3://us-west-2.opendata.source.coop/tge-labs/aef/"):
        path = path.replace("s3://us-west-2.opendata.source.coop/tge-labs/aef/", SOURCE_PREFIX, 1)
    elif path.startswith("v1/"):
        path = SOURCE_PREFIX + path
    elif not path.startswith("http"):
        path = SOURCE_PREFIX + path.lstrip("/")
    if path.endswith(".vrt"):
        path = path[:-4] + ".tiff"
    return path


def main():
    scenes = {}
    for scene_id in SCENES:
        metadata_path = ROOT / f"data/s1s2_water/metadata/{scene_id}/sentinel12_{scene_id}_meta.json"
        metadata = json.loads(metadata_path.read_text())
        reference = ROOT / f"data/s1s2_water/test/{scene_id}/sentinel12_s1_{scene_id}_img.tif"
        with rasterio.open(reference) as source:
            bbox = transform_bounds(source.crs, "EPSG:4326", *source.bounds, densify_pts=21)
            grid = {
                "crs": source.crs.to_string(), "transform": list(source.transform),
                "width": source.width, "height": source.height,
            }
        scenes[scene_id] = {
            "scene_id": scene_id,
            "year": int(metadata["properties"]["date_s1"][:4]),
            "date_s1": metadata["properties"]["date_s1"],
            "bbox_wgs84": list(bbox),
            "reference": str(reference),
            "grid": grid,
            "tiles": [],
        }

    with INDEX.open(newline="") as source:
        for row in csv.DictReader(source):
            year = int(row["year"])
            tile_bbox = tuple(float(row[name]) for name in (
                "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"))
            for scene in scenes.values():
                if scene["year"] == year and overlaps(tile_bbox, scene["bbox_wgs84"]):
                    crs = row["crs"]
                    if crs.isdigit():
                        crs = f"EPSG:{crs}"
                    scene["tiles"].append({
                        "url": to_url(row["path"]),
                        "crs": crs,
                        "bbox_wgs84": list(tile_bbox),
                    })
    for scene in scenes.values():
        if not scene["tiles"]:
            raise RuntimeError(f"No AlphaEarth tiles found for scene {scene['scene_id']}")
        scene["tiles"].sort(key=lambda tile: tile["url"])
    result = {"index": str(INDEX), "scenes": scenes}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({scene_id: len(scene["tiles"]) for scene_id, scene in scenes.items()}, indent=2))


if __name__ == "__main__":
    main()
