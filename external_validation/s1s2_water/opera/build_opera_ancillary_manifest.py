#!/usr/bin/env python3
"""Build deterministic OPERA ancillary tile lists from exact SLC footprints."""

import json
import math
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SLC_MANIFEST = ROOT / "data/rtc/slc_manifest.json"
OUTPUT = ROOT / "data/ancillary/opera_ancillary_manifest.json"


def ns(value: int, width: int = 2) -> str:
    return f"{'N' if value >= 0 else 'S'}{abs(value):0{width}d}"


def ew(value: int, width: int = 3) -> str:
    return f"{'E' if value >= 0 else 'W'}{abs(value):0{width}d}"


def jrc_coord(value: int, positive: str, negative: str) -> str:
    return f"{abs(value)}{positive if value >= 0 else negative}"


def bounds(records):
    points = [
        point
        for record in records
        for polygon in record["geometry"]["coordinates"]
        for point in polygon
    ]
    longitudes = [point[0] for point in points]
    latitudes = [point[1] for point in points]
    return min(longitudes), min(latitudes), max(longitudes), max(latitudes)


def build_scene(scene_id, records):
    west, south, east, north = bounds(records)
    one_degree = [
        (latitude, longitude)
        for latitude in range(math.floor(south), math.floor(north) + 1)
        for longitude in range(math.floor(west), math.floor(east) + 1)
    ]
    dem = []
    hand = []
    for latitude, longitude in one_degree:
        stem = (
            f"Copernicus_DSM_COG_10_{ns(latitude)}_00_"
            f"{ew(longitude)}_00"
        )
        dem_name = f"{stem}_DEM.tif"
        hand_name = f"{stem}_HAND.tif"
        dem.append({
            "name": dem_name,
            "url": f"https://copernicus-dem-30m.s3.amazonaws.com/{stem}_DEM/{dem_name}",
        })
        hand.append({
            "name": hand_name,
            "url": f"https://glo-30-hand.s3.us-west-2.amazonaws.com/v1/2021/{hand_name}",
        })

    worldcover = []
    for latitude in range(3 * math.floor(south / 3), 3 * math.floor(north / 3) + 1, 3):
        for longitude in range(3 * math.floor(west / 3), 3 * math.floor(east / 3) + 1, 3):
            name = (
                f"ESA_WorldCover_10m_2020_v100_{ns(latitude)}"
                f"{ew(longitude)}_Map.tif"
            )
            worldcover.append({
                "name": name,
                "url": (
                    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
                    f"v100/2020/map/{name}"
                ),
            })

    reference_water = []
    for jrc_north in range(
        10 * math.ceil(south / 10), 10 * math.ceil(north / 10) + 1, 10
    ):
        for jrc_west in range(
            10 * math.floor(west / 10), 10 * math.floor(east / 10) + 1, 10
        ):
            jrc_name = (
                f"occurrence_{jrc_coord(jrc_west, 'E', 'W')}_"
                f"{jrc_coord(jrc_north, 'N', 'S')}v1_4_2021.tif"
            )
            reference_water.append({
                "name": jrc_name,
                "url": (
                    "https://storage.googleapis.com/global-surface-water/"
                    f"downloads2021/occurrence/{jrc_name}"
                ),
            })
    return {
        "scene_id": scene_id,
        "slc_union_bounds_wgs84": [west, south, east, north],
        "tile_selection_rule": {
            "dem_hand": "all 1-degree southwest-origin tiles intersecting the closed SLC union bbox",
            "worldcover": "all 3-degree southwest-origin tiles intersecting the closed SLC union bbox",
            "reference_water": "all 10-degree tiles intersecting the closed SLC union bbox, named by west and north edges",
        },
        "dem": dem,
        "hand": hand,
        "worldcover": worldcover,
        "reference_water": reference_water,
    }


def main():
    records = json.loads(SLC_MANIFEST.read_text())["records"]
    scene_ids = sorted({record["scene_id"] for record in records})
    scenes = []
    for scene_id in scene_ids:
        scenes.append(build_scene(
            scene_id,
            [record for record in records if record["scene_id"] == scene_id],
        ))
    result = {
        "source_manifest": str(SLC_MANIFEST),
        "scene_count": len(scenes),
        "scenes": scenes,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(OUTPUT),
        "scene_count": len(scenes),
        "tile_counts": {
            str(scene["scene_id"]): {
                key: len(scene[key])
                for key in ("dem", "hand", "worldcover", "reference_water")
            }
            for scene in scenes
        },
    }, indent=2))


if __name__ == "__main__":
    main()
