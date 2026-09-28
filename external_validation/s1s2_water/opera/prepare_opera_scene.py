#!/usr/bin/env python3
"""Select intersecting SLC bursts and create an exact-grid RTC-S1 runconfig."""

import argparse
import csv
import json
import pathlib

import yaml
from pyproj import Transformer
from s1reader import load_bursts
from shapely.geometry import box
from shapely.ops import transform, unary_union


ROOT = pathlib.Path(__file__).resolve().parents[1]
SLC_MANIFEST = ROOT / "data/rtc/slc_manifest.json"
ORBIT_MANIFEST = ROOT / "data/rtc/orbit_manifest.json"
GRIDS = ROOT / "evaluation/grids_30m.csv"


def grid_for(scene_id):
    with GRIDS.open(newline="") as stream:
        return next(row for row in csv.DictReader(stream) if int(row["scene_id"]) == scene_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, required=True)
    args = parser.parse_args()
    scene_id = args.scene
    slcs = [x for x in json.loads(SLC_MANIFEST.read_text())["records"]
            if x["scene_id"] == scene_id]
    orbits = [x for x in json.loads(ORBIT_MANIFEST.read_text())["records"]
              if x["scene_id"] == scene_id]
    if not slcs or len(orbits) != len(slcs):
        raise RuntimeError(f"Incomplete SLC/orbit lineage for scene {scene_id}")
    grid = grid_for(scene_id)
    epsg = int(grid["crs"].split(":")[1])
    left, bottom, right, top = [float(grid[key]) for key in ("left", "bottom", "right", "top")]
    aoi = box(left, bottom, right, top)
    to_grid = Transformer.from_crs(4326, epsg, always_xy=True).transform
    scene_dir = ROOT / "data/rtc" / f"scene_{scene_id}"
    work_dir = ROOT / "work/opera" / f"scene_{scene_id}"
    occurrences = []
    selected_ids = []
    for slc in sorted(slcs, key=lambda x: x["source_order"]):
        safe_path = scene_dir / slc["file_name"]
        if not safe_path.is_file() or safe_path.stat().st_size != slc["bytes"]:
            raise RuntimeError(f"Missing or incomplete exact SLC: {safe_path}")
        orbit_record = next(x for x in orbits if x["source_order"] == slc["source_order"])
        orbit_path = pathlib.Path(orbit_record["path"])
        for iw in (1, 2, 3):
            for burst in load_bursts(
                str(safe_path), str(orbit_path), iw, "vv", flag_apply_eap=False
            ):
                footprint = transform(to_grid, unary_union(burst.border))
                overlap = footprint.intersection(aoi).area
                if overlap <= 0:
                    continue
                burst_id = str(burst.burst_id)
                if burst_id not in selected_ids:
                    selected_ids.append(burst_id)
                occurrences.append({
                    "burst_id": burst_id,
                    "iw": iw,
                    "source_order": slc["source_order"],
                    "slc_product_id": slc["slc_product_id"],
                    "intersection_area_m2": overlap,
                    "fraction_of_frozen_grid": overlap / aoi.area,
                })
    if not selected_ids:
        raise RuntimeError(f"No SLC bursts intersect frozen grid for scene {scene_id}")
    work_dir.mkdir(parents=True, exist_ok=True)
    selection_path = scene_dir / "selected_bursts.json"
    selection_path.write_text(json.dumps({
        "scene_id": scene_id,
        "selection_rule": "positive-area intersection between s1-reader burst border and frozen 30 m grid in the grid CRS",
        "frozen_grid": {key: grid[key] for key in (
            "crs", "resolution_m", "left", "bottom", "right", "top", "width", "height"
        )},
        "unique_burst_ids": selected_ids,
        "occurrences_in_source_order": occurrences,
    }, indent=2) + "\n")

    safe_paths = [str(scene_dir / x["file_name"]) for x in sorted(slcs, key=lambda x: x["source_order"])]
    orbit_paths = []
    for item in sorted(orbits, key=lambda x: x["source_order"]):
        if item["path"] not in orbit_paths:
            orbit_paths.append(item["path"])
    config = {
        "runconfig": {
            "name": f"s1s2_water_scene_{scene_id}_rtc_v1_0_4",
            "groups": {
                "primary_executable": {"product_type": "RTC_S1"},
                "pge_name_group": {"pge_name": "RTC_S1_PGE"},
                "input_file_group": {
                    "safe_file_path": safe_paths,
                    "orbit_file_path": orbit_paths,
                    "burst_id": selected_ids,
                },
                "dynamic_ancillary_file_group": {
                    "dem_file": str(ROOT / "data/ancillary/dem" / f"scene_{scene_id}" / "glo30_aws" / f"scene{scene_id}_dem.vrt"),
                    "dem_file_description": "Copernicus DEM GLO-30 Public, AWS Open Data tiles",
                },
                "static_ancillary_file_group": {"burst_database_file": None},
                "product_group": {
                    "processing_type": "CUSTOM",
                    "product_version": "1.0.4-retrospective",
                    "product_path": str(ROOT / "products/opera" / f"scene_{scene_id}" / "rtc"),
                    "scratch_path": str(work_dir / "rtc_scratch"),
                    "output_dir": str(work_dir / "rtc_output"),
                    "product_id": (
                        "OPERA_L2_RTC-S1_{burst_id}_{sensing_start_datetime}_"
                        "{processing_datetime}_{sensor}_{pixel_spacing}_"
                        "{product_version}_RETROSPECTIVE"
                    ),
                    "save_bursts": True,
                    "save_mosaics": True,
                    "save_browse": True,
                    "output_imagery_format": "COG",
                    "output_imagery_compression": "DEFLATE",
                    "output_imagery_nbits": 32,
                    "save_secondary_layers_as_hdf5": False,
                    "save_metadata": True,
                },
                "processing": {
                    "check_ancillary_inputs_coverage": True,
                    "polarization": "dual-pol",
                    "num_workers": 14,
                    "apply_absolute_radiometric_correction": True,
                    "apply_thermal_noise_correction": True,
                    "apply_rtc": True,
                    "apply_bistatic_delay_correction": True,
                    "apply_static_tropospheric_delay_correction": True,
                    "rtc": {
                        "output_type": "gamma0", "algorithm_type": "area_projection",
                        "input_terrain_radiometry": "beta0", "rtc_min_value_db": -30,
                        "dem_upsampling": 1,
                    },
                    "geocoding": {
                        "apply_valid_samples_sub_swath_masking": True,
                        "apply_shadow_masking": True,
                        "algorithm_type": "area_projection", "memory_mode": "auto",
                        "save_incidence_angle": True, "save_local_inc_angle": True,
                        "save_projection_angle": False, "save_rtc_anf_projection_angle": False,
                        "save_range_slope": False, "save_nlooks": True, "save_rtc_anf": True,
                        "save_rtc_anf_gamma0_to_sigma0": False, "save_dem": False,
                        "save_mask": True, "abs_rad_cal": 1, "upsample_radargrid": False,
                        "bursts_geogrid": {
                            "output_epsg": epsg, "x_posting": 30, "y_posting": 30,
                            "x_snap": 30, "y_snap": 30,
                            "top_left": {"x": int(left), "y": int(top)},
                            "bottom_right": {"x": int(right), "y": int(bottom)},
                        },
                    },
                    "mosaicking": {
                        "mosaic_mode": "first",
                        "mosaic_geogrid": {
                            "output_epsg": epsg, "x_posting": 30, "y_posting": 30,
                            "x_snap": 30, "y_snap": 30,
                            "top_left": {"x": int(left), "y": int(top)},
                            "bottom_right": {"x": int(right), "y": int(bottom)},
                        },
                    },
                },
            },
        },
    }
    config_path = work_dir / "rtc_runconfig.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({"scene_id": scene_id, "burst_count": len(selected_ids),
                      "occurrence_count": len(occurrences), "runconfig": str(config_path)}, indent=2))


if __name__ == "__main__":
    main()
