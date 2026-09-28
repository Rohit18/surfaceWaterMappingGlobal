#!/usr/bin/env python3
"""Create OPERA-compatible RTC aliases and a corrected DSWx-S1 runconfig."""

import argparse
import json
import pathlib

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, required=True)
    args = parser.parse_args()
    scene_id = args.scene
    scene_work = ROOT / "work/opera" / f"scene_{scene_id}"
    selection = json.loads(
        (ROOT / "data/rtc" / f"scene_{scene_id}" / "selected_bursts.json").read_text()
    )
    input_directories = []
    for burst_id in selection["unique_burst_ids"]:
        source_directory = scene_work / "rtc_output" / burst_id
        sources = sorted(source_directory.glob("*RTC-S1_*"))
        if not sources:
            raise RuntimeError(f"No RTC burst outputs in {source_directory}")
        alias_directory = scene_work / "dswx_input" / burst_id
        alias_directory.mkdir(parents=True, exist_ok=True)
        for source in sources:
            legacy_prefix = f"S1S2WATER_SCENE{scene_id}_RTC-S1_"
            if source.name.startswith(legacy_prefix):
                alias_name = "OPERA_L2_RTC-S1_" + source.name[len(legacy_prefix):]
            elif source.name.startswith("OPERA_L2_RTC-S1_"):
                alias_name = source.name
            else:
                continue
            destination = alias_directory / alias_name
            if destination.is_symlink() and destination.resolve() == source.resolve():
                continue
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.symlink_to(source.resolve())
        required = list(alias_directory.glob("OPERA_L2_RTC-S1_*_VV.tif"))
        required += list(alias_directory.glob("OPERA_L2_RTC-S1_*_VH.tif"))
        if len(required) != 2:
            raise RuntimeError(f"Missing aliased VV/VH RTC inputs in {alias_directory}")
        input_directories.append(str(alias_directory))

    ancillary = ROOT / "data/ancillary"
    config = {
        "runconfig": {
            "name": f"scene_{scene_id}_dswx_s1_v1_2_posttag_boundary_fix",
            "groups": {
                "pge_name_group": {"pge_name": "DSWX_S1_PGE"},
                "input_file_group": {"input_file_path": input_directories},
                "dynamic_ancillary_file_group": {
                    "dem_file": str(ancillary / "dem" / f"scene_{scene_id}" / "glo30_aws" / f"scene{scene_id}_dem.vrt"),
                    "dem_file_description": "Copernicus GLO-30 Public DEM, AWS Open Data mirror",
                    "worldcover_file": str(ancillary / "worldcover" / f"scene_{scene_id}" / f"scene{scene_id}_worldcover.vrt"),
                    "worldcover_file_description": "ESA WorldCover 2020 v100",
                    "reference_water_file": str(ancillary / "reference_water" / f"scene_{scene_id}" / f"scene{scene_id}_reference_water.vrt"),
                    "reference_water_file_description": "JRC Global Surface Water occurrence v1.4, 1984-2021",
                    "hand_file": str(ancillary / "hand" / f"scene_{scene_id}" / f"scene{scene_id}_hand.vrt"),
                    "hand_file_description": "ASF GLO-30 HAND v1 derived from 2021 Copernicus GLO-30",
                    "algorithm_parameters": str(ROOT / "code/DSWX-SAR-v1.2-boundary-fix/src/dswx_sar/defaults/algorithm_parameter_s1.yaml"),
                },
                "static_ancillary_file_group": {"static_ancillary_inputs_flag": False},
                "primary_executable": {"product_type": "dswx_s1"},
                "product_path_group": {
                    "product_path": str(scene_work / "dswx_output"),
                    "scratch_path": str(scene_work / "dswx_scratch"),
                    "sas_output_path": str(scene_work / "dswx_output"),
                    "product_version": "1.2-posttag-e7f20a",
                    "output_imagery_format": "COG",
                    "output_imagery_compression": "DEFLATE",
                    "output_imagery_nbits": 32,
                },
                "browse_image_group": {
                    "save_browse": True, "browse_image_height": 1024,
                    "browse_image_width": 1024, "flag_collapse_wtr_classes": True,
                    "exclude_inundated_vegetation": False,
                    "set_not_water_to_nodata": False, "set_hand_mask_to_nodata": True,
                    "set_layover_shadow_to_nodata": True,
                    "set_ocean_masked_to_nodata": False, "save_tif_to_output": True,
                },
                "log_file": str(ROOT / "logs" / f"dswx_scene_{scene_id}.log"),
            },
        },
    }
    config_path = scene_work / "dswx_runconfig.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({"scene_id": scene_id, "input_count": len(input_directories),
                      "runconfig": str(config_path)}, indent=2))


if __name__ == "__main__":
    main()
