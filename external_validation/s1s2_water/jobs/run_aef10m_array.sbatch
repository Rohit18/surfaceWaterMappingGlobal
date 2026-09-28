#!/bin/bash
#SBATCH --job-name=swval-aef10m
#SBATCH --account=m4612
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=23,25,29,33,35,47,53,57,75,77,78,80,82,88,89%8
#SBATCH --output=/pscratch/sd/r/rohit9/surface_water_validation/logs/aef10m_%A_%a.out
#SBATCH --error=/pscratch/sd/r/rohit9/surface_water_validation/logs/aef10m_%A_%a.err
# AlphaEarth on the 10 m grid shared with export_gee_s1_scene.py (benchmark CRS and upper-left corner).

set -euo pipefail
cd /pscratch/sd/r/rohit9/surface_water_validation
export GDAL_CACHEMAX=4096
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tiff,.tif
/global/homes/r/rohit9/.conda/envs/s1aef-train/bin/python code/download_scene_aef.py \
  --scene "${SLURM_ARRAY_TASK_ID}" --pixel-size 10 \
  --output-dir "work/model/scene_${SLURM_ARRAY_TASK_ID}/inputs_gee10m"
