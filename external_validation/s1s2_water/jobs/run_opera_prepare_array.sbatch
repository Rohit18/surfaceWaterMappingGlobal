#!/bin/bash
#SBATCH --job-name=swval-rtc-prep
#SBATCH --account=m4612
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=23,25,29,33,35,47,53,75,77,78,80,82,88,89%4
#SBATCH --output=/pscratch/sd/r/rohit9/surface_water_validation/logs/rtc_prep_%A_%a.out
#SBATCH --error=/pscratch/sd/r/rohit9/surface_water_validation/logs/rtc_prep_%A_%a.err

set -euo pipefail
validation_root=/pscratch/sd/r/rohit9/surface_water_validation
scene_work="${validation_root}/work/opera/scene_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${scene_work}/mplconfig" "${scene_work}/fontconfig"
export MPLCONFIGDIR="${scene_work}/mplconfig"
export XDG_CACHE_HOME="${scene_work}/fontconfig"
export PYTHONNOUSERSITE=1
cd "${validation_root}"
envs/rtc-1.0.4/bin/python code/prepare_opera_scene.py --scene "${SLURM_ARRAY_TASK_ID}"
