#!/bin/bash
#SBATCH --job-name=swval-rtc
#SBATCH --account=m4612
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=23,25,29,33,35,47,53,75,77,78,80,82,88,89%2
#SBATCH --output=/pscratch/sd/r/rohit9/surface_water_validation/logs/rtc_%A_%a.out
#SBATCH --error=/pscratch/sd/r/rohit9/surface_water_validation/logs/rtc_%A_%a.err

set -euo pipefail
validation_root=/pscratch/sd/r/rohit9/surface_water_validation
rtc_env="${validation_root}/envs/rtc-1.0.4"
scene_work="${validation_root}/work/opera/scene_${SLURM_ARRAY_TASK_ID}"
export PYTHONNOUSERSITE=1
export PATH="${rtc_env}/bin:${PATH}"
export PROJ_LIB="${rtc_env}/share/proj"
export PROJ_DATA="${rtc_env}/share/proj"
export GDAL_DATA="${rtc_env}/share/gdal"
export MPLCONFIGDIR="${scene_work}/mplconfig"
export XDG_CACHE_HOME="${scene_work}/fontconfig"
export TMPDIR="${scene_work}/tmp"
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=4
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${TMPDIR}" \
    "${validation_root}/products/opera/scene_${SLURM_ARRAY_TASK_ID}/rtc" \
    "${scene_work}/rtc_scratch" "${scene_work}/rtc_output"
cd "${validation_root}"
srun --cpu-bind=cores "${rtc_env}/bin/rtc_s1.py" \
    "${scene_work}/rtc_runconfig.yaml" \
    --log "${validation_root}/logs/rtc_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" \
    --full-log-format
