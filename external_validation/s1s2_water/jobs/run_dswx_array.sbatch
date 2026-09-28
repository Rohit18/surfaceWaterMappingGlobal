#!/bin/bash
#SBATCH --job-name=swval-dswx
#SBATCH --account=m4612
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=23,25,29,33,35,47,53,75,77,78,80,82,88,89%2
#SBATCH --output=/pscratch/sd/r/rohit9/surface_water_validation/logs/dswx_%A_%a.out
#SBATCH --error=/pscratch/sd/r/rohit9/surface_water_validation/logs/dswx_%A_%a.err

set -euo pipefail
validation_root=/pscratch/sd/r/rohit9/surface_water_validation
dswx_env="${validation_root}/envs/dswx-sar-1.2"
scene_work="${validation_root}/work/opera/scene_${SLURM_ARRAY_TASK_ID}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${validation_root}/code/DSWX-SAR-v1.2-boundary-fix/src"
export PATH="${dswx_env}/bin:${PATH}"
export PROJ_LIB="${dswx_env}/share/proj"
export PROJ_DATA="${dswx_env}/share/proj"
export GDAL_DATA="${dswx_env}/share/gdal"
export MPLCONFIGDIR="${scene_work}/dswx_mplconfig"
export XDG_CACHE_HOME="${scene_work}/dswx_fontconfig"
export TMPDIR="${scene_work}/dswx_tmp"
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${TMPDIR}" \
    "${scene_work}/dswx_scratch" "${scene_work}/dswx_output"
cd "${validation_root}"
"${dswx_env}/bin/python" code/prepare_dswx_scene.py --scene "${SLURM_ARRAY_TASK_ID}"
srun --cpu-bind=cores "${dswx_env}/bin/dswx_s1.py" \
    "${scene_work}/dswx_runconfig.yaml" \
    --log "${validation_root}/logs/dswx_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
