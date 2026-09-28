#!/bin/bash
#SBATCH --job-name=swval-compare
#SBATCH --account=m4612
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/pscratch/sd/r/rohit9/surface_water_validation/logs/intercomparison_%j.out
#SBATCH --error=/pscratch/sd/r/rohit9/surface_water_validation/logs/intercomparison_%j.err

set -euo pipefail
validation_root=/pscratch/sd/r/rohit9/surface_water_validation
cd "${validation_root}"
export MPLCONFIGDIR="${validation_root}/work/intercomparison_mplconfig"
export XDG_CACHE_HOME="${validation_root}/work/intercomparison_cache"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
/global/homes/r/rohit9/.conda/envs/s1aef-train/bin/python code/evaluate_intercomparison.py
