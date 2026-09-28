#!/bin/bash
#SBATCH --job-name=swval-lc-eval
#SBATCH --account=m4612
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/pscratch/sd/r/rohit9/surface_water_validation/logs/lc_evaluation_%j.out
#SBATCH --error=/pscratch/sd/r/rohit9/surface_water_validation/logs/lc_evaluation_%j.err

set -euo pipefail
cd /pscratch/sd/r/rohit9/surface_water_validation
export GDAL_NUM_THREADS=16
/global/homes/r/rohit9/.conda/envs/s1aef-train/bin/python code/evaluate_labelclass_models.py
