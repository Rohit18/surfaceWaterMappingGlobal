#!/bin/bash
#SBATCH --job-name=swval-ancillary
#SBATCH --account=m4612
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=23,25,29,33,35,47,53,57,75,77,78,80,82,88,89%4
#SBATCH --output=/pscratch/sd/r/rohit9/surface_water_validation/logs/ancillary_%A_%a.out
#SBATCH --error=/pscratch/sd/r/rohit9/surface_water_validation/logs/ancillary_%A_%a.err

set -euo pipefail
cd /pscratch/sd/r/rohit9/surface_water_validation
python3 code/download_scene_ancillary.py --scene "${SLURM_ARRAY_TASK_ID}" --workers 8
