#!/bin/bash
#SBATCH -J openwater-repair
#SBATCH -A m4612
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -L scratch
#SBATCH -o /pscratch/sd/r/rohit9/S1ML/training_supplement/v1/logs/repair-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/global/u2/r/rohit9/surfaceWaterMappingGlobal}"
PYTHON_BIN="${PYTHON_BIN:-/global/u2/r/rohit9/.conda/envs/s1aef-train/bin/python}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/repair_supplement_coverage.py" \
  --manifest "${REPO_ROOT}/outputs/audit/supplement_samples.csv" \
  --ee-project rohit-global-water
