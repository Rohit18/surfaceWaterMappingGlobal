#!/bin/bash
#SBATCH -J openwater-manifest
#SBATCH -A m4612
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -L scratch
#SBATCH -o /pscratch/sd/r/rohit9/S1ML/training_supplement/paper_retrain_v1/build-manifest-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/global/u2/r/rohit9/surfaceWaterMappingGlobal}"
PYTHON_BIN="${PYTHON_BIN:-/global/u2/r/rohit9/.conda/envs/s1aef-train/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/pscratch/sd/r/rohit9/S1ML/training_supplement/paper_retrain_v1}"

mkdir -p "${OUTPUT_ROOT}"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_supplemented_training_manifest.py" \
  --output-root "${OUTPUT_ROOT}"
