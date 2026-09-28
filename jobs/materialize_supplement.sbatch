#!/bin/bash
#SBATCH -J openwater-export
#SBATCH -A m4612
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 12:00:00
#SBATCH --time-min=01:00:00
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -L scratch
#SBATCH --array=0-11%4
#SBATCH -o /pscratch/sd/r/rohit9/S1ML/training_supplement/v1/logs/export-%A_%a.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/global/u2/r/rohit9/surfaceWaterMappingGlobal}"
PYTHON_BIN="${PYTHON_BIN:-/global/u2/r/rohit9/.conda/envs/s1aef-train/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/pscratch/sd/r/rohit9/S1ML/training_supplement/v1}"
NUM_SHARDS="${NUM_SHARDS:-12}"

mkdir -p "${OUTPUT_ROOT}/logs"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/materialize_supplement_samples.py" \
  --manifest "${REPO_ROOT}/outputs/audit/supplement_samples.csv" \
  --output-root "${OUTPUT_ROOT}" \
  --ee-project rohit-global-water \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --minimum-valid-fraction 0.90 \
  --maximum-gap-hours 24 \
  --timeout 300
