#!/bin/bash
#SBATCH -J openwater-table
#SBATCH -A m4612
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -L scratch
#SBATCH -o /pscratch/sd/r/rohit9/S1ML/training_runs/paper_retrain_openwater_v1/logs/aggregate-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/global/u2/r/rohit9/surfaceWaterMappingGlobal}"
PYTHON_BIN="${PYTHON_BIN:-/global/u2/r/rohit9/.conda/envs/s1aef-train/bin/python}"
TRAIN_ARRAY_JOB_ID="${TRAIN_ARRAY_JOB_ID:?Set TRAIN_ARRAY_JOB_ID}"
REPORT_ROOT="${REPORT_ROOT:-/global/u2/r/rohit9/Workspace/reports/paper_retrain_openwater_v1/${TRAIN_ARRAY_JOB_ID}}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/aggregate_paper_retrain.py" \
  --report-root "${REPORT_ROOT}" \
  --output-root "${REPO_ROOT}/results/paper_retrain_openwater_v1/${TRAIN_ARRAY_JOB_ID}"
