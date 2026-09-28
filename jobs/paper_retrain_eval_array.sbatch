#!/bin/bash
#SBATCH -J openwater-eval
#SBATCH -A m4612
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 02:00:00
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -L scratch
#SBATCH --array=0-30%12
#SBATCH -o /pscratch/sd/r/rohit9/S1ML/training_runs/paper_retrain_openwater_v1/logs/eval-%A_%a.out

set -euo pipefail

WORKDIR="${WORKDIR:-/global/homes/r/rohit9/Workspace}"
PYTHON_BIN="${PYTHON_BIN:-/global/u2/r/rohit9/.conda/envs/s1aef-train/bin/python}"
TRAIN_ARRAY_JOB_ID="${TRAIN_ARRAY_JOB_ID:?Set TRAIN_ARRAY_JOB_ID}"
DATA_ROOT="${DATA_ROOT:-/pscratch/sd/r/rohit9/S1ML/intercomparison_s1aef}"
PREDICTION_ROOT_BASE="${PREDICTION_ROOT_BASE:-${DATA_ROOT}/predictions/paper_retrain_openwater_v1/${TRAIN_ARRAY_JOB_ID}}"
REPORT_ROOT="${REPORT_ROOT:-/global/u2/r/rohit9/Workspace/reports/paper_retrain_openwater_v1/${TRAIN_ARRAY_JOB_ID}}"

SEEDS=(42 43 44)
WIDTHS=(0 1 2 3 4 8 16)
TASK="${SLURM_ARRAY_TASK_ID}"
if (( TASK == 30 )); then
  EXPERIMENT="width_k16_seed42"
  TAG="${EXPERIMENT}_tminus1_tta"
  MANIFEST="inference_inputs_nearest_SID_tminus1.csv"
  PRIMARY_THRESHOLD="0.30"
else
  SEED="${SEEDS[$((TASK % 3))]}"
  if (( TASK < 21 )); then
    WIDTH="${WIDTHS[$((TASK / 3))]}"
    EXPERIMENT="width_k${WIDTH}_seed${SEED}"
  else
    SIZES=(1000 2000 3000)
    SIZE="${SIZES[$(((TASK - 21) / 3))]}"
    EXPERIMENT="train${SIZE}_k16_seed${SEED}"
  fi
  TAG="${EXPERIMENT}_current_tta"
  MANIFEST="inference_inputs_primary_protocol_permanent.csv"
  PRIMARY_THRESHOLD="0.50"
fi

if [[ "${AEF_TIME:-current}" == "tminus1" ]]; then
  TAG="${EXPERIMENT}_tminus1_tta"
  MANIFEST="inference_inputs_nearest_SID_tminus1.csv"
  PRIMARY_THRESHOLD="0.30"
fi

OUTPUT_DIR="${REPORT_ROOT}/evaluation/${TAG}"
mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" "${WORKDIR}/code/evaluate_intercomparison_s1aef_predictions.py" \
  --data-root "${DATA_ROOT}" \
  --prediction-root "${PREDICTION_ROOT_BASE}/${TAG}" \
  --manifest-glob "${MANIFEST}" \
  --threshold "${PRIMARY_THRESHOLD}" \
  --sweep-thresholds 0.30 0.50 \
  --output-dir "${OUTPUT_DIR}" \
  --output-prefix "${TAG}"

# The paper's main comparison is reported at 0.30; retain separate per-scene
# metrics for the seed-42 S1-only and acquisition-year AEF models.
if [[ "${AEF_TIME:-current}" != "tminus1" ]] && (( TASK == 0 || TASK == 18 )); then
  PAPER_DIR="${REPORT_ROOT}/evaluation/${TAG}_paper030"
  mkdir -p "${PAPER_DIR}"
  "${PYTHON_BIN}" "${WORKDIR}/code/evaluate_intercomparison_s1aef_predictions.py" \
    --data-root "${DATA_ROOT}" \
    --prediction-root "${PREDICTION_ROOT_BASE}/${TAG}" \
    --manifest-glob "${MANIFEST}" \
    --threshold 0.30 \
    --sweep-thresholds 0.30 0.50 \
    --output-dir "${PAPER_DIR}" \
    --output-prefix "${TAG}_paper030"
fi
