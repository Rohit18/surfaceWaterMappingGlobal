#!/bin/bash
#SBATCH -J openwater-infer
#SBATCH -A m4612_g
#SBATCH -C gpu&hbm40g
#SBATCH -q shared
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -L scratch
#SBATCH --array=0-30%6
#SBATCH -o /pscratch/sd/r/rohit9/S1ML/training_runs/paper_retrain_openwater_v1/logs/infer-%A_%a.out

set -euo pipefail

WORKDIR="${WORKDIR:-/global/homes/r/rohit9/Workspace}"
ENV_PREFIX="${ENV_PREFIX:-/global/common/software/m4612/rohit9/conda-envs/s1aef-train}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
TRAIN_ARRAY_JOB_ID="${TRAIN_ARRAY_JOB_ID:?Set TRAIN_ARRAY_JOB_ID to the completed training array job}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/pscratch/sd/r/rohit9/S1ML/training_runs/paper_retrain_openwater_v1}"
DATA_ROOT="${DATA_ROOT:-/pscratch/sd/r/rohit9/S1ML/intercomparison_s1aef}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/predictions/paper_retrain_openwater_v1/${TRAIN_ARRAY_JOB_ID}}"
TORCH_HOME="${TORCH_HOME:-/pscratch/sd/r/rohit9/S1ML/model_cache/torch}"

SEEDS=(42 43 44)
WIDTHS=(0 1 2 3 4 8 16)
TASK="${SLURM_ARRAY_TASK_ID}"
if (( TASK == 30 )); then
  WIDTH=16
  SEED=42
  EXPERIMENT="width_k16_seed42"
  OUTPUT_TAG="${EXPERIMENT}_tminus1_tta"
  INPUT_MANIFEST="${DATA_ROOT}/manifests/inference_inputs_nearest_SID_tminus1.csv"
else
  SEED="${SEEDS[$((TASK % 3))]}"
  if (( TASK < 21 )); then
    WIDTH="${WIDTHS[$((TASK / 3))]}"
    EXPERIMENT="width_k${WIDTH}_seed${SEED}"
  else
    SIZES=(1000 2000 3000)
    SIZE="${SIZES[$(((TASK - 21) / 3))]}"
    WIDTH=16
    EXPERIMENT="train${SIZE}_k16_seed${SEED}"
  fi
  OUTPUT_TAG="${EXPERIMENT}_current_tta"
  INPUT_MANIFEST="${DATA_ROOT}/manifests/inference_inputs_primary_protocol_permanent.csv"
fi

if [[ "${AEF_TIME:-current}" == "tminus1" ]]; then
  OUTPUT_TAG="${EXPERIMENT}_tminus1_tta"
  INPUT_MANIFEST="${DATA_ROOT}/manifests/inference_inputs_nearest_SID_tminus1.csv"
fi

RUN_DIR="${ARTIFACT_ROOT}/runs/${TRAIN_ARRAY_JOB_ID}/${EXPERIMENT}"
RUNTIME_DIR="${OUTPUT_ROOT}/runtime/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${RUNTIME_DIR}" "${OUTPUT_ROOT}" "${TORCH_HOME}"
cp "${WORKDIR}/code/infer_s1_probability_models.py" "${RUNTIME_DIR}/infer_s1_probability_models.py"
cp "${WORKDIR}/code/infer_pnw_s1aef_tiles.py" "${RUNTIME_DIR}/infer_pnw_s1aef_tiles.py"
cp "${WORKDIR}/code/infer_intercomparison_s1_tiles.py" "${RUNTIME_DIR}/infer_intercomparison_s1_tiles.py"

COMMON=(
  --data-root "${DATA_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --output-tag "${OUTPUT_TAG}"
  --input-manifest "${INPUT_MANIFEST}"
  --run-dir "${RUN_DIR}"
  --infer-core-script "${RUNTIME_DIR}/infer_s1_probability_models.py"
  --train-script "${RUN_DIR}/runtime/train.py"
  --tile 512
  --overlap 64
  --batch-size 4
  --tta
)

export SLURM_CPU_BIND="${SLURM_CPU_BIND:-cores}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_HOME
echo "experiment=${EXPERIMENT} run_dir=${RUN_DIR} output_tag=${OUTPUT_TAG} input_manifest=${INPUT_MANIFEST}"
if [[ "${WIDTH}" == "0" ]]; then
  "${PYTHON_BIN}" "${RUNTIME_DIR}/infer_intercomparison_s1_tiles.py" "${COMMON[@]}" --validate-only
  "${PYTHON_BIN}" "${RUNTIME_DIR}/infer_intercomparison_s1_tiles.py" "${COMMON[@]}"
else
  "${PYTHON_BIN}" "${RUNTIME_DIR}/infer_pnw_s1aef_tiles.py" "${COMMON[@]}" --n-proj-bands "${WIDTH}" --validate-only
  "${PYTHON_BIN}" "${RUNTIME_DIR}/infer_pnw_s1aef_tiles.py" "${COMMON[@]}" --n-proj-bands "${WIDTH}"
fi
