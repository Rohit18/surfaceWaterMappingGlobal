#!/bin/bash
#SBATCH -J openwater-train
#SBATCH -A m4612_g
#SBATCH -C gpu&hbm40g
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -L scratch
#SBATCH --array=0-29%6
#SBATCH -o /pscratch/sd/r/rohit9/S1ML/training_runs/paper_retrain_openwater_v1/logs/train-%A_%a.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/global/u2/r/rohit9/surfaceWaterMappingGlobal}"
ENV_PREFIX="${ENV_PREFIX:-/global/common/software/m4612/rohit9/conda-envs/s1aef-train}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
MANIFEST_ROOT="${MANIFEST_ROOT:-/pscratch/sd/r/rohit9/S1ML/training_supplement/paper_retrain_v1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/pscratch/sd/r/rohit9/S1ML/training_runs/paper_retrain_openwater_v1}"
TORCH_HOME="${TORCH_HOME:-/pscratch/sd/r/rohit9/S1ML/model_cache/torch}"
SAMPLE_ID="${SAMPLE_ID:-paper_sword4678_plus_openwater600_v1}"

SEEDS=(42 43 44)
WIDTHS=(0 1 2 3 4 8 16)
TASK="${SLURM_ARRAY_TASK_ID}"
ARTIFACT_ARRAY_JOB_ID="${ARTIFACT_ARRAY_JOB_ID:-${SLURM_ARRAY_JOB_ID}}"
SEED="${SEEDS[$((TASK % 3))]}"
MAX_TRAIN_RECORDS=""
if (( TASK < 21 )); then
  WIDTH="${WIDTHS[$((TASK / 3))]}"
  EXPERIMENT="width_k${WIDTH}_seed${SEED}"
else
  SIZES=(1000 2000 3000)
  SIZE="${SIZES[$(((TASK - 21) / 3))]}"
  WIDTH=16
  MAX_TRAIN_RECORDS="${SIZE}"
  EXPERIMENT="train${SIZE}_k16_seed${SEED}"
fi

ARTIFACT_DIR="${ARTIFACT_ROOT}/runs/${ARTIFACT_ARRAY_JOB_ID}/${EXPERIMENT}"
RUNTIME_DIR="${ARTIFACT_DIR}/runtime"
mkdir -p "${RUNTIME_DIR}" "${ARTIFACT_ROOT}/logs" "${TORCH_HOME}"
cp "${REPO_ROOT}/src/train.py" "${RUNTIME_DIR}/train.py"
cp "${REPO_ROOT}/jobs/paper_retrain_array.sbatch" "${RUNTIME_DIR}/paper_retrain_array.sbatch"

CMD=(
  env TORCH_HOME="${TORCH_HOME}" "${PYTHON_BIN}" "${RUNTIME_DIR}/train.py"
  --mode train
  --triplet_manifest "${MANIFEST_ROOT}/triplets.csv"
  --triplet_sample_id "${SAMPLE_ID}"
  --fixed_split_manifest "${MANIFEST_ROOT}/fixed_split.csv"
  --artifact_dir "${ARTIFACT_DIR}"
  --n_s1_bands 3
  --n_aef_bands 64
  --n_proj_bands "${WIDTH}"
  --loss ce_dice
  --batch_size 4
  --crop_size 512
  --num_workers 8
  --seed "${SEED}"
  --split_seed 42
  --train_subset_seed "${SEED}"
  --selection_metric water_iou
  --freeze_epochs 2
  --partial_unfreeze_epochs 2
  --finetune_epochs 12
  --finetune_tail_epochs 10
  --warmup_lr 0.0001
  --partial_unfreeze_lr 0.00001
  --unfreeze_lr 0.000003
  --finetune_tail_lr 0.0000015
  --weight_decay 0.01
  --grad_clip 1.0
  --patience 4
  --min_delta 0.001
  --rotate_max_deg 5.0
  --rotate_p 0.5
  --eval_thresholds 0.30 0.50
  --disable_pin_memory
  --use_fp16
)
if [[ "${WIDTH}" == "0" ]]; then
  CMD+=(--n_aef_bands 0 --n_proj_bands 0)
fi
if [[ -n "${MAX_TRAIN_RECORDS}" ]]; then
  CMD+=(--max_train_records "${MAX_TRAIN_RECORDS}")
fi

printf 'experiment=%s\nseed=%s\nwidth=%s\nmax_train_records=%s\nartifact_dir=%s\n' \
  "${EXPERIMENT}" "${SEED}" "${WIDTH}" "${MAX_TRAIN_RECORDS}" "${ARTIFACT_DIR}"
"${CMD[@]}"
