#!/bin/bash
# Strict t-1 training and evaluation of k=16 with three seeds.
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/global/u2/r/rohit9/surfaceWaterMappingGlobal}"
export ENV_PREFIX="${ENV_PREFIX:-/global/common/software/m4612/rohit9/conda-envs/s1aef-train}"
export PYTHON_BIN="${ENV_PREFIX}/bin/python"
export MANIFEST_ROOT="${MANIFEST_ROOT:-/pscratch/sd/r/rohit9/S1ML/training_supplement/paper_tminus1_v1}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/pscratch/sd/r/rohit9/S1ML/training_runs/paper_tminus1_v1}"
export SAMPLE_ID=paper_openwater_strict_tminus1_v1
export AEF_TIME=tminus1
TASKS="${TASKS:-18-20%3}"
DRY_RUN="${DRY_RUN:-0}"
submit() {
  if [[ "${DRY_RUN}" == 1 ]]; then
    printf '[dry-run]' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    printf DRYRUN
  else
    "$@"
  fi
}
if [[ "${DRY_RUN}" != 1 ]]; then
  mkdir -p "${ARTIFACT_ROOT}/logs"
  if [[ -e "${ARTIFACT_ROOT}/submission.txt" ]]; then
    echo 'A submission already exists; inspect it before resubmitting.' >&2
    exit 1
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/prepare_tminus1_training.py" prepare --root "${MANIFEST_ROOT}"
fi
export_job="$(submit sbatch --parsable -o "${ARTIFACT_ROOT}/logs/aef-%A_%a.out" "${REPO_ROOT}/jobs/tminus1_materialize.sbatch")"
manifest_job="$(submit sbatch --parsable --dependency="afterok:${export_job}" -o "${ARTIFACT_ROOT}/logs/manifest-%j.out" "${REPO_ROOT}/jobs/tminus1_finalize.sbatch")"
train_job="$(submit sbatch --parsable --job-name=tminus1-train --array="${TASKS}" --dependency="afterok:${manifest_job}" -o "${ARTIFACT_ROOT}/logs/train-%A_%a.out" "${REPO_ROOT}/jobs/paper_retrain_array.sbatch")"
export TRAIN_ARRAY_JOB_ID="${train_job}"
export OUTPUT_ROOT="/pscratch/sd/r/rohit9/S1ML/intercomparison_s1aef/predictions/paper_tminus1_v1/${train_job}"
export PREDICTION_ROOT_BASE="${OUTPUT_ROOT}"
export REPORT_ROOT="${ARTIFACT_ROOT}/reports/${train_job}"
infer_job="$(submit sbatch --parsable --job-name=tminus1-infer --array="${TASKS}" --dependency="afterok:${train_job}" -o "${ARTIFACT_ROOT}/logs/infer-%A_%a.out" "${REPO_ROOT}/jobs/paper_retrain_infer_array.sbatch")"
eval_job="$(submit sbatch --parsable --job-name=tminus1-eval --array="${TASKS}" --dependency="afterok:${infer_job}" -o "${ARTIFACT_ROOT}/logs/eval-%A_%a.out" "${REPO_ROOT}/jobs/paper_retrain_eval_array.sbatch")"
summary="$(printf 'export_job=%s\nmanifest_job=%s\ntrain_job=%s\ninfer_job=%s\neval_job=%s\nreport_root=%s\n' "$export_job" "$manifest_job" "$train_job" "$infer_job" "$eval_job" "$REPORT_ROOT")"
printf '%s\n' "${summary}"
if [[ "${DRY_RUN}" != 1 ]]; then
  printf '%s\n' "${summary}" > "${ARTIFACT_ROOT}/submission.txt"
fi
