#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/global/u2/r/rohit9/surfaceWaterMappingGlobal}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/pscratch/sd/r/rohit9/S1ML}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p \
    "${SCRATCH_ROOT}/training_supplement/v1/logs" \
    "${SCRATCH_ROOT}/training_supplement/paper_retrain_v1" \
    "${SCRATCH_ROOT}/training_runs/paper_retrain_openwater_v1/logs"
fi

submit() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run]' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    printf 'DRYRUN'
  else
    "$@"
  fi
}

export_job="$(submit sbatch --parsable "${REPO_ROOT}/jobs/materialize_supplement.sbatch")"
manifest_job="$(submit sbatch --parsable --dependency="afterok:${export_job}" "${REPO_ROOT}/jobs/build_supplemented_manifest.sbatch")"
train_job="$(submit sbatch --parsable --dependency="afterok:${manifest_job}" "${REPO_ROOT}/jobs/paper_retrain_array.sbatch")"
infer_job="$(submit sbatch --parsable --dependency="afterok:${train_job}" --export=ALL,TRAIN_ARRAY_JOB_ID="${train_job}" "${REPO_ROOT}/jobs/paper_retrain_infer_array.sbatch")"
eval_job="$(submit sbatch --parsable --dependency="afterok:${infer_job}" --export=ALL,TRAIN_ARRAY_JOB_ID="${train_job}" "${REPO_ROOT}/jobs/paper_retrain_eval_array.sbatch")"
aggregate_job="$(submit sbatch --parsable --dependency="afterok:${eval_job}" --export=ALL,TRAIN_ARRAY_JOB_ID="${train_job}" "${REPO_ROOT}/jobs/paper_retrain_aggregate.sbatch")"

printf '%s\n' \
  "export_job=${export_job}" \
  "manifest_job=${manifest_job}" \
  "train_job=${train_job}" \
  "infer_job=${infer_job}" \
  "eval_job=${eval_job}" \
  "aggregate_job=${aggregate_job}"
