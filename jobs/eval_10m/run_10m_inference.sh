#!/bin/bash
# 10 m inference for the 53 GSWD scenes (GRSL revision, Task 1). Same code path and arguments as
# repo_snapshot/jobs/paper_retrain_infer_array.sbatch (tile 512, overlap 64, batch 4, four-flip TTA, fp16 default,
# band statistics from each run dir, validate-only pass first); only --input-manifest and --output-root differ.
# Runs are spread over the GPUs of one node as plain background processes (one per GPU), not srun steps.
# Repository copy: paths come from environment variables (scripts/eval_10m/README.md), the inference code is
# copied from scripts/eval_10m/runtime/, and the final grep (which made job 59050165 exit 1 when no error rows
# existed) is replaced by a count of error rows, as in run_ablation_10m_inference.sh.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${SWM_EVAL10M:?Set SWM_EVAL10M to the output root of the 10 m run}
ENV_PREFIX=${ENV_PREFIX:?Set ENV_PREFIX to the conda environment with requirements.txt}
PY=${ENV_PREFIX}/bin/python
RT=${OUT}/task1_10m/runtime
OW=${OPENWATER_RUNS:?Set OPENWATER_RUNS to the run directories of training array 58321212}
TM=${TMINUS1_RUNS:?Set TMINUS1_RUNS to the run directories of training array 58321217}
M=${OUT}/task1_10m/manifests
P=${OUT}/predictions_10m
export TORCH_HOME=${TORCH_HOME:-${OUT}/torch_cache} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8

mkdir -p "${RT}" "${OUT}/logs/infer10m"
for f in infer_s1_probability_models.py infer_pnw_s1aef_tiles.py infer_intercomparison_s1_tiles.py; do
  [[ -f ${RT}/${f} ]] || cp "${REPO}/scripts/eval_10m/runtime/${f}" "${RT}/${f}"
done

# run spec: output_root|run_dir|width|manifest|output_tag
SPECS=(
  "${P}/openwater|${OW}/width_k0_seed42|0|${M}/inputs_10m_t.csv|width_k0_seed42_current_tta"
  "${P}/openwater|${OW}/width_k16_seed42|16|${M}/inputs_10m_t.csv|width_k16_seed42_current_tta"
  "${P}/openwater|${OW}/width_k16_seed42|16|${M}/inputs_10m_tminus1.csv|width_k16_seed42_tminus1_tta"
  "${P}/tminus1|${TM}/width_k16_seed42|16|${M}/inputs_10m_tminus1.csv|width_k16_seed42_tminus1_tta"
  "${P}/openwater|${OW}/width_k0_seed43|0|${M}/inputs_10m_t.csv|width_k0_seed43_current_tta"
  "${P}/openwater|${OW}/width_k0_seed44|0|${M}/inputs_10m_t.csv|width_k0_seed44_current_tta"
  "${P}/openwater|${OW}/width_k16_seed43|16|${M}/inputs_10m_t.csv|width_k16_seed43_current_tta"
  "${P}/openwater|${OW}/width_k16_seed44|16|${M}/inputs_10m_t.csv|width_k16_seed44_current_tta"
  "${P}/openwater_daymosaic|${OW}/width_k0_seed42|0|${M}/inputs_10m_daymosaic_t.csv|width_k0_seed42_current_tta"
  "${P}/openwater_daymosaic|${OW}/width_k16_seed42|16|${M}/inputs_10m_daymosaic_t.csv|width_k16_seed42_current_tta"
  "${P}/openwater_daymosaic|${OW}/width_k16_seed42|16|${M}/inputs_10m_daymosaic_tminus1.csv|width_k16_seed42_tminus1_tta"
  "${P}/tminus1_daymosaic|${TM}/width_k16_seed42|16|${M}/inputs_10m_daymosaic_tminus1.csv|width_k16_seed42_tminus1_tta"
  "${P}/openwater_nomargin|${OW}/width_k0_seed42|0|${M}/inputs_10m_nomargin_t.csv|width_k0_seed42_current_tta"
  "${P}/openwater_nomargin|${OW}/width_k16_seed42|16|${M}/inputs_10m_nomargin_t.csv|width_k16_seed42_current_tta"
  "${P}/openwater_nomargin|${OW}/width_k16_seed42|16|${M}/inputs_10m_nomargin_tminus1.csv|width_k16_seed42_tminus1_tta"
  "${P}/tminus1_nomargin|${TM}/width_k16_seed42|16|${M}/inputs_10m_nomargin_tminus1.csv|width_k16_seed42_tminus1_tta"
)

run_spec() {  # gpu spec
  local gpu=$1 spec=$2 root run_dir width manifest tag common log
  IFS='|' read -r root run_dir width manifest tag <<< "${spec}"
  log=${OUT}/logs/infer10m/$(basename "${root}")__${tag}.log
  common=(--data-root "${OUT}/inputs_10m" --output-root "${root}" --output-tag "${tag}" --input-manifest "${manifest}"
          --run-dir "${run_dir}" --infer-core-script "${RT}/infer_s1_probability_models.py"
          --train-script "${run_dir}/runtime/train.py" --tile 512 --overlap 64 --batch-size 4 --tta)
  {
    echo "$(date -Is) gpu=${gpu} root=${root} run_dir=${run_dir} width=${width} manifest=${manifest} tag=${tag}"
    if [[ "${width}" == "0" ]]; then
      CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${RT}/infer_intercomparison_s1_tiles.py" "${common[@]}" --validate-only &&
      CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${RT}/infer_intercomparison_s1_tiles.py" "${common[@]}"
    else
      CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${RT}/infer_pnw_s1aef_tiles.py" "${common[@]}" --n-proj-bands "${width}" --validate-only &&
      CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${RT}/infer_pnw_s1aef_tiles.py" "${common[@]}" --n-proj-bands "${width}"
    fi
    echo "$(date -Is) exit=$?"
  } > "${log}" 2>&1
  tail -1 "${log}"
}

NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
(( NGPU < 1 )) && NGPU=1
echo "start $(date -Is) job=${SLURM_JOB_ID:-none} node=$(hostname) gpus=${NGPU}"
nvidia-smi -L
worker() {
  local gpu=$1 i
  for (( i = gpu; i < ${#SPECS[@]}; i += NGPU )); do run_spec "${gpu}" "${SPECS[$i]}"; done
}
for (( g = 0; g < NGPU; g++ )); do worker "${g}" & done
wait
echo "end $(date -Is)"
n_err=$(cat ${P}/*/*/scene_tile_summary.csv 2>/dev/null | grep -c ",error," || true)
echo "scene rows with status error: ${n_err}"
exit 0
