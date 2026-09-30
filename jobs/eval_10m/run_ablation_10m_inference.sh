#!/bin/bash
# 10 m inference for the 24 label-class ablation runs of array 58321212 not run at 10 m on 29 Sep (GRSL revision,
# Task B2): widths k = 1, 2, 3, 4, 8 and training sizes 1,000 / 2,000 / 3,000 tiles (k = 16), seeds 42/43/44.
# Copy of revision_checks_20260929/task1_10m/scripts/run_10m_inference.sh. Unchanged: wrappers, arguments (tile 512,
# overlap 64, batch 4, four-flip TTA, fp16 default, band statistics from each run dir, validate-only pass first),
# 10 m inputs and acquisition-year manifest of the 29 Sep run (read only). Changed: OUT, the run list, and the end
# of the script (the 29 Sep version ended with a grep that returns 1 when no error rows exist).
# Repository copy: paths come from environment variables (scripts/eval_10m/README.md).
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${SWM_ABLATION10M:?Set SWM_ABLATION10M to the output root of the ablation run}
PREV=${SWM_EVAL10M:?Set SWM_EVAL10M to the output root of the 10 m run (inputs and manifests)}
ENV_PREFIX=${ENV_PREFIX:?Set ENV_PREFIX to the conda environment with requirements.txt}
PY=${ENV_PREFIX}/bin/python
RT=${OUT}/runtime
OW=${OPENWATER_RUNS:?Set OPENWATER_RUNS to the run directories of training array 58321212}
M=${PREV}/task1_10m/manifests
P=${OUT}/predictions_10m
export TORCH_HOME=${TORCH_HOME:-${OUT}/torch_cache} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8

mkdir -p "${RT}" "${OUT}/logs/infer10m"
for f in infer_s1_probability_models.py infer_pnw_s1aef_tiles.py infer_intercomparison_s1_tiles.py; do
  [[ -f ${RT}/${f} ]] || cp "${REPO}/scripts/eval_10m/runtime/${f}" "${RT}/${f}"
done

# run spec: output_root|run_dir|width|manifest|output_tag
SPECS=(
  "${P}/openwater|${OW}/width_k1_seed42|1|${M}/inputs_10m_t.csv|width_k1_seed42_current_tta"
  "${P}/openwater|${OW}/width_k1_seed43|1|${M}/inputs_10m_t.csv|width_k1_seed43_current_tta"
  "${P}/openwater|${OW}/width_k1_seed44|1|${M}/inputs_10m_t.csv|width_k1_seed44_current_tta"
  "${P}/openwater|${OW}/width_k2_seed42|2|${M}/inputs_10m_t.csv|width_k2_seed42_current_tta"
  "${P}/openwater|${OW}/width_k2_seed43|2|${M}/inputs_10m_t.csv|width_k2_seed43_current_tta"
  "${P}/openwater|${OW}/width_k2_seed44|2|${M}/inputs_10m_t.csv|width_k2_seed44_current_tta"
  "${P}/openwater|${OW}/width_k3_seed42|3|${M}/inputs_10m_t.csv|width_k3_seed42_current_tta"
  "${P}/openwater|${OW}/width_k3_seed43|3|${M}/inputs_10m_t.csv|width_k3_seed43_current_tta"
  "${P}/openwater|${OW}/width_k3_seed44|3|${M}/inputs_10m_t.csv|width_k3_seed44_current_tta"
  "${P}/openwater|${OW}/width_k4_seed42|4|${M}/inputs_10m_t.csv|width_k4_seed42_current_tta"
  "${P}/openwater|${OW}/width_k4_seed43|4|${M}/inputs_10m_t.csv|width_k4_seed43_current_tta"
  "${P}/openwater|${OW}/width_k4_seed44|4|${M}/inputs_10m_t.csv|width_k4_seed44_current_tta"
  "${P}/openwater|${OW}/width_k8_seed42|8|${M}/inputs_10m_t.csv|width_k8_seed42_current_tta"
  "${P}/openwater|${OW}/width_k8_seed43|8|${M}/inputs_10m_t.csv|width_k8_seed43_current_tta"
  "${P}/openwater|${OW}/width_k8_seed44|8|${M}/inputs_10m_t.csv|width_k8_seed44_current_tta"
  "${P}/openwater|${OW}/train1000_k16_seed42|16|${M}/inputs_10m_t.csv|train1000_k16_seed42_current_tta"
  "${P}/openwater|${OW}/train1000_k16_seed43|16|${M}/inputs_10m_t.csv|train1000_k16_seed43_current_tta"
  "${P}/openwater|${OW}/train1000_k16_seed44|16|${M}/inputs_10m_t.csv|train1000_k16_seed44_current_tta"
  "${P}/openwater|${OW}/train2000_k16_seed42|16|${M}/inputs_10m_t.csv|train2000_k16_seed42_current_tta"
  "${P}/openwater|${OW}/train2000_k16_seed43|16|${M}/inputs_10m_t.csv|train2000_k16_seed43_current_tta"
  "${P}/openwater|${OW}/train2000_k16_seed44|16|${M}/inputs_10m_t.csv|train2000_k16_seed44_current_tta"
  "${P}/openwater|${OW}/train3000_k16_seed42|16|${M}/inputs_10m_t.csv|train3000_k16_seed42_current_tta"
  "${P}/openwater|${OW}/train3000_k16_seed43|16|${M}/inputs_10m_t.csv|train3000_k16_seed43_current_tta"
  "${P}/openwater|${OW}/train3000_k16_seed44|16|${M}/inputs_10m_t.csv|train3000_k16_seed44_current_tta"
)

run_spec() {  # gpu spec
  local gpu=$1 spec=$2 root run_dir width manifest tag common log
  IFS='|' read -r root run_dir width manifest tag <<< "${spec}"
  log=${OUT}/logs/infer10m/$(basename "${root}")__${tag}.log
  common=(--data-root "${PREV}/inputs_10m" --output-root "${root}" --output-tag "${tag}" --input-manifest "${manifest}"
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
