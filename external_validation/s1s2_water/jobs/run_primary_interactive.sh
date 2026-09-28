#!/bin/bash
# Primary label-class models (k0 and k16, seed 42) on Earth Engine 10 m inputs for all 15 scenes,
# on one 4-GPU node: four independent processes (one per GPU, CUDA_VISIBLE_DEVICES), not srun steps.
# Then the primary evaluation (released + these two models) into evaluation/models_labelclass_primary/.
# The queued array 58699266 skips any probability this writes (infer.py --skip-existing is the default).
set -uo pipefail
cd /pscratch/sd/r/rohit9/surface_water_validation
python=/global/homes/r/rohit9/.conda/envs/s1aef-train/bin/python
infer=code/surfaceWaterGlobal-release/src/infer.py
runs=/pscratch/sd/r/rohit9/S1ML/training_runs/paper_labelclass_v1/openwater/runs/58321212
export TORCH_HOME=/pscratch/sd/r/rohit9/S1ML/model_cache/torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
log=logs/primary_interactive_${SLURM_JOB_ID:-local}
echo "start $(date -Is) job=${SLURM_JOB_ID:-none} node=$(hostname)"
nvidia-smi -L

worker() {  # gpu scene...
  local gpu=$1; shift
  local scene width run_dir aef status=0
  for scene in "$@"; do
    for width in 16 0; do
      run_dir=${runs}/width_k${width}_seed42
      args=(--model-kind s1)
      if (( width > 0 )); then
        aef=$(find work/model/scene_${scene}/inputs_gee10m -maxdepth 1 -name 'alphaearth_*.tif' -print -quit)
        args=(--model-kind s1aef --aef-path "${aef}")
      fi
      echo "$(date -Is) gpu=${gpu} scene=${scene} k${width}"
      CUDA_VISIBLE_DEVICES=${gpu} "${python}" "${infer}" "${args[@]}" \
        --scenes-root work/model/scene_${scene}/inputs_gee10m \
        --output-root work/model/scene_${scene}/inference_inputs_gee10m --output-tag lc_k${width}_seed42_tta \
        --run-dir "${run_dir}" --train-script "${run_dir}/runtime/train.py" --device cuda \
        --tta --fp16 --tile 512 --overlap 64 --batch-size 4 || { echo "FAILED scene=${scene} k${width}"; status=1; }
    done
  done
  return ${status}
}

worker 0 23 33 53 78 > ${log}_gpu0.log 2>&1 & p0=$!
worker 1 25 35 57 80 > ${log}_gpu1.log 2>&1 & p1=$!
worker 2 29 47 75 82 > ${log}_gpu2.log 2>&1 & p2=$!
worker 3 77 88 89    > ${log}_gpu3.log 2>&1 & p3=$!
failed=0
for p in ${p0} ${p1} ${p2} ${p3}; do wait ${p} || failed=1; done
echo "inference finished $(date -Is) failed=${failed}"
grep -h "FAILED" ${log}_gpu*.log || true
if (( failed )); then exit 1; fi

"${python}" code/evaluate_labelclass_models.py --processes 8 \
  --methods released_s1_only released_s1_aef lc_s1_only_gee_seed42 lc_s1_aef_gee_seed42 \
  --output-dir evaluation/models_labelclass_primary
echo "evaluation finished $(date -Is) exit=$?"
