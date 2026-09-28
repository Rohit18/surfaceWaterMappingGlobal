#!/bin/bash
# Remaining label-class S1S2-Water inference on one 4-GPU node: seeds 43/44 on Earth Engine 10 m inputs and
# seed 42 on the benchmark S1 (input-source sensitivity). Four workers pull tasks from a shared list
# (flock), one process per GPU. Then the full evaluation (all methods) into evaluation/models_labelclass/.
# Replaces queued array 58699266 and evaluation 58699267 (infer.py skips existing outputs).
set -uo pipefail
cd /pscratch/sd/r/rohit9/surface_water_validation
python=/global/homes/r/rohit9/.conda/envs/s1aef-train/bin/python
infer=code/surfaceWaterGlobal-release/src/infer.py
runs=/pscratch/sd/r/rohit9/S1ML/training_runs/paper_labelclass_v1/openwater/runs/58321212
export TORCH_HOME=/pscratch/sd/r/rohit9/S1ML/model_cache/torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
tag=secondary_interactive_${SLURM_JOB_ID:-local}
queue=work/${tag}_tasks.txt
lock=work/${tag}_tasks.lock
echo "start $(date -Is) job=${SLURM_JOB_ID:-none} node=$(hostname)"

# Benchmark-grid tasks first (largest rasters: 6-10 m), then Earth Engine seeds 43/44.
: > "${queue}"
for scene in 25 77 33 35 53 57 75 78 23 29 80 82 88 89 47; do
  for width in 16 0; do echo "${scene} inputs ${width} 42" >> "${queue}"; done
done
for seed in 43 44; do
  for scene in 23 25 29 33 35 47 53 57 75 77 78 80 82 88 89; do
    for width in 16 0; do echo "${scene} inputs_gee10m ${width} ${seed}" >> "${queue}"; done
  done
done
echo "tasks: $(wc -l < "${queue}")"

next_task() {
  flock "${lock}" bash -c "head -n 1 '${queue}'; sed -i '1d' '${queue}'"
}

worker() {
  local gpu=$1 status=0 task scene input_set width seed run_dir args aef
  while true; do
    task=$(next_task)
    [[ -z "${task}" ]] && break
    read -r scene input_set width seed <<< "${task}"
    run_dir=${runs}/width_k${width}_seed${seed}
    args=(--model-kind s1)
    if (( width > 0 )); then
      aef=$(find work/model/scene_${scene}/${input_set} -maxdepth 1 -name 'alphaearth_*.tif' -print -quit)
      args=(--model-kind s1aef --aef-path "${aef}")
    fi
    echo "$(date -Is) gpu=${gpu} ${task}"
    CUDA_VISIBLE_DEVICES=${gpu} "${python}" "${infer}" "${args[@]}" \
      --scenes-root work/model/scene_${scene}/${input_set} \
      --output-root work/model/scene_${scene}/inference_${input_set} --output-tag lc_k${width}_seed${seed}_tta \
      --run-dir "${run_dir}" --train-script "${run_dir}/runtime/train.py" --device cuda \
      --tta --fp16 --tile 512 --overlap 64 --batch-size 4 > /dev/null || { echo "FAILED ${task}"; status=1; }
  done
  return ${status}
}

for gpu in 0 1 2 3; do worker ${gpu} > logs/${tag}_gpu${gpu}.log 2>&1 & pids[gpu]=$!; done
failed=0
for gpu in 0 1 2 3; do wait ${pids[gpu]} || failed=1; done
echo "inference finished $(date -Is) failed=${failed}"
grep -h "FAILED" logs/${tag}_gpu*.log || true
if (( failed )); then exit 1; fi

"${python}" code/evaluate_labelclass_models.py --processes 8
echo "evaluation finished $(date -Is) exit=$?"
