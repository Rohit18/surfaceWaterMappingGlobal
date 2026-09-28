#!/bin/bash
#SBATCH --job-name=swval-lc-models
#SBATCH --account=m4612_g
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=23,25,29,33,35,47,53,57,75,77,78,80,82,88,89
#SBATCH --output=/pscratch/sd/r/rohit9/surface_water_validation/logs/lc_models_%A_%a.out
#SBATCH --error=/pscratch/sd/r/rohit9/surface_water_validation/logs/lc_models_%A_%a.err
# Label-class paper reruns (paper_labelclass_v1, openwater array 58321212) on S1S2-Water.
# Primary input: Earth Engine COPERNICUS/S1_GRD at 10 m (inputs_gee10m), as in the paper.
# Sensitivity input: the benchmark's own S1 raster (inputs), as used for the released models.
# Seed 42 runs first for both models; seeds 43/44 follow.

set -euo pipefail
cd /pscratch/sd/r/rohit9/surface_water_validation
python=/global/homes/r/rohit9/.conda/envs/s1aef-train/bin/python
infer=code/surfaceWaterGlobal-release/src/infer.py
runs=/pscratch/sd/r/rohit9/S1ML/training_runs/paper_labelclass_v1/openwater/runs/58321212
scene=${SLURM_ARRAY_TASK_ID}
export TORCH_HOME=/pscratch/sd/r/rohit9/S1ML/model_cache/torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_one() {  # input_set width seed
  local input_set=$1 width=$2 seed=$3
  local inputs=work/model/scene_${scene}/${input_set}
  local run_dir=${runs}/width_k${width}_seed${seed}
  local tag=lc_k${width}_seed${seed}_tta
  local outputs=work/model/scene_${scene}/inference_${input_set}
  local kind=s1 aef_args=()
  if (( width > 0 )); then
    kind=s1aef
    local aef_path
    aef_path=$(find "${inputs}" -maxdepth 1 -type f -name 'alphaearth_*.tif' -print -quit)
    [[ -n "${aef_path}" ]] || { echo "No AlphaEarth input in ${inputs}" >&2; exit 1; }
    aef_args=(--aef-path "${aef_path}")
  fi
  echo "scene=${scene} input_set=${input_set} run=${run_dir} tag=${tag}"
  "${python}" "${infer}" --model-kind "${kind}" --scenes-root "${inputs}" "${aef_args[@]}" \
    --output-root "${outputs}" --output-tag "${tag}" \
    --run-dir "${run_dir}" --train-script "${run_dir}/runtime/train.py" --device cuda \
    --tta --fp16 --tile 512 --overlap 64 --batch-size 4
}

for seed in 42 43 44; do
  for width in 0 16; do
    run_one inputs_gee10m "${width}" "${seed}"
  done
done
for width in 0 16; do
  run_one inputs "${width}" 42
done
