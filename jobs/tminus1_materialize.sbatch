#!/bin/bash
#SBATCH -J tminus1-aef
#SBATCH -A m4612
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -L scratch
#SBATCH --array=0-11%4
set -euo pipefail
export GDAL_HTTP_TIMEOUT=300
export GDAL_HTTP_MAX_RETRY=5
export GDAL_HTTP_RETRY_DELAY=10
"${ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/prepare_tminus1_training.py" materialize \
  --root "${MANIFEST_ROOT}" --shard "${SLURM_ARRAY_TASK_ID}" --shards 12
