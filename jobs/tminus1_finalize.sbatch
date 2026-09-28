#!/bin/bash
#SBATCH -J tminus1-manifest
#SBATCH -A m4612
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -L scratch
set -euo pipefail
"${ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/prepare_tminus1_training.py" finalize --root "${MANIFEST_ROOT}"
