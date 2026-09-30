#!/bin/bash
# Part S rescoring on one interactive-QOS CPU node (15 scenes in parallel; ~5 min and <= ~6 GB per scene).
# DRY_RUN=1 (default) prints the command only.
set -euo pipefail
OUT=${SWM_RESOLUTION:?Set SWM_RESOLUTION to the output root of the resolution-matched rescoring}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PYTHON_BIN:-python}
DRY_RUN="${DRY_RUN:-1}"
submit() { if [[ "${DRY_RUN}" == 1 ]]; then printf '[dry-run]' >&2; printf ' %q' "$@" >&2; printf '\n' >&2; else "$@"; fi; }
submit salloc -A m4612 -C cpu -q interactive -N 1 -t 00:30:00 -J rev-res-s1s2 \
  srun -n 1 -c 256 bash -c "cd ${OUT} && OMP_NUM_THREADS=4 ${PY} ${REPO}/scripts/eval_10m/s1s2_rescore.py --procs 15"
