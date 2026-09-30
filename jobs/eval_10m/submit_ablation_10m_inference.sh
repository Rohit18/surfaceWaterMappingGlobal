#!/bin/bash
# Launch scripts/run_ablation_10m_inference.sh on one interactive-QOS GPU node (the shared GPU queue had ~4,070 pending jobs on
# 2026-09-28); used again for the ablation runs (Task B2, 30 Sep). DRY_RUN=1 (default) prints the command only, as in submit_paper_labelclass.sh.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DRY_RUN="${DRY_RUN:-1}"
submit() { if [[ "${DRY_RUN}" == 1 ]]; then printf '[dry-run]' >&2; printf ' %q' "$@" >&2; printf '\n' >&2; else "$@"; fi; }
submit salloc -A m4612_g -C gpu -q interactive -N 1 --gpus-per-node 4 -t 00:40:00 -J rev10m-abl \
  srun -n 1 -c 128 --gpus-per-task 4 bash "${HERE}/run_ablation_10m_inference.sh"
