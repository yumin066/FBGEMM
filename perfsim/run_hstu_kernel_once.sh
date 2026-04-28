#!/usr/bin/env bash
# run_hstu_kernel_once.sh — Shell wrapper for run_hstu_kernel_once.py.
# Used as the workload `binary` in workload_hstu_sm120.yml so that env vars
# (PYTHONUSERBASE, HSTU_ARCH_PYSITE) are set before python3 is invoked.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "${SCRIPT_DIR}/run_hstu_kernel_once.py" \
    --bs 8 --seq 4096 --heads 16 --headdim 128 --causal --iters 6
