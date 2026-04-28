#!/usr/bin/env bash
# launch_smart_hstu.sh — Submit a pre-collected HSTU SM120 FP8 WS CUDA trace to Smart.
#
# Run on a login node (sc-xterm / computelab-frontend), NOT on a compute node.
# First collect cuda.tgz on a GB202/SM120 compute-node docker/srun environment
# with collect_hstu_trace.sh.
#
# Usage:
#   ./launch_smart_hstu.sh --chip gb202 --trace /path/to/cuda.tgz
#   ./launch_smart_hstu.sh --chip gb202  # uses traces/hstu_latest_cuda.tgz
#   ./launch_smart_hstu.sh --chip gb202 --node <healthy-computelab-node>
set -euo pipefail

PERFSIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PERFSIM_DIR}/.." && pwd)"

FLOW_SMART="${FLOW_SMART:-/home/scratch.svc_compute_arch/release/flow.smart/latest/flow.smart}"
CONFIG="${PERFSIM_DIR}/config.smart.yml"
CHIP="${CHIP:-gb202}"
TRACE="${PERFSIM_DIR}/traces/hstu_latest_cuda.tgz"
USE_SMART2=1
export PATH="${PERFSIM_DIR}/tools:${PATH}"

if [[ ! -x "${FLOW_SMART}" ]]; then
    echo "ERROR: flow.smart not found at: ${FLOW_SMART}" >&2
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Smart config not found at: ${CONFIG}" >&2
    exit 1
fi

COMMIT=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date +"%Y%m%d_%H%M")
OUT_DIR="${PERFSIM_DIR}/results/hstu_trace_${CHIP}_${COMMIT}_${TIMESTAMP}"
NODE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chip) CHIP="$2"; shift 2 ;;
        --dir)  OUT_DIR="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --trace) TRACE="$2"; shift 2 ;;
        --node) NODE_ARGS+=(-node "$2"); shift 2 ;;
        --no-use-smart2) USE_SMART2=0; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "${TRACE}" ]]; then
    echo "ERROR: trace not found at: ${TRACE}" >&2
    echo "Collect it first on the GB202 compute-node docker/srun environment with:" >&2
    echo "  ${PERFSIM_DIR}/collect_hstu_trace.sh" >&2
    exit 1
fi

echo "Submitting Smart from pre-collected HSTU CUDA trace"
echo "  flow.smart : ${FLOW_SMART}"
echo "  Chip       : ${CHIP}"
echo "  Trace      : ${TRACE}"
echo "  Config     : ${CONFIG}"
echo "  Out        : ${OUT_DIR}"
if [[ "${#NODE_ARGS[@]}" -gt 0 ]]; then
    echo "  Nodes      : ${NODE_ARGS[*]}"
fi
echo ""

ARGS=(
    run
    -chip "${CHIP}"
    -pic
    -cuda2ctl
    -enableMorph
    -trace "${TRACE}"
    -dir "${OUT_DIR}"
    -config "${CONFIG}"
)

if [[ "${#NODE_ARGS[@]}" -gt 0 ]]; then
    ARGS+=("${NODE_ARGS[@]}")
fi

if [[ "${USE_SMART2}" == "1" ]]; then
    ARGS+=(-useSmart2)
fi

"${FLOW_SMART}" "${ARGS[@]}"

echo ""
echo "Job submitted. Monitor at: https://compute-nexus.nvidia.com/workflows/runs"
echo "You will receive an email when the run completes."
