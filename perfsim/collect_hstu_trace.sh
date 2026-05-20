#!/usr/bin/env bash
# collect_hstu_trace.sh - Collect CUDA traces for the SM120 cuTile FP8 D256 kernel.
#
# Run this on an RTX Pro 6000 / SM120 GPU machine.  The script uses the repo-local
# docker_cutile.sh wrapper by default so that cuda.tile, nv-triton tileiras, and
# the cuTile PYTHONPATH are the same as the validated cuTile benchmark setup.
#
# Usage:
#   ./perfsim/collect_hstu_trace.sh
#   ./perfsim/collect_hstu_trace.sh --full
#   ./perfsim/collect_hstu_trace.sh --causal --bs 1 --seq 2048 --heads 1
#   ./perfsim/collect_hstu_trace.sh --tile-m 32 --tile-n 64 --frange 3:3
#   HSTU_CUTILE_TRACE_USE_DOCKER=0 ./perfsim/collect_hstu_trace.sh  # already in cuTile env
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERFSIM_DIR="${REPO_ROOT}/perfsim"
RUNNER_MODULE="hstu_blackwell_sm120_cudatile_fp8_ws.sweep_d256"

# ── cuda_apic_capture.pl path ─────────────────────────────────────────────────
# Use the same APIC version as flow.smart's cudaReplayer (0.1.2026041608261776353168)
# to ensure trace format compatibility. Using "latest" causes format mismatch.
CUDA_APIC="${CUDA_APIC:-/home/scratch.svc_compute_arch/release/cuda_apic/linux64/release/0.1.2026041608261776353168/cuda_apic_capture.pl}"
if [[ ! -x "${CUDA_APIC}" ]]; then
    echo "ERROR: cuda_apic_capture.pl not found at: ${CUDA_APIC}" >&2
    echo "       Set CUDA_APIC env var to the correct path." >&2
    exit 1
fi

strip_ld_path_entry() {
    local entry_to_strip="$1"
    local old_path="${2:-}"
    local new_path=""
    local entry

    IFS=':' read -r -a entries <<< "${old_path}"
    for entry in "${entries[@]}"; do
        if [[ -n "${entry}" && "${entry}" != "${entry_to_strip}" ]]; then
            if [[ -n "${new_path}" ]]; then
                new_path="${new_path}:${entry}"
            else
                new_path="${entry}"
            fi
        fi
    done

    echo "${new_path}"
}

# -- Parse args ----------------------------------------------------------------
BS=1
SEQ=2048
HEADS=1
HEADDIM=256
CAUSAL_FLAG=""
FRANGE="3:3"      # 4th kernel invocation: warmup=3, first measured launch.
WARMUP=3
ITERS=1
TILE_M="${HSTU_CUTILE_FP8_TILE_M:-32}"
TILE_N="${HSTU_CUTILE_FP8_TILE_N:-64}"
OCCUPANCY="${HSTU_CUTILE_FP8_OCCUPANCY:-1}"
NUM_CTAS="${HSTU_CUTILE_FP8_NUM_CTAS:-}"
WORKER_WARPS="${HSTU_CUTILE_FP8_WORKER_WARPS:-}"
USE_DOCKER="${HSTU_CUTILE_TRACE_USE_DOCKER:-1}"
KERNEL_RE="${HSTU_CUTILE_TRACE_KERNEL_RE:-.*hstu_fp8_d256.*}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)      CAUSAL_FLAG=""; shift ;;
        --causal)    CAUSAL_FLAG="--causal"; shift ;;
        --bs)        BS="$2"; shift 2 ;;
        --seq)       SEQ="$2"; shift 2 ;;
        --heads)     HEADS="$2"; shift 2 ;;
        --headdim)   HEADDIM="$2"; shift 2 ;;
        --warmup)    WARMUP="$2"; shift 2 ;;
        --iters)     ITERS="$2"; shift 2 ;;
        --frange)    FRANGE="$2"; shift 2 ;;
        --tile-m)    TILE_M="$2"; shift 2 ;;
        --tile-n)    TILE_N="$2"; shift 2 ;;
        --occupancy) OCCUPANCY="$2"; shift 2 ;;
        --num-ctas)  NUM_CTAS="$2"; shift 2 ;;
        --worker-warps) WORKER_WARPS="$2"; shift 2 ;;
        --kernel-re) KERNEL_RE="$2"; shift 2 ;;
        --use-docker) USE_DOCKER=1; shift ;;
        --no-docker) USE_DOCKER=0; shift ;;
        *)           echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "${HEADDIM}" != "256" ]]; then
    echo "ERROR: cuTile FP8 trace script only supports D256; got --headdim ${HEADDIM}" >&2
    exit 1
fi

# -- Output directory ----------------------------------------------------------
MASK_TAG=$( [[ -n "${CAUSAL_FLAG}" ]] && echo "causal" || echo "full" )
COMMIT=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date +"%Y%m%d_%H%M")
TILE_TAG="${TILE_M}x${TILE_N}"
OUT_DIR="${PERFSIM_DIR}/traces/cutile_fp8_${TILE_TAG}_bs${BS}_seq${SEQ}_h${HEADS}_${MASK_TAG}_${COMMIT}_${TIMESTAMP}"
mkdir -p "${OUT_DIR}"
echo "Output dir : ${OUT_DIR}"

# -- Kernel name regex ---------------------------------------------------------
# DumpControl regex is matched against the CUDA kernel's demangled name.  cuTile
# generated names can vary across toolchains, so keep the default broad.
DUMP_CONTROL="(func=${KERNEL_RE}&frange=${FRANGE})"

# -- Build runner command ------------------------------------------------------
RUNNER_ARGS="-m ${RUNNER_MODULE} --batch ${BS} --seqlen ${SEQ} --heads ${HEADS} --warmup ${WARMUP} --iters ${ITERS}"
if [[ -n "${CAUSAL_FLAG}" ]]; then
    RUNNER_ARGS="${RUNNER_ARGS} --causal"
fi

echo "Kernel RE  : ${DUMP_CONTROL}"
echo "Runner args: ${RUNNER_ARGS}"
echo "cuTile cfg : TILE_M=${TILE_M} TILE_N=${TILE_N} OCCUPANCY=${OCCUPANCY} NUM_CTAS=${NUM_CTAS:-unset} WORKER_WARPS=${WORKER_WARPS:-unset}"
echo "Use docker : ${USE_DOCKER}"
echo ""

run_capture_direct() {
    local apic_ld_library_path
    apic_ld_library_path="$(strip_ld_path_entry "/usr/local/cuda/compat/lib" "${LD_LIBRARY_PATH:-}")"

    HSTU_CUTILE_FP8_TILE_M="${TILE_M}" \
    HSTU_CUTILE_FP8_TILE_N="${TILE_N}" \
    HSTU_CUTILE_FP8_OCCUPANCY="${OCCUPANCY}" \
    HSTU_CUTILE_FP8_NUM_CTAS="${NUM_CTAS}" \
    HSTU_CUTILE_FP8_WORKER_WARPS="${WORKER_WARPS}" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False" \
    LD_LIBRARY_PATH="${apic_ld_library_path}" \
    "${CUDA_APIC}" \
        --app_binary="$(which python3)" \
        --app_cmd_args="${RUNNER_ARGS}" \
        --knob DumpControl="${DUMP_CONTROL}" \
        --knob TrackAllModuleFunc=1 \
        --knob PrintLevel=ALL \
        --out_dir="${OUT_DIR}"
}

run_capture_docker() {
    local docker_cutile="${REPO_ROOT}/docker_cutile.sh"
    if [[ ! -x "${docker_cutile}" ]]; then
        echo "ERROR: docker_cutile.sh not executable at: ${docker_cutile}" >&2
        echo "       Set HSTU_CUTILE_TRACE_USE_DOCKER=0 if you are already in the cuTile environment." >&2
        exit 1
    fi

    HSTU_CUTILE_FP8_TILE_M="${TILE_M}" \
    HSTU_CUTILE_FP8_TILE_N="${TILE_N}" \
    HSTU_CUTILE_FP8_OCCUPANCY="${OCCUPANCY}" \
    HSTU_CUTILE_FP8_NUM_CTAS="${NUM_CTAS}" \
    HSTU_CUTILE_FP8_WORKER_WARPS="${WORKER_WARPS}" \
    CUTILE_USE_NVTRITON_TILEIRAS="${CUTILE_USE_NVTRITON_TILEIRAS:-1}" \
    CUTILE_DOCKER_PULL="${CUTILE_DOCKER_PULL:-0}" \
    "${docker_cutile}" bash -lc "
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

\"${CUDA_APIC}\" \
    --app_binary=\"\$(which python3)\" \
    --app_cmd_args=\"${RUNNER_ARGS}\" \
    --knob DumpControl=\"${DUMP_CONTROL}\" \
    --knob TrackAllModuleFunc=1 \
    --knob PrintLevel=ALL \
    --out_dir=\"${OUT_DIR}\"
"
}

# -- Collect traces ------------------------------------------------------------
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False disables PyTorch VMM allocator
# (cuMemCreate/cuMemMap), forcing legacy cudaMalloc so cudaReplayer doesn't segfault.
if [[ "${USE_DOCKER}" == "1" ]]; then
    run_capture_docker
else
    run_capture_direct
fi

TRACE=$(find "${OUT_DIR}" -name "cuda.tgz" 2>/dev/null | head -1)
if [[ -z "${TRACE}" ]]; then
    echo "ERROR: cuda.tgz not found under ${OUT_DIR}" >&2
    exit 1
fi

LATEST_TRACE="${PERFSIM_DIR}/traces/cutile_fp8_latest_${TILE_TAG}_cuda.tgz"
ln -sfn "${TRACE}" "${LATEST_TRACE}"
ln -sfn "${TRACE}" "${PERFSIM_DIR}/traces/cutile_fp8_latest_cuda.tgz"
ln -sfn "${TRACE}" "${PERFSIM_DIR}/traces/hstu_latest_cuda.tgz"

echo ""
echo "Trace collected → ${TRACE}"
echo "Latest symlink → ${LATEST_TRACE}"
echo "cuTile symlink → ${PERFSIM_DIR}/traces/cutile_fp8_latest_cuda.tgz"
echo "Compat symlink → ${PERFSIM_DIR}/traces/hstu_latest_cuda.tgz"
echo ""
echo "Next step: on a login node (sc-xterm / computelab-frontend), run:"
echo "  ${PERFSIM_DIR}/launch_smart_hstu.sh --chip gb202 --trace ${TRACE}"
