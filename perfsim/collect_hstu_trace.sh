#!/usr/bin/env bash
# collect_hstu_trace.sh — Collect CUDA traces for the HSTU SM120 FP8 WS kernel.
#
# Run this on the RTX Pro 6000 GPU machine (SM120).
# The resulting cuda.tgz is then fed to launch_smart_hstu.sh on a login node.
#
# Usage:
#   ./collect_hstu_trace.sh               # causal, bs=8 seq=4096
#   ./collect_hstu_trace.sh --full        # full attention
#   ./collect_hstu_trace.sh --bs 4 --seq 2048 --heads 8 --headdim 128
#   ./collect_hstu_trace.sh --frange 2:2  # capture 3rd kernel invocation (0-indexed)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERFSIM_DIR="${REPO_ROOT}/perfsim"
RUNNER_PY="${PERFSIM_DIR}/run_hstu_kernel_once.py"

# ── cuda_apic_capture.pl path ─────────────────────────────────────────────────
# Use the same APIC version as flow.smart's cudaReplayer (0.1.2026041608261776353168)
# to ensure trace format compatibility. Using "latest" causes format mismatch.
CUDA_APIC="${CUDA_APIC:-/home/scratch.svc_compute_arch/release/cuda_apic/linux64/release/0.1.2026041608261776353168/cuda_apic_capture.pl}"
if [[ ! -x "${CUDA_APIC}" ]]; then
    echo "ERROR: cuda_apic_capture.pl not found at: ${CUDA_APIC}" >&2
    echo "       Set CUDA_APIC env var to the correct path." >&2
    exit 1
fi

# ── Python environment ────────────────────────────────────────────────────────
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHONUSERBASE_DIR="/home/scratch.minyu_gpu/project/.cache/pip-user"
ARCH_PYSITE="${PYTHONUSERBASE_DIR}/lib/python${PYVER}/site-packages"

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

# ── Parse args ────────────────────────────────────────────────────────────────
BS=8
SEQ=4096
HEADS=16
HEADDIM=128
CAUSAL_FLAG="--causal"
FRANGE="3:6"      # 4th invocation (skip 3 init/warmup launches), 0-indexed
ITERS=6           # total invocations (must be > frange end)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)      CAUSAL_FLAG=""; shift ;;
        --causal)    CAUSAL_FLAG="--causal"; shift ;;
        --bs)        BS="$2"; shift 2 ;;
        --seq)       SEQ="$2"; shift 2 ;;
        --heads)     HEADS="$2"; shift 2 ;;
        --headdim)   HEADDIM="$2"; shift 2 ;;
        --iters)     ITERS="$2"; shift 2 ;;
        --frange)    FRANGE="$2"; shift 2 ;;
        *)           echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Output directory ──────────────────────────────────────────────────────────
MASK_TAG=$( [[ -n "${CAUSAL_FLAG}" ]] && echo "causal" || echo "full" )
COMMIT=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date +"%Y%m%d_%H%M")
OUT_DIR="${PERFSIM_DIR}/traces/hstu_sm120_fp8_bs${BS}_seq${SEQ}_h${HEADS}_${MASK_TAG}_${COMMIT}_${TIMESTAMP}"
mkdir -p "${OUT_DIR}"
echo "Output dir : ${OUT_DIR}"

# ── Kernel name regex ─────────────────────────────────────────────────────────
# Matches any instantiation of hstu_fwd_kernel_sm120_fp8_ws_tma (causal or full).
# DumpControl regex is matched against the CUDA kernel's demangled name.
KERNEL_RE="hstu_fwd_kernel_sm120_fp8_ws_tma"
DUMP_CONTROL="(func=${KERNEL_RE}\\w*&frange=${FRANGE})"

# ── Build runner command ──────────────────────────────────────────────────────
RUNNER_ARGS="--bs ${BS} --seq ${SEQ} --heads ${HEADS} --headdim ${HEADDIM} --iters ${ITERS}"
if [[ -n "${CAUSAL_FLAG}" ]]; then
    RUNNER_ARGS="${RUNNER_ARGS} --causal"
fi

echo "Kernel RE  : ${DUMP_CONTROL}"
echo "Runner args: ${RUNNER_ARGS}"
echo ""

# ── Collect traces ────────────────────────────────────────────────────────────
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False disables PyTorch VMM allocator
# (cuMemCreate/cuMemMap), forcing legacy cudaMalloc so cudaReplayer doesn't segfault.
APIC_LD_LIBRARY_PATH="$(strip_ld_path_entry "/usr/local/cuda/compat/lib" "${LD_LIBRARY_PATH:-}")"
PYTHONUSERBASE="${PYTHONUSERBASE_DIR}" \
HSTU_ARCH_PYSITE="${ARCH_PYSITE}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False" \
LD_LIBRARY_PATH="${APIC_LD_LIBRARY_PATH}" \
"${CUDA_APIC}" \
    --app_binary="$(which python3)" \
    --app_cmd_args="${RUNNER_PY} ${RUNNER_ARGS}" \
    --knob DumpControl="${DUMP_CONTROL}" \
    --knob TrackAllModuleFunc=1 \
    --knob PrintLevel=ALL \
    --out_dir="${OUT_DIR}"

TRACE=$(find "${OUT_DIR}" -name "cuda.tgz" 2>/dev/null | head -1)
if [[ -z "${TRACE}" ]]; then
    echo "ERROR: cuda.tgz not found under ${OUT_DIR}" >&2
    exit 1
fi

ln -sfn "${TRACE}" "${PERFSIM_DIR}/traces/hstu_latest_cuda.tgz"

echo ""
echo "Trace collected → ${TRACE}"
echo "Latest symlink → ${PERFSIM_DIR}/traces/hstu_latest_cuda.tgz"
echo ""
echo "Next step: on a login node (sc-xterm / computelab-frontend), run:"
echo "  ${PERFSIM_DIR}/launch_smart_hstu.sh --chip gb202 --trace ${TRACE}"
