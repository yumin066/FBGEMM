#!/usr/bin/env bash
# collect_hstu_cpp_trace.sh - Collect CUDA traces for the C++ HSTU FP8 WS kernel.
#
# Run this on an RTX Pro 6000 / SM120 GPU machine. This script uses
# docker_computelab.sh by default because the C++ extension runner depends on
# the older HSTU/PyTorch environment from the sm120 branch.
#
# Usage:
#   ./perfsim/collect_hstu_cpp_trace.sh
#   ./perfsim/collect_hstu_cpp_trace.sh --full --bs 1 --seq 2048 --heads 1 --headdim 256
#   ./perfsim/collect_hstu_cpp_trace.sh --causal --frange 3:3
#   HSTU_CPP_TRACE_USE_DOCKER=0 ./perfsim/collect_hstu_cpp_trace.sh  # already in C++ env
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERFSIM_DIR="${REPO_ROOT}/perfsim"
RUNNER_PY="${PERFSIM_DIR}/run_hstu_kernel_once.py"

# Use the same APIC version as flow.smart's cudaReplayer to keep trace format
# compatibility. Using "latest" has caused replayer format mismatches before.
CUDA_APIC="${CUDA_APIC:-/home/scratch.svc_compute_arch/release/cuda_apic/linux64/release/0.1.2026041608261776353168/cuda_apic_capture.pl}"
if [[ ! -x "${CUDA_APIC}" ]]; then
    echo "ERROR: cuda_apic_capture.pl not found at: ${CUDA_APIC}" >&2
    echo "       Set CUDA_APIC env var to the correct path." >&2
    exit 1
fi

if [[ ! -f "${RUNNER_PY}" ]]; then
    echo "ERROR: C++ FP8 runner not found at: ${RUNNER_PY}" >&2
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

python_site_for_current_python() {
    local pythonuserbase="$1"
    python3 - <<PY
import sys
print("${pythonuserbase}/lib/python%d.%d/site-packages" % (
    sys.version_info.major,
    sys.version_info.minor,
))
PY
}

# -- Parse args ----------------------------------------------------------------
BS=1
SEQ=2048
HEADS=1
HEADDIM=256
CAUSAL_FLAG=""
FRANGE="3:3"       # 4th target-kernel invocation.
ITERS=6            # total target-kernel invocations; must exceed frange end.
USE_DOCKER="${HSTU_CPP_TRACE_USE_DOCKER:-1}"
KERNEL_RE="${HSTU_CPP_TRACE_KERNEL_RE:-.*hstu_fwd_kernel_.*fp8_ws_tma.*}"
PRINT_LEVEL="${HSTU_CPP_TRACE_PRINT_LEVEL:-INFO}"
PYTHONUSERBASE_DIR="${HSTU_CPP_TRACE_PYTHONUSERBASE:-/home/scratch.minyu_gpu/project/.cache/pip-user}"

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
        --kernel-re) KERNEL_RE="$2"; shift 2 ;;
        --print-level) PRINT_LEVEL="$2"; shift 2 ;;
        --use-docker) USE_DOCKER=1; shift ;;
        --no-docker) USE_DOCKER=0; shift ;;
        *)           echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# -- Output directory ----------------------------------------------------------
MASK_TAG=$( [[ -n "${CAUSAL_FLAG}" ]] && echo "causal" || echo "full" )
COMMIT=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date +"%Y%m%d_%H%M")
OUT_DIR="${PERFSIM_DIR}/traces/hstu_cpp_fp8_bs${BS}_seq${SEQ}_h${HEADS}_hdim${HEADDIM}_${MASK_TAG}_${COMMIT}_${TIMESTAMP}"
mkdir -p "${OUT_DIR}"

# DumpControl regex is matched against the CUDA kernel's demangled name. Current
# sm120_cutile C++ builds use hstu_fwd_kernel_blackwell_rtx_fp8_ws_tma; older
# sm120 builds used hstu_fwd_kernel_sm120_fp8_ws_tma.
DUMP_CONTROL="(func=${KERNEL_RE}&frange=${FRANGE})"

RUNNER_ARGS="${RUNNER_PY} --bs ${BS} --seq ${SEQ} --heads ${HEADS} --headdim ${HEADDIM} --iters ${ITERS}"
if [[ -n "${CAUSAL_FLAG}" ]]; then
    RUNNER_ARGS="${RUNNER_ARGS} --causal"
fi

echo "Output dir : ${OUT_DIR}"
echo "Kernel RE  : ${DUMP_CONTROL}"
echo "Runner args: ${RUNNER_ARGS}"
echo "Use docker : ${USE_DOCKER}"
echo ""

run_capture_direct() {
    local apic_ld_library_path
    local arch_pysite

    apic_ld_library_path="$(strip_ld_path_entry "/usr/local/cuda/compat/lib" "${LD_LIBRARY_PATH:-}")"
    arch_pysite="$(python_site_for_current_python "${PYTHONUSERBASE_DIR}")"

    PYTHONUSERBASE="${PYTHONUSERBASE_DIR}" \
    HSTU_ARCH_PYSITE="${arch_pysite}" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False" \
    LD_LIBRARY_PATH="${apic_ld_library_path}" \
    "${CUDA_APIC}" \
        --app_binary="$(which python3)" \
        --app_cmd_args="${RUNNER_ARGS}" \
        --knob DumpControl="${DUMP_CONTROL}" \
        --knob TrackAllModuleFunc=1 \
        --knob PrintLevel="${PRINT_LEVEL}" \
        --out_dir="${OUT_DIR}"
}

run_capture_docker() {
    local docker_computelab="${REPO_ROOT}/docker_computelab.sh"
    if [[ ! -x "${docker_computelab}" ]]; then
        echo "ERROR: docker_computelab.sh not executable at: ${docker_computelab}" >&2
        echo "       Pull it from the sm120 branch or set HSTU_CPP_TRACE_USE_DOCKER=0 inside a C++ FP8 env." >&2
        exit 1
    fi

    "${docker_computelab}" bash -lc "
set -euo pipefail
cd \"${REPO_ROOT}\"
export PYTHONUSERBASE=\"${PYTHONUSERBASE_DIR}\"
export HSTU_ARCH_PYSITE=\"${PYTHONUSERBASE_DIR}/lib/\$(python3 -c 'import sys; print(f\"python{sys.version_info.major}.{sys.version_info.minor}\")')/site-packages\"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export LD_LIBRARY_PATH=\"\$(python3 - <<'PY'
import os
parts = [
    part for part in os.environ.get('LD_LIBRARY_PATH', '').split(':')
    if part and part != '/usr/local/cuda/compat/lib'
]
print(':'.join(parts))
PY
)\"

\"${CUDA_APIC}\" \
    --app_binary=\"\$(which python3)\" \
    --app_cmd_args=\"${RUNNER_ARGS}\" \
    --knob DumpControl=\"${DUMP_CONTROL}\" \
    --knob TrackAllModuleFunc=1 \
    --knob PrintLevel=\"${PRINT_LEVEL}\" \
    --out_dir=\"${OUT_DIR}\"
"
}

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False disables PyTorch VMM
# allocator, forcing legacy cudaMalloc so cudaReplayer is less likely to fail.
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

LATEST_HDIM_TRACE="${PERFSIM_DIR}/traces/hstu_cpp_fp8_latest_hdim${HEADDIM}_cuda.tgz"
ln -sfn "${TRACE}" "${LATEST_HDIM_TRACE}"
ln -sfn "${TRACE}" "${PERFSIM_DIR}/traces/hstu_cpp_fp8_latest_cuda.tgz"
ln -sfn "${TRACE}" "${PERFSIM_DIR}/traces/hstu_latest_cuda.tgz"

echo ""
echo "Trace collected -> ${TRACE}"
echo "Latest C++ hdim symlink -> ${LATEST_HDIM_TRACE}"
echo "Latest C++ symlink -> ${PERFSIM_DIR}/traces/hstu_cpp_fp8_latest_cuda.tgz"
echo "Compat symlink -> ${PERFSIM_DIR}/traces/hstu_latest_cuda.tgz"
echo ""
echo "Next step: on a login node (sc-xterm / computelab-frontend), run:"
echo "  ${PERFSIM_DIR}/launch_smart_hstu.sh --chip gb202 --trace ${TRACE}"
