#!/usr/bin/env bash
# bench_cross_arch.sh — Cross-architecture HSTU attention kernel-only benchmark.
#
# Usage:
#   ./bench_cross_arch.sh                   # build + benchmark (default)
#   ./bench_cross_arch.sh --no-build        # benchmark only (skip compilation)
#   ./bench_cross_arch.sh --causal          # build + causal-only benchmark
#   ./bench_cross_arch.sh --seqlens 1024 2048 4096
#
# Build artifacts are isolated per architecture under .build/sm<N>x/ so that
# machines sharing the same filesystem do not overwrite each other's compiled .so.
#
#   .build/
#     sm8x/lib/python3.12/site-packages/   ← SM80 build
#     sm9x/lib/python3.12/site-packages/   ← SM90 build
#     sm12x/lib/python3.12/site-packages/  ← SM120 build
#
# Environment overrides:
#   MAX_JOBS=16 ./bench_cross_arch.sh --build
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HSTU_SRC="${REPO_ROOT}/fbgemm_gpu/experimental/hstu"
BENCH_PY="${HSTU_SRC}/benchmark/bench_hstu_cross_arch.py"

# ── GPU detection ──────────────────────────────────────────────────────────────
SM_MAJOR=$(python3 -c "import torch; print(torch.cuda.get_device_capability()[0])" 2>/dev/null) || {
    echo "ERROR: Failed to query GPU via PyTorch. Is CUDA/PyTorch installed?" >&2
    exit 1
}
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name())" 2>/dev/null)
echo "Detected GPU : ${GPU_NAME}  (SM${SM_MAJOR}x)"

# ── Per-arch isolated build prefix ────────────────────────────────────────────
ARCH_TAG="sm${SM_MAJOR}x"
ARCH_DIR="${REPO_ROOT}/.build/${ARCH_TAG}"
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ARCH_PYSITE="${ARCH_DIR}/lib/python${PYVER}/site-packages"

# ── Parse flags (consume shell-only flags; forward remaining args to Python) ───
BUILD=1
HAS_OUT_CSV=0
PYTHON_ARGS=()
for arg in "$@"; do
    case "${arg}" in
        --no-build|-n)   BUILD=0 ;;
        --build|-b)      BUILD=1 ;;   # kept for backward compat; now the default
        --out-csv)       HAS_OUT_CSV=1; PYTHON_ARGS+=("${arg}") ;;
        *)               PYTHON_ARGS+=("${arg}") ;;
    esac
done

# ── Optional build ─────────────────────────────────────────────────────────────
if [[ "${BUILD}" -eq 1 ]]; then
    echo "Building HSTU CUDA extension for ${ARCH_TAG} → ${ARCH_DIR} ..."
    mkdir -p "${ARCH_DIR}"
    cd "${HSTU_SRC}"

    # Run build in a subshell so build-only env vars don't leak into the benchmark.
    (
    export HSTU_DISABLE_BACKWARD=TRUE
    export MAX_JOBS="${MAX_JOBS:-32}"

    # Use PYTHONUSERBASE + --user so pip writes only to ARCH_DIR and never
    # touches system site-packages (avoids permission errors when a previous
    # editable install exists in /usr/local/lib/).
    export PYTHONUSERBASE="${ARCH_DIR}"

    case "${SM_MAJOR}" in
        8)
            HSTU_ARCH_LIST="8.0" \
                pip install --no-build-isolation --user . -q
            ;;
        9)
            HSTU_ARCH_LIST="9.0" \
                pip install --no-build-isolation --user . -q
            ;;
        12)
            HSTU_ARCH_LIST="12.0" \
            HSTU_DISABLE_HDIM32=TRUE \
            HSTU_DISABLE_HDIM64=TRUE \
            HSTU_DISABLE_HDIM256=TRUE \
                pip install --no-build-isolation --user . -q
            ;;
        10|11)
            echo "SM${SM_MAJOR}x (Blackwell GB) uses a Python implementation; no CUDA build needed."
            ;;
        *)
            echo "WARNING: Unrecognised SM major version ${SM_MAJOR}; skipping build." >&2
            ;;
    esac

    ) # end build subshell
    cd "${REPO_ROOT}"
    echo "Build complete → ${ARCH_PYSITE}"
fi

# ── Log file ──────────────────────────────────────────────────────────────────
LOG_DIR="${REPO_ROOT}/6cross_bench_results"
mkdir -p "${LOG_DIR}"
COMMIT=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date +"%Y%m%d_%H%M")
# Derive a short filesystem-safe slug from the GPU name:
#   "NVIDIA RTX PRO 6000 Blackwell" → "rtx_pro_6000_blackwell"
#   "NVIDIA H100 80GB HBM3"         → "h100_80gb_hbm3"
GPU_SLUG=$(echo "${GPU_NAME}" \
    | sed 's/^[Nn][Vv][Ii][Dd][Ii][Aa] //' \
    | tr '[:upper:]' '[:lower:]' \
    | tr ' /' '_' \
    | tr -cd '[:alnum:]_-')
LOG_FILE="${LOG_DIR}/${GPU_SLUG}_${COMMIT}_${TIMESTAMP}.log"
CSV_FILE="${LOG_FILE%.log}.csv"
echo "Logging to : ${LOG_FILE}"

# ── Run Python benchmark ───────────────────────────────────────────────────────
# Tell the Python script which arch-specific site-packages to prefer, so it
# loads the correct compiled .so instead of any stale source-tree artifact.
export HSTU_ARCH_PYSITE="${ARCH_PYSITE}"
# Only inject --out-csv when the user hasn't already passed one.
if [[ "${HAS_OUT_CSV}" -eq 0 ]]; then
    PYTHON_ARGS+=(--out-csv "${CSV_FILE}")
fi
python3 "${BENCH_PY}" \
    "${PYTHON_ARGS[@]+"${PYTHON_ARGS[@]}"}" 2>&1 | tee "${LOG_FILE}"
