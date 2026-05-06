#!/usr/bin/env bash
# SM120 HSTU attention kernel profiling script.
# Usage:
#   ./run_profile.sh NNN <description> [all|bf16|fp8|paged]
#   e.g.: ./run_profile.sh 020 phase8_lb1 paged
#
# Captures:
#   1. nsys profile  → 3profile_results/NNN_<desc>.nsys-rep + .sqlite
#   2. ncu --set full (FP8 WS kernel) → 3profile_results/NNN_<desc>_ncu_fp8.ncu-rep
#   3. ncu --set full (BF16 kernel)   → 3profile_results/NNN_<desc>_ncu_bf16.ncu-rep
#   4. ncu --set full (FP8 paged KV)  → 3profile_results/NNN_<desc>_ncu_paged.ncu-rep
#   5. nsys stats CSV and ncu CSV exports
#
# Prerequisites:
#   - GPU clock locked:  sudo nvidia-smi -lgc <MHz>
#   - .so built and deployed (run pip install first if needed)

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
NCU_TARGET="${REPO}/fbgemm_gpu/experimental/hstu/benchmark/ncu_hstu_attn.py"
PROFILE_DIR="${REPO}/3profile_results"
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user

NNN="${1:-000}"
DESC="${2:-profile}"
TARGETS="${3:-all}"
BASE="${PROFILE_DIR}/${NNN}_${DESC}"
WINDOW="${HSTU_PROFILE_WINDOW:-causal}"
BS="${HSTU_PROFILE_BS:-4}"
SEQ="${HSTU_PROFILE_SEQ:-2048}"
HEADS="${HSTU_PROFILE_HEADS:-16}"
DIM="${HSTU_PROFILE_DIM:-128}"
WARMUP="${HSTU_PROFILE_WARMUP:-5}"
ITERS="${HSTU_PROFILE_ITERS:-3}"

mkdir -p "${PROFILE_DIR}"
mkdir -p /tmp/claude

COMMIT=$(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
CLOCK=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "unknown")
echo "========================================"
echo "Profiling: ${NNN}_${DESC}"
echo "Commit: ${COMMIT}  GPU clock: ${CLOCK} MHz"
echo "Targets: ${TARGETS}  Window: ${WINDOW}  BS=${BS} SEQ=${SEQ} H=${HEADS} D=${DIM}"
echo "Output base: ${BASE}"
echo "========================================"

want_target() {
    local t="$1"
    [[ "${TARGETS}" == "all" || "${TARGETS}" == "${t}" || "${TARGETS}" == *"${t}"* ]]
}

target_arg_for_nsys() {
    case "${TARGETS}" in
        bf16|fp8|paged) echo "${TARGETS}" ;;
        *) echo "all" ;;
    esac
}

# ── 1. nsys profile ──────────────────────────────────────────────────────────
echo ""
echo "[1/4] nsys profile → ${BASE}.nsys-rep"
PYTHONUSERBASE=${PYTHONUSERBASE} \
nsys profile \
    --output "${BASE}" \
    --trace cuda,nvtx \
    --force-overwrite true \
    python3 "${NCU_TARGET}" \
        --target "$(target_arg_for_nsys)" \
        --window "${WINDOW}" \
        --bs "${BS}" --seq "${SEQ}" --heads "${HEADS}" --dim "${DIM}" \
        --warmup "${WARMUP}" --iters "${ITERS}"

# ── 2. nsys stats ────────────────────────────────────────────────────────────
echo ""
echo "[2/4] nsys stats → ${BASE}_stats.log"
nsys stats \
    --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_pushpop_sum \
    --force-export true \
    --format csv \
    "${BASE}.nsys-rep" \
    2>&1 | tee "${BASE}_stats.log"

# ── 3. ncu --set full reports ────────────────────────────────────────────────
if want_target fp8; then
    echo ""
    echo "[3/5] ncu FP8 WS kernel → ${BASE}_ncu_fp8.ncu-rep"
    PYTHONUSERBASE=${PYTHONUSERBASE} \
    ncu -f \
        -o "${BASE}_ncu_fp8" \
        --set full \
        --launch-skip "${WARMUP}" \
        --launch-count 1 \
        --kernel-name "hstu_fwd_kernel_sm120_fp8_ws_tma" \
        python3 "${NCU_TARGET}" \
            --target fp8 --window "${WINDOW}" \
            --bs "${BS}" --seq "${SEQ}" --heads "${HEADS}" --dim "${DIM}" \
            --warmup "${WARMUP}" --iters "${ITERS}"
fi

if want_target paged; then
    echo ""
    echo "[4/5] ncu FP8 paged KV kernel → ${BASE}_ncu_paged.ncu-rep"
    PYTHONUSERBASE=${PYTHONUSERBASE} \
    ncu -f \
        -o "${BASE}_ncu_paged" \
        --set full \
        --launch-skip "${WARMUP}" \
        --launch-count 1 \
        --kernel-name "hstu_fwd_kernel_sm120_fp8_ws_tma" \
        python3 "${NCU_TARGET}" \
            --target paged --window causal \
            --bs "${BS}" --seq "${SEQ}" --heads "${HEADS}" --dim "${DIM}" \
            --warmup "${WARMUP}" --iters "${ITERS}"
fi

if want_target bf16; then
    echo ""
    echo "[5/5] ncu BF16 kernel → ${BASE}_ncu_bf16.ncu-rep"
    PYTHONUSERBASE=${PYTHONUSERBASE} \
    ncu -f \
        -o "${BASE}_ncu_bf16" \
        --set full \
        --launch-skip "${WARMUP}" \
        --launch-count 1 \
        --kernel-name "hstu_fwd_kernel_sm120" \
        python3 "${NCU_TARGET}" \
            --target bf16 --window "${WINDOW}" \
            --bs "${BS}" --seq "${SEQ}" --heads "${HEADS}" --dim "${DIM}" \
            --warmup "${WARMUP}" --iters "${ITERS}"
fi

# ── 5. Export ncu reports to CSV for easy metric extraction ──────────────────
echo ""
echo "[done] Exporting ncu reports to CSV..."
for kind in fp8 paged bf16; do
    if [[ -f "${BASE}_ncu_${kind}.ncu-rep" ]]; then
        ncu --import "${BASE}_ncu_${kind}.ncu-rep" --csv --print-units base \
            > "${BASE}_ncu_${kind}.csv" 2>&1
        echo "${kind} CSV rows: $(wc -l < "${BASE}_ncu_${kind}.csv")"
    fi
done

# ── 6. Quick metric summary ───────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Quick metric summary"
echo "========================================"
SUMMARY_ARGS=()
[[ -f "${BASE}_ncu_fp8.csv" ]] && SUMMARY_ARGS+=("FP8 WS" "${BASE}_ncu_fp8.csv")
[[ -f "${BASE}_ncu_paged.csv" ]] && SUMMARY_ARGS+=("FP8 paged KV" "${BASE}_ncu_paged.csv")
[[ -f "${BASE}_ncu_bf16.csv" ]] && SUMMARY_ARGS+=("BF16" "${BASE}_ncu_bf16.csv")
python3 - "${SUMMARY_ARGS[@]}" << 'PYEOF'
import csv, sys

KEYS = [
    "Duration",
    "Grid Size",
    "Waves Per SM",
    "Registers Per Thread",
    "Local Memory Spilling Requests",
    "Memory Throughput",
    "DRAM Throughput",
    "L1/TEX Hit Rate",
    "L2 Hit Rate",
    "Compute (SM) Throughput",
    "Issue Slots Busy",
    "No Eligible",
    "Eligible Warps Per Scheduler",
    "Achieved Occupancy",
]

def load_csv(path):
    out = {}
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                name = row.get("Metric Name","")
                val  = row.get("Metric Value","")
                unit = row.get("Metric Unit","")
                if name:
                    out[name] = f"{val} {unit}".strip()
    except:
        pass
    return out

for i in range(1, len(sys.argv), 2):
    label, path = sys.argv[i], sys.argv[i + 1]
    data = load_csv(path)
    print(f"\n--- {label} ---")
    for k in KEYS:
        if k in data:
            print(f"  {k}: {data[k]}")
        else:
            # partial match
            matches = [(n,v) for n,v in data.items() if k.lower() in n.lower()]
            for n,v in matches[:2]:
                print(f"  {n}: {v}")
PYEOF

echo ""
echo "All done. Files in ${PROFILE_DIR}/"
ls -lh "${BASE}"* 2>/dev/null | awk '{print $5, $9}'
