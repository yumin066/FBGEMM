#!/usr/bin/env bash
# SM120 HSTU attention kernel profiling script.
# Usage:
#   ./run_profile.sh NNN <description>
#   e.g.: ./run_profile.sh 020 phase8_lb1
#
# Captures:
#   1. nsys profile  → 3profile_results/NNN_<desc>.nsys-rep + .sqlite
#   2. ncu --set full (FP8 WS kernel) → 3profile_results/NNN_<desc>_ncu_fp8.ncu-rep
#   3. ncu --set full (BF16 kernel)   → 3profile_results/NNN_<desc>_ncu_bf16.ncu-rep
#   4. nsys stats CSV                 → 3profile_results/NNN_<desc>_stats.log
#   5. ncu CSV export                 → 3profile_results/NNN_<desc>_ncu_fp8.csv
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
BASE="${PROFILE_DIR}/${NNN}_${DESC}"

mkdir -p "${PROFILE_DIR}"
mkdir -p /tmp/claude

COMMIT=$(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
CLOCK=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "unknown")
echo "========================================"
echo "Profiling: ${NNN}_${DESC}"
echo "Commit: ${COMMIT}  GPU clock: ${CLOCK} MHz"
echo "Output base: ${BASE}"
echo "========================================"

# ── 1. nsys profile ──────────────────────────────────────────────────────────
echo ""
echo "[1/4] nsys profile → ${BASE}.nsys-rep"
PYTHONUSERBASE=${PYTHONUSERBASE} \
nsys profile \
    --output "${BASE}" \
    --trace cuda,nvtx \
    --force-overwrite true \
    python3 "${NCU_TARGET}"

# ── 2. nsys stats ────────────────────────────────────────────────────────────
echo ""
echo "[2/4] nsys stats → ${BASE}_stats.log"
nsys stats \
    --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_pushpop_sum \
    --force-export true \
    --format csv \
    "${BASE}.nsys-rep" \
    2>&1 | tee "${BASE}_stats.log"

# ── 3. ncu --set full (FP8 WS kernel) ────────────────────────────────────────
echo ""
echo "[3/4] ncu FP8 WS kernel → ${BASE}_ncu_fp8.ncu-rep"
PYTHONUSERBASE=${PYTHONUSERBASE} \
ncu -f \
    -o "${BASE}_ncu_fp8" \
    --set full \
    --launch-skip 5 \
    --launch-count 1 \
    --kernel-name "hstu_fwd_kernel_sm120_fp8_ws_tma" \
    python3 "${NCU_TARGET}"

# ── 4. ncu --set full (BF16 kernel) ──────────────────────────────────────────
echo ""
echo "[4/4] ncu BF16 kernel → ${BASE}_ncu_bf16.ncu-rep"
PYTHONUSERBASE=${PYTHONUSERBASE} \
ncu -f \
    -o "${BASE}_ncu_bf16" \
    --set full \
    --launch-skip 5 \
    --launch-count 1 \
    --kernel-name "hstu_fwd_kernel_sm120" \
    python3 "${NCU_TARGET}"

# ── 5. Export ncu reports to CSV for easy metric extraction ──────────────────
echo ""
echo "[done] Exporting ncu FP8 report to CSV..."
ncu --import "${BASE}_ncu_fp8.ncu-rep" --csv --print-units base \
    > "${BASE}_ncu_fp8.csv" 2>&1
echo "FP8 CSV rows: $(wc -l < "${BASE}_ncu_fp8.csv")"

echo ""
echo "Exporting ncu BF16 report to CSV..."
ncu --import "${BASE}_ncu_bf16.ncu-rep" --csv --print-units base \
    > "${BASE}_ncu_bf16.csv" 2>&1
echo "BF16 CSV rows: $(wc -l < "${BASE}_ncu_bf16.csv")"

# ── 6. Quick metric summary ───────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Quick metric summary (FP8 WS kernel)"
echo "========================================"
python3 - "${BASE}_ncu_fp8.csv" "${BASE}_ncu_bf16.csv" << 'PYEOF'
import csv, sys

KEYS = [
    "Duration",
    "launch__registers_per_thread",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "sm__inst_executed_pipe_lsu_mem_local_op_ld.sum",
    "sm__inst_executed_pipe_lsu_mem_local_op_st.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "l2_read_hit_rate",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
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

labels = ["FP8 WS", "BF16"]
for label, path in zip(labels, sys.argv[1:]):
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
