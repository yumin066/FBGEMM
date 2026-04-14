#!/bin/bash
# Dump full SASS for one fp8_ws_tma kernel from the compiled .so (cuobjdump).
#
# Usage:
#   ./dump_sass_fp8_ws.sh
#   SO=/path/to/fbgemm_gpu_experimental_hstu....so ./dump_sass_fp8_ws.sh
#   KERNEL_SUBSTR='...' OUT_FILE=./my.sass ./dump_sass_fp8_ws.sh
#
# Default KERNEL_SUBSTR matches the mangled name for:
#   hstu_fwd_kernel_sm120_fp8_ws_tma< Hstu_fwd_kernel_traits_sm120_fp8_ws< 128,128,128,8, ... > >

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SO="${SO:-${SCRIPT_DIR}/fbgemm_gpu/experimental/hstu/hstu/fbgemm_gpu_experimental_hstu.cpython-312-x86_64-linux-gnu.so}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/4sass_dump_ws}"
# Itanium-mangled substring; must appear on the "Function : ..." line from cuobjdump
KERNEL_SUBSTR="${KERNEL_SUBSTR:-hstu_fwd_kernel_sm120_fp8_ws_tmaI35Hstu_fwd_kernel_traits_sm120_fp8_wsILi128ELi128ELi128ELi8ELb0ELb0ELb0ELb0ELb0ELi0ELb0ELb1ELb1EN7}"
OUT_FILE="${OUT_FILE:-${OUT_DIR}/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full.sass}"
PY=/tmp/parse_sass_fp8_ws_one.py

mkdir -p "${OUT_DIR}"

cat > "${PY}" << 'PYEOF'
import sys, re, os

out_file = sys.argv[1]
kernel_substr = sys.argv[2]

current_lines = []
matched = False

def write_out(lines):
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {out_file} ({len(lines)} lines)")

for line in sys.stdin:
    line = line.rstrip("\n")
    if "Function :" in line:
        if kernel_substr in line:
            if current_lines:
                write_out(current_lines)
            current_lines = [line]
            matched = True
            print(f"  Matched Function line (prefix): {line[:120]}...")
        elif current_lines:
            write_out(current_lines)
            current_lines = []
    elif current_lines:
        current_lines.append(line)

if current_lines:
    write_out(current_lines)

if not matched:
    print("ERROR: no kernel whose Function line contains KERNEL_SUBSTR.", file=sys.stderr)
    print(f"  KERNEL_SUBSTR={kernel_substr!r}", file=sys.stderr)
    sys.exit(1)

print("\n=== Max register index (target kernel) ===")
with open(out_file) as f:
    content = f.read()
regs = [int(x) for x in re.findall(r"\bR(\d+)\b", content)]
maxreg = max(regs) if regs else 0
print(f"  MaxReg=R{maxreg}  ({out_file})")
PYEOF

echo "=== cuobjdump SASS for kernel containing ==="
echo "    ${KERNEL_SUBSTR}"
echo "=== SO: ${SO} ==="
if [[ ! -f "${SO}" ]]; then
  echo "ERROR: .so not found: ${SO}" >&2
  exit 1
fi

set -e
cuobjdump --dump-sass "${SO}" 2>&1 | python3 "${PY}" "${OUT_FILE}" "${KERNEL_SUBSTR}"
echo "=== Done: ${OUT_FILE} ==="
