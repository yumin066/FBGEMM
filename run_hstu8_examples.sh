#!/usr/bin/env bash
# Run 14 explicit @example cases for HSTU8Test (quant_mode=2, SM120 FP8 block-scale).
# Covers causal / +rab / +drab / local / context / target / arbitrary at seq=128 and seq=256.
#
# Usage:
#   bash run_hstu8_examples.sh [log_file]
#
# Default log: 1test_results/NNN_hstu8_examples_qm2.log (auto-numbered)

REPO="$(cd "$(dirname "$0")" && pwd)"

if [ -n "$1" ]; then
    LOG="$1"
else
    N=$(ls "${REPO}/1test_results/"*.log 2>/dev/null | wc -l)
    N=$(printf "%03d" $((N + 1)))
    LOG="${REPO}/1test_results/${N}_hstu8_examples_qm2.log"
fi

echo "Log: ${LOG}"

PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user \
python -u - > "${LOG}" 2>&1 <<'PYEOF'
import torch, sys
REPO = '/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu'
sys.path.insert(0, f'{REPO}/fbgemm_gpu/experimental/hstu')
sys.path.insert(0, f'{REPO}/fbgemm_gpu/experimental/hstu/test')
import hstu  # noqa
from hstu_test import HSTU8Test

t = HSTU8Test()
inner = t.test_hstu_attn_fp8.hypothesis.inner_test

cases = [
    # --- seq=128 ---
    dict(batch_size=4, heads=1, seq_len_params=(128, 128), max_context_len=0,
         target_params=(0, (-1, 0), 1, False), attn_hidden_dims=(128, 128), alpha=1.0,
         rab_params=(False, False, None), dtype=torch.float8_e4m3fn,
         quant_mode_full_batch=(2, True)),   # causal baseline
    dict(batch_size=4, heads=1, seq_len_params=(128, 128), max_context_len=0,
         target_params=(0, (-1, 0), 1, False), attn_hidden_dims=(128, 128), alpha=1.0,
         rab_params=(True, False, None), dtype=torch.float8_e4m3fn,
         quant_mode_full_batch=(2, True)),   # causal + rab
    dict(batch_size=4, heads=1, seq_len_params=(128, 128), max_context_len=0,
         target_params=(0, (-1, 0), 1, False), attn_hidden_dims=(128, 128), alpha=1.0,
         rab_params=(True, True, None), dtype=torch.float8_e4m3fn,
         quant_mode_full_batch=(2, True)),   # causal + drab
    dict(batch_size=4, heads=1, seq_len_params=(128, 128), max_context_len=0,
         target_params=(0, (64, 16), 1, False), attn_hidden_dims=(128, 128), alpha=1.0,
         rab_params=(False, False, None), dtype=torch.float8_e4m3fn,
         quant_mode_full_batch=(2, True)),   # local mask
    dict(batch_size=4, heads=1, seq_len_params=(128, 128), max_context_len=128,
         target_params=(0, (-1, 0), 1, False), attn_hidden_dims=(128, 128), alpha=1.0,
         rab_params=(False, False, None), dtype=torch.float8_e4m3fn,
         quant_mode_full_batch=(2, True)),   # context + causal (total_k=256)
    dict(batch_size=4, heads=1, seq_len_params=(128, 128), max_context_len=0,
         target_params=(128, (-1, 0), 1, False), attn_hidden_dims=(128, 128), alpha=1.0,
         rab_params=(False, False, None), dtype=torch.float8_e4m3fn,
         quant_mode_full_batch=(2, True)),   # target + causal (total_k=256)
    dict(batch_size=4, heads=1, seq_len_params=(128, 128), max_context_len=0,
         target_params=(0, (-1, -1), 1, True), attn_hidden_dims=(128, 128), alpha=1.0,
         rab_params=(False, False, None), dtype=torch.float8_e4m3fn,
         quant_mode_full_batch=(2, True)),   # arbitrary masking
    # --- seq=256 ---
    dict(batch_size=4, heads=1, seq_len_params=(256, 256), max_context_len=0,
         target_params=(0, (-1, 0), 1, False), attn_hidden_dims=(128, 128), alpha=1.0,
         rab_params=(False, False, None), dtype=torch.float8_e4m3fn,
         quant_mode_full_batch=(2, True)),   # causal baseline
    dict(batch_size=4, heads=1, seq_len_params=(256, 256), max_context_len=0,
         target_params=(0, (-1, 0), 1, False), attn_hidden_dims=(128, 128), alpha=1.0,
         rab_params=(True, False, None), dtype=torch.float8_e4m3fn,
         quant_mode_full_batch=(2, True)),   # causal + rab
    dict(batch_size=4, heads=1, seq_len_params=(256, 256), max_context_len=0,
         target_params=(0, (-1, 0), 1, False), attn_hidden_dims=(128, 128), alpha=1.0,
         rab_params=(True, True, None), dtype=torch.float8_e4m3fn,
         quant_mode_full_batch=(2, True)),   # causal + drab
    dict(batch_size=4, heads=1, seq_len_params=(256, 256), max_context_len=0,
         target_params=(0, (128, 16), 1, False), attn_hidden_dims=(128, 128), alpha=1.0,
         rab_params=(False, False, None), dtype=torch.float8_e4m3fn,
         quant_mode_full_batch=(2, True)),   # local mask
    dict(batch_size=4, heads=1, seq_len_params=(256, 256), max_context_len=256,
         target_params=(0, (-1, 0), 1, False), attn_hidden_dims=(128, 128), alpha=1.0,
         rab_params=(False, False, None), dtype=torch.float8_e4m3fn,
         quant_mode_full_batch=(2, True)),   # context + causal (total_k=512)
    dict(batch_size=4, heads=1, seq_len_params=(256, 256), max_context_len=0,
         target_params=(256, (-1, 0), 1, False), attn_hidden_dims=(128, 128), alpha=1.0,
         rab_params=(False, False, None), dtype=torch.float8_e4m3fn,
         quant_mode_full_batch=(2, True)),   # target + causal (total_k=512)
    dict(batch_size=4, heads=1, seq_len_params=(256, 256), max_context_len=0,
         target_params=(0, (-1, -1), 1, True), attn_hidden_dims=(128, 128), alpha=1.0,
         rab_params=(False, False, None), dtype=torch.float8_e4m3fn,
         quant_mode_full_batch=(2, True)),   # arbitrary masking
]

labels = [
    'seq=128 causal',         'seq=128 causal+rab',   'seq=128 causal+drab',
    'seq=128 local',          'seq=128 context+causal','seq=128 target+causal',
    'seq=128 arbitrary',
    'seq=256 causal',         'seq=256 causal+rab',   'seq=256 causal+drab',
    'seq=256 local',          'seq=256 context+causal','seq=256 target+causal',
    'seq=256 arbitrary',
]

passed = 0
for i, (kw, label) in enumerate(zip(cases, labels)):
    try:
        inner(t, **kw)
        print(f'case {i+1:2d} PASS  [{label}]')
        passed += 1
    except Exception as e:
        print(f'case {i+1:2d} FAIL  [{label}] -- {e}')
    sys.stdout.flush()

print(f'\nResult: {passed}/{len(cases)} passed')
sys.exit(0 if passed == len(cases) else 1)
PYEOF

EXIT=$?
echo "exit=${EXIT}"
grep -E "^case|^Result" "${LOG}"
exit ${EXIT}
