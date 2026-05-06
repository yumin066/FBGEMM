#!/usr/bin/env bash
# Run 14 explicit @example cases for HSTU8Test plus paged KV mirrors.
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
from hstu.cuda_hstu_attention import hstu_attn_varlen_func
from hstu_test import HSTU8Test, generate_paged_kv_input, _hstu_paged_kv_attention

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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

def metric_report(a, b):
    diff = (a.float() - b.float()).abs()
    cos = float(torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)
    ).item())
    return cos, float(diff.max().item()), float(diff.mean().item())


def run_paged_kv_case(batch_size, heads, new_history_len, prev_history_len, target_len):
    D = 128
    page_size = 64
    alpha = 1.0
    torch.manual_seed(SEED + batch_size * 100 + heads * 10 + new_history_len + prev_history_len + target_len)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + batch_size * 100 + heads * 10 + new_history_len + prev_history_len + target_len)

    (
        _,
        cu_seqlens_q,
        cu_seqlens_k,
        num_targets,
        page_offsets,
        page_ids,
        last_page_lens,
        q,
        k,
        v,
        kv_cache,
        mask,
    ) = generate_paged_kv_input(
        batch_size=batch_size,
        heads=heads,
        max_seq_len_q=new_history_len,
        max_seq_len_k=prev_history_len,
        max_target_len=target_len,
        attn_dim=D,
        hidden_dim=D,
        page_size=page_size,
        dtype=torch.float16,
        full_batch=True,
    )

    max_seqlen_q = new_history_len + target_len
    max_seqlen_k = new_history_len + prev_history_len + target_len
    ref = _hstu_paged_kv_attention(
        num_heads=heads,
        attention_dim=D,
        linear_dim=D,
        seqlen_q=max_seqlen_q,
        seqlen_k=max_seqlen_k,
        q=q,
        k=k,
        v=v,
        q_offsets=cu_seqlens_q,
        k_offsets=cu_seqlens_k,
        num_targets=num_targets,
        invalid_attn_mask=mask,
        alpha=alpha,
        upcast=True,
        kv_cache=kv_cache,
        page_offsets=page_offsets,
        page_ids=page_ids,
        last_page_lens=last_page_lens,
    )
    out = hstu_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        scaling_seqlen=-1,
        num_contexts=None,
        num_targets=num_targets,
        target_group_size=1,
        window_size=(-1, 0),
        alpha=alpha,
        rab=None,
        has_drab=False,
        kv_cache=kv_cache,
        page_offsets=page_offsets,
        page_ids=page_ids,
        last_page_lens=last_page_lens,
        func=None,
        quant_mode=2,
    )
    torch.cuda.synchronize()
    cos, max_err, mean_err = metric_report(out, ref)
    if cos < 0.995:
        raise AssertionError(f'paged KV fp8_gt_cos={cos:.6f} < 0.995')
    return cos, max_err, mean_err, last_page_lens.detach().cpu().tolist()


def paged_case_from_example(kw):
    seq_len = kw['seq_len_params'][1]
    target_len = kw['target_params'][0]
    return dict(
        batch_size=kw['batch_size'],
        heads=kw['heads'],
        new_history_len=seq_len,
        prev_history_len=kw['max_context_len'],
        target_len=target_len,
    )


# SM120 FP8 paged KV currently supports no-RAB causal/target semantics only.
# These mirrors cover every example's batch/head/seq/context/target dimensions;
# unsupported RAB/local/arbitrary features are not enabled on the paged path.
PAGED_CASES = [
    (f'paged kv mirror {label}', paged_case_from_example(kw))
    for kw, label in zip(cases, labels)
] + [
    ('paged kv edge partial-last-page', dict(batch_size=1, heads=2, new_history_len=64, prev_history_len=32, target_len=64)),
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

case_id = len(cases)
for label, kw in PAGED_CASES:
    case_id += 1
    try:
        cos, max_err, mean_err, last_page = run_paged_kv_case(**kw)
        print(
            f'case {case_id:2d} PASS  [{label}] '
            f'cos={cos:.6f} max={max_err:.6f} mean={mean_err:.6f} last_page={last_page}'
        )
        passed += 1
    except Exception as e:
        print(f'case {case_id:2d} FAIL  [{label}] -- {e}')
    sys.stdout.flush()

total = len(cases) + len(PAGED_CASES)
print(f'\nResult: {passed}/{total} passed')
sys.exit(0 if passed == total else 1)
PYEOF

EXIT=$?
echo "exit=${EXIT}"
grep -E "^case|^Result" "${LOG}"
exit ${EXIT}
