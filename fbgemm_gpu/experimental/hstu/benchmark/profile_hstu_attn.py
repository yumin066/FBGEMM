#!/usr/bin/env python3
"""
nsys profile script for SM120 HSTU attention.
Run via:
  nsys profile --output <trace> --trace cuda,nvtx --force-overwrite true \
    python profile_hstu_attn.py
"""
import sys
import torch

sys.path.insert(0, "/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu")

from hstu.cuda_hstu_attention import (
    quantize_for_block_scale_qk_along_d,
    quantize_for_block_scale_v_along_n,
    pack_descale_to_e8m0x4_int32,
)

# Profile config: bs=4, seq=2048, h=16, d=128 — large enough to see kernel structure
BS, SEQ, H, D = 4, 2048, 16, 128
WARMUP = 5
ITERS  = 20

total = BS * SEQ
cu = torch.zeros(BS + 1, dtype=torch.int32, device="cuda")
cu[1:] = torch.cumsum(torch.full((BS,), SEQ, dtype=torch.int32, device="cuda"), dim=0)

# BF16 inputs
q = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")

# FP8 inputs (quantize once, outside profile region)
q_in = q.to(torch.float8_e4m3fn).to(torch.bfloat16)
k_in = k.to(torch.float8_e4m3fn).to(torch.bfloat16)
v_in = v.to(torch.float8_e4m3fn).to(torch.bfloat16)
q_fp8, q_dsc, cu_q_blk = quantize_for_block_scale_qk_along_d(q_in, cu, fp8_type=torch.float8_e4m3fn)
k_fp8, k_dsc, cu_kv_blk = quantize_for_block_scale_qk_along_d(k_in, cu, fp8_type=torch.float8_e4m3fn)
v_fp8, v_dsc, cu_v_blk = quantize_for_block_scale_v_along_n(v_in, cu, block_size=128, fp8_type=torch.float8_e4m3fn)
sf_q = pack_descale_to_e8m0x4_int32(q_dsc)
sf_k = pack_descale_to_e8m0x4_int32(k_dsc)
sf_v = pack_descale_to_e8m0x4_int32(v_dsc).repeat_interleave(128, dim=1)

def run_bf16():
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        q, k, v,
        cu, cu,          # cu_seqlens_q, cu_seqlens_k
        None, None,      # seqused_q, seqused_k
        SEQ, SEQ,        # max_seqlen_q, max_seqlen_k
        SEQ,             # scaling_seqlen
        None, None, 1,   # num_contexts, num_targets, target_group_size
        -1, -1,          # window_size_left, window_size_right
        1.0,             # alpha
        None,            # rab
        None,            # func_boundaries
        -1,              # quant_mode (BF16)
        None, None, None,        # descale_q/k/v
        None, None, None,        # sf_q/k/v
        None, None, None,        # cu_q_blk/kv_blk/v_blk
    )
    return out

def run_fp8():
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        q_fp8, k_fp8, v_fp8,
        cu, cu,
        None, None,
        SEQ, SEQ,
        SEQ,
        None, None, 1,
        -1, -1,
        1.0,
        None,            # rab
        None,            # func_boundaries
        2,               # quant_mode (FP8 block-scale)
        q_dsc, k_dsc, v_dsc,
        sf_q, sf_k, sf_v,
        cu_q_blk, cu_kv_blk, cu_v_blk,
    )
    return out

# Warmup
for _ in range(WARMUP):
    run_bf16()
    run_fp8()
torch.cuda.synchronize()

# Profile region: BF16
torch.cuda.nvtx.range_push("BF16_attn")
for _ in range(ITERS):
    run_bf16()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

# Profile region: FP8
torch.cuda.nvtx.range_push("FP8_attn")
for _ in range(ITERS):
    run_fp8()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

print(f"Profile done: BS={BS} SEQ={SEQ} H={H} D={D}, warmup={WARMUP} iters={ITERS}")
