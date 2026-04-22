#!/usr/bin/env python3
"""
ncu target script for SM120 HSTU attention.
Run via:
  ncu --kernel-name-base function --kernel-name hstu_fwd_kernel_sm120 \
      --launch-skip 5 --launch-count 3 \
      <metrics> python ncu_hstu_attn.py
"""
import sys
import torch

sys.path.insert(0, "/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu")

from hstu.cuda_hstu_attention import (
    quantize_for_block_scale_qk_along_d,
    quantize_for_block_scale_v_along_n,
    pack_descale_to_e8m0x4_int32,
)

BS, SEQ, H, D = 4, 2048, 16, 128
WARMUP = 5

total = BS * SEQ
cu = torch.zeros(BS + 1, dtype=torch.int32, device="cuda")
cu[1:] = torch.cumsum(torch.full((BS,), SEQ, dtype=torch.int32, device="cuda"), dim=0)

q = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")

q_in = q.to(torch.float8_e4m3fn).to(torch.bfloat16)
k_in = k.to(torch.float8_e4m3fn).to(torch.bfloat16)
v_in = v.to(torch.float8_e4m3fn).to(torch.bfloat16)
q_fp8, q_dsc, cu_q_blk   = quantize_for_block_scale_qk_along_d(q_in, cu, fp8_type=torch.float8_e4m3fn)
k_fp8, k_dsc, cu_kv_blk  = quantize_for_block_scale_qk_along_d(k_in, cu, fp8_type=torch.float8_e4m3fn)
v_fp8, v_dsc, cu_v_blk   = quantize_for_block_scale_v_along_n(v_in, cu, block_size=128, fp8_type=torch.float8_e4m3fn)
sf_q = pack_descale_to_e8m0x4_int32(q_dsc)
sf_k = pack_descale_to_e8m0x4_int32(k_dsc)
sf_v = pack_descale_to_e8m0x4_int32(v_dsc).repeat_interleave(128, dim=1)

def run_bf16():
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        q, k, v, cu, cu, None, None,
        SEQ, SEQ, SEQ, None, None, 1, -1, -1, 1.0,
        None, None, -1,
        None, None, None, None, None, None, None, None, None,
    )
    return out

def run_fp8():
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        q_fp8, k_fp8, v_fp8, cu, cu, None, None,
        SEQ, SEQ, SEQ, None, None, 1, -1, -1, 1.0,
        None, None, 2,
        q_dsc, k_dsc, v_dsc,
        sf_q, sf_k, sf_v,
        cu_q_blk, cu_kv_blk, cu_v_blk,
    )
    return out

# Warmup (ncu --launch-skip will skip these)
for _ in range(WARMUP):
    run_bf16()
    run_fp8()
torch.cuda.synchronize()

# Profile targets: 3 BF16 then 3 FP8
# ncu --launch-skip 10 --launch-count 3 profiles launches 11-13 (BF16)
# ncu --launch-skip 13 --launch-count 3 profiles launches 14-16 (FP8)
for _ in range(3):
    run_bf16()
torch.cuda.synchronize()
for _ in range(3):
    run_fp8()
torch.cuda.synchronize()

print(f"ncu target done: BS={BS} SEQ={SEQ} H={H} D={D}")
