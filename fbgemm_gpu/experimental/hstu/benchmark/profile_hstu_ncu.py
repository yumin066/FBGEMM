#!/usr/bin/env python3
"""
NCU profile script for SM120 HSTU attention.
Usage:
  ncu --metrics <metrics> --kernel-name <name> \
      python profile_hstu_ncu.py [--mode row|col]

Supports row-major V (baseline) and column-major V (experimental) modes.
"""
import sys
import argparse
import torch

sys.path.insert(0, "/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu")

from hstu.cuda_hstu_attention import (
    quantize_for_block_scale_qk_along_d,
    quantize_for_block_scale_v_along_n,
    pack_descale_to_e8m0x4_int32,
)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="col", choices=["row", "col"],
                    help="V layout: row=row-major (baseline), col=column-major (experimental)")
args = parser.parse_args()

# Profile config: bs=4, seq=2048, h=16, d=128
BS, SEQ, H, D = 4, 2048, 16, 128
WARMUP = 3
ITERS  = 5

total = BS * SEQ
cu = torch.zeros(BS + 1, dtype=torch.int32, device="cuda")
cu[1:] = torch.cumsum(torch.full((BS,), SEQ, dtype=torch.int32, device="cuda"), dim=0)

# BF16 inputs
q = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")

# FP8 quantization
q_in = q.to(torch.float8_e4m3fn).to(torch.bfloat16)
k_in = k.to(torch.float8_e4m3fn).to(torch.bfloat16)
v_in = v.to(torch.float8_e4m3fn).to(torch.bfloat16)
q_fp8, q_dsc, cu_q_blk = quantize_for_block_scale_qk_along_d(q_in, cu, fp8_type=torch.float8_e4m3fn)
k_fp8, k_dsc, cu_kv_blk = quantize_for_block_scale_qk_along_d(k_in, cu, fp8_type=torch.float8_e4m3fn)
v_fp8_row, v_dsc, cu_v_blk = quantize_for_block_scale_v_along_n(v_in, cu, block_size=128, fp8_type=torch.float8_e4m3fn)
sf_q = pack_descale_to_e8m0x4_int32(q_dsc)
sf_k = pack_descale_to_e8m0x4_int32(k_dsc)
sf_v = pack_descale_to_e8m0x4_int32(v_dsc).repeat_interleave(128, dim=1)

# Select V layout
if args.mode == "col":
    # Column-major V: shape [total_k, h_k, d], strides [d, d*total_k, 1]
    v_fp8 = v_fp8_row.permute(1, 0, 2).contiguous().permute(1, 0, 2)
    print(f"[mode] column-major V: strides={tuple(v_fp8.stride())}", flush=True)
else:
    v_fp8 = v_fp8_row
    print(f"[mode] row-major V: strides={tuple(v_fp8.stride())}", flush=True)

print(f"Config: BS={BS} SEQ={SEQ} H={H} D={D}", flush=True)


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
        None,
        None,
        2,
        q_dsc, k_dsc, v_dsc,
        sf_q, sf_k, sf_v,
        cu_q_blk, cu_kv_blk, cu_v_blk,
    )
    return out


# Warmup (outside NCU capture range)
for _ in range(WARMUP):
    run_fp8()
torch.cuda.synchronize()

# Profile region
for _ in range(ITERS):
    run_fp8()
torch.cuda.synchronize()

print("Done.", flush=True)
