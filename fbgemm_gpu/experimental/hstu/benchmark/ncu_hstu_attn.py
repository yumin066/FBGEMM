#!/usr/bin/env python3
"""NCU target script for SM120 HSTU attention."""
import argparse
import sys
import torch

sys.path.insert(0, "/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu")

from hstu.cuda_hstu_attention import (
    quantize_for_block_scale_qk_along_d,
    quantize_for_block_scale_v_along_n,
    quantize_paged_kv_cache_for_block_scale,
    pack_descale_to_e8m0x4_int32,
)

parser = argparse.ArgumentParser()
parser.add_argument("--target", choices=["bf16", "fp8", "paged", "all"], default="all")
parser.add_argument("--window", choices=["full", "causal"], default="causal")
parser.add_argument("--bs", type=int, default=4)
parser.add_argument("--seq", type=int, default=2048)
parser.add_argument("--heads", type=int, default=16)
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--iters", type=int, default=3)
args = parser.parse_args()

BS, SEQ, H, D = args.bs, args.seq, args.heads, args.dim
WARMUP = args.warmup
ITERS = args.iters
WINDOW = (-1, 0) if args.window == "causal" else (-1, -1)
FP8_BLOCK_N = 64 if D == 128 else 128

total = BS * SEQ
cu = torch.zeros(BS + 1, dtype=torch.int32, device="cuda")
cu[1:] = torch.cumsum(torch.full((BS,), SEQ, dtype=torch.int32, device="cuda"), dim=0)
num_targets0 = torch.zeros(BS, dtype=torch.int32, device="cuda")

q = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")

q_in = q.to(torch.float8_e4m3fn).to(torch.bfloat16)
k_in = k.to(torch.float8_e4m3fn).to(torch.bfloat16)
v_in = v.to(torch.float8_e4m3fn).to(torch.bfloat16)
q_fp8, q_dsc, cu_q_blk   = quantize_for_block_scale_qk_along_d(q_in, cu, fp8_type=torch.float8_e4m3fn)
k_fp8, k_dsc, cu_kv_blk  = quantize_for_block_scale_qk_along_d(k_in, cu, fp8_type=torch.float8_e4m3fn)
v_fp8, v_dsc, cu_v_blk   = quantize_for_block_scale_v_along_n(v_in, cu, block_size=FP8_BLOCK_N, fp8_type=torch.float8_e4m3fn)
sf_q = pack_descale_to_e8m0x4_int32(q_dsc)
sf_k = pack_descale_to_e8m0x4_int32(k_dsc)
sf_v = pack_descale_to_e8m0x4_int32(v_dsc).repeat_interleave(FP8_BLOCK_N, dim=1)

page_offsets = torch.zeros(BS + 1, dtype=torch.int32, device="cuda")
pages_per_batch = SEQ // FP8_BLOCK_N
page_offsets[1:] = torch.arange(1, BS + 1, dtype=torch.int32, device="cuda") * pages_per_batch
page_ids = torch.arange(BS * pages_per_batch, dtype=torch.int32, device="cuda")
last_page_lens = torch.full((BS,), FP8_BLOCK_N, dtype=torch.int32, device="cuda")
kv_cache = torch.randn(
    BS * pages_per_batch, 2, FP8_BLOCK_N, H, D,
    dtype=torch.bfloat16,
    device="cuda",
).to(torch.float8_e4m3fn).to(torch.bfloat16)
kv_cache_fp8, sf_k_cache, sf_v_cache = quantize_paged_kv_cache_for_block_scale(
    kv_cache, block_size=FP8_BLOCK_N, fp8_type=torch.float8_e4m3fn)
sf_k_paged = torch.cat([sf_k_cache, sf_k], dim=1).contiguous()
sf_v_paged = torch.cat([sf_v_cache, sf_v], dim=1).contiguous()

def run_bf16():
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        q, k, v, cu, cu, None, None,
        SEQ, SEQ, SEQ, None, None, 1, WINDOW[0], WINDOW[1], 1.0,
        None, None, -1,
        None, None, None, None, None, None, None, None, None,
    )
    return out

def run_fp8():
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        q_fp8, k_fp8, v_fp8, cu, cu, None, None,
        SEQ, SEQ, SEQ, None, None, 1, WINDOW[0], WINDOW[1], 1.0,
        None, None, 2,
        q_dsc, k_dsc, v_dsc,
        sf_q, sf_k, sf_v,
        cu_q_blk, cu_kv_blk, cu_v_blk,
    )
    return out

def run_paged():
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        q_fp8, k_fp8, v_fp8, cu, cu, None, None,
        SEQ, SEQ, SEQ, None, num_targets0, 1, -1, 0, 1.0,
        None, None, 2,
        q_dsc, k_dsc, v_dsc,
        sf_q, sf_k_paged, sf_v_paged,
        cu_q_blk, cu_q_blk, cu_v_blk,
        kv_cache_fp8, page_offsets, page_ids, last_page_lens,
    )
    return out

RUNNERS = {
    "bf16": run_bf16,
    "fp8": run_fp8,
    "paged": run_paged,
}
targets = ["bf16", "fp8", "paged"] if args.target == "all" else [args.target]

for target in targets:
    fn = RUNNERS[target]
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()

print(
    f"ncu target done: target={args.target} window={args.window} "
    f"BS={BS} SEQ={SEQ} H={H} D={D} warmup={WARMUP} iters={ITERS}"
)
