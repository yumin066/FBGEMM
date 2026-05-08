#!/usr/bin/env python3
"""NCU target script for SM120 HSTU attention."""
import argparse
import sys
import torch

sys.path.insert(0, "/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu")
sys.path.insert(0, "/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu/benchmark")

from bench_hstu_attn_sm120 import (
    BenchCase,
    make_case_inputs,
    quantize_fp8_bs,
    quantize_paged_kv_inputs,
    run_kernel_bf16,
    run_kernel_fp8bs,
    run_kernel_fp8bs_paged,
)

parser = argparse.ArgumentParser()
parser.add_argument("--target", choices=["bf16", "fp8", "paged", "all"], default="all")
parser.add_argument("--window", choices=["full", "causal"], default="causal")
parser.add_argument("--mask-config", choices=["full", "causal", "local", "context", "target", "arbitrary"], default=None)
parser.add_argument("--bias-config", choices=["none", "rab", "drab"], default="none")
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

mask_config = args.mask_config if args.mask_config is not None else args.window
case = BenchCase(mask_config, args.bias_config)
inputs = make_case_inputs(BS, SEQ, H, D, case)
fp8_inputs = quantize_fp8_bs(
    inputs["q"],
    inputs["k"],
    inputs["v"],
    inputs["cu_q"],
    inputs["cu_k"],
    D,
    case.has_rab,
)
paged_inputs = quantize_paged_kv_inputs(inputs, D)

def run_bf16():
    return run_kernel_bf16(inputs)

def run_fp8():
    return run_kernel_fp8bs(inputs, fp8_inputs)

def run_paged():
    return run_kernel_fp8bs_paged(inputs, paged_inputs)

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
    f"ncu target done: target={args.target} mask={case.label} "
    f"BS={BS} SEQ={SEQ} H={H} D={D} warmup={WARMUP} iters={ITERS}"
)
