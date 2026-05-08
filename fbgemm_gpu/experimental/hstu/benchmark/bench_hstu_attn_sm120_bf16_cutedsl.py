#!/usr/bin/env python3
"""Compare current SM120 BF16 HSTU with the CuTe DSL BF16 prototype.

The default benchmark compares dense full-batch non-paged inputs against the
current C++ BF16 path.  The prototype wrapper itself also supports compact
varlen inputs, wrapper-side paged KV materialization, and D256 RAB/DRAB.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
HSTU_ROOT = REPO_ROOT / "fbgemm_gpu" / "experimental" / "hstu"
sys.path.insert(0, str(HSTU_ROOT / "src"))
sys.path.insert(0, str(HSTU_ROOT))
sys.path.insert(0, str(HSTU_ROOT / "test"))
sys.path.insert(0, str(HSTU_ROOT / "benchmark"))

import hstu  # noqa: F401,E402
import bench_hstu_attn_sm120 as base_bench  # noqa: E402
from bench_hstu_attn_sm120 import (  # noqa: E402
    BenchCase,
    make_case_inputs,
    run_kernel_bf16,
    time_kernel,
)
from hstu_blackwell_sm120_cutedsl import hstu_varlen_fwd_120_bf16_cutedsl  # noqa: E402


def run_kernel_bf16_dsl(inputs: dict, *, scale_output: bool = True):
    out, _ = hstu_varlen_fwd_120_bf16_cutedsl(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["cu_q"],
        inputs["cu_k"],
        inputs["max_seqlen_q"],
        inputs["max_seqlen_k"],
        inputs["num_contexts"],
        inputs["num_targets"],
        inputs["target_group_size"],
        inputs["window_size"][0],
        inputs["window_size"][1],
        1.0,
        inputs["rab"],
        inputs["func"],
        scale_output=scale_output,
    )
    return out


def bench_case(
    bs: int,
    seq: int,
    nheads: int,
    headdim: int,
    mask: str,
    bias: str,
    warmup: int,
    iters: int,
):
    case = BenchCase(mask, bias)
    inputs = make_case_inputs(bs, seq, nheads, headdim, case)
    pairs = inputs["pairs"]
    flops = 4.0 * pairs * nheads * headdim

    ref = run_kernel_bf16(inputs)
    dsl = run_kernel_bf16_dsl(inputs)
    torch.cuda.synchronize()
    diff = (dsl.float() - ref.float()).abs()
    cos = F.cosine_similarity(
        dsl.float().flatten().unsqueeze(0),
        ref.float().flatten().unsqueeze(0),
    ).item()

    cur_ms, _, _ = time_kernel(
        lambda: run_kernel_bf16(inputs), warmup=warmup, iters=iters
    )
    dsl_ms, _, _ = time_kernel(
        lambda: run_kernel_bf16_dsl(inputs), warmup=warmup, iters=iters
    )
    cur_tflops = flops / (cur_ms * 1e-3) / 1e12
    dsl_tflops = flops / (dsl_ms * 1e-3) / 1e12
    ratio = dsl_tflops / cur_tflops if cur_tflops else 0.0

    return {
        "cur_ms": cur_ms,
        "dsl_ms": dsl_ms,
        "cur_tflops": cur_tflops,
        "dsl_tflops": dsl_tflops,
        "ratio": ratio,
        "max_err": diff.max().item(),
        "mean_err": diff.mean().item(),
        "cos": cos,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="SM120 BF16 CuTe DSL prototype benchmark"
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--seqlens", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--nheads", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--headdims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--masks", nargs="+", default=["full", "causal"])
    parser.add_argument("--biases", nargs="+", default=["none"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def dsl_supported(headdim: int, mask: str, bias: str) -> tuple[bool, str]:
    if headdim not in (32, 64, 128, 256):
        return False, "unsupported headDim"
    if mask not in ("full", "causal", "local", "context", "target", "arbitrary"):
        return False, "unsupported mask"
    if bias not in ("none", "rab", "drab"):
        return False, "unsupported bias"
    return True, ""


def compare_supported(headdim: int, mask: str, bias: str) -> tuple[bool, str]:
    ok, reason = dsl_supported(headdim, mask, bias)
    if not ok:
        return ok, reason
    base_reason = base_bench.bf16_unsupported_reason(headdim, BenchCase(mask, bias))
    if base_reason:
        return False, f"current C++ BF16 compare unavailable: {base_reason}"
    return True, ""


def main():
    args = parse_args()

    base_bench.WARMUP = args.warmup
    base_bench.ITERS = args.iters

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Capability: SM{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")
    print(f"Warmup={args.warmup} Iters={args.iters}")
    print("Scope: BF16 non-paged dense equal-length batches")
    print("DSL supported: D32/D64/D128/D256 all masks x none/RAB/DRAB")
    print("Wrapper also supports compact varlen and wrapper-side paged KV materialization.")
    print("This benchmark reports current C++ BF16 vs DSL only when the current C++ BF16 column is available.")
    print("Note: DSL CUDA-event timing includes Python/CuTe DSL wrapper launch overhead; use nsys/ncu for GPU kernel-only.")
    print()
    header = (
        f"{'Config':<42} {'Current(ms)':>11} {'Current TF':>11} "
        f"{'DSLevt(ms)':>10} "
        f"{'DSL TF':>9} {'DSL/Cur':>8} "
        f"{'cos':>10} {'max_err':>10} {'mean_err':>10}"
    )
    print(header)
    print("-" * len(header))

    ratios = []
    for bs in args.batch_sizes:
        for seq in args.seqlens:
            for h in args.nheads:
                for d in args.headdims:
                    for mask in args.masks:
                        for bias in args.biases:
                            ok, reason = compare_supported(d, mask, bias)
                            cfg = f"bs={bs} seq={seq} h={h} d={d} {mask}+{bias}"
                            if not ok:
                                print(f"{cfg:<42} N/A: {reason}")
                                continue
                            try:
                                res = bench_case(
                                    bs, seq, h, d, mask, bias, args.warmup, args.iters
                                )
                                ratios.append(res["ratio"])
                                print(
                                    f"{cfg:<42} {res['cur_ms']:11.4f} {res['cur_tflops']:11.1f} "
                                    f"{res['dsl_ms']:10.4f} {res['dsl_tflops']:9.1f} "
                                    f"{res['ratio']:8.3f} {res['cos']:10.7f} "
                                    f"{res['max_err']:10.6f} {res['mean_err']:10.6f}"
                                )
                            except Exception as exc:
                                print(f"{cfg:<42} ERROR: {exc}")

    if ratios:
        ratios_t = torch.tensor(ratios, dtype=torch.float64)
        print()
        print(
            f"Geomean DSL/Current TFLOPS ratio: "
            f"{torch.exp(torch.log(ratios_t).mean()).item():.3f}"
        )


if __name__ == "__main__":
    main()
