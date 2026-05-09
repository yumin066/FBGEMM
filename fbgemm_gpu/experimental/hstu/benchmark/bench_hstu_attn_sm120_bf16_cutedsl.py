#!/usr/bin/env python3
"""Compare SM120 BF16 C++ HSTU with the CuTe DSL BF16 prototype.

Dense cases use the C++ BF16 torch.ops path directly.  Paged cases use the
same wrapper-side dense materialization semantics as the DSL prototype, then
call the existing C++ BF16 kernel.
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
    make_paged_cache_from_varlen_kv,
    run_kernel_bf16,
    time_kernel,
)
from hstu.cuda_hstu_attention import hstu_attn_varlen_func  # noqa: E402
from hstu_blackwell_sm120_cutedsl import hstu_varlen_fwd_120_bf16_cutedsl  # noqa: E402


def run_kernel_bf16_dsl(
    inputs: dict,
    paged_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    *,
    scale_output: bool = True,
):
    if paged_inputs is None:
        kv_cache = page_offsets = page_ids = last_page_lens = None
    else:
        kv_cache, page_offsets, page_ids, last_page_lens = paged_inputs
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
        kv_cache=kv_cache,
        page_offsets=page_offsets,
        page_ids=page_ids,
        last_page_lens=last_page_lens,
        scale_output=scale_output,
    )
    return out


def run_kernel_bf16_cpp(
    inputs: dict,
    paged_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
):
    if paged_inputs is None:
        return run_kernel_bf16(inputs)
    kv_cache, page_offsets, page_ids, last_page_lens = paged_inputs
    return hstu_attn_varlen_func(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        cu_seqlens_q=inputs["cu_q"],
        cu_seqlens_k=inputs["cu_k"],
        seqused_q=inputs["seqused_q"],
        seqused_k=inputs["seqused_k"],
        max_seqlen_q=inputs["max_seqlen_q"],
        max_seqlen_k=inputs["max_seqlen_k"],
        scaling_seqlen=-1,
        num_contexts=inputs["num_contexts"],
        num_targets=inputs["num_targets"],
        target_group_size=inputs["target_group_size"],
        window_size=inputs["window_size"],
        alpha=1.0,
        rab=inputs["rab"],
        has_drab=inputs["case"].has_drab,
        func=inputs["func"],
        kv_cache=kv_cache,
        page_offsets=page_offsets,
        page_ids=page_ids,
        last_page_lens=last_page_lens,
        quant_mode=-1,
    )


def bench_case(
    bs: int,
    seq: int,
    nheads: int,
    headdim: int,
    mask: str,
    bias: str,
    layout: str,
    page_size: int,
    warmup: int,
    iters: int,
):
    case = BenchCase(mask, bias)
    inputs = make_case_inputs(bs, seq, nheads, headdim, case)
    paged_inputs = (
        make_paged_cache_from_varlen_kv(inputs, page_size)
        if layout == "paged"
        else None
    )
    pairs = inputs["pairs"]
    flops = 4.0 * pairs * nheads * headdim

    ref = run_kernel_bf16_cpp(inputs, paged_inputs)
    dsl = run_kernel_bf16_dsl(inputs, paged_inputs)
    torch.cuda.synchronize()
    diff = (dsl.float() - ref.float()).abs()
    cos = F.cosine_similarity(
        dsl.float().flatten().unsqueeze(0),
        ref.float().flatten().unsqueeze(0),
    ).item()

    cur_ms, _, _ = time_kernel(
        lambda: run_kernel_bf16_cpp(inputs, paged_inputs), warmup=warmup, iters=iters
    )
    dsl_ms, _, _ = time_kernel(
        lambda: run_kernel_bf16_dsl(inputs, paged_inputs), warmup=warmup, iters=iters
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
    parser.add_argument("--layouts", nargs="+", default=["dense"], choices=["dense", "paged"])
    parser.add_argument("--page-size", type=int, default=64)
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
    print(f"Layouts={args.layouts} page_size={args.page_size}")
    print("DSL supported: D32/D64/D128/D256 all masks x none/RAB/DRAB")
    print("C++ BF16 supports dense directly and paged through wrapper-side dense materialization.")
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
                            for layout in args.layouts:
                                ok, reason = compare_supported(d, mask, bias)
                                cfg = f"bs={bs} seq={seq} h={h} d={d} {layout} {mask}+{bias}"
                                if not ok:
                                    print(f"{cfg:<42} N/A: {reason}")
                                    continue
                                try:
                                    res = bench_case(
                                        bs, seq, h, d, mask, bias, layout,
                                        args.page_size, args.warmup, args.iters
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
