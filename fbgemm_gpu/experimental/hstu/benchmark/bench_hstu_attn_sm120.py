#!/usr/bin/env python3
"""
SM120 (Blackwell RTX Pro) HSTU Attention Benchmark.

Measures kernel-only and end-to-end throughput for BF16 and FP8 block-scale
(quant_mode=2) attention on SM120.

Kernel-only timing bypasses the Python-level quantization overhead to measure
the raw CUDA kernel performance (TFLOPS).

Usage:
    python benchmark/bench_hstu_attn_sm120.py
    python benchmark/bench_hstu_attn_sm120.py --mode kernel
    python benchmark/bench_hstu_attn_sm120.py --mode e2e
    python benchmark/bench_hstu_attn_sm120.py --mode all
    python benchmark/bench_hstu_attn_sm120.py --seqlens 512 1024 2048 4096
"""

import argparse
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu")

try:
    from hstu.cuda_hstu_attention import (
        quantize_for_block_scale_qk_along_d,
        quantize_for_block_scale_v_along_n,
        pack_descale_to_e8m0x4_int32,
    )
    import hstu  # noqa: F401
except ImportError as e:
    print(f"ERROR: Failed to import hstu: {e}", file=sys.stderr)
    sys.exit(1)


WARMUP = 10
ITERS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_sm120() -> None:
    assert torch.cuda.is_available(), "CUDA not available"
    major, minor = torch.cuda.get_device_capability()
    if major < 12:
        print(
            f"WARNING: SM120 kernels require SM12.x (RTX Pro 6000 Blackwell). "
            f"Detected SM{major}{minor}. FP8 block-scale benchmarks may fail.",
            file=sys.stderr,
        )


def make_cu_seqlens(batch_size: int, seqlen: int, device="cuda") -> torch.Tensor:
    lengths = torch.full((batch_size,), seqlen, dtype=torch.int32, device=device)
    cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(lengths, dim=0)
    return cu


def make_bf16_inputs(batch_size, seqlen, nheads, headdim):
    total = batch_size * seqlen
    q = torch.randn(total, nheads, headdim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(total, nheads, headdim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(total, nheads, headdim, dtype=torch.bfloat16, device="cuda")
    cu_seqlens = make_cu_seqlens(batch_size, seqlen)
    return q, k, v, cu_seqlens


def quantize_fp8_bs(q_bf16, k_bf16, v_bf16, batch_size, seqlen, nheads, headdim):
    """Quantize BF16 Q/K/V to FP8 block-scale (quant_mode=2) format."""
    cu_seqlens = make_cu_seqlens(batch_size, seqlen)

    # Round to FP8 range first (same as sweep_accuracy.py)
    q_in = q_bf16.to(torch.float8_e4m3fn).to(torch.bfloat16)
    k_in = k_bf16.to(torch.float8_e4m3fn).to(torch.bfloat16)
    v_in = v_bf16.to(torch.float8_e4m3fn).to(torch.bfloat16)

    # Q/K: block-scale along D (headdim axis)
    q_fp8, q_descale, cu_q_blk = quantize_for_block_scale_qk_along_d(
        q_in, cu_seqlens, fp8_type=torch.float8_e4m3fn
    )
    k_fp8, k_descale, cu_kv_blk = quantize_for_block_scale_qk_along_d(
        k_in, cu_seqlens, fp8_type=torch.float8_e4m3fn
    )

    # V: block-scale along N (sequence axis)
    v_fp8, v_descale, cu_v_blk = quantize_for_block_scale_v_along_n(
        v_in, cu_seqlens, block_size=128, fp8_type=torch.float8_e4m3fn
    )
    # Pack descale factors to e8m0×4 int32 format for kernel.
    # sf_v is expanded from [H, total_blocks] → [H, total_tokens] via repeat_interleave
    # so TMA SFV can load kBlockN-element tiles indexed by nb_abs (same as SFB).
    sf_q = pack_descale_to_e8m0x4_int32(q_descale)
    sf_k = pack_descale_to_e8m0x4_int32(k_descale)
    sf_v = pack_descale_to_e8m0x4_int32(v_descale).repeat_interleave(128, dim=1)

    return (q_fp8, k_fp8, v_fp8,
            sf_q, sf_k, sf_v,
            q_descale, k_descale, v_descale,
            cu_q_blk, cu_kv_blk, cu_v_blk,
            cu_seqlens)


def run_kernel_bf16(q, k, v, cu_seqlens, seqlen, batch_size, window_size=(-1, -1)):
    """Run BF16 kernel directly via torch.ops."""
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        q, k, v,
        cu_seqlens, cu_seqlens,   # cu_seqlens_q, cu_seqlens_k
        None, None,                # seqused_q, seqused_k
        seqlen, seqlen,            # max_seqlen_q, max_seqlen_k
        seqlen,                    # scaling_seqlen (must be int)
        None,                      # num_contexts
        None,                      # num_targets
        1,                         # target_group_size
        window_size[0], window_size[1],
        1.0,                       # alpha
        None,                      # rab
        None,                      # func_boundaries
        -1,                        # quant_mode (BF16)
        None, None, None,          # descale_q/k/v
        None, None, None,          # sf_q/k/v
        None, None, None,          # cu_q_blk/kv_blk/v_blk
    )
    return out


def run_kernel_fp8bs(q_fp8, k_fp8, v_fp8, sf_q, sf_k, sf_v,
                     q_descale, k_descale, v_descale,
                     cu_seqlens, cu_q_blk, cu_kv_blk, cu_v_blk,
                     seqlen, window_size=(-1, -1)):
    """Run FP8 block-scale kernel directly via torch.ops."""
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        q_fp8, k_fp8, v_fp8,
        cu_seqlens, cu_seqlens,
        None, None,
        seqlen, seqlen,
        seqlen,                     # scaling_seqlen (must be int)
        None, None, 1,
        window_size[0], window_size[1],
        1.0,
        None, None,
        2,                          # quant_mode=2 (FP8 block-scale)
        q_descale, k_descale, v_descale,  # raw descales (for reference, unused by mode=2)
        sf_q, sf_k, sf_v,           # block-scale SF tensors (packed e8m0×4 int32)
        cu_q_blk, cu_kv_blk, cu_v_blk,
    )
    return out


def time_kernel(fn, warmup=WARMUP, iters=ITERS):
    """Time a CUDA kernel function; returns (avg_ms, min_ms, max_ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_evts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_evts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_evts[i].record()
        fn()
        end_evts[i].record()
    torch.cuda.synchronize()

    lats = [start_evts[i].elapsed_time(end_evts[i]) for i in range(iters)]
    return sum(lats) / len(lats), min(lats), max(lats)


def attention_flops(
    batch_size: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    window_size: Tuple[int, int],
) -> float:
    """FWD attention FLOPs for GEMM1 + GEMM2 over valid (query, key) pairs."""
    left, right = window_size
    if left < 0 and right < 0:
        pairs_per_sequence = seqlen * seqlen
    else:
        pairs_per_sequence = 0
        for q_idx in range(seqlen):
            k_begin = 0 if left < 0 else max(0, q_idx - left)
            k_end = seqlen - 1 if right < 0 else min(seqlen - 1, q_idx + right)
            if k_end >= k_begin:
                pairs_per_sequence += k_end - k_begin + 1

    return 4.0 * batch_size * pairs_per_sequence * nheads * headdim


# ---------------------------------------------------------------------------
# Kernel-only benchmark: measures only the CUDA attention kernel time
# ---------------------------------------------------------------------------

def bench_kernel_only(
    batch_size: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    window_size: Tuple[int, int],
) -> dict:
    """
    Returns dict with keys: bf16_ms, fp8_ms, bf16_tflops, fp8_tflops, speedup_pct.
    Total FLOPs = 4 * valid_attention_pairs * H * D (FWD attention: GEMM1 + GEMM2).
    """
    result = {}

    try:
        q_bf16, k_bf16, v_bf16, cu_seqlens = make_bf16_inputs(
            batch_size, seqlen, nheads, headdim
        )

        # BF16 kernel
        bf16_avg, _, _ = time_kernel(
            lambda: run_kernel_bf16(q_bf16, k_bf16, v_bf16, cu_seqlens, seqlen,
                                    batch_size, window_size)
        )
        result["bf16_ms"] = bf16_avg

    except Exception as e:
        result["bf16_error"] = str(e)

    try:
        # Prepare FP8 inputs (quantization done once, not timed)
        q_bf16_2, k_bf16_2, v_bf16_2, _ = make_bf16_inputs(
            batch_size, seqlen, nheads, headdim
        )
        (q_fp8, k_fp8, v_fp8,
         sf_q, sf_k, sf_v,
         q_dsc, k_dsc, v_dsc,
         cu_q_blk, cu_kv_blk, cu_v_blk,
         cu_seqlens2) = quantize_fp8_bs(
            q_bf16_2, k_bf16_2, v_bf16_2, batch_size, seqlen, nheads, headdim
        )

        fp8_avg, _, _ = time_kernel(
            lambda: run_kernel_fp8bs(
                q_fp8, k_fp8, v_fp8, sf_q, sf_k, sf_v,
                q_dsc, k_dsc, v_dsc,
                cu_seqlens2, cu_q_blk, cu_kv_blk, cu_v_blk,
                seqlen, window_size
            )
        )
        result["fp8_ms"] = fp8_avg

    except Exception as e:
        result["fp8_error"] = str(e)

    # Compute TFLOPS
    total_flops = attention_flops(batch_size, seqlen, nheads, headdim, window_size)
    if "bf16_ms" in result:
        result["bf16_tflops"] = total_flops / (result["bf16_ms"] * 1e-3) / 1e12
    if "fp8_ms" in result:
        result["fp8_tflops"] = total_flops / (result["fp8_ms"] * 1e-3) / 1e12

    # Speedup
    if "bf16_tflops" in result and "fp8_tflops" in result:
        result["speedup_pct"] = (
            result["fp8_tflops"] / result["bf16_tflops"] - 1.0
        ) * 100.0

    return result


# ---------------------------------------------------------------------------
# End-to-end benchmark: includes quantization overhead
# ---------------------------------------------------------------------------

def run_e2e_bf16(q_bf16, k_bf16, v_bf16, cu_seqlens, seqlen,
                 batch_size, window_size):
    return run_kernel_bf16(q_bf16, k_bf16, v_bf16, cu_seqlens, seqlen,
                           batch_size, window_size)


def run_e2e_fp8bs(q_bf16, k_bf16, v_bf16, seqlen, batch_size, nheads, headdim,
                  window_size):
    """Full end-to-end FP8: quantization + kernel."""
    (q_fp8, k_fp8, v_fp8,
     sf_q, sf_k, sf_v,
     q_dsc, k_dsc, v_dsc,
     cu_q_blk, cu_kv_blk, cu_v_blk,
     cu_seqlens) = quantize_fp8_bs(
        q_bf16, k_bf16, v_bf16, batch_size, seqlen, nheads, headdim
    )
    return run_kernel_fp8bs(
        q_fp8, k_fp8, v_fp8, sf_q, sf_k, sf_v,
        q_dsc, k_dsc, v_dsc,
        cu_seqlens, cu_q_blk, cu_kv_blk, cu_v_blk,
        seqlen, window_size
    )


def bench_e2e(
    batch_size: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    window_size: Tuple[int, int],
) -> dict:
    """Returns dict with end-to-end BF16 and FP8 latencies and speedup."""
    result = {}

    try:
        q_bf16, k_bf16, v_bf16, cu_seqlens = make_bf16_inputs(
            batch_size, seqlen, nheads, headdim
        )
        bf16_avg, _, _ = time_kernel(
            lambda: run_e2e_bf16(q_bf16, k_bf16, v_bf16, cu_seqlens, seqlen,
                                 batch_size, window_size)
        )
        result["bf16_ms"] = bf16_avg

        total_flops_e2e = attention_flops(
            batch_size, seqlen, nheads, headdim, window_size
        )
        result["bf16_tflops"] = total_flops_e2e / (bf16_avg * 1e-3) / 1e12

    except Exception as e:
        result["bf16_error"] = str(e)

    try:
        q_bf16_2, k_bf16_2, v_bf16_2, _ = make_bf16_inputs(
            batch_size, seqlen, nheads, headdim
        )
        fp8_avg, _, _ = time_kernel(
            lambda: run_e2e_fp8bs(q_bf16_2, k_bf16_2, v_bf16_2, seqlen,
                                  batch_size, nheads, headdim, window_size)
        )
        result["fp8_ms"] = fp8_avg

        total_flops_e2e = attention_flops(
            batch_size, seqlen, nheads, headdim, window_size
        )
        result["fp8_tflops"] = total_flops_e2e / (fp8_avg * 1e-3) / 1e12

    except Exception as e:
        result["fp8_error"] = str(e)

    if "bf16_ms" in result and "fp8_ms" in result:
        result["speedup_pct"] = (
            result["fp8_ms"] / result["bf16_ms"] - 1.0
        ) * -100.0  # positive = FP8 faster

    return result


# ---------------------------------------------------------------------------
# Pretty print helpers
# ---------------------------------------------------------------------------

def _ws_str(ws):
    if ws == (-1, 0):
        return "causal"
    elif ws == (-1, -1):
        return "full"
    else:
        return f"w={ws}"


def print_kernel_table(
    batch_sizes, seqlens, nheads_list, headdims, window_sizes
):
    print("=" * 100)
    print("KERNEL-ONLY BENCHMARK  (FP8 quantization not included)")
    print("=" * 100)
    hdr = (f"{'Config':<56} {'BF16(ms)':>9} {'BF16 TFLOPS':>12}"
           f" {'FP8(ms)':>9} {'FP8 TFLOPS':>12} {'Speedup':>9}")
    print(hdr)
    print("-" * len(hdr))

    for bs in batch_sizes:
        for seqlen in seqlens:
            for nheads in nheads_list:
                for headdim in headdims:
                    for ws in window_sizes:
                        cfg = (f"bs={bs} seq={seqlen} h={nheads} "
                               f"d={headdim} {_ws_str(ws)}")
                        res = bench_kernel_only(bs, seqlen, nheads, headdim, ws)

                        bf16_str = (f"{res['bf16_ms']:9.3f}"
                                    if "bf16_ms" in res
                                    else f"{'ERR':>9}")
                        bf16_tf = (f"{res['bf16_tflops']:12.1f}"
                                   if "bf16_tflops" in res
                                   else f"{'ERR':>12}")
                        fp8_str = (f"{res['fp8_ms']:9.3f}"
                                   if "fp8_ms" in res
                                   else f"{'ERR':>9}")
                        fp8_tf = (f"{res['fp8_tflops']:12.1f}"
                                  if "fp8_tflops" in res
                                  else f"{'ERR':>12}")
                        speedup = (f"{res['speedup_pct']:+8.1f}%"
                                   if "speedup_pct" in res
                                   else f"{'N/A':>9}")

                        print(f"  {cfg:<54} {bf16_str} {bf16_tf} {fp8_str} {fp8_tf} {speedup}")

                        if "bf16_error" in res:
                            print(f"    BF16 ERROR: {res['bf16_error']}")
                        if "fp8_error" in res:
                            print(f"    FP8  ERROR: {res['fp8_error']}")
    print()


def print_e2e_table(
    batch_sizes, seqlens, nheads_list, headdims, window_sizes
):
    print("=" * 100)
    print("END-TO-END BENCHMARK  (includes FP8 quantization overhead)")
    print("=" * 100)
    hdr = (f"{'Config':<56} {'BF16(ms)':>9} {'BF16 TFLOPS':>12}"
           f" {'FP8 e2e(ms)':>11} {'FP8 TFLOPS':>12} {'Speedup':>9}")
    print(hdr)
    print("-" * len(hdr))

    for bs in batch_sizes:
        for seqlen in seqlens:
            for nheads in nheads_list:
                for headdim in headdims:
                    for ws in window_sizes:
                        cfg = (f"bs={bs} seq={seqlen} h={nheads} "
                               f"d={headdim} {_ws_str(ws)}")
                        res = bench_e2e(bs, seqlen, nheads, headdim, ws)

                        bf16_str = (f"{res['bf16_ms']:9.3f}"
                                    if "bf16_ms" in res
                                    else f"{'ERR':>9}")
                        bf16_tf = (f"{res['bf16_tflops']:12.1f}"
                                   if "bf16_tflops" in res
                                   else f"{'ERR':>12}")
                        fp8_str = (f"{res['fp8_ms']:11.3f}"
                                   if "fp8_ms" in res
                                   else f"{'ERR':>11}")
                        fp8_tf = (f"{res['fp8_tflops']:12.1f}"
                                  if "fp8_tflops" in res
                                  else f"{'ERR':>12}")
                        speedup_raw = res.get("speedup_pct")
                        if speedup_raw is not None:
                            speedup = f"{speedup_raw:+8.1f}%"
                        else:
                            speedup = f"{'N/A':>9}"

                        print(f"  {cfg:<54} {bf16_str} {bf16_tf} {fp8_str} {fp8_tf} {speedup}")

                        if "bf16_error" in res:
                            print(f"    BF16 ERROR: {res['bf16_error']}")
                        if "fp8_error" in res:
                            print(f"    FP8  ERROR: {res['fp8_error']}")
    print()


# ---------------------------------------------------------------------------
# Accuracy check
# ---------------------------------------------------------------------------

def check_accuracy(
    batch_sizes, seqlens, nheads_list, headdims, window_sizes
):
    print("=" * 100)
    print("ACCURACY CHECK  (FP8 quant_mode=2 vs BF16)")
    print("=" * 100)
    hdr = f"{'Config':<56} {'max_err':>10} {'mean_err':>10} {'cos_sim':>10} {'PASS?':>6}"
    print(hdr)
    print("-" * len(hdr))

    all_pass = True
    for bs in batch_sizes:
        for seqlen in seqlens:
            for nheads in nheads_list:
                for headdim in headdims:
                    for ws in window_sizes:
                        cfg = (f"bs={bs} seq={seqlen} h={nheads} "
                               f"d={headdim} {_ws_str(ws)}")
                        try:
                            q_bf16, k_bf16, v_bf16, cu_seqlens = make_bf16_inputs(
                                bs, seqlen, nheads, headdim
                            )
                            # BF16 reference
                            ref = run_kernel_bf16(
                                q_bf16, k_bf16, v_bf16, cu_seqlens,
                                seqlen, bs, ws
                            ).float()

                            # FP8 output
                            (q_fp8, k_fp8, v_fp8,
                             sf_q, sf_k, sf_v,
                             q_dsc, k_dsc, v_dsc,
                             cu_q_blk, cu_kv_blk, cu_v_blk,
                             cu_seqlens2) = quantize_fp8_bs(
                                q_bf16, k_bf16, v_bf16, bs, seqlen, nheads, headdim
                            )
                            fp8_out = run_kernel_fp8bs(
                                q_fp8, k_fp8, v_fp8, sf_q, sf_k, sf_v,
                                q_dsc, k_dsc, v_dsc,
                                cu_seqlens2, cu_q_blk, cu_kv_blk, cu_v_blk,
                                seqlen, ws
                            ).float()

                            max_err = (fp8_out - ref).abs().max().item()
                            mean_err = (fp8_out - ref).abs().mean().item()
                            cos_sim = F.cosine_similarity(
                                fp8_out.flatten().unsqueeze(0),
                                ref.flatten().unsqueeze(0),
                            ).item()
                            passed = cos_sim > 0.99
                            if not passed:
                                all_pass = False
                            status = "PASS" if passed else "FAIL"
                            print(f"  {cfg:<54} {max_err:10.5f} {mean_err:10.5f} {cos_sim:10.6f} {status:>6}")
                        except Exception as e:
                            all_pass = False
                            print(f"  {cfg:<54} ERROR: {e}")
    print()
    if all_pass:
        print("All accuracy checks PASSED.")
    else:
        print("Some accuracy checks FAILED!")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SM120 HSTU Attention Benchmark (BF16 vs FP8 block-scale)"
    )
    parser.add_argument(
        "--mode",
        choices=["kernel", "e2e", "accuracy", "all"],
        default="all",
        help="Which benchmark to run (default: all)",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 8],
        metavar="BS",
    )
    parser.add_argument(
        "--seqlens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096],
        metavar="S",
    )
    parser.add_argument(
        "--nheads", type=int, nargs="+", default=[4, 16],
        metavar="H",
    )
    parser.add_argument(
        "--headdims", type=int, nargs="+", default=[128],
        metavar="D",
    )
    parser.add_argument(
        "--causal-only", action="store_true",
        help="Only benchmark causal attention",
    )
    parser.add_argument(
        "--full-only", action="store_true",
        help="Only benchmark full attention",
    )
    parser.add_argument(
        "--warmup", type=int, default=WARMUP,
        help=f"Number of warmup iterations (default: {WARMUP})",
    )
    parser.add_argument(
        "--iters", type=int, default=ITERS,
        help=f"Number of timed iterations (default: {ITERS})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    check_sm120()

    global WARMUP, ITERS
    WARMUP = args.warmup
    ITERS = args.iters

    if args.causal_only:
        window_sizes = [(-1, 0)]
    elif args.full_only:
        window_sizes = [(-1, -1)]
    else:
        window_sizes = [(-1, -1), (-1, 0)]

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Capability: SM{torch.cuda.get_device_capability()[0]}"
          f"{torch.cuda.get_device_capability()[1]}")
    print(f"Warmup={WARMUP} Iters={ITERS}")
    print()

    if args.mode in ("accuracy", "all"):
        # Use smaller config for accuracy check to save time
        check_accuracy(
            batch_sizes=[1],
            seqlens=[128, 256, 512],
            nheads_list=[1, 4],
            headdims=args.headdims,
            window_sizes=[(-1, -1)],
        )

    if args.mode in ("kernel", "all"):
        print_kernel_table(
            batch_sizes=args.batch_sizes,
            seqlens=args.seqlens,
            nheads_list=args.nheads,
            headdims=args.headdims,
            window_sizes=window_sizes,
        )

    if args.mode in ("e2e", "all"):
        print_e2e_table(
            batch_sizes=args.batch_sizes,
            seqlens=args.seqlens,
            nheads_list=args.nheads,
            headdims=args.headdims,
            window_sizes=window_sizes,
        )


if __name__ == "__main__":
    main()
