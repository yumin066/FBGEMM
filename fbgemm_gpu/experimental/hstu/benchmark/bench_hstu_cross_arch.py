#!/usr/bin/env python3
"""
Cross-architecture HSTU Attention Kernel-Only Benchmark.

Benchmarks the raw CUDA attention forward kernel (excluding Python-level
quantization overhead) across SM80 (Ampere), SM90 (Hopper), and SM120
(Blackwell RTX Pro).

Supported dtypes per architecture:
  SM80  — BF16
  SM90  — BF16
  SM120 — BF16, FP8 block-scale (quant_mode=2)

SM100 (Blackwell GB) uses a Python implementation with no stable torch.ops
kernel entry point; it is excluded from this kernel-only comparison.

Usage:
  python bench_hstu_cross_arch.py
  python bench_hstu_cross_arch.py --seqlens 1024 2048 4096
  python bench_hstu_cross_arch.py --batch-sizes 1 4 8 --heads 16
  python bench_hstu_cross_arch.py --causal          # causal only
  python bench_hstu_cross_arch.py --full            # full only
  python bench_hstu_cross_arch.py --out-csv results.csv
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

# ── sys.path setup ────────────────────────────────────────────────────────────
# Source-tree path is always registered as a fallback so the script can be
# invoked directly during development (without bench_cross_arch.sh --build).
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_HSTU_ROOT = os.path.normpath(os.path.join(_BENCH_DIR, ".."))
if _HSTU_ROOT not in sys.path:
    sys.path.insert(0, _HSTU_ROOT)

# When bench_cross_arch.sh builds into a per-arch prefix (.build/sm<N>x/), it
# exports HSTU_ARCH_PYSITE so we load that arch's compiled .so in preference
# to any stale artifact that might live in the shared source tree.
_arch_pysite = os.environ.get("HSTU_ARCH_PYSITE", "")
if _arch_pysite and os.path.isdir(_arch_pysite) and _arch_pysite not in sys.path:
    sys.path.insert(0, _arch_pysite)

try:
    from hstu.cuda_hstu_attention import (
        pack_descale_to_e8m0x4_int32,
        quantize_for_block_scale_qk_along_d,
        quantize_for_block_scale_v_along_n,
    )
    import hstu  # noqa: F401 — side-effect: registers torch.ops.fbgemm.*
except ImportError as exc:
    sys.exit(f"ERROR: Cannot import hstu from {_HSTU_ROOT}: {exc}\n"
             "Make sure the CUDA extension is built (run with --build).")

WARMUP: int = 10
ITERS:  int = 50


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchConfig:
    batch_size: int
    seqlen:     int
    nheads:     int
    headdim:    int
    causal:     bool = False

    def flops(self) -> float:
        """GEMM-equivalent FLOPs: 2× (Q@K^T) + 2× (P@V), each MAC = 2 FLOPs."""
        B, S, H, D = self.batch_size, self.seqlen, self.nheads, self.headdim
        if self.causal:
            # effective pairs per sequence: S*(S+1)/2
            return 2.0 * B * H * D * S * (S + 1)
        return 4.0 * B * H * D * S * S

    def window(self) -> Tuple[int, int]:
        return (-1, 0) if self.causal else (-1, -1)

    def label(self) -> str:
        mask = "causal" if self.causal else "full"
        return f"bs={self.batch_size} seq={self.seqlen} h={self.nheads} {mask}"


# ─────────────────────────────────────────────────────────────────────────────
# Input helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cu_seqlens(batch_size: int, seqlen: int) -> torch.Tensor:
    cu = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    cu[1:] = torch.arange(1, batch_size + 1, dtype=torch.int32, device="cuda") * seqlen
    return cu


def make_bf16_inputs(cfg: BenchConfig):
    total = cfg.batch_size * cfg.seqlen
    q = torch.randn(total, cfg.nheads, cfg.headdim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(total, cfg.nheads, cfg.headdim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(total, cfg.nheads, cfg.headdim, dtype=torch.bfloat16, device="cuda")
    cu = _cu_seqlens(cfg.batch_size, cfg.seqlen)
    return q, k, v, cu


def make_fp8_inputs_sm120(cfg: BenchConfig, q_bf16, k_bf16, v_bf16):
    """
    Quantize BF16 Q/K/V to FP8 block-scale format required by SM120 quant_mode=2.

    Q/K: block-scale along D (headdim), granularity=128.
    V  : block-scale along N (sequence), block_size=128.
    SF tensors are packed to e8m0×4 int32 as expected by the kernel.
    """
    cu = _cu_seqlens(cfg.batch_size, cfg.seqlen)
    # Clamp to FP8 representable range before quantization
    q_in = q_bf16.to(torch.float8_e4m3fn).to(torch.bfloat16)
    k_in = k_bf16.to(torch.float8_e4m3fn).to(torch.bfloat16)
    v_in = v_bf16.to(torch.float8_e4m3fn).to(torch.bfloat16)

    q_fp8, q_dsc, cu_q_blk  = quantize_for_block_scale_qk_along_d(
        q_in, cu, fp8_type=torch.float8_e4m3fn)
    k_fp8, k_dsc, cu_kv_blk = quantize_for_block_scale_qk_along_d(
        k_in, cu, fp8_type=torch.float8_e4m3fn)
    v_fp8, v_dsc, cu_v_blk  = quantize_for_block_scale_v_along_n(
        v_in, cu, block_size=128, fp8_type=torch.float8_e4m3fn)

    sf_q = pack_descale_to_e8m0x4_int32(q_dsc)
    sf_k = pack_descale_to_e8m0x4_int32(k_dsc)
    # SFV is tiled: each scale covers kBlockN=128 tokens, repeat to match token count
    sf_v = pack_descale_to_e8m0x4_int32(v_dsc).repeat_interleave(128, dim=1)

    return (q_fp8, k_fp8, v_fp8,
            sf_q, sf_k, sf_v,
            q_dsc, k_dsc, v_dsc,
            cu, cu_q_blk, cu_kv_blk, cu_v_blk)


# ─────────────────────────────────────────────────────────────────────────────
# Arch-specific kernel runners (call torch.ops directly to exclude Python overhead)
# ─────────────────────────────────────────────────────────────────────────────

def run_sm80_bf16(q, k, v, cu, cfg: BenchConfig):
    wl, wr = cfg.window()
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_80(
        q, k, v,
        cu, cu,
        None, None,                   # seqused_q, seqused_k
        cfg.seqlen, cfg.seqlen, cfg.seqlen,
        None, None, 1,                # num_contexts, num_targets, target_group_size
        wl, wr,
        1.0,                          # alpha
        None, None,                   # rab, func
        None, None, None, None,       # kv_cache, page_offsets, page_ids, last_page_lens
    )
    return out


def run_sm90_bf16(q, k, v, cu, cfg: BenchConfig):
    wl, wr = cfg.window()
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_90(
        q, k, v,
        cu, cu,
        None, None,                   # seqused_q, seqused_k
        cfg.seqlen, cfg.seqlen, cfg.seqlen,
        None, None, 1,                # num_contexts, num_targets, target_group_size
        wl, wr,
        1.0,                          # alpha
        None, None,                   # rab, func
        -1,                           # quant_mode=-1 → BF16 pass-through
        0,                            # output_dtype: 0=BF16
        None, None,                   # vt, cu_seqlens_vt_descale
        None, None, None, None,       # q/k/v/vt descale
        None, None,                   # cu_seqlens_q/kv_block_descale
    )
    return out


def run_sm120_bf16(q, k, v, cu, cfg: BenchConfig):
    wl, wr = cfg.window()
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        q, k, v,
        cu, cu,
        None, None,                   # seqused_q, seqused_k
        cfg.seqlen, cfg.seqlen, cfg.seqlen,
        None, None, 1,                # num_contexts, num_targets, target_group_size
        wl, wr,
        1.0,                          # alpha
        None, None,                   # rab, func
        -1,                           # quant_mode=-1 → BF16
        None, None, None,             # descale_q/k/v
        None, None, None,             # sf_q/k/v
        None, None, None,             # cu_q_blk/kv_blk/v_blk
    )
    return out


def run_sm120_fp8(fp8_inputs, cfg: BenchConfig):
    (q_fp8, k_fp8, v_fp8,
     sf_q, sf_k, sf_v,
     q_dsc, k_dsc, v_dsc,
     cu, cu_q_blk, cu_kv_blk, cu_v_blk) = fp8_inputs
    wl, wr = cfg.window()
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        q_fp8, k_fp8, v_fp8,
        cu, cu,
        None, None,
        cfg.seqlen, cfg.seqlen, cfg.seqlen,
        None, None, 1,
        wl, wr,
        1.0,
        None, None,
        2,                            # quant_mode=2 → FP8 block-scale
        q_dsc, k_dsc, v_dsc,
        sf_q, sf_k, sf_v,
        cu_q_blk, cu_kv_blk, cu_v_blk,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────

def time_kernel(fn, warmup: int = WARMUP, iters: int = ITERS) -> Tuple[float, float]:
    """Returns (avg_ms, min_ms) over `iters` timed iterations after `warmup`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    lats = [starts[i].elapsed_time(ends[i]) for i in range(iters)]
    return sum(lats) / len(lats), min(lats)


# ─────────────────────────────────────────────────────────────────────────────
# Per-architecture benchmark dispatch
# ─────────────────────────────────────────────────────────────────────────────

_BF16_RUNNERS = {
    8:  run_sm80_bf16,
    9:  run_sm90_bf16,
    12: run_sm120_bf16,
}


def bench_config(cfg: BenchConfig, sm_major: int, gpu_name: str = "") -> Dict:
    result: Dict = {"config": cfg.label(), "gpu_name": gpu_name, "flops": cfg.flops()}
    q, k, v, cu = make_bf16_inputs(cfg)

    # BF16 kernel
    runner = _BF16_RUNNERS.get(sm_major)
    if runner is None:
        result["bf16_error"] = f"No BF16 runner for SM{sm_major}x"
    else:
        try:
            avg_ms, _ = time_kernel(lambda: runner(q, k, v, cu, cfg))
            result["bf16_ms"]     = avg_ms
            result["bf16_tflops"] = cfg.flops() / (avg_ms * 1e-3) / 1e12
        except Exception as exc:
            result["bf16_error"] = str(exc)

    # FP8 block-scale kernel — SM120 only
    if sm_major == 12:
        try:
            fp8_inputs = make_fp8_inputs_sm120(cfg, q, k, v)
            avg_ms, _ = time_kernel(lambda: run_sm120_fp8(fp8_inputs, cfg))
            result["fp8_ms"]     = avg_ms
            result["fp8_tflops"] = cfg.flops() / (avg_ms * 1e-3) / 1e12
            if "bf16_tflops" in result:
                result["speedup_pct"] = (
                    result["fp8_tflops"] / result["bf16_tflops"] - 1.0
                ) * 100.0
        except Exception as exc:
            result["fp8_error"] = str(exc)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(result: Dict, ms_key: str, tf_key: str) -> Tuple[str, str]:
    err_key = tf_key.replace("_tflops", "_error")
    if err_key in result:
        return "ERR", "ERR"
    if ms_key not in result:
        return "N/A", "N/A"
    return f"{result[ms_key]:.3f}", f"{result[tf_key]:.1f}"


def print_results(results: List[Dict], sm_major: int, gpu_name: str) -> None:
    has_fp8 = sm_major == 12
    w = 80 if not has_fp8 else 100
    print(f"\n{'=' * w}")
    print(f"  HSTU Attention — Kernel-Only Benchmark")
    print(f"  GPU : {gpu_name}  (SM{sm_major}x)   "
          f"warmup={WARMUP}  iters={ITERS}")
    print(f"{'=' * w}")

    if has_fp8:
        header = (f"{'Config':<40} {'BF16 ms':>8} {'BF16 TF/s':>10}"
                  f" {'FP8 ms':>7} {'FP8 TF/s':>9} {'Speedup':>8}")
    else:
        header = f"{'Config':<40} {'BF16 ms':>8} {'BF16 TF/s':>10}"
    print(header)
    print("-" * (len(header) + 2))

    for r in results:
        bf16_ms, bf16_tf = _fmt(r, "bf16_ms", "bf16_tflops")
        row = f"{r['config']:<40} {bf16_ms:>8} {bf16_tf:>10}"
        if has_fp8:
            fp8_ms, fp8_tf = _fmt(r, "fp8_ms", "fp8_tflops")
            spd = (f"{r['speedup_pct']:>+7.1f}%" if "speedup_pct" in r
                   else "    N/A")
            row += f" {fp8_ms:>7} {fp8_tf:>9} {spd:>8}"
        print(row)
    print()

    if has_fp8:
        # Summary: best FP8 speedup
        valid = [r for r in results if "speedup_pct" in r]
        if valid:
            best = max(valid, key=lambda r: r["speedup_pct"])
            print(f"  Best FP8 speedup: {best['speedup_pct']:+.1f}%  ({best['config']})")
            print()


def save_csv(results: List[Dict], path: str) -> None:
    fields = ["config", "gpu_name", "bf16_ms", "bf16_tflops", "fp8_ms", "fp8_tflops",
              "speedup_pct", "bf16_error", "fp8_error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"Results saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global WARMUP, ITERS

    parser = argparse.ArgumentParser(
        description="Cross-architecture HSTU attention kernel-only benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[4, 8],     metavar="B")
    parser.add_argument("--seqlens",     nargs="+", type=int,
                        default=[512, 1024, 2048, 4096], metavar="S")
    parser.add_argument("--heads",       nargs="+", type=int,
                        default=[8],        metavar="H")
    parser.add_argument("--headdim",     type=int,  default=128, metavar="D")
    parser.add_argument("--causal",      action="store_true",
                        help="Run causal-mask configs only")
    parser.add_argument("--full",        action="store_true",
                        help="Run full-attention configs only")
    parser.add_argument("--warmup",      type=int,  default=WARMUP)
    parser.add_argument("--iters",       type=int,  default=ITERS)
    parser.add_argument("--out-csv",     type=str,  default=None,
                        metavar="PATH",    help="Save results to CSV file")
    args = parser.parse_args()

    WARMUP = args.warmup
    ITERS  = args.iters

    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA not available.")

    sm_major, _ = torch.cuda.get_device_capability()
    gpu_name    = torch.cuda.get_device_name()

    if sm_major == 10:
        print(
            "NOTE: SM100 (Blackwell GB) uses a Python implementation with no "
            "torch.ops kernel entry point.\n"
            "      This script only benchmarks SM80/SM90/SM120 CUDA kernels.",
            file=sys.stderr,
        )
        sys.exit(0)

    if sm_major not in (8, 9, 12):
        sys.exit(f"ERROR: Unsupported SM major version {sm_major}. "
                 f"Expected 8 (Ampere), 9 (Hopper), or 12 (Blackwell RTX Pro).")

    if sm_major == 12 and args.headdim not in (64, 128):
        sys.exit(f"ERROR: SM120 kernel only supports headdim 64 or 128, got {args.headdim}.")

    # Build mask list: default = both
    mask_flags: List[bool] = []
    if args.causal:
        mask_flags.append(True)
    if args.full:
        mask_flags.append(False)
    if not mask_flags:
        mask_flags = [False, True]   # full first, then causal

    configs = [
        BenchConfig(b, s, h, args.headdim, causal)
        for b     in args.batch_sizes
        for s     in args.seqlens
        for h     in args.heads
        for causal in mask_flags
    ]

    print(f"Running {len(configs)} configurations ...")
    results: List[Dict] = []
    for cfg in configs:
        print(f"  {cfg.label()} ...", end="\r", flush=True)
        results.append(bench_config(cfg, sm_major, gpu_name))

    print_results(results, sm_major, gpu_name)

    if args.out_csv:
        save_csv(results, args.out_csv)


if __name__ == "__main__":
    main()
