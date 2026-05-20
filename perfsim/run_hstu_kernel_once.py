#!/usr/bin/env python3
"""
Minimal HSTU SM120 FP8 WS kernel runner for CUDA trace capture.

Runs the FP8 kernel N times so cuda_apic_capture.pl can capture
a specific invocation via frange (default: capture 4th, frange=3:3).

Usage:
  python run_hstu_kernel_once.py [--bs 8] [--seq 4096] [--heads 16] [--causal]
"""

import argparse
import os
import sys

# Arch-specific site-packages (set by collect_hstu_trace.sh)
_pysite = os.environ.get("HSTU_ARCH_PYSITE", "")
if _pysite and os.path.isdir(_pysite) and _pysite not in sys.path:
    sys.path.insert(0, _pysite)

_repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(_repo_root, "fbgemm_gpu/experimental/hstu"))

import torch

try:
    from hstu.cuda_hstu_attention import (
        quantize_for_block_scale_qk_along_d,
        quantize_for_block_scale_v_along_n,
        pack_descale_to_e8m0x4_int32,
    )
except ImportError:
    try:
        from hstu.cuda_hstu_attention import (
            quantize_for_block_scale,
            pack_descale_to_e8m0x4_int32,
        )
    except ImportError as e:
        print(f"ERROR: cannot import hstu quantization helpers: {e}", file=sys.stderr)
        sys.exit(1)

    def quantize_for_block_scale_qk_along_d(x, seq_offsets, fp8_type=torch.float8_e4m3fn):
        return quantize_for_block_scale(
            x,
            seq_offsets,
            fp8_type=fp8_type,
            scale_mode="token_dchunk",
            d_chunk_size=128,
            round_to_e8m0=True,
        )

    def quantize_for_block_scale_v_along_n(
        x,
        seq_offsets,
        block_size=128,
        fp8_type=torch.float8_e4m3fn,
    ):
        return quantize_for_block_scale(
            x,
            seq_offsets,
            block_size=block_size,
            fp8_type=fp8_type,
            scale_mode="seq_block",
            round_to_e8m0=True,
        )

try:
    import hstu  # noqa: F401 — triggers compiled .so load
except ImportError as e:
    print(f"ERROR: cannot import hstu: {e}", file=sys.stderr)
    sys.exit(1)


def make_cu_seqlens(batch_size, seqlen):
    lengths = torch.full((batch_size,), seqlen, dtype=torch.int32, device="cuda")
    cu = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    cu[1:] = torch.cumsum(lengths, dim=0)
    return cu


def prepare_fp8_inputs(batch_size, seqlen, nheads, headdim):
    total = batch_size * seqlen
    cu_seqlens = make_cu_seqlens(batch_size, seqlen)

    q_bf16 = torch.randn(total, nheads, headdim, dtype=torch.bfloat16, device="cuda")
    k_bf16 = torch.randn(total, nheads, headdim, dtype=torch.bfloat16, device="cuda")
    v_bf16 = torch.randn(total, nheads, headdim, dtype=torch.bfloat16, device="cuda")

    q_in = q_bf16.to(torch.float8_e4m3fn).to(torch.bfloat16)
    k_in = k_bf16.to(torch.float8_e4m3fn).to(torch.bfloat16)
    v_in = v_bf16.to(torch.float8_e4m3fn).to(torch.bfloat16)

    q_fp8, q_descale, cu_q_blk = quantize_for_block_scale_qk_along_d(
        q_in, cu_seqlens, fp8_type=torch.float8_e4m3fn
    )
    k_fp8, k_descale, cu_kv_blk = quantize_for_block_scale_qk_along_d(
        k_in, cu_seqlens, fp8_type=torch.float8_e4m3fn
    )
    v_fp8, v_descale, cu_v_blk = quantize_for_block_scale_v_along_n(
        v_in, cu_seqlens, block_size=128, fp8_type=torch.float8_e4m3fn
    )

    sf_q = pack_descale_to_e8m0x4_int32(q_descale)
    sf_k = pack_descale_to_e8m0x4_int32(k_descale)
    sf_v = pack_descale_to_e8m0x4_int32(v_descale).repeat_interleave(128, dim=1)

    return (q_fp8, k_fp8, v_fp8,
            sf_q, sf_k, sf_v,
            q_descale, k_descale, v_descale,
            cu_seqlens, cu_q_blk, cu_kv_blk, cu_v_blk)


def run_kernel(bs, seq, nheads, headdim, causal, n_iters):
    window_size = (-1, 0) if causal else (-1, -1)
    inputs = prepare_fp8_inputs(bs, seq, nheads, headdim)
    (q_fp8, k_fp8, v_fp8, sf_q, sf_k, sf_v,
     q_descale, k_descale, v_descale,
     cu_seqlens, cu_q_blk, cu_kv_blk, cu_v_blk) = inputs

    mask = "causal" if causal else "full"
    print(f"HSTU FP8 WS kernel: bs={bs} seq={seq} h={nheads} d={headdim} {mask} × {n_iters}")

    for i in range(n_iters):
        out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
            q_fp8, k_fp8, v_fp8,
            cu_seqlens, cu_seqlens,
            None, None,
            seq, seq,
            seq,
            None, None, 1,
            window_size[0], window_size[1],
            1.0,
            None, None,
            2,
            q_descale, k_descale, v_descale,
            sf_q, sf_k, sf_v,
            cu_q_blk, cu_kv_blk, cu_v_blk,
        )
        torch.cuda.synchronize()
        print(f"  iter {i}: output shape {out.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--iters", type=int, default=6)
    args = parser.parse_args()

    major, _ = torch.cuda.get_device_capability()
    if major < 12:
        print(f"WARNING: SM{major}x detected, expected SM12x", file=sys.stderr)

    run_kernel(args.bs, args.seq, args.heads, args.headdim, args.causal, args.iters)


if __name__ == "__main__":
    main()
