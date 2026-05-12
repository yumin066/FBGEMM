"""Standalone benchmark for the isolated cuTile D256 FP8 prototype."""

from __future__ import annotations

import argparse
import time

import torch

from .test_d256 import _make_inputs
from .wrapper import TileToolchainError, hstu_fp8_d256_cudatile, require_available


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    try:
        require_available()
    except TileToolchainError as exc:
        print(f"SKIP: {exc}")
        return

    (
        q,
        k,
        v,
        cu,
        _q_descale,
        _k_descale,
        _v_descale,
        _cu_q_sf,
        _cu_k_sf,
        _cu_v_sf,
        sf_q,
        sf_k,
        sf_v,
    ) = _make_inputs(args.batch, args.seqlen, args.heads)

    def run():
        return hstu_fp8_d256_cudatile(
            q,
            k,
            v,
            cu,
            cu,
            args.seqlen,
            args.seqlen,
            sf_q_packed=sf_q,
            sf_k_packed=sf_k,
            sf_v_packed=sf_v,
            causal=args.causal,
            scaling_seqlen=args.seqlen,
        )

    try:
        ms = _time_ms(run, args.warmup, args.iters)
    except TileToolchainError as exc:
        print(f"SKIP: {exc}")
        return
    total_pairs = args.batch * args.heads * args.seqlen * args.seqlen
    if args.causal:
        total_pairs = args.batch * args.heads * args.seqlen * (args.seqlen + 1) // 2
    flops = 4.0 * total_pairs * 256
    tflops = flops / (ms * 1e-3) / 1e12
    print(
        f"D256 cuTile FP8 causal={args.causal} "
        f"batch={args.batch} seqlen={args.seqlen} heads={args.heads} "
        f"ms={ms:.4f} tflops={tflops:.1f}"
    )


if __name__ == "__main__":
    main()
