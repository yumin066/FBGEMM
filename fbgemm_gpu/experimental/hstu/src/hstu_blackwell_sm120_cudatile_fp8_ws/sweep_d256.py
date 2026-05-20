"""Kernel-only sweep for the isolated cuTile D256 FP8 prototype."""

from __future__ import annotations

import argparse
import math

import torch

from . import kernel_d256
from .wrapper import TileToolchainError, require_available


def _time_kernel_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    try:
        require_available()
    except TileToolchainError as exc:
        print(f"SKIP: {exc}")
        return

    shape = (args.batch, args.seqlen, args.heads, 256)
    q = torch.randn(
        shape,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    k = torch.randn(
        shape,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    v = torch.randn(
        shape,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    q_scale = torch.ones(
        (args.batch, args.heads, args.seqlen, kernel_d256.QK_SCALE_BLOCKS_D256),
        dtype=torch.float8_e8m0fnu,
        device="cuda",
    )
    k_scale = torch.ones_like(q_scale)
    v_scale = torch.ones(
        (
            args.batch,
            args.heads,
            math.ceil(args.seqlen / kernel_d256.TILE_N_D256),
            kernel_d256.PV_SCALE_BLOCKS_BN64,
            256,
        ),
        dtype=torch.float8_e8m0fnu,
        device="cuda",
    )
    q_lengths = torch.full((args.batch,), args.seqlen, dtype=torch.int32, device="cuda")
    k_lengths = torch.full_like(q_lengths, args.seqlen)
    out = torch.empty(
        (args.batch, args.seqlen, args.heads, 256),
        dtype=torch.float16,
        device=q.device,
    )
    total_tiles = (
        args.batch
        * args.heads
        * math.ceil(args.seqlen / kernel_d256.TILE_M_D256)
    )
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = (
        total_tiles
        if args.causal or not kernel_d256.KERNEL_PERSISTENT_D256
        else min(total_tiles, sms),
        1,
        1,
    )
    stream = torch.cuda.current_stream()

    def run() -> None:
        kernel_d256.ct.launch(
            stream,
            grid,
            kernel_d256.hstu_fp8_d256_kernel,
            (
                q,
                k,
                v,
                q_scale,
                k_scale,
                v_scale,
                q_lengths,
                k_lengths,
                out,
                1.0,
                args.seqlen,
                bool(args.causal),
                not bool(args.causal),
                kernel_d256.TILE_M_D256,
                kernel_d256.TILE_N_D256,
            ),
        )

    ms = _time_kernel_ms(run, args.warmup, args.iters)
    total_pairs = args.batch * args.heads * args.seqlen * args.seqlen
    if args.causal:
        total_pairs = args.batch * args.heads * args.seqlen * (args.seqlen + 1) // 2
    flops = 4.0 * total_pairs * 256
    tflops = flops / (ms * 1e-3) / 1e12
    print(
        "RESULT "
        f"batch={args.batch} seqlen={args.seqlen} heads={args.heads} "
        f"causal={int(args.causal)} "
        f"tile_m={kernel_d256.TILE_M_D256} tile_n={kernel_d256.TILE_N_D256} "
        f"occ={kernel_d256.KERNEL_OCCUPANCY_D256} "
        f"num_ctas={kernel_d256.KERNEL_NUM_CTAS_D256} "
        f"worker_warps={kernel_d256.KERNEL_WORKER_WARPS_D256} "
        f"lat_q={kernel_d256.Q_LATENCY_D256} "
        f"lat_k={kernel_d256.K_LATENCY_D256} "
        f"lat_v={kernel_d256.V_LATENCY_D256} "
        f"lat_s={kernel_d256.SCALE_LATENCY_D256} "
        f"lat_o={kernel_d256.O_LATENCY_D256} "
        f"grid={grid[0]} ms={ms:.6f} tflops={tflops:.2f}"
    )


if __name__ == "__main__":
    main()
