"""Kernel-only matrix benchmark for the D256 cuTile FP8 prototype."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace

import torch

from . import kernel_d256
from .wrapper import (
    TileToolchainError,
    _autotune_compiler_timeout_seconds,
    _autotune_search_space,
    _compiler_hints,
    _compiler_timeout,
    _num_launch_ctas,
    require_available,
)

try:
    from cuda.tile.tune import exhaustive_search
    import cuda.tile.tune._tune as tune_impl
except Exception as exc:  # pragma: no cover - availability path.
    exhaustive_search = None
    tune_impl = None
    _TUNE_IMPORT_ERROR = exc
else:
    _TUNE_IMPORT_ERROR = None


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


def _make_dense_inputs(batch: int, seqlen: int, heads: int):
    shape = (batch, seqlen, heads, 256)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(shape, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    q_scale = torch.ones(
        (batch, heads, seqlen, kernel_d256.QK_SCALE_BLOCKS_D256),
        dtype=torch.float8_e8m0fnu,
        device="cuda",
    )
    k_scale = torch.ones_like(q_scale)
    q_lengths = torch.full((batch,), seqlen, dtype=torch.int32, device="cuda")
    k_lengths = torch.full_like(q_lengths, seqlen)
    out = torch.empty(shape, dtype=torch.float16, device="cuda")
    v_scale_cache: dict[int, torch.Tensor] = {}

    def v_scale_for(tile_n: int) -> torch.Tensor:
        cached = v_scale_cache.get(tile_n)
        if cached is not None:
            return cached
        v_scale = torch.ones(
            (batch, heads, math.ceil(seqlen / tile_n), tile_n // 32, 256),
            dtype=torch.float8_e8m0fnu,
            device="cuda",
        )
        v_scale_cache[tile_n] = v_scale
        return v_scale

    return q, k, v, q_scale, k_scale, q_lengths, k_lengths, out, v_scale_for


def _grid(batch: int, seqlen: int, heads: int, tile_m: int, causal: bool) -> tuple[int, int, int]:
    total_tiles = batch * heads * math.ceil(seqlen / tile_m)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    return (_num_launch_ctas(total_tiles, sms, causal), 1, 1)


def _autotune_case(
    q,
    k,
    v,
    q_scale,
    k_scale,
    q_lengths,
    k_lengths,
    out,
    v_scale_for,
    batch: int,
    seqlen: int,
    heads: int,
    causal: bool,
    stream,
):
    if exhaustive_search is None:
        raise TileToolchainError(f"cuda.tile.tune import failed: {_TUNE_IMPORT_ERROR}")
    _configure_autotune_measurement()
    with _compiler_timeout(_autotune_compiler_timeout_seconds()):
        return exhaustive_search(
            _autotune_search_space(),
            stream,
            grid_fn=lambda cfg: _grid(batch, seqlen, heads, cfg.TILE_M, causal),
            kernel=kernel_d256.hstu_fp8_d256_kernel,
            args_fn=lambda cfg: (
                q,
                k,
                v,
                q_scale,
                k_scale,
                v_scale_for(cfg.TILE_N),
                q_lengths,
                k_lengths,
                out,
                1.0,
                seqlen,
                causal,
                not causal,
                cfg.TILE_M,
                cfg.TILE_N,
            ),
            hints_fn=_compiler_hints,
            quiet=True,
        )


def _configure_autotune_measurement() -> None:
    if tune_impl is None:
        return
    tune_impl._WARM_UP_STEPS = int(
        os.environ.get("HSTU_CUTILE_FP8_AUTOTUNE_WARMUP_STEPS", "3")
    )
    tune_impl._MIN_REPEATS = int(os.environ.get("HSTU_CUTILE_FP8_AUTOTUNE_MIN_REPEATS", "5"))
    tune_impl._MAX_REPEATS = int(os.environ.get("HSTU_CUTILE_FP8_AUTOTUNE_MAX_REPEATS", "100"))


def _run_case(
    *,
    batch: int,
    seqlen: int,
    heads: int,
    causal: bool,
    warmup: int,
    iters: int,
    autotune: bool,
) -> dict[str, object]:
    q, k, v, q_scale, k_scale, q_lengths, k_lengths, out, v_scale_for = _make_dense_inputs(
        batch, seqlen, heads
    )
    stream = torch.cuda.current_stream()
    if autotune:
        result = _autotune_case(
            q,
            k,
            v,
            q_scale,
            k_scale,
            q_lengths,
            k_lengths,
            out,
            v_scale_for,
            batch,
            seqlen,
            heads,
            causal,
            stream,
        )
        cfg = result.best.config
        tune_us = result.best.mean_us
        ok = len(result.successes)
        failed = len(result.failures)
    else:
        cfg = SimpleNamespace(
            TILE_M=kernel_d256.DEFAULT_TILE_M_D256,
            TILE_N=kernel_d256.DEFAULT_TILE_N_D256,
            num_ctas=kernel_d256.KERNEL_NUM_CTAS_D256,
            occupancy=kernel_d256.KERNEL_OCCUPANCY_D256,
            num_worker_warps=kernel_d256.KERNEL_WORKER_WARPS_D256,
        )
        tune_us = float("nan")
        ok = 0
        failed = 0

    tuned_kernel = kernel_d256.hstu_fp8_d256_kernel.replace_hints(
        **_compiler_hints(cfg)
    )
    grid = _grid(batch, seqlen, heads, cfg.TILE_M, causal)

    def launch() -> None:
        kernel_d256.ct.launch(
            stream,
            grid,
            tuned_kernel,
            (
                q,
                k,
                v,
                q_scale,
                k_scale,
                v_scale_for(cfg.TILE_N),
                q_lengths,
                k_lengths,
                out,
                1.0,
                seqlen,
                causal,
                not causal,
                int(cfg.TILE_M),
                int(cfg.TILE_N),
            ),
        )

    ms = _time_kernel_ms(launch, warmup, iters)
    pairs = batch * heads * seqlen * seqlen
    if causal:
        pairs = batch * heads * seqlen * (seqlen + 1) // 2
    tflops = (4.0 * pairs * 256) / (ms * 1e-3) / 1e12
    return {
        "bs": batch,
        "seq": seqlen,
        "heads": heads,
        "d": 256,
        "mask": "causal" if causal else "full",
        "ms": ms,
        "tflops": tflops,
        "tile_m": cfg.TILE_M,
        "tile_n": cfg.TILE_N,
        "num_ctas": cfg.num_ctas,
        "occupancy": cfg.occupancy,
        "num_worker_warps": getattr(cfg, "num_worker_warps", None),
        "autotune_us": tune_us,
        "autotune_successes": ok,
        "autotune_failures": failed,
        "grid": grid[0],
    }


def _load_cpp_baseline(path: str | None) -> dict[tuple[int, int, int, str], float]:
    if path is None:
        return {}
    baseline: dict[tuple[int, int, int, str], float] = {}
    pattern = re.compile(r"bs=(\d+) seq=(\d+) h=(\d+) d=256 (full|causal)\s+(.+)")
    for line in Path(path).read_text().splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        bs, seq, heads, mask, tail = match.groups()
        nums = []
        for token in tail.split():
            try:
                nums.append(float(token))
            except ValueError:
                continue
        if nums:
            baseline[(int(bs), int(seq), int(heads), mask)] = nums[0]
    return baseline


def _write_markdown(path: str, rows: list[dict[str, object]]) -> None:
    lines = [
        "# D256 cuTile FP8 Matrix Benchmark",
        "",
        "| bs | seq | h | mask | ms | TFLOPS | best tile | hints | grid | autotune ok/fail | C++ ms | cuTile/C++ |",
        "|---:|---:|---:|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        cxx_ms = row.get("cpp_fp8_ms")
        ratio = row.get("cutile_over_cpp")
        cxx = "" if cxx_ms is None else f"{float(cxx_ms):.6f}"
        ratio_text = "" if ratio is None else f"{float(ratio):.3f}x"
        hints = (
            f"occ={row['occupancy']},ctas={row['num_ctas']},"
            f"warps={row['num_worker_warps']}"
        )
        lines.append(
            f"| {row['bs']} | {row['seq']} | {row['heads']} | {row['mask']} "
            f"| {float(row['ms']):.6f} | {float(row['tflops']):.2f} "
            f"| {row['tile_m']}x{row['tile_n']} | {hints} | {row['grid']} "
            f"| {row['autotune_successes']}/{row['autotune_failures']} "
            f"| {cxx} | {ratio_text} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--seqlens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096]
    )
    parser.add_argument("--heads", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--masks", nargs="+", choices=("full", "causal"), default=["full", "causal"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-autotune", action="store_true")
    parser.add_argument("--csv", default="")
    parser.add_argument("--md", default="")
    parser.add_argument("--cpp-baseline", default="")
    args = parser.parse_args()

    require_available()
    baseline = _load_cpp_baseline(args.cpp_baseline or None)
    print(
        "SEARCH_SPACE "
        f"candidates={len(_autotune_search_space())} "
        f"autotune={int(not args.no_autotune)}"
    )

    rows: list[dict[str, object]] = []
    for bs in args.batch_sizes:
        for seq in args.seqlens:
            for heads in args.heads:
                for mask in args.masks:
                    row = _run_case(
                        batch=bs,
                        seqlen=seq,
                        heads=heads,
                        causal=(mask == "causal"),
                        warmup=args.warmup,
                        iters=args.iters,
                        autotune=not args.no_autotune,
                    )
                    cpp_ms = baseline.get((bs, seq, heads, mask))
                    if cpp_ms is not None:
                        row["cpp_fp8_ms"] = cpp_ms
                        row["cutile_over_cpp"] = float(row["ms"]) / cpp_ms
                    rows.append(row)
                    ratio = row.get("cutile_over_cpp")
                    ratio_text = "" if ratio is None else f" cxx_ratio={float(ratio):.3f}x"
                    print(
                        "RESULT "
                        f"bs={bs} seq={seq} h={heads} d=256 mask={mask} "
                        f"ms={float(row['ms']):.6f} tflops={float(row['tflops']):.2f} "
                        f"tile={row['tile_m']}x{row['tile_n']} "
                        f"occ={row['occupancy']} num_ctas={row['num_ctas']} "
                        f"worker_warps={row['num_worker_warps']} "
                        f"grid={row['grid']} "
                        f"autotune_ok={row['autotune_successes']} "
                        f"autotune_fail={row['autotune_failures']}"
                        f"{ratio_text}"
                    )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if args.md:
        _write_markdown(args.md, rows)


if __name__ == "__main__":
    main()
