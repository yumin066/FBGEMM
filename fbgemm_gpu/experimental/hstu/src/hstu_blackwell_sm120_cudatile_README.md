# SM120 cuTile HSTU Prototypes

This documents the local environment and commands for the experimental SM120
cuTile HSTU prototypes:

- `hstu_blackwell_sm120_cudatile_bf16`: BF16 D256 dense full/causal prototype.
- `hstu_blackwell_sm120_cudatile_fp8_ws`: FP8 D256 block-scale full/causal
  prototype.

These paths are not wired into production HSTU dispatch.

## Environment

Use the repo-local Docker wrapper from the repository root. The currently
validated FP8 path uses the cuTile nightly package plus the `nv-triton` bundled
`tileiras`; the pip RC `nvidia-cuda-tileiras==13.3.44` was observed to fail on
the current `mma_scaled` kernel.

```bash
export CUTILE_DOCKER_GPUS=2
export CUTILE_DOCKER_PULL=0
export CUTILE_DOCKER_IMAGE=nvcr.io/nvidia/pytorch:26.03-py3
export CUTILE_USE_NVTRITON_TILEIRAS=1
```

One-time setup:

```bash
./docker_cutile.sh install-nvtriton-tileiras
./docker_cutile.sh bootstrap-cutile
./docker_cutile.sh test-mma-scaled
```

Interactive shell:

```bash
./docker_cutile.sh
```

## Smoke Tests

BF16 wrapper correctness:

```bash
./docker_cutile.sh python -m hstu_blackwell_sm120_cudatile_bf16.test_d256
```

FP8 direct kernel smoke. This does not import the production `hstu` package and
is useful when the full HSTU extension is not built in the cuTile container:

```bash
./docker_cutile.sh python -m hstu_blackwell_sm120_cudatile_fp8_ws.sweep_d256 \
  --batch 1 --seqlen 128 --heads 1 --warmup 1 --iters 1
```

FP8 wrapper correctness requires the HSTU Python package and its CUDA extension
to be importable:

```bash
./docker_cutile.sh python -m hstu_blackwell_sm120_cudatile_fp8_ws.test_d256
```

## Benchmarks

The FP8 direct kernel benchmark is the clean kernel-only path:

```bash
./docker_cutile.sh python -m hstu_blackwell_sm120_cudatile_fp8_ws.sweep_d256 \
  --batch 1 --seqlen 2048 --heads 4 --warmup 5 --iters 20
./docker_cutile.sh python -m hstu_blackwell_sm120_cudatile_fp8_ws.sweep_d256 \
  --batch 1 --seqlen 2048 --heads 4 --causal --warmup 5 --iters 20
```

Wrapper benchmarks include dense materialization, output compaction, and
autotune overhead on the first call:

```bash
./docker_cutile.sh python -m hstu_blackwell_sm120_cudatile_bf16.bench_d256
./docker_cutile.sh python -m hstu_blackwell_sm120_cudatile_fp8_ws.bench_d256
```

Both wrappers default to autotune. The selected config is cached per
`(B, H, max_q, max_k, causal)` shape in-process.

The autotune search spaces are generated programmatically instead of using the
old fixed 8-candidate list. By default:

- FP8 D256: `TILE_M=16,32,64,128,256`, `TILE_N=32,64,128,256`,
  `num_ctas=None,1`, `occupancy=1,2`.
- BF16 D256: the same tile/CTA/occupancy grid.

The current cuTile compiler options do not support `num_worker_warps`, so that
hint is not included unless a future toolchain exposes it.
