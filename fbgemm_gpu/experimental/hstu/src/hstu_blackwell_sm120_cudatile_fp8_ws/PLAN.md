# cuTile FP8 WS Plan

This directory is an isolated experiment for a cuTile implementation of SM120
HSTU FP8 block-scale forward. It does not change the production C++/CuTe path
and should not be wired into default dispatch until correctness, SASS, and
locked-clock benchmarks justify that.

## Scope

- Target: SM120, `quant_mode=2`, `torch.float8_e4m3fn` Q/K/V.
- First head dimension: `D=256`.
- First layouts: dense/non-paged Q/K/V materialized as `[B, S, H, 256]`.
- First masks: full and pure causal.
- First bias mode: no RAB/DRAB.
- Output dtype: FP16, matching the current SM120 FP8 path.

## Toolchain Requirements

- `cuda.tile` importable.
- `tileiras` with development bytecode support for `ct.mma_scaled`.
- PyTorch with `torch.float8_e4m3fn` and `torch.float8_e8m0fnu`.

The currently validated local environment is documented in
`../hstu_blackwell_sm120_cudatile_README.md`. In short, use
`docker_cutile.sh` with the cuTile nightly package and the `nv-triton` bundled
`tileiras`. The pip RC `nvidia-cuda-tileiras==13.3.44` was observed to fail on
the current `mma_scaled` kernel, so it is not the recommended path.

## Milestones

1. D256 dense full no-RAB correctness. Done.
2. D256 dense causal no-RAB correctness. Done.
3. Static grid-stride persistent full/causal. Done.
4. Pure causal paired persistent scheduler.
5. D256 RAB/DRAB direct-global add.
6. D256 local/context/target/arbitrary masks.
7. D128/D64/D32 backfill.
8. Paged KV: wrapper-side dense materialization first, native page-id path only
   after the dense path is validated.

## D256 Scale Mapping

The production HSTU FP8 path packs up to four e8m0 scales in one `int32`.
For D256 Q/K there are two 128-D chunks. cuTile `ct.mma_scaled` expects scale
blocks along K of size 32, so each 128-D scale is repeated across four 32-D
blocks:

- packed lane 0 -> scale blocks 0, 1, 2, 3
- packed lane 1 -> scale blocks 4, 5, 6, 7

For GEMM2, `P` uses unit e8m0 scale. V has one scale per 64-token block in the
current path; for cuTile it is expanded to two 32-token K-scale blocks and
replicated across the 256 output columns.

## Validation

Run only the scripts in this directory for this experimental backend. Existing
HSTU tests and benchmarks remain the authority for the production path.

See `../hstu_blackwell_sm120_cudatile_README.md` for the environment setup.
Direct kernel smoke:

```bash
./docker_cutile.sh python -m hstu_blackwell_sm120_cudatile_fp8_ws.sweep_d256 \
  --batch 1 --seqlen 128 --heads 1 --warmup 1 --iters 1
```

Benchmark:

```bash
./docker_cutile.sh python -m hstu_blackwell_sm120_cudatile_fp8_ws.sweep_d256 \
  --batch 1 --seqlen 2048 --heads 4 --warmup 5 --iters 20
```
