# cuTile BF16 Plan

This directory is an isolated cuTile BF16 prototype for SM120 HSTU forward.
It is not wired into production dispatch and does not modify the existing
C++/CUDA HSTU kernels.

## Scope

- Target: SM120 BF16 forward.
- First head dimension: D256.
- First layout: dense/non-paged Q/K/V materialized as `[B, S, H, 256]`.
- First masks: full and pure causal.
- First bias mode: no RAB/DRAB.
- Output dtype: BF16.

## Environment

BF16 uses `ct.mma`, so the current CUDA toolkit/tileiras 13.2 environment is
enough. It does not depend on `ct.mma_scaled` or TileIR bytecode 13.3.
The shared Docker setup and command list is documented in
`../hstu_blackwell_sm120_cudatile_README.md`.

## Milestones

1. D256 dense full no-RAB correctness against PyTorch reference. Done.
2. D256 dense causal no-RAB correctness against PyTorch reference. Done.
3. Wrapper-side autotune for D256 full/causal. Done.
4. Compare against the production BF16 op when the extension is built in the
   active container.
5. Add local/context/target/arbitrary masks.
6. Add direct-global RAB/DRAB.
7. Add wrapper-side dense materialization for paged KV.
8. Backfill D128/D64/D32.

## Current Results

Local CUDA execution with the cuTile `_cext` overlay:

- `B=1/2, S=128/256, H=1/4, D=256`, full and causal: all directed cases
  reached `cos=1.000000` against the PyTorch BF16 reference.
- Production BF16 comparison for `B=1, S=128, H=1, D=256`:
  - full: `cos=0.99999994`, `max_err=0.00390625`
  - causal: `cos=0.99999988`, `max_err=0.0078125`
- Small diagnostic timings, unlocked:
  - `B=1, S=128, H=1`: about `0.21 ms`
  - `B=2, S=256, H=4`: about `0.25-0.26 ms`

## Validation

```bash
./docker_cutile.sh install-nvtriton-tileiras
./docker_cutile.sh bootstrap-cutile
./docker_cutile.sh python -m hstu_blackwell_sm120_cudatile_bf16.test_d256
```
