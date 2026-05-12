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

`nvcr.io/nvidia/pytorch:26.04-py3` provides PyTorch 2.12/CUDA 13.2 and
`torch.float8_e8m0fnu`. The cuTile C++ extension builds in that image with
development features enabled, but the image and current pip package only provide
`tileiras 13.2.78`. `ct.mma_scaled` requires TileIR bytecode 13.3, so this
directory keeps runtime checks explicit and reports that blocker before launch.

`Dockerfile_cutile_cuda133` and `docker_cutile_cuda133.sh` are the dedicated
CUDA 13.3 environment path. The Dockerfile tries, in order:

- NVIDIA CUDA apt packages such as `cuda-tileiras-13-3`;
- pip packages such as `nvidia-cuda-tileiras>=13.3,<13.4`;
- optional private package URLs passed via `CUDA133_TILEIRAS_DEB_URL` and
  `CUDA133_COMPILER_DEB_URL`.

As of 2026-05-10 in this environment, public apt, PyPI, NGC PyTorch
`26.04-py3`, and the existing private HSTU base do not expose a tileiras that
accepts TileIR bytecode 13.3. The strict image build therefore fails with
`tileiras_bytecode=13.2`, which is the correct behavior. Build with
`CUDA133_STRICT=0` only for diagnostics; it will still make the D256 test skip
before kernel launch.

Additional checks:

- `pip install -U "cuda-tile[tileiras]"` installs `cuda-tile 1.3.0`, but its
  dependency range is `nvidia-cuda-tileiras<13.3,>=13.2`, so the installed
  compiler remains `13.2.78`. That wheel does not expose public
  `ct.mma_scaled`.
- `/home/scratch.minyu_gpu/project/shopee/cutile-python` is already at upstream
  `main` (`8e25f53c031d8b9fcbfecdb03fefaf9790204aec`). Building it with
  `--enable-dev-features` exposes the development `mma_scaled` Python surface
  and `_cext`, but kernel compilation still depends on the external tileiras
  binary, which currently probes as bytecode 13.2.

## Milestones

1. D256 dense full no-RAB correctness.
2. D256 dense causal no-RAB correctness.
3. Static grid-stride persistent full/causal.
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

Container bootstrap:

```bash
CUTILE_DOCKER_PULL=0 ./docker_cutile.sh bootstrap-cutile
```

CUDA 13.3 strict image build:

```bash
./docker_cutile_cuda133.sh build
```

Diagnostic non-strict image build:

```bash
CUDA133_STRICT=0 ./docker_cutile_cuda133.sh build
```

When a private 13.3 package is available:

```bash
CUDA133_TILEIRAS_DEB_URL=<tileiras-13.3.deb> \
CUDA133_COMPILER_DEB_URL=<compiler-13.3.deb> \
./docker_cutile_cuda133.sh build
```

Then:

```bash
./docker_cutile_cuda133.sh bootstrap-cutile
./docker_cutile_cuda133.sh test
```

Correctness:

```bash
CUTILE_DOCKER_PULL=0 ./docker_cutile.sh \
  python -m hstu_blackwell_sm120_cudatile_fp8_ws.test_d256
```

On the 26.04 image this currently exits successfully with:

```text
SKIP: ct.mma_scaled requires tileiras 13.3 or later, found 13.2
```

Benchmark:

```bash
sudo nvidia-smi -lgc 2407
CUTILE_DOCKER_PULL=0 ./docker_cutile.sh \
  python -m hstu_blackwell_sm120_cudatile_fp8_ws.bench_d256
sudo nvidia-smi -rgc
```
