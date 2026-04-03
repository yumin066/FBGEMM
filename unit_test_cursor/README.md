# TMA K Copy Unit Test

`tma_copy_k_sm120_ws_unit_test.cu` extracts the K-path TMA copy flow used in
`hstu_fwd_kernel_fp8_ws.h`:

- build TMA descriptor on host with `make_tma_copy`
- in kernel issue `cute::copy(tma.with(mbar), ...)` from GMEM to SMEM
- wait with `mbarrier.test_wait.parity`
- write SMEM tile to GMEM and byte-compare with source

## Build

Run from repository root:

```bash
nvcc -std=c++17 -O2 \
  -I./external/cutlass/include \
  -I./fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120 \
  unit_test/tma_copy_k_sm120_ws_unit_test.cu \
  -o unit_test/tma_copy_k_sm120_ws_unit_test \
  -arch=sm_120
```

Build with prefetch enabled (optional debug path):

```bash
nvcc -std=c++17 -O2 \
  -DUNIT_TEST_ENABLE_TMA_PREFETCH=1 \
  -I./external/cutlass/include \
  -I./fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120 \
  unit_test/tma_copy_k_sm120_ws_unit_test.cu \
  -o unit_test/tma_copy_k_sm120_ws_unit_test_prefetch \
  -arch=sm_120
```

## Run

```bash
./unit_test/tma_copy_k_sm120_ws_unit_test
```

Expected output:

- `PASS` if TMA copy executes and copied bytes match.
- `FAIL` if copied bytes mismatch.
- On hardware/driver without executable TMA support, runtime may report `cudaErrorIllegalInstruction`.
- If prefetch mode is enabled and CuTe runtime gates are not satisfied, runtime can assert at
  `prefetch_tma_descriptor`.

---

# TMA B Copy Unit Test (Blockscaled GEMM)

`tma_copy_b_sm120_blockscaled_unit_test.cu` extracts the B-operand GMEM->SMEM
copy path from `6KD_fp8_block_scale/.../sm120_blockscaled_gemm_impl.cuh`:

- `tma_load_b.get_tma_tensor(make_shape(N, K, L))`
- `local_tile(... Step<X,_1,_1>)` for B tile addressing
- `partition_S / partition_D`
- `tma_load_b.with(*recast_ptr<ProducerBarrierType>(&barrier))`
- `cute::copy(...)` + `arrive_and_expect_tx(...)`

## Build

```bash
nvcc -std=c++17 -O2 \
  -I./external/cutlass/include \
  unit_test/tma_copy_b_sm120_blockscaled_unit_test.cu \
  -o unit_test/tma_copy_b_sm120_blockscaled_unit_test \
  -arch=sm_120
```

## Run

```bash
./unit_test/tma_copy_b_sm120_blockscaled_unit_test
```

Expected output:

- `PASS` if B tile bytes copied from GMEM to SMEM match reference.
- `FAIL` if byte mismatch is detected.
- On hardware/driver without executable TMA support, runtime may report `cudaErrorIllegalInstruction`.
