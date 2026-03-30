## Phase 5 COMPLETE: TMA K+V^T (no kernel-side transpose) (2026-03-30)

### 实现内容
- **核心变更**: FP8 路径从 cp.async K+V 升级为 TMA K+V^T
  - V 以转置视图 [d, total_v, h_k] strides [1, v_row_stride, v_head_stride] 描述给 TMA
  - TMA 直接将 [kHeadDim, kBlockN] tile 加载到 SmemLayoutVt_TMA（SW128 SMEM）
  - 消除了 ~64 SMEM ops/thread 的 element-wise V 转置循环
- **关键技术**:
  - `uint64_t*` 原始 mbarrier（CuTe `with()` 需要 `uint64_t&`，不是 `ClusterTransactionBarrier`）
  - PTX inline: `mbarrier.init`, `mbarrier.arrive.expect_tx`, `mbarrier.test_wait`
  - `local_tile` with `make_coord(_, _)`（全 tile）+ 在 `copy` 时按 `nb_abs` 索引
- **新模板参数**: `bool Use_TMA_KV = false`，FP8 路径默认为 true
- **新内核**: `hstu_fwd_kernel_sm120_fp8_tma<Kernel_traits, TMA_K_t, TMA_Vt_t>`
- **新参数结构**: `Hstu_fwd_params_fp8_tma<TMA_K_t, TMA_Vt_t>`

### 测试结果
- sweep_accuracy.py: fp8_gt_cos ≈ 0.9996，全部通过
- 日志: `test_results/039_phase5_tma_kv_vt_notranspose.log`

### 性能结果 (2026-03-30, RTX PRO 6000 Blackwell SM120, bs=4, h=16, d=128)
| SEQ  | BF16(ms) | FP8(ms) | BF16 TFLOPS | FP8 TFLOPS | FP8/BF16 ratio |
|------|---------|---------|------------|-----------|--------------|
| 512  | 0.044   | 0.053   | 98.6        | 80.3       | 1.23x slower |
| 1024 | 0.119   | 0.144   | 144.6       | 119.5      | 1.21x slower |
| 2048 | 0.447   | 0.544   | 153.7       | 126.2      | 1.22x slower |
| 4096 | 1.551   | 1.912   | 177.2       | 143.8      | 1.23x slower |
- **Phase 5 vs Phase 4**: FP8 kernel 快 ~6-7%（TMA 消除 V 转置开销）
- 日志: `test_results/041_phase5_tma_kv_bench.log`

---

## Phase 4 COMPLETE: K+V double-prefetch pipeline (2026-03-27)

### 实现内容
- **核心变更**: 将 V[nb] 和 K[nb] 在上一轮迭代末尾一起预取，消除迭代中途的 blocking `cp_async_wait` for V
- **文件**: `hstu_blackwell_sm120/hstu_fwd_kernel.h`
  - preamble: 同时发出 K[n0] 和 V[n0]
  - fwd_step_fp8bs: 删除中途 V issue+wait，增加 `__syncthreads()` 保护 s2r P 和 Vt 写入之间的冲突
  - 迭代末尾: 同时发出 K[nb_next] 和 V[nb_next]
- **library.py 修复**: `_should_load_hstu_lib()` 函数，条件 `cap < (10,0) or cap[0] >= 12`（之前 `< (10,0)` 阻止了 SM120 加载库）

### 测试结果
- sweep_accuracy.py: fp8_gt_cos ≈ 0.9996，全部通过（Phase 3 同等准确度）
- hstu_test.py: Exit 0，max diff 0.00098

### 性能 Benchmark 结果 (2026-03-27, RTX PRO 6000 Blackwell SM120)
- **Benchmark 脚本**: `fbgemm_gpu/experimental/hstu/benchmark/bench_hstu_attn_sm120.py`（新建）
- **精度**: FP8 quant_mode=2 vs BF16, cos_sim ≈ 0.9986，全部 PASS
- **kernel-only 性能** (bs=4, h=16, d=128, full attention):
  - seq=512:  BF16 0.048ms 181 TFLOPS  →  FP8 0.057ms 151 TFLOPS  (-16%)
  - seq=1024: BF16 0.126ms 273 TFLOPS  →  FP8 0.154ms 224 TFLOPS  (-18%)
  - seq=2048: BF16 0.479ms 287 TFLOPS  →  FP8 0.582ms 236 TFLOPS  (-18%)
- **Phase 4 vs Phase 3 改善**: Phase 3 有 mid-iteration blocking V wait（GMEM→SMEM full roundtrip delay），Phase 4 K+V 预取消除此 stall，FP8 kernel 延迟约减少 1 个 V DMA 的 GMEM 延迟
- **FP8 仍比 BF16 慢原因**: SM120 QMMA block-scale 指令有 scale factor 加载开销；V^T transpose 步骤增加额外工作；kernel 整体 compute-bound
- **End-to-end**: FP8 比 BF16 慢 5-100x（Python 量化开销 ~0.8-1.9ms）

### 日志
- `test_results/036_phase4_library_fix.log`
- `test_results/037_hstu_test_phase4_final.log`
- `test_results/038_sm120_bench_phase4.log`

