## Phase 6 结论：WS cp.async 不可行（2026-03-31 更新）

### 当前代码状态（git branch: sm120）

**当前 dispatch（hstu_fwd_kernel.h ~line 1669）**：所有 FP8 路径使用 Phase 4 cp.async。
**WS 代码**：`hstu_fwd_kernel_fp8_ws.h` 已改写为 cp.async 版本（TMA 已移除），但 dispatch 已回退到 Phase 4。

**Phase 4 dispatch 通过验证**：`sweep_accuracy.py` fp8_gt_cos ≈ 0.9996，全部 6 个测试用例通过（测试 051, 053）。

---

### Phase 6 WS cp.async 实验（测试 052）

#### 实现
将 WS kernel 中 TMA 替换为 cp.async：
- Load warp（32 threads）使用 `GmemLayoutAtom_LW = Layout<Shape<_4,_8>,Stride<_8,_1>>` 发送 K+V cp.async
- `cp_async_wait<0>` 后 `fence.proxy.async` 后 `mbarrier.arrive(load_mbar)`
- Math warps 等待 `load_mbar` 后执行 GEMM1 + silu + GEMM2
- 在 GEMM2 前加入 V→Vt 元素级转置（Phase 4 同样的模式）
- **准确度**：通过（fp8_gt_cos ≈ 0.9996）

#### 性能结果（对比 Phase 4 benchmark_results/038）
| seq  | Phase 4 FP8 | WS FP8 | 倍数 |
|------|------------|--------|------|
| 512  | ~0.057ms   | 0.182ms | 3.2x 慢 |
| 1024 | ~0.154ms   | 0.343ms | 2.2x 慢 |
| 2048 | ~0.582ms   | 1.965ms | 3.4x 慢 |

#### 根本原因：WS + cp.async 没有 overlap

1. **32-thread load warp 效率低**：发出 64 条 cp.async（K+V 各 32 步），而 Phase 4 的 256 线程只需 8 步
2. **`cp_async_wait<0>` 阻塞 load warp**：必须等 copy 完成才能 `mbarrier.arrive`，math warps 等待整个 copy 时间，无计算-内存 overlap
3. **额外 bar.sync 开销**：每迭代多 2 次 `bar.sync 1, 256`（Vt transpose 前后）
4. **TMA 才是 WS 的正确搭配**：TMA 可以非阻塞通知（`arrive.expect_tx`），cp.async 不行

#### 结论
Phase 6 WS cp.async 比 Phase 4 慢 2-4x，不可行。WS kernel 适合 TMA（异步信号），不适合 cp.async（同步等待）。

**下一步方向（如果仍要进行）**：
- 方案A：等待有 TMA 硬件支持的 GPU 测试（原 Phase 6 WS + TMA 方案）
- 方案B：在当前 Phase 4 基础上做寄存器/指令级优化，减少 register spill 和 scale factor 加载开销
- 方案C：将 FP8 量化融合进 kernel（消除 Python 侧量化 overhead，当前 end-to-end FP8 比 BF16 慢 100x）

---

### TMA 不可用原因（2026-03-31 确认）

- 硬件：RTX PRO 6000 Blackwell **Server Edition** (SM120, 不是 SM120a)
- `CLUSTER_LAUNCH_SUPPORTED=0`, `CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED=1`（只是说支持属性查询，不代表 cp.async.bulk 可执行）
- `cp.async.bulk.tensor.3d.shared::cluster`（Hopper 路径）→ "Illegal instruction" (error 715)
- `cp.async.bulk.tensor.3d.shared::cta`（SM120 路径，定义了 `CUTE_ARCH_TMA_SM120_ENABLED`）→ 仍然 "Illegal instruction"
- **结论**：SM120 consumer GPU 的 TMA 硬件不工作，无法使用 `cp.async.bulk.tensor` 指令
- **测试日志**：050 (cta path still illegal), 051 (Phase 4 passes)

---

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
|------|---------|---------|------------|-----------|--------------|\n| 512  | 0.044   | 0.053   | 98.6        | 80.3       | 1.23x slower |
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
