## Phase 6 进行中：Warp-Specialized Kernel（2026-03-30，srun 到期前保存）

### 当前状态：已完成所有代码修改，尚未成功编译验证

### 核心修复（已写入 hstu_fwd_kernel.h）

#### Bug：V^T TMA descriptor 初始化失败（"Failed to initialize the TMA descriptor 1"）
- **根因**：V^T GMEM tensor 描述为 `(d, total_k, h_k):(_1{}, v_row_stride, v_head_stride)`，`_1{}` 在 position 0
  - CuTe `make_tma_copy` 的 dim-reorder 路径**不会**被触发（只有 `_1{}` 不在 position 0 时才触发）
  - 导致 `globalStrides[1] = 1 byte`，而 `cuTensorMapEncodeTiled` 要求所有 globalStrides 必须是 16 字节的倍数
- **修复**：改用与 K 相同的维度顺序 `(total_k, d, h_k):(v_row_stride, _1{}, v_head_stride)`
  - `_1{}` 在 position 1 → 触发 CuTe dim-reorder → CuTe 内部重排为 `(d, total_k, h_k)` → 正确计算 globalStrides
  - TMA box `(kBlockN, kHeadDim)` 在 CuTe 重排后对应 SMEM layout `(kHeadDim, kBlockN)` → V^T[d,n]=V[n,d] 正确

#### 四处修改（均已应用到 hstu_fwd_kernel.h）

1. **`issue_tma_kv` device lambda（~line 978）**：
   ```cpp
   // 改为 (total_k, d, h_k) 顺序
   auto mVt_tma = params.tma_vt.get_tma_tensor(
       make_shape(params.total_k, params.d, params.h_k));
   auto gVt_tiles = local_tile(gVt_head,
       Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _));
   auto tVtgVt = tma_slice_Vt.partition_S(gVt_tiles(_, _, _, Int<0>{}));
   ```

2. **`run_hstu_fwd_sm120_fp8_tma_impl` host 函数（~line 2225）**：
   - 删除 debug printf
   - tensor_Vt_full 改为 `(total_k, d, h_k):(v_row_stride, _1{}, v_head_stride)`
   - TMA tile 改为 `(kBlockN, kHeadDim)`

3. **`run_hstu_fwd_sm120_fp8_ws_impl` host 函数（~line 2309）**：同上

4. **`run_hstu_fwd_sm120` dispatch（~line 2370）**：恢复正确 dispatch：
   ```cpp
   if constexpr (!Has_rab) {
     run_hstu_fwd_sm120_fp8_ws_impl<...>(params, stream);
   } else {
     run_hstu_fwd_sm120_fp8_tma_impl<...>(params, stream);
   }
   ```

### 还未完成
- [ ] 编译验证（srun 到期前未完成）
- [ ] sweep_accuracy.py 验证
- [ ] benchmark 性能对比

### 恢复工作指令
```bash
# 1. 重新申请 srun
# 2. 进入工作目录
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu

# 3. 编译
mkdir -p /tmp/claude && \
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user \
HSTU_ARCH_LIST="12.0" \
HSTU_DISABLE_BACKWARD=TRUE \
HSTU_DISABLE_DETERMINISTIC=FALSE \
HSTU_DISABLE_HDIM32=TRUE \
HSTU_DISABLE_HDIM64=TRUE \
HSTU_DISABLE_HDIM256=TRUE \
MAX_JOBS=32 \
pip install --no-build-isolation --config-settings editable_mode=compat -e . 2>&1 | tee /tmp/claude/compile_ws.log
grep -E "error:|static_assert|Successfully" /tmp/claude/compile_ws.log | head -20

# 4. 准确度测试
HSTU_SWEEP_FP8_QUANT_MODE=2 python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/sweep_accuracy.py \
  2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/test_results/042_phase6_ws_accuracy.log

# 5. benchmark
python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu/benchmark/bench_hstu_attn_sm120.py \
  2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/test_results/043_phase6_ws_bench.log
```

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

