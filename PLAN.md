# HSTU SM120 FP8 优化计划

> 最后更新：2026-04-23

---

## Phase 18 COMPLETE：cvt.e4m3x2 双路 FP8 转换 + Epilogue STSM（2026-04-23）

### 18a：Epilogue STSM（先完成，无运行时收益）

将 math warp epilogue 中逐元素 `STS.32` × 32 替换为 `stmatrix.sync.aligned.x4.m8n8.shared.b16`（STSM.16.M88.4）× 8，指令数降低 4×，使用 SM120 原生矩阵存储路径。运行时与 Phase 17 持平（epilogue 不是主瓶颈，No Eligible ~60% 仍由 mbarrier wait 主导）。

### 18b：cvt.e4m3x2 双路 FP8 转换（本 session 完成，+1-4%）

**问题**：`F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C Rx, RZ, Rx, RZ` 每条指令只转一个 FP32，srcB=RZ 浪费一路；后续需要 `IMAD+LOP3+PRMT+IADD` 链完成 4 字节打包。

**方案**：改用 PTX `cvt.rn.satfinite.e4m3x2.f32`，2 条指令转 4 个 FP32→FP8，`mov.b32 {lo, hi}` 完成打包，消除原有复杂打包链。

```cpp
CUTE_UNROLL
for (int flat = 0; flat < kAccSElems; flat += 4) {
  uint32_t out;
  asm volatile(
      "{\n"
      ".reg .b16 lo, hi;\n"
      "cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
      "mov.b32 %0, {lo, hi};\n"
      "}\n"
      : "=r"(out)
      : "f"(float(acc_s(flat+0))), "f"(float(acc_s(flat+1))),
        "f"(float(acc_s(flat+2))), "f"(float(acc_s(flat+3))));
  acc_s_packed[flat / 4] = out;
}
```

**验证**：
- [x] sweep_accuracy.py：全部 fp8_gt_cos ≥ 0.9996（`1test_results/087_phase18_fp8_cvt2.log`）
- [x] run_hstu8_examples.sh：14/14 PASS
- [x] benchmark 054：FP8 全面 +1-4%，bs=8 seq=4096 causal 1113.0 TFLOPS

**性能结果（benchmark 054，RTX PRO 6000 Blackwell SM120）**：

| Config | BF16 TFLOPS | FP8 TFLOPS | FP8 vs BF16 | vs Phase 17 |
|--------|-------------|------------|-------------|-------------|
| bs=8 seq=4096 h=16 causal | 644.2 | **1113.0** | **+72.8%** | **+2.9%** |
| bs=4 seq=4096 h=16 causal | 619.2 | **1059.4** | **+71.1%** | **+2.2%** |
| bs=1 seq=4096 h=16 causal | 492.7 | **872.1** | **+77.0%** | +1.2% |
| bs=8 seq=4096 h=16 full | 362.5 | **630.8** | **+74.0%** | **+2.5%** |
| bs=8 seq=2048 h=16 full | 340.8 | **610.7** | **+79.2%** | **+3.6%** |

---

## Phase 19 COMPLETE：TMA Store Output（2026-04-23）

将 epilogue SMEM→GMEM 拷贝从 256 线程 LDS.128+STG.E.128 改为单线程 TMA async bulk store，255 个 math thread 提前退出。

**核心变更**：
- **host side**：`Hstu_fwd_params_fp8_ws_tma` 加 `TMA_O_t tma_o` 字段；`run_hstu_fwd_sm120_fp8_ws_tma_impl` 创建 `make_tma_copy(SM90_TMA_STORE{}, tensor_O_full, SmemLayoutO_TMA_t{}, ...)` descriptor
- **device side**：S5 bar.sync 后，partial tile 清零 OOB 行（全 256 线程合作 + bar.sync），然后 `if (tidx_math != 0) return;`，thread 0 执行 `fence.proxy.async.shared::cta` + TMA store + `tma_store_arrive` + `tma_store_wait<0>()`
- **关键 bug（已修正）**：TMA descriptor 必须用 `(total_q, d, h)` dim ordering（d 有 stride 1 为 innermost），与 Q/K/V 保持一致；用 `(total_q, h, d)` 会导致 boxDim 超过 globalDim h=1，descriptor 初始化失败

**验证**：
- [x] sweep_accuracy.py：全部 fp8_gt_cos ≥ 0.9996（`1test_results/088_phase19_tma_store.log`）
- [x] run_hstu8_examples.sh：14/14 PASS
- [x] benchmark 055：bs=8 seq=4096 causal **1123.6 TFLOPS**（+0.95% vs Phase 18）

**性能结果（benchmark 055，RTX PRO 6000 Blackwell SM120）**：

| Config | BF16 TFLOPS | FP8 TFLOPS | FP8 vs BF16 | vs Phase 18 |
|--------|-------------|------------|-------------|-------------|
| bs=8 seq=4096 h=16 causal | 644.7 | **1123.6** | **+74.3%** | **+0.95%** |
| bs=4 seq=4096 h=16 causal | 619.9 | **1068.2** | **+72.3%** | **+0.83%** |
| bs=1 seq=4096 h=16 causal | 493.6 | **872.6** | **+76.8%** | +0.06% |
| bs=8 seq=4096 h=16 full | 362.3 | **629.7** | **+73.8%** | -0.2% |
| bs=8 seq=2048 h=16 full | 340.6 | **612.1** | **+79.7%** | +0.2% |

收益主要来自大 batch causal 配置（epilogue 占比较高），full 和小 seq 基本持平。受 Block Limit SMEM=1（1 CTA/SM）限制，提前释放 warp 无法帮助调度其他 CTA。

---

## Phase 20 候选方向

当前主瓶颈：**No Eligible stall ~60%**，根本原因是 math warp 等 K/V TMA（mbarrier wait）期间调度器无可发射指令。单 CTA 内已无进一步可隐藏的空间，下一步需从结构层面突破。

| 方向 | 思路 | 预期收益 | 难度 |
|------|------|---------|------|
| **kBlockN 128→256** | 更大 tile，算术密度提升，TMA 占比下降；需重新计算 SMEM（V tile 256×128=32KB FP8，需检查 85KB 预算） | +10-20% | 中 |
| **SMEM 降至 50KB** | 解除 Block Limit SMEM=1（1 CTA/SM），occupancy 翻倍，No Eligible 分摊到更多 warp（SM120 每 SM 100KB，需降至 50KB 才能 2 CTA/SM） | 理论显著，实测依赖具体布局 | 高 |
| **Persistent kernel / 双 CTA wave** | 2 个 CTA 交替发 TMA，互相隐藏 wait | +10-30%（若 2 CTA/SM 可行） | 高 |
| **BF16 kernel 同类优化** | 将 Phase 13-19 的 FP8 WS 优化移植到 BF16 路径 | +5-15% | 中 |

**推荐下一步**：评估 kBlockN=256 的 SMEM 可行性（85KB 预算分析），或先做 NCU profile 056 确认 Phase 19 后的新瓶颈分布。

---

## Phase 17 COMPLETE：同步点精简（2026-04-23）

### 背景

NCU Profile 052 显示 No Eligible stall 59.76%，mbarrier wait 仍是主瓶颈。分析 CTA 内全部同步点：

| 同步点 | 原实现 | 问题 | 新实现 |
|--------|--------|------|--------|
| S3 | `__syncthreads()` | 纯集合点，无数据依赖 | 删除 |
| S2 | `__syncthreads()` | Q TMA wait 属 math warp 内部 | `bar.sync 2,256` + `wait_mbar_parity` |
| q_tma_mbar init | load warp thread 256 | 多余 CTA sync | 迁移至 math warp thread 0 |
| S5 | `__syncthreads()` | epilogue math-warp-only，load warp 白白锁死 | `bar.sync 1,256`，load warp 提前退出 |

**关键 bug**：`bar.sync 2,256` 必须在 `mbarrier.init` 与 `wait_mbar_parity` 之间，否则 race condition 导致 flaky crash。

### 验证

- [x] sweep_accuracy.py：全部 fp8_gt_cos ≥ 0.9996
- [x] run_hstu8_examples.sh：14/14 PASS（连续 5 次）
- [x] benchmark 053：bs=1 causal +6.7%，bs=4/8 causal +0.4-0.8%，full +0.6-1.8%

### 性能结果（benchmark 053）

| Config | BF16 TFLOPS | FP8 TFLOPS | FP8 vs BF16 | vs Phase 16 |
|--------|-------------|------------|-------------|-------------|
| bs=8 seq=4096 h=16 causal | 644.3 | **1081.8** | **+67.9%** | +0.4% |
| bs=4 seq=4096 h=16 causal | 619.8 | **1036.6** | **+67.2%** | +0.8% |
| bs=1 seq=4096 h=16 causal | 476.4 | **861.9** | **+80.9%** | **+6.7%** |
| bs=8 seq=4096 h=16 full | 362.3 | **615.4** | **+69.8%** | +0.6% |
| bs=8 seq=2048 h=16 full | 340.1 | **589.5** | **+73.3%** | +1.8% |

---

## 关键文件

| 文件 | 作用 |
|------|------|
| `hstu_fwd_kernel_fp8_ws.h` | FP8 WS 内核主体（全部 Phase 6+ 修改） |
| `sm120_qmma_builder.h` | SM120 QMMA builder（Phase 12+） |
| `kernel_traits.h` | BF16 + FP8 内核 traits |
| `2benchmark_results/054_phase18_fp8_cvt2.log` | 当前最新 benchmark |
| `1test_results/087_phase18_fp8_cvt2.log` | 当前最新准确性日志 |
