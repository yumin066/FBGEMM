# Phase 16 实现计划：Q+SFA Preload 隐藏 mbarrier Wait Latency

> 最后更新：2026-04-23

---

## 背景与问题

### NCU Profile 050（Phase 15 基线）瓶颈分析

```
SM Busy:          61.49%
No Eligible:      59.85%   ← 主要瓶颈
Math Throttle:    13.4%
Long Scoreboard:   3.1%
Registers/Thread: 168（static max = 224）
SMEM:             85 KB / CTA
Occupancy:        25%（Block Limit SMEM = 1，binding）
```

**No Eligible 59.85% 的根本原因**：

math warp 主循环每次迭代开头必须等待 K/V TMA 数据落入 SMEM（`mbarrier.test_wait` 自旋）。在此等待期间，所有 12 个 math warp 同时阻塞，调度器无任何可发射指令 → "No Eligible" 计数飙升。

典型 K TMA 延迟估算（seq=4096，kBlockN=128）：
- K tile 大小：128×128 bytes = 16 KB FP8
- SM120 TMA 带宽 ≈ ~1 TB/s → 每 K-tile 约 16 ns ≈ 40 cycles（锁频 2407 MHz）
- 实际 mbarrier wait 含调度 overhead，约 100-200 cycles/iteration

### SMEM Bank Conflict 分析（次要）

NCU 报告 epilogue partition_C scatter-write 7.9-way conflicts，NCU 估计 42.92% 加速。
**实际贡献重新计算**：
- 超额 wavefront: ~1.835M
- 总 SM cycles: ~109M（由 FP8 kernel duration 估算）
- 实际占比: 1.835M / 109M ≈ **0.44%**

→ NCU 高估了 epilogue 影响（误用了峰值 cycle 而非实际占比）。**本 Phase 不处理 bank conflict**。

---

## 优化策略

### 核心思路：LDSM 延迟隐藏

将主循环中与 TMA 无关的操作提前到 mbarrier wait 前发射：

| 操作 | TMA 依赖？ | 可提前？ | 说明 |
|------|-----------|---------|------|
| 指针算术（sfb_cur, sK_cur, sVt_cur）| 否 | ✅ | 纯整数加减 |
| SFA load（smem_sfa_ptr[sfa_row]）| 否 | ✅ | 主循环前已就绪 |
| Q LDSM（sQ_persist → tCrQ）| 否 | ✅ | sQ_persist 主循环前已就绪且只读 |
| SFB load（smem_sfb_cur[...]）| **是** | ❌ | 来自 TMA 写入的 SMEM 区域 |
| K LDSM（sK_cur → tCrK）| **是** | ❌ | TMA 数据 |
| V LDSM（sVt_cur → tCrV）| **是** | ❌ | TMA 数据 |

**收益估算**：
- Q LDSM 延迟：16 regs × ~5 cycles/reg = ~80 cycles
- K TMA wait：~100-200 cycles
- 这 80 cycles 完全隐藏于 wait 内 → 无额外 No Eligible stall
- 理论减少 No Eligible：~80 cycles / 每迭代 → 在 total cycle 占比中减少 ~5-8%

### 寄存器压力

预加载 Q+SFA 在 wait 前需额外持有 17 regs（Q=16, SFA=1）：

```
Phase 15 baseline peak:  168 regs（主循环内）
Phase 16 wait 前峰值:    168 + 17 = 185 regs
静态分配上限（setmaxnreg）: 224 regs
余量:                    39 regs
```

**无 spill 风险**。

---

## 代码变更详情

**文件**：`fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/hstu_fwd_kernel_fp8_ws.h`

**主循环 mbarrier wait 区域重构**：

```cpp
// ============================================================
// Phase 16：将 stage-invariant 指针计算 + SFA + Q 提前到 wait 前
// ============================================================

// [提前] 指针计算（纯算术，无 TMA 依赖）
int32_t* smem_sfb_cur = smem_sfa_ptr + 2 * kBlockM + math_stage * kBlockN;
FP8Elem* sK_cur  = reinterpret_cast<FP8Elem*>(smem_q)
                   + math_stage * 2 * kSmemKVElems;
FP8Elem* sVt_cur = reinterpret_cast<FP8Elem*>(smem_q)
                   + kSmemKVElems + math_stage * 2 * kSmemKVElems;

// [提前] SFA 加载（smem_sfa_ptr 主循环前已就绪，只读）
Tensor tCrSFA = BS1::partition_fragment_SFA(sSFA(_,_,_0{}), thr_mma_g1);
tCrSFA(0, 0, 0) = smem_sfa_ptr[sfa_row];
auto tCrSFA_frg = BS1::transform_fragment_for_qmma(tCrSFA);

// [提前] Q 预加载（sQ_persist 主循环前已就绪，只读）
// LDSM 发射后，其 ~80-cycle 延迟隐藏于下方的 mbarrier wait 中
Tensor tCrQ = thr_mma_g1.partition_fragment_A(sQ_persist_pi);
load_a_z_pattern(sQ_persist_pi, tCrQ, 0, kHeadDim / 32);

// Wait for K/V/SFB/SFV TMA to land（Q LDSM latency 在此期间被隐藏）
{ /* mbarrier.test_wait spin loop */ }
if (math_stage) { tma_parity1 ^= 1; } else { tma_parity0 ^= 1; }
asm volatile("" ::: "memory");

// 以下操作依赖 TMA 数据，必须在 wait 后
Tensor sSFB_ = make_tensor(make_smem_ptr(smem_sfb_cur), SmemLayoutSFB{});
auto   sSFB  = as_position_independent_swizzle_tensor(sSFB_);
Tensor tCrSFB = BS1::partition_fragment_SFB(sSFB, thr_mma_g1);
{ /* SFB load 循环 */ }
auto tCrSFB_frg = BS1::transform_fragment_for_qmma(tCrSFB);

Tensor acc_s = partition_fragment_C(tiled_mma_g1, ...);
clear(acc_s);

auto sK_cur_pi = as_position_independent_swizzle_tensor(...);
Tensor tCrK = thr_mma_g1.partition_fragment_B(sK_cur_pi);
load_b_z_pattern(sK_cur_pi, tCrK, 0, kHeadDim / 32);

cute::gemm(tiled_mma_g1, zip(tCrQ, tCrSFA_frg), zip(tCrK, tCrSFB_frg), acc_s);
```

**不变量确认**：
1. `sQ_persist` 在 Phase 15 的 preamble TMA（math warp thread 0 发出）完成后保持只读 → 主循环内提前读取安全
2. `smem_sfa_ptr` 指向 SMEM preamble 区，主循环内不被覆写 → 提前读取安全
3. 指针算术中的 `math_stage`（0/1）在本次迭代开头已确定 → 提前计算安全

---

## 实现状态

### 已完成

- [x] **代码修改**：主循环 mbarrier wait 前插入 Q+SFA+指针计算（见代码变更详情）
- [x] **编译通过**：`HSTU_ARCH_LIST="12.0" pip install --no-build-isolation ...`（无 error/warning）

### 待完成

- [x] **数值验证**：`sweep_accuracy.py` → 全部 fp8_gt_cos ≥ 0.9996（`1test_results/086_phase16_q_preload.log`）
- [x] **example 验证**：`run_hstu8_examples.sh` → 14/14 PASS
- [x] **benchmark（未锁频）**：`2benchmark_results/051_phase16_q_preload.log`；FP8/BF16 比值 causal 配置 +2-5pp，无退化
- [x] **benchmark（锁频 2407MHz）**：`2benchmark_results/052_phase16_q_preload_lgc2407.log`；causal +1~3pp，full 持平
- [ ] **profile**：ncu 确认 No Eligible stall 下降（可选，已验证正确性）

---

## 验证命令

### 数值验证

```bash
# 命令1
mkdir -p /tmp/claude

# 命令2
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu

# 命令3（编译，如需重编）
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user HSTU_ARCH_LIST="12.0" HSTU_DISABLE_BACKWARD=TRUE HSTU_DISABLE_DETERMINISTIC=FALSE HSTU_DISABLE_HDIM32=TRUE HSTU_DISABLE_HDIM64=TRUE HSTU_DISABLE_HDIM256=TRUE MAX_JOBS=32 pip install --no-build-isolation --config-settings editable_mode=compat -e . 2>&1 | grep -E "error:|note:|static_assert" | head -60
```

```bash
HSTU_SWEEP_FP8_QUANT_MODE=2 python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/sweep_accuracy.py 2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/1test_results/086_phase16_q_preload.log
```

**通过条件**：所有配置 `fp8_gt_cos ≥ 0.995`

### Example 验证

```bash
bash /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/run_hstu8_examples.sh
```

**通过条件**：`14/14 passed`

### Benchmark

```bash
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu/benchmark/bench_hstu_attn_sm120.py 2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/2benchmark_results/051_phase16_q_preload.log
```

### SASS 验证（确认无新增 spill）

```bash
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu
./dump_sass_fp8_ws.sh
grep -c "LDL\|STL" 4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full.sass
```

**目标**：LDL+STL 计数维持 0。

### NCU Profile

```bash
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu
./run_profile.sh 051 phase16_q_preload
```

**关注指标**：No Eligible stall（目标 < 45%）、SM Busy、Math Throttle。

---

## 预期结果

| 指标 | Phase 15 基线（050） | Phase 16 目标 |
|------|-------------------|-------------|
| No Eligible stall | 59.85% | < 45% |
| SM Busy | 61.49% | > 65% |
| FP8 TFLOPS（seq=4096 causal） | ~1060 | +3-8% |
| FP8 TFLOPS（seq=4096 full） | ~620 | +3-8% |
| LDL/STL（SASS） | 0 | 0（不变） |
| fp8_gt_cos | ≥ 0.9996 | ≥ 0.9996 |

---

## 风险与决策树

```
数值验证通过？
  ├─ 否 → debug（Q 预加载是否读到正确数据？sQ_persist 地址是否正确？）
  │        → 检查 sQ_persist_pi 在提前到 wait 前后是否变化
  └─ 是 →
       benchmark 提升？
         ├─ 是（+3%以上）→ 完成 Phase 16，进入 Phase 17
         └─ 否（<1%）→
              NCU 确认 No Eligible 是否下降：
                ├─ 下降但 TFLOPS 无变化 → 瓶颈已转移，分析新主导 stall
                └─ 无下降 → ptxas 可能已做了同样优化，考虑 pipeline overlap
```

---

## 关键文件

| 文件 | 修改位置 | 变更内容 |
|------|---------|---------|
| `hstu_fwd_kernel_fp8_ws.h` | 主循环 mbarrier wait 前（约 856-940 行） | 指针计算 + SFA + Q 提前到 wait 前 |
