# Phase 13 调试计划：AtomLayout `<_8,_1,_1>` 精度修复

> 最后更新：2026-04-20（换机器前存档）

---

## 当前状态

**Branch**: sm120  
**cos_sim**: ~0.97（目标 ≥ 0.995）  
**编译状态**: 通过  
**上一次有效 log**: `1test_results/091_phase13_revert_load_patterns.log`

### 本次会话做了什么

1. 实施并验证了 Codex 建议的 `load_a_z_pattern` / `load_b_z_pattern` 字节级 gather "修复"
2. 结果 cos_sim 从 0.97 跌至 0.054 → 证明 Codex 分析错误
3. 回滚到 K-major 格式（uint32 连续 K 字节加载），cos_sim 恢复 0.97
4. 重编译成功，`091_...log` 确认状态

---

## 已确认正确的部分

| 组件 | 状态 | 根据 |
|------|------|------|
| `load_a_z_pattern` K-major 公式 | ✅ | cos_sim=0.97 vs 0.054（M-major/字节 gather） |
| `load_b_z_pattern` K-major + PermMmaTileN 顺序 | ✅ | N_base=(nr%4)*32+(nr/4)*8 正确 |
| SFB N_base 公式 | ✅ | 与 load_b_z_pattern 对称，两者一致 |
| P staging 写入（thr_mma_g1.partition_C 直接映射） | ✅ | [DBG][SPBUF]=[DBG][TCRP] 一致 |
| PermMmaTileN 推导 | ✅ | 数学验证：nr=0→N=0, nr=1→N=32, nr=4→N=8... |

---

## 最可能的 bug（待验证）

### Bug 1：SFV 缺少 `n_row_sfv` 偏移（高置信度）

**位置**：`hstu_fwd_kernel_fp8_ws.h` 约 1127-1130 行

**当前代码（可能错误）**：
```cpp
for (int nr = 0; nr < kNAtomsSFV; ++nr) {
  const int N_base = (nr % 4) * 32 + (nr / 4) * 8;
  tCrSFV(0, nr, 0) = smem_sfv_ptr[math_stage][N_base];   // ← 缺少 n_row 偏移！
}
```

**对比 SFB（正确，有 n_row_sfb）**：
```cpp
const int n_row_sfb = (tidx_math & 31) >> 2;
tCrSFV(0, nr, 0) = smem_sfb_ptr[math_stage][N_base + n_row_sfb];
```

**分析**：
- GEMM2 中 B 操作数 = V^T，N-轴 = head-dim（d 方向，0..127）
- SFBLayout: `T_contrib = 0*(t%4) + 1*(t/4) = t/4 = lane/4`
- 每个 N-atom 内 8 个线程持有不同 d-位置，各需不同 SFV
- 应加 `n_row_sfv = lane >> 2`（与 SFB 相同公式）

**修复**：
```cpp
const int n_row_sfv = (tidx_math & 31) >> 2;
for (int nr = 0; nr < kNAtomsSFV; ++nr) {
  const int N_base = (nr % 4) * 32 + (nr / 4) * 8;
  tCrSFV(0, nr, 0) = smem_sfv_ptr[math_stage][N_base + n_row_sfv];
}
```

**为什么当前 cos_sim 是 0.97 而不是 0.054**：  
测试数据中 SFV 值基本均匀（诊断输出 `sfv[0][0..3]=0x79797979`），偏差小时错误小。  
非均匀 scale 会放大误差。

### Bug 2：SFA 公式可能错误（中置信度，待确认）

**位置**：`hstu_fwd_kernel_fp8_ws.h` 约 853-856 行

**当前代码**：
```cpp
const int warp_m  = tidx_math / 32;
const int t0      = tidx_math % 4;
const int sfa_row = warp_m * 16 + t0 * 2;
tCrSFA(0, 0, 0)  = smem_sfa_ptr[sfa_row];
```

**SFALayout 要求**：
```
SFALayout: T_contrib = 8*(t%2) + t/4
→ SFA M-position = 8*(lane%2) + (lane/4)
→ 正确公式应为: sfa_row = warp_m*16 + 8*(lane&1) + (lane>>2)
```

**当前公式错误情况**：
- lane=0: 当前=0, 正确=0 ✓
- lane=1: 当前=2, 正确=8 ✗
- lane=2: 当前=4, 正确=0 ✗
- lane=4: 当前=0, 正确=1 ✗

**注意**：当前测试数据所有 SFA 均为 0x78（均匀），无法从 cos_sim 验证此 bug。  
可能是次要误差源，或在 SFV bug 修复后通过 all-ones 测试进一步验证。

---

## 下一步执行顺序

### Step A：修复 SFV n_row_sfv（先做，影响更大）

修改 `hstu_fwd_kernel_fp8_ws.h` 约 1127-1130：
```cpp
const int n_row_sfv = (tidx_math & 31) >> 2;
for (int nr = 0; nr < kNAtomsSFV; ++nr) {
  const int N_base = (nr % 4) * 32 + (nr / 4) * 8;
  tCrSFV(0, nr, 0) = smem_sfv_ptr[math_stage][N_base + n_row_sfv];
}
```

编译 → `sweep_accuracy.py` → 看 cos_sim 是否提升。

### Step B：若 Step A 后 cos_sim 仍 < 0.995，修复 SFA 公式

修改 `hstu_fwd_kernel_fp8_ws.h` 约 853-856：
```cpp
const int sfa_row = warp_m * 16 + 8*(lane & 1) + (lane >> 2);
tCrSFA(0, 0, 0)  = smem_sfa_ptr[sfa_row];
```

### Step C：若上述修复后 cos_sim ≥ 0.995，运行完整验证

```bash
bash /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/run_hstu8_examples.sh
```

---

## 关键文件位置

- 主 kernel：`fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/hstu_fwd_kernel_fp8_ws.h`
- QMMA builder：`fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/sm120_qmma_builder.h`
- SFV bug 行号：约 1121-1131（`tCrSFV` 加载循环）
- SFA bug 行号：约 853-856（`tCrSFA` 直接加载）

## 编译命令（三条独立 Bash 调用）

```bash
# 1
mkdir -p /tmp/claude
# 2
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu
# 3
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user HSTU_ARCH_LIST="12.0" HSTU_DISABLE_BACKWARD=TRUE HSTU_DISABLE_HDIM32=TRUE HSTU_DISABLE_HDIM64=TRUE HSTU_DISABLE_HDIM256=TRUE MAX_JOBS=32 pip install --no-build-isolation --config-settings editable_mode=compat -e .
```

## 精度验证命令

```bash
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user HSTU_SWEEP_FP8_QUANT_MODE=2 python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/sweep_accuracy.py 2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/1test_results/092_phase13_fix_sfv_nrow.log
```
