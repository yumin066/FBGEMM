# Phase 11 实现计划：SM100_U8x16_LDSM_T 消除 SMEM Transpose

> 经两轮 Codex 技术审查 + diff 原型确认（2026-04-16）。

## 原则

- Phase 11 代码改动**无论结果如何一律不回退**
- 若数值不正确或性能回退，在 CLAUDE.md 中记录结论，继续推进后续工作

---

## 目标

用 `ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8`（`SM100_U8x16_LDSM_T`）直接从 MN_SW128 SMEM 加载 V^T 并转置到寄存器，消除每个 N-block 迭代中的协作 SMEM transpose。

**当前路径**：
```
sVt_cur (MN_SW128) → bar.sync → transpose loop → bar.sync → sK_cur (K_SW128) → LDSM_N → 寄存器
```

**目标路径**：
```
sVt_cur (MN_SW128) → LDSM_T (.trans) → 寄存器
```

预期消除每 N-block：2×`bar.sync 1,256` + 16384 元素标量 copy。

---

## 修改文件

**仅修改**：`fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/hstu_fwd_kernel_fp8_ws.h`

### 修改 1：新增 GEMM2-V 专用 copy object（第 624 行附近）

在 `s2r_thr_B2` 定义之后紧接着加入：

```cpp
auto s2r_copy_Vt  = make_tiled_copy_B(Copy_Atom<SM100_U8x16_LDSM_T, FP8Elem>{}, tiled_mma_g2);
auto s2r_thr_Vt   = s2r_copy_Vt.get_thread_slice(tidx_math);
```

> 不修改全局 `s2r_copy_B2`（仍保留给 K 路径使用）。

### 修改 2：删除 SmemLayoutVt_K_SW128（第 741-742 行）

```cpp
// 删除：
using SmemLayoutVt_K_SW128 = decltype(tile_to_shape(typename BS2::SmemLayoutAtomB{},
    Shape<Int<kHeadDim>, Int<kBlockN>>{}));
```

### 修改 3：删除 bar.sync + transpose loop + bar.sync（第 907-917 行）

```cpp
// 全部删除：
asm volatile("bar.sync 1, 256;\n" : : : "memory");   // 907行
{                                                      // 909行
  Tensor sVt_mn = make_tensor(make_smem_ptr(sVt_cur), SmemLayoutVt_SW128{});
  Tensor sVt_k  = make_tensor(make_smem_ptr(sK_cur),  SmemLayoutVt_K_SW128{});
  for (int i = tidx_math; i < kHeadDim * kBlockN; i += kNMathThreads) {
    sVt_k(i / kBlockN, i % kBlockN) = sVt_mn(i / kBlockN, i % kBlockN);
  }
}                                                      // 916行
asm volatile("bar.sync 1, 256;\n" : : : "memory");   // 917行
```

> **保留** 876 行（sK→P buffer 重用保护）和 898 行（P 写入可见性）的 bar.sync。

### 修改 4：替换 s2r V^T（第 920-928 行）

```cpp
// 删除旧 s2r（从 K_SW128 的 sK_cur 加载）：
auto sVt_k_pi = as_position_independent_swizzle_tensor(
    make_tensor(make_smem_ptr(sK_cur), SmemLayoutVt_K_SW128{}));
Tensor tCrV = thr_mma_g2.partition_fragment_B(sVt_k_pi);
{
  auto tXsVt = s2r_thr_B2.partition_S(sVt_k_pi);
  auto tXrV  = s2r_thr_B2.retile_D(tCrV);
  cute::copy(s2r_copy_B2, tXsVt, tXrV);
}

// 替换为新 s2r（从 MN_SW128 的 sVt_cur 用 LDSM_T 加载）：
auto sVt_mn_pi = as_position_independent_swizzle_tensor(
    make_tensor(make_smem_ptr(sVt_cur), SmemLayoutVt_SW128{}));
Tensor tCrV = thr_mma_g2.partition_fragment_B(sVt_mn_pi);
{
  auto tXsVt = s2r_thr_Vt.partition_S(sVt_mn_pi);
  auto tXrV  = s2r_thr_Vt.retile_D(tCrV);
  cute::copy(s2r_copy_Vt, tXsVt, tXrV);
}
```

---

## 验证流程

### 编译
```bash
mkdir -p /tmp/claude && \
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu && \
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user \
HSTU_ARCH_LIST="12.0" HSTU_DISABLE_BACKWARD=TRUE \
HSTU_DISABLE_HDIM32=TRUE HSTU_DISABLE_HDIM64=TRUE HSTU_DISABLE_HDIM256=TRUE \
MAX_JOBS=32 pip install --no-build-isolation --config-settings editable_mode=compat -e . \
  2>&1 | grep -E "error:|note:|static_assert|undefined" | head -60
```

### 数值验证
```bash
HSTU_SWEEP_FP8_QUANT_MODE=2 \
python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/sweep_accuracy.py \
  2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/1test_results/080_phase11_ldsm_t.log
```
通过条件：所有 `fp8_gt_cos ≥ 0.995`

### Example case 验证
```bash
bash /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/run_hstu8_examples.sh
```
通过条件：`14/14 passed`

### 性能基准
```bash
REPO=/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu
COMMIT=$(git -C ${REPO} rev-parse --short HEAD)
CLOCK=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits | head -1 | tr -d ' ')
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user \
python ${REPO}/fbgemm_gpu/experimental/hstu/benchmark/bench_hstu_attn_sm120.py \
  2>&1 | tee ${REPO}/2benchmark_results/030_${COMMIT}_gpu${CLOCK}MHz_phase11_ldsm_t.log
```

---

## 实施结果（2026-04-16）

| 方案 | 结果 | 根因 |
|------|------|------|
| 方案 A：make_tiled_copy_B + SM100_U8x16_LDSM_T | 编译失败 | MN_SW128 per-thread source slice 无法 vectorize 为 uint128_t |
| 方案 B（Fallback）：K_SW128 重解释 sVt_cur | 编译通过，fp8_gt_cos ≈ 0.02 | addr_MN(i,j) == addr_K(j,i)（转置关系），非相同地址，重解释读错字节 |
| 方案 C：直接 PTX + 手算 swizzle 地址 | 编译通过，CUDA error: misaligned address | `ldmatrix.m16n16.x2.trans.b8` 是 SM100A 专属 ISA，SM120 不支持此 PTX 指令，运行时非法指令错误伪装为地址对齐报错 |

**最终结论（更新 2026-04-16 第二轮）**：
- `SM100_U8x16_LDSM_T`（`ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8`）虽然在 SM120 上可用（ISA 确认支持）
- 之前 fp8_gt_cos ≈ 0.02~0.12 的根因是**寄存器分配公式错误**：
  - 错误假设 fragment shape = (2, 16, 4) = 128 uint32/thread（全 128 N-列）
  - 正确：fragment shape = (2, 4, 4) = 32 uint32/thread（每 warp 仅 32 N-列）
  - BS2 AtomLayout<_2,_4,_1> 有 4 个 N-warp，每 warp `n_warp = warp_id % 4`，d_base = n_warp * 32
  - base 公式：`(ni & 1) + 8*(ni >> 1)`，NOT `(ni & 1) + di*8 + 32*(ni>>1)`
- **修复后验证（2026-04-16）**：fp8_gt_cos ≈ 0.9996，14/14 PASS
- 2×bar.sync + 16KB cooperative copy 已删除，LDSM_T 直接从 D-major SMEM 加载

**Phase 11 最终状态**：COMPLETE，数值通过。
**Phase 11 swizzle 优化（2026-04-16）**：SmemLayoutVt_TMA 从 D-major 改为 MN_SW128，LDSM_T 地址加入 `swizzle_xor = (n_off & 7) << 4`，16B 对齐保持。fp8_gt_cos ≈ 0.9996，14/14 PASS。
- bs=4 seq=1024 h=16：FP8 +2.0% vs BF16（full），+4.3%（causal）
- bs=4 seq=2048 h=16：FP8 +4.4% vs BF16（full），+12.9%（causal）
- bs=4 seq=4096 h=16：FP8 +6.7% vs BF16（full），+15.7%（causal）

**Phase 12**：代码整洁化 ✅ 已完成（2026-04-17，commit `0aa00eca`）。

---

# Phase 12 实现计划：代码整洁化（已完成）

> 经 Codex 代码分析 + 独立确认（2026-04-16）；实施完成（2026-04-17）。
>
> **完成结果**：`sm120_qmma_builder.h` 新建（296行），`hstu_fwd_kernel.h` 净删 437 行（6KD 依赖 + 死代码），`kernel_traits.h` 精简 56 行。

## 目标

消除 `hstu_blackwell_sm120/` 目录的历史积累代码腐化：
1. 删除绝对路径 6KD include，改为本地自包含 header
2. 删除死代码（Phase 5 遗留 impl + Phase 4 generic FP8 kernel body）
3. 收口所有 CuTe include 到 `kernel_traits.h`
4. 清理 `kernel_traits.h` 中未被 WS FP8 路径使用的死 MMA 定义

**不修改任何功能逻辑，不修改 Python 侧代码，每步均可独立验证。**

---

## 前置分析结论

### 现有调用链（FP8 路径）

```
hstu_ops_gpu.cpp: hstu_varlen_fwd_120()
  → hstu_fwd_launch_template.h: run_hstu_fwd_sm120<...>()
    → if (!Has_rab):  run_hstu_fwd_sm120_fp8_ws_tma_impl()  ← 生产路径
    → if (Has_rab):   run_hstu_fwd_sm120_impl<..., Is_fp8=true>()  ← 回退路径

死代码（无调用入口）：
  run_hstu_fwd_sm120_fp8_tma_impl()  ← Phase 5 遗留，hstu_fwd_kernel.h:1556
```

### MMA 使用情况

| MMA 定义 | 定义位置 | 实际使用路径 | 是否可删 |
|---------|---------|------------|---------|
| `SM80_16x8x16_F32BF16BF16F32_TN` | `Flash_kernel_traits_sm120::MMA_Atom_Arch` | BF16 kernel（活跃） | **不可删** |
| `SM89_16x8x32_F32E4M3E4M3F32_TN` | `Hstu_fwd_kernel_traits_sm120_fp8::MMA_Atom_Arch` + `TiledMma` | Has_rab FP8 回退路径（活跃），WS 路径不用 | Step 5 决策（见下） |
| `SM120::BLOCKSCALED::SM120_16x8x32_TN_VS` | `SM120BlockScaledBuilder::TiledMma`（via BS1/BS2） | WS FP8 kernel（生产路径，已在用） | 迁入本地后保留 |

---

## Step 1：删除死代码

**改动文件**：`hstu_fwd_kernel.h`

**删除内容**：
- `run_hstu_fwd_sm120_fp8_tma_impl` 函数（lines 1539–1617，约 80 行）
- 相关 `Hstu_fwd_params_fp8_tma` struct（若仅此函数使用）

**前提检查**（执行后再删）：
```bash
grep -r "run_hstu_fwd_sm120_fp8_tma_impl" \
  /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/ --include="*.h" --include="*.cpp" --include="*.cu"
```
确认只有定义，无调用点。

**验证**：
```bash
# 编译
cd fbgemm_gpu/experimental/hstu && HSTU_ARCH_LIST="12.0" ... pip install -e .
# 功能验证
bash run_hstu8_examples.sh  # 14/14 PASS
```

---

## Step 2：新建 `sm120_qmma_builder.h`

**新建文件**：`hstu_blackwell_sm120/sm120_qmma_builder.h`

**内容**：从 `sm120_blockscaled_utils.cuh` 中提取 WS FP8 实际使用的最小子集：

```cpp
#pragma once
// SM120 Block-Scaled QMMA builder — local copy, only the subset used by hstu_fwd_kernel_fp8_ws.h.
// Original source: 6KD_fp8_block_scale/kernels/.../sm120_blockscaled_utils.cuh
// DO NOT add new content here. Extend kernel_traits.h instead.

// Note: kernel_traits.h must be included before this file (provides cute includes)

namespace hstu_sm120_qmma {

template <int TileM_ = 128, int TileN_ = 128, int Stages_ = 4>
struct SM120QmmaBuilder {
  using ElementA        = cute::float_e4m3_t;
  using ElementB        = cute::float_e4m3_t;
  using ElementSFLoad   = int32_t;
  using ElementSFCompute= cute::float_ue8m0_t;
  using ElementAccum    = float;

  static constexpr int kTileM    = TileM_;
  static constexpr int kTileN    = TileN_;
  static constexpr int kTileK    = 128;
  static constexpr int AB_Stages = Stages_;

  // SM120 native block-scale QMMA: 16x8x32, FP8 inputs, block scale UE8M0
  using MmaAtom = cute::MMA_Atom<
      cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
          cute::float_e4m3_t, cute::float_e4m3_t, float, cute::float_ue8m0_t, 32>>;
  using TiledMma = cute::TiledMMA<
      MmaAtom,
      cute::Layout<cute::Shape<cute::_2, cute::_4, cute::_1>,
                   cute::Stride<cute::_4, cute::_1, cute::_0>>,
      cute::Tile<
          cute::Int<32>,
          cute::Layout<cute::Shape<cute::_8, cute::_4, cute::_4>,
                       cute::Stride<cute::_1, cute::_32, cute::_8>>,
          cute::Underscore>>;

  using SmemCopyAtomA   = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementA>;
  using SmemCopyAtomB   = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementB>;
  using SmemLayoutAtomA = cute::GMMA::Layout_K_SW128_Atom<ElementA>;
  using SmemLayoutAtomB = cute::GMMA::Layout_K_SW128_Atom<ElementB>;
  using SmemCopyAtomSF  = cute::Copy_Atom<cute::DefaultCopy, ElementSFLoad>;

  using SmemLayoutSFA = cute::Layout<cute::Shape<cute::Int<kTileM>, cute::Int<1>>,
                                     cute::Stride<cute::_1, cute::Int<kTileM>>>;
  using SmemLayoutSFB = cute::Layout<cute::Shape<cute::Int<kTileN>, cute::Int<1>>,
                                     cute::Stride<cute::_1, cute::Int<kTileN>>>;

  // Static helper methods (ported from SM120BlockScaledBuilder verbatim)
  template <class SFATensor, class ThrMma>
  CUTE_HOST_DEVICE static constexpr auto
  partition_fragment_SFA(SFATensor&&, ThrMma&);

  template <class SFBTensor, class ThrMma>
  CUTE_HOST_DEVICE static constexpr auto
  partition_fragment_SFB(SFBTensor&&, ThrMma&);

  template <class TiledMma_>
  CUTE_HOST_DEVICE static constexpr auto
  get_layoutSFA_TV(TiledMma_&);

  template <class TiledMma_>
  CUTE_HOST_DEVICE static constexpr auto
  get_layoutSFB_TV(TiledMma_&);

  template <class Tensor>
  CUTE_HOST_DEVICE static constexpr auto
  transform_fragment_for_qmma(Tensor&&);
};

} // namespace hstu_sm120_qmma
```

**关键要求**：
- 静态方法体与 6KD 原文完全一致（逐字 port，不做任何修改）
- 命名空间改为 `hstu_sm120_qmma`，避免与 6KD 命名空间冲突
- 调用方 `BS1/BS2` 类型别名改为 `using BS1 = hstu_sm120_qmma::SM120QmmaBuilder<kBlockM, kBlockN, 4>`

**验证**：
```bash
HSTU_SWEEP_FP8_QUANT_MODE=2 python sweep_accuracy.py  # 6/6 fp8_gt_cos ≥ 0.995
bash run_hstu8_examples.sh                             # 14/14 PASS
```

---

## Step 3：删除 6KD include

**改动文件**：`hstu_fwd_kernel.h`

**改动**：
```diff
-// Ensure cutlass::gemm namespace exists before sm120_blockscaled_utils.cuh tries
-#include "/home/.../6KD_fp8_block_scale/.../sm120_blockscaled_utils.cuh"
+#include "sm120_qmma_builder.h"
```

同时在 `kernel_traits.h` 添加：
```cpp
#include <cute/atom/mma_traits_sm120.hpp>  // SM120::BLOCKSCALED QMMA
```

**风险**：`CUTE_ARCH_TMA_SM120_ENABLED` 宏（`hstu_fwd_kernel.h:45`）必须在 `copy_sm90_tma.hpp` include 之前定义，收口时不可打乱此顺序。

**验证**：同 Step 2。

---

## Step 4：收口 CuTe includes 到 `kernel_traits.h`

**改动文件**：`hstu_fwd_kernel.h`

**删除**：
```diff
-#include <cute/tensor.hpp>
-#include <cute/arch/copy_sm90_tma.hpp>
```

这两个 include 通过 `kernel_traits.h` 已间接引入（`cute/tensor.hpp` 已在 `kernel_traits.h:18`）。

**注意**：`cute/arch/copy_sm90_tma.hpp` 需要在 `CUTE_ARCH_TMA_SM120_ENABLED` 宏定义之后 include。`kernel_traits.h` 中无宏定义，需将宏定义（`hstu_fwd_kernel.h:43-47`）迁移到 `kernel_traits.h`，或保留在 `hstu_fwd_kernel.h` 中（`copy_sm90_tma.hpp` 只在该文件中使用）。

**推荐方案**：TMA include 保留在 `hstu_fwd_kernel.h`（因宏定义依赖），仅迁移 `cute/tensor.hpp`（`kernel_traits.h` 已有）。

**验证**：编译通过即可。

---

## Step 5：清理 `kernel_traits.h` 死 MMA 定义

**决策依据**：
- `Hstu_fwd_kernel_traits_sm120_fp8::TiledMma`（基于 `SM89_16x8x32`）是否还被 Has_rab=true 的 FP8 回退路径使用？
  - 是：保留 SM89 include + SM89 MMA_Atom_Arch，但加注释说明其用途仅限 Has_rab 回退
  - 否（Has_rab 回退路径也切到 SM120 QMMA 后）：删除 SM89 include + SM89 dead defs

**当前建议**（保守方案）：
- 保留 `mma_sm89.hpp` / `mma_traits_sm89.hpp` include 以及 SM89 `MMA_Atom_Arch`（Has_rab 回退路径仍依赖）
- 添加注释明确标注哪套定义属于哪条路径
- 将 `Flash_kernel_traits_sm120` base struct 中的 SM80 `MMA_Atom_Arch` 重命名为 `MMA_Atom_Arch_BF16` 并注释，与 SM89 定义区分

**改动**：
- 删除 `Hstu_fwd_kernel_traits_sm120_fp8::TiledMma`（SM89 版本，WS 路径不用）
- 在所有 MMA 定义上方加注释，明确生效路径

**验证**：
```bash
HSTU_SWEEP_FP8_QUANT_MODE=2 python sweep_accuracy.py  # 6/6 PASS
bash run_hstu8_examples.sh                             # 14/14 PASS（含 Has_rab cases）
```

---

## 验证矩阵

每个 Step 完成后必须执行：

| 验证项 | 命令 | 通过标准 |
|--------|------|---------|
| 编译 | `pip install -e .`（HSTU_ARCH_LIST=12.0）| 无 error |
| 数值正确性 | `HSTU_SWEEP_FP8_QUANT_MODE=2 python sweep_accuracy.py` | fp8_gt_cos ≥ 0.995 |
| 全场景覆盖 | `bash run_hstu8_examples.sh` | 14/14 PASS |

## 文件改动汇总

| 文件 | Step | 改动 |
|------|------|------|
| `sm120_qmma_builder.h` | Step 2 | **新建**：SM120BlockScaledBuilder 最小子集（重命名为 SM120QmmaBuilder） |
| `hstu_fwd_kernel.h` | Step 1 | 删除 `run_hstu_fwd_sm120_fp8_tma_impl` 死代码 |
| `hstu_fwd_kernel.h` | Step 3 | 替换 6KD 绝对路径 include 为 `sm120_qmma_builder.h` |
| `hstu_fwd_kernel.h` | Step 4 | 删除已在 `kernel_traits.h` 覆盖的 cute include |
| `kernel_traits.h` | Step 3 | 新增 `mma_traits_sm120.hpp` include |
| `kernel_traits.h` | Step 5 | 删除 SM89 FP8 TiledMma（dead），为各 MMA 定义加注释 |
| `hstu_fwd_kernel_fp8_ws.h` | — | **无需改动** |

---

# Phase 13 实现计划：P staging 同步优化

> 技术分析经 Codex 二次确认（2026-04-17）。

## 背景

GEMM1 输出 `acc_s`（float32，kBlockM×kBlockN=128×128，每线程 64 元素）通过以下步骤转化为 GEMM2 输入 `tCrP`：

```
acc_s（寄存器）
  → F32→FP8 量化（acc_s_packed）
  → bar.sync 1,256          ← 第一个屏障（等所有线程量化完毕再写 SMEM）
  → 写 sPbuf（aliased sK_cur）
  → bar.sync 1,256          ← 第二个屏障（等所有线程写完再读 SMEM）
  → LDSM → tCrP（寄存器）
```

两个 `bar.sync 1,256` 串行阻塞全部 256 个 math 线程。本 Phase 目标是减少这两个同步屏障的代价。

---

## 技术分析结论

### 为何纯 warp shuffle 不可行

- GEMM1 使用 `TiledMMA<AtomLayout<_2,_4,_1>>` — 8 个 warp 排列为 2M×4N
- warp j（N 方向）持有 P[:,j*32:(j+1)*32)（N-cols 范围），以 D-format 存于寄存器
- GEMM2 K-step j 需要**全部 8 个 warp** 读取第 j 个 N-col 块 → 跨 N-warp 广播
- `__shfl_sync` 仅作用于单个 warp 内 32 条 lane；跨 warp 数据交换必须经过 SMEM

### `AB_Stages=4` 不产生 N 方向分片

- `AB_Stages=4`（SM120QmmaBuilder 模板参数）控制 scale factor 流水线（`kNumTileKPerSF = 512/128 = 4`），**非** N 方向分区
- 一次 `cute::gemm(tiled_mma, tCrQ, tCrK, acc_s)` 对 HeadDim 方向所有 4 个 K=32 子步累加，最终结果是完整的 128×128 P 矩阵（所有位置同时更新），无法在 K 步间截取 N-col 分片
- 结论：warp shuffle 优化 P staging **在数学上不可行**

---

## 可行优化方案

### 方案 A（主方案）：拆分 `bar.sync 1,256` 为两个 `bar.sync 128`

**原理**：P 矩阵的 M 方向由两个 warpgroup（WG0: warp 0-3，WG1: warp 4-7）分别负责不同的 M 行：
- WG0（warp 0-3）：P 行 0..63
- WG1（warp 4-7）：P 行 64..127

这两半的写入和读取**完全独立**，无需等对方。可将 `bar.sync 1,256` 拆为：
- `bar.sync A,128`（WG0 内部同步）
- `bar.sync B,128`（WG1 内部同步）

两组可独立推进，减少等待。

**实现**：
```cpp
// 替换 bar.sync 1,256
const int wg_id = tidx_math / 128;          // 0 = WG0, 1 = WG1
const int bar_id = (wg_id == 0) ? 2 : 3;   // 用 bar ID 2,3（0,1 已被 tma/math mbar 占用）
asm volatile("bar.sync %0, 128;" :: "r"(bar_id));
```

**注意**：需在 `kernel_traits.h` 的 `kNSyncBarriers` 中为这两个 bar 预留槽位（如果有显式声明）。

### 方案 B（可选）：P 使用独立 SMEM buffer，消除第一个屏障

**原理**：当前 sPbuf aliased 到 `sK_cur`，写入前需要确保 K 被 GEMM1 读完（第一个 bar.sync 的语义）。若给 P 分配独立 SMEM buffer（不与 K/V^T 复用），第一个 bar.sync 可省略。

**代价**：新增 128×128×1 = 16KB SMEM（已经很紧张），需评估 occupancy 影响后决定是否实施。

---

## 实施步骤

### Step 1：替换第一个 `bar.sync 1,256`（写前同步）

**文件**：`hstu_fwd_kernel_fp8_ws.h`，找到 GEMM1 结束后的第一个 `bar.sync 1,256`

```cpp
// 删除
asm volatile("bar.sync 1, 256;");

// 替换为
const int wg_bar = (tidx_math < 128) ? 2 : 3;
asm volatile("bar.sync %0, 128;" :: "r"(wg_bar));
```

### Step 2：替换第二个 `bar.sync 1,256`（读前同步）

```cpp
// 删除
asm volatile("bar.sync 1, 256;");

// 替换为
asm volatile("bar.sync %0, 128;" :: "r"(wg_bar));
// wg_bar 在 Step 1 中已定义，可直接复用
```

### Step 3：验证 bar ID 可用性

检查 `hstu_fwd_kernel_fp8_ws.h` 中已使用的 bar ID（tma_mbar/math_mbar 用 0,1），确认 2,3 未被占用。

---

## 验证流程

```bash
# 编译
mkdir -p /tmp/claude && \
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu && \
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user \
HSTU_ARCH_LIST="12.0" HSTU_DISABLE_BACKWARD=TRUE \
HSTU_DISABLE_HDIM32=TRUE HSTU_DISABLE_HDIM64=TRUE HSTU_DISABLE_HDIM256=TRUE \
MAX_JOBS=32 pip install --no-build-isolation --config-settings editable_mode=compat -e . \
  2>&1 | grep -E "error:|note:|static_assert|undefined" | head -60

# 数值验证
HSTU_SWEEP_FP8_QUANT_MODE=2 \
python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/sweep_accuracy.py \
  2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/1test_results/090_phase13_bar_split.log

# 全场景验证
bash /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/run_hstu8_examples.sh
```

**通过标准**：fp8_gt_cos ≥ 0.995，14/14 PASS

---

## 预期收益与风险

| 项 | 说明 |
|----|------|
| 预期收益 | 每个 N-tile 迭代节省 1-2 个 256-thread 全局屏障，WG0 和 WG1 可独立推进写/读 P |
| 风险 | bar ID 冲突（需确认 2,3 未被 mbarrier slot 占用）；SM120 上 named-barrier 语义与 mbarrier 语义不同，需确认兼容性 |
| 方案 B 条件 | 仅在 SMEM 预算允许（+16KB）且 occupancy 不下降时实施 |
