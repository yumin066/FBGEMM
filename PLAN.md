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

# Phase 13 实现计划：消除 P staging SMEM — AtomLayout `<_8,_1,_1>` 重构

> 方案经 Codex 二次确认（2026-04-17）；替代旧方案（bar.sync 256→128 分割，已废弃）。

## 背景与目标

### 当前 P staging 开销

每个 N-block 迭代（N_total 次）：
```
acc_s (F32, 64 elem/thread)
  → F32→FP8 量化 (acc_s_packed，4 FP8/uint32)
  → bar.sync 1,256          ← 阻塞全部 256 math threads
  → SMEM 写 sPbuf (16KB, aliased sK_cur)
  → bar.sync 1,256          ← 再次阻塞
  → LDSM → tCrP (FP8, A-layout, 256 elem/thread)
```

根因：`<_2,_4,_1>` 下 4 个 N-warp 各持有 P 的 32/128 列，GEMM2 A-fragment 需要全部 128 列，必须跨 warp 通过 SMEM 交换。

### 新方案

将 AtomLayout 改为 `<_8,_1,_1>`（8M-warp × 1N-warp = 8 warp）：
- 每个 warp 覆盖 **16M 行 × 全部 128N 列**（MMA_M=1, MMA_N=16）
- GEMM1 结束后每个 warp 寄存器已持有本 warp 所有 M 行对应的完整 128 列 P 数据
- D→A 格式转换纯在 warp 内完成：lane t（t%4=p）在 K-rep k_a 所需的 8 个 K 列全来自 N-rep `s=k_a*4+p`，源 lane `{t&~3 .. t&~3+3}`，约 12 次 int32 `__shfl_sync`
- **完全消除** 2×`bar.sync 1,256` + 16KB SMEM 写读

---

## 关键设计决策（经 Codex 确认）

| 决策 | 说明 |
|------|------|
| PermMmaTileN | `Layout<Shape<_8,_1,_16>, Stride<_1,_128,_8>>`（size=128，1N-warp × 16 atom，identity 排列）|
| SFB bug 修正 | `partition_fragment_SFB` 的 `get<1>(thr_vmnk)` 改为 `get<2>`（当前在 `<_2,_4,_1>` 下被 stride-0 collapse 掩盖，新 layout 下会产生实错）|
| D→A 转换时机 | FP8 量化**之后**操作（不在 F32 acc_s 上），`acc_s_packed` = 4 FP8/uint32 |
| Pack 设计 | 不复用当前 `acc_s_packed` 直接 shuffle；需设计面向 `tCrP` 目标 layout 的 shuffle-friendly pack |
| acc_o 大小 | 每 warp 仍为 2048 float32（16M×128N），总量不变；epilogue `partition_C` 写回逻辑无需手改 |

---

## 实施步骤

### Step 1：改 builder 骨架 + 修复 SFB bug

**文件**：`sm120_qmma_builder.h`

- `:73-98`：
  ```cpp
  using PermMmaTileN = Layout<Shape<_8,_1,_16>, Stride<_1,_128,_8>>;
  // TiledMma AtomLayout:
  Layout<Shape<_8,_1,_1>, Stride<_1,_0,_0>>
  ```
- `:241-255`（`partition_fragment_SFB`）：`get<1>(thr_vmnk)` → `get<2>(thr_vmnk)`

**验证**：编译通过（静态 assert 不报错）

**最大风险**：builder helper 内其他处隐含了旧 2×4 warp 网格语义

---

### Step 2：只切 BS2，重写 V^T 手写 ldmatrix（保留 P staging）

**文件**：`hstu_fwd_kernel_fp8_ws.h`

- `:49-52`：`BS2 = SM120QmmaBuilder<kBlockM, kHeadDim, 4>` 使用新 layout builder
- `:916-997`：V^T 手写 ldmatrix 重写，新循环结构：
  ```cpp
  // 外层 ni=0..7（K-group），内层 dg=0..3（D-group）
  // fragment 逻辑 shape: (reg=2, n_atom=16, k_step=4)
  // d_group_base = dg * 32;
  // addr = vt_base + n_k_row * kHeadDim + (d_start ^ swizzle_xor);
  ```

**验证**：编译 + sweep_accuracy.py fp8_gt_cos ≥ 0.995

**最大风险**：`tCrV` fragment shape 与预期不符导致寄存器落点错位

---

### Step 3：切 BS1 到新 layout，验证 GEMM1（保留 P staging）

**文件**：`hstu_fwd_kernel_fp8_ws.h:49-52`

- `BS1 = SM120QmmaBuilder<kBlockM, kBlockN, 4>` 也使用新 layout builder
- P staging 代码**暂时保留**，仅验证 GEMM1 acc_s 分布是否正确

**验证**：编译 + sweep_accuracy.py fp8_gt_cos ≥ 0.995 + run_hstu8_examples.sh 14/14 PASS

**最大风险**：acc_s lane/col 坐标分布与纸面推导存在偏差

---

### Step 4：去掉 P staging，替换为 warp 内 D→A 转换

**文件**：`hstu_fwd_kernel_fp8_ws.h:839-914`

- 删除：2×`bar.sync 1,256` + sPbuf write + LDSM read
- 替换为：
  1. acc_s → FP8 量化（保留现有逻辑）
  2. 设计 shuffle-friendly pack（4 FP8/uint32，按 tCrP 目标 layout 排列）
  3. 每个 k_a（0..3）做 1 次 quad gather（3 次 `__shfl_sync`），直接填入 tCrP

**验证**：编译 + sweep_accuracy.py fp8_gt_cos ≥ 0.995 + 14/14 PASS + benchmark 对比基线

**最大风险**：pack 设计与 tCrP 目标 layout 不匹配，导致额外拆包重组抵消收益

---

## 验证命令

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
  2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/1test_results/090_phase13_atom_layout.log

# 全场景验证
bash /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/run_hstu8_examples.sh
```

---

## 预期收益与风险

| 项 | 说明 |
|----|------|
| 预期收益 | 每 N-block 消除 2×`bar.sync 256` + 16KB SMEM write/read；12 次 int32 shuffle 代价远小于当前 SMEM roundtrip |
| 新增成本 | `tCrV` 寄存器从 MMA_N=4 增至 MMA_N=16（4×），需确认 spill 不升高 |
| SMEM V^T 访问 | 8 个 M-warp 读同一组 128 列，无 bank conflict，但总读带宽集中，需实测 |
| 回退方案 | 如 Step 3 acc_s 分布验证失败，回退至旧方案（bar.sync 256→128 分割，archived below）|
