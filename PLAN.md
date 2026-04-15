# Phase 11 实现计划：原生 SM120 CuTe 类型替换

## 背景

当前 FP8 WS kernel 存在两处"借用"非 SM120 原生类型的地方：

| 借用来源 | 当前用途 | SM120 原生替换 |
|---------|--------|--------------|
| `mma_sm89.hpp` / `mma_traits_sm89.hpp` | GEMM1/GEMM2 的 MMA atom（`SM89_16x8x32_F32E4M3E4M3F32_TN`） | `SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<..., VS=32>`（`mma_sm120.hpp`/`mma_traits_sm120.hpp`） |
| `mma_traits_sm90_gmma.hpp` | SMEM layout atom（`GMMA::Layout_K_SW128_Atom`）+ SmemLayoutVt_SW128 | 保留（SM120 SMEM layout atom 沿用 SW128 swizzle，原 header 仍需） |

Phase 11 还包含：
- 用 `SM100_U8x16_LDSM_T` 替换 SMEM→SMEM transpose（消除 bar.sync + 16KB copy loop）

---

## 关键信息

### A. 当前 SM89 MMA（待替换）

**文件**：`kernel_traits.h:26-28, 252, 289-292, 300-301`

```cpp
// 待删除的 include
#include "cute/arch/mma_sm89.hpp"
#include "cute/atom/mma_traits_sm89.hpp"

// MMA atom（第 252 行）
using MMA_Atom_Arch = MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>;
// ISA: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
// 无内置 block scale → 计算后手动乘 descale_qk

// SMEM Layout Q（第 300-301 行）
using SmemLayoutAtomQ = Layout<Shape<_16, _32>, Stride<_32, _1>>;  // 无 swizzle

// TiledMma（第 289-292 行）
using TiledMma = TiledMMA<
    MMA_Atom_Arch,
    Layout<Shape<Int<kNWarps>, _1, _1>>,
    Tile<Int<16 * kNWarps>, _16, _32>>;
```

### B. SM120 原生 block-scale MMA（替换目标）

**定义**：`external/cutlass/include/cute/arch/mma_sm120.hpp`  
**Traits**：`external/cutlass/include/cute/atom/mma_traits_sm120.hpp`  
**参考**：`6KD_fp8_block_scale/kernels/include/sm120_blockscaled_gemm/sm120_blockscaled_utils.cuh`

```cpp
// 新 MMA atom
using MMA_Atom_Arch = MMA_Atom<
    SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
        float_e4m3_t, float_e4m3_t, float, float_ue8m0_t, /*VS=*/32>>;
// ISA: mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
// 内置 block scale：SFA/SFB 作为额外 operand 传入，无需 post-multiply

// 新 SMEM Layout Atom Q
using SmemLayoutAtomQ = GMMA::Layout_K_SW128_Atom<Element>;
// Swizzle<3,4,3>, 8×128 FP8 atom（与 SmemLayoutAtomB 相同）
```

**关键**：SM120 block-scale MMA 的 `MMA_Traits` 继承自 `SM120_16x8x32_TN`，后者的 Shape_MNK、ThrID、ALayout、BLayout、CLayout 与 SM89 完全相同 → **register 格式不变，GEMM 调用侧代码无需修改**。

### C. Scale Factor 处理变化（最重要）

| 项目 | 当前（SM89 方式） | SM120 原生方式 |
|------|----------------|--------------|
| SFB/SFV 加载 | TMA（已完成） | 不变 |
| 应用方式 | GEMM 后 `acc_s *= descale_qk`（float 乘法） | 作为 SFA/SFB 参数传入 MMA 指令 |
| 格式 | int32（4×e8m0 packed） | 每次 MMA 需 1 个 uint8_t（e8m0），连续 4 次 MMA 共用同一个 scale |
| kHeadDim=128 时 | 4 次 MMA-K iteration（各 32K），共用 1 个 SFB | 每次 MMA 传同一 SFB，循环展开 4 次 |

具体：`sf_k_packed` 每个 int32 = 4 个 e8m0 packed（covers 4×128=512 K-elements）  
kHeadDim=128 → 4 次 MMA-K 共用 1 个 e8m0 scale → 从 int32 中取 bits[0:7]（block 0）传入 SFB

### D. LDSM_T（消除 SMEM transpose）

**位置**：`hstu_fwd_kernel_fp8_ws.h:909-917`  
**Guard 问题**：`SM100_U8x16_LDSM_T` 需要 `CUTE_ARCH_LDSM_SM100A_ENABLED`，SM120 消费级不自动定义 → 需显式 force-define（参照 `hstu_fwd_kernel.h:36-49` 的 TMA guard 先例）。  
**Byte interleave 风险**：LDSM_T 内含 byte reorder（为 SM100A WGMMA 设计），替换 SM89 MMA 为 SM120 后，B operand register format 可能与 LDSM_T 输出匹配 → 需数值验证。  
**Fallback**：以 K_SW128 view 直接重解释 sVt_cur（对 128×128 方形 tile 物理字节等价），省去 transpose loop。

---

## 实现步骤

### Step 1：更新 include（`kernel_traits.h:26-28`）

```cpp
// 删除
// #include "cute/arch/mma_sm89.hpp"
// #include "cute/atom/mma_traits_sm89.hpp"

// 新增
#include "cute/arch/mma_sm120.hpp"
#include "cute/atom/mma_traits_sm120.hpp"
#include "cute/arch/copy_sm100.hpp"          // SM100_U8x16_LDSM_T
#include "cute/atom/copy_traits_sm100.hpp"   // Copy_Traits<SM100_U8x16_LDSM_T>

// 保留（SW128 layout atom 仍需）
#include "cute/atom/mma_traits_sm90_gmma.hpp"
```

### Step 2：替换 MMA atom（`kernel_traits.h:252`）

```cpp
// 删除
using MMA_Atom_Arch = MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>;

// 替换为
using MMA_Atom_Arch = MMA_Atom<
    SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
        cute::float_e4m3_t, cute::float_e4m3_t, float, cute::float_ue8m0_t, 32>>;
```

### Step 3：更新 SmemLayoutAtomQ（`kernel_traits.h:300-301`）

```cpp
// 删除
using SmemLayoutAtomQ = Layout<Shape<_16, _32>, Stride<_32, _1>>;

// 替换为（与 SmemLayoutAtomB / SW128 对齐）
using SmemLayoutAtomQ = GMMA::Layout_K_SW128_Atom<Element>;
```

注：`SmemLayoutQ` 由 `tile_to_shape(SmemLayoutAtomQ{}, Shape<kBlockM, kHeadDim>{})` 派生，自动适配新 atom。

### Step 4：更新 TiledMma 配置（`kernel_traits.h:289-292`）

SM120 block-scale MMA Traits 继承自 SM120_16x8x32_TN → Shape/ThrID 不变，thread layout 需验证与 SM120BlockScaledBuilder 的一致性（参考实现用 `Layout<Shape<_2,_4,_1>>`）。  
**保守策略**：先保持当前 thread layout `Layout<Shape<kNWarps,_1,_1>>`，编译通过后对比数值，再根据需要调整。

### Step 5：修改 GEMM1/GEMM2 中 scale factor 传递

**文件**：`hstu_fwd_kernel_fp8_ws.h`，找到 GEMM1 调用（约 808 行）和 post-multiply descale（`acc_s *= descale_qk`）。

**当前流程**：
```cpp
cute::gemm(tiled_mma_g1, tCrQ, tCrK, acc_s);   // 无 scale
acc_s *= descale_qk;                             // 手动 post-multiply
```

**SM120 新流程**（伪代码，具体 API 参考 `sm120_blockscaled_gemm_impl.cuh`）：
```cpp
// SFA（query scale）：从 smem_sfa_ptr 读取 uint8_t，转为 float_ue8m0_t
auto sfa_val = reinterpret_cast<cute::float_ue8m0_t>(smem_sfa_ptr[sfa_idx]);
// SFB（key scale）：从 smem_sfb_ptr 读取 int32 → 取 bits[0:7] = block 0 的 e8m0
uint8_t sfb_u8 = static_cast<uint8_t>(smem_sfb_val & 0xFF);
auto sfb_val = reinterpret_cast<cute::float_ue8m0_t>(sfb_u8);

// SM120 block-scale MMA 调用（SFA/SFB 作为额外参数）
cute::gemm(tiled_mma_g1, tCrQ, tCrK, tCrSFA, tCrSFB, acc_s);
// 内置 scaling，无需 acc_s *= descale_qk
```

> 具体 API 参考 `sm120_blockscaled_gemm_impl.cuh` 中的 `gemm_blockscaled()` 或 `MMA_Traits<SM120_BLOCKSCALED_...>::fma()` 函数签名。

类似地处理 GEMM2（acc_o）中的 SFV。

### Step 6：添加 LDSM Guard + 消除 SMEM transpose（11-B，在 11-A 验证通过后进行）

**6a** — 在 `hstu_fwd_kernel.h` 的 guard 区域（第 36–49 行）追加：
```cpp
#if defined(__CUDA_ARCH__)
#ifndef CUTE_ARCH_LDSM_SM100A_ENABLED
#define CUTE_ARCH_LDSM_SM100A_ENABLED
#endif
#endif
```

**6b** — 在 `hstu_fwd_kernel_fp8_ws.h` 删除 transpose loop（909–917），替换 s2r 段（920–928）：
```cpp
// 删除 transpose loop + bar.sync（909-917）

// 新 s2r：直接从 MN_SW128 sVt_cur 用 LDSM_T 转置加载
using SmemCopyAtomVt = Copy_Atom<SM100_U8x16_LDSM_T, FP8Elem>;
auto s2r_copy_Vt = make_tiled_copy_B(SmemCopyAtomVt{}, tiled_mma_g2);
auto sVt_pi = as_position_independent_swizzle_tensor(
    make_tensor(make_smem_ptr(sVt_cur), SmemLayoutVt_SW128{}));
Tensor tCrV = thr_mma_g2.partition_fragment_B(sVt_pi);
{
  auto thr_copy_Vt = s2r_copy_Vt.get_thread_slice(tidx_math);
  auto tXsVt = thr_copy_Vt.partition_S(sVt_pi);
  auto tXrV  = thr_copy_Vt.retile_D(tCrV);
  cute::copy(s2r_copy_Vt, tXsVt, tXrV);
}
```

---

## 关键文件

| 文件 | 修改内容 |
|------|--------|
| `src/hstu_blackwell_sm120/kernel_traits.h` | include 替换（Step 1）、MMA_Atom_Arch（Step 2）、SmemLayoutAtomQ（Step 3）、TiledMma（Step 4） |
| `src/hstu_blackwell_sm120/hstu_fwd_kernel.h` | LDSM guard（Step 6a） |
| `src/hstu_blackwell_sm120/hstu_fwd_kernel_fp8_ws.h` | GEMM scale 传递（Step 5）、transpose 消除（Step 6b） |
| `external/cutlass/include/cute/arch/mma_sm120.hpp` | 只读参考 |
| `external/cutlass/include/cute/atom/mma_traits_sm120.hpp` | 只读参考 |
| `6KD_fp8_block_scale/.../sm120_blockscaled_gemm_impl.cuh` | 只读参考（SM120 scale 传递 API） |

---

## 实施顺序

**11-A**（先做）：Step 1~5（include + MMA atom + SmemLayoutAtomQ + scale 传递）  
→ 编译 + `sweep_accuracy.py` 验证（cos_sim ≥ 0.995）+ 14-case 验证

**11-B**（后做）：Step 6（LDSM_T 消除 SMEM transpose）  
→ 再次验证 + benchmark

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
  2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/1test_results/080_phase11_sm120_mma.log
```
**通过条件**：所有 `fp8_gt_cos ≥ 0.995`

### Example case 验证
```bash
bash /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/run_hstu8_examples.sh
```
**通过条件**：`14/14 passed`

### 性能基准
```bash
REPO=/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu
COMMIT=$(git -C ${REPO} rev-parse --short HEAD)
CLOCK=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits | head -1 | tr -d ' ')
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user \
python ${REPO}/fbgemm_gpu/experimental/hstu/benchmark/bench_hstu_attn_sm120.py \
  2>&1 | tee ${REPO}/2benchmark_results/030_${COMMIT}_gpu${CLOCK}MHz_phase11.log
```

---

## 决策树

```
Step 1-4 编译通过？
  ├─ 否 → 检查 SM120::BLOCKSCALED 命名空间；检查 SmemLayoutAtomQ 与 tile_to_shape 兼容性
  └─ 是 →
       Step 5（scale 传递）编译通过？
         ├─ 否 → 参考 sm120_blockscaled_gemm_impl.cuh 的 cute::gemm 签名
         └─ 是 →
              sweep_accuracy cos_sim ≥ 0.99？
                ├─ 是 → Step 6（LDSM_T）→ 再次验证 → benchmark
                └─ 否 → 调试：GEMM1 单独验证（acc_s all-ones test），定位 scale 传递 or layout 问题
```
