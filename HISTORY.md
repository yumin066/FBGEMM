# HSTU SM120 历史阶段详细记录

本文件存档已完成阶段的技术细节、Bug 修复记录和性能数据，供查阅参考。
当前活跃阶段（Phase 13）的详细设计见 `CLAUDE.md` 和 `PLAN.md`。

---

## Phase 4：K+V Double-Prefetch Pipeline（2026-03-27）

**目标**：K+V double-prefetch pipeline，消除迭代中途 blocking V wait。

**性能结果**（RTX PRO 6000 SM120，bs=4, h=16, d=128, full attention）：

| seq  | BF16      | FP8 (qm=2) | 差距  |
|------|-----------|------------|-------|
| 512  | 0.048ms 181 TFLOPS | 0.057ms 151 TFLOPS | -16% |
| 1024 | 0.126ms 273 TFLOPS | 0.154ms 224 TFLOPS | -18% |
| 2048 | 0.479ms 287 TFLOPS | 0.582ms 236 TFLOPS | -18% |

---

## Phase 5：TMA K+V^T（2026-03-30）

**目标**：用 TMA 直接加载转置后的 V^T，消除 kernel 内逐元素 V 转置循环；K 从 cp.async 升级为 TMA。

### 实现内容

- **V^T TMA**：V 以转置视图 `[d, total_v, h_k]`、strides `[1, v_row_stride, v_head_stride]` 描述给 TMA，直接加载到 `SmemLayoutVt_TMA`（SW128）
- **K TMA**：K 加载从 cp.async 升级为 TMA，与 V^T 共用同一 `uint64_t` mbarrier
- **新模板参数** `bool Use_TMA_KV`：FP8 路径走 TMA，BF16 路径维持 cp.async

### 关键技术要点

- `with()` 需要 `uint64_t&`（不是 `ClusterTransactionBarrier`），mbarrier 操作全部用 PTX inline
- `local_tile` 须用 `make_coord(_, _)` 取全部 tile，在 `copy` 时以 `nb_abs` 索引

### 性能结果（RTX PRO 6000 SM120，2026-03-30）

| seq  | BF16    | FP8 (qm=2) | 差距   |
|------|---------|------------|--------|
| 512  | 0.044ms | 0.053ms    | -20.5% |
| 1024 | 0.119ms | 0.144ms    | -21.0% |
| 2048 | 0.447ms | 0.544ms    | -21.7% |
| 4096 | 1.551ms | 1.912ms    | -23.3% |

---

## Phase 6：Warp-Specialized FP8 Kernel（2026-04-03）

**目标**：warp 0 专职发 TMA（K+V^T），warp 1-8 专职 QMMA，2-stage double-buffer 实现 TMA 与 MMA 真正重叠。

### 实现内容

- **新文件** `hstu_fwd_kernel_fp8_ws.h`：WS 内核主体
- **Warp 分工**：warp 0 = load warp（288 线程总计，32 load + 256 math）
- **两对 barrier（4个 uint64_t）**：
  - `tma_mbar[2]`：TMA 完成信号，math warps 直接等待
  - `math_mbar[2]`：SMEM 消费完成信号，load warp 等待
- **Double-buffer 时序**：load warp preamble 预发两个 tile（s0、s1），TMA 与 MMA 真正并行

### 关键 Bug 修复记录

#### CuTe SM90 TMA header debug printf（严重性能问题）
- **文件**：`external/cutlass/include/cute/arch/copy_sm90_tma.hpp:172`
- **问题**：`SM90_TMA_LOAD_3D::copy()` 在 `CUTE_ARCH_TMA_SM120_ENABLED` 路径下有 `printf(...)` 遗留调试代码，造成 FP8 kernel 慢 **30-80x**
- **修复**：删除该 printf 行

#### TMA descriptor 初始化失败
- **原因**：`SmemLayoutVt_TMA` 用 K_SW128（kBlockN 内层），TMA dim0=kBlockN 映射到 total_k（非 stride-1）
- **修复**：改为 MN_SW128（kHeadDim 内层），dim0=kHeadDim → d（stride-1）

#### LDSM_N 编译报错（MN_SW128 不兼容）
- **原因**：`SM75_U32x4_LDSM_N` 要求 K-inner（K_SW128），MN_SW128 是 N-inner → static_assert 失败
- **修复**：绕过 LDSM_N，改用手动 SMEM transpose + 标准 LDSM_N

#### fp8_gt_cos = 0.13~0.23（GEMM2 数值错误）
- **原因**：`partition_B(MN_SW128 tensor)` 按 N-first 分配寄存器，SM120 QMMA 按 K-first 解释，元素放错寄存器
- **修复**：256 个 math 线程协同将 V^T 从 MN_SW128 SMEM 转写到 K_SW128 SMEM，再用标准 LDSM_N

### 验证结果（2026-04-03）

sweep_accuracy.py（quant_mode=2）：6/6 PASS，fp8_gt_cos ≈ 0.9996~0.9997  
hstu_test.py：**14/14 PASS**

### 性能结果（2026-04-03，bs=4, h=16, d=128，full attention）

| seq  | BF16    | BF16 TFLOPS | FP8 (qm=2) | FP8 TFLOPS | FP8/BF16 |
|------|---------|-------------|------------|------------|----------|
| 512  | 0.032ms | 267         | 0.070ms    | 122        | 2.19x 慢 |
| 1024 | 0.084ms | 408         | 0.174ms    | 197        | 2.07x 慢 |
| 2048 | 0.271ms | 508         | 0.518ms    | 265        | 1.91x 慢 |
| 4096 | 0.919ms | 598         | 1.734ms    | 317        | 1.89x 慢 |

---

## Phase 7：TMA SFB（2026-04-04）

**目标**：将 K 侧 block scale（SFB）从 GMEM 标量加载升级为 TMA，与 K+V^T 在同一 mbarrier 批次内一起发射。

### 实现内容

- **SmemLayoutSFB_TMA_t**：`Layout<[kBlockN, 1], [1, kBlockN]>`（kBlockN 个 int32，512 字节）
- **双缓冲 SFB**：`smem_sfb_ptr[2]`，与 K/V^T 共用同一 `tma_mbar[s]`
- **expect_tx**：每 stage 加上 `kSmemSFBBytes`（128×4=512）

### 关键 Bug：SMEM 128B 对齐（Is_arbitrary=true 崩溃）

- **现象**：Is_arbitrary=true 时 `CUDA error: misaligned address`
- **根因**：`kSmemWsDataSizePadded` 仅填充到 8B 对齐，Is_arbitrary=true 时 `kSmemWsFuncEnd=67604` → 填充到 67608，但 `67608 mod 128 ≠ 0`，违反 TMA 目标地址 128B 对齐要求
- **修复**：`kernel_traits.h` 将填充改为 128B：`((kSmemWsFuncEnd + 127) / 128) * 128`
- **定位**：compute-sanitizer memcheck

### 验证结果（2026-04-04）

hstu_test.py：**14/14 PASS**；benchmark 日志：`2benchmark_results/006_709899ce_gpu2430MHz_phase7_tma_sfb.log`

---

## Phase 8：TMA SFV（2026-04-04）

**目标**：将 V 侧 block scale（SFV）从 math warp 标量 GMEM 读取升级为 load warp 的 TMA。

### 实现内容

- **Python 侧**：`sf_v_packed` 通过 `.repeat_interleave(kBlockN, dim=1)` 扩展，使 TMA 可用相同 tile 索引
- **SmemLayoutSFV_TMA_t**：与 SFB 相同；双缓冲 `smem_sfv_ptr[2]`
- **expect_tx**：每 stage 加上 `kSmemSFVBytes`（kBlockN×4=512B）

### 关键 Bug：TMA descriptor globalDim < boxDim

- **现象**：`Error: Failed to initialize the TMA descriptor 1`
- **根因**：`sweep_accuracy.py` 未做 `repeat_interleave`，`sf_v_packed.size(1) = 1 < boxDim[0]=128`
- **修复**：在 `sweep_accuracy.py` 和 `bench_hstu_attn_sm120.py` 加 `repeat_interleave(128, dim=1)`

### 验证结果（2026-04-04）

sweep_accuracy.py（quant_mode=2）：6/6 PASS，fp8_gt_cos ≈ 0.9996~0.9997  
hstu_test.py：**14/14 PASS**；benchmark 日志：`2benchmark_results/007_e10772bc_gpu2347MHz_phase8_tma_sfv.log`

---

## Phase 9：12-warp 结构 + setmaxnreg 216（2026-04-08）

**目标**：通过 `setmaxnreg` 动态重分配寄存器，让 math warp 获得 216 个寄存器，减少 register spill。

### Warp 结构

```
warp 0-7  (256线程) = math warps（WG0 + WG1）→ setmaxnreg.inc 216
warp 8-11 (128线程) = load warpgroup（WG2）   → setmaxnreg.dec 64
  warp 8：active load warp（负责发射 TMA）
  warp 9-11：idle（保持 WG2 完整，使 WARPSYNC.ALL 能完成）
```

### 关键设计原因

`setmaxnreg.inc` 在 SASS 层是 `USETMAXREG.TRY_ALLOC.CTAPOOL`，retry loop 内含 `WARPSYNC.ALL`，要求同一 warpgroup 内所有 4 个 warp 同时到达。WG2 不完整会导致 deadlock → 必须 12 warp（3 个完整 warpgroup）。

### 验证结果（2026-04-08）

sweep_accuracy.py：全部 PASS，cos_sim ≈ 0.9985~0.9987  
hstu_test.py：**14/14 PASS**；benchmark 日志：`2benchmark_results/022_147b3416_gpu2212MHz_setmaxnreg_216.log`

### 性能结果（2026-04-08，gpu=2212MHz）

| seq  | BF16 TFLOPS | FP8 TFLOPS | FP8/BF16 |
|------|-------------|------------|----------|
| 512  | 197         | 127        | 1.55x 慢 |
| 1024 | 294         | 188        | 1.56x 慢 |
| 2048 | 313         | 206        | 1.52x 慢 |
| 4096 | 359         | 237        | 1.52x 慢 |

与 Phase 8 对比，FP8 TFLOPS 提升 +27~51%（含频率修正）。

---

## Phase 10：消除 SMEM Transpose（col-major V，2026-04-13）

**目标**：Python 侧将 V 改为列主序 + kernel 侧 `SmemLayoutVt_TMA` 改为 K_SW128，TMA 直接写入 GEMM2 兼容的 SMEM layout，彻底消除 SMEM→SMEM transpose（2×`bar.sync 1,256` + 16KB copy）。

### 关键原理

| layout | SMEM fast axis | GMEM 要求 |
|--------|---------------|-----------|
| MN_SW128（旧） | kHeadDim | V d 轴 stride-1（行主序）|
| K_SW128（新） | kBlockN | V token 轴 stride-1（列主序）|

Python 侧 `v_fp8.permute(2,1,0).contiguous().permute(2,1,0)` → shape `[n,h,d]` strides `[1, n, h*n]`，token 轴 stride-1。

### 改动文件

- `kernel_traits.h`：`SmemLayoutVt_TMA` 改为 `SmemLayoutAtomSW128{}`（K_SW128）
- `hstu_fwd_kernel_fp8_ws.h`：删除 SMEM transpose，s2r V^T 直接从 `sVt_cur`（K_SW128）读取
- `hstu.h`：新增 `v_d_stride` 字段
- `hstu_ops_gpu.cpp`：`v_d_stride = v.stride(-1)`, `v_row_stride = v.stride(-3)`
- `hstu_fwd_kernel.h`：TMA descriptor 从硬编码 `_1{}` 改为 `params.v_d_stride`
- Python 4 处加 `v_fp8 = v_fp8.permute(2,1,0).contiguous().permute(2,1,0)`

### 验证结果（2026-04-13）

sweep_accuracy.py：6/6 PASS，fp8_gt_cos ≈ 0.9985~0.9987  
hstu_test.py：**14/14 PASS**；benchmark 日志：`2benchmark_results/029_1e241bfff_py_v_transpose.log`

### 性能结果

| seq  | BF16 TFLOPS | FP8 TFLOPS | FP8/BF16 |
|------|-------------|------------|----------|
| 512  | 191         | 180        | -5.9%    |
| 1024 | 287         | 281        | -1.9%    |
| 2048 | 306         | 310        | **+1.4%** |
| 4096 | 357         | 370        | **+3.5%** |

与 Phase 9 对比：FP8 TFLOPS 提升 +42~56%。消除 SMEM transpose 是迄今最大的单项性能提升。

---

## Phase 11：PTX LDSM_T + MN_SW128 Swizzle 优化（2026-04-16）

**目标**：消除协作 SMEM 转置开销（2×`bar.sync 1,256` + 16KB copy loop）。

### 方案 A：`make_tiled_copy_B` + `SM100_U8x16_LDSM_T`
- **结果**：编译失败（`copy_traits.hpp:128: static_assert: src failed to vectorize`）
- **根因**：MN_SW128 layout 通过 `partition_S` 的 per-thread source slice 不满足 uint128_t 向量化条件

### 方案 B：K_SW128 重解释 `sVt_cur`
- **结果**：编译通过，fp8_gt_cos ≈ 0.02
- **根因**：MN_SW128 和 K_SW128 物理地址对同一逻辑索引不同，指针重解释读错数据

### 方案 C：直接内联 PTX LDSM_T
- **结果**：编译通过，运行时 "misaligned address"
- **根因**：`ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8` 是 **SM100A 专属指令**，SM120 ISA 不存在此指令

### 技术结论

- `ldmatrix.m16n16.x2.trans.b8`：SM100A ISA，**SM120 不支持**
- MN_SW128 → K_SW128 重解释：128×128 tile 物理地址不同，不可直接重解释

### 最终实现（fp8_gt_cos 0.12 → 0.9996）

使用手算 swizzle 地址的直接 SMEM 寻址，修复 fragment shape 错误：
- 错误假设：shape = (2, 16, 4) = 128 uint32/thread
- 正确：shape = (2, 4, 4) = 32 uint32/thread，4 个 N-warp 各持 32 N 列
- base 公式：`(ni & 1) + 8*(ni >> 1)`

MN_SW128 swizzle 优化：`SmemLayoutVt_TMA` 改为 `tile_to_shape(Layout_MN_SW128_Atom, [kHeadDim, kBlockN])`；LDSM_T 地址加 `swizzle_xor = (n_off & 7) << 4`，bank conflict 从 16-way 降至 4-way。

### 验证结果（2026-04-16）

sweep_accuracy.py：6/6 PASS，fp8_gt_cos ≈ 0.9996  
hstu_test.py：**14/14 PASS**；benchmark 日志：`2benchmark_results/034_89991df1c_ldsmt_w_swz.log`

### 性能结果（bs=4, h=16, d=128）

| seq  | BF16 TFLOPS | FP8 TFLOPS | FP8/BF16 |
|------|-------------|------------|----------|
| 512  | 192 / 251 (causal) | 176 / 243 | -8.4% / -2.9% |
| 1024 | 284 / 419 | 293 / 433 | **+3.0% / +3.4%** |
| 2048 | 326 / 532 | 340 / 599 | **+4.5% / +12.6%** |
| 4096 | 361 / 628 | 387 / 724 | **+7.1% / +15.2%** |

seq≥1024 FP8 全面超越 BF16；seq=4096 causal 领先 15.2%。

---

## Phase 12：代码整洁化（2026-04-17）

**目标**：消除历史代码腐化，使 `hstu_blackwell_sm120/` 自包含可维护。

### 问题清单

| # | 问题 | 影响 |
|---|------|------|
| 1 | `hstu_fwd_kernel.h:60` 绝对路径 include 6KD 目录 | 代码不可移植 |
| 2 | `run_hstu_fwd_sm120_fp8_tma_impl` 死代码（Phase 5 遗留） | 误导阅读者 |
| 3 | cute include 散落多处 | include 顺序隐式依赖 |
| 4 | `kernel_traits.h` 混用 SM80/SM89/SM120 三套 MMA，SM89 对 WS 路径是死代码 | 歧义 |

### 实施结果

commit `0aa00eca Refactor code`：
- `sm120_qmma_builder.h` 新建 296 行（SM120BlockScaledBuilder 最小子集本地化）
- `hstu_fwd_kernel.h` 净删除 437 行（死代码清除）
- `kernel_traits.h` 精简 56 行（SM89 废弃定义清理）

### 关键发现

- **WS FP8 路径已在用 SM120 原生 QMMA**：WS 内核全程直接用 `BS1/BS2::TiledMma`（SM120 QMMA），`Kernel_traits::MMA_Atom_Arch`（SM89）对 WS 路径是纯死代码
- **SM89 trait 保留**：Has_rab=true 的非 WS 回退路径仍使用 `Hstu_fwd_kernel_traits_sm120_fp8::TiledMma`（SM89）

验证：sweep_accuracy.py 6/6 PASS + hstu_test.py 14/14 PASS
