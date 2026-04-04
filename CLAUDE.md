# HSTU FBGEMM 项目说明

## 项目概述

本项目是 FBGEMM 中 HSTU（Hierarchical Sequential Transduction Unit）注意力机制的 CUDA 实现，upstream 代码已支持 Hopper（SM90），本项目目标是让 Blackwell（SM120）GPU 架构支持 HSTU 注意力，包含 BF16 和 FP8 block-scale 量化模式。

---

## 当前进度

- [x] **Phase 0**：BF16 路径完整实现，`sweep_accuracy.py` quant_mode=-1 通过
- [x] **Phase 1**：FP8 block-scale `else{}` 框架 + cp.async + unit SF + make_zip_tensor → 编译通过
- [x] **Phase 2**：cp.async 升级为 TMA（已合并入 Phase 3 调试过程中完成）
- [x] **Phase 3**：数值验证通过（fp8_gt_cos ≈ 0.9996，远超 0.95 阈值）
- [x] **Phase 4**：性能优化（K+V double-prefetch pipeline，消除迭代中途 blocking V wait）
- [x] **Phase 5**：TMA K+V^T（消除 kernel-side 逐元素 V 转置，K 从 cp.async 升级为 TMA）
- [x] **Phase 6**：Warp-specialized kernel（load warp 专职 TMA，math warps 专职 MMA，producer/consumer pipeline）— 数值验证通过（2026-04-03）
- [x] **Phase 7**：TMA SFB（Scale Factor B via TMA，load warp 每 tile 与 K+V^T 同批 TMA）— 14/14 全通过（2026-04-04）

### Phase 4 最终性能结果（RTX PRO 6000 Blackwell SM120，2026-03-27）

kernel-only（bs=4, h=16, d=128, full attention）：

| seq  | BF16      | FP8 (qm=2) | 差距  |
|------|-----------|------------|-------|
| 512  | 0.048ms 181 TFLOPS | 0.057ms 151 TFLOPS | -16% |
| 1024 | 0.126ms 273 TFLOPS | 0.154ms 224 TFLOPS | -18% |
| 2048 | 0.479ms 287 TFLOPS | 0.582ms 236 TFLOPS | -18% |

FP8 仍比 BF16 慢的原因：SM120 QMMA block-scale 指令有 scale factor 加载开销 + V^T transpose 额外工作 + kernel compute-bound。

---

## Phase 5 完成：TMA K+V^T（2026-03-30）

**目标**：用 TMA 直接加载转置后的 V^T，消除 kernel 内逐元素 V 转置循环；同时将 K 加载从 cp.async 升级为 TMA。

### 实现内容

- **V^T TMA**：V 以转置视图 `[d, total_v, h_k]`、strides `[1, v_row_stride, v_head_stride]` 描述给 TMA，直接将 `[kHeadDim, kBlockN]` tile 加载到 `SmemLayoutVt_TMA`（SW128），消除原来的逐元素转置循环（Phase 4 中约 64 SMEM ops/thread/tile）
- **K TMA**：K 加载从 cp.async 升级为 TMA，使用同一 `uint64_t` mbarrier 与 V^T 共同管理
- **新模板参数** `bool Use_TMA_KV`：FP8 路径走 TMA，BF16 路径维持 cp.async
- **新参数结构** `Hstu_fwd_params_fp8_tma<TMA_K_t, TMA_Vt_t>`：携带 TMA 描述符
- **新内核** `hstu_fwd_kernel_sm120_fp8_tma`

### 关键技术要点

- `with()` 需要 `uint64_t&`（不是 `ClusterTransactionBarrier`），mbarrier 操作全部用 PTX inline
- `local_tile` 须用 `make_coord(_, _)` 取全部 tile，在 `copy` 时以 `nb_abs` 索引（避免 ArithmeticTuple 错误）

### 性能结果（RTX PRO 6000 Blackwell SM120，2026-03-30）

kernel-only（bs=4, h=16, d=128, full attention）：

| seq  | BF16    | FP8 (qm=2) | 差距   |
|------|---------|------------|--------|
| 512  | 0.044ms | 0.053ms    | -20.5% |
| 1024 | 0.119ms | 0.144ms    | -21.0% |
| 2048 | 0.447ms | 0.544ms    | -21.7% |
| 4096 | 1.551ms | 1.912ms    | -23.3% |

注：Phase 5 与 Phase 4 相比绝对延迟下降 ~6-7%，但 BF16 路径未改动也有相近降幅，判断该差异在测量噪声范围内，**实质性能收益待 warp-specialized（Phase 6）验证**。

---

## Phase 6 完成：Warp-Specialized FP8 Kernel with Double-Buffer（2026-04-03）

**目标**：warp 0 专职发 TMA（K+V^T），warp 1-8 专职 QMMA，2-stage double-buffer 实现 TMA 与 MMA 真正重叠。

**当前状态**：double-buffer overlap 已实现，数值正确，14/14 测试通过。

### 实现内容

- **新文件** `hstu_fwd_kernel_fp8_ws.h`：WS 内核主体，由 `hstu_fwd_kernel.h` include
- **Warp 分工**：warp 0 = load warp（288 线程总计，32 load + 256 math）
- **两对 barrier（4个 uint64_t）**：
  - `tma_mbar[2]`：TMA 完成信号，math warps **直接**等待（无 load_mbar 中间层）
  - `math_mbar[2]`：SMEM 消费完成信号，load warp 等待
- **Double-buffer 时序**（真正 overlap）：
  ```
  Load:   [issue TMA N-1→s0] [issue TMA N-2→s1] [等math_mbar[0]] [issue TMA N-3→s0] ...
  Math:   [wait tma_mbar[s0]] [GEMM tile N-1] [signal math_mbar[0]] [wait tma_mbar[s1]] ...
  TMA:    ---N-1 in flight---  ---N-2 与 N-1 计算重叠---  ---N-3 与 N-2 计算重叠---
  ```
  load warp 在 preamble 预发两个 tile（s0、s1），math warp 直接等 tma_mbar，TMA 与 MMA 真正并行。
- **dispatch**：FP8 + kBlockN%128==0 + !Has_rab → WS TMA 路径；Has_rab → Phase 5 非 WS 路径

---

### ⚠️ 已知性能问题：SMEM→SMEM 手动 transpose（待优化）

**当前 V^T 数据路径**：
```
GMEM V → TMA → MN_SW128 SMEM → [256线程手动transpose] → K_SW128 SMEM → LDSM_N → 寄存器
```

**为什么需要手动 transpose**：TMA 和 LDSM_N 对 SMEM layout 的要求互相冲突：
- TMA 要求：SMEM 内层维 = GMEM stride-1 维。GMEM 中 V^T 的 stride-1 维是 d=kHeadDim，因此 SMEM 内层必须是 kHeadDim → 只能用 **MN_SW128**
- LDSM_N 要求：SMEM 内层维 = K-dim（kBlockN）→ 只能用 **K_SW128**
- 两者不兼容，中间必须加一次 transpose

**transpose 开销**：每次 N-block 迭代中，256 个 math warp 线程各搬 64 字节（共 16KB），加 2 次 `bar.sync 1, 256`。此部分开销未被 double-buffer overlap 覆盖，是 FP8 仍比 BF16 慢的主要原因之一。

**消除 transpose 的方案**：Python 侧将 V 改为列主序存储，strides 从 `[v_row_stride, 1]` 改为 `[1, v_col_stride]`，使 kBlockN 维成为 stride-1，则 TMA 可直接写入 K_SW128 SMEM，无需 transpose。代价是修改量化+存储 API。

---

### 关键 Bug 修复记录

#### CuTe SM90 TMA header debug printf（严重性能问题）
- **文件**：`external/cutlass/include/cute/arch/copy_sm90_tma.hpp:172`
- **问题**：`SM90_TMA_LOAD_3D::copy()` 在 `CUTE_ARCH_TMA_SM120_ENABLED` 路径下有一行 `printf(...)` 遗留调试代码，每次 TMA 调用执行一次，造成 FP8 kernel 慢 **30-80x**
- **修复**：删除该 printf 行（已提交）

#### 关键调试难点记录

#### 1. TMA descriptor 初始化失败
- **原因**：`SmemLayoutVt_TMA` 用 K_SW128（kBlockN 内层），TMA dim0=kBlockN 映射到 total_k（非 stride-1）
- **修复**：改为 MN_SW128（kHeadDim 内层），dim0=kHeadDim → d（stride-1）✓

#### 2. LDSM_N 编译报错（MN_SW128 不兼容）
- **原因**：`SM75_U32x4_LDSM_N` 要求 K-inner（K_SW128），MN_SW128 是 N-inner → static_assert 失败
- **修复**：绕过 LDSM_N，改用手动 SMEM transpose + 标准 LDSM_N（见性能问题一）

#### 3. fp8_gt_cos = 0.13~0.23（GEMM2 数值错误）
- **原因**：`partition_B(MN_SW128 tensor)` 内部 `make_ordered_layout` 按 N-first 顺序将元素分配到寄存器槽位，而 SM120 QMMA block-scale 指令按 K-first 顺序解释寄存器（与 K_SW128 的排列一致），导致 V^T 元素放错寄存器位置
- **修复**：s2r P 完成后 smem_q 空闲，256 个 math 线程协同将 V^T 从 MN_SW128 SMEM 转写到 K_SW128 SMEM（smem_q），再用标准 LDSM_N s2r，寄存器排列恢复正确

#### 4. Is_arbitrary=true 时 CUDA misaligned address（Phase 7 TMA SFB）
- **现象**：Is_arbitrary=false（cases 1-6）正常，Is_arbitrary=true（case 7，seq=128 arbitrary masking）在 TMA SFA copy 处崩溃
- **根因**：`kSmemWsDataSizePadded` 仅填充到 8B 对齐。Is_arbitrary=true 时 `kSmemWsFuncEnd=67604`，填充到 8B 得 67608，但 `67608 mod 16 = 8`，不满足 TMA 要求的 128B 目标地址对齐。Is_arbitrary=false 时 `kSmemWsFuncEnd=65536`（已 128B 对齐）故不触发
- **修复**：`kernel_traits.h` 将 `((kSmemWsFuncEnd + 7) / 8) * 8` 改为 `((kSmemWsFuncEnd + 127) / 128) * 128`
- **定位**：compute-sanitizer memcheck 在 `hstu_fwd_kernel_fp8_ws.h:344` 精确报告 "Misaligned shared or local address"

---

### 验证结果（2026-04-03）

**sweep_accuracy.py**（quant_mode=2）：

| D | H | SEQ | fp8_gt_cos |
|---|---|-----|------------|
| 128 | 1 | 128 | 0.9996 ✓ |
| 128 | 1 | 256 | 0.9997 ✓ |
| 128 | 1 | 512 | 0.9997 ✓ |
| 128 | 4 | 128 | 0.9996 ✓ |
| 128 | 4 | 256 | 0.9996 ✓ |
| 128 | 4 | 512 | 0.9996 ✓ |

**hstu_test.py 14 explicit @example cases**（quant_mode=2，seq=128/256，causal/rab/drab/local/context/target/arbitrary）：**14/14 PASS**

测试脚本：`run_hstu8_examples.sh`（项目根目录）
测试日志：`test_results/011_ws_phase6_final.log`（Phase 6），`test_results/021_hstu8_examples_qm2.log`（Phase 7）

### Phase 6 性能结果（2026-04-03，RTX PRO 6000 Blackwell SM120，bs=4, h=16, d=128）

kernel-only（full attention，CuTe debug printf 已删除后测量）：

| seq  | BF16    | BF16 TFLOPS | FP8 (qm=2) | FP8 TFLOPS | FP8/BF16 |
|------|---------|-------------|------------|------------|----------|
| 512  | 0.032ms | 267         | 0.070ms    | 122        | 2.19x 慢 |
| 1024 | 0.084ms | 408         | 0.174ms    | 197        | 2.07x 慢 |
| 2048 | 0.271ms | 508         | 0.518ms    | 265        | 1.91x 慢 |
| 4096 | 0.919ms | 598         | 1.734ms    | 317        | 1.89x 慢 |

**注意**：BF16 绝对值与历史数据差异较大（历史 0.044ms vs 今日 0.032ms at seq=512），原因是 GPU 热状态/boost clock 不同，跨 session 绝对值不可直接比较。

FP8 仍比 BF16 慢的主要原因：SMEM transpose 开销（每 tile 16KB + 2×bar.sync）+ WS 分工损失 1/9 计算资源 + QMMA block-scale 指令本身更重。

### 后续优化方向
- [ ] **消除 SMEM transpose**：Python 侧 V 列主序存储，TMA 直接写 K_SW128，预计节省每 tile 约 2×bar.sync + 16KB 搬运
- [ ] **nsys profile**：量化各部分耗时（GEMM1/GEMM2/transpose/SFB load），确定实际瓶颈

---

## Phase 7 完成：TMA SFB（Scale Factor B via TMA，2026-04-04）

**目标**：将 K 侧 block scale（SFB）从 GMEM 标量加载升级为 TMA，与 K+V^T TMA 在同一 mbarrier 批次内一起发射，消除逐元素 GMEM 读取开销。

### 实现内容

- **SmemLayoutSFB_TMA_t**：`Layout<[kBlockN, 1], [1, kBlockN]>`，对应 `kBlockN` 个 int32（128×4=512 字节）
- **TMA 描述符**：`sf_k_packed [H, SEQ]` 以 `[kv_block_descale_head_stride, 1, h_k]` 展开，tile `[kBlockN, 1]`，单 int32 swizzle
- **双缓冲 SFB**：`smem_sfb_ptr[2]`，与 K/V^T 共用同一 `tma_mbar[s]` 管理；load warp preamble 预发两个 tile，主循环轮转
- **expect_tx**：每 stage 的 `expect_tx` 加上 `kSmemSFBBytes`（128×4=512），mbarrier 计数正确

### 关键 Bug 修复：SMEM 128B 对齐（Is_arbitrary=true 崩溃）

- **现象**：Is_arbitrary=false（cases 1-6）正常，Is_arbitrary=true（case 7）出现 `CUDA error: misaligned address`
- **根因**：`kSmemWsDataSizePadded = ((kSmemWsFuncEnd + 7) / 8) * 8`，对 Is_arbitrary=true：
  - `kSmemWsFuncEnd = 65536 + 2048 + 20 = 67604`
  - 填充到 8B → `67608`
  - `67608 mod 16 = 8` → **TMA 要求目标 SMEM 地址 128B 对齐，67608 不满足**
  - Is_arbitrary=false 时 `kSmemWsFuncEnd = 65536`（已 128B 对齐），所以不触发
- **修复**：`kernel_traits.h` 中将填充改为 128B：`((kSmemWsFuncEnd + 127) / 128) * 128`
- **定位方式**：compute-sanitizer memcheck，在 `hstu_fwd_kernel_fp8_ws.h:344` 处精确报告 misaligned shared address

### 验证结果（2026-04-04）

**hstu_test.py 14 explicit @example cases**（quant_mode=2，seq=128/256，causal/rab/drab/local/context/target/arbitrary）：**14/14 PASS**

测试日志：`test_results/021_hstu8_examples_qm2.log`

---

## 核心文件

### SM120 原生内核（当前主目录）
`fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/`
- `hstu_fwd_kernel.h` — 前向内核（BF16 + FP8 非WS路径 + include WS文件）★核心
- `hstu_fwd_kernel_fp8_ws.h` — Phase 6 WS FP8 内核主体（由 hstu_fwd_kernel.h include）★
- `kernel_traits.h` — BF16 + FP8 内核 traits（含 SmemLayoutVt_TMA = MN_SW128）
- `hstu_ops_gpu.cpp` — PyTorch 入口（`hstu_varlen_fwd_120`）
- `hstu.h` — Params 结构体（含 FP8 descale 字段）
- `utils.h` — tile 大小、类型转换、silu 辅助
- `hstu_fwd_launch_template.h` — 启动模板（WS dispatch 逻辑在此）

### FP8 block-scale 参考实现
`6KD_fp8_block_scale/kernels/include/sm120_blockscaled_gemm/`
- `sm120_blockscaled_utils.cuh` — MMA/SMEM 类型定义
- `sm120_blockscaled_gemm_impl.cuh` — GEMM 主实现（TMA + MMA block scale）

---

## 通用技术规则

**数据格式**：

| 张量 | GEMM | K 方向 |
|------|------|--------|
| Q, K | GEMM1 (Q×K^T) | headDim = 128 |
| V    | GEMM2 (P×V)   | kBlockN = 128 |

**FP8 dtype**：Q/K/V 均为 `cute::float_e4m3_t`（torch: `torch.float8_e4m3fn`）

**SF 格式**：每 128 个 K 元素对应 1 个 e8m0 scale，4 个连续块打包为 1 个 int32：
- bits 0-7 = block 0 [K: 0,128)，bits 8-15 = block 1，bits 16-23 = block 2，bits 24-31 = block 3

**FP8 量化模式**：
- `quant_mode=-1`：禁用 FP8（纯 BF16）
- `quant_mode=2`：FP8 block scale（SM120 支持此模式）

**kBlockN 约束**：必须整除 128（launch template 已有 guard）

---

## 当前运行环境

Claude 直接运行在主机（无需 docker exec），通过 bwrap sandbox 隔离，GPU 设备透传已配置。

**重编译命令**（MAX_JOBS=32 加速并行编译）：
```bash
mkdir -p /tmp/claude && \
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu && \
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user \
HSTU_ARCH_LIST="12.0" \
HSTU_DISABLE_BACKWARD=TRUE \
HSTU_DISABLE_DETERMINISTIC=FALSE \
HSTU_DISABLE_HDIM32=TRUE \
HSTU_DISABLE_HDIM64=TRUE \
HSTU_DISABLE_HDIM256=TRUE \
MAX_JOBS=32 \
pip install --no-build-isolation --config-settings editable_mode=compat -e .
```

注：`mkdir -p /tmp/claude` 必须先执行（sandbox 将 TMPDIR 设为该路径，nvcc 编译需要它存在）。

**运行准确度测试**：
```bash
HSTU_SWEEP_FP8_QUANT_MODE=2 python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/sweep_accuracy.py \
  2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/test_results/NNN_xxx.log
```

**测试日志命名规范**：文件名格式 `NNN_<修改内容简述>.log`，NNN 为三位数字顺序编号。

**环境版本**：PyTorch 2.10.0（nvcr.io/nvidia/pytorch:26.01-py3）；CUDA 13.1。

---

## 性能分析流程（每轮优化后必须执行）

每次跑完 benchmark 之后，必须按以下步骤做 nsys profile 分析，找到性能瓶颈并制定下一步优化方向。

**文件命名规范**：所有 benchmark 产物（bench log、nsys trace、stats log）统一存放在 `benchmark_results/` 目录，文件名格式与 `test_results/` 相同：`NNN_<内容简述>.<ext>`，NNN 为三位数字顺序编号。

### 步骤 1：运行 benchmark
```bash
REPO=/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu
python ${REPO}/fbgemm_gpu/experimental/hstu/benchmark/bench_hstu_attn_sm120.py \
  2>&1 | tee ${REPO}/benchmark_results/NNN_bench.log
```

### 步骤 2：nsys profile 抓取 trace
```bash
REPO=/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu
TRACE=${REPO}/benchmark_results/NNN_profile
nsys profile \
  --output ${TRACE} \
  --trace cuda,nvtx \
  --force-overwrite true \
  python ${REPO}/fbgemm_gpu/experimental/hstu/benchmark/profile_hstu_attn.py
```

### 步骤 3：nsys stats 分析 trace
```bash
nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_pushpop_sum \
  --force-export true \
  --format csv \
  ${TRACE}.nsys-rep \
  2>&1 | tee ${REPO}/benchmark_results/NNN_stats.log
```

注：nsys 2025.6.1 的 report 名称为 `cuda_gpu_kern_sum,cuda_api_sum,nvtx_pushpop_sum`（非旧版的 `gputrace,cudaapisum,nvtxsum`）。

### 步骤 4：分析瓶颈并制定优化方向

看 `cuda_gpu_kern_sum` 输出，重点关注：
- **HSTU kernel 耗时占比**：是否 compute-bound 还是 memory-bound
- **kernel 内部 stall**：通过 duration 与理论 FLOP/s 对比判断
- **量化 kernel 耗时**：`quant_mode=2` 时 Python 侧量化 kernel 的开销（出现在 CUDA API 调用中）
- **SM 利用率**：kernel duration × SM count vs 理论峰值

分析完成后，根据瓶颈类型制定下一步优化方向：
- 若 compute-bound → 考虑 instruction-level 优化（减少 SF 加载、register spill）
- 若 memory-bound → 考虑增加 prefetch stage 数、TMA multicast
- 若量化开销主导 → 考虑将量化融合进 kernel

---

## 目录结构

```
fbgemm_gpu/experimental/hstu/
├── benchmark/
│   ├── bench_hstu_attn.py               # 吞吐量基准测试（通用）
│   ├── bench_hstu_attn_sm120.py         # SM120 专用 benchmark ★
│   └── profile_hstu_attn.py             # nsys 性能分析脚本
├── hstu/
│   ├── cuda_hstu_attention.py           # Python 入口，SM 版本分发
│   └── library.py                       # 命名空间包检测（已修复）
├── src/
│   ├── generate_kernels.py              # 生成 Hopper/Blackwell .cu 文件
│   ├── hstu_ampere/                     # Ampere (SM80) 原生内核（upstream）
│   ├── hstu_blackwell/                  # SM100 原生内核（upstream，CuTe-DSL 实现）
│   ├── hstu_blackwell_sm120/            # SM120 最终版本（Phase 4，当前主目录）★
│   │   ├── hstu_fwd_kernel.h            # 前向内核（BF16 + FP8 block-scale + K/V 预取）★
│   │   ├── kernel_traits.h
│   │   ├── hstu_ops_gpu.cpp
│   │   ├── hstu.h
│   │   ├── utils.h
│   │   ├── hstu_fwd_launch_template.h
│   │   └── instantiations/              # 编译单元（generate_kernels.py 生成）
│   └── hstu_hopper/                     # Hopper (SM90) 原生内核（upstream）
├── test/
│   └── hstu_test.py
├── CMakeLists.txt                       # 含 Blackwell 源文件和 SM120 gencode
└── setup.py                             # 含 "12.0" arch 支持

6KD_fp8_block_scale/                     # SM120a FP8 block scale 参考实现
└── kernels/include/sm120_blockscaled_gemm/
    ├── sm120_blockscaled_gemm_impl.cuh  # GEMM 主实现（TMA + block scale MMA）
    └── sm120_blockscaled_utils.cuh      # 类型定义（tile/MMA/SMEM/Barrier）

（项目根目录）
├── sweep_accuracy.py                    # FP8 vs BF16 准确度扫描 ★
├── memory.md                            # Claude 对话记忆（跨会话）
└── test_results/
    └── NNN_*.log                        # 编号调试日志
```

---

## 编译注意事项

编译时可能出现 CuTe static_assert、CUDA 类型不匹配等错误。每次编译须：
1. 仔细观察输出，主动捕获错误（不要只看最后几行）
2. 编译失败后先读相关源文件理解上下文，再修改，不要盲目重试
3. 用如下命令捕获完整错误（ninja 并行编译时错误可能被截断）：
   ```bash
   pip install ... 2>&1 | grep -E "error:|note:|static_assert|undefined" | head -60
   ```

## 运行时调试技巧

### Illegal Memory Access：使用 compute-sanitizer
```bash
HSTU_SWEEP_FP8_QUANT_MODE=2 compute-sanitizer --tool memcheck \
  python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/sweep_accuracy.py \
  2>&1 | tee /tmp/claude/sanitizer_out.log
```
- `compute-sanitizer` 会精确报告出错的 PTX 指令地址、线程 ID 和访问地址
- 常见原因：mbarrier wait phase 错误、SMEM 越界访问、TMA descriptor 地址计算错误

### 数值不正确：GEMM1-only + all-ones 缩小范围

在 kernel 内部通过条件编译或临时硬编码来分步验证：

**Step 1**：将所有输入（Q/K/SFA/SFB）置为全 1，只跑 GEMM1（`acc_s = Q × K^T`），跳过 silu 和 GEMM2，直接将 `acc_s` 输出为结果
- 预期：所有元素 = kHeadDim（128 次 1×1 乘加）
- 若不符：GEMM1 数据路径有问题（SMEM layout、MMA warp partition、TMA 加载错误）

**Step 2**：恢复正常输入但只跑 GEMM1，检查与参考实现（BF16 路径）的 cosine similarity
- 若 GEMM1 结果 cos < 0.99：K 加载、SMEM layout、scale factor 有问题
- 若 GEMM1 结果 cos ≈ 1.0 但最终结果差：问题在 GEMM2（V^T 加载、SMEM layout）

**实现方式**：在 kernel 里加 `if constexpr (kDebugGemm1Only)` 分支，或在 hstu_fwd_launch_template.h 传入临时 debug 模板参数。

---

## 语言要求

**任何时候只用中文回答，不要显示韩文或日文。**

## 知识点整理

**将用户提问中涉及的所有技术知识点（CUDA、CuTe、FP8、MMA、内核优化等）整理到项目根目录的 `knowledge.md` 中。**
- 每次回答技术问题后，将该知识点以结构化方式追加到 `knowledge.md`
- 格式：`## <主题>` + 简明说明 + 关键结论
- 不重复已有条目，可在已有条目上补充

## Memory 持久化

**将所有 memory 保存到本项目根目录下的 `memory.md`**，而非默认的 `~/.claude/` 路径。
每次对话开始时读取 `/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/memory.md` 以恢复上下文。

## Plan 持久化

**将实现计划（plan）写到当前目录的 `PLAN.md`** 中，而非其他位置。

## 文件删除限制（重要）

**严禁执行任何删除文件的命令**，包括但不限于：
- `rm`、`rm -f`、`rm -rf`
- `unlink`、`find ... -delete`、`shutil.rmtree`

如需清理文件，必须先告知用户，等待明确授权后方可执行。
