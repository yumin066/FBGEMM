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
- [ ] **Phase 5**：Warp-specialized kernel（load warp 专职 TMA，math warps 专职 MMA，producer/consumer pipeline）

### Phase 4 最终性能结果（RTX PRO 6000 Blackwell SM120，2026-03-27）

kernel-only（bs=4, h=16, d=128, full attention）：

| seq  | BF16      | FP8 (qm=2) | 差距  |
|------|-----------|------------|-------|
| 512  | 0.048ms 181 TFLOPS | 0.057ms 151 TFLOPS | -16% |
| 1024 | 0.126ms 273 TFLOPS | 0.154ms 224 TFLOPS | -18% |
| 2048 | 0.479ms 287 TFLOPS | 0.582ms 236 TFLOPS | -18% |

FP8 仍比 BF16 慢的原因：SM120 QMMA block-scale 指令有 scale factor 加载开销 + V^T transpose 额外工作 + kernel compute-bound。

---

## Phase 5 任务：前向 Warp-Specialized Kernel（当前目标）

**目标**：将 `hstu_blackwell_sm120/hstu_fwd_kernel.h` 改造为 warp-specialized 前向设计，通过 TMA + producer/consumer pipeline 进一步提升吞吐量。**暂不实现后向。**

### 参考文件（必读）
- **SM100 前向参考**：`src/hstu_blackwell/hstu_fwd.py`
  - 前向 warp 分工：`load_warp_id=9`（TMA issue）、`mma_warp_id=8`（tcgen05 MMA）、`silu0_warp_ids=(0-3)`、`silu1_warp_ids=(4-7)`、`empty_warp_ids=(10,11)`，共 12 warps
  - pipeline barrier：`load_mma_Q/K/V_mbar_ptr` 管理 TMA→MMA 同步；`mma_compute_S_mbar_ptr` 管理 MMA→silu 同步
  - 多 stage pipeline：`kv_stage=4`（FP8）/`kv_stage=3`（BF16），`q_stage=2`
  - TMA 操作：`cpasync.CopyBulkTensorTileG2SOp` + `make_tiled_tma_atom_A/B`

### SM120 与 SM100 的关键差异

| 特性 | SM100（参考） | SM120（目标） |
|------|--------------|--------------|
| Tensor Core 指令 | `tcgen05`（全 CTA 共享 TMEM） | `mma.sync`（per-warp，block-scale QMMA） |
| MMA 调用 | 单 mma_warp 代理全 CTA | 所有 math warps 各自执行 QMMA |
| TMA API | 相同（SM90+） | 相同 |
| SF 处理 | 无（非 block-scale） | SFA/SFB/SFV 需随 K/V tile 预取 |
| V transpose | TMA 预转置 GMEM layout | Phase 4 用逐元素循环，Phase 5 同样用 TMA 预转置 |

### Phase 5 Warp 分工方案（SM120 适配）

SM120 无 TMEM，mma.sync 是 per-warp 的，因此所有 math warps 都要执行 MMA（不能单独指定一个 mma_warp）：

```
warp 0      : load warp（专职 TMA issue：Q/K/V + SF）
warp 1-7    : math warps（QMMA block-scale + silu + softmax）
warp 8-X    : 可选 epilogue / empty warps
```

**线程数**：FP8 路径维持 `BS1::kNumMathThreads=256`（8 warps），load warp 额外 +1 → 共 9 warps = 288 threads（待确认与 `SM120BlockScaledBuilder` 的兼容性）。

### Phase 5 V Transpose 解决方案（关键优化）

**Phase 4 现状**：V 以 `[kBlockN, kHeadDim]` 加载到 SMEM，再用逐元素循环 `sVt_buf(d, k) = sV_curr(k, d)` 手动转置，引入额外 SMEM 读写延迟（`hstu_fwd_kernel.h` 第 1131 行）。

**SM100 的做法**（`hstu_fwd.py` 第 212-213 行）：在创建 TMA 描述符之前，直接对 GMEM 张量做 layout 变换，交换 seq 维和 head 维的步长：
```python
V_layout_transpose = [1, 0, 2]   # 先 KV_layout_transpose 后再交换 dim0/dim1
mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))
# mV 现在是 [headdim, seqlen, batch]，MN-major
tma_atom_V, tma_tensor_V = make_tiled_tma_atom_B(tma_load_op, mV, sV_layout, ...)
```
TMA 直接将 V 的一个 `[kHeadDim, kBlockN]` tile 搬到 SMEM，到达 SMEM 时已是转置后的形状，内核中无需任何额外操作。

**SM120 Phase 5 对应实现**（C++）：
```cpp
// 在 run_hstu_fwd_sm120_impl 中，创建 tma_V 之前：
// tensor_V 原始 shape: [seqlen, headdim]，stride: [headdim, 1]
// 交换 dim0/dim1 的 stride，让 TMA 看到 [headdim, seqlen]
auto tensor_Vt = make_tensor(tensor_V.data(),
    make_layout(make_shape(get<1>(tensor_V.shape()), get<0>(tensor_V.shape())),
                make_stride(get<1>(tensor_V.stride()), get<0>(tensor_V.stride()))));
// sV_smem 布局改为 [kHeadDim, kBlockN]（已转置）
auto tma_V = make_tma_copy(SM90_TMA_LOAD{}, tensor_Vt, smem_layout_Vt, tile_shape_Vt, Int<1>{});
```
GEMM2 的 B operand 直接从 SMEM 读 `[kHeadDim, kBlockN]`，不再需要转置步骤。

### Phase 5 核心实现步骤

1. **TMA 对象创建**（在 `run_hstu_fwd_sm120_impl` 中）：
   - Q、K：标准 `[kBlockM/N, kHeadDim]` layout
   - V：**预转置** GMEM layout（交换 stride dim0/dim1），SMEM 目标 layout 为 `[kHeadDim, kBlockN]`

2. **SharedStorage 扩展**：加入 TMA transaction barrier（`cutlass::arch::ClusterTransactionBarrier`）或 `cutlass::pipeline::NamedBarrier`，阶段数与 K/V stage 对应。

3. **Load warp 职责**：
   ```cpp
   if (warp_idx == load_warp_id) {
       // preamble: 发出 K[0] + V[0] 的 TMA（V 已预转置，直接落 [kHeadDim, kBlockN] SMEM）
       // 主循环: 等待 math warps 消费完毕后，发出下一 tile 的 TMA
   }
   ```

4. **Math warp 职责**：
   ```cpp
   else {
       // 等待 load warp 完成当前 tile 的 TMA
       // GEMM1: QMMA block-scale Q×K → acc_s
       // silu + softmax
       // GEMM2: QMMA block-scale P×Vt（Vt 已在 SMEM，无需转置）
   }
   ```

5. **SF prefetch 与 MMA overlap**：SFA/SFB/SFV 在上一轮 MMA 执行期间由 load warp 预取到 SMEM。

### 验证目标
- 编译通过，`sweep_accuracy.py` fp8_gt_cos > 0.95
- bench 性能优于 Phase 4（预期：消除逐元素 V transpose + GMEM 读延迟气泡）

---

## 核心文件

### SM120 原生内核（当前主目录）
`fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/`
- `hstu_fwd_kernel.h` — 前向内核（BF16 + FP8 block-scale 路径）★核心
- `kernel_traits.h` — BF16 + FP8 内核 traits
- `hstu_ops_gpu.cpp` — PyTorch 入口（`hstu_varlen_fwd_120`）
- `hstu.h` — Params 结构体（含 FP8 descale 字段）
- `utils.h` — tile 大小、类型转换、silu 辅助
- `hstu_fwd_launch_template.h` — 启动模板

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
