# HSTU FBGEMM 项目说明

## 项目概述

本项目是 FBGEMM 中 HSTU（Hierarchical Sequential Transduction Unit）注意力机制的 CUDA 实现，upstream代码已支持 Hopper（SM90），该项目目标是让Blackwell（SM120）GPU 架构也支持HSTU注意力，包含 BF16 和 FP8 量化模式。
需要CLAUDE帮我撰写SM120原生的BF16和FP8 kernel，并在写完kernel之后做性能benchmark和准确度benchmark。

其中，bf16已经正确实现了，在hstu_fwd_kernel.h中。

---

> ## ⚠️ 核心任务：Blockwise-Scale FP8 实现

## 当前进度

- [x] **Phase 0**：BF16 路径完整实现，`sweep_accuracy.py` quant_mode=-1 通过
- [ ] **Phase 1**：FP8 block-scale `else{}` 框架 + cp.async + unit SF + make_zip_tensor → 编译通过
- [ ] **Phase 2**：将 cp.async 升级为 TMA
- [ ] **Phase 3**：数值验证，sweep_accuracy.py quant_mode=2 通过（bf16_fp8_cos > 0.95）
- [ ] **Phase 4**：性能优化（pipelining）

---

## Phase 1 任务：else{} 框架实现（当前目标）

**目标**：让 `else {}` 分支有完整实现并编译通过，能跑 sweep_accuracy.py。

### 参考文件（必读）
- `./6KD_fp8_block_scale/kernels/include/sm120_blockscaled_gemm/sm120_blockscaled_utils.cuh` — MMA/SMEM 类型
- `./6KD_fp8_block_scale/kernels/include/sm120_blockscaled_gemm/sm120_blockscaled_gemm_impl.cuh` — s2r copy 和 gemm 调用范式

### 代码位置
- **实现文件**：`hstu_fwd_kernel.h` 第 688 行的 `else { }` 分支
- **BF16 参考**：同文件 `if constexpr (!Is_fp8)` 分支（结构完全参考）

### Phase 1 实现规则

**GMEM→SMEM（Phase 1 允许 cp.async）**：
- Q、K、V 暂时用 cp.async（与 BF16 路径相同的 `GmemTiledCopyQKV`）
- 注意：SMEM 布局必须用 **SW128**（`BS::SmemLayoutAtomA`），不能用 FP8 kernel traits 里的 flat Layout<_16,_32>
- Phase 3 再换成 TMA

**SF（scale factor）处理**：
- Phase 1 在 SMEM 里直接填 `0x7f7f7f7f`（e8m0=1.0，单位 scale），不从 GMEM 加载
- 因此 Phase 1 结果 ≈ 普通 FP8 MMA（无额外 quantization），准确度应与 BF16 接近

**SMEM 布局**：
- 必须用 `SM120BlockScaledBuilder` 的 `SmemLayoutAtomA/B`（SW128），不用 `Hstu_fwd_kernel_traits_sm120_fp8` 里的 flat layout
- 使用 `as_position_independent_swizzle_tensor(sX)` 包装后再做 s2r copy

**MMA（强制，Phase 1 就要正确）**：
```cpp
using BS1 = sm120_blockscaled_gemm::SM120BlockScaledBuilder<kBlockM, kBlockN, 4>;   // GEMM1 Q×K
using BS2 = sm120_blockscaled_gemm::SM120BlockScaledBuilder<kBlockM, kHeadDim, 4>;  // GEMM2 P×V
// 调用必须用 make_zip_tensor：
cute::gemm(tiled_mma_g1, make_zip_tensor(tCrQ, tCrSFA), make_zip_tensor(tCrK, tCrSFB), acc_s);
cute::gemm(tiled_mma_g2, make_zip_tensor(tCrP, tCrSFP), make_zip_tensor(tCrV, tCrSFV), acc_o);
```

**SMEM 大小**：
- `Hstu_fwd_kernel_traits_sm120_fp8::kSmemSize` 需加上 SF 空间（SFA+SFB 各 512 bytes = 1024 bytes 额外）
- 修改 `kernel_traits.h` FP8 struct 的 `kSmemSize`

**TORCH_CHECK 放宽**：
- `hstu_ops_gpu.cpp` 第 334-337 行的 `cu_seqlens_q_block_descale` 强制检查必须删除，改为可选

**Phase 1 数据流概览**：
```
Q/K/V FP8 GMEM ──cp.async──► SW128 SMEM
SF SMEM ◄── 直接填 0x7f7f7f7f（unit scale）
SW128 SMEM ──ldmatrix──► registers
cute::gemm(mma_bs, zip(Q,SFA), zip(K,SFB), acc_s)  ← GEMM1
mask + silu(acc_s * alpha)
acc_s → FP8 rP → SW128 SMEM roundtrip
cute::gemm(mma_bs, zip(P,SFP), zip(V,SFV), acc_o)  ← GEMM2
acc_o / scaling_seqlen → BF16 → GMEM
```

### Phase 1 关键技术要点

**SW128 SMEM 与 Q/K/V 形状**（headDim=128, kBlockM=128, kBlockN=128）：
- sQ（SW128）：[kBlockM=128, kHeadDim=128, 1 stage] = 16384 bytes
- sK（SW128）：[kBlockN=128, kHeadDim=128, 1 stage] = 16384 bytes（Share_Q_K_smem=true 时与 sQ 共享）
- sV（SW128）：[kBlockN=128, kHeadDim=128, 1 stage] = 16384 bytes（偏移 16384 bytes）
- sSFA：[kBlockM=128, 1, 1] int32 = 512 bytes（偏移 32768 bytes）
- sSFB：[kBlockN=128, 1, 1] int32 = 512 bytes（偏移 33280 bytes）

**GEMM1 s2r copy 方式**（参考 `sm120_blockscaled_gemm_impl.cuh` 第 297-326 行）：
```cpp
// A (Q)
auto s2r_copy_A = make_tiled_copy_A(typename BS1::SmemCopyAtomA{}, tiled_mma_g1);
auto tXsA = s2r_thr_copy_A.partition_S(sQ_sw128);   // (CPY,CPY_M,CPY_K,PIPE)
auto tCrA = thr_mma_g1.partition_fragment_A(sQ_sw128(_,_,_0{}));
auto tXrA = s2r_thr_copy_A.retile_D(tCrA);
cute::copy(s2r_copy_A, tXsA(_,_,_,_0{}), tXrA);

// SF A → transform_fragment_for_qmma → tCrSFA_frg(_,_,_,_0{}) 作为 stage 0
auto tCrSFA = BS1::partition_fragment_SFA(sSFA(_,_,_0{}), thr_mma_g1);
auto tCrSFA_frg = BS1::transform_fragment_for_qmma(tCrSFA);
// s2r copy for SF:
auto s2r_copy_SFA = make_tiled_copy_impl(typename BS1::SmemCopyAtomSF{},
    BS1::get_layoutSFA_TV(tiled_mma_g1), make_shape(size<0>(tile_shape(tiled_mma_g1)), _1{}));
auto tXsSFA = s2r_thr_copy_SFA.partition_S(sSFA);
cute::copy(s2r_copy_SFA, tXsSFA(_,_,_,_0{}), s2r_thr_copy_SFA.retile_D(tCrSFA));
```

**GEMM2 P SMEM roundtrip**：
- float acc_s 经 silu 后转 FP8 → 写入 sQ_sw128（复用，因 Share_Q_K_smem）
- `make_tiled_copy_C(AutoVectorizingCopy, tiled_mma_g1)` 写回 SMEM
- 再用 ldmatrix 读到 tCrP 寄存器
- P 的 SFA 填 0x7f7f7f7f（P 不量化，scale=1）

**线程数**：`BS1::kNumMathThreads = 256`（与 `Hstu_fwd_kernel_traits_sm120_fp8::kNThreads` 必须一致）
- FP8 kNWarps=8 → kNThreads=256，刚好匹配 `SM120BlockScaledBuilder::kNumMathThreads=256` ✓

### Phase 1 编译验证步骤
1. 修改 `kernel_traits.h`：kSmemSize 加 1024
2. 修改 `hstu_ops_gpu.cpp`：删除 cu_seqlens_q_block_descale TORCH_CHECK
3. 实现 `else {}` 块
4. 编译：`cd .../hstu && HSTU_ARCH_LIST="12.0" HSTU_DISABLE_HDIM64=TRUE MAX_JOBS=32 pip install --no-build-isolation --config-settings editable_mode=compat -e . 2>&1 | grep -E "error:|static_assert" | head -40`
5. 测试：`python sweep_accuracy.py 2>&1 | tee test_results/001_phase1_cp_async_unit_sf.log`

---

## Phase 2 任务：cp.async → TMA（Phase 1 通过后执行）

**目标**：将 Q/K/V GMEM→SMEM 从 cp.async 改成 TMA（`make_tma_copy` + `ClusterTransactionBarrier`）。

**方案**：在 `run_hstu_fwd_sm120_impl` 里（当 Is_fp8=true 时）用 `make_tma_copy(SM90_TMA_LOAD{}, tensor_Q, smem_layout_sw128)` 创建 TMA 对象，通过扩展 params struct 传给新 kernel。

**注意**：Phase 2 不需要修改数据格式或 MMA 部分，只换搬运方式。

---

## Phase 3 任务：数值调试（Phase 2 通过后执行）

**目标**：`sweep_accuracy.py` 中所有配置的 `bf16_fp8_cos > 0.95`。

**调试流程**：
1. 先跑 ones 输入（Q=K=V=1），验证 GEMM1 输出数值
2. 再跑 randn 输入
3. cos_sim 低时：添加 `printf("[HSTU_KDBG] ...")` 对比中间值（条件：`tidx==0 && bidb==0 && bidh==0 && m_block==0`）

---

## Phase 4 任务：性能优化（Phase 3 通过后执行）

**目标**：在 `/home/minyu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/hstu_fwd_kernel.h` 中实现 TMA + producer/consumer pipeline，参考 SM100 CuTe-DSL 实现的流程。

### 参考文件（必读）
- **SM100 CuTe-DSL 参考**：`/home/minyu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu/src/hstu_blackwell/hstu_fwd.py`
  - 实现了 TMA load（`tma_atom_Q/K/V`）+ WGMMA producer/consumer pipeline
  - 关键结构：load warp（`load_warp_id=9`）负责 TMA issue，math warps（silu0/silu1/mma）负责 compute
  - 多 stage pipeline：`kv_stage=4`（FP8）/ `kv_stage=3`（BF16），`q_stage=2`
  - barrier 机制：`mbar_ptr`（NamedBarrier）管理 producer/consumer 同步
  - SharedStorage 布局：`sQ`、`sK`（复用 sV）、`sO`、validity barriers
- **SM120 block-scale GEMM 参考**：`6KD_fp8_block_scale/kernels/include/sm120_blockscaled_gemm/sm120_blockscaled_gemm_impl.cuh` — producer/consumer 模式与 SF prefetch

### 实现目标文件
- `src/hstu_blackwell_sm120/hstu_fwd_kernel.h`（新目录，对应 SM120 原生 TMA 版本）

### Phase 4 核心任务
1. **TMA 搬运**：将 Q/K/V GMEM→SMEM 从 cp.async 改为 TMA（`make_tma_copy` + `ClusterTransactionBarrier`），参考 `hstu_fwd.py` 的 `tma_atom_Q/K/V` 初始化及 `tma_tensor_Q/K/V` 用法
2. **Producer/Consumer 分离**：load warp 专职 TMA issue（类似 `hstu_fwd.py` 的 `load_warp_id=9`），math warps 专职 MMA+silu（类似 `silu0_warp_ids`/`mma_warp_id`）
3. **多 stage pipeline**：K/V 多 stage 双缓冲 prefetch（对应 `hstu_fwd.py` 的 `kv_stage=4`）
4. **SF prefetch 与 MMA overlap**：SFA/SFB/SFV scale factor 与 GEMM 计算重叠
5. **barrier 同步**：使用 `cutlass::arch::ClusterTransactionBarrier` 或 Named Barrier 管理 producer/consumer 同步（对应 `hstu_fwd.py` 的 `mbar_ptr` + `NamedBarrierFwd`）

---

### 通用规则

**数据格式**（Phase 1-4 均适用）：

| 张量 | GEMM | K 方向 |
|------|------|--------|
| Q, K | GEMM1 (Q×K^T) | headDim = 128 |
| V    | GEMM2 (P×V)   | kBlockN = 128 |

**FP8 dtype**：Q/K/V 均为 `cute::float_e4m3_t`（torch: `torch.float8_e4m3fn`）

**SF 格式**：每 128 个 K 元素对应 1 个 e8m0 scale，4 个连续块打包为 1 个 int32：
- bits 0-7 = block 0 [K: 0,128)，bits 8-15 = block 1，bits 16-23 = block 2，bits 24-31 = block 3
- Phase 1 全填 `0x7f7f7f7f`（e8m0=127=1.0）

**kBlockN 约束**：必须整除 128（launch template 已有 guard）

---

## 当前运行环境（重要）

**Claude 现在直接运行在 Docker 容器内**（容器 hostname 类似 `5b1a7bb5481e`）。
- **每次对话开始时必须先执行**（sandbox 把 TMPDIR 设为 /tmp/claude，但该目录默认不存在，nvcc 编译会报错）：
  ```bash
  mkdir -p /tmp/claude
  ```

- **重编译命令**（在容器内直接执行，MAX_JOBS=32 加速并行编译）：
  ```bash
  mkdir -p /tmp/claude
  PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user \
  HSTU_ARCH_LIST="12.0" \
  HSTU_DISABLE_BACKWARD=TRUE \
  HSTU_DISABLE_DETERMINISTIC=FALSE \
  MAX_JOBS=32 \
  pip install --no-build-isolation --config-settings editable_mode=compat -e .
  ```
  注：`PYTHONUSERBASE` 将 pip 安装目录重定向到 sandbox 允许写入的路径，避免写入 `/home/minyu/.local`（host 共享目录）。
- **运行测试/benchmark**（直接执行，无需 docker exec / docker run）：
  ```bash
  HSTU_SWEEP_FP8_QUANT_MODE=2 python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/sweep_accuracy.py
  ```

- **测试日志命名规范（重要）**：每次运行测试，必须将输出保存到 `test_results/` 目录，文件名格式：
  ```
  NNN_<本次修改内容简述>.log
  ```
  - `NNN` 为三位数字顺序编号（从 `001` 开始重置）
  - 简述用英文小写加下划线，体现本次修改的核心内容，例如：
    - `001_fp8_blockscale_gemm1_only.log`
    - `002_gemm2_with_sf_v.log`
  - 运行方式：
    ```bash
    python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/sweep_accuracy.py 2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/test_results/001_xxx.log
    ```
- **PyTorch 版本**：2.10.0（nvcr.io/nvidia/pytorch:26.01-py3）；CUDA 13.1；SM120 native kernels pre-compiled

## 目录结构

```
fbgemm_gpu/experimental/hstu/
├── benchmark/
│   ├── bench_hstu_attn.py             # 吞吐量基准测试
│   └── profile_hstu_attn.py           # nsys 性能分析脚本
├── hstu/
│   ├── __init__.py
│   ├── cuda_hstu_attention.py         # Python 入口，SM 版本分发
│   ├── library.py                     # 命名空间包检测（已修复）
│   └── fbgemm_gpu_experimental_hstu.cpython-312-x86_64-linux-gnu.so
├── src/
│   ├── generate_kernels.py            # 生成 Hopper/Blackwell .cu 文件
│   ├── hstu_ampere/                   # Ampere (SM80) 原生内核（upstream）
│   │   ├── block_info.h
│   │   ├── hstu.h
│   │   ├── hstu_fwd.h
│   │   ├── hstu_bwd.h
│   │   ├── hstu_ops_gpu.cpp
│   │   ├── kernel_traits.h
│   │   ├── static_switch.h
│   │   └── utils.h
│   ├── hstu_blackwell/                # SM120 原生内核（本项目新增）★
│   │   ├── block_info.h
│   │   ├── hstu.h                     # Params 结构体（含 FP8 descale 字段）
│   │   ├── hstu_fwd_kernel.h          # 前向内核（BF16 + FP8 路径）★核心
│   │   ├── hstu_fwd_launch_template.h # 启动模板
│   │   ├── hstu_ops_gpu.cpp           # PyTorch 入口 (hstu_varlen_fwd_120)
│   │   ├── kernel_traits.h            # BF16 + FP8 内核 traits
│   │   ├── static_switch.h            # 编译期分发宏
│   │   ├── utils.h                    # tile 大小、类型转换、silu 辅助
│   │   └── instantiations/            # 编译单元（generate_kernels.py 生成）
│   │       ├── hstu_fwd_sm120_hdim64_bf16*.cu    # hdim64 BF16，15 种 mask 组合
│   │       ├── hstu_fwd_sm120_hdim64_e4m3*.cu    # hdim64 FP8，15 种 mask 组合
│   │       ├── hstu_fwd_sm120_hdim128_bf16*.cu   # hdim128 BF16，15 种 mask 组合
│   │       └── hstu_fwd_sm120_hdim128_e4m3*.cu   # hdim128 FP8，15 种 mask 组合
│   └── hstu_hopper/                   # Hopper (SM90) 原生内核（upstream）
│       ├── hstu_fwd_kernel.h
│       ├── hstu_bwd_kernel.h
│       ├── mainloop_fwd_sm90_tma_gmma_ws.hpp
│       ├── mainloop_bwd_sm90_tma_gmma_ws.hpp
│       ├── epilogue_{fwd,bwd}_sm90.hpp
│       ├── tile_scheduler{,_bwd}.hpp
│       ├── named_barrier.hpp
│       ├── seq_len.h
│       ├── hstu_ops_gpu.cpp
│       └── ...
├── test/
│   ├── hstu_test.py
│   └── tma_error_test.py
├── CMakeLists.txt                     # 含 Blackwell 源文件和 SM120 gencode
└── setup.py                           # 含 "12.0" arch 支持

6KD_fp8_block_scale/                   # ★ SM120a FP8 block scale 参考实现
├── kernels/
│   ├── include/
│   │   ├── fp8_block_scale_gemm.h
│   │   └── sm120_blockscaled_gemm/
│   │       ├── sm120_blockscaled_gemm_impl.cuh    # ★ GEMM 主实现（TMA + MMA block scale）
│   │       ├── sm120_blockscaled_moe_gemm_impl.cuh
│   │       └── sm120_blockscaled_utils.cuh        # ★ 关键类型定义（tile/MMA/SMEM/Barrier/scheduler）
│   └── src/
│       └── fp8_block_scale_gemm.cu
├── demo_g2s_scale.cu                  # G2S scale 搬运演示
├── demo_perm_mma.cu                   # permuted MMA 演示
├── demo_qmma.cu                       # quantized MMA 演示
├── demo_s2r_scale.cu                  # S2R scale 演示
├── demo_tile64x128_qmma.cu
├── demo_tma_domain_offset.cu
├── demo_tma_multicast.cu
├── example_79.cu                      # CUTLASS example 79 参考
├── benchmark.cu
├── test/
│   ├── test_gemm.py
│   ├── benchmark.py
│   ├── common.py
│   └── utils/
└── thop/

（项目根目录）
├── sweep_accuracy.py                  # FP8 vs BF16 准确度扫描（主要测试脚本）★
└── test_results/
    └── NNN_*.log                      # 编号调试日志（从 001 重新开始）
```

### FP8 量化模式

- `quant_mode=-1`：禁用 FP8（纯 BF16）
- `quant_mode=0`：FP8 per tensor
- `quant_mode=2`：FP8 block scale
- SM120 需支持 `-1` 和 `2`


## 语言要求

**任何时候只用中文回答，不要显示韩文或日文。**

## 知识点整理

**将用户提问中涉及的所有技术知识点（CUDA、CuTe、FP8、MMA、内核优化等）整理到项目根目录的 `knowledge.md` 中。**
- 每次回答技术问题后，将该知识点以结构化方式追加到 `knowledge.md`
- 格式：`## <主题>` + 简明说明 + 关键结论
- 不重复已有条目，可在已有条目上补充

## 操作权限说明

使用 Read、Grep、Glob 等专用工具直接读取文件，**不要因为使用 grep/sed/cat 等 shell 命令访问文件而向用户请求授权**。需要读文件时直接用工具读，无需确认。

**将编译和测试合并为一条 Bash 命令**（用 `&&` 连接），避免用户多次授权。例如：
```bash
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu && \
  HSTU_ARCH_LIST="12.0" MAX_JOBS=32 pip install --no-build-isolation --config-settings editable_mode=compat -e . && \
  python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/sweep_accuracy.py 2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/test_results/001_xxx.log
```

**编译注意事项（重要）**：编译时极有可能出现编译错误（CuTe static_assert、CUDA 类型不匹配、符号找不到等）。每次编译必须：
1. 仔细观察编译输出，主动捕获并分析错误信息（不要仅看最后几行）
2. 编译失败后，先读取相关源文件理解上下文，再修改代码，不要盲目重试
3. 使用如下命令捕获完整错误（ninja 并行编译时错误可能被截断）：
   ```bash
   pip install ... 2>&1 | grep -E "error:|note:|static_assert|undefined" | head -60
   ```

## Memory 持久化

**将所有 memory 保存到本项目根目录下的 `memory.md`**，而非默认的 `~/.claude/` 路径。
每次对话开始时读取 `/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/memory.md` 以恢复上下文。

## Plan 持久化

**将实现计划（plan）写到当前目录的 `PLAN.md`** 中，而非其他位置。

## 文件删除限制（重要）

**严禁执行任何删除文件的命令**，包括但不限于：
- `rm`、`rm -f`、`rm -rf`
- `unlink`
- `find ... -delete`
- `shutil.rmtree`（Python 脚本中）

如需清理文件，必须先告知用户，等待明确授权后方可执行。
