## Core Instruction for CodeX MCP

在任何时刻，你必须思考当前过程可以如何与codex进行协作，如何调用Codex 为你提供的MCP工具作为你客观全面分析的保障。
其中你**务必执行**以下几个步骤：
**1** 在你对用户需求形成初步分析后，将用户需求、初始思路告知codex，并要求其完善需求分析和实施计划。
**2** 在实施具体编码任务前，**必须向codex索要代码实现原型（要求codex仅给出unified diff patch，严禁对代码做任何真实修改）**。在获取代码原型后，你**只能以此为逻辑参考，再次对代码修改进行重写**，形成企业生产级别、可读性极高、可维护性极高的代码后，才能实施具体编程修改任务。
**3** 无论何时，只要完成切实编码行为后，**必须立即使用codex review代码改动和对应需求完成程度**。
**4** codex只能给出参考，你**必须有自己的思考，甚至需要对codex的回答提出置疑**。尽信书则不如无书，你与codex的最终使命都是达成统一、全面、精准的意见，所以你们必须不断争辩已找到通向真理的唯一途径。


## Codex Tool Invocation Specification

 1. 工具概述

  codex MCP 提供了一个工具 `codex`，用于执行 AI 辅助的编码任务。该工具**通过 MCP 协议调用**，无需使用命令行。

  2. 工具参数

  **必选**参数：
  - PROMPT (string): 发送给 codex 的任务指令
  - cd (Path): codex 执行任务的工作目录根路径

  可选参数：
  - sandbox (string): 沙箱策略，可选值：
    - "read-only" (默认): 只读模式，最安全
    - "workspace-write": 允许在工作区写入
    - "danger-full-access": 完全访问权限
  - SESSION_ID (UUID | null): 用于继续之前的会话以与codex进行多轮交互，默认为 None（开启新会话）
  - skip_git_repo_check (boolean): 是否允许在非 Git 仓库中运行，默认 False
  - return_all_messages (boolean): 是否返回所有消息（包括推理、工具调用等），默认 False
  - image (List[Path] | null): 附加一个或多个图片文件到初始提示词，默认为 None
  - model (string | null): 指定使用的模型，默认为 None（使用用户默认配置）
  - yolo (boolean | null): 无需审批运行所有命令（跳过沙箱），默认 False
  - profile (string | null): 从 `~/.codex/config.toml` 加载的配置文件名称，默认为 None（使用用户默认配置）

  返回值：
  {
    "success": true,
    "SESSION_ID": "uuid-string",
    "agent_messages": "agent回复的文本内容",
    "all_messages": []  // 仅当 return_all_messages=True 时包含
  }
  或失败时：
  {
    "success": false,
    "error": "错误信息"
  }

  3. 使用方式

  开启新对话：
  - 不传 SESSION_ID 参数（或传 None）
  - 工具会返回新的 SESSION_ID 用于后续对话

  继续之前的对话：
  - 将之前返回的 SESSION_ID 作为参数传入
  - 同一会话的上下文会被保留

  4. 调用规范

  **必须遵守**：
  - 每次调用 codex 工具时，必须保存返回的 SESSION_ID，以便后续继续对话
  - cd 参数必须指向存在的目录，否则工具会静默失败
  - 严禁codex对代码进行实际修改，使用 sandbox="read-only" 以避免意外，并要求codex仅给出unified diff patch即可

  推荐用法：
  - 如需详细追踪 codex 的推理过程和工具调用，设置 return_all_messages=True
  - 对于精准定位、debug、代码原型快速编写等任务，优先使用 codex 工具

  5. 注意事项

  - 会话管理：始终追踪 SESSION_ID，避免会话混乱
  - 工作目录：确保 cd 参数指向正确且存在的目录
  - 错误处理：检查返回值的 success 字段，处理可能的错误


# ⚠️ 严格要求：所有回复必须用中文，禁止输出韩文或日文 ⚠️

# HSTU FBGEMM 项目说明

## 项目概述

本项目是 FBGEMM 中 HSTU（Hierarchical Sequential Transduction Unit）注意力机制的 CUDA 实现，upstream 代码已支持 Hopper（SM90），本项目目标是让 Blackwell（SM120）GPU 架构支持 HSTU 注意力，包含 BF16 和 FP8 block-scale 量化模式。

---

## 当前进度

| Phase | 描述 | 状态 | 关键成果 |
|-------|------|------|----------|
| 0-3 | BF16 实现 + FP8 基础框架 + TMA | ✅ | fp8_gt_cos ≈ 0.9996 |
| 4-5 | K+V double-prefetch + TMA K/V^T | ✅ | 延迟 -6-7% |
| 6 | Warp-specialized kernel（double-buffer） | ✅ | 14/14 PASS（2026-04-03）|
| 7-8 | TMA SFB + TMA SFV | ✅ | 14/14 PASS（2026-04-04）|
| 9 | 12-warp + setmaxnreg 216 | ✅ | FP8 TFLOPS +27-51%（2026-04-08）|
| 10 | 消除 SMEM transpose（col-major V） | ✅ | seq≥2048 FP8 超 BF16（2026-04-13）|
| 11 | PTX LDSM_T + MN_SW128 swizzle | ✅ | seq≥1024 FP8 超 BF16（2026-04-16）|
| 12 | 代码整洁化（本地 sm120_qmma_builder.h） | ✅ | 无功能变化（2026-04-17）|
| **13** | AtomLayout `<_8,_1,_1>` + 消除 P staging（warp shuffle） | ✅ | seq≥1024 全面 +10~30%（2026-04-22）|

详细历史记录见 `HISTORY.md`。最新基线性能（Phase 13，benchmark 039）：seq=4096 causal，FP8 **833 TFLOPS** vs BF16 658 TFLOPS（**+26.7%**）；seq=4096 full：FP8 **458 TFLOPS** vs BF16 364 TFLOPS（**+25.8%**）。

---

## Phase 13 COMPLETE：消除 P Staging SMEM — AtomLayout `<_8,_1,_1>` + Warp Shuffle（2026-04-22）

**核心变更**：
- `sm120_qmma_builder.h`：TiledMMA AtomLayout `<_2,_4,_1>` → `<_8,_1,_1>`（8M×1N warps），`partition_fragment_SFB` bug fix（`get<1>` → `get<2>`）
- `hstu_fwd_kernel_fp8_ws.h`：GEMM1/GEMM2 均改为 4×K=32 k_step streaming；P staging（2×bar.sync + 16KB SMEM 写读 + LDSM）替换为 warp 内 `__shfl_sync` + `__byte_perm`

**warp shuffle 核心逻辑**（关键 bug 修复记录）：
- C-fragment N-atom → A-fragment K-slot 映射：必须先 shuffle 两个 half（pk0/pk1），再由目标线程按 `tiq>>1` 选择，不能在 shuffle 前选（否则 src_lane 的 tiq 不同导致 N-atom 错配，cos_sim ≈ 0.66）
- 代码位置：`hstu_fwd_kernel_fp8_ws.h` 约第 1040-1065 行

**验证结果**（2026-04-22）：
- sweep_accuracy.py：所有配置 fp8_gt_cos ≥ 0.9996（日志 `1test_results/081_phase13_shfl_fix.log`）
- run_hstu8_examples.sh：14/14 PASS

**性能结果（benchmark 039，RTX PRO 6000 Blackwell SM120）**：

| Config | BF16 TFLOPS | FP8 TFLOPS | FP8 vs BF16 |
|--------|-------------|------------|-------------|
| bs=8 seq=4096 h=16 causal | 658 | **833** | **+26.7%** |
| bs=4 seq=4096 h=16 causal | 633 | **792** | **+25.2%** |
| bs=1 seq=4096 h=16 causal | 463 | **591** | **+27.9%** |
| bs=8 seq=4096 h=16 full | 364 | **457** | **+25.8%** |
| bs=8 seq=2048 h=16 full | 326 | **429** | **+31.9%** |

Phase 12 基线（035）→ Phase 13（039）FP8 kernel TFLOPS 提升：seq≥1024 full +15~25pp，causal +9~13pp。
seq=512 causal 有约 -10pp 小幅退化（k_step overhead 与 causal masking 交互），属预期内。

---

## 核心文件

`fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/`
- `hstu_fwd_kernel.h` — 前向内核（BF16 + FP8 非WS路径 + include WS文件）★
- `hstu_fwd_kernel_fp8_ws.h` — FP8 WS 内核主体（Phase 6+）★
- `sm120_qmma_builder.h` — SM120 QMMA builder 本地子集（Phase 12+）
- `kernel_traits.h` — BF16 + FP8 内核 traits
- `hstu_ops_gpu.cpp` — PyTorch 入口（`hstu_varlen_fwd_120`）
- `hstu.h` / `utils.h` / `hstu_fwd_launch_template.h`

参考实现：`6KD_fp8_block_scale/kernels/include/sm120_blockscaled_gemm/`（只读）

---

## 通用技术规则

**数据格式**：Q/K 为 GEMM1（K 方向=headDim=128），V 为 GEMM2（K 方向=kBlockN=128）  
**FP8 dtype**：`cute::float_e4m3_t`（torch: `torch.float8_e4m3fn`）  
**SF 格式**：每 128 个 K 元素对应 1 个 e8m0 scale，4 个连续块打包为 1 个 int32（bits 0-7 = block 0, 8-15 = block 1, ...）  
**FP8 量化模式**：`quant_mode=-1`（BF16）、`quant_mode=2`（FP8 block scale，SM120 支持）  
**kBlockN 约束**：必须整除 128

---

## 当前运行环境

Claude 直接运行在主机（无需 docker exec），bwrap sandbox 隔离，GPU 设备透传已配置。  
**环境**：PyTorch 2.10.0（nvcr.io/nvidia/pytorch:26.01-py3）；CUDA 13.1。

**重编译命令**（必须分三次独立 Bash 调用，严禁 `&&`、`;`、`|` 组合）：

```bash
# 命令1
mkdir -p /tmp/claude

# 命令2
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu

# 命令3
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user HSTU_ARCH_LIST="12.0" HSTU_DISABLE_BACKWARD=TRUE HSTU_DISABLE_DETERMINISTIC=FALSE HSTU_DISABLE_HDIM32=TRUE HSTU_DISABLE_HDIM64=TRUE HSTU_DISABLE_HDIM256=TRUE MAX_JOBS=32 pip install --no-build-isolation --config-settings editable_mode=compat -e .
```

---

## 功能验证流程

**日志命名规范**：`1test_results/NNN_<修改内容简述>.log`

### 步骤 1：sweep_accuracy.py（数值正确性）

```bash
HSTU_SWEEP_FP8_QUANT_MODE=2 python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/sweep_accuracy.py 2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/1test_results/NNN_xxx.log
```

检查各配置 `fp8_gt_cos` ≥ 0.995。

### 步骤 2：run_hstu8_examples.sh（14 个 example case）

```bash
bash /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/run_hstu8_examples.sh
```

检查输出为 `14/14 passed`。

---

## 数值错误 Debug 流程（强制规程）

**当修改代码后仍无法定位问题时，禁止再次盲目读代码。必须按以下步骤依次进行 debug，一旦某步出错立即停止并在该阶段插入 printf 观察。**

### Debug 步骤顺序

**Step 1：gemm1_only + all_ones**
- 设置 `params.debug_gemm1_only=true`，Q/K/SFA/SFB 全设为 1（e8m0=0x3F=1.0）
- 预期：acc_o 所有元素 = kHeadDim = 128
- 若失败：在 GEMM1 调用前后插入 printf 观察 tCrQ/tCrK/tCrSFA/tCrSFB 的值

**Step 2：gemm1_only（真实数据）**
- 设置 `params.debug_gemm1_only=true`，使用真实 Q/K/SFA/SFB
- 对比 acc_o 与 BF16 参考的 cos_sim
- 若 cos_sim < 0.995：问题在 GEMM1 路径（Q/K 加载或 SFA/SFB 加载）
- 若 cos_sim ≥ 0.995：问题在 GEMM2 路径（V/SFV 加载或 P staging）

**Step 3：all_ones（全链路）**
- 不开 gemm1_only，Q/K/V/SFA/SFB/SFV 全设为 1
- 预期：acc_o 所有元素 = kBlockN * kHeadDim = 128 * 128 = 16384 * alpha * silu(alpha)
- 若失败：结合 Step 1/2 结论缩小范围

### Printf 观察要点

一旦确定出错阶段，插入 printf 观察（加 `if ((tidx_math & 31) == 0 && tidx_math < 64)` guard 避免刷屏）：
- **输入 A/B**：`tXrQ(0)`、`tXrK(0)` 等寄存器值
- **SFA/SFB/SFV**：`tCrSFA(0,0,0)`、`tCrSFB(0,nr,0)` 等
- **输出 acc**：`acc_s(0)`、`acc_o(0)` 等

---

## 性能分析流程

**文件命名规范**：
- benchmark log：`2benchmark_results/NNN_<commit_short>_gpu<MHz>MHz_<描述>.log`
- profile 产物：`3profile_results/NNN_<描述>.*`
- SASS：`4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full.sass`

### 前置：锁定 GPU 频率

```bash
sudo nvidia-smi -lgc 2407   # 锁频（RTX PRO 6000 最大 boost clock）
sudo nvidia-smi -rgc         # 跑完后解锁
```

### 步骤 1：bench_hstu_attn_sm120.py

```bash
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu/benchmark/bench_hstu_attn_sm120.py 2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/2benchmark_results/NNN_xxx.log
```

### 步骤 2：run_profile.sh（nsys + ncu）

```bash
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu
./run_profile.sh NNN <描述>
```

**ncu 分析重点**：`launch__registers_per_thread`（目标 216）、`l1tex__t_sectors_pipe_lsu_mem_local_op_ld/st.sum`（spill）、Long Scoreboard stall、Math Throttle、SM 利用率。

### 步骤 3：dump_sass_fp8_ws.sh

```bash
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu
./dump_sass_fp8_ws.sh
```

SASS 中搜索 `LDL`/`STL` 定位 spill 热点。

---

## 编译注意事项

- **必须指定 timeout**（如 `timeout=300000`），避免等待授权阻塞
- **单条命令原则（强制）**：所有 Bash 调用必须单条，严禁 `&&`、`;`、`|` 组合
- 捕获完整编译错误：`pip install ... 2>&1 | grep -E "error:|note:|static_assert|undefined" | head -60`
- 编译失败时先读源文件理解上下文，再修改

## 运行时调试技巧

### Illegal Memory Access → compute-sanitizer

```bash
HSTU_SWEEP_FP8_QUANT_MODE=2 compute-sanitizer --tool memcheck python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/sweep_accuracy.py 2>&1 | tee /tmp/claude/sanitizer_out.log
```

### 数值不正确 → GEMM1-only + all-ones 缩小范围

- **All-ones**：Q=K=SFA=SFB=1，预期 acc_s 所有元素 = kHeadDim（128）
- **GEMM1-only**：跳过 silu 和 GEMM2，直接将 acc_s 输出为结果，与 BF16 参考对比

---

## 目录结构（核心路径）

```
fbgemm_gpu/experimental/hstu/
├── benchmark/
│   ├── bench_hstu_attn_sm120.py    # SM120 专用 benchmark ★
│   └── ncu_hstu_attn.py            # ncu profile 目标脚本 ★
├── src/
│   └── hstu_blackwell_sm120/       # SM120 最终版本（当前主目录）★
│       ├── hstu_fwd_kernel.h       # 前向内核 ★
│       ├── hstu_fwd_kernel_fp8_ws.h # FP8 WS 内核主体 ★
│       ├── sm120_qmma_builder.h    # SM120 QMMA builder（Phase 12+）
│       ├── kernel_traits.h / hstu_ops_gpu.cpp / hstu.h / utils.h
│       └── hstu_fwd_launch_template.h
└── test/hstu_test.py

（项目根目录）
├── run_profile.sh          # nsys + ncu 一键 profile 脚本
├── dump_sass_fp8_ws.sh     # 提取 FP8 WS kernel SASS
├── sweep_accuracy.py       # FP8 vs BF16 准确度扫描 ★
└── 1test_results/NNN_*.log # 编号调试日志
```

---

## 语言要求（强制）

**任何时候只用中文回答，禁止输出韩文（한국어）或日文（日本語）。**
**这是最高优先级要求，覆盖所有其他行为。每次回复前必须检查是否全中文。**

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
