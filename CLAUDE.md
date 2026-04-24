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
| 13 | AtomLayout `<_8,_1,_1>` + 消除 P staging（warp shuffle） | ✅ | seq≥1024 全面 +10~30%（2026-04-22）|
| **14** | setmaxnreg 224/56 + 消除全部 LDL/STL reg spill（math+load warp 零 spill） | ✅ | FP8 突破 1 TFLOPS（2026-04-22）|
| **15** | Q+SFA TMA 迁移到 math warp，直接打到 sQ_persist，删除 load warp Q copy + bar.sync | ✅ | FP8/BF16 比值 1.633（2026-04-22）|
| **16** | Q+SFA preload 移到 mbarrier wait 前，隐藏 LDSM latency，减少 No Eligible stall | ✅ | causal +1~3%，全验证通过（2026-04-23）|
| **17** | 同步点精简：删 S3、S2→wait_mbar_parity、q_tma_mbar 迁 math warp、S5→bar.sync 1,256 | ✅ | bs=1 causal +6.7%，全验证通过（2026-04-23）|
| **18** | cvt.e4m3x2 双路 FP8 转换：F2FP 指令数 64→32，打包链 IMAD+LOP3+PRMT+IADD→mov.b32 | ✅ | FP8 全面 +1-4%（2026-04-23）|
| **19** | TMA Store Output：epilogue SMEM→GMEM 改为 thread 0 单线程 TMA bulk store，255 线程提前退出 | ✅ | bs=8 causal +0.95%，全验证通过（2026-04-23）|
| **20** | kBlockM=256 M-direction Streaming（尝试后因寄存器压力导致严重性能回退，暂时搁置） | ⏸️ | 搁置（2026-04-24）|
| **21** | Opt A（N-loop 不变量提升：tCrQ+tCrSFA 移出循环）+ Opt C（split N-loop + if constexpr masking） | ✅ | bs=8 causal +0.7%，full +1.6%（2026-04-24）|

详细历史记录见 `HISTORY.md`。最新基线性能（Phase 21，benchmark 056）：seq=4096 causal，FP8 **1131.8 TFLOPS** vs BF16 647.0 TFLOPS（**+74.9%**）；seq=4096 full：FP8 **640.0 TFLOPS** vs BF16 365.6 TFLOPS（**+75.1%**）。

---

## Phase 21 COMPLETE：Opt A + Opt C — N-loop 不变量提升 + split N-loop（2026-04-24）

### 优化内容

**Opt A（N-loop 不变量提升）**：将 `tCrQ`（Q fragment，16 regs，来自 sQ_persist TMA）和 `tCrSFA`/`tCrSFA_frg`（Q scale factor，来自 smem_sfa_ptr TMA）从每 N-tile 重复加载改为在 N-loop 前一次性加载，消除每个 N-tile 的 4× ldmatrix + 1× LDS。

- 安全性保证：`sQ_persist` 和 `smem_sfa_ptr` 由 Q+SFA TMA 写入（Phase 15 迁移到 math warp），在 S2 `wait_mbar_parity(q_tma_mbar_ptr, 0)` 之后对全部 math warp 可见，N-loop 期间不再修改。
- **不提升 tCrSFP**：`smem_sfp_ptr` 由分布式线程写（`for i = tidx_math: smem_sfp_ptr[i] = 0x7f7f7f7f`），提升后 SFP copy 在写入后无显式 barrier，导致 H=4 配置出现 fp8_gt_cos=0.9918 的 race condition。已恢复 per-tile（利用 mbarrier wait + GEMM1 ~500 cycles 隐式可见性）。

**Opt C（split N-loop + if constexpr masking）**：将 N-loop 主体封装为 `run_n_tile` lambda（`auto kIsMasking_c` 模板参数），对 `Is_causal && !Is_arbitrary && !Is_local` 拆分为：
- Phase 1：前 `n_masking_steps` 个 tile，`kIsMasking=true`，`apply_mask_bs` 编译入
- Phase 2：剩余 tile，`kIsMasking=false`，`if constexpr` 死代码消除 `apply_mask_bs` 调用

### 验证结果

- [x] sweep_accuracy.py：全部 fp8_gt_cos ≥ 0.9996（日志 `1test_results/090_phase21_optA_optC_fix.log`）
- [x] run_hstu8_examples.sh：14/14 PASS
- [x] benchmark 056：见下表

**性能结果（benchmark 056，RTX PRO 6000 Blackwell SM120）**：

| Config | BF16 TFLOPS | FP8 TFLOPS | FP8 vs BF16 | vs Phase 19 |
|--------|-------------|------------|-------------|-------------|
| bs=8 seq=4096 h=16 causal | 647.0 | **1131.8** | **+74.9%** | **+0.7%** |
| bs=4 seq=4096 h=16 causal | 621.4 | **1075.7** | **+73.1%** | +0.7% |
| bs=1 seq=4096 h=16 causal | 476.1 | 853.3 | +79.2% | -2.2% |
| bs=8 seq=4096 h=16 full | 365.6 | **640.0** | **+75.1%** | **+1.6%** |
| bs=8 seq=2048 h=16 full | 335.2 | 612.0 | +82.6% | ~0% |

收益主要来自大 batch（bs=8）：减少每 tile 的 SMEM load 指令数，SMEM bank 压力略降，IPC 微升。bs=1 小幅回退可能是 GPU 调度 variance。主瓶颈（No Eligible stall ~63%，mbarrier wait 主导）未变。

---

## Phase 20 搁置记录：kBlockM=256 M-direction Streaming（2026-04-24）

### 搁置原因

**根本约束**：kNMSubtiles=2 时，`acc_o_0`（64 F32）和 `acc_o_1`（64 F32）共 128 个寄存器全程活跃，是不可规避的硬性约束。

**寄存器压力分析**：

| 阶段 | 固定活跃 regs | 峰值额外 | 总峰值 | vs 232（setmaxnreg math） |
|------|-------------|---------|--------|------------------------|
| GEMM1（全 K） | acc_o_0+o_1=128, acc_s=64, SFA/SFB=20, 杂项=15 | tCrK=128 | **355** | +123 → **严重 spill** |
| GEMM2（全 K） | acc_o_0+o_1=128, acc_s_packed=16, SFP/SFV=16, 杂项=15 | tCrV=128 | **303** | +71 → **严重 spill** |

**N-streaming=2 分析**（尝试将 N-loop 内 tCrK live range 从 128 降至 32）：
- GEMM1 k_step peak = 128+64+4+32+20+15 = **263 regs > 232** → GEMM1 仍 spill
- GEMM2 k_step peak = 128+16+4+32+16+15 = **211 regs < 232** → GEMM2 可 fit
- 结论：即使做 K-step streaming，GEMM1 阶段仍无法避免 spill，kBlockM=256 根本不可行

**实测性能（benchmark 057/058）**：

| 版本 | bs=8 seq=4096 h=16 causal | vs Phase 19 |
|------|--------------------------|-------------|
| Phase 19 基线 | **1123.6 TFLOPS** | — |
| Phase 20b（kBlockM=256 直接实现） | ~524 TFLOPS | **-53%** |
| Fix A（K-step streaming） | ~524 TFLOPS | **-53%** |
| Fix B（进一步调整） | ~366 TFLOPS | **-67%** |

### 结论

kBlockM=256 路线在当前 setmaxnreg=232 约束下不可行。唯一突破路径是大幅削减 acc_o 之外的寄存器用量（例如将 K/V fragment 完全流式化到 SMEM，每步只保留极少 regs），目前尚无可行方案。**暂时搁置，维持 Phase 19 为当前基线。**

---

## Phase 19 COMPLETE：TMA Store Output（2026-04-23）

### 优化目标

将 epilogue SMEM→GMEM 拷贝从 256 线程 LDS.128+STG.E.128 改为 thread 0 单线程 TMA async bulk store，其余 255 个 math thread 在 bar.sync 1,256 后提前退出。

### 核心变更

**host side**（`hstu_fwd_kernel.h`）：
- `Hstu_fwd_params_fp8_ws_tma` 新增 `TMA_O_t tma_o` 字段（第 7 个模板参数）
- `run_hstu_fwd_sm120_fp8_ws_tma_impl` 创建 TMA O descriptor：
  - GMEM tensor dim ordering：`(total_q, d, h)`，与 Q/K/V 保持一致（d 为 innermost，stride=1）
  - `make_tma_copy(SM90_TMA_STORE{}, tensor_O_full, SmemLayoutO_TMA_t{}, tile_shape, _1{})`

**device side**（`hstu_fwd_kernel_fp8_ws.h`）：
- S5 改为 `bar.sync 1, 256`（math warp only）
- OOB 清零：partial tile（varlen 最后一块）256 线程协作将 SMEM 越界行清零，再做第二个 `bar.sync 1, 256`
- `if (tidx_math != 0) return` — 255 线程提前退出
- thread 0：`fence.proxy.async.shared::cta` → `UTMASTG.3D`（TMA store）→ `UTMACMDFLUSH`（arrive）→ `DEPBAR.LE SB0, 0x0`（wait）→ EXIT

**关键 bug（已修正）**：TMA descriptor 必须用 `(total_q, d, h)` dim ordering（d stride=1 为 innermost）。若用 `(total_q, h, d)` 则 tile 的第二维 boxDim[1]=kHeadDim=128 > globalDim[1]=h=1，descriptor 初始化失败报 "Failed to initialize the TMA descriptor 1"。

### SASS 验证

SASS（`4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full.sass` offset 0x79e0）确认生成正确 TMA 指令序列：
```
UTMASTG.3D [UR8], [UR4]   ← TMA bulk store
UTMACMDFLUSH               ← tma_store_arrive (commit group)
DEPBAR.LE SB0, 0x0         ← tma_store_wait<0> (wait all groups)
EXIT
```

### SMEM 安全性

`tma_store_wait<0>()` 阻塞 thread 0 直到 TMA 完成数据提交。thread 0 是 CTA 最后存活线程，CTA 的 SMEM 仅在所有线程 EXIT 后才归还 SM，故 SMEM 不会被提前释放。

### 验证结果

- [x] sweep_accuracy.py：全部 fp8_gt_cos ≥ 0.9996（日志 `1test_results/088_phase19_tma_store.log`）
- [x] run_hstu8_examples.sh：14/14 PASS
- [x] benchmark 055：bs=8 seq=4096 causal **1123.6 TFLOPS**（+0.95% vs Phase 18）

**性能结果（benchmark 055，RTX PRO 6000 Blackwell SM120）**：

| Config | BF16 TFLOPS | FP8 TFLOPS | FP8 vs BF16 | vs Phase 18 |
|--------|-------------|------------|-------------|-------------|
| bs=8 seq=4096 h=16 causal | 644.7 | **1123.6** | **+74.3%** | **+0.95%** |
| bs=4 seq=4096 h=16 causal | 619.9 | **1068.2** | **+72.3%** | **+0.83%** |
| bs=1 seq=4096 h=16 causal | 493.6 | **872.6** | **+76.8%** | +0.06% |
| bs=8 seq=4096 h=16 full | 362.3 | **629.7** | **+73.8%** | -0.2% |
| bs=8 seq=2048 h=16 full | 340.6 | **612.1** | **+79.7%** | +0.2% |

收益主要来自大 batch causal（epilogue 占比较高）；full 和小 seq 受 Block Limit SMEM=1 限制（1 CTA/SM，255 线程提前退出无法帮助调度其他 CTA）基本持平。

---

## Phase 18 COMPLETE：cvt.e4m3x2 双路 FP8 转换（2026-04-23）

### 优化目标

`F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` 支持同时转两个 FP32→FP8（srcA+srcB），但原始代码传 srcB=RZ，浪费一路。改用 PTX `cvt.rn.satfinite.e4m3x2.f32` 同时转一对，并用 `mov.b32 {lo, hi}` 拼装，消除了原先复杂的 `IMAD+LOP3+PRMT+IADD` 打包链。

### 代码变更（`hstu_fwd_kernel_fp8_ws.h`）

```cpp
// Before: 4× scalar F2FP + IMAD+LOP3+PRMT+IADD packing chain
// After: 2× cvt.e4m3x2 + mov.b32
CUTE_UNROLL
for (int flat = 0; flat < kAccSElems; flat += 4) {
  uint32_t out;
  asm volatile(
      "{\n"
      ".reg .b16 lo, hi;\n"
      "cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
      "mov.b32 %0, {lo, hi};\n"
      "}\n"
      : "=r"(out)
      : "f"(float(acc_s(flat+0))), "f"(float(acc_s(flat+1))),
        "f"(float(acc_s(flat+2))), "f"(float(acc_s(flat+3))));
  acc_s_packed[flat / 4] = out;
}
```

### 验证结果

- [x] sweep_accuracy.py：全部 fp8_gt_cos ≥ 0.9996（日志 `1test_results/087_phase18_fp8_cvt2.log`）
- [x] run_hstu8_examples.sh：14/14 PASS
- [x] benchmark 054：见下表

**性能结果（benchmark 054，RTX PRO 6000 Blackwell SM120）**：

| Config | BF16 TFLOPS | FP8 TFLOPS | FP8 vs BF16 | vs Phase 17 |
|--------|-------------|------------|-------------|-------------|
| bs=8 seq=4096 h=16 causal | 644.2 | **1113.0** | **+72.8%** | **+2.9%** |
| bs=4 seq=4096 h=16 causal | 619.2 | **1059.4** | **+71.1%** | **+2.2%** |
| bs=1 seq=4096 h=16 causal | 492.7 | **872.1** | **+77.0%** | +1.2% |
| bs=8 seq=4096 h=16 full | 362.5 | **630.8** | **+74.0%** | **+2.5%** |
| bs=8 seq=2048 h=16 full | 340.8 | **610.7** | **+79.2%** | **+3.6%** |

---

## Phase 17 COMPLETE：同步点精简 — 删 S3 / S2→mbarrier wait / S5→bar.sync（2026-04-23）

### 优化目标

NCU Profile 052（Phase 16 基线）显示 No Eligible stall 仍达 **59.76%**，与 Phase 15 几乎相同，说明 mbarrier wait 依然是主瓶颈。在此基础上分析 CTA 内全部同步点，发现存在三类冗余：

| 同步点 | 原实现 | 问题 | 新实现 |
|--------|--------|------|--------|
| S3 | `__syncthreads()` | 纯集合点，无数据依赖，load/math 双方都不需要 | **删除** |
| S2 | `__syncthreads()` | Q+SFA TMA 等待本质是 math warp 内部事务，无需阻塞 load warp | `bar.sync 2,256` + `wait_mbar_parity(q_tma_mbar_ptr,0)` |
| q_tma_mbar init | load warp thread 256 初始化，再经 S1 同步 | 增加无谓 CTA 同步；q_tma_mbar 完全属于 math warp | 迁移至 math warp thread 0（含 `fence.proxy.async.shared::cta`） |
| S5 | `__syncthreads()` | epilogue 只有 math warp 参与，3 个 idle load warp 被白白锁死 | `bar.sync 1,256`（math-warp-only），load warp 提前退出 |

### 关键 bug：bar.sync 2,256 修复 race condition

thread 0 初始化 `q_tma_mbar_ptr`（SMEM write）后，threads 1-255 立即调用 `wait_mbar_parity`，但 SMEM 写入对其他线程不可见（无隐式 coherence）。修复：在 `if (tidx_math == 0)` 块之后、`wait_mbar_parity` 之前插入 `asm volatile("bar.sync 2, 256;\n")` 使 init 写入对全部 256 个 math thread 可见。未修复前表现为 flaky crash（非确定性，0-100% 概率出现）。

### 代码变更（`hstu_fwd_kernel_fp8_ws.h`）

**同步点 map（变更后）**：
```
S_arb : __syncthreads()         — Is_arbitrary 分支
S1    : __syncthreads()         — load warp 初始化 tma_mbar/math_mbar 后
S2    : bar.sync 2,256 + wait_mbar_parity(q_tma_mbar_ptr,0)  — math warp only
S3    : 删除
S5    : bar.sync 1,256          — math warp only；load warp 主循环结束后直接退出
```

**load warp 变更**：
- 删除 q_tma_mbar 初始化块（从 thread 256 的 mbarrier.init 和 arrive 语句）
- 删除 S3 `__syncthreads()`
- 删除 S5 `__syncthreads()`，load warp 主循环结束后直接 return

**math warp 变更**：
- thread 0 的 Q+SFA TMA preamble 中新增 `mbarrier.init`（含 `fence.proxy.async.shared::cta`）
- S2 从 `__syncthreads()` 改为 `bar.sync 2, 256` + 所有 256 线程自旋 `wait_mbar_parity`
- 删除 S3 `__syncthreads()`
- S5 从 `__syncthreads()` 改为 `asm volatile("bar.sync 1, 256;\n")`

### 验证结果

- [x] sweep_accuracy.py：全部 fp8_gt_cos ≥ 0.9996
- [x] run_hstu8_examples.sh：14/14 PASS（连续 5 次全部通过）
- [x] benchmark 053：见下表

**性能结果（benchmark 053，RTX PRO 6000 Blackwell SM120）**：

| Config | BF16 TFLOPS | FP8 TFLOPS | FP8 vs BF16 | vs Phase 16 |
|--------|-------------|------------|-------------|-------------|
| bs=8 seq=4096 h=16 causal | 644.3 | **1081.8** | **+67.9%** | +0.4% |
| bs=4 seq=4096 h=16 causal | 619.8 | **1036.6** | **+67.2%** | +0.8% |
| bs=1 seq=4096 h=16 causal | 476.4 | **861.9** | **+80.9%** | **+6.7%** |
| bs=8 seq=4096 h=16 full | 362.3 | **615.4** | **+69.8%** | +0.6% |
| bs=8 seq=2048 h=16 full | 340.1 | **589.5** | **+73.3%** | +1.8% |

bs=1 causal 收益最大（+6.7%），因为该配置 CTA 数少、load warp idle 占比更高，释放后 scheduler 调度收益更明显。

---

## Phase 16 COMPLETE：Q+SFA preload 移到 mbarrier wait 前（2026-04-23）

### 优化目标

**NCU Profile 050（Phase 15 基线）关键发现**：

| 指标 | 数值 | 说明 |
|------|------|------|
| SM Busy | 61.49% | SM 整体利用率 |
| No Eligible stall | **59.85%** | 主要瓶颈：调度器无可发射 warp |
| Registers/Thread | 168 | 静态 max = 224，有余量 |
| SMEM 占用 | 85 KB / CTA | Block Limit SMEM = 1（绑定） |
| SMEM Bank Conflicts | 7.9-way（epilogue）| **实际贡献 ~0.44%**，可忽略 |

**No Eligible 59.85% 根本原因**：math warp 在主循环每次迭代中，必须等 K/V TMA 数据就绪（`mbarrier.test_wait` 自旋），这段时间 12 个 math warp 全部阻塞，调度器无可发射指令，直接体现为 "No Eligible" stall。

**优化方案**：将以下与 TMA 无关、仅依赖 `sQ_persist` 和 `smem_sfa_ptr` 的计算移到 mbarrier wait 前：
- 指针计算：`smem_sfb_cur`、`sK_cur`、`sVt_cur`（纯算术，无内存依赖）
- SFA 加载：`tCrSFA` 从 `smem_sfa_ptr[sfa_row]`（sQ persist 缓冲区，主循环前已就绪）
- Q 预加载：`load_a_z_pattern(sQ_persist_pi, tCrQ, ...)` （`sQ_persist` 主循环前已就绪）

**预期收益**：Q LDSM 延迟（~80 cycles，16 regs × ~5 cycle/reg SMEM load）隐藏在 K TMA 等待时间（~100-200 cycles）中，减少 No Eligible stall 约 40-80 cycles/iteration。

### 代码变更

**文件**：`hstu_fwd_kernel_fp8_ws.h`，主循环 mbarrier wait 区域

**变更前**（原始结构）：
```cpp
// Wait for TMA K/V/SFB/SFV
{ mbarrier.test_wait spin loop }
if (math_stage) { tma_parity1 ^= 1; } else { tma_parity0 ^= 1; }
asm volatile("" ::: "memory");
// 所有指针计算 + SFA + Q 加载在 wait 之后
int32_t* smem_sfb_cur = smem_sfa_ptr + 2*kBlockM + math_stage*kBlockN;
FP8Elem* sK_cur = ...; FP8Elem* sVt_cur = ...;
Tensor tCrSFA = ...; tCrSFA(0,0,0) = smem_sfa_ptr[sfa_row];
Tensor tCrQ = ...; load_a_z_pattern(sQ_persist_pi, tCrQ, 0, kHeadDim/32);
// GEMM1...
```

**变更后**（Phase 16）：
```cpp
// Phase 16: 指针计算 + SFA + Q 预加载 — 在 mbarrier wait 前发射 LDSM
int32_t* smem_sfb_cur = smem_sfa_ptr + 2*kBlockM + math_stage*kBlockN;
FP8Elem* sK_cur = ...; FP8Elem* sVt_cur = ...;
Tensor tCrSFA = ...; tCrSFA(0,0,0) = smem_sfa_ptr[sfa_row];
Tensor tCrQ = ...; load_a_z_pattern(sQ_persist_pi, tCrQ, 0, kHeadDim/32);
// Wait for TMA K/V/SFB/SFV（Q LDSM latency 在此期间隐藏）
{ mbarrier.test_wait spin loop }
if (math_stage) { tma_parity1 ^= 1; } else { tma_parity0 ^= 1; }
asm volatile("" ::: "memory");
// SFB + K 加载（依赖 TMA 数据，必须在 wait 后）
Tensor tCrSFB = ...; auto tCrSFB_frg = ...;
Tensor tCrK = ...; load_b_z_pattern(sK_cur_pi, tCrK, 0, kHeadDim/32);
// GEMM1...
```

### 寄存器压力分析

| 阶段 | 新增寄存器 | 峰值 | 余量（vs 224） |
|------|-----------|------|--------------|
| Wait 前预加载 Q（16 regs） + SFA（1 reg） | +17 | 168+17=185 | 39 |
| Wait 后与原有变量重叠 | 0 | 185 | 39 |

寄存器不超 224，无 spill 风险。

### 预期效果

| 指标 | Phase 15 基线 | Phase 16 目标 |
|------|-------------|-------------|
| No Eligible stall | 59.85% | < 45%（隐藏约 40-80 cycles/iter） |
| FP8 TFLOPS（seq=4096 causal）| ~1060 | 预计 +3-8% |
| FP8 TFLOPS（seq=4096 full）| ~620 | 预计 +3-8% |
| LDL/STL（SASS）| 0 | 保持 0 |

### 验证结果

- [x] 代码已修改、编译通过
- [x] sweep_accuracy.py：全部 fp8_gt_cos ≥ 0.9996（`1test_results/086_phase16_q_preload.log`）
- [x] run_hstu8_examples.sh：14/14 PASS
- [x] benchmark 052（锁频 2407MHz）：见下表

**性能结果（benchmark 052，锁频 2407MHz，RTX PRO 6000 Blackwell SM120）**：

| Config | BF16 TFLOPS | FP8 TFLOPS | FP8 vs BF16 |
|--------|-------------|------------|-------------|
| bs=8 seq=4096 h=16 causal | 646 | **1077.7** | **+66.8%** |
| bs=4 seq=4096 h=16 causal | 620 | **1028.4** | **+65.9%** |
| bs=1 seq=4096 h=16 causal | 471 | **808.0** | **+71.7%** |
| bs=8 seq=4096 h=16 full | 362 | **611.8** | **+69.0%** |
| bs=8 seq=2048 h=16 full | 334 | **579.1** | **+73.5%** |

vs Phase 15（048，无锁频）：causal 配置 FP8/BF16 比值 +1~3pp；full 配置基本持平。  
实际收益低于预期（3-8%）：ptxas 可能已做部分指令前移，或实际 TMA wait 窗口不足以完全隐藏 Q LDSM 延迟。

---

## Phase 15 COMPLETE：Q+SFA TMA 迁移到 math warp（2026-04-22）

**核心变更**（`hstu_fwd_kernel_fp8_ws.h` + `kernel_traits.h`）：
- **kernel_traits.h**：`kSmemMbarSize` 32→40（新增 `q_tma_mbar` 第5个 barrier）；`kSmemWsQPersistOffset` 对齐 128B→2048B（与 SW128 swizzle 周期一致，保证 TMA 绝对写地址 = PI-swizzle LDSM 读地址）
- **load warp**：删除 Q+SFA TMA preamble block（约30行），S2 变纯同步屏障
- **math warp**：新增 `q_tma_mbar_ptr` 声明；thread 0 在 S1 后发 Q+SFA TMA → `wait_mbar_parity(q_tma_mbar_ptr, 0)` 自旋等待完成；S2 使 Q 对所有 math warp 可见
- **主循环**：`tma_parity0 = 1` → `0`（Q TMA 不再占用 tma_mbar[0]，K TMA 从 parity 0 起算）

**两个关键 bug 修复**（Codex review 发现）：
1. `q_tma_mbar` init `expected=2` → `1`（避免 `wait_mbar_parity` 死锁）
2. `tma_parity0 = 1` → `0`（Q 不再用 tma_mbar[0]，parity 修正）

**验证结果**（2026-04-22）：
- sweep_accuracy.py：所有配置 fp8_gt_cos ≥ 0.9996（日志 `1test_results/085_phase15_q_tma_math_warp.log`）
- run_hstu8_examples.sh：14/14 PASS
- FP8/BF16 TFLOPS 比值：1.619（Phase 14: 1.616），无退化（benchmark 048）

**性能结果（benchmark 048，RTX PRO 6000 Blackwell SM120）**：

| Config | BF16 TFLOPS | FP8 TFLOPS | FP8 vs BF16 |
|--------|-------------|------------|-------------|
| bs=8 seq=4096 h=16 causal | 654 | **1059.2** | **+61.9%** |
| bs=4 seq=4096 h=16 causal | 627 | **1025.2** | **+63.6%** |
| bs=1 seq=4096 h=16 causal | 472 | **804.2** | **+70.3%** |
| bs=8 seq=4096 h=16 full | 366 | **619.0** | **+69.3%** |
| bs=8 seq=2048 h=16 full | 329 | **581.1** | **+76.7%** |

---

## Phase 14 COMPLETE：setmaxnreg 224/56 + 消除全部 LDL/STL 寄存器 Spill（2026-04-22）

**核心变更**（`hstu_fwd_kernel_fp8_ws.h`）：
- **math warp**：`smem_base32` 替换 6 个持久化 64-bit 指针数组（freed ~24 regs）；`tma_wait_parity[2]` → 独立标量 `tma_parity0/1`
- **load warp**：`tma_mbar_ptr[2]`/`math_mbar_ptr[2]` → 独立标量指针 `tma_mbar_ptr0/1`/`math_mbar_ptr0/1`；mbar 地址用字节偏移加到 32-bit SMEM 地址上；`math_wait_parity[2]` → 独立标量 `math_wait_parity0/1`
- **setmaxnreg**：load warp `56`，math warp `224`

**优化历程（SASS LDL/STL 计数）**：

| 阶段 | LDL/STL 总数 | 说明 |
|------|------------|------|
| Phase 13 基线 | 24 | math warp 热路径 6×STL.128 + 1×STL.64 |
| smem_base32 替换指针数组 | 11 | math warp 热路径清零 |
| tma/math_wait_parity 标量化 | 4 | math warp 零 spill，余 load warp |
| load warp mbar 指针标量化 | **0** | **全部清零** |

**验证结果**：
- sweep_accuracy.py：所有配置 fp8_gt_cos ≥ 0.9996（日志 `1test_results/084_phase15_tma_parity_scalar.log`）
- run_hstu8_examples.sh：14/14 PASS

**性能结果（benchmark 047，RTX PRO 6000 Blackwell SM120）**：

| Config | BF16 TFLOPS | FP8 TFLOPS | FP8 vs BF16 |
|--------|-------------|------------|-------------|
| bs=8 seq=4096 h=16 causal | 661 | **1068.7** | **+61.6%** |
| bs=4 seq=4096 h=16 causal | 633 | **1012.3** | **+59.9%** |
| bs=1 seq=4096 h=16 causal | 473 | **783.9** | **+65.8%** |
| bs=8 seq=4096 h=16 full | 367 | **619.2** | **+68.9%** |
| bs=8 seq=2048 h=16 causal | 577 | **860.0** | **+49.1%** |

Phase 13（039）→ Phase 16（047）FP8 kernel TFLOPS 提升：seq=4096 causal +28%，seq=4096 full +35%。

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
