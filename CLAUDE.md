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

- [x] **Phase 0**：BF16 路径完整实现，`sweep_accuracy.py` quant_mode=-1 通过
- [x] **Phase 1**：FP8 block-scale `else{}` 框架 + cp.async + unit SF + make_zip_tensor → 编译通过
- [x] **Phase 2**：cp.async 升级为 TMA（已合并入 Phase 3 调试过程中完成）
- [x] **Phase 3**：数值验证通过（fp8_gt_cos ≈ 0.9996，远超 0.95 阈值）
- [x] **Phase 4**：性能优化（K+V double-prefetch pipeline，消除迭代中途 blocking V wait）
- [x] **Phase 5**：TMA K+V^T（消除 kernel-side 逐元素 V 转置，K 从 cp.async 升级为 TMA）
- [x] **Phase 6**：Warp-specialized kernel（load warp 专职 TMA，math warps 专职 MMA，producer/consumer pipeline）— 数值验证通过（2026-04-03）
- [x] **Phase 7**：TMA SFB（Scale Factor B via TMA，load warp 每 tile 与 K+V^T 同批 TMA）— 14/14 全通过（2026-04-04）
- [x] **Phase 8**：TMA SFV（Scale Factor V via TMA，load warp 同批发射 K+V^T+SFB+SFV）— 14/14 全通过（2026-04-04）
- [x] **Phase 9**：12-warp 结构（3 个完整 warpgroup）+ `setmaxnreg 216` → math warp 寄存器提升至 216，消除 TRY_ALLOC deadlock，FP8/BF16 比从 ~1.7-2.1x 改善至 ~1.52-1.56x — 验证通过（2026-04-08）
- [x] **Phase 10**：消除 SMEM transpose（Python 侧 V 列主序 + kernel SmemLayoutVt_TMA 改为 K_SW128）→ seq≥2048 时 FP8 超过 BF16，seq=1024 基本持平 — 验证通过（2026-04-13）
- [x] **Phase 11**：直接 PTX LDSM_T（`ldmatrix.m16n16.x2.trans.b8`）+ 手算 swizzle 地址，从 MN_SW128 SMEM 加载 V^T 到寄存器，消除 2×bar.sync + 16KB cooperative copy — 数值通过（fp8_gt_cos ≈ 0.9996，14/14 PASS）；附 MN_SW128 swizzle 优化，seq≥1024 FP8 全面超越 BF16（2026-04-16）
- [x] **Phase 12**：代码整洁化——消除 6KD 绝对路径依赖、删除死代码（Phase 5 遗留 impl + Phase 4 generic FP8 body）、新建本地 `sm120_qmma_builder.h`、清理废弃 MMA 定义 — 完成（2026-04-17）

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

### ~~已知性能问题：SMEM→SMEM 手动 transpose~~ ✅ Phase 10 已消除

**Phase 6 时的 V^T 数据路径（已废弃）**：
```
GMEM V → TMA → MN_SW128 SMEM → [256线程手动transpose] → K_SW128 SMEM → LDSM_N → 寄存器
```

**Phase 10 后的 V^T 数据路径**：
```
GMEM V（列主序，token stride-1）→ TMA → K_SW128 SMEM → LDSM_N → 寄存器
```

消除方案：Python 侧 `v_fp8.permute(2,1,0).contiguous().permute(2,1,0)` 使 token 轴 stride-1，K_SW128 TMA 直接写入正确 layout，无需 transpose。详见 Phase 10。

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
测试日志：`1test_results/011_ws_phase6_final.log`（Phase 6），`1test_results/021_hstu8_examples_qm2.log`（Phase 7）

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

### 后续优化方向（Phase 6 时规划，部分已完成）
- [x] **TMA SFB**：SFB 从 scalar GMEM 读取改为 TMA，与 K+V^T 同批发射（Phase 7 完成）
- [x] **TMA SFV**：SFV 从 scalar GMEM 读取改为 TMA，与 K+V^T+SFB 同批发射（Phase 8 完成）
- [ ] **消除 SMEM transpose**：Python 侧 V 列主序存储，TMA 直接写 K_SW128，预计节省每 tile 约 2×bar.sync + 16KB 搬运
- [ ] **nsys profile**：量化各部分耗时（GEMM1/GEMM2/transpose/SFB+SFV load），确定实际瓶颈

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

测试日志：`1test_results/021_hstu8_examples_qm2.log`

### Phase 7 性能结果（2026-04-04，RTX PRO 6000 Blackwell SM120）

**准确度**：全部 PASS（cos_sim ≈ 0.9985~0.9986）

**kernel-only**（bs=4, h=16, d=128，full attention）：

| seq  | BF16    | BF16 TFLOPS | FP8 (qm=2) | FP8 TFLOPS | FP8/BF16 |
|------|---------|-------------|------------|------------|----------|
| 512  | 0.045ms | 193         | 0.090ms    | 96         | 2.00x 慢 |
| 1024 | 0.121ms | 285         | 0.234ms    | 147        | 1.93x 慢 |
| 2048 | 0.451ms | 305         | 0.775ms    | 177        | 1.72x 慢 |
| 4096 | 1.523ms | 361         | 2.750ms    | 200        | 1.81x 慢 |

**注**：与 Phase 6 对比，FP8/BF16 延迟比从约 1.89~2.19x 小幅改善至 1.72~2.0x（GPU boost clock 跨 session 不同，绝对值不可直接比较，比值有参考意义）。

**end-to-end**：Python 侧量化开销约 0.78~1.8ms（随 bs 线性增长），远超 kernel 本身，FP8 端到端仍显著劣于 BF16。

benchmark 日志：`2benchmark_results/006_709899ce_gpu2430MHz_phase7_tma_sfb.log`

---

## Phase 8 完成：TMA SFV（Scale Factor V via TMA，2026-04-04）

**目标**：将 V 侧 block scale（SFV）从 math warp 的标量 GMEM 读取升级为 load warp 的 TMA，与 K+V^T+SFB 在同一 mbarrier 批次内一起发射，消除 math warp 内每 tile 的 scalar GMEM read + broadcast-to-SMEM 开销。

### 实现内容

- **Python 侧数据格式**：`sf_v_packed` 从 `[H, total_blocks]` 通过 `.repeat_interleave(kBlockN, dim=1)` 扩展为 `[H, total_tokens]`，使每个 block 的 scale 值在 token 维度重复 kBlockN 次，从而与 `sf_k_packed` 结构相同，TMA SFV 可用相同的 tile 索引 `nb_abs = binfo.sum_s_k / kBlockN + nb`
- **SmemLayoutSFV_TMA_t**：`Layout<[kBlockN, 1], [1, kBlockN]>`（与 SFB 相同）
- **双缓冲 SFV**：`smem_sfv_ptr[2] = {smem_sfa_ptr + kBlockM + 2*kBlockN, + 3*kBlockN}`，与 K/V^T/SFB 共用同一 `tma_mbar[s]` 管理
- **kSmemWsSFSize**：从 `Base::kSmemSFSize + kBlockN*4` 增加到 `Base::kSmemSFSize + 3*kBlockN*4`（SFA 512B + SFB[0] 512B + SFB[1] 512B + SFV[0] 512B + SFV[1] 512B = 2560B）
- **expect_tx**：每 stage 加上 `kSmemSFVBytes`（kBlockN×4=512B）
- **math warp**：移除标量 GMEM SFV 读取，改从 `smem_sfv_ptr[math_stage]` 进行 s2r SFV
- **`v_block_descale_head_stride`**：C++ 侧从 `sf_v_packed.size(1)`（扩展后的 total_tokens）读取
- **需更新的调用方**：`sweep_accuracy.py` 和 `bench_hstu_attn_sm120.py` 均需在 `pack_descale_to_e8m0x4_int32` 后加 `.repeat_interleave(128, dim=1)`；`cuda_hstu_attention.py`（高层入口）已更新

### 关键 Bug：TMA descriptor globalDim < boxDim

- **现象**：`Error: Failed to initialize the TMA descriptor 1`，`globalDim = (1,1,1,1,1)`，`boxDim = (128,1,1,1,1)`
- **根因**：`sweep_accuracy.py` 直接调用 C++ op，未做 `repeat_interleave`，导致 `sf_v_packed.size(1) = 1`（SEQ=128 时 total_blocks=1），TMA descriptor 的 `globalDim[0]=1 < boxDim[0]=128` → 非法指令
- **修复**：在 `sweep_accuracy.py` 和 `bench_hstu_attn_sm120.py` 中加 `repeat_interleave(128, dim=1)`

### 验证结果（2026-04-04）

**sweep_accuracy.py**（quant_mode=2）：6/6 PASS，fp8_gt_cos ≈ 0.9996~0.9997

**hstu_test.py 14 explicit @example cases**：14/14 PASS

测试日志：`1test_results/024_hstu8_examples_qm2.log`

### Phase 8 性能结果（2026-04-04，RTX PRO 6000 Blackwell SM120）

**kernel-only**（bs=4, h=16, d=128，full attention）：

| seq  | BF16    | BF16 TFLOPS | FP8 (qm=2) | FP8 TFLOPS | FP8/BF16 |
|------|---------|-------------|------------|------------|----------|
| 512  | 0.045ms | 191         | 0.093ms    | 93         | 2.07x 慢 |
| 1024 | 0.121ms | 285         | 0.240ms    | 143        | 1.98x 慢 |
| 2048 | 0.453ms | 304         | 0.786ms    | 175        | 1.73x 慢 |
| 4096 | 1.523ms | 361         | 2.805ms    | 196        | 1.84x 慢 |

**与 Phase 7 对比**（FP8/BF16 比值改善）：Phase 7 约 1.72~2.0x，Phase 8 约 1.73~2.07x（同 GPU boost clock 下比值略有波动，属测量噪声范围，总体基本持平）。

benchmark 日志：`2benchmark_results/007_e10772bc_gpu2347MHz_phase8_tma_sfv.log`

### 后续优化方向（更新）
- [ ] **消除 SMEM transpose**：Python 侧 V 列主序存储，TMA 直接写 K_SW128，预计节省每 tile 约 2×bar.sync + 16KB 搬运
- [ ] **nsys profile**：量化各部分耗时（GEMM1/GEMM2/transpose/SFB+SFV load），确定实际瓶颈
- [ ] **GQA 支持**：SFV 目前假设 h=h_k（非 GQA），GQA 场景需改用 bidh 而非 bidh_kv 索引 SFV

---

## Phase 9 进行中：12-warp 结构 + setmaxnreg 216（2026-04-08）

**目标**：通过 `setmaxnreg` 动态重分配寄存器，让 math warp 获得 216 个寄存器（而非编译器静态分配的 ~176），减少 register spill（LDL/STL 指令），提升 GEMM 吞吐。

### 当前 warp 结构

```
warp 0-7 (256线程) = math warps（WG0 + WG1，2 个完整 warpgroup）
  → setmaxnreg.inc 216 → 每线程 216 个寄存器
warp 8-11 (128线程) = load warpgroup（WG2，1 个完整 warpgroup）
  → setmaxnreg.dec 64 → 每线程 64 个寄存器
    warp 8：active load warp（负责发射 TMA K/V^T/SFB/SFV）
    warp 9-11：idle（仅为保持 WG2 完整，使 setmaxnreg TRY_ALLOC 的 WARPSYNC.ALL 能完成）
```

**总线程数**：384（`kNThreads = 384`）

### 关键设计原因

- `setmaxnreg.inc` 的 PTX 实现在 SASS 层面是 `USETMAXREG.TRY_ALLOC.CTAPOOL`
- TRY_ALLOC 在寄存器池不足时会进入 retry loop，loop 内含 `WARPSYNC.ALL`
- `WARPSYNC.ALL` 要求同一 warpgroup 内所有 4 个 warp 都到达该指令，才能继续
- 若 warpgroup 不完整（如原 9-warp 方案中 WG2 仅有 warp 8），WARPSYNC.ALL 永远无法完成 → deadlock
- **修复**：扩展到 12 warp，WG2 = warp 8-11（完整），warps 9-11 空闲但参与所有同步

### 寄存器预算分析（setmaxnreg 后）

| 变量 | 寄存器 | 说明 |
|------|--------|------|
| acc_o | 64 | GEMM2 跨 tile 累加器，全程存活 |
| acc_s | 64 | GEMM1 当前 tile 输出，GEMM1 期间存活 |
| Q/K/Vt fragment | ~80 | 各 ~26-32 regs |
| SF + 循环变量 | ~20 | sfb/sfv/loop/addr |
| **合计** | **~228** | **仍超出 216 → 少量 spill** |

### 代码关键点

- `kernel_traits.h`：`kNLoadWarps = 4`，`kNThreads = 384`
- `hstu_fwd_kernel_fp8_ws.h`：load warp 入口 `setmaxnreg.dec 64`；math warp 入口 `setmaxnreg.inc 216`；`is_active_load = (tidx < kNMathThreads + 32)`，warp 8 做 TMA，warp 9-11 参与 S1-S5 syncthreads 后 idle

### 验证结果（2026-04-08）

**sweep_accuracy.py**（quant_mode=2）：全部 PASS，cos_sim ≈ 0.9985~0.9987

**hstu_test.py 14 explicit @example cases**：14/14 PASS

benchmark 日志：`2benchmark_results/022_147b3416_gpu2212MHz_setmaxnreg_216.log`

### Phase 9 性能结果（2026-04-08，RTX PRO 6000 Blackwell SM120，gpu=2212MHz）

**kernel-only**（bs=4, h=16, d=128，full attention）：

| seq  | BF16    | BF16 TFLOPS | FP8 (qm=2) | FP8 TFLOPS | FP8/BF16 |
|------|---------|-------------|------------|------------|----------|
| 512  | 0.044ms | 197         | 0.068ms    | 127        | 1.55x 慢 |
| 1024 | 0.117ms | 294         | 0.183ms    | 188        | 1.56x 慢 |
| 2048 | 0.439ms | 313         | 0.668ms    | 206        | 1.52x 慢 |
| 4096 | 1.531ms | 359         | 2.325ms    | 237        | 1.52x 慢 |

**与 Phase 8 对比**（同归一化后，FP8 TFLOPS 显著提升）：

| seq  | Phase 8 FP8 TFLOPS | Phase 9 FP8 TFLOPS | 提升    |
|------|--------------------|--------------------|---------|
| 512  | 93                 | 127                | +37%    |
| 1024 | 143                | 188                | +31%    |
| 2048 | 175                | 206                | +18%    |
| 4096 | 196                | 237                | +21%    |

注：Phase 8 在 2347MHz，Phase 9 在 2212MHz，实际提升幅度经频率修正后约 +27~51%，收益来自 math warp 寄存器从 ~176 提升至 216、register spill（LDL/STL）大幅减少。

### 后续优化方向（Phase 9 后）
- [x] **消除 SMEM transpose**：Python 侧 V 列主序存储，TMA 直接写 K_SW128（Phase 10 完成）
- [ ] **进一步减少 spill**：acc_o(64) + acc_s(64) = 128 regs 同时存活，加上 fragment 约 228 regs 超出 216 → 少量 spill 仍存在
- [ ] **nsys profile**：量化各部分耗时（GEMM1/GEMM2），确定剩余瓶颈
- [ ] **GQA 支持**

---

## Phase 10 完成：消除 SMEM Transpose（col-major V，2026-04-13）

**目标**：通过 Python 侧将 V 改为列主序（token 轴 stride-1）+ kernel 侧 `SmemLayoutVt_TMA` 改为 K_SW128，使 TMA 直接写入 GEMM2 兼容的 SMEM layout，彻底消除每 N-block 迭代中的 SMEM→SMEM transpose（2×`bar.sync 1,256` + 16KB copy）。

### 关键原理

| layout | SMEM fast axis | GMEM 要求 |
|--------|---------------|-----------|
| MN_SW128（旧） | kHeadDim | V d 轴 stride-1（行主序）|
| K_SW128（新） | kBlockN | V token 轴 stride-1（列主序）|

Python 侧 `v_fp8.permute(2,1,0).contiguous().permute(2,1,0)` → shape `[n,h,d]` strides `[1, n, h*n]`，token 轴 stride-1，满足 K_SW128 TMA 要求。

### 实现内容

- **`kernel_traits.h`**：`SmemLayoutVt_TMA` 从 `GMMA::Layout_MN_SW128_Atom<Element>{}` 改为 `SmemLayoutAtomSW128{}`（即 K_SW128）
- **`hstu_fwd_kernel_fp8_ws.h`**：删除 SMEM transpose（`bar.sync B` + 16KB copy loop + `bar.sync C`），s2r V^T 直接从 `sVt_cur`（K_SW128）读取
- **`hstu.h`**：新增 `v_d_stride` 字段（V 的 d 轴 stride；行主序=1，列主序=h*total_k）
- **`hstu_ops_gpu.cpp`**：`v_d_stride = v.stride(-1)`，`v_row_stride = v.stride(-3)`；支持 `[n,h,d]` 和 `[d,h,n]` 两种 V 布局检测
- **`hstu_fwd_kernel.h`**：TMA descriptor 从硬编码 `_1{}` 改为 `params.v_d_stride`
- **Python（4 处）**：`v_fp8 = v_fp8.permute(2,1,0).contiguous().permute(2,1,0)`（sweep_accuracy.py、bench_hstu_attn_sm120.py、ncu_hstu_attn.py、cuda_hstu_attention.py）

### 验证结果（2026-04-13）

**sweep_accuracy.py**（quant_mode=2）：6/6 PASS，fp8_gt_cos ≈ 0.9985~0.9987

**hstu_test.py 14 explicit @example cases**：14/14 PASS

测试日志：`1test_results/072_hstu8_examples_qm2.log`

### Phase 10 性能结果（RTX PRO 6000 Blackwell SM120，bs=4, h=16, d=128，full attention）

kernel-only（2benchmark_results/029_1e241bfff_py_v_transpose.log）：

| seq  | BF16 TFLOPS | FP8 TFLOPS | FP8/BF16 |
|------|-------------|------------|----------|
| 512  | 191         | 180        | -5.9%    |
| 1024 | 287         | 281        | -1.9%    |
| 2048 | 306         | 310        | **+1.4%** |
| 4096 | 357         | 370        | **+3.5%** |

**与 Phase 9 对比**（Phase 9 gpu=2212MHz）：

| seq  | Phase 9 FP8 TFLOPS | Phase 10 FP8 TFLOPS | 提升   |
|------|--------------------|--------------------|--------|
| 512  | 127                | 180                | +42%   |
| 1024 | 188                | 281                | +49%   |
| 2048 | 206                | 310                | +50%   |
| 4096 | 237                | 370                | +56%   |

**关键结论**：seq≥2048 时 FP8 已超过 BF16，seq=1024 基本持平（-2%）。消除 SMEM transpose 是迄今最大的单项性能提升。

### 后续优化方向（Phase 10 后）
- [ ] **Phase 11 - 消除 SMEM transpose**：使用 `SM100_U8x16_LDSM_T`（`ldmatrix.m16n16.x2.trans.b8`），在 LDSM 从 SMEM 拷贝到寄存器时同步完成转置，省掉 MN_SW128→K_SW128 的协作转置循环（2×`bar.sync 1,256` + 16KB copy）
- [ ] **Phase 12 - 原生 SM120 CuTe 类型**：用 `cute/arch/copy_sm100.hpp`、`cute/arch/mma_sm120.hpp`、`cute/atom/copy_traits_sm100.hpp`、`cute/atom/mma_traits_sm120.hpp` 替换当前 `mma_sm89.hpp`/`mma_traits_sm89.hpp`/`mma_traits_sm90_gmma.hpp`
- [ ] **进一步减少 spill**：acc_o(64) + acc_s(64) = 128 regs 同时存活，加上 fragment 约 228 regs 超出 216 → 少量 spill 仍存在
- [ ] **nsys profile**：量化 GEMM1/GEMM2/SFB+SFV load 各部分耗时，确定剩余瓶颈
- [ ] **GQA 支持**：SFV 目前假设 h=h_k（非 GQA）

---

## Phase 11 完成（结论：数值不通过，代码保留）

**目标**：消除每个 N-block 迭代中的协作 SMEM 转置开销（2×`bar.sync 1,256` + 16KB copy loop）。

### 实施结果（2026-04-16）

**方案 A：`make_tiled_copy_B` + `SM100_U8x16_LDSM_T`**
- **结果**：编译失败（`copy_traits.hpp:128: static_assert: Copy_Traits: src failed to vectorize`）
- **根因**：`make_tiled_copy_B` 要求每个 lane 的 source fragment 能被 `recast<uint128_t>` 成恰好 1 个 16B 向量，而 MN_SW128 layout 通过 `partition_S` 得到的 per-thread source slice 不满足此条件
- **结论**：SM100_U8x16_LDSM_T 与 `make_tiled_copy_B` + MN_SW128 从根本上不兼容

**方案 B（Fallback）：K_SW128 重解释 `sVt_cur`，使用现有 `s2r_copy_B2`**
- **结果**：编译通过，数值不通过（fp8_gt_cos ≈ 0.02，远低于 0.995）
- **根因**：MN_SW128 和 K_SW128 的物理字节地址对同一逻辑索引 [i,j] 是不同的——`addr_MN(k,n) ≠ addr_K(k,n)`。Codex 的 `addr_MN(k,n) == addr_K(n,k)` 声明在理论上描述的是一种坐标变换，但用同一指针直接重解释 layout 实际上读取了不同的物理字节，产生了彻底错误的数据（cos_sim ≈ 0）
- **测试日志**：`1test_results/081_phase11_k_sw128_reinterpret.log`
- **当前代码状态**：删除了 transpose loop + 2×bar.sync，改为 K_SW128 重解释（数值错误状态，已保留不回退）

**方案 C：直接内联 PTX + 手算 swizzle 地址**
- **结果**：编译通过，运行时 `CUDA error: misaligned address`
- **根因**：`ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8` 是 SM100A 专属 PTX 指令（需要 `sm_100a` ISA 和 `CUTE_ARCH_LDSM_SM100A_ENABLED`）。SM120（消费级 Blackwell / RTX PRO 6000）不属于 SM100A，该指令根本不存在于 SM120 ISA 中。运行时非法指令错误被 GPU 报告为 "misaligned address"，这是假象——问题是 ISA 不兼容，而非地址计算错误。
- **确认来源**：`external/cutlass/include/cute/arch/copy_sm100.hpp` 中 `SM100_U8x16_LDSM_T::copy()` 被 `#if defined(CUTE_ARCH_LDSM_SM100A_ENABLED)` 完整包裹；SM120 kernel 目录中无任何地方定义此宏。
- **关键教训**：无论地址计算多么精准，都无法绕过硬件 ISA 层的根本限制。

### 技术结论

- **MN_SW128 → K_SW128 重解释**：对 128×128 正方形 tile 不成立，两种 layout 物理地址不同，不能用指针重解释代替 SMEM 显式拷贝
- **LDSM_T + make_tiled_copy_B**：需要 source tensor 的 per-thread slice 满足 uint128_t 向量化条件，MN_SW128 在此路径下不满足
- **`ldmatrix.m16n16.x2.trans.b8` 硬件不兼容**：该 PTX 指令属于 SM100A ISA，**SM120 不支持**，任何方式内联此 PTX 都会导致运行时非法指令错误
- **SMEM transpose 无法简单消除**：在 SM120 上，cooperative SMEM transpose（16KB copy + 2×bar.sync）仍是 MN_SW128→K_SW128 layout 转换中不可绕过的步骤（Phase 10 的 Python 侧 V 列主序化已经消除了这个需求）

### Phase 11 修复与 swizzle 优化最终结果（2026-04-16）

**根因修复**（fp8_gt_cos 0.12 → 0.9996）：
- 错误假设 fragment shape = (2, 16, 4) = 128 uint32/thread（全 128 N-列）
- 正确：fragment shape = (2, 4, 4) = 32 uint32/thread（每 warp 仅 32 N-列）
- BS2 AtomLayout<_2,_4,_1> 有 4 个 N-warp，`n_warp = (tidx_math / 32) % 4`
- base 公式：`(ni & 1) + 8*(ni >> 1)`（移除错误的 `di` 外层循环）

**MN_SW128 swizzle 优化**（减少 bank conflict 从 16-way 至 4-way）：
- `SmemLayoutVt_TMA` 改为 `tile_to_shape(GMMA::Layout_MN_SW128_Atom<Element>{}, [kHeadDim, kBlockN])`
- LDSM_T 地址加入 `swizzle_xor = (n_off & 7) << 4`，16B 对齐保持

**性能结果**（benchmark log：`2benchmark_results/034_89991df1c_ldsmt_w_swz.log`，RTX PRO 6000 SM120，bs=4, h=16, d=128）：

| seq  | BF16 TFLOPS | FP8 TFLOPS | FP8/BF16 |
|------|-------------|------------|----------|
| 512  | 192 (full) / 251 (causal) | 176 / 243 | -8.4% / -2.9% |
| 1024 | 284 / 419 | 293 / 433 | **+3.0% / +3.4%** |
| 2048 | 326 / 532 | 340 / 599 | **+4.5% / +12.6%** |
| 4096 | 361 / 628 | 387 / 724 | **+7.1% / +15.2%** |

seq≥1024 FP8 全面超越 BF16；seq=4096 causal 领先 15.2%。

### 后续方向（Phase 13+）

详见 PLAN.md Phase 13 章节。

---

## Phase 12：代码整洁化（已完成，2026-04-17）

**目标**：消除历史积累的代码腐化，使 `hstu_blackwell_sm120/` 目录结构清晰、自包含、可维护。不涉及任何功能修改或性能优化，纯代码质量改善。

### 背景与现状问题

| # | 问题 | 影响 |
|---|------|------|
| 1 | `hstu_fwd_kernel.h:60` 使用绝对路径 include 6KD 目录 | 代码不可移植，依赖外部目录结构 |
| 2 | FP8 调用链含死代码 `run_hstu_fwd_sm120_fp8_tma_impl`（Phase 5 遗留，无调用入口） | 代码体积膨胀，误导阅读者 |
| 3 | `cute` include 散落在 `hstu_fwd_kernel.h`、`utils.h`、6KD header 中 | include 顺序隐式依赖，难以维护 |
| 4 | `kernel_traits.h` 混用 SM80/SM89/SM120 三套 MMA 定义，其中 SM89 的 `MMA_Atom_Arch` 和 `TiledMma` 对 WS FP8 路径是死代码 | 读者无法判断哪套定义实际在用 |

### 关键发现（经 Codex 二次确认）

- **WS FP8 路径已在用 SM120 原生 QMMA**：`SM120BlockScaledBuilder::TiledMma` 使用 `SM120::BLOCKSCALED::SM120_16x8x32_TN_VS`，Phase 12 的实质是把这个定义从 6KD 搬进本地 header，消除外部依赖
- **SM89 MMA_Atom_Arch 在 WS 路径是死代码**：WS 内核完全不用 `Kernel_traits::MMA_Atom_Arch/TiledMma`，全程直接用 `BS1/BS2::TiledMma`；SM89 trait 只在 Has_rab=true 的非 WS 回退路径中仍然使用
- **`run_hstu_fwd_sm120_fp8_tma_impl` 是纯死代码**：`run_hstu_fwd_sm120` dispatcher 只走 WS 或 Has_rab 回退，Phase 5 的 TMA-only（非 WS）impl 没有任何调用入口

### 实施子步骤（按顺序，每步独立可验证）

**Step 1：删除死代码** — `run_hstu_fwd_sm120_fp8_tma_impl`

- 删除 `hstu_fwd_kernel.h:1539-1617` 的 `run_hstu_fwd_sm120_fp8_tma_impl` 函数及相关 TMA params struct
- 同时删除 `hstu_fwd_kernel.h` 中 FP8 非 WS generic kernel body（`else{}` 分支内的 Phase 4 路径，约 800 行）
- 风险：确认 `run_hstu_fwd_sm120_fp8_tma_impl` 在全仓无调用后方可删除
- 验证：编译通过 + 14/14 examples PASS

**Step 2：新建 `sm120_qmma_builder.h`，迁移 SM120BlockScaledBuilder 最小必要子集**

- 在 `hstu_blackwell_sm120/` 目录新建 `sm120_qmma_builder.h`
- 仅迁移 WS 内核实际使用的部分：
  - `TiledMma`（= `SM120::BLOCKSCALED::SM120_16x8x32_TN_VS` + AtomLayout + PermMmaTileN）
  - `SmemCopyAtomA/B`（= `Copy_Atom<SM75_U32x4_LDSM_N, float_e4m3_t>`）
  - `SmemLayoutAtomA/B`（= `GMMA::Layout_K_SW128_Atom<float_e4m3_t>`）
  - `SmemLayoutSFA/SFB`
  - 静态方法：`partition_fragment_SFA/SFB`、`transform_fragment_for_qmma`、`get_layoutSFA/SFB_TV`
- 不迁移：store/TMA/shared-storage 等 6KD 用于 GEMM standalone 的其他部分
- 风险：静态方法的 CuTe layout 计算必须保持二进制语义一致；迁移后需跑 sweep_accuracy.py 验证
- 验证：编译通过 + sweep_accuracy.py 6/6 PASS + 14/14 examples PASS

**Step 3：删除 6KD include，改用本地 `sm120_qmma_builder.h`**

- `hstu_fwd_kernel.h:57-60`：删除注释+绝对路径 include，改为 `#include "sm120_qmma_builder.h"`
- 同步在 `kernel_traits.h` 中添加 `#include <cute/atom/mma_traits_sm120.hpp>`（SM120 QMMA 所需）
- 风险：`CUTE_ARCH_TMA_SM120_ENABLED` 宏必须在 `copy_sm90_tma.hpp` 之前定义（当前在 `hstu_fwd_kernel.h:45`），收口时需保住此顺序
- 验证：同 Step 2

**Step 4：收口 CuTe includes 到 `kernel_traits.h`**

- `hstu_fwd_kernel.h:51-52` 删除 `cute/tensor.hpp` 和 `cute/arch/copy_sm90_tma.hpp`，确认已通过 `kernel_traits.h` 间接引入
- `utils.h` 中的 cute include 转为依赖 `kernel_traits.h` 的间接传递（utils.h 是私有 header，`kernel_traits.h` → `utils.h` 顺序保证）
- 风险：include 传递顺序变化可能导致某些前向声明失效
- 验证：编译通过

**Step 5：清理 `kernel_traits.h` 死 MMA 定义**

- 删除 `Hstu_fwd_kernel_traits_sm120_fp8` struct 中的 `MMA_Atom_Arch = SM89` 和对应 `TiledMma` 定义
- 删除 `kernel_traits.h` 中 `cute/arch/mma_sm89.hpp` 和 `cute/atom/mma_traits_sm89.hpp` 的 include（前提：Has_rab 回退路径也已切换到 SM120 QMMA，或接受保留 SM89 include 仅用于 BF16 base struct）
- 保留：BF16 base struct 的 `SM80 MMA_Atom_Arch`（BF16 活路径在用）
- 风险：Has_rab 回退路径 `run_hstu_fwd_sm120_impl<..., Is_fp8=true>` 仍使用 `Hstu_fwd_kernel_traits_sm120_fp8::TiledMma`（SM89），该路径若仍保留则不能删 SM89 定义
- 验证：编译通过 + 14/14 examples PASS（含 Has_rab case）

### 文件改动汇总

| 文件 | 改动 |
|------|------|
| `hstu_blackwell_sm120/sm120_qmma_builder.h` | **新建**：SM120BlockScaledBuilder 最小子集 |
| `kernel_traits.h` | 新增 `mma_traits_sm120.hpp` include；删除 SM89 include（Step 5 后）；删除 FP8 traits 中 SM89 dead defs |
| `hstu_fwd_kernel.h` | 删除 6KD 绝对路径 include；删除死代码（Phase 5 impl + Phase 4 generic FP8 body）；删除 cute includes（已在 kernel_traits.h 引入） |
| `hstu_fwd_kernel_fp8_ws.h` | 无需改动（已经直接用 BS1/BS2） |

### 完成结果（2026-04-17）

commit `0aa00eca Refactor code`：
- `sm120_qmma_builder.h` 新建 296 行，SM120BlockScaledBuilder 最小子集本地化，消除 6KD 绝对路径依赖
- `hstu_fwd_kernel.h` 净删除 437 行：`run_hstu_fwd_sm120_fp8_tma_impl` 死代码 + Phase 4 generic FP8 kernel body 全部清除
- `kernel_traits.h` 精简 56 行：SM89 废弃 MMA 定义清理，FP8 traits 添加注释说明 WS 路径直接用 BS1/BS2

---

## Phase 13：P staging 同步优化（规划中）

**目标**：减少每个 N-block 迭代中 GEMM1→GEMM2 P 矩阵中转的 barrier 开销。

### 当前 P staging 路径及开销

每个 N-block（N_total 次）：
```
acc_s (F32, D-layout) → F32→FP8 convert → bar.sync 1,256 → SMEM write (16 STS/thread)
  → bar.sync 1,256 → LDSM (16 LDS/thread) → tCrP (FP8, A-layout)
```

- 2×`bar.sync 1,256`（256线程全局同步，代价高）
- 16 STS.128 + 16 LDS.128 per thread（32 SMEM 操作）

### 为什么 warp shuffle 无法完全替代 SMEM

**技术根因**（经 Codex 双重确认）：

- `Stages_=4`（BS1/BS2 的 `AB_Stages=4`）对应 GEMM1 的 **HeadDim-K inner dimension** 的 4 个累积步骤（K=32 × 4 = 128），与 N 方向（P 的列）**正交**
- GEMM1 K-step s 对 **所有** N-warp 的 P 列都有贡献，即 `P[m,n] = Σ_{s=0}^{3} Σ_{k∈[32s,32s+32)} Q[m,k]·K[n,k]`——P[:,j\*32:(j+1)\*32) 的最终值需等全部 4 个 K-step 完成
- 与此同时，N-warp j 持有 P[:,j\*32:(j+1)\*32) 的 D 格式数据，但 GEMM2 同一 M-half 的其他 N-warp 也需要该数据→需要 1→4 跨 warp 广播
- `__shfl_sync` 只在 32-lane warp 内工作，跨 warp 广播必须通过 SMEM

**能做的 within-warp D→A 转换**（仅限 own K-slice）：
- N-warp j 持有 P[:,j\*32:(j+1)\*32) 的 4 个 D atom（各 8 N-col），可用 group-of-4 内的 `__shfl_sync` 将 D 格式重排为 A 格式
- A-lane (q,t_A=0) 需 D-lane t_D=0（own）和 t_D=1 的 FP8 数据；A-lane t_A=1 需 D-lane t_D=2 和 t_D=3 的数据（均 within-warp）
- 但这只覆盖 1/4 GEMM2 K-step，其余 3/4 仍需 SMEM → 无法消除 bar.sync

### 可行优化方案

**方案 A（主要）：256-thread → 2×128-thread barrier**
- 各 M-half（warps 0-3 / warps 4-7）的 P rows 不重叠（P[0..63] 和 P[64..127]）
- 两个 M-half 可独立同步，各自用 128-thread barrier
- `bar.sync 1,256 × 2` → `bar.sync A,128 + bar.sync B,128`（可并发）
- 预期收益：同步代价减半

**方案 B（可选）：消除第一个 bar.sync**
- 第一个 `bar.sync` 的目的：等 sK（SMEM）被 GEMM1 消费完，再覆盖写 P
- `sPbuf = sK_cur`（alias）是原因
- 若给 P 分配独立 SMEM 区域（不与 sK 共用），第一个 bar.sync 可消除
- 代价：需额外 16KB SMEM（128×128 FP8），需检查 SMEM 预算
- 当前 kSmemSize 已很紧，可能不可行

### 验证流程
编译通过 + sweep_accuracy.py 6/6 PASS + run_hstu8_examples.sh 14/14 PASS + benchmark 对比 Phase 12 基线。

---

## 核心文件

### SM120 原生内核（当前主目录）
`fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/`
- `hstu_fwd_kernel.h` — 前向内核（BF16 + FP8 非WS路径 + include WS文件）★核心
- `hstu_fwd_kernel_fp8_ws.h` — Phase 6 WS FP8 内核主体（由 hstu_fwd_kernel.h include）★
- `kernel_traits.h` — BF16 + FP8 内核 traits（SmemLayoutVt_TMA = K_SW128，Phase 10 后）
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

**环境版本**：PyTorch 2.10.0（nvcr.io/nvidia/pytorch:26.01-py3）；CUDA 13.1。

---

## 功能验证流程

每次代码修改后必须依次执行以下两步验证。

**日志命名规范**：`1test_results/NNN_<修改内容简述>.log`，NNN 为三位数字顺序编号。

### 步骤 1：sweep_accuracy.py（数值正确性）

```bash
HSTU_SWEEP_FP8_QUANT_MODE=2 \
python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/sweep_accuracy.py \
  2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/1test_results/NNN_xxx.log
```

检查各配置的 `fp8_gt_cos` ≥ 0.995。

### 步骤 2：run_hstu8_examples.sh（14 个 example case）

```bash
bash /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/run_hstu8_examples.sh
```

检查输出为 `14/14 passed`，覆盖 causal/rab/drab/local/context/target/arbitrary 等所有掩码类型。

---

## 性能分析流程

每次优化后按以下三步分析，定位瓶颈并制定下一步方向。

**文件命名规范**：
- benchmark log：`2benchmark_results/NNN_<commit_short>_gpu<MHz>MHz_<描述>.log`
- profile 产物：`3profile_results/NNN_<描述>.*`（nsys-rep、ncu-rep、csv、stats.log）
- SASS：`4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full.sass`

### 前置：锁定 GPU 频率

跑 benchmark 前必须锁频，保证结果可复现、跨 session 可比较。

```bash
# 锁定（需要 sudo 或 persistence mode 已开启）
sudo nvidia-smi -lgc 2407   # 2407 = RTX PRO 6000 Blackwell 的最大 boost clock

# 验证
nvidia-smi --query-gpu=clocks.current.graphics,clocks.max.graphics \
  --format=csv,noheader,nounits

# 跑完后解锁
sudo nvidia-smi -rgc
```

### 步骤 1：bench_hstu_attn_sm120.py（整体延迟 / TFLOPS）

```bash
REPO=/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu
COMMIT=$(git -C ${REPO} rev-parse --short HEAD)
CLOCK=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits | head -1 | tr -d ' ')
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user \
python ${REPO}/fbgemm_gpu/experimental/hstu/benchmark/bench_hstu_attn_sm120.py \
  2>&1 | tee ${REPO}/2benchmark_results/NNN_${COMMIT}_gpu${CLOCK}MHz_<描述>.log
```

关注 FP8/BF16 延迟比值和 TFLOPS，与上一版本对比。

### 步骤 2：run_profile.sh（nsys + ncu）

```bash
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu
./run_profile.sh NNN <描述>
# 例如：./run_profile.sh 031 phase10
```

**输出产物**（均在 `3profile_results/`）：

| 产物 | 说明 |
|------|------|
| `NNN_<desc>.nsys-rep` | nsys timeline trace |
| `NNN_<desc>_stats.log` | nsys stats CSV（kernel 耗时汇总） |
| `NNN_<desc>_ncu_fp8.ncu-rep` | FP8 WS kernel ncu full profile |
| `NNN_<desc>_ncu_bf16.ncu-rep` | BF16 kernel ncu full profile |
| `NNN_<desc>_ncu_fp8.csv` | FP8 ncu 指标 CSV |
| `NNN_<desc>_ncu_bf16.csv` | BF16 ncu 指标 CSV |

脚本结束后自动打印关键指标摘要：寄存器数、LDL/STL spill、warp stall 分布、SM 吞吐。

profile 目标脚本：`benchmark/ncu_hstu_attn.py`（固定 seq=512, bs=1/4, h=16, d=128）。

注：nsys 2025.6.1 report 名称为 `cuda_gpu_kern_sum,cuda_api_sum,nvtx_pushpop_sum`。

**分析重点**（ncu CSV / 摘要）：
- `launch__registers_per_thread`：实际寄存器分配（math warp 目标 216）
- `l1tex__t_sectors_pipe_lsu_mem_local_op_ld/st.sum`：LDL/STL spill 次数，越低越好
- `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct`：GMEM/SMEM 依赖 stall
- `smsp__warp_issue_stalled_math_throttle_per_warp_active.pct`：MMA pipe 压力
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`：SM 利用率

### 步骤 3：dump_sass_fp8_ws.sh（SASS / reg spill）

```bash
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu
./dump_sass_fp8_ws.sh
# 输出：4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full.sass
```

脚本输出 MaxReg（最大寄存器号），直接反映编译器实际寄存器分配。SASS 中搜索 `LDL`/`STL` 指令可定位 spill 热点。

**分析方向**：
- spill 多（LDL/STL > 0）→ 减少同时存活寄存器（acc_o 64 + acc_s 64 是主要压力）
- Long Scoreboard stall 高 → 增加 prefetch 距离或减少 SMEM 依赖
- Math Throttle 高 → 接近 compute bound，减少非 MMA 指令
- 量化开销主导 → 考虑将量化融合进 kernel

---

## 目录结构

```
fbgemm_gpu/experimental/hstu/
├── benchmark/
│   ├── bench_hstu_attn.py               # 吞吐量基准测试（通用）
│   ├── bench_hstu_attn_sm120.py         # SM120 专用 benchmark ★
│   ├── ncu_hstu_attn.py                 # ncu profile 目标脚本（固定 seq/bs/h/d）★
│   └── profile_hstu_attn.py             # nsys 性能分析脚本（旧版，已被 run_profile.sh 覆盖）
├── hstu/
│   ├── cuda_hstu_attention.py           # Python 入口，SM 版本分发
│   └── library.py                       # 命名空间包检测（已修复）
├── src/
│   ├── generate_kernels.py              # 生成 Hopper/Blackwell .cu 文件
│   ├── hstu_ampere/                     # Ampere (SM80) 原生内核（upstream）
│   ├── hstu_blackwell/                  # SM100 原生内核（upstream，CuTe-DSL 实现）
│   ├── hstu_blackwell_sm120/            # SM120 最终版本（当前主目录）★
│   │   ├── hstu_fwd_kernel.h            # 前向内核（BF16 + FP8 block-scale + include WS文件）★
│   │   ├── hstu_fwd_kernel_fp8_ws.h     # FP8 WS 内核主体（Phase 6+）★
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
├── run_profile.sh                       # nsys + ncu 一键 profile 脚本 ★
│                                        # 用法：./run_profile.sh NNN <描述>
│                                        # 产物：3profile_results/NNN_<描述>.*
├── dump_sass_fp8_ws.sh                  # 从 .so 提取 FP8 WS kernel SASS ★
│                                        # 用法：./dump_sass_fp8_ws.sh
│                                        # 产物：4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full.sass

（项目根目录）
├── sweep_accuracy.py                    # FP8 vs BF16 准确度扫描 ★
├── memory.md                            # Claude 对话记忆（跨会话）
└── 1test_results/
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
