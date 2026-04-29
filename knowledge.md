# HSTU 架构实现对比：Blackwell (SM120) vs Hopper (SM90) vs Ampere (SM80)

> 更新日期：2026-03-10

---

## 18. WS 分支同步死锁排查（`__syncthreads` vs `bar.sync`）

**场景**：在 FP8 WS kernel 中，为了做 Q 持久化复制，在 math 分支新增了一个 `__syncthreads()`，结果 `seq=128/256` 可跑，`seq=512` 卡死。

**根因**：
- `__syncthreads()` 是 **CTA 级** barrier，要求所有 warps（包括 load warps）参与。
- load/math 两个分支原本约定了固定的 CTA barrier 序列（S1/S2/S3/S5）。
- 在 math 分支额外插入 CTA barrier 后，序列错位：短序列时可能“偶然对齐”，长序列进入 producer/consumer 循环后会稳定死锁。

**修复原则**：
- 只需 math warps 同步时，用 `bar.sync 1, 256`（或等价 warpgroup/子集同步），不要额外加 `__syncthreads()`。
- 保持 load 和 math 分支的 CTA 级 barrier 数量与顺序严格一致。

---

## 19. `fast_silu` / FP8 pack 的寄存器降峰值策略

**背景**：在 WS FP8 内核里，`acc_s` 的 `silu + F32->E4M3 pack` 段常出现大量 `MUFU.TANH + F2FP`，并伴随 `LDL/STL`。即使 Q/SFQ 生命周期缩短后，这一段仍可能是 spill 主来源。

**策略**：
- 保持数学语义不变，新增 `fast_silu_no_unroll()` 与 `convert_type_safe_no_unroll()`；
- 用 `#pragma unroll 1` 限制该段完全展开，降低单个 basic block 的峰值临时寄存器数量；
- 仅在 WS 路径切换到 no-unroll 版本，减少对其他路径的影响。

**预期**：
- Spill（`LDL/STL`）下降；
- 可能牺牲少量 ILP，需要通过 kernel-only benchmark 验证净收益（尤其 full vs causal 差异）。

---

## 20. 为什么 `acc_s` 不能直接变成 `tCrP`

**关键区别**：
- `acc_s`：GEMM1 的 C fragment（按 MMA-C 片段布局分发到线程寄存器）
- `tCrP`：GEMM2 的 A fragment（由 K_SW128 + LDSM_N 路径定义的寄存器布局）

两者的线程内/线程间元素映射不同，不能简单“同寄存器重解释”。

**可行优化**：
- 去掉 `rP` 全量中间寄存器 tensor（避免 `acc_s` 与 `rP` 长时间重叠）；
- 改为 `acc_s` 元素直接量化后写入 `sPbuf`；
- 再通过既有 `s2r P` 路径（LDSM_N）生成正确布局的 `tCrP`。

这属于“减少 RF 峰值”的低风险优化，不改变 `tCrP` 的正确布局来源。

---

## 21. FP8 量化回归：`static_cast` vs CUTLASS NumericConverter

**现象**：去掉 `rP` 中间 tensor 后，若逐元素用 `static_cast<FP8Elem>` 写入 `sPbuf`，`arbitrary` case 出现明显数值回归（target/arbitrary 失败）。

**原因**：
- `arbitrary/local/causal` 路径会对 `acc_s` 注入大量 `-INF`（mask）；
- `static_cast` 的 FP8 转换语义与原先 `convert_type_safe`（CUTLASS `NumericArrayConverter`）不一致，尤其在极值/特殊值处理上。

**修复**：
- 保留“无 `rP` 中间 tensor”的优化；
- 逐元素转换改用 `cutlass::NumericConverter<FP8Elem, float>`，保持与 CUTLASS 量化语义一致。

---

## 22. FP8 WS persistent wrapper 的 SMEM 对齐约束

**现象**：把 FP8 WS kernel 改成 persistent loop 时，如果在 kernel wrapper 中新增普通 `__shared__ int` 保存 next tile，首个 FP8 case 会报 `misaligned address`。

**原因**：
- WS body 使用 `extern __shared__ char smem_[]` 作为 K/V/Q-persist/SF/mbarrier 的绝对地址基准。
- K/V/Q-persist 路径依赖 SW128/TMA 的绝对地址对齐，尤其 Q-persist offset 按 2048B swizzle period 设计。
- wrapper 里的 static shared memory 会改变 dynamic shared memory 的绝对起点；即使手动 padding static shared，也不能假设 compiler 最终布局满足 WS 的绝对对齐要求。

**修复原则**：
- persistent wrapper 不要新增普通 static shared memory。
- 需要在 CTA 内广播 next tile 时，可以在 tile 结束、所有线程 `__syncthreads()` 后临时复用 dynamic SMEM 的首字节；读出 next tile 后再同步一次，确保没有线程进入下一轮 compute 覆写该位置。
- 如果后续改为更高性能的 next-work prefetch，应把状态放进既有 WS shared layout 的显式保留区域，并把 offset 纳入 `Kernel_traits::kSmemSize`/alignment 计算。

---

## 23. FP8 WS causal 静态首尾配对 queue

**结论**：对 pure causal no-RAB FP8 WS，首尾配对的静态 launch order 比 persistent wrapper 更稳。

**有效做法**：
- pure causal launch 使用一维 `grid.x = total_tiles`；
- `blockIdx.x` 映射为 `0, total_tiles-1, 1, total_tiles-2, ...`；
- full/local/context/target/arbitrary 保持原始三维 grid，避免无收益路径被 causal queue 影响。

**原因**：
- causal 的 M tile 工作量从重到轻变化，原始顺序容易在 wave/tail 上产生不均；
- heavy/light 交错可以静态拉平 launch order；
- 不把 `hstu_compute_attn_1rowblock_sm120_fp8_ws` 包进 persistent loop，就不会因为 inline body 跨 tile 循环而引入 `STACK`/`LDL`/`STL`。

**已拒绝做法**：
- dynamic persistent queue：correctness 可过，但需要 tile 间 CTA sync/atomic，收益小且波动；
- paired persistent loop：causal 有单次高分，但 full 回退，SASS 出现 stack/local spill；
- wrapper 中 static shared state：会移动 dynamic SMEM 起点，破坏 SW128/TMA 绝对对齐。

---

## 15. Context Parallel（CP）原理与梯度缩放

**定义**：Context Parallel（CP）是把序列维度（context/tokens）切分到多个 rank 上，每个 rank 只处理该序列的一段 token。与 TP（切 hidden 维）和 PP（切层）不同，CP 主要切的是 `seq_len` 维度。

**核心收益**：
- 降低单卡激活显存（每卡只保留局部 token 的中间激活）。
- 允许更长上下文（长序列训练/推理时更容易扩展）。

**训练时的关键点（和 loss 归一化强相关）**：
- 如果某个 loss 在本地是 `partial-sum`（仅局部 token 求和，未做 token 归一），后续在 `dp_cp_group` 做梯度平均时，容易出现“额外 `1/cp_size` 缩小”，需要检查是否补偿。
- 如果某个 loss 在本地已是 `per-token mean`（先除以本地 token 数），再做 `dp_cp_group` 平均通常是预期归一化，不一定需要额外补偿。

**与 MTP / MoE aux 的常见差异**（Megatron-LM issue #3943 讨论结论）：
- MTP 主 loss 路径通常是 local per-token mean，CP 下不一定有额外缩放 bug。
- MoE aux loss 历史上出现过 partial-sum 导致的 CP 额外缩小，需要补偿（见 PR #2217 的思路）。

**一句话结论**：
> CP 本身不是 bug 来源；真正要警惕的是“loss 进入 autoscaler 前的统计口径（partial-sum vs per-token mean）”与“后续 group averaging”是否发生重复归一化。

## 16. 为什么 CP 在 loss 缩放上更容易出问题（对比 DP/TP）

**核心原因**：CP 直接切分 token 维度（样本统计维）。loss 通常就是按 token 聚合（sum/mean），所以 CP 会直接改变“本地 loss 的统计口径”。

**CP 的特殊点**：
- 每个 rank 只看到局部 token，若本地先做 `partial-sum`，再在 `dp_cp_group` 平均，可能产生额外 `1/cp_size` 缩小（重复归一化）。
- 若本地先做 `per-token mean`，再 group 平均通常是预期行为，不会自动触发同类问题。

**为什么 DP 通常没这类争议**：
- DP 是样本副本并行，各 rank 通常处理完整样本子集，loss 统计口径在 rank 间较一致。
- DDP 默认“梯度求和/平均”语义已被广泛验证，通常不会再引入“切 token 维导致的额外因子”。

**为什么 TP 通常没这类争议**：
- TP 切 hidden/channel 等特征维，不切 token 统计维；最终每个 token 的 loss 仍按完整语义计算。
- TP 的通信更多是张量重组（all-reduce/all-gather）以恢复算子语义，不直接改变“每 token 统计口径”。

**一句话**：
> CP 之所以敏感，是因为它切的正是 loss 的统计维（token）；DP/TP 通常不直接切这个维度，因此较少出现“partial-sum + group avg”带来的重复归一化问题。

## 17. CP 风险最小数值例子（cp_size=2）

设一条样本共有 4 个 token，其 token-level 损失分别是：
`[1, 3, 5, 7]`，全局 mean 应为 `(1+3+5+7)/4 = 4`。

CP=2 时切分：
- rank0 看到 `[1, 3]`
- rank1 看到 `[5, 7]`

### 错误高风险路径：local partial-sum + group average

每个 rank 先算局部和：
- `L0 = 1+3 = 4`
- `L1 = 5+7 = 12`

随后在 CP 组做平均（很多 DDP 逻辑等价于平均）：
- `L = (L0 + L1) / 2 = 8`

得到的是 8，而不是全局 mean=4。  
此时若再在别处除 token 或套 autoscaler，常会出现“口径不一致”，最终表现为梯度额外缩放（常见为多一个 `1/cp_size` 或缺一个归一因子）。

### 低风险路径：local per-token mean + group average（token 数相等时）

每个 rank 先本地 mean：
- `M0 = (1+3)/2 = 2`
- `M1 = (5+7)/2 = 6`

再做 CP 平均：
- `M = (M0 + M1) / 2 = 4`

恰好得到全局 mean。  
这就是为什么“先 per-token mean，再做 group avg”通常更稳。

### 关键提示

- 风险不在 CP 本身，而在“局部统计口径”和“后续平均规则”是否一致。
- 一旦使用 local partial-sum，就必须明确：后续是否还会平均、是否还会再除 token、这些除法是否重复。

## 18. Autoscaler 与 token 数的关系

**本质**：Autoscaler 只是“把一个标量系数稳定地注入反向梯度”。  
它本身不认识 token 语义；token 数是否参与缩放，取决于传给 autoscaler 的 loss scale 里有没有 `1/num_tokens`。

### 常见写法

若前向定义为：
`scaled_loss = base_scale * loss_sum / safe_num_tokens`

则链式法则下，反向会带出：
`grad ~ backward_scale * (base_scale / safe_num_tokens) * d(loss_sum)/dθ`

可见 token 数影响来自 `1/safe_num_tokens` 这一项，而不是 autoscaler “自动推断”出来的。

### 直觉

- token 越多，若使用 per-token mean（除以 token 数），单个 token 对总梯度贡献会按平均规则被归一。
- token 越少，`1/safe_num_tokens` 越大，局部梯度幅度会相应变大（避免小 batch/token 时梯度过小）。

### 在并行训练中的注意点

- 若本地已除 `safe_num_tokens`（per-token mean），后续再做 group averaging 通常是预期组合。
- 若本地没除 token（partial-sum），但后续有 averaging，就要检查是否缺少或重复 token 归一化。

**一句话**：
> Autoscaler 决定“按什么系数传梯度”，token 数通常通过 `loss_scale`（如 `1/num_tokens`）间接影响梯度幅度。

## 19. Megatron Issue #3943 的风险点（按数值例子）

问题链接：[Issue #3943](https://github.com/NVIDIA/Megatron-LM/issues/3943)

### 19.1 争议焦点

该 issue 讨论的是：CP 开启后，`MTPLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)` 是否会让 MTP 梯度被“多缩一次”（缺少 `cp_size` 补偿）。

维护者在讨论中给出的区分是：
- MTP 主 loss：local per-token mean 路径（通常不需要额外 CP 补偿）。
- MTP MoE aux loss：历史上 partial-sum 路径（这类路径更容易需要 CP 补偿，见 PR #2217）。

### 19.2 用同一组数字解释风险

设 `cp_size=2`，全局 token loss 为 `[1,3,5,7]`。

#### A. MTP 主 loss（低风险路径）

local per-token mean：
- rank0: `(1+3)/2 = 2`
- rank1: `(5+7)/2 = 6`

cp 组平均：
- `(2+6)/2 = 4`（与全局 mean 一致）

若 autoscaler 再乘统一系数 `β`，两边一致乘，不会引入额外 CP 因子。

#### B. MoE aux partial-sum（高风险路径）

local partial-sum：
- rank0: `1+3 = 4`
- rank1: `5+7 = 12`

cp 组平均：
- `(4+12)/2 = 8`（不是全局 mean 4）

这说明“local partial-sum + group average”口径容易不一致。若后续逻辑还按 mean 口径解释该量，就可能出现额外缩放（常见体感是梯度偏小，需要补偿 `cp_size`）。

### 19.3 在 #3943 中真正要检查的点

1. 进入 `MTPLossAutoScaler` 前是 `partial-sum` 还是 `per-token mean`。  
2. `set_loss_scale(loss_scale / num_microbatches)` 的 `num_microbatches` 是否已包含 CP 维（与 group averaging 是否重复）。  
3. 反向后 DDP 是否在 `dp_cp_group` 上做平均（平均因子是否与前向归一口径匹配）。  
4. 若 MTP 层中包含 MoE aux 分支，主 loss 与 aux loss 不可共用同一“是否补偿 CP”的结论。

**结论**：
> #3943 的核心风险不是 autoscaler 本身，而是“前向 loss 统计口径 + set_loss_scale 因子 + dp_cp_group 平均”三者是否一致。  
> MTP 主 loss 常见为低风险；MTP MoE aux partial-sum 路径是高风险重点排查对象。

## 20. `num_microbatches` 与并行维度的关系（Megatron 语境）

### 20.1 它是什么

`num_microbatches` 是一个 mini-batch 在一次 optimizer step 中被切成多少个 micro-batch（梯度累积步数）。  
常见直觉：`global_batch = micro_batch_size * data_parallel_size * num_microbatches`（忽略其他细节时的近似关系）。

### 20.2 它主要属于哪类并行

- **最直接相关**：Pipeline Parallel（PP）与 Gradient Accumulation。  
  PP 调度（1F1B 等）依赖 micro-batch 数量来填充流水线、提高吞吐。
- **也和 DP 有关系**：DP size 变大时，在固定 global batch 下常会影响需要的 `num_microbatches`。

### 20.3 为什么会在 CP 讨论里出现

在 #3943 里，争议点之一是：
`MTPLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)`。  
当某些代码路径里 `num_microbatches` 的定义/有效值受 CP 影响（例如被乘上 CP 因子，或等价地在 CP 维上重复计数）时，就可能与后续 `dp_cp_group` 平均形成“重复归一化”。

### 20.4 一句话理解

`num_microbatches` 本质是“时间维度上的分批/累积因子”（PP/GA 核心变量），  
但一旦它被用于 loss scaling，就会和 DP/CP 的“空间并行平均因子”耦合；若口径没对齐，就会引发 #3943 这类缩放争议。

## 21. Forward / Backward / Loss / Optimize 关系总览

### 21.1 一次训练 step 的主流程

1. **Forward**：`y_hat = f(x; θ)`，得到预测和中间激活（autograd tape）。
2. **Loss**：`L = loss(y_hat, y)`，把任务目标压成标量（或可规约成标量）。
3. **Backward**：计算参数梯度 `g = ∂L/∂θ`，写入每个参数的 `.grad`。
4. **Optimize**：优化器用 `θ`、`g` 以及自身状态（如动量）更新参数。

### 21.2 公式视角

- 参数更新统一写法：
  `θ_{t+1} = θ_t - update_t`
- SGD:
  `update_t = lr * g_t`
- AdamW（简化）:
  - `m_t = β1 m_{t-1} + (1-β1) g_t`
  - `v_t = β2 v_{t-1} + (1-β2) g_t^2`
  - `update_t = lr * m_t / (sqrt(v_t)+eps) + lr * wd * θ_t`

### 21.3 Optimize 需要哪些结果

优化器更新至少需要：
- 当前参数值 `θ`
- 当前梯度 `g = ∂L/∂θ`（来自 backward）
- 优化器状态（如 `m/v`、step 计数）
- 超参数（`lr`、`weight_decay`、`betas`、`eps`）

在混合精度/分布式训练中通常还需要：
- （可选）grad scaling/unscale 后的梯度
- （可选）梯度裁剪结果（global grad norm）
- （分布式）all-reduce/average 后的一致梯度

### 21.4 loss scale 与优化的连接点

若定义 `L' = α * L`，则 `∂L'/∂θ = α * ∂L/∂θ`。  
因此 `α`（来自 num_tokens、num_microbatches、autoscaler、并行平均口径）会直接影响优化器看到的梯度幅度，进而影响收敛稳定性。

## 1. MMA 指令类型

| 架构 | 指令模型 | MMA 原子 | K 步长 |
|------|----------|----------|--------|
| **Blackwell (SM120)** | Per-warp `mma.sync` | `SM80_16x8x16_F32BF16BF16F32_TN`（BF16）<br>`SM89_16x8x32_F32E4M3E4M3F32_TN`（FP8） | BF16: K=16<br>FP8: K=32 |
| **Hopper (SM90)** | Warpgroup WGMMA | `SM90_64x64x16_F32BF16BF16F32_SS_TN`等 | K=16（按 warpgroup） |
| **Ampere (SM80)** | Per-warp `mma.sync` | `SM80_16x8x16_F32BF16BF16F32_TN` | K=16 |

**关键差异**：Hopper 引入了 WGMMA（Warp Group MMA），一次操作 128 个线程（4 warp），理论计算吞吐是 mma.sync 的 2 倍。Blackwell SM120 退回到与 Ampere 相同的 per-warp `mma.sync`，但增加了 SM89 原生 FP8 MMA 支持（K=32）。

---

## 2. Kernel Traits 结构对比

### Blackwell (SM120) — BF16
```cpp
struct Hstu_fwd_kernel_traits_sm120 {
    using Element = cutlass::bfloat16_t;
    using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
    using TiledMma = TiledMMA<MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        Tile<Int<16 * kNWarps>, _16, _16>>;

    static constexpr int kBlockKSmem = (kHeadDim % 64 == 0) ? 64 : 32;
    static constexpr int kSwizzle = (kBlockKSmem == 32) ? 2 : 3;

    // Swizzle SMEM，防 bank conflict
    using SmemLayoutAtomQ = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, BF16>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, BF16>;
};
```

### Blackwell (SM120) — FP8 Phase 2
```cpp
struct Hstu_fwd_kernel_traits_sm120_fp8 {
    using Element    = cutlass::float_e4m3_t;   // GMEM + SMEM 均 FP8
    using ElementSmem = cutlass::float_e4m3_t;
    using OutputType = cutlass::bfloat16_t;     // 输出 BF16

    // SM89 FP8 MMA，K=32
    using MMA_Atom_Arch = MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>;
    using TiledMma = TiledMMA<MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        Tile<Int<16 * kNWarps>, _16, _32>>;

    static constexpr int kBlockKSmem = 32;

    // 平坦布局（无 Swizzle），保证 FP8 ldmatrix 正确性
    using SmemLayoutAtomQ = Layout<Shape<_16, _32>, Stride<_32, _1>>;
    using SmemCopyAtom = Copy_Atom<DefaultCopy, float_e4m3_t>;
};
```

> **为何 FP8 用平坦布局（无 Swizzle）**：SM89 FP8 MMA (K=32) 的 `ldmatrix.x4` 要求 16×32 FP8 K-tile 内 thread T 的数据在 `T×16 bytes` 偏移处。Swizzle 的 XOR 操作会错开地址，导致数据加载错误（cos_sim ≈ 0.31 的精度崩溃）。改为 `SmemLayoutAtomQ = Layout<_16, _32>` 后精度恢复正常。
>
> **FP8 kBlockKSmem=32 的原因**：K=32 的 MMA 原子每步处理 32 列，SMEM atom 必须与此对齐，使每个 MMA K-step 的 sub-tile 连续存储。

### Hopper (SM90)
```
- TiledMma 分两种：TiledMma0（WGMMA，计算主循环）、TiledMma1（尾声，输出累加）
- 引入 ClusterShape_MNK，支持跨 block 的 Cluster 协同
- TMA descriptor 做 SMEM 异步加载（Producer warpgroup 专职）
- Consumer warpgroup 直接读 SMEM descriptor 到寄存器
```

### Ampere (SM80)
```
与 Blackwell BF16 几乎相同（两者共享 mma.sync 模型）：
- 同款 SM80_16x8x16_F32BF16BF16F32_TN
- 同款 Swizzle<3,3,3> + 8×kBlockKSmem SMEM atom
- 同款 SM75_U32x4_LDSM_N copy atom
- 仅支持 BF16，无 FP8
```

---

## 3. FP8 处理方式

| 架构 | FP8 策略 | SMEM 数据类型 | descale 时机 |
|------|----------|--------------|-------------|
| **Blackwell Phase 2** | GMEM FP8 → SMEM FP8 → SM89 FP8 MMA | FP8 | GEMM1 后 × `descale_q × descale_k`；epilogue × `descale_v` |
| **Blackwell Phase 1**（已废弃） | GMEM FP8 → 转 BF16 → SMEM BF16 → BF16 MMA | BF16 | GEMM1 后乘 descale |
| **Hopper** | 多模式：quant_mode=0（CUDA 量化）/ 1,2,3（Python 量化）；V 转置 pipeline | FP8 | 复杂的 `permute_regs_C_to_A()` 字节重排 |
| **Ampere** | 不支持 FP8 | — | — |

**Blackwell Phase 2 FP8 计算流程（已修复版本）**：
```
1. GMEM 读 Q/K（FP8） → SMEM（FP8，平坦布局）
2. （若 Has_rab）加载 RAB（BF16）到寄存器 rRab
3. GEMM1: FP8 × FP8 → FP32 累加（清零后计算，仅 Q×K 部分）
4. 乘 descale_q × descale_k（仅对 Q×K 部分）
5. （若 Has_rab）acc_s += float(rRab)（RAB 在 descale 之后加，避免 RAB 被 descale 错误缩放）
6. 乘 alpha，SiLU 激活（FP32）
7. FP32 clamp → FP8 rP（C-fragment 寄存器布局）
8. 【SMEM Roundtrip 修复】写 rP → sP_buf (flat row-major SMEM) → __syncthreads()
   → make_tiled_copy_A + DefaultCopy 重载 → 正确的 SM89 FP8 A-fragment 布局
9. GEMM2: FP8(P) × FP8(V^T) → FP32 累加（acc_o += P × V^T）
10. 乘 descale_v / scaling_seqlen，写出 BF16 输出
```

> **RAB+descale 顺序 bug（已修复，2026-03-10）**：原代码在 Has_rab=true 时先将 RAB 填入 acc_s，再执行 GEMM，再 `acc_s *= descale_qk`，导致 `descale_qk` 错误地缩放了 RAB（不应被量化 descale）。
> 修复：先 `clear(acc_s)` → GEMM → `*= descale_qk` → 再 `+= float(rRab)`。

> **GEMM2 A 寄存器布局 bug（已修复，2026-03-10）**：`convert_layout_acc_Aregs<K=32>` 的 `_4` fold 将 C-fragment 的 `(4, MMA_M, MMA_N)` 重排为 `((4,4), MMA_M, MMA_N/4)`，每个 uint32_t 内 4 个 FP8 来自**同一列、不同行**。但 SM89 FP8 MMA ALayout 要求每个 uint32_t 内是 `{row0_K0, row0_K16, row1_K0, row1_K16}`（行-K 交替排列）。字节排列不匹配导致 GEMM2 计算错误（cos_sim ≈ 0.36）。
> 修复：通过 SMEM roundtrip：FP8 rP → **写入 flat row-major SMEM（sP_buf）** → __syncthreads() → **make_tiled_copy_A + DefaultCopy 重载** → 正确的 A-fragment 布局。
> sP_buf 复用 sQ 的 SMEM 空间（Q 已在寄存器中），kBlockM × kBlockN ≤ kSmemQSize 恒成立。

---

## 4. SMEM 布局与 Tiling 策略

### Blackwell & Ampere（mma.sync 家族）
```
SMEM atom（BF16）: Swizzle<3,3,3> + Shape<8, 64>, Stride<64, 1>
  - 8 行 × 64 列，每行 128 bytes（= 64 × 2 bytes BF16）
  - Swizzle 消除 ldmatrix 的 bank conflict

SMEM atom（FP8）: Shape<16, 32>, Stride<32, 1>  [平坦]
  - 16 行 × 32 列，每行 32 bytes（= 32 × 1 byte FP8）
  - kBlockKSmem=32，每次 MMA K-step 加载 16×32=512 bytes

Q SMEM 尺寸: kBlockM × kHeadDim（整个 Q tile 驻留 SMEM）
K/V SMEM 尺寸: kBlockN × kHeadDim（K/V tile 滑动加载）
```

### Hopper（WGMMA）
```
Producer warpgroup 使用 TMA 描述符异步加载到 SMEM
Consumer warpgroup 直接用 WGMMA 从 SMEM 描述符读数据
无需显式 ldmatrix（由 WGMMA 硬件处理地址对齐）
多级 pipeline（kStages=3+），pipeline 隐藏内存延迟
```

---

## 5. Tile 大小（utils.h 中的查找表）

| 架构 | hdim=128, 无 RAB | hdim=128, 有 RAB |
|------|-----------------|-----------------|
| **SM120 BF16** | `{128, 128, 8}` | `{64, 64, 4}` |
| **SM120 FP8** | `{128, 128, 8}` | `{128, 128, 8}` |
| **SM80 BF16** | `{128, 64, 8}` | `{64, 64, 4}` |

**注**：`{kBlockM, kBlockN, kNWarps}`。RAB（Relative Attention Bias）为 BF16，会占用额外 SMEM；具体 tile 以 `utils.h::get_tile_size_fwd_sm120` 为准。

---

## 6. 流水线与同步模型

| 架构 | 流水线深度 | 内存拷贝方式 | 同步原语 |
|------|----------|------------|---------|
| **Blackwell (SM120)** | kStages=1（单阶段） | `cp.async`（SM80 异步拷贝） | `__syncthreads()` |
| **Hopper (SM90)** | kStages=3+（多阶段 TMA 流水） | TMA（Tensor Memory Accelerator） | Cluster barrier + `bar.sync` |
| **Ampere (SM80)** | kStages=1（单阶段） | `cp.async` | `__syncthreads()` |

**关键差异**：Hopper 的 TMA + 多阶段流水线允许内存加载与计算完全重叠（生产者-消费者模型），而 SM120/SM80 每次循环都需要等待 cp.async 完成后才能计算。

---

## 7. 文件组织对比

### Blackwell (SM120)：`src/hstu_blackwell/`
```
kernel_traits.h            — BF16 + FP8 两套 traits，同文件
hstu_fwd_kernel.h          — 前向内核，模板特化区分 BF16/FP8 路径
hstu_fwd_launch_template.h — 启动模板（include kernel.h）
hstu_ops_gpu.cpp           — PyTorch 入口（hstu_varlen_fwd_120）
utils.h                    — tile 大小查找表 + 类型转换 + silu 辅助
hstu.h                     — Params 结构体（扩展自 Ampere + FP8 descale 字段）
static_switch.h            — 编译期分发宏（FP16/BF16/FP8）
block_info.h               — block 分配信息（与 Ampere 相同）
```

### Hopper (SM90)：`src/hstu_hopper/`（更复杂）
```
kernel_traits.h            — 多套 traits（含 cluster、TMA、WGMMA 配置）
hstu_fwd_kernel.h          — 两个内核：compute_attn_ws（BF16）/ compute_attn_ws_fp8（FP8）
mainloop_fwd_sm90_tma_gmma_ws.hpp — 独立的 mainloop collective
epilogue_fwd_sm90.hpp      — 独立的 epilogue collective
tile_scheduler.hpp         — 跨 cluster 的 tile 调度
```

### Ampere (SM80)：`src/hstu_ampere/`（最简单）
```
kernel_traits.h            — 单套 BF16 traits
hstu_fwd.h                 — 单一计算内核
无独立 collective 或 scheduler
```

---

## 8. 关键实现差异总结

| 方面 | Blackwell (SM120) | Hopper (SM90) | Ampere (SM80) |
|------|-------------------|---------------|---------------|
| **MMA 模型** | Per-warp `mma.sync` | Warpgroup WGMMA | Per-warp `mma.sync` |
| **FP8 支持** | 原生 SM89 FP8 MMA（Phase 2） | 多模式（CUDA + Python） | 无 |
| **流水线** | 单阶段 | 多阶段 TMA | 单阶段 |
| **内存拷贝** | `cp.async` | TMA | `cp.async` |
| **同步** | `__syncthreads()` | Cluster barrier | `__syncthreads()` |
| **FP8 SMEM 布局** | 平坦（无 Swizzle，kBlockKSmem=32） | 复杂（含 permute） | — |
| **架构复杂度** | 中（基于 SM80 扩展 FP8） | 高（全新 WGMMA+TMA） | 低（基线） |

---

## 9. SM120 实现的核心设计决策

### 为何不使用 WGMMA？
SM120（Blackwell）的 WGMMA 语义与 SM90（Hopper）不同（SM120 引入了新的 `stmatrix`/`tcgen05` 指令集），目前 CuTe 对 SM120 WGMMA 的支持尚未成熟。采用 SM80 `mma.sync` 是最稳健的选择。

### FP8 Phase 1 → Phase 2 的演进
- **Phase 1**（已弃用）：GMEM FP8 → 转 BF16 → SMEM → BF16 MMA
  问题：节省了带宽（FP8 = BF16 一半大小），但 kNWarps=4（受 SMEM 限制），且 FP8→BF16 转换开销使 SM120 上 FP8 反而比 BF16 慢 34%
- **Phase 2**（当前）：FP8 直接驻 SMEM → SM89 FP8 MMA（K=32，是 BF16 K=16 的两倍）
  优势：kNWarps=8（FP8 SMEM 减半），计算密度翻倍，带宽和计算均有改善

### FP8 平坦 SMEM 布局的原因
使用 `SmemLayoutAtomQ = Layout<_16, _32>` 而非 Swizzle，原因：
- SM89 FP8 MMA 的 K=32 步长要求 16×32 tile 中 thread T 的数据在 byte 偏移 `T×16` 处
- Swizzle 的 XOR 操作使地址错位，导致 cos_sim ≈ 0.31 的精度崩溃
- 平坦布局保证地址正确性，代价是接受可能的 SMEM bank conflict（实测影响较小）

---

## 10. 性能基准参考

### H100 (SM90)，2026-03-06，bs=4/8，h=16，d=128，causal
- FP8 q=0 vs BF16：seq=1024 时 -52.8%，seq=4096 时 -12.6%
- FP8 注意力内核本身快 21%（418μs vs 507μs，seq=4096）
- 但 3× cudaStreamSynchronize（~129μs 每次）拉低端到端性能

### RTX 6000 Pro (SM120)，2026-03-08/09，bs=4/8，h=16，d=128，causal
- **Phase 1 FP8 比 BF16 慢**（与 H100 相反）
  - BF16 内核：882μs；FP8 内核：1180μs（+34%）
  - BF16 seq=4096 → 675 TFLOPS；FP8 → 193 TFLOPS（-71%）
  - 根因：kNWarps=4 + FP8→BF16 转换开销，seq=4096 计算瓶颈时带宽节省无效
- **Phase 2 FP8**（TODO）：SM89 原生 FP8 MMA + kNWarps=8，预期获得 2× 计算吞吐

---

## 13. TRT-LLM SM120 FP8 GEMM 实现解读

> 参考文件：
> - `/home/minyu/project/shopee/trtllm/.../sm120_utils.cuh`
> - `/home/minyu/project/shopee/trtllm/.../sm120_fp8_gemm_1d1d.cuh`

### 13.1 文件命名含义："1d1d"

`1d1d` = A 和 B 两个操作数都采用 **1D block scaling（一维块量化）**：

- **1D scaling**：scale factor 只沿 K 维度变化，每 128 个 K 元素共享一个 scale
  - A 的 scale shape：`(M, K/512, L)`（每 512 个 K 元素一个 e8m0）
  - B 的 scale shape：`(N, K/512, L)`
- **1d1d** = A 用 1D scale + B 用 1D scale（对称）

```cpp
static constexpr int kSFVecSize = 128;   // 固定 1×128 量化粒度
static constexpr int kTileK = 128;       // 每次 MMA 处理 128 个 K 元素
static constexpr int kNumTileKPerSF = 512 / kTileK;  // 4 个 K-tile 共享一个 SF tile
```

对比其他命名：
- `2d` = 2D block scaling（M×K 两个方向都分块）
- `per_tensor` = 整个 tensor 单一 scale

---

### 13.2 SmemLayoutAtomA vs SmemCopyAtomA：职责区分

这两个名字都含 "Smem" 但职责**完全不同**：

| 名字 | 类型 | 负责阶段 | 作用 |
|------|------|----------|------|
| `SmemLayoutAtomA` | Layout（布局描述符） | 两个阶段都用 | 描述 SMEM 物理排列（swizzle 模式、行/列步长） |
| `SmemCopyAtomA` | Copy_Atom（拷贝指令） | **SMEM → REG** | `ldmatrix` 指令，把 SMEM 数据搬入寄存器 |
| `TMA_A` / `GmemTiledCopyQKV` | TMA / Copy_Atom | **GMEM → SMEM** | 把全局内存数据搬入共享内存 |

完整数据流：
```
GMEM
  │  TMA (SM90_TMA_LOAD) / cp.async      ← 由 TMA_A 或 GmemTiledCopyQKV 驱动
  │  写入地址按 SmemLayoutA 计算          ← SmemLayoutAtomA 决定物理排列
  ▼
SMEM（SW128 swizzle 物理排布）
  │  ldmatrix.x4 (SmemCopyAtomA)         ← 由 SmemCopyAtomA = LDSM_N 驱动
  │  读出地址同样按 SmemLayoutA 计算      ← 写入和读出布局必须一致！
  ▼
寄存器（MMA fragment 格式）
  │  mma.sync (MMA_Atom)
  ▼
累加器寄存器
```

**关键约束**：`SmemCopyAtomA` 和 `SmemLayoutAtomA` 必须配套使用。LDSM_N 的地址计算依赖 K-fast SW128 的物理排列；若布局改变（如改为 flat）而 copy atom 不变，读出的元素顺序就会错乱。这正是 HSTU FP8 GEMM2 V-operand bug 的根因。

---

### 13.3 sm120_utils.cuh 核心配置详解（SM120BlockScaledBuilder）

#### MMA Atom — SM120 专属 blockscaled FP8

```cpp
using MMA_Atom = MMA_Atom<SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    float_e4m3_t, float_e4m3_t, float, float_ue8m0_t, 32>>;
```

- `SM120_16x8x32_TN_VS`：SM120A（B100/B200）专属指令，**不同于** HSTU 用的 `SM89_16x8x32_F32E4M3E4M3F32_TN`
- `_VS` = "Vector Scaled"：scale 因子直接参与 MMA 硬件指令（非软件 descale）
- `float_ue8m0_t`：scale 类型 = UE8M0（只有指数位无尾数，表示 2^e，8 种量级）
- `32`：scale vector size = 每 32 个 FP8 元素共享一个硬件 scale（MMA 粒度，不是量化粒度 128）

#### TiledMMA 线程排列

```cpp
using PermMmaTileM = Int<32>;
using PermMmaTileN = Layout<Shape<_8, _4, _4>, Stride<_1, _32, _8>>;
using TiledMma = TiledMMA<MMA_Atom,
    Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>,   // 2×4=8 warps
    Tile<PermMmaTileM, PermMmaTileN, PermMmaTileK>>;
static constexpr int kNumMathThreads = size(TiledMma::ThrLayoutVMNK{});  // = 256
static constexpr int kNumMathWarps   = kNumMathThreads / 32;             // = 8
```

- `Shape<_2,_4,_1>` = M方向2个warp × N方向4个warp × K方向1个
- `Stride<_4,_1,_0>` = warp排列步长（N方向最快，M方向步长4）
- `PermMmaTileN = Layout<Shape<_8,_4,_4>,Stride<_1,_32,_8>>`：N方向 8×4×4=128，非平凡的置换保证 warp 间 N-tile 无冲突

#### SMEM Copy Atoms — A 和 B 都用 LDSM_N

```cpp
using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
using SmemCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>;  // B 也用非转置！
```

前提：A 和 B 的 SMEM 布局完全对称（都是 K-fast SW128），所以不需要 LDSM_T。

#### SMEM Layout — K-major + SW128

```cpp
using SmemLayoutAtomA = GMMA::Layout_K_SW128_Atom<ElementA>;
// FP8: ((_8,_16),(_128,_1)):((_128,_1024),(_1,_0)) → 128行×128列
using SmemLayoutAtomB = GMMA::Layout_K_SW128_Atom<ElementB>;
// 与 A 完全相同结构

using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{},
    make_shape(shape<0>(TileShape{}),    // kTileM=32
               shape<2>(TileShape{}),   // kTileK=128
               Int<AB_Stages>{}),       // Stages=4
    Step<_1,_2,_3>{}));
// 最终尺寸: (32, 128, 4) × FP8 = 16 KB

using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{},
    make_shape(shape<1>(TileShape{}),   // kTileN=128
               shape<2>(TileShape{}),  // kTileK=128
               Int<AB_Stages>{}),
    Step<_1,_2,_3>{}));
// 最终尺寸: (128, 128, 4) × FP8 = 64 KB
```

#### Scale Factor 布局（1D block scaling 细节）

```cpp
static constexpr int kSFVecSize      = 128;  // 量化粒度：每 128 个 FP8 一个 scale
static constexpr int kTileSF         = 1;    // 每个 SF tile = 1 个 scale block
static constexpr int kNumTileKPerSF  = 512 / kTileK;  // = 4 个 K-tile 共享一个 SF
static constexpr int kNumStagePerSF  = kNumTileKPerSF / AB_Stages;  // = 1

using ElementSFLoad    = int32_t;          // 加载类型：4个 e8m0 打包为 int32
using ElementSFCompute = cute::float_ue8m0_t;  // 计算类型：解包后的 e8m0
```

Scale 物理布局（column-major）：
```cpp
// A scale: (M_aligned, K/512, L) column-major
int64_t scale_k = ceil_div(K, 128 * 4);   // 每 512 K 元素（4个128-block）一个 e8m0
// → A_scale shape = (M, K/512)，M 方向 stride=1（连续）

// B scale: (N_aligned, K/512, L) column-major
// → B_scale shape = (N, K/512)，N 方向 stride=1（连续）
```

Scale SMEM copy：
```cpp
using SmemCopyAtomSF = Copy_Atom<AutoVectorizingCopy, ElementSFLoad>;
// 自动向量化 LDS，加载 int32（4个e8m0打包）
```

#### TMA 配置（注意：SM120 复用 SM90 TMA 接口）

```cpp
using TMA_A = decltype(make_tma_copy(SM90_TMA_LOAD{},   // SM120 使用 SM90 TMA 接口
    make_tensor(recast_ptr<ElementA>(nullptr), StrideA{}),
    SmemLayoutA{}(_, _, Int<0>{}),
    make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})), _1{}));
```

> SM120 硬件支持 SM90 风格的 TMA 指令，`SM90_TMA_LOAD` 可直接复用。

---

### 13.4 sm120_fp8_gemm_1d1d.cuh 内核详解

#### 线程组织（生产者-消费者完全分离）

```cpp
static constexpr int kNumTMAThreads  = 128;   // 4 warps：TMA producer
static constexpr int kNumMathThreads = 256;   // 8 warps：MMA consumer
static constexpr int MaxThreadsPerBlock = 384; // 总线程数

// warp 分工：
// warp 0~7 (math warps):  MMA 计算 + epilogue 写 sD
// warp 8   (epi warp):    TMA store sD → GMEM
// warp 9   (ab warp):     TMA load A/B GMEM → SMEM
// warp 10  (sf warp):     TMA load Scale GMEM → SMEM
```

#### 三路独立流水线（mbarrier 同步）

```
SF warp ─→ load_sf() ─→ sf_full_mbar[0]
                              │
AB warp ─→ load_ab() ─→ ab_full_mbar[0..3]
                              │
Math warps ←── mma() ←────────┘
                │
                └→ 写 sD → store_full_mbar[0]
                              │
Epilogue warp ←── store() ←───┘ ─→ TMA store → GMEM
```

#### load_ab() — A/B 数据搬运（4-stage ping-pong）

```cpp
int32_t k_tile_count = sf_tile_count * KT::kNumTileKPerSF;  // 总 K-tile 数
for (int32_t k_tile_idx = 0; k_tile_idx < k_tile_count; k_tile_idx += KT::AB_Stages) {
    cute::for_each(cute::make_int_sequence<KT::AB_Stages>{},  // AB_Stages=4
        [&](auto write_stage) {
            ab_empty_mbar[write_stage].wait(phase);           // 等消费者释放
            auto tma_copy_a = params.tma_load_a.with(ab_full_barrier);
            cute::copy(tma_copy_a, tAgA(_, _, _, k_tile+write_stage),
                                   tAsA(_, _, _, write_stage)); // TMA 异步写 SMEM
            // 同理 B
            ab_full_mbar[write_stage].arrive_and_expect_tx(TmaABTransactionBytes);
        });
    phase ^= 1;
}
```

#### load_sf() — Scale 搬运（独立于 AB，1-stage）

```cpp
// 每个 SF tile 对应 kNumTileKPerSF=4 个 AB K-tile（512 K 元素 = 一个量化单元）
for (int32_t sf_tile_idx = 0; sf_tile_idx < sf_tile_count; ++sf_tile_idx) {
    sf_empty_mbar[0].wait(phase);
    cute::copy(tma_copy_sfa, tAgSFA(_, _, _, sf_tile_idx), tAsSFA(_, _, _, 0));
    cute::copy(tma_copy_sfb, tBgSFB(_, _, _, sf_tile_idx), tBsSFB(_, _, _, 0));
    sf_full_mbar[0].arrive_and_expect_tx(TmaSFTransactionBytes);
    phase ^= 1;
}
```

#### mma() — 核心计算函数

**SMEM → 寄存器（S2R）设置：**

```cpp
// A: make_tiled_copy_A + LDSM_N
auto s2r_copy_A = make_tiled_copy_A(typename KT::SmemCopyAtomA{}, mma);
auto tXsA = s2r_thr_copy_A.partition_S(sA);  // (CPY,CPY_M,CPY_K,PIPE)
auto tXrA = s2r_thr_copy_A.retile_D(tCrA);   // (CPY,CPY_M,CPY_K)

// B: 同样用 LDSM_N（与 A 完全对称）
auto s2r_copy_B = make_tiled_copy_B(typename KT::SmemCopyAtomB{}, mma);
auto tXsB = s2r_thr_copy_B.partition_S(sB);
auto tXrB = s2r_thr_copy_B.retile_D(tCrB);
```

**Scale 分片（thrfrg_SFA）：**

```cpp
// A scale 的线程分配布局
using AtomLayoutSFA_TV = Layout<
    Shape<Shape<_2, _2, _8>, _1>,
    Stride<Stride<_8, _0, _1>, _16>>;
// Shape<_2,_2,_8> = 32 thread slots，但 Stride<_8,_0,_1>：
//   - _8  stride：每组8线程
//   - _0  stride：广播（不同线程持有相同 scale！）
//   - _1  stride：线程内递增
// → 实际只有 16 个线程持有有效 scale，其余通过广播共享
```

```cpp
// B scale 的线程分配布局
using AtomLayoutSFB_TV = Layout<
    Shape<Shape<_4, _8>, _1>,
    Stride<Stride<_0, _1>, _8>>;
// _0 stride 同样表示广播
```

**Scale 格式转换 transform_fragment_for_qmma：**

```cpp
// int32 → float_ue8m0_t 重解释（4个e8m0打包在int32中）
auto new_layout = make_layout(
    make_shape(_32{}, num_mn, _4{}, _4{}),
    make_stride(_0{}, _4{}, _0{}, _1{}));  // stride _0 = 广播，_1 = 逐元素
auto new_tensor = make_tensor(recast_ptr<ElementSFCompute>(old_ptr), new_layout);
// 将 (Val, MN, K) 的 int32 fragment 重解释为
// (32, MN, 4, 4) 的 ue8m0 layout，匹配 SM120 blockscaled MMA 对 scale 格式的要求
```

**实际 MMA 调用（数据+scale 打包）：**

```cpp
for (int32_t sf_tile_idx = 0; ...) {
    // 加载 scale
    cute::copy(s2r_copy_SFA, tXsSFA(_, _, _, 0), tXrSFA);
    cute::copy(s2r_copy_SFB, tXsSFB(_, _, _, 0), tXrSFB);
    auto tCrSFA_frg = KT::transform_fragment_for_qmma(tCrSFA);  // 重解释为 ue8m0

    cute::for_each(cute::make_int_sequence<KT::kNumStagePerSF>{},  // =1
        [&](auto iter) {
            cute::for_each(cute::make_int_sequence<KT::AB_Stages>{},  // =4
                [&](auto read_stage) {
                    ab_full_mbar[read_stage].wait(ab_phase);
                    cute::copy(s2r_copy_A, tXsA(_, _, _, read_stage), tXrA);
                    cute::copy(s2r_copy_B, tXsB(_, _, _, read_stage), tXrB);
                    ab_empty_mbar[read_stage].arrive();  // 通知生产者可以复用

                    auto tCrSFA_stage = tCrSFA_frg(_, _, _, iter*AB_Stages+read_stage);
                    auto tCrSFB_stage = tCrSFB_frg(_, _, _, iter*AB_Stages+read_stage);

                    // 关键：make_zip_tensor 将数据和 scale 打包，硬件内部完成缩放
                    cute::gemm(mma,
                        make_zip_tensor(tCrA, tCrSFA_stage),
                        make_zip_tensor(tCrB, tCrSFB_stage),
                        accum);
                });
            ab_phase ^= 1;
        });
    sf_phase ^= 1;
}
```

---

### 13.5 TRT-LLM SM120 中 B 也用 LDSM_N 的原因

```cpp
// sm120_utils.cuh line 258-259
using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
using SmemCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>;  // B 也用 LDSM_N！
```

B 使用非转置 LDSM_N 而非 LDSM_T 的前提：**B 的 SMEM 布局与 A 完全对称——都是 K-fast（K 为 inner 维度）**：

```cpp
using SmemLayoutAtomA = GMMA::Layout_K_SW128_Atom<ElementA>;  // (N, K) K-fast
using SmemLayoutAtomB = GMMA::Layout_K_SW128_Atom<ElementB>;  // (N, K) K-fast（相同！）
```

A 存为 `(kTileM, kTileK)` K-fast → LDSM_N 读 K 方向 ✓
B 存为 `(kTileN, kTileK)` K-fast → LDSM_N 读 K 方向 ✓

对比 HSTU BF16 中 B（V^T）的处理：
- V 自然存为 `(kBlockN, kHeadDim)` → kHeadDim-fast（N-fast）
- 转置视图 sVt 使 kBlockN（K）变为 fast 维度
- 用 **LDSM_T**（转置 ldmatrix）读取，产生 N 方向遍历 → 正确

---

### 13.4 CuTe Layout 记号详解

以 `GMMA::Layout_K_SW128_Atom<float_e4m3_t>` 为例：

```
((_8,_16), (_128,_1)) : ((_128,_1024), (_1,_0))
 ─────────────────────   ─────────────────────
       Shape                    Stride
   dim0     dim1            dim0      dim1
  (行/MN)  (K列)          (行步长)  (K步长)
```

#### Q：这个 Atom 是多大？

```
dim0 总大小：_8 × _16 = 128 行
dim1 总大小：_128 × _1 = 128 列
→ Atom 总共覆盖 128行 × 128列 = 16384 FP8 = 16 KB
```

> ⚠️ 注意：**Atom 是 128×128，不是 8×128**。`_8` 是内层维度（swizzle group 大小），`_16` 是外层维度（group 数量），两者相乘才是总行数。

对比 BF16 版本（`Layout_K_SW128_Atom<bfloat16_t>`）：
- 形状 `((_8,_16), (_64,_1))` = 128行 × 64列（因 BF16 是 2 字节，SW128 = 64 个 BF16）
- FP8 是 1 字节，SW128 = 128 个 FP8，所以 K-dim 翻倍

#### 各数字含义

| 数字 | 维度 | 含义 |
|------|------|------|
| `_8` | dim0 inner | swizzle 周期 = 8 行（Swizzle<3,3,3> 的 XOR 周期 = 2³） |
| `_16` | dim0 outer | swizzle group 数量 = 128行 / 8行 = 16 组 |
| `_128` | dim1 inner | K 方向连续 128 个 FP8 = 128 字节 = **SW128** 命名来源 |
| `_1` | dim1 outer | K 方向只有 1 个 swizzle 周期（边界标记） |
| `_128`（stride）| dim0 inner stride | 相邻行间距 128 bytes（K-major：K 元素连续） |
| `_1024`（stride）| dim0 outer stride | 相邻 group 间距 = 8行 × 128bytes = 1 KB |
| `_1`（stride） | dim1 inner stride | K 元素连续，stride=1 ✓ |
| `_0`（stride） | dim1 outer stride | **XOR swizzle 编码**：stride=0 表示此维度参与 XOR 而非加法 |

#### 物理地址计算（逻辑坐标 → SMEM 偏移）

```
逻辑坐标：row = (a, b)，a ∈ [0,8)，b ∈ [0,16)
         col = (c, d)，c ∈ [0,128)，d = 0

无 swizzle 时（加法）：
  offset = a*128 + b*1024 + c*1 + d*0

有 SW128 swizzle（XOR）：
  swizzle_bits = (b & 7) << 5       # b 的低3位 XOR 进 col 的 bit[5:8]
  phys_col = c XOR swizzle_bits
  offset = a*128 + b*1024 + phys_col
```

#### SW128 Swizzle 消除 bank conflict 示例

```
无 swizzle（存在 bank conflict）：
  行0, col=0 → 物理地址 0   → bank 0
  行1, col=0 → 物理地址 128 → bank 0  ← CONFLICT!

SW128 swizzle 后（各行访问不同 bank）：
  行0, col=0 → 物理地址 0            → bank  0  ✓
  行1, col=0 → 物理地址 0 XOR 128=128 → bank  8  ✓
  行2, col=0 → 物理地址 0 XOR 256=256 → bank 16  ✓
  行7, col=0 → 物理地址 0 XOR 896=896 → bank 28  ✓
```

8 行访问 8 个不同 bank → ldmatrix.x4 (LDSM) 完全无冲突 ✓

---

### 13.5 SmemLayoutAtomA 与 SmemLayoutA 的关系

**一句话：Atom 是瓷砖图案，Layout 是整面墙。**

```cpp
// Atom：最小可重复单元（FP8: 128行×128列）
using SmemLayoutAtomA = GMMA::Layout_K_SW128_Atom<ElementA>;

// Full Layout：用 tile_to_shape 把 Atom 平铺到整个 SMEM tile
using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{},
    make_shape(shape<0>(TileShape{}),   // kTileM = 32
               shape<2>(TileShape{}),   // kTileK = 128
               Int<AB_Stages>{}),       // Stages = 4
    Step<_1, _2, _3>{}));
```

```
SmemLayoutAtomA 覆盖：128行 × 128列（FP8）
      ↓ tile_to_shape(target=(32, 128, 4))

目标尺寸：kTileM=32，kTileK=128，4 个流水 stage

M 方向：32 / 128 = 0.25 < 1 → 这里 tile_to_shape 实际上只取 Atom 的子集？
```

> 注意：当 Atom 比 target 大时，`tile_to_shape` 的行为需要仔细核查。对于 TRT-LLM 中 kTileM=32、Atom 行数=128 的情况，这里可能是 tile_to_shape 截取 Atom 的一部分，或者 kTileM 应理解为 32 而 Atom 的 128 行是包含多个 warp 覆盖范围的完整 tile。**实际上 TRT-LLM 的 kTileM=32 对应 kNumMathWarps×16=2×16=32，而 Atom 覆盖 128 行可能是为了支持不同 TileM 配置。**

`Step<_1, _2, _3>` 控制平铺顺序（M 优先，K 其次，Stage 最外）。

**关键约束**：Atom 尺寸必须整除（或包含）Full Shape：
```
M_full % M_atom == 0  或  M_full <= M_atom
K_full % K_atom == 0  →  128 % 128 == 0  ✓
```

**Atom 的 Swizzle 被自动继承**：整个 SmemLayoutA 的任意 `(row, col)` 访问都自动走 SW128 swizzle 地址计算，不需要额外处理。

---

### 13.6 TRT-LLM SM120 整体架构（sm120_fp8_gemm_1d1d.cuh）

**线程组织（生产者-消费者分离）：**
```
总线程数 = kNumMathThreads(256) + kNumTMAThreads(128) = 384
  ├── Math warps (warp 0~7):  负责 MMA 计算
  ├── Epilogue warp (warp 8): 负责 store D → GMEM
  ├── AB warp (warp 9):       负责 TMA load A/B → SMEM
  └── SF warp (warp 10):      负责 TMA load Scale A/B → SMEM
```

**流水线：**
- `AB_Stages = 4`：A/B 各有 4 个 SMEM ping-pong buffer
- `SF_Stages = 1`：Scale 单 buffer（scale 更新频率低）
- 通过 `mbarrier`（ClusterTransactionBarrier）同步生产者-消费者

**Blockscaled MMA（区别于 HSTU 的软件 descale）：**
```cpp
// scale 打包进 zip_tensor，直接传给 MMA 指令
cute::gemm(mma,
    make_zip_tensor(tCrA, tCrSFA_stage),   // A 数据 + A scale
    make_zip_tensor(tCrB, tCrSFB_stage),   // B 数据 + B scale
    accum);
// 硬件 MMA 内部完成缩放，无需软件 descale 循环
```

---

### 13.7 与 HSTU FP8 实现的关键差异

| 特性 | TRT-LLM SM120 | HSTU SM120 FP8 |
|------|--------------|----------------|
| **MMA 指令** | SM120::BLOCKSCALED（SM120A 专用，B100/B200）| SM89_16x8x32（SM89+ 通用，含 SM120 consumer）|
| **Scale 集成** | 硬件 blockscaled（MMA 内部缩放）| 软件 descale（GEMM 后乘标量）|
| **B 的 Copy Atom** | LDSM_N（与 A 对称）| 需与 SMEM 写入布局一致（当前存在布局不一致 bug）|
| **B 的 SMEM 布局** | K-fast SW128（写入=读取，一致）| flat 写入，需确保读取也用 flat 布局 |
| **GMEM→SMEM** | TMA（硬件异步，生产者 warp 专职）| cp.async（软件）|
| **线程模型** | 生产者-消费者分离（mbarrier 同步）| 所有 warp 同时做计算+load |
| **适用硬件** | SM120A（B100/B200 数据中心）| SM120 consumer（RTX 6000 Pro）|

**最重要的启发（对 HSTU FP8 bug 修复）**：
TRT-LLM B 也能用 LDSM_N 的原因是**写入和读取使用同一种 K-fast SW128 布局**。HSTU FP8 的核心问题是 V 的写入（flat SmemLayoutKV）与读取（SW128-based SmemLayoutVtransposed）布局不一致，导致 tOsVt 读到错误的物理地址数据。

---

## 附：各架构关键文件路径

```
# Blackwell (SM120)
fbgemm_gpu/experimental/hstu/src/hstu_blackwell/

# Hopper (SM90)
fbgemm_gpu/experimental/hstu/src/hstu_hopper/

# Ampere (SM80)
fbgemm_gpu/experimental/hstu/src/hstu_ampere/

# Python 入口（SM 版本分发）
fbgemm_gpu/experimental/hstu/hstu/cuda_hstu_attention.py

# 内核生成脚本
fbgemm_gpu/experimental/hstu/src/generate_kernels.py
```

---

## 11. SM120 FP8 实现代码导读

### 11.1 调用链（自顶向下）

```
hstu_ops_gpu.cpp: hstu_varlen_fwd_120()          ← PyTorch 入口
  └→ run_hstu_fwd_blackwell()                     ← dtype/mask/rab/causal 多级 static dispatch
       └→ run_hstu_fwd_headdim_sm120<FP8Type,...>() ← dispatch head_dim (64/128)
            └→ run_hstu_fwd_sm120<120, e4m3, hdim,...>() ← 选 tile size，区分 BF16/FP8
                 └→ run_hstu_fwd_sm120_impl<..., Is_fp8=true>() ← launch kernel
                      └→ hstu_fwd_kernel_sm120<Kernel_traits>()  ← GPU kernel 入口
                           └→ hstu_compute_attn_1rowblock_sm120() ← 核心计算
```

BF16 和 FP8 共用同一个 kernel 函数 `hstu_compute_attn_1rowblock_sm120()`，通过 `if constexpr (!Is_fp8)` 在编译期分支。

### 11.2 数据类型体系

```
输入: Q(FP8 e4m3), K(FP8), V(FP8), RAB(BF16, optional), descale_q/k/v(float)
中间: acc_s(float), acc_o(float)
输出: O(BF16)
```

FP8 traits 定义（kernel_traits.h）：
- `Element = float_e4m3_t` — Q/K/V 在 GMEM 和 SMEM 中的类型
- `ElementSmem = float_e4m3_t` — Phase 2 核心：SMEM 中保持 FP8，不转 BF16
- `OutputType = bfloat16_t` — 输出固定 BF16
- `MMA_Atom_Arch = SM89_16x8x32_F32E4M3E4M3F32_TN` — K=32 FP8 MMA

### 11.3 MMA 指令与 TiledMma 配置

```cpp
using TiledMma = TiledMMA<
    MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>,  // 单个 MMA 原子
    Layout<Shape<Int<kNWarps>, _1, _1>>,           // kNWarps 个 warp 在 M 方向堆叠
    Tile<Int<16 * kNWarps>, _16, _32>>;            // 总覆盖 M×N×K
```

- AtomLayout: `(kNWarps, 1, 1)` — warp 仅在 M 维展开
- TileShape: `(16*kNWarps, 16, 32)` — M=128(8 warps), N=16(每步 2 个 N=8 原子), K=32

### 11.4 SMEM 布局：Flat vs Swizzle

**BF16**：使用 `Swizzle<3,3,3>` 消除 `ldmatrix.x4` bank conflict
**FP8**：使用 flat layout（无 swizzle）

```cpp
// FP8 SMEM atom — 无 swizzle
using SmemLayoutAtomQ = Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>;
```

原因：FP8 是 1-byte 元素，swizzle 的 XOR 粒度（16-byte group）与 `ldmatrix` 的 16-byte 加载粒度不匹配，导致 CuTe 静态向量化检查失败。

### 11.5 SMEM → Register 拷贝：DefaultCopy

```cpp
// FP8 path — 逐元素拷贝，保证寄存器排布正确
using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, Element>;
```

不能使用 `SM75_U32x4_LDSM_N`（ldmatrix.x4），因为 ldmatrix 以 b16 模式工作，每 16-bit slot 打包 2 个 FP8，产生的寄存器排布与 `SM89_16x8x32` MMA 期望不匹配。DefaultCopy 虽慢但保证正确。

### 11.6 FP8 Kernel 主循环（fwd_step_fp8）

每个 n_block 的计算步骤：

```
Step 1: cp.async V → SMEM (FP8)
Step 2: GEMM1 Q×K^T
        - (Has_rab) 加载 RAB(BF16) → 寄存器 → 转 float → 初始化 acc_s
        - flash::gemm: acc_s += Q_fp8 × K_fp8 (SM89 MMA, FP32 累加)
Step 3: Mask → descale (acc_s *= descale_q * descale_k) → alpha → fast_silu
Step 4: GEMM2 P×V
        - float → FP8: convert_type_safe(acc_s, rP)
        - layout 变换: convert_layout_acc_Aregs (K=32 → _4 divide)
        - flash::gemm_rs: acc_o += P_fp8 × V_fp8
Step 5: Output scale (acc_o *= descale_v / scaling_seqlen)
Step 6: Epilogue: FP32 → BF16 → SMEM(复用 smem_) → GMEM
```

### 11.7 Tile Size 选择

```cpp
// utils.h: get_tile_size_fwd_sm120<Headdim, Has_rab, Is_fp8>()
// FP8 无 RAB → kNWarps=8 (SMEM 减半, warp 翻倍)
// FP8 有 RAB → kNWarps=4 (RAB 是 BF16, 占额外 SMEM)
```

| hdim | Has_rab | kBlockM | kBlockN | kNWarps |
|------|---------|---------|---------|---------|
| 128 | false | 128 | 64 | 8 |
| 128 | true | 64 | 64 | 4 |
| 64 | false | 128 | 64 | 8 |
| 64 | true | 128 | 64 | 4 |

### 11.8 SMEM 空间规划（FP8, hdim=128, Share_Q_K_smem=true）

```
smem_[0 .. kSmemSizeQKV):
  ├── sK: 64×128×1B = 8 KB     (Share_Q_K_smem: sK.data() == sQ.data())
  ├── sV: 64×128×1B = 8 KB     (紧跟 sK 之后)
  └── sO: 128×128×2B = 32 KB   (复用 smem_ 起始, epilogue 时写 BF16)
  kSmemSizeQKV = max(sQ+sKV, sO) = max(8+16, 32) = 32 KB
```

### 11.9 Host 侧入口要点（hstu_ops_gpu.cpp）

- `quant_mode >= 0` 时要求输入 `float8_e4m3fn`，目前仅支持 `quant_mode=0`（per-tensor FP8）
- `descale_q/k/v` 均为必选参数
- 输出固定 BF16：`out = torch::empty({q.size(0), num_heads, head_size}, q.options().dtype(at::kBFloat16))`
- 通过 `TORCH_LIBRARY_FRAGMENT` 注册为 `fbgemm::hstu_varlen_fwd_120`

---

## 12. FP8 GEMM2 Accuracy 问题分析

### 12.1 问题现象

SM120 FP8 实现的 cos_sim ≈ 0.36（与 BF16 reference 对比），表明存在结构性错误。

### 12.2 `convert_layout_acc_Aregs` 的作用

GEMM2（P×V）中，P 来自 GEMM1 的输出（经 silu 后）。为避免 SMEM roundtrip，Flash Attention 直接在寄存器中将 C fragment（GEMM1 输出）重排为 A fragment（GEMM2 输入）。该函数负责 layout 变换：

```cpp
// utils.h
template <typename MMA_traits, typename Layout>
auto convert_layout_acc_Aregs(Layout acc_layout) {
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    if constexpr (mma_shape_K == 16) {
        // BF16: 2 N-tiles → 8 BF16 = 4 uint32_t × 2 BF16
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
        return make_layout(make_layout(get<0>(l), get<2,0>(l)), get<1>(l), get<2,1>(l));
    } else {  // K=32
        // FP8: 4 N-tiles → 16 FP8 = 4 uint32_t × 4 FP8
        auto l = logical_divide(acc_layout, Shape<X, X, _4>{});
        return make_layout(make_layout(get<0>(l), get<2,0>(l)), get<1>(l), get<2,1>(l));
    }
}
```

### 12.3 ALayout 结构对比

CuTe 的 MMA traits 定义了每个操作数的 Thread-Value 布局（mma_traits_sm80.hpp / mma_traits_sm89.hpp）：

```
CLayout（C/D fragment, 两者共用）:
  Shape:  ((4,8), (2,2))
  Stride: ((32,1), (16,8))
  → 每线程 4 个 float 值

SM80 ALayout (BF16, K=16):
  Shape:  ((4,8), (2, 2, 2))
  Stride: ((32,1), (16, 8, 128))
  → 每线程 8 个 BF16 = 4×uint32_t×2

SM89 ALayout (FP8, K=32):
  Shape:  ((4,8), (4, 2, 2))
  Stride: ((64,1), (16, 8, 256))
  → 每线程 16 个 FP8 = 4×uint32_t×4
```

核心差异：A value 模式第一维从 `_2`（BF16）变为 `_4`（FP8），stride 保持 16。

### 12.4 ALayout 的 uint32_t 内部排列

以 thread 0 为例，分析 SM89 ALayout 每个 uint32_t 寄存器内 4 个 FP8 的矩阵位置（A 矩阵 16×32, row-major）：

```
a[0] (v_linear 0-3): {A[0,0], A[0,16], A[1,0], A[1,16]}
    → 2 行(0,1) × 2 K-列(0,16) 交替排列

a[1] (v_linear 4-7): {A[0,8], A[0,24], A[1,8], A[1,24]}
    → 2 行(0,1) × 2 K-列(8,24) 交替排列

a[2] (v_linear 8-11): {A[8,0], A[8,16], A[9,0], A[9,16]}
a[3] (v_linear 12-15): {A[8,8], A[8,24], A[9,8], A[9,24]}
```

**每个 uint32_t 内部是 row-K 交替排列**：`{row0_K0, row0_K16, row1_K0, row1_K16}`

### 12.5 `convert_layout_acc_Aregs` 实际产生的排列

`_4` fold 将 4 个连续 N-tile 的 C atom 值堆叠进 mode-0，产生：

```
a[0] = {c[0]_N0, c[1]_N0, c[2]_N0, c[3]_N0}
```

根据 CLayout，thread 0 的 4 个 C atom 值来自 S 矩阵的**同一列、不同行**：
```
c[0]_N0 = P[0, col_N0]     ← row 0, K=col_N0
c[1]_N0 = P[2, col_N0]     ← row 2, K=col_N0  (*)
c[2]_N0 = P[1, col_N0]     ← row 1, K=col_N0
c[3]_N0 = P[3, col_N0]     ← row 3, K=col_N0  (*)
```

### 12.6 Mismatch

| a[0] byte | ALayout 期望 | `_4` fold 实际给出 | 匹配？ |
|-----------|-------------|-------------------|--------|
| 0 | P[0, 0] | P[0, 0] | ✓ |
| 1 | **P[0, 16]** | **P[2, 0]** | ✗ 错误的行和列 |
| 2 | P[1, 0] | P[1, 0] | ✓ |
| 3 | **P[1, 16]** | **P[3, 0]** | ✗ 错误的行和列 |

**ALayout 期望 row-K 交替（2 行 × 2 K-列），但 `_4` fold 给出同 K-列 4 行。**

a[0], a[1] 中约 50% 字节错误；a[2], a[3] 因行组不同全部错误。总体约 25% 正确值 → 产生非零但远低于 1.0 的 cos_sim，与观测到的 0.36 一致。

### 12.7 为何 BF16 K=16 不受影响

SM80 BF16 ALayout 每 uint32_t 仅含 2 BF16（`{row0_K, row1_K}`，同一 K 列、不同行）。`_2` fold 将 2 个 N-tile 折叠进 mode-0，每 uint32_t 仍然保持"同 K 列、不同行"的结构，与 ALayout 一致。

**本质区别**：
- BF16: uint32_t 内部 2 个值来自**同一 K 列** → `_2` fold 兼容
- FP8: uint32_t 内部 4 个值来自 **2 个不同 K 列 × 2 行交替** → `_4` fold 不兼容

### 12.8 修复方案

**方案 A：SMEM Roundtrip（安全，有性能开销）**

将 P 值写入 SMEM（C layout），再用 `make_tiled_copy_A` + `DefaultCopy` 按 ALayout 重新加载到寄存器。CuTe 自动处理 C→A 的 layout 转换。

```cpp
// 伪代码
convert_type_safe(acc_s, rP_fp8);           // float → FP8
copy(smem_tiled_copy_P_write, rP_fp8, sP); // regs → SMEM (C layout)
__syncthreads();
copy(smem_tiled_copy_P_read, sP, tOrP);    // SMEM → regs (A layout)
```

**方案 B：Warp Shuffle（高性能，复杂）**

根据 CLayout→ALayout 的精确映射，用 `__shfl_sync` 在线程间交换值，在寄存器中完成重排。

**方案 C：查证 CuTe 是否提供内置 C→A 变换**

CuTe/CUTLASS 可能在新版本中提供了针对 FP8 K=32 的 `permute_regs_C_to_A()` 或类似工具（Hopper 路径已有此类函数）。

### 12.9 其他已知 accuracy 问题

**RAB + descale 顺序 bug**（Has_rab=true + FP8 时）：
原代码：`acc_s = rab → gemm(acc_s += Q×K) → acc_s *= descale_qk` → descale 错误地缩放了 rab。
修复：先 GEMM + descale，再加 rab。（详见第 3 节 FP8 计算流程）

---

## 14. TRT-LLM SM120 FP8 GEMM 完整详解

> 源文件路径：
> - `/home/minyu/project/shopee/trtllm/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/sm120_blockwise_gemm/sm120_utils.cuh`
> - `/home/minyu/project/shopee/trtllm/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/sm120_blockwise_gemm/sm120_fp8_gemm_1d1d.cuh`

### 14.1 文件概览

这两个文件实现了 **SM120（Blackwell RTX Pro）上的 FP8 Block-Scaled GEMM**，使用 SM120A 专属的 **硬件 Blockscaled MMA 指令**（`SM120::BLOCKSCALED::SM120_16x8x32_TN_VS`）。

- **`sm120_utils.cuh`**：模板结构体 `SM120BlockScaledBuilder`，定义所有类型别名和布局（MMA atom、TiledMMA、SmemCopyAtom、SmemLayout、TMA descriptor、Scale Factor 布局）
- **`sm120_fp8_gemm_1d1d.cuh`**：kernel 类 `SM120BlockScaledKernel`，实现完整的 load_ab、load_sf、mma（含 epilogue）、store 函数，采用**生产者-消费者（producer-consumer）**线程模型

**命名中的"1d1d"含义**：
- 1D Block Scaling（Block-Scaled Quantization）：量化粒度为 1 维（沿 K 方向，每 128 个元素共用一个 scale 值）
- 两个"1d"分别指 A 矩阵和 B 矩阵都使用 1D block scaling
- 对应的量化格式：`float_e4m3_t`（FP8 E4M3）数据 + `float_ue8m0_t`（UE8M0）scale

---

### 14.2 sm120_utils.cuh 详解

#### 14.2.1 MMA Atom：`SM120_16x8x32_TN_VS`

```cpp
using MMA_Atom = MMA_Atom<SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    float_e4m3_t, float_e4m3_t, float, float_ue8m0_t, 32>>;
```

- **指令族**：`SM120::BLOCKSCALED`（SM120A 数据中心 GPU 专属，B100/B200）
- **指令名**：`SM120_16x8x32_TN_VS`
  - `16×8×32`：M=16, N=8, K=32（单次 MMA 覆盖）
  - `TN`：A 矩阵 T（row-major = K-fast）+ B 矩阵 N（col-major = K-fast）
  - `VS`：Vector Scale（硬件内置 scale factor 处理）
- **模板参数**：`ElementA=e4m3, ElementB=e4m3, ElementC=float, ElementSF=ue8m0, ScaleK=32`（每 32 个元素的 scale 粒度，注意：外层 kSFVecSize=128 是量化粒度，这里 32 是 MMA 内部 scale 粒度）
- **与 HSTU FP8 的对比**：HSTU 使用 `SM89_16x8x32_F32E4M3E4M3F32_TN`，这是 SM89+ 通用指令（无 VS），不带硬件 scale，需要软件 descale

#### 14.2.2 TiledMMA 配置

```cpp
using PermMmaTileM = Int<32>;
using PermMmaTileN = Layout<Shape<_8, _4, _4>, Stride<_1, _32, _8>>;
using TiledMma = TiledMMA<
    MMA_Atom,
    Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>,  // AtomLayout (Thr)
    Tile<PermMmaTileM, PermMmaTileN, PermMmaTileK>>; // ValLayout (Perm)
```

- AtomLayout `(2,4,1):(4,1,0)` → 总 8 个 atom，8 warp × 32 线程 = **256 math threads**
- TileM=32（2×16），TileN=128（4×8=32 atom N × permutation），TileK=128

#### 14.2.3 SmemCopyAtom：两者均为 LDSM_N

```cpp
using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
using SmemCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>;
```

- **A 和 B 均使用 `LDSM_N`**（非转置 ldmatrix）
- 原因：A 和 B 的 SMEM 布局都是 **K-fast**（K 方向连续），LDSM_N 读取 K 方向连续的 16 个字节，与 K-fast 布局天然匹配
- 不需要 LDSM_T（转置），因为写入和读取布局一致，均 K-fast

**与 HSTU BF16 的对比**：
- HSTU BF16 的 V 是 **N-fast**（N 方向连续写入），读取时需要 LDSM_T 转置
- TRT-LLM B（=V）是 **K-fast** 写入 + K-fast 读取，故 LDSM_N 即可

#### 14.2.4 SmemLayoutAtom：K-major SW128

```cpp
using SmemLayoutAtomA = GMMA::Layout_K_SW128_Atom<ElementA>;
// 展开 = ((_8,_16),(_128,_1)):((_128,_1024),(_1,_0))
using SmemLayoutAtomB = GMMA::Layout_K_SW128_Atom<ElementB>;
// 展开 = ((_8,_16),(_128,_1)):((_128,_1024),(_1,_0))
```

**重要：这个 Atom 是 128×128，不是 8×128**：
- dim0（行/MN方向）：`_8 × _16 = 128` 行
- dim1（K方向）：`_128 × _1 = 128` 列
- Atom 总共覆盖 128行 × 128列 = 16384 FP8 = 16 KB

**SW128 Swizzle 编码方式**：
- stride `_1024` = 相邻 group 间距（8行 × 128字节），是普通加法 stride
- stride `_0`（dim1 outer）= **XOR swizzle 编码**，`_0` 不是真的 stride=0，而是 CuTe 用 0 表示"此维度参与 XOR"
- 地址计算：`offset = a*128 + b*1024 + (c XOR (b_low3 << 5))`，将行号的低 3 位异或进 K 偏移的高位

**为什么 A 和 B 都用 K-fast（非 N-fast）**：
- A：M×K 矩阵，K-fast = row-major，GMEM 通常也是 row-major → 写入时自然对齐
- B：N×K 矩阵，K-fast = col-major（以 N 为行看）= 每个 K 元素在内存中按 N 方向存储
- 对于 GEMM B 操作数（TransposeB=N），TN 格式要求 B 按 K-fast 加载，所以 SMEM 也用 K-fast

#### 14.2.5 SmemLayoutA（Full Layout）

```cpp
using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{},
    make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<AB_Stages>{}),
    Step<_1, _2, _3>{}));
// kTileM=32, kTileK=128, Stages=4
```

`SmemLayoutAtomA` 覆盖 128行 × 128列，`tile_to_shape` 目标是 (32, 128, 4)。
- M 方向：32 < 128（Atom 比 target 大），取 Atom 的子集
- K 方向：128 = 128 ✓，恰好一个 Atom
- Stage 方向：4 stages（pipeline buffer）

**Atom 的 Swizzle 被自动继承**：整个 SmemLayoutA 内任意 `(row, col)` 访问都自动应用 SW128 swizzle。

#### 14.2.6 Scale Factor 1D 布局

```cpp
// GMEM 布局（列优先，按 K 方向切分）
static auto deduce_sfa_layout(ProblemShape) {
    int64_t scale_m = get_tma_aligned_size(M);
    int64_t scale_k = ceil_div(K, 128 * 4);  // K/512
    return make_layout(make_shape(scale_m, scale_k, L),
                       make_stride(Int<1>{}, scale_m, scale_m * scale_k));
}
```

- **量化粒度**：每 128 个 FP8 元素共用一个 `float_ue8m0_t` scale
- **`kSFVecSize = 128`**：1D block scaling（只沿 K 方向，不沿 M/N）
- `scale_k = K/512`：每 4 个 kTileK（4×128=512）对应一个 SF tile
- SMEM 使用 `ElementSFLoad = int32_t` 加载（4字节对齐，内含 4 个 ue8m0 按字节打包）
- 计算时 `ElementSFCompute = float_ue8m0_t`，通过 `transform_fragment_for_qmma` 转换

#### 14.2.7 TMA 配置

```cpp
using TMA_A = decltype(make_tma_copy(SM90_TMA_LOAD{}, ...));
using TMA_B = decltype(make_tma_copy(SM90_TMA_LOAD{}, ...));
```

- 使用 **SM90_TMA_LOAD**（Hopper TMA 接口，SM120 上也可用）
- TMA 直接将 GMEM → SMEM，无需 cp.async，硬件异步搬运
- 每次加载整个 tile（kTileM × kTileK 或 kTileN × kTileK）

#### 14.2.8 `thrfrg_SFA`：Scale Factor 线程分区

```cpp
using AtomLayoutSFA_TV = Layout<
    Shape<Shape<_2, _2, _8>, _1>,
    Stride<Stride<_8, _0, _1>, _16>>;
```

- 形状 `((_2,_2,_8), _1)`：共 `2×2×8=32` 个线程 ID，每线程 1 个 scale 值
- stride `_0`：**广播维度**，`_0` 表示此维度的不同线程 ID 映射到同一物理地址（多个线程读同一个 scale）
- 原理：对于 1D block scaling，同一个 MNK tile 内的 16 个 FP8 元素（一行 × 16 K 元素）共用同一个 scale → 需要广播给多个线程
- 具体广播规则：`Stride<_8, _0, _1>` 中 `_0` 对应 `_2`（2个线程广播同一 scale）

#### 14.2.9 `transform_fragment_for_qmma`：int32 → ue8m0 layout 重解释

```cpp
auto transform_fragment_for_qmma(Tensor&& tensor) {
    auto new_ptr = recast_ptr<ElementSFCompute>(old_ptr);  // int32 → ue8m0
    auto new_layout = make_layout(
        make_shape(_32{}, num_mn, _4{}, _4{}),
        make_stride(_0{}, _4{}, _0{}, _1{}));
    return make_tensor(new_ptr, new_layout);
}
```

- 将以 `int32_t` 加载的 scale fragment 重新解释为 `float_ue8m0_t` 布局
- `_32{} stride _0{}`：32 个线程广播同一个 int32（每 int32 内含 4 个 ue8m0）
- `_4{} stride _1{}`：int32 内 4 个字节分别对应 4 个连续 ue8m0
- 这个 new_layout 直接对应硬件 MMA 期望的 scale operand 布局

---

### 14.3 sm120_fp8_gemm_1d1d.cuh 详解

#### 14.3.1 线程组织（生产者-消费者分离）

```
总线程数 = kNumMathThreads(256) + kNumTMAThreads(128) = 384
  kNumMathWarps = 256 / 32 = 8
  kNumTMAWarps  = 128 / 32 = 4（但实际只用 3 个 warp）

warp 0~7  (thread 0~255):   Math warps → 执行 mma()
warp 8    (thread 256~287): Epilogue warp → 执行 store()
warp 9    (thread 288~319): AB warp → 执行 load_ab()（仅 lane 0 active）
warp 10   (thread 320~351): SF warp → 执行 load_sf()（仅 lane 0 active）
```

关键代码：
```cpp
if (warp_idx >= KT::kNumMathWarps) {
    constexpr int epi_warp_idx = KT::kNumMathWarps;     // warp 8
    constexpr int ab_warp_idx  = epi_warp_idx + 1;      // warp 9
    constexpr int sf_warp_idx  = ab_warp_idx + 1;       // warp 10
    if (warp_idx == ab_warp_idx && lane_predicate) load_ab(...);
    if (warp_idx == sf_warp_idx && lane_predicate) load_sf(...);
    if (warp_idx == epi_warp_idx && lane_predicate) store(...);
} else {
    mma(...);  // warp 0~7 全部做 MMA
}
```

#### 14.3.2 三个流水线

| 流水线 | Buffer 数 | 同步机制 | 用途 |
|--------|-----------|----------|------|
| AB pipeline | `AB_Stages=4` | `ClusterTransactionBarrier` (ab_full/ab_empty) | A/B SMEM ping-pong |
| SF pipeline | `SF_Stages=1` | `ClusterTransactionBarrier` (sf_full/sf_empty) | Scale Factor 单 buffer |
| Store pipeline | 1 | `ClusterBarrier` (store_full/store_empty) | MMA→Epilogue 同步 |

SF 流水线只有 1 个 buffer，因为 1 个 SF tile 覆盖 `kNumTileKPerSF=4` 个 AB tile（512/128=4），SF 加载频率比 AB 低 4×。

#### 14.3.3 `load_ab()` 函数

```cpp
// 每次循环加载 AB_Stages=4 个 K-tile（批量填充 pipeline buffer）
int32_t k_tile_count = sf_tile_count * KT::kNumTileKPerSF;
for (int32_t k_tile_idx = 0; k_tile_idx < k_tile_count; k_tile_idx += KT::AB_Stages) {
    cute::for_each(cute::make_int_sequence<KT::AB_Stages>{}, [&](auto write_stage) {
        ab_empty_mbar[write_stage].wait(phase);  // 等待该 stage 空闲
        auto tma_copy_a = params.tma_load_a.with(ab_full_mbar[write_stage]);
        cute::copy(tma_copy_a, tAgA(_, _, _, k_tile_idx + write_stage), tAsA(_, _, _, write_stage));
        auto tma_copy_b = params.tma_load_b.with(ab_full_mbar[write_stage]);
        cute::copy(tma_copy_b, tBgB(_, _, _, k_tile_idx + write_stage), tBsB(_, _, _, write_stage));
        ab_full_mbar[write_stage].arrive_and_expect_tx(KT::TmaABTransactionBytes);
    });
    phase ^= 1;
}
```

- TMA 异步加载：`cute::copy` 立即返回，数据异步传输
- `arrive_and_expect_tx`：告知 barrier 预期传输字节数，TMA 完成后自动 arrive
- math warp 侧通过 `ab_full_mbar[stage].wait()` 等待数据就绪

#### 14.3.4 `load_sf()` 函数

```cpp
for (int32_t sf_tile_idx = 0; sf_tile_idx < sf_tile_count; ++sf_tile_idx) {
    sf_empty_mbar[0].wait(phase);  // 等待 math warp 消费完
    cute::copy(tma_copy_sfa, tAgSFA(_, _, _, sf_tile_idx), tAsSFA(_, _, _, Int<0>{}));
    cute::copy(tma_copy_sfb, tBgSFB(_, _, _, sf_tile_idx), tBsSFB(_, _, _, Int<0>{}));
    sf_full_mbar[0].arrive_and_expect_tx(KT::TmaSFTransactionBytes);
    phase ^= 1;
}
```

- SF 独立流水线，不与 AB 耦合
- 单 buffer，每次只加载 1 个 SF tile（覆盖 4 个 AB k-tile）

#### 14.3.5 `mma()` 函数核心

**SMEM → Register 拷贝设置**：
```cpp
// A: make_tiled_copy_A + LDSM_N
auto s2r_copy_A = make_tiled_copy_A(typename KT::SmemCopyAtomA{}, mma);
auto tXsA = s2r_thr_copy_A.partition_S(sA);  // (CPY, CPY_M, CPY_K, PIPE)
auto tXrA = s2r_thr_copy_A.retile_D(tCrA);   // (CPY, CPY_M, CPY_K)

// B: make_tiled_copy_B + LDSM_N（与 A 对称）
auto s2r_copy_B = make_tiled_copy_B(typename KT::SmemCopyAtomB{}, mma);
auto tXsB = s2r_thr_copy_B.partition_S(sB);
auto tXrB = s2r_thr_copy_B.retile_D(tCrB);
```

**Scale Factor 拷贝设置**（自定义，非标准 make_tiled_copy）：
```cpp
// 使用 get_layoutSFA_TV 获取 SF 的 Thread-Value 布局
auto s2r_copy_SFA = make_tiled_copy_impl(
    typename KT::SmemCopyAtomSF{},  // AutoVectorizingCopy
    KT::get_layoutSFA_TV(mma),
    make_shape(size<0>(tile_shape(mma)), _1{}));
auto tXsSFA = s2r_thr_copy_SFA.partition_S(sSFA);
auto tCrSFA = KT::partition_fragment_SFA(sSFA(_, _, Int<0>{}), thr_mma);
auto tCrSFA_frg = KT::transform_fragment_for_qmma(tCrSFA);  // int32→ue8m0 重解释
```

**MMA 主循环（双层 for_each）**：
```cpp
cute::for_each(cute::make_int_sequence<KT::kNumStagePerSF>{}, [&](auto iter) {
    cute::for_each(cute::make_int_sequence<KT::AB_Stages>{}, [&](auto read_stage) {
        ab_full_mbar[read_stage].wait(ab_phase);
        cute::copy(s2r_copy_A, tXsA(_, _, _, read_stage), tXrA);  // SMEM→REG A
        cute::copy(s2r_copy_B, tXsB(_, _, _, read_stage), tXrB);  // SMEM→REG B
        ab_empty_mbar[read_stage].arrive();                         // 通知 AB producer 可写

        auto tCrSFA_stage = tCrSFA_frg(_, _, _, iter * KT::AB_Stages + read_stage);
        auto tCrSFB_stage = tCrSFB_frg(_, _, _, iter * KT::AB_Stages + read_stage);
        cute::gemm(mma,
            make_zip_tensor(tCrA, tCrSFA_stage),  // A data + A scale → zip
            make_zip_tensor(tCrB, tCrSFB_stage),  // B data + B scale → zip
            accum);                                 // 硬件 blockscaled MMA
    });
    ab_phase ^= 1;
});
```

**`make_zip_tensor`**：将 data tensor 和 scale tensor 打包成一个 "zip" tensor，传给 `cute::gemm`。硬件 blockscaled MMA 指令（`SM120_16x8x32_TN_VS`）接受这种打包格式，在 MMA 内部完成 FP8×scale 的缩放，累加结果到 float accum，**无需软件 descale 循环**。

#### 14.3.6 Epilogue（在 `mma()` 末尾）

```cpp
// 转换 float accum → BF16
auto epi_frg = recast<Array<ElementD, 2>>(epi);
NumericArrayConverter<ElementD, ElementAccum, 2> converter;
for_each(make_int_sequence<size(epi_frg)>{}, [&](auto i) { epi_frg(i) = converter(accum_frg(i)); });

// REG → SMEM (via STSM_N)
copy(tiled_copy_r2s, tRS_rD, tRS_sD(_, _, _, Int<0>{}));
tma_store_fence();
NamedBarrier::sync(kNumMathThreads, 0);  // 等所有 math warp 写完 SMEM
store_full_mbar[0].arrive();              // 通知 epilogue warp 可 TMA store
```

Epilogue warp（warp 8）收到信号后，用 TMA 将 SMEM → GMEM。

---

### 14.4 TRT-LLM vs HSTU FP8 关键对比

| 特性 | TRT-LLM SM120 | HSTU SM120 FP8 |
|------|--------------|----------------|
| **MMA 指令** | `SM120::BLOCKSCALED::SM120_16x8x32_TN_VS`（SM120A 专用，B100/B200 数据中心）| `SM89_16x8x32_F32E4M3E4M3F32_TN`（SM89+ 通用，含 SM120 consumer RTX 6000 Pro）|
| **Scale 集成** | 硬件 blockscaled（`make_zip_tensor`，MMA 内部缩放）| 软件 descale（GEMM 后乘标量 `acc_s *= descale_qk`）|
| **A 的 Copy Atom** | `LDSM_N`（K-fast SMEM → REG）| `DefaultCopy`（GEMM1 Q/K；`LDSM_N` 应也可用，若 SMEM 是 K-fast）|
| **B 的 Copy Atom** | `LDSM_N`（与 A 对称，因 B 也是 K-fast）| 当前 FP8 路径存在 bug（`SmemCopyAtomTransposed` 与 SMEM 写入布局不一致）|
| **B 的 SMEM 写入布局** | K-fast SW128（TMA 直接以 K-fast 格式写入）| flat SmemLayoutKV（按 N-fast 写入 V）|
| **B 的 SMEM 读取布局** | K-fast SW128（与写入完全一致，无 mismatch）| `SmemLayoutVtransposedNoSwizzle`（基于 SW128，与 flat 写入不一致 → **bug 根因**）|
| **GMEM→SMEM** | TMA（`SM90_TMA_LOAD`，硬件异步）| `cp.async`（软件异步）|
| **线程模型** | 生产者-消费者分离（专属 warp 负责 load/store）| 所有 warp 同时做计算+load（类 Flash Attention 风格）|
| **Pipeline** | AB 4-stage + SF 1-stage + Store 1-stage，mbarrier 同步 | 单 buffer，`cp_async_wait<0>` 同步 |
| **适用硬件** | SM120A（B100/B200，数据中心 Blackwell）| SM120 consumer（RTX 6000 Pro，工作站 Blackwell）|

**最重要的启发（对 HSTU FP8 V-operand bug 的修复）**：

TRT-LLM 中 B 操作数使用 `LDSM_N` 的原因是**写入 SMEM 和从 SMEM 读取使用完全相同的 K-fast SW128 布局**（write layout = read layout）。这是关键原则：

> **SMEM copy atom 的选择必须与 SMEM 的物理布局匹配。**
> - K-fast SMEM → LDSM_N（非转置，沿 K 方向连续读取 16 字节）
> - N-fast SMEM → LDSM_T（转置，沿 N 方向读取后 transpose 进寄存器）

HSTU FP8 的 V-operand bug 根因：
1. V 写入 SMEM 时用 `SmemLayoutKV`（flat，N-fast：V[row, col] 中 row 是 N 方向，stride=1 for col）
2. 读取时用 `SmemLayoutVtransposedNoSwizzle`（基于 SW128 的 K-fast 布局的转置视图）
3. 两者描述的物理地址映射不同 → 读到错误数据

修复方向：统一写入和读取布局（选择其中一种，并相应地选择 LDSM_N 或 LDSM_T）。

---

### 14.5 `SmemCopyAtomA` vs `SmemLayoutAtomA` 功能区分

这是一个常见混淆点，以下是精确区分：

| 概念 | 类型别名 | 作用阶段 | 描述 |
|------|---------|---------|------|
| `SmemLayoutAtomA` | `GMMA::Layout_K_SW128_Atom<ElementA>` | **GMEM→SMEM**（write）AND **SMEM→REG**（read） | 描述 SMEM 的物理排布（swizzle 模式）；TMA 按此 layout 写入 SMEM；`make_tiled_copy_A` 也按此 layout 分析线程访问模式 |
| `SmemCopyAtomA` | `Copy_Atom<SM75_U32x4_LDSM_N, ElementA>` | **SMEM→REG**（read only） | 实际执行 SMEM→REG 的硬件指令；必须与 `SmemLayoutAtomA` 兼容（K-fast 布局 + LDSM_N = 零冲突）|

**关键点**：
- `SmemLayoutAtomA` 同时影响两个阶段（write 和 read），因为它描述了 SMEM 的物理结构
- `SmemCopyAtomA` 只影响 read 阶段（SMEM→REG 的具体指令）
- 两者必须兼容：layout 决定了数据在 SMEM 中的排列，copy atom 决定了如何高效读取该排列
- TMA write 路径：TMA 使用 `SmemLayoutA`（即 Atom 平铺后的完整 layout）直接写入，无需指定 copy atom
- CP.async write 路径（HSTU）：需要单独的 `GmemCopyAtom`（LDGSTS），`SmemLayoutAtom` 决定写入地址

**一句话总结**：
> `SmemLayoutAtom` = SMEM 的地图（决定每个元素住哪里）
> `SmemCopyAtom` = 读 SMEM 的交通工具（决定用什么指令去取元素）
> 地图和交通工具必须匹配，否则取到错误的元素。

---

## 17. `setmaxnreg` 与 nvcc 寄存器分配 / spill

**现象**：`setmaxnreg.inc` 在运行时提高某 warp 可用的硬件寄存器上限，但 `ptxas -v` 仍显示约 96 个 GPR，且出现严重 spill；SASS 中可能出现较大寄存器编号区间却未承载编译器分配的变量。

**原因**：
- **静态分配**（nvcc/ptxas）决定在指令里实际用多少寄存器、是否 spill；**`setmaxnreg` 不替代**这一步。
- 单参数 `__launch_bounds__(N)` 等价于第二参数为 0，可能仍让编译器用**其它启发式**（追求 occupancy）压低每线程寄存器并选择 spill。
- 同一线程块内**所有线程共享同一套**编译期寄存器分配；load/math 分叉会迫使编译器做折中。

**常用手段**：
- `__launch_bounds__(maxThreadsPerBlock, 1)`：明确「每 SM 至少常驻 1 个 block」，通常**抬高**允许的每线程寄存器上限、有利于减少 spill（与 `setmaxnreg` 目标一致时需再结合实测）。
- 编译选项：`--maxrregcount=224`（或 255）作为**上限**与硬件预算对齐；必要时尝试 `-Xptxas --register-usage-level=...`（依 CUDA 版本文档）。
- 代码侧：缩短 live range（`__syncthreads`/作用域拆分）、减少大块模板同时存活，比单纯依赖 `setmaxnreg` 更直接。

---

## 18. Phase 22：BF16 默认路径优化结论

Phase 22 第一轮结论：当前待提交 diff 只保留默认 BF16 cp.async 路径中的有效性能改动，不保留 BF16 WS/TMA 实验代码。

保留结论：
- BF16 no-RAB、headDim=128 默认主配置从 `{kBlockM=128, kBlockN=64, kNWarps=8}` 切到 `{kBlockM=128, kBlockN=128, kNWarps=8}`。
- 长序列 benchmark 有实际收益：`bs=8 seq=4096 h=16 full` 约 `365.0 -> 377.8/379.1 TFLOPS`，causal 约 `648.3 -> 656.3/656.8 TFLOPS`。
- BF16 GEMM1-only debug reference 必须跟随当前 BF16 `kBlockN`。Phase 22 默认 BF16 no-RAB headDim128 已切到 `kBlockN=128`。
- BF16 默认路径已经使用 LDSM：Q/K 使用 `SM75_U32x4_LDSM_N`，V transposed view 使用 `SM75_U16x8_LDSM_T`。本轮性能提升不是由新增 LDSM 带来的。

已回退的实验结论：
- BF16 WS/TMA correctness 可以成立，但性能显著低于默认 cp.async；不作为当前性能路径。
- Q SMEM reuse 可行，但需要严格延迟 release；过早释放会在随机数据 GEMM1 中出现不稳定错误。
- BF16 O TMA store 可行，但 benchmark 基本不变，说明该实验路径的主要瓶颈不在 epilogue store。
- FP8 WS 的手写 `stmatrix.sync.aligned.x4.m8n8.shared.b16` 输出路径不能直接用于 BF16。FP8 路径依赖 SM120 QMMA C fragment 的 `PermMmaTileN` 排列；BF16 使用 SM80 `mma.sync` C fragment，直接套用会导致输出重排，实测 `bf16_gt_cos` 约 0.27-0.36。
- BF16 WS/TMA 若后续重启，需要先解决 TMA descriptor 对任意 `cu_seqlens` offset 的安全性，以及双缓冲 SMEM 压力和 load/math warp 同步开销。
- 单缓冲 BF16 路径尝试把 K_next `cp.async` 提前到 GEMM1 后：不加 CTA barrier 会因为部分 warp 提前覆盖 `sK` 而导致 SEQ=512 BF16 correctness 下降；加 GEMM1 后 barrier 后 correctness 恢复，但 `bs=8 seq=4096 h=16` kernel-only 从 107 基线 full `377.8`/causal `656.3` TFLOPS 降到 full `369.3`/causal `644.5` TFLOPS。该单缓冲提前预取方案不保留。
- K-only ping-pong SMEM 从容量上可行：BF16 no-RAB hdim128 当前 dynamic SMEM 为 64KB，额外 K stage 后约 96KB，低于 SM120 当前 100KB/SM 配置上限。实测 correctness 通过（`1test_results/261_phase22_bf16_konly_double_buffer_accuracy.log`、`262_phase22_bf16_konly_double_buffer_hstu_test_main.log`），但 benchmark 无稳定收益：`2benchmark_results/109_phase22_bf16_konly_double_buffer_kernel.log` 为 full `372.4`/causal `650.8` TFLOPS，repeat `110_phase22_bf16_konly_double_buffer_repeat.log` 为 full `370.6`/causal `648.0` TFLOPS，低于 107 基线 full `377.8`/causal `656.3`。该方案不保留。
- BF16 no-RAB hdim128 causal 的 2CTA occupancy 候选 `{kBlockM=64, kBlockN=64, kNWarps=4}` 必须配合 `__launch_bounds__(128, 2)` 检查 SASS。当前候选 SASS：`4sass_dump_ws/hstu_fwd_kernel_sm120_bf16_hdim128_causal_tile64x64x4_lb2.sass`，`cuobjdump --dump-resource-usage` 显示 `REG:199 STACK:0 LOCAL:0`，SASS 中 `LDL/STL` 计数为 0；说明编成 2CTA 目标没有引入 register spill。该候选 benchmark 不稳定，不能只凭 quick run 认定有收益，后续必须用 NCU 确认实际 occupancy 和 stall 分布。
- BF16 no-RAB hdim128 causal 的 `{kBlockM=128, kBlockN=64, kNWarps=4} + __launch_bounds__(128, 2)` 候选已验证不应保留。SASS：`4sass_dump_ws/hstu_fwd_kernel_sm120_bf16_hdim128_causal_tile128x64x4_lb2.sass`，`cuobjdump --dump-resource-usage` 为 `REG:255 STACK:200 LOCAL:0`，SASS 中 `LDL=57`、`STL=55`。`hstu_test.py` BF16 主用例通过（`1test_results/266_phase22_bf16_tile128x64x4_lb2_hstu_test_main.log`），但 `bs=8 seq=4096 h=16` kernel-only benchmark（`2benchmark_results/117_phase22_bf16_causal_tile128x64x4_lb2_kernel.log`）显示 full `376.8`、causal `496.7 TFLOPS`；causal 相对 107 基线 `656.3 TFLOPS` 大幅回退，主要原因是 2CTA launch bound 下寄存器压力过高导致 spill。
- BF16 no-RAB hdim128 causal 的 `{64,128,4} + K/V SMEM alias + __launch_bounds__(128,2)` 已验证不保留。该方案把 K/V 共用同一块 SMEM，SASS 无 spill（`REG:230 STACK:0 LOCAL:0`，`LDL/STL=0`，`4sass_dump_ws/hstu_fwd_kernel_sm120_bf16_hdim128_causal_tile64x128x4_kv_alias_lb2.sass`），correctness 通过（`1test_results/268_phase22_bf16_tile64x128x4_kv_alias_lb2_hstu_test_main.log`、`269_phase22_bf16_tile64x128x4_kv_alias_lb2_accuracy.log`），但 benchmark `bs=8 seq=4096 h=16` causal 只有 `580.0 TFLOPS`（`2benchmark_results/119_phase22_bf16_tile64x128x4_kv_alias_lb2_kernel_repeat.log`）。NCU（`3profile_results/059_phase22_bf16_tile64x128x4_kv_alias_lb2_causal_ncu_bf16.csv`）显示 dynamic SMEM 为 `32768B`、active warps/scheduler 仍为 `1.93`，且 `No Eligible` 从 `{64,64,4}+lb2` 的 `78.12%` 恶化到 `81.02%`；延后 V load 破坏了原先 overlap。
- BF16 no-RAB hdim128 causal 的 `{64,64,4} + __launch_bounds__(128,3)` 已验证不保留。该方案无 spill（`REG:168 STACK:0 LOCAL:0`，`LDL/STL=0`，`4sass_dump_ws/hstu_fwd_kernel_sm120_bf16_hdim128_causal_tile64x64x4_lb3.sass`），correctness 通过（`1test_results/270_phase22_bf16_tile64x64x4_lb3_hstu_test_main.log`、`271_phase22_bf16_tile64x64x4_lb3_accuracy.log`）。NCU（`3profile_results/060_phase22_bf16_tile64x64x4_lb3_causal_ncu_bf16.csv`）显示 active warps/scheduler 提升到 `2.80`、achieved occupancy `23.36%`、No Eligible 降到 `75.92%`；但同环境 benchmark 不赢默认 `{128,128,8}`：`bs=8 seq=4096 h=16` causal 为 `653.0 TFLOPS`（`2benchmark_results/120_phase22_bf16_tile64x64x4_lb3_kernel.log`），默认 baseline 为 `668.6 TFLOPS`（`2benchmark_results/121_phase22_bf16_baseline_kbn128_sameenv_kernel.log`）；`bs=4 seq=2048 h=16` causal 也基本持平/略低（`507.9` vs baseline `508.7 TFLOPS`，`122/123`）。提高 occupancy 本身不足以抵消更小 M/N tile 的额外开销。
