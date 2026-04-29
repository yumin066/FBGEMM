# HSTU SM120 FP8 当前计划

来源：`CLAUDE.md`、`PLAN.md`、`memory.md`。仓库中没有找到小写 `plan.md`，实际存在的是 `PLAN.md`。`PLAN.md` 最后更新到 Phase 19 候选方向，`CLAUDE.md` 中包含更新的 Phase 20/21 状态，因此本文件以 `CLAUDE.md` 的 Phase 21 记录为最新状态。

## 当前状态

项目当前位于 SM120 FP8 WS kernel 优化后续阶段。Phase 21 已完成并验证，最新基线为 benchmark 056。Phase 22 已完成第一轮 BF16 迁移评估：当前提交只保留默认 BF16 路径的 `kBlockN=128` 性能改动；WS/TMA 实验因性能回退不进入待提交 diff。Phase 23 已评估 FP8 WS persistent kernel，并保留 pure causal 的静态首尾配对 launch queue。

最新有效基线：

- Phase 21：Opt A + Opt C，完成于 2026-04-24。
- Phase 23：FP8 WS pure causal 使用静态首尾配对 launch queue，full/其他 mask 保持原始 3D grid。
- bs=8 seq=4096 h=16 causal：FP8 1131.8 TFLOPS，BF16 647.0 TFLOPS，FP8 比 BF16 快 74.9%。
- bs=8 seq=4096 h=16 full：FP8 640.0 TFLOPS，BF16 365.6 TFLOPS，FP8 比 BF16 快 75.1%。
- 准确性：`sweep_accuracy.py` 全部 `fp8_gt_cos >= 0.9996`。
- examples：`run_hstu8_examples.sh` 为 14/14 PASS。

## Phase 22：BF16 默认路径优化

目标：只保留能提高默认 BF16 kernel 性能的改动，避免把 WS/TMA 实验代码放进待提交 diff。

当前保留的性能改动：

- 默认 BF16 no-RAB headDim128 主配置从 `{kBlockM=128, kBlockN=64, kNWarps=8}` 切到 `{kBlockM=128, kBlockN=128, kNWarps=8}`。
- `sweep_accuracy.py` 的 GEMM1-only reference 已同步为默认 BF16 `kBlockN=128`。

已回退的实验方向：

- BF16 WS/TMA path、Q SMEM reuse、O TMA store 实验 correctness 可过，但 benchmark 低于默认 cp.async，不进入当前提交。
- BF16 split-mask 实验没有稳定收益，已回退。

验证记录：

- `1test_results/121_phase22_bf16_default_kbn128_accuracy.log`：默认 BF16 `kBlockN=128` correctness 通过，BF16 `bf16_gt_cos=1.0000`，FP8 `fp8_gt_cos>=0.9996`。
- `1test_results/122_phase22_bf16_default_kbn128_gemm1_reference.log`：默认 BF16 `kBlockN=128` GEMM1-only reference 通过，BF16/FP8 `*_gt_cos=1.0000`。
- `1test_results/123_phase22_bf16_default_kbn128_hstu_test_main.log`：`hstu_test.py` BF16 主用例通过，max diff `0.0001220703125`。
- `1test_results/124_phase22_bf16_default_kbn128_final_accuracy.log`：最终默认路径 sweep 通过，BF16 `bf16_gt_cos=1.0000`。
- `2benchmark_results/106_phase22_bf16_default_kbn128_kernel.log`：`bs=8 seq=4096 h=16 full` 为 `379.1 TFLOPS`，causal 为 `656.8 TFLOPS`。
- `2benchmark_results/107_phase22_bf16_default_kbn128_kernel_repeat.log`：重复 benchmark，full 为 `377.8 TFLOPS`，causal 为 `656.3 TFLOPS`。相对旧 `kBlockN=64` 记录约 `365.0/648.3 TFLOPS` 有稳定提升。

下一步：

- 在清理后的默认 BF16 `kBlockN=128` 路径上重新跑 NCU，确认当前 `No Eligible`、Long Scoreboard、Math Throttle、SMEM bank conflict、register spill 和 occupancy。

## Phase 23：FP8 WS 静态首尾配对 queue

目标：评估 FP8 WS persistent kernel，并利用 causal tile 工作量从重到轻的结构做静态负载拉平。

当前实现：

- 仅 pure causal no-RAB WS kernel 使用静态首尾配对 launch order。
- pure causal launch 改为一维 `grid.x = total_tiles`，`blockIdx.x` 映射为 `0, total_tiles-1, 1, total_tiles-2, ...`。
- full、local、context、target、arbitrary 等其他 WS 实例保持原始三维 grid：`dim3(num_m_block, h, b)`。
- 这样保留 causal 的 heavy/light launch-order 拉平，同时避免 persistent wrapper loop 造成的寄存器 spill。

已拒绝的中间实现：

- 纯 static grid-stride persistent correctness 可过，但 causal 性能从同环境基线 `1105.1 TFLOPS` 降到 `1037.8 TFLOPS`，主要原因是 tile 分配相位固定导致 causal 重 tile 分布不均。
- dynamic persistent work queue correctness 可过，但收益小且 tile 间必须做 CTA sync/atomic，同环境 100 iters full/causal 约 `644.0/1111.7 TFLOPS`，200 iters repeat causal 回落到 `1098.1 TFLOPS`。
- 在 wrapper 中新增普通或 padded static `__shared__` 保存 next tile 会破坏 WS dynamic SMEM 绝对对齐，触发 `misaligned address`。
- paired persistent loop 可以把 causal 推到 `1138.3 TFLOPS` 的单次结果，但 full 明显回退；SASS 显示 wrapper loop 让 full/causal 实例出现 `STACK` 和大量 `LDL/STL`。因此不保留 persistent loop 形态。

验证记录：

- `1test_results/287_phase23_fp8_ws_causal_static_launch_pair_accuracy.log`：FP8 sweep 通过，`fp8_gt_cos >= 0.9996`。
- `1test_results/288_phase23_fp8_ws_causal_static_launch_pair_examples.log`：`run_hstu8_examples.sh` 为 14/14 PASS。
- `2benchmark_results/135_27011a7f_gpu_unlocked_phase23_fp8_ws_causal_static_launch_pair_kernel200.log`：bs=8 seq=4096 h=16 full FP8 `654.1 TFLOPS`，causal FP8 `1135.5 TFLOPS`。
- `2benchmark_results/136_27011a7f_gpu_unlocked_phase23_fp8_ws_causal_static_launch_pair_kernel100.log`：repeat full FP8 `649.9 TFLOPS`，causal FP8 `1113.4 TFLOPS`。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full_static_launch_pair.sass`：full 实例 `REG:168 STACK:0 LOCAL:0`，`LDL/STL=0`。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_causal_static_launch_pair.sass`：causal 实例 `REG:168 STACK:0 LOCAL:0`，`LDL/STL=0`。

下一步：

- 对 static launch pair 版本跑 NCU，重点确认 tail wave 是否改善，以及 No Eligible/Long Scoreboard 是否变化。
- 如果继续做真正 persistent kernel，需要先把 `hstu_compute_attn_1rowblock_sm120_fp8_ws` 拆成 producer/consumer/store 三段，避免 wrapper loop + inline body 导致 spill。

## 已完成阶段摘要

- Phase 0-3：BF16 实现、FP8 基础框架、TMA 基础能力，`fp8_gt_cos ~= 0.9996`。
- Phase 4-5：K+V double-prefetch、TMA K/V^T，延迟降低约 6-7%。
- Phase 6：Warp-specialized TMA double-buffer kernel，14/14 PASS。
- Phase 7-8：TMA SFB、TMA SFV，14/14 PASS。
- Phase 9：12-warp + `setmaxnreg 216`，FP8 TFLOPS 提升 27-51%。
- Phase 10：消除 SMEM transpose，使用 col-major V，seq>=2048 FP8 超过 BF16。
- Phase 11：PTX `LDSM_T` + MN_SW128 swizzle，seq>=1024 FP8 超过 BF16。
- Phase 12：代码整洁化，引入本地 `sm120_qmma_builder.h`，无功能变化。
- Phase 13：AtomLayout `<_8,_1,_1>` + warp shuffle 消除 P staging，seq>=1024 提升 10-30%。
- Phase 14：`setmaxnreg 224/56` + 消除全部 LDL/STL register spill，FP8 突破 1 TFLOPS。
- Phase 15：Q+SFA TMA 迁移到 math warp，删除 load warp Q copy + barrier。
- Phase 16：Q+SFA preload 移到 mbarrier wait 前，隐藏 LDSM latency，causal 提升 1-3%。
- Phase 17：同步点精简，删除 S3，S2 改为 `wait_mbar_parity`，S5 改为 `bar.sync 1,256`。
- Phase 18：`cvt.e4m3x2` 双路 FP8 转换，F2FP 指令数 64 降到 32，FP8 全面提升 1-4%。
- Phase 19：TMA Store Output，epilogue SMEM 到 GMEM 改为 thread 0 TMA bulk store。
- Phase 20：`kBlockM=256` M-direction streaming 已尝试并搁置，因寄存器压力导致严重性能回退。
- Phase 21：N-loop 不变量提升 + split N-loop + `if constexpr` masking，bs=8 causal 提升 0.7%，full 提升 1.6%。
- Phase 23：FP8 WS pure causal 静态首尾配对 launch queue，correctness 通过，full 不回退，causal 有小幅/波动收益。

## Phase 21 细节

Opt A：N-loop 不变量提升。

- 将 `tCrQ`、`tCrSFA`、`tCrSFA_frg` 从每个 N tile 重复加载改为 N-loop 前一次加载。
- 安全依据：`sQ_persist` 与 `smem_sfa_ptr` 由 Q+SFA TMA 写入，在 S2 `wait_mbar_parity(q_tma_mbar_ptr, 0)` 后对 math warp 可见，N-loop 期间不再修改。
- 明确不提升 `tCrSFP`。曾尝试提升后 H=4 出现 `fp8_gt_cos=0.9918` race condition，已恢复 per-tile。

Opt C：split N-loop + `if constexpr` masking。

- 将 N-loop 主体封装为 `run_n_tile` lambda。
- 对 `Is_causal && !Is_arbitrary && !Is_local` 拆分：
  - 前 `n_masking_steps` 个 tile 使用 `kIsMasking=true`。
  - 后续 tile 使用 `kIsMasking=false`，让 `apply_mask_bs` 被编译期消除。

验证：

- `1test_results/090_phase21_optA_optC_fix.log`：全部 `fp8_gt_cos >= 0.9996`。
- `run_hstu8_examples.sh`：14/14 PASS。
- benchmark 056：bs=8 seq=4096 causal FP8 1131.8 TFLOPS，full FP8 640.0 TFLOPS。

## 暂停路线

Phase 20 的 `kBlockM=256` 路线暂时搁置。

原因：

- `kNMSubtiles=2` 时，`acc_o_0` 和 `acc_o_1` 共 128 个 F32 accumulator 寄存器全程活跃。
- GEMM1 峰值估算约 355 regs，远超 `setmaxnreg math=232`。
- 即使做 N-streaming 或 K-step streaming，GEMM1 阶段仍约 263 regs，仍会 spill。
- 实测 Phase 20b 和后续修复版本性能回退 53-67%。

结论：除非能大幅削减 accumulator 以外的寄存器活跃量，当前约束下不继续推进 `kBlockM=256`。

## 当前瓶颈

主要瓶颈仍是 No Eligible stall 和 mbarrier wait。

已知约束：

- FP8 WS kernel 约 85KB SMEM/CTA，受 Block Limit SMEM=1 约束，通常只能 1 CTA/SM。
- math warp 等待 K/V TMA 数据时，调度器经常无可发射 warp。
- Phase 16-21 已逐步隐藏或消除部分同步和加载开销，但结构性瓶颈未消失。
- register spill 已在 Phase 14 清零，后续改动必须继续用 SASS 检查 `LDL`/`STL`。
- BF16 no-RAB hdim128 causal 的 `{64,64,4} + __launch_bounds__(128,2)` 候选已完成 SASS 检查：目标 kernel `REG:199 STACK:0 LOCAL:0`，`LDL/STL` 计数为 0。该候选没有因 2CTA 编译产生 register spill；性能波动需要继续用 NCU 看实际 occupancy、No Eligible 和 tile 工作量变化。
- BF16 no-RAB hdim128 causal 的 `{128,64,4} + __launch_bounds__(128,2)` 候选已拒绝：目标 kernel `REG:255 STACK:200 LOCAL:0`，SASS 中 `LDL=57`、`STL=55`；benchmark `bs=8 seq=4096 h=16` causal 仅 `496.7 TFLOPS`，明显低于 107 基线 `656.3 TFLOPS`。
- BF16 causal occupancy 后续实验已拒绝：`{64,128,4}+K/V alias+lb2` 无 spill但 causal 只有 `580.0 TFLOPS`；`{64,64,4}+lb3` 无 spill且 NCU occupancy 提升到 `23.36%`，但同环境 benchmark 仍低于默认 `{128,128,8}`（`653.0` vs `668.6 TFLOPS`）。当前 BF16 性能路径应回到默认 `{128,128,8}`。

## 下一步候选

优先级建议：

1. 对 Phase 21 基线做新的 NCU profile，确认 No Eligible、Long Scoreboard、SMEM、register、occupancy 的最新分布。
2. 评估 `kBlockN=256` 的 SMEM 可行性和寄存器压力，重点检查是否超过 85KB 预算以及是否恶化 1 CTA/SM 限制。
3. 评估能否把 SMEM 降到约 50KB，以解除 1 CTA/SM 限制；这是高难度方向，但可能直接改善 No Eligible。
4. 对 Phase 23 static launch pair 版本做 NCU；若收益来自 tail-wave 拉平，继续评估其他 shape 的静态 queue；若收益不稳，回到 Phase 21 调度。
5. Phase 22 下一步：停止通过降低 BF16 causal tile 或单纯提高 launch bound 追求 occupancy；已测方案没有超过默认 `{128,128,8}`。后续若继续 BF16，应优先寻找减少指令/同步/冗余工作且不缩小主 tile 的方案，或做 runtime 多 kernel dispatch 但必须证明目标 shape 有稳定收益。

不建议立即继续：

- 不继续 `kBlockM=256`，除非先提出明确的寄存器活跃量削减方案。
- 不把 `tCrSFP` 提升到 N-loop 外；已有 race condition 记录。

## 验证清单

每次 CUDA/CuTe 内核改动后至少执行：

- 重新编译 HSTU extension。
- `HSTU_SWEEP_FP8_QUANT_MODE=2 python sweep_accuracy.py`，要求 `fp8_gt_cos >= 0.995`，理想值 `>= 0.9996`。
- BF16 相关改动运行 `hstu_test.py` 的 `HSTU16Test` 定向用例，至少覆盖 aligned WS 场景和不对齐 fallback 场景。
- FP8 相关改动运行 `bash run_hstu8_examples.sh`，要求 14/14 PASS。

性能相关改动还需要：

- 运行 `bench_hstu_attn_sm120.py`，记录到 `2benchmark_results/NNN_xxx.log`。
- 必要时运行 `run_profile.sh NNN <描述>`。
- 必要时运行 `dump_sass_fp8_ws.sh`，检查 `LDL`/`STL` spill。

## 关键风险

- TMA descriptor 维度顺序必须保证 innermost 维度 stride=1；TMA Store Output 已因 `(total_q, h, d)` 触发过 descriptor 初始化失败，正确顺序为 `(total_q, d, h)`。
- `mbarrier.init` 后其他线程等待前需要同步；Phase 17 记录中缺少 `bar.sync 2,256` 会导致 flaky crash。
- Q/SFA TMA math-warp 化后，`expected` 和 parity 必须保持正确；错误会导致死锁或等待错误 stage。
- warp shuffle 重排必须先 shuffle pk0/pk1，再由目标线程按 `tiq >> 1` 选择；否则会出现 N-atom 错配。
- SMEM TMA 目标地址必须满足 128B 对齐。
- FP8 WS wrapper 不要新增普通 static `__shared__`，否则可能移动 dynamic SMEM 绝对地址并破坏 SW128/TMA 对齐。persistent wrapper loop 还容易让 inline WS body 增加 stack/local spill；保留前必须查 SASS。
- CuTe TMA header 曾有 debug `printf` 导致严重性能回退；若性能突然退化，检查相关 header 和 SASS/profile。
