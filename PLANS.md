# HSTU SM120 FP8 当前计划

来源：`CLAUDE.md`、`PLAN.md`、`memory.md`。仓库中没有找到小写 `plan.md`，实际存在的是 `PLAN.md`。`PLAN.md` 最后更新到 Phase 19 候选方向，`CLAUDE.md` 中包含更新的 Phase 20/21 状态，因此本文件以 `CLAUDE.md` 的 Phase 21 记录为最新状态。

## 当前状态

项目当前位于 SM120 FP8 WS kernel 优化阶段。Phase 21 已完成并验证，最新基线为 benchmark 056。

最新有效基线：

- Phase 21：Opt A + Opt C，完成于 2026-04-24。
- bs=8 seq=4096 h=16 causal：FP8 1131.8 TFLOPS，BF16 647.0 TFLOPS，FP8 比 BF16 快 74.9%。
- bs=8 seq=4096 h=16 full：FP8 640.0 TFLOPS，BF16 365.6 TFLOPS，FP8 比 BF16 快 75.1%。
- 准确性：`sweep_accuracy.py` 全部 `fp8_gt_cos >= 0.9996`。
- examples：`run_hstu8_examples.sh` 为 14/14 PASS。

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

## 下一步候选

优先级建议：

1. 对 Phase 21 基线做新的 NCU profile，确认 No Eligible、Long Scoreboard、SMEM、register、occupancy 的最新分布。
2. 评估 `kBlockN=256` 的 SMEM 可行性和寄存器压力，重点检查是否超过 85KB 预算以及是否恶化 1 CTA/SM 限制。
3. 评估能否把 SMEM 降到约 50KB，以解除 1 CTA/SM 限制；这是高难度方向，但可能直接改善 No Eligible。
4. 评估 persistent kernel 或双 CTA wave，让 CTA 间交替隐藏 TMA wait；前提是资源允许。
5. 评估将 Phase 13-21 中适合 BF16 的优化移植到 BF16 路径，预计收益较小但风险可控。

不建议立即继续：

- 不继续 `kBlockM=256`，除非先提出明确的寄存器活跃量削减方案。
- 不把 `tCrSFP` 提升到 N-loop 外；已有 race condition 记录。

## 验证清单

每次 CUDA/CuTe 内核改动后至少执行：

- 重新编译 HSTU extension。
- `HSTU_SWEEP_FP8_QUANT_MODE=2 python sweep_accuracy.py`，要求 `fp8_gt_cos >= 0.995`，理想值 `>= 0.9996`。
- `bash run_hstu8_examples.sh`，要求 14/14 PASS。

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
- CuTe TMA header 曾有 debug `printf` 导致严重性能回退；若性能突然退化，检查相关 header 和 SASS/profile。
