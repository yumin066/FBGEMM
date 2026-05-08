# HSTU SM120 项目背景摘要

本文件保留从 `memory.md` 和 `CLAUDE.md` 中迁移出的长期背景。它不是完整历史；完整阶段细节仍可查 `HISTORY.md`、`CLAUDE.md` 和历史 benchmark/profile 日志。

## 项目定位

本仓库基于 FBGEMM HSTU attention。上游已有 Hopper SM90 相关实现，本项目目标是让 Blackwell SM120 支持并优化 HSTU attention，重点是 FP8 block-scale 路径，同时保留 BF16 路径用于对照。

当前主要活跃目录：

- `fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/`
- `fbgemm_gpu/experimental/hstu/benchmark/`
- 项目根目录下的 `sweep_accuracy.py`、`run_profile.sh`、`dump_sass_fp8_ws.sh`

## 当前最佳基线

最新记录为 Phase 21，benchmark 056：

- bs=8 seq=4096 h=16 causal：FP8 1131.8 TFLOPS，BF16 647.0 TFLOPS，FP8 比 BF16 快 74.9%。
- bs=8 seq=4096 h=16 full：FP8 640.0 TFLOPS，BF16 365.6 TFLOPS，FP8 比 BF16 快 75.1%。
- `sweep_accuracy.py` 全部 `fp8_gt_cos >= 0.9996`。
- `run_hstu8_examples.sh` 为 14/14 PASS。

## 长期有效的技术事实

- FP8 dtype 为 `cute::float_e4m3_t`，PyTorch 为 `torch.float8_e4m3fn`。
- `quant_mode=2` 表示 SM120 FP8 block-scale，`quant_mode=-1` 表示 BF16。
- Q/K 是 GEMM1 输入，V 是 GEMM2 输入。
- scale factor 使用 e8m0，每 128 个 K 元素对应一个 scale，4 个 scale 打包为一个 `int32`。
- FP8 WS 内核采用 12 warp 结构：8 个 math warp 和 4 个 load warpgroup warp，其中 active load warp 发 TMA。
- `setmaxnreg` 需要完整 warpgroup 配合，否则可能因 `WARPSYNC.ALL` 死锁。
- register spill 必须通过 SASS 中的 `LDL`/`STL` 检查；Phase 14 已做到全部清零。
- 当前结构性瓶颈是 math warp 等 K/V TMA 的 mbarrier wait，表现为 No Eligible stall 高。
- 约 85KB SMEM/CTA 导致 Block Limit SMEM=1，是后续性能突破的重要约束。

## 重要 bug 与经验

- CuTe `copy_sm90_tma.hpp` 中曾有调试 `printf`，导致 FP8 kernel 慢 30-80x。遇到异常性能退化时检查 header、SASS 和 profile。
- TMA descriptor 的 global tensor 维度顺序必须让 innermost 维度 stride=1。TMA O store 正确顺序为 `(total_q, d, h)`。
- TMA 目标 SMEM 地址需要 128B 对齐；Is_arbitrary case 曾因填充只到 8B 对齐而 misaligned。
- `bar.sync 2,256` 对 Phase 17 的 q_tma_mbar 初始化可见性是必要的，否则会 flaky crash。
- Q+SFA TMA 迁移到 math warp 后，mbarrier `expected` 和 parity 错误会导致死锁。
- warp shuffle 消除 P staging 时，C fragment 到 A fragment 重排必须先 shuffle pk0/pk1，再由目标线程选择，否则 cosine similarity 会降到约 0.66。
- `tCrSFP` 不应提升到 N-loop 外；历史尝试导致 H=4 race condition，`fp8_gt_cos=0.9918`。
- `kBlockM=256` 路线已尝试并搁置，寄存器压力导致 53-67% 性能回退。

## 历史脉络

早期 `memory.md` 记录了 Phase 6 之前的一些关键背景：

- Phase 4 的 K+V double-prefetch 消除了迭代中途 blocking V wait。
- Phase 5 用 TMA K+V^T 消除 kernel 侧 V transpose 的一部分开销。
- Phase 6 建立 warp-specialized TMA double-buffer pipeline，并确认 TMA 才适合 WS overlap；WS + cp.async 路线不可行。
- 后续 Phase 7-21 逐步迁移 SFB/SFV/Q/SFA/O 到 TMA，调整 QMMA layout，消除 P staging、register spill 和若干同步点。

## 当前文档约定

- `AGENTS.md`：Codex 长期协作规则、构建、验证和 debug 规程。
- `PLANS.md`：当前计划、最新状态、候选方向和风险。
- `MEMORY.md`：项目背景摘要。
- `HISTORY.md`：详细历史阶段记录。
- `CLAUDE.md`、`PLAN.md`、`memory.md`：迁移来源，暂时保留不改。

## 换机 handoff：Phase 33

时间：2026-05-08。当前机器即将释放，本段记录下一台机器继续工作的入口。

当前目标：

- BF16 CuTe DSL：GPU kernel-only 性能至少达到 BF16 C++ 的 95%。当前可信 kernel-only 约 87-88%，wrapper-call geomean 约 0.203 另算。
- FP8 WS：聚焦 `bs=1,seq=2048,h=4,d=256`，分析并优化 RAB/DRAB 降 TFLOPS、causal 低于 full、context TFLOPS 偏高，以及 benchmark `596` vs `173` 的 D128 regression。

本轮已完成：

- 两个只读 sub-agent 已完成分析，均未改文件。
- 已更新 `PLANS.md` 和 `knowledge.md`，写入 Phase 33 证据、计划和风险。
- 已锁频跑 FP8 WS NCU，并已解锁 GPU clocks。产物：
  - `3profile_results/phase33_fp8_d256_s2048_h4_full_none_ncu.ncu-rep`
  - `3profile_results/phase33_fp8_d256_s2048_h4_full_rab_ncu.ncu-rep`
  - `3profile_results/phase33_fp8_d256_s2048_h4_causal_none_ncu.ncu-rep`
  - `3profile_results/phase33_fp8_d256_s2048_h4_causal_rab_ncu.ncu-rep`
  - 同名 `.csv`
- 没有完成任何业务代码优化改动；当前只应把 `PLANS.md`、`knowledge.md`、`MEMORY.md` 视为本轮新增 handoff。

关键结论：

- BF16 CuTe DSL kernel 差距最可能来自没有复刻 C++ 的 `Q-in-reg + Share_Q_K_smem`：DSL 每个 N block 重复 LDSM Q，dynamic SMEM 约 99KB；C++ D128 约 64KB。
- FP8 WS RAB/DRAB 降 TFLOPS 的第一主因是 math critical path 上的 global scalar RAB load。D256 full+RAB NCU 无 local spill，但 DRAM throughput 升高、L2 hit 降低、No Eligible 升高。
- D256 causal no-RAB latency 与 full 接近，而 valid pairs 只有 full 的约一半；同时 D256 non-paged causal 当前不走 paired persistent scheduler。
- benchmark `596` 相比 `173`：D128 full 无 regression；D128 causal long-seq 有约 6% 小回退。

下一步建议：

1. 优先实现 BF16 DSL D64/D128 no-RAB 的 Q-in-reg/Share-Q-K-smem，先用 kernel-only profile 验证是否接近 95%。
2. FP8 WS RAB 不要重复单 K warp 预取 RAB 到 SMEM 的旧方案；如优化，优先考虑独立 RAB TMA/异步搬运、layout/packing，或 D256 中复用已消费 K/V stage。
3. 所有性能结论继续锁频后记录到 `2benchmark_results/`；CUDA/CuTe 改动后按 `AGENTS.md` 跑 `hstu_test.py`、`sweep_accuracy.py` 和 `run_hstu8_examples.sh`。
