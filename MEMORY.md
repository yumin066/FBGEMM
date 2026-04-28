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
