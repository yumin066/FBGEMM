# Codex 协作规则

本文件由 `CLAUDE.md`、`PLAN.md` 和 `memory.md` 迁移而来，用于 Codex 在本仓库内长期协作。Claude 专属的 MCP 调用流程不再适用；Codex 应直接阅读代码、提出判断、编辑文档或代码，并在修改后用本文件列出的验证流程闭环。

## 回复与协作

- 默认使用中文回复。
- 先理解仓库上下文，再修改文件；对 CUDA/CuTe/FP8 内核改动必须先定位相关实现和现有约束。
- 业务代码修改前应说明将改哪些文件、为什么改。
- 不要执行文件删除命令，包括 `rm`、`rm -f`、`rm -rf`、`unlink`、`find ... -delete`、`shutil.rmtree`。确需清理时先取得用户明确授权。
- 保持用户已有改动，不要回滚无关变更。
- 技术知识点需要沉淀时，更新项目根目录 `knowledge.md`，避免重复条目。
- 当前计划保存在 `PLANS.md`；历史背景摘要保存在 `MEMORY.md`。

## 项目概述

本仓库是 FBGEMM 中 HSTU attention 的 CUDA 实现。当前重点是让 Blackwell SM120 GPU 支持并优化 HSTU attention，包含 BF16 路径和 FP8 block-scale 量化路径。

核心目标：

- FP8 dtype：`cute::float_e4m3_t`，PyTorch 对应 `torch.float8_e4m3fn`。
- FP8 block-scale 量化模式：`quant_mode=2`；BF16 为 `quant_mode=-1`。
- 当前 SM120 FP8 forward 主路径覆盖 headDim32/headDim64/headDim128/headDim256；Phase 31 已修复 D64 paged full residual register spill，并完成 D32/D64/D128/D256 全量 FP8+paged kernel-only benchmark。
- `kBlockN` 必须整除 128；当前 FP8 paged KV 第一约束仍是 `page_size == kBlockN`。
- Q/K 用于 GEMM1，scale 沿 headDim 每 128 个元素 1 个 e8m0 scale；headDim128 有 1 个 D chunk，headDim256 有 2 个 D chunk。
- V 用于 GEMM2，scale 沿 N block 维度组织；4 个连续 scale block 打包为 1 个 `int32`。

当前工作焦点：

- Phase 33：FP8 WS RAB 已完成三轮优化。K-warp RAB TMA/SMEM 替代 direct-global RAB 后显著提升 RAB/DRAB，但 dense per-head BF16 RAB 仍受 RAB 字节量限制，full+RAB 达不到 no-RAB `95%`。最新保留改动是 head-shared RAB scheduler：当 `Has_rab && h_rab == 1 && h > 1` 时，persistent tile decode 改为同一 `(b,m)` 下 head-fast，相邻 CTA 复用同一 RAB tile 的 L2；non-persistent grid 改为 `dim3(h, num_m_block, b)`。`h=1` 保持旧路径。验证日志：`1test_results/638_hstu_test_rab_h1_scheduler_final_retry.log` 为 `3 passed, 1 skipped`，`1test_results/629_sweep_rab_h1_scheduler.log` 通过，`1test_results/630_hstu8_examples_rab_h1_scheduler.log` 为 `300/300 passed`。锁频 focused benchmark：per-head `2benchmark_results/633_gpu2407MHz_rab_h1_scheduler_per_head_focus.log`、head-shared `634`、no-RAB `635`；`rab_h=1` 相比 per-head RAB TFLOPS geomean `+8.3%`，但同形状对 no-RAB 仍只有约 `66.6%`。`h_rab==1` direct-global RAB add 实验 `636` full 回退，不保留；额外 masked tile predicate 没有新增可跳过 tile，也不保留。
- Phase 32：SM120 BF16 CuTe DSL prototype 已补齐 D32/D64/D128/D256 × full/causal/local/context/target/context+target/arbitrary × none/RAB/DRAB，BF16 paged KV 通过 wrapper-side dense materialization 支持，默认 C++ dispatch 不变。当前性能口径必须区分 kernel-only 和 Python wrapper-call：`nsys` kernel-only 代表 case 显示 DSL kernel 约为 C++ BF16 的 `87%`，但锁频默认 CUDA-event wrapper-call geomean 仍只有 `0.203`，不适合作为生产端到端替代。后续继续优化应先用 `nsys/ncu` 看 kernel-only 差距，再考虑是否需要把 DSL launch 下沉到 C++/extension 层，不能用 Python wrapper event geomean 直接判断 kernel 本体。
- Phase 31：已修复 `D=64 paged full` register spill。最终 SASS `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I64_paged_full_final.sass` 对应 resource 为 `REG:168 STACK:0 LOCAL:0`，无 `LDL/STL`。最终 correctness：`1test_results/phase31_final_hstu_test.log` 为 `3 passed, 1 skipped`，`1test_results/phase31_final_sweep.log` 通过，`1test_results/690_hstu8_examples_qm2.log` 为 `284/284 passed`。最终锁频全量 benchmark：`2benchmark_results/phase31_gpu2407MHz_full_fp8_paged_kernel_final.log`。
- Phase 30：SM120 FP8 `quant_mode=2` forward 已支持 headDim32/headDim64，并覆盖当前 headDim128/headDim256 已支持的所有配置组合：non-paged/paged、full/causal/local/context/target/arbitrary、none/RAB/DRAB、irregular seqlen。最新 correctness：`1test_results/phase30_hdim32_64_hstu8_matrix.log` 通过，`1test_results/phase30_hdim32_64_hstu_test_full.log` 为 `3 passed, 1 skipped`，`1test_results/670_hstu8_examples_qm2.log` 为 `284/284 passed`，`1test_results/phase30_hdim32_64_sweep.log` 通过。D32/D64 锁频 kernel-only benchmark：`2benchmark_results/phase30_gpu2407MHz_hdim32_64_kernel_only_after_guard.log`，1296 行 FP8+paged 实测，无 unsupported/ERR。
- Phase 29：已完成 repo 自带 `hstu_test.py` 的 SM120 FP8 `quant_mode=2` irregular seqlen 和 paged KV 验收矩阵。wrapper 侧构造 block-aligned physical Q/K/V/SF/func layout，向 SM120 kernel 传 padded `cu_seqlens` 和 actual `seqused`，输出再 unpad；`(99,99)` 等非 block 对齐输入现在真实运行，不再由 alignment guard 跳过。最新日志：fixed matrix `1test_results/592_phase29_hstu8_matrix_k_offsets.log`，全文件 pytest `1test_results/592_phase29_hstu_test_full_after_k_offsets.log`。
- Phase 28：补齐 SM120 FP8 paged KV + RAB/DRAB 覆盖。当前 D=128/D=256 的 full、pure causal、context+causal、target+causal、local、arbitrary + paged KV + RAB/DRAB 都已真实运行通过；local paged 不再是 `PASS-UNSUPPORTED`。
- Phase 27：逐步将 SM120 FP8 non-paged RAB/DRAB 切到 WS。当前已完成 D=128/D=256 non-paged RAB/DRAB 全配置：full、pure causal、context+causal、target+causal、local、arbitrary 都走 FP8 WS TMA；D=128 RAB 已改用 `kBlockN=64`。
- Phase 26：支持 SM120 FP8 paged KV cache 与 RAB/DRAB 配置。Phase 28 后 paged RAB/DRAB 覆盖已扩展到 complex mask/local。
- Phase 25：SM120 headDim256 forward 已接入；FP8 hdim256 correctness 路径已打通，pure non-paged causal residual register spill 已清理。
- BF16 hdim256 作为 correctness/dispatch 基线；backward 不进第一阶段。

FP8 WS Phase 23 稳定基线：

- FP8 WS pure full no-RAB 路径当前使用 branch-local grid-stride persistent kernel：host 端 `grid.x = min(total_tiles, SM_count)`，device 端 load/math 分支各自以 `gridDim.x` 为 stride 遍历 tile。
- FP8 WS pure causal no-RAB 路径当前使用 branch-local paired persistent kernel：host 端 `grid.x = min(total_tile_pairs, SM_count)`，device 端 load/math 分支各自以 `gridDim.x` 为 stride 遍历 tile-pair。
- 每个 tile-pair 处理 `tile_pair` 和 `total_tiles - 1 - tile_pair`，用于拉平 causal heavy/light tile 的 wave/tail。
- tile id 使用静态本地 decode：load/math 两边运行同一确定性 scheduler，不使用 dynamic work queue、atomic counter 或 shared tile-id state。
- FP8 WS local、context、target、arbitrary 等其他实例保持原始三维 grid：`dim3(num_m_block, h, b)`。
- epilogue/O-store 已归到 active load warp：math warps 负责写 O 到 SMEM，active load warp 负责 `tma_o` store 并等待完成。
- persistent full/causal 路径已把 K/V 双缓冲的 producer/consumer mbarrier 初始化和 CTA S1 提到 static scheduler loop 外，并跨 tile 维护 mbarrier parity 和 K/V stage；`S_arb` 仍只属于 arbitrary/non-persistent 路径。
- 当前实验版进一步把 4 个 load warp 分工为 Q/SFA load、K/SFB load、V/SFV load、O store；Q/K/V/O 使用 ready/empty mbarrier 做细粒度 producer/consumer 同步，已移除 tile-end load-only `bar.sync 4,128`。O 不新增 double buffer，而是复用当前 tile 最后被 math 消费的 K/V stage。math epilogue 直接用 `math_stage ^ 1` 选 O stage；O-store warp 使用同一 tile 参数和当前起始 stage 推导 O stage。
- 不保留 dynamic persistent work queue：correctness 可过，但 tile 间需要 CTA sync/atomic，性能收益小且波动。
- 当前 K/V stage persistent 版本已通过 correctness，full/causal SASS 均为 `REG:168 STACK:0 LOCAL:0` 且无 `LDL/STL`；当前 O-stage N-tile parity 公式版 benchmark `156` 为 full/causal `643.7/1194.6 TFLOPS`。前一版逐 tile 翻转 O-store stage 的 benchmark `155` 为 `642.2/1204.1 TFLOPS`。full 仍低于 benchmark `148` 的 `646.5 TFLOPS` 和 pre-hoist `140` 的 `649.5 TFLOPS`。`q_consumed_mbar + bar.sync 4,96` 实验 correctness 可过，但 benchmark `146` 回退到 `621.6/1155.0 TFLOPS`，不保留。

FP8 paged KV Phase 24/26/28 稳定基线：

- SM120 FP8 paged KV 当前支持范围：`quant_mode=2`、headDim32/headDim64/headDim128/headDim256、`page_size=64`、forward；full、causal、context+causal、target+causal、local、arbitrary paged mirror 已通过；Phase 28 后 RAB/DRAB 的 full、pure causal、context+causal、target+causal、local、arbitrary paged mirror 也已通过；Phase 30 后同一支持面扩展到 D=32/D=64；BF16 paged KV 在 SM120 当前路径仍未接入。
- paged full 使用 full persistent scheduler，host grid 使用 `min(total_tiles, SM_count)`；paged causal/target 使用 paired persistent scheduler，host grid 使用 `min(total_tile_pairs, SM_count)`。
- paged history K/V 已有 page-cache TMA 版本：host 端为 physical `kv_cache` 建 `[page_size, d, h_k, total_pages]` TMA descriptor，load warp 读取 runtime `page_id` 后把它作为 TMA 坐标直接搬 `[64,128]` tile 到 WS SW128 SMEM。
- 当前保留性能启发式：`n_block_paged >= 16` 时 history K/V/SFB/SFV 走 paged TMA；更小 history 仍走 warp 内 `cp.async.cg.shared.global` 16B copy，避免小 seq 被 TMA 固定开销拖慢。paged target tail 也仍走 guarded cp.async copy，因为 target 起点可能不按 page/tile 对齐。
- history SFB/SFV 已复用现有 scale-factor TMA descriptor，使用 `page_id * page_size / kBlockN` 作为 scale tile id，并与 K/V 使用同一个 ready mbarrier。
- profile `176/177/178` 显示 `BS=4,SEQ=2048,H=16,D=128 causal` paged duration `467.1us -> 350.4us -> 200.2us`，同轮 non-paged FP8 为 `169.6us`；grid `1024 -> 188`，local spilling requests `261632 -> 30928 -> 0`。
- kernel benchmark `176` 显示 36 个 paged causal case 相比旧 benchmark `175` 平均 latency 提升约 `62%`，`seq>=1024` 平均提升约 `110%`；paged 相对 non-paged FP8 差距从约 `-58.8%` 收敛到约 `-14.2%`。
- kernel benchmark `181` 的 paged TMA + SF TMA 版相对 `176` cp.async paged：`seq>=1024` 平均 paged TFLOPS 提升约 `+4.42%`，`seq>=4096` 提升约 `+6.27%`；paged 相对 non-paged FP8 的 `seq>=4096` 平均差距约 `-8.52%`。
- full paged KV 已接入同一 paged TMA history 路径：correctness 日志为 `1test_results/447_phase24_paged_full_guard_sweep.log` 和 `448_phase24_paged_full_guard_examples.log`，后者为 `31/31 passed`。公平误差对比应使用 dequantized FP8/e8m0 reference；`1test_results/449_phase24_paged_fair_ref_sweep.log` 显示 paged causal/full `cos=0.9996~0.9997`，`450_phase24_paged_fair_ref_edges.log` 显示 partial page/target/full edge `cos>=0.99962`。same-input 对照 `451_phase24_paged_same_input_sweep.log` 用同一份 raw Q/K/V 和同一 FP8 block scale 构造 non-paged 与 paged，full/causal 的 `max_err=0`。benchmark `182` 显示 full paged 相对 non-paged FP8：全部 36 个 full case 平均 `-3.0%`，`seq>=1024` 平均 `-0.1%`，`seq>=4096` 平均 `+1.5%`；causal paged 仍平均约 `-8.2%`。
- `run_hstu8_examples.sh` 已把原 14 个 non-paged example 逐个镜像到 paged KV，并覆盖 D=32/D=64/D=128/D=256。Phase 27 后额外覆盖 non-paged full/local/context/target/arbitrary + RAB/DRAB；Phase 28 后这些 RAB/DRAB extra cases 也逐个镜像到 paged KV。Phase 29 后 partial-last-page/target tail 也按 K physical offset 通过。Phase 33 后新增 D128/D256、h=4 的 `rab_h1/drab_h1` non-paged 与 paged mirror；最新日志 `1test_results/630_hstu8_examples_rab_h1_scheduler.log` 为 `300/300 passed`。`sweep_accuracy.py` 最新 head-shared scheduler 验证日志 `1test_results/629_sweep_rab_h1_scheduler.log` 通过。
- paged causal TMA SASS：`4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_paged_causal_tma_sf_tma.sass` 含 `UTMALDG.4D`，资源为 `REG:168 STACK:0 LOCAL:0`，无 `LDL/STL` 命中；`UTMALDG` 计数为 `10`，包含 paged K/V 和 SFB/SFV TMA。
- 剩余差距主要来自 page-id/descriptor 额外控制流、paged target tail guarded copy，以及小 seq 下 page TMA descriptor 固定开销。

FP8 headDim256 Phase 25 当前状态：

- SM120 编译、dispatch、runtime guard 已支持 hdim256；Phase 25 构建应使用 `HSTU_DISABLE_HDIM256=FALSE`。
- FP8 WS hdim256 使用 `{kBlockM=128,kBlockN=64,kHeadDim=256,kNWarps=8}`；Q/K e8m0 scale 按每 token 一个 int32 打包连续 128-D chunk，hdim128 scale layout 保持兼容。
- hdim256 GEMM1 分两个 128-D chunk 执行并累加到同一 `acc_s`；K stage 的 `k_empty` release 必须在两个 chunk 的 K 都从 SMEM 消费完成之后，不能沿用 hdim128 的“load K 后立即 release”位置。
- hdim256 paged KV history K/V/SFB/SFV 走 page-cache TMA；aligned full-block target tail 可走 contiguous TMA，非满或不对齐 target tail 仍用 guarded copy fallback。
- hdim256 arbitrary 为了满足 SM120 dynamic SMEM opt-in limit，使用 single K/V stage；stage1 指针 alias stage0，load/math stage 不翻转。
- `run_hstu8_examples.sh` 当前覆盖 D=128/D=256 的 14 个 non-paged example、paged mirror、paged edge 和 paged full。最新 spill-fix 后日志 `1test_results/553_phase25_hdim256_spill_final_examples.log` 为 `62/62 passed`。
- `sweep_accuracy.py` 当前覆盖 D=128/D=256 主表，并追加 same-input paged vs non-paged 对照；最新 spill-fix 后日志 `1test_results/553_phase25_hdim256_spill_final_sweep.log` 通过，D=256 full/causal same-input `max_err=0`。
- BF16 hdim256 定向 full/causal correctness 日志 `1test_results/542_phase25_bf16_hdim256_directed.log` 通过；`hstu_test.py` 的 SM120 guard 已允许 `attn_dim=256`。
- hdim256 SASS 代表实例：`I256_full`、`I256_causal`、`I256_paged_full`、`I256_paged_causal` 均为 `REG:168 STACK:0 LOCAL:0`，且 `LDL=0/STL=0`。pure non-paged causal 的修复方式是 hdim256 不再走 paired persistent scheduler，改用 runtime mask 单体 N-loop，并关闭 hdim256 in-mainloop O-store；hdim128 causal 和 hdim256 paged causal 仍保留 persistent。

FP8 paged RAB/DRAB Phase 26/28 当前状态：

- paged+RAB/DRAB 走 FP8 WS TMA kernel；Phase 28 后 D=128/D=256 full、pure causal、context+causal、target+causal、local、arbitrary + paged KV + RAB/DRAB 都已支持。
- hdim128 RAB 必须使用 `kBlockN=64`，不能沿用旧 non-paged hdim128 RAB fallback 的 `kBlockN=128`；Python block-scale wrapper 对 `rab is not None && dim == 128` 返回 BN64，paged KV 下也继续以 `kv_cache.shape[2]` 作为 BN。
- RAB/DRAB bias 在 GEMM1 后、mask/activation 前加到 `acc_s`。Phase 33 后 aligned RAB tile 由 K load warp 发起 TMA 到单-stage RAB SMEM，再由 math warp 从 SMEM 加；不适合 TMA 的 paged target tail 等 unaligned case 仍 fallback 到 direct-global RAB add。
- Phase 33 head-shared RAB：当 `params.h_rab == 1 && params.h > 1` 时，kernel 仍按 `bidh_rab=0` 读取共享 RAB，但 scheduler 把同一 `(b,m)` 的不同 head 相邻执行以增强 L2 reuse。该优化只改善 RAB 数据复用，不减少每个 head 的 RAB TMA/SMEM read/add，因此不能期望接近 no-RAB。
- Scheduler 口径：paged full RAB/DRAB 走 full persistent；D=128 paged pure causal/target+causal RAB/DRAB 走 paired persistent；D=256 paged RAB/DRAB 以及 context/local/arbitrary 走 3D grid。
- host guard 当前允许 no-target paged full/local/arbitrary/context window；带 `num_targets` 的 paged KV 仍要求 `window_size_left < 0 && window_size_right == 0`，即 target/local-window 组合仍不属于当前支持范围。
- SASS：`I128_paged_causal_rab` 为 `REG:168 STACK:0 LOCAL:0`，无 `LDL/STL`；`I256_paged_causal_rab` 为 `REG:168 STACK:8 LOCAL:0`，有 1 个 `STL` 和 1 个 `LDL`。尝试将 hdim256 RAB-add loop 改为 `#pragma unroll 1` 会恶化到 `STACK:128`，不保留。

FP8 irregular seqlen Phase 29 当前状态：

- SM120 FP8 block-scale 主路径是 `quant_mode=2`、headDim32/headDim64/headDim128/headDim256、forward。Phase 29 已补齐 `hstu_test.py` 中的 irregular length fixed matrix，当前显式覆盖 `seq=99/128/256`。
- 实现口径采用 Hopper-style actual/padded seqlen：Python wrapper 对 SM120 FP8 `quant_mode=2` 在需要时 padding physical Q/K/V、scale factor table 和 arbitrary `func`；传给 kernel 的 `cu_seqlens_q/k` 是 padded offset，`seqused_q/k` 是 actual total length。aligned case 继续走原 fast path。
- Q/SFA padding 对齐 `kBlockM=128`；non-paged K/V/SFB/SFV padding 对齐实际 `kBlockN`，当前 FP8 hdim128/256 RAB 主路径为 64。输出从 SM120 kernel 返回后按 actual `seqused_q` unpad 回原 API 期望的 compact `[total_q, H, D]`。
- paged KV target tail 不能用 Q offset 推导 K/V physical start。actual K 可能包含已经在 page cache 中的 previous history，而 contiguous K/V 只包含 new history + target；wrapper 必须在 K/V physical layout 中插入 last-page gap，让 target tail 从 `kBlockN` 边界开始，kernel target load 使用 `sum_s_k + actual_seqlen_k - actual_seqlen_t + last_page_offset + target_block*kBlockN`。
- RAB/DRAB 坐标仍按 logical K col 取 bias；paged target block 需要减去 `last_page_offset`，history page padding col 需要跳过。
- `hstu_test.py::HSTU8Test::test_sm120_fp8_blockscale_matrix` 覆盖 D=32/D=64/D=128/D=256、seq=99/128/256、full/causal/local/context/target/arbitrary、none/RAB/DRAB、non-paged/paged。最新日志 `1test_results/phase30_hdim32_64_hstu8_matrix.log` 为 `1 passed`。
- 最新全文件 pytest 日志 `1test_results/592_phase29_hstu_test_full_after_k_offsets.log` 为 `3 passed, 1 skipped`。`run_hstu8_examples.sh` 最新日志 `1test_results/592_phase29_examples_after_k_offsets.log` 为 `142/142 passed`；`sweep_accuracy.py` 最新日志 `1test_results/592_phase29_sweep.log` 通过；全量 benchmark 日志为 `2benchmark_results/592_282d811e_phase29_irregular_paged_hstu_test_full_benchmark.log`。benchmark 中 BF16 hdim256+RAB/DRAB/arbitrary baseline 可出现 unsupported/invalid-argument 行，不作为 SM120 FP8 correctness 失败。

FP8 non-paged RAB/DRAB Phase 27 当前状态：

- D=128/D=256 non-paged RAB/DRAB 已切到 FP8 WS TMA，覆盖 full、pure causal、context+causal、target+causal、local、arbitrary，使用 `{kBlockM=128,kBlockN=64,kNWarps=8}`。Phase 33 后 aligned RAB 主路径使用 K-warp RAB TMA/SMEM；direct-global RAB add 只作为 unaligned fallback。
- `h_rab==1` 是当前支持的 head-shared RAB 快路径。benchmark 使用 `bench_hstu_attn_sm120.py --rab-heads shared` 生成 `heads_rab=1` 输入；默认 `--rab-heads per-head` 仍代表每个 head 一份 RAB。
- D=128 non-paged RAB/DRAB 的 V block-scale BN 已从旧 fallback 的 128 改为 64；Python wrapper、C++ tile-size 和 WS specialization 必须保持一致。
- Scheduler 口径：full RAB 对 D=128/D=256 使用 full persistent；pure causal RAB 只有 D=128 使用 paired persistent；D=256 pure causal RAB 和 context/target/local/arbitrary RAB 使用 3D grid。
- D=256 arbitrary + non-paged RAB/DRAB 也切到 WS；同频 kernel-only 对照中 WS 与 fallback 基本持平，因此这是路径统一改动，不是性能优化改动。
- Resource usage：D=128 non-paged RAB WS 代表实例为 `REG:168 LOCAL:0`，当前扫描到的 non-paged 代表实例 `STACK:0`；D=256 non-paged RAB WS 代表实例为 `REG:168 LOCAL:0`，full `STACK:0`，pure causal `STACK:16`，context `STACK:8`，target `STACK:24`，local `STACK:32`，arbitrary `STACK:16`。资源日志见 `1test_results/590_phase27_d128_d256_all_rab_ws_resource.log` 和 `1test_results/590_phase27_d128_rab_ws_resource.log`。
- Kernel-only 对照日志：WS `2benchmark_results/586_phase27_d256_nonarbitrary_rab_ws_kernel_only.log`，fallback `2benchmark_results/587_phase27_d256_rab_fallback_kernel_only.log`。`bs=2,seq=1024,h=4,d=256` latency：full `0.0693ms vs 0.1111ms`，local `0.0583ms vs 0.0825ms`，context `0.1264ms vs 0.2118ms`，target `0.0798ms vs 0.1246ms`。arbitrary 切 WS 是路径统一；此前同形状 fallback 约 `0.0897ms`，WS 无明确性能收益但也未观察到实质损失。

## 核心路径

主要代码目录：

- `fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/hstu_fwd_kernel.h`：前向内核入口，含 BF16、FP8 非 WS 路径和 WS include。
- `fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/hstu_fwd_kernel_fp8_ws.h`：FP8 warp-specialized 内核主体。
- `fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/sm120_qmma_builder.h`：本地 SM120 QMMA builder 子集。
- `fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/kernel_traits.h`：BF16/FP8 kernel traits 与 SMEM 布局。
- `fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/hstu_ops_gpu.cpp`：PyTorch 入口，包含 `hstu_varlen_fwd_120`。
- `fbgemm_gpu/experimental/hstu/benchmark/bench_hstu_attn_sm120.py`：SM120 benchmark。
- `fbgemm_gpu/experimental/hstu/benchmark/ncu_hstu_attn.py`：NCU profile 目标脚本。
- `sweep_accuracy.py`：FP8 vs BF16 准确度扫描。
- `run_profile.sh`：nsys + ncu profile 脚本。
- `dump_sass_fp8_ws.sh`：提取 FP8 WS kernel SASS。

参考实现：

- `6KD_fp8_block_scale/kernels/include/sm120_blockscaled_gemm/` 只读参考，不要直接改。

## 构建

当前环境记录：

- PyTorch 2.10.0，`nvcr.io/nvidia/pytorch:26.01-py3`。
- CUDA 13.1。
- GPU 目标为 RTX PRO 6000 Blackwell SM120。

重编译命令应分开执行，避免把多个无关步骤串成一个 shell 命令：

```bash
mkdir -p /tmp/claude
```

```bash
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu
```

```bash
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user HSTU_ARCH_LIST="12.0" HSTU_DISABLE_BACKWARD=TRUE HSTU_DISABLE_DETERMINISTIC=FALSE HSTU_DISABLE_HDIM32=FALSE HSTU_DISABLE_HDIM64=FALSE HSTU_DISABLE_HDIM256=FALSE MAX_JOBS=32 pip install --no-build-isolation --config-settings editable_mode=compat -e .
```

Phase 30 headDim32/64/256 开发应使用 `HSTU_DISABLE_HDIM32=FALSE HSTU_DISABLE_HDIM64=FALSE HSTU_DISABLE_HDIM256=FALSE`，并在测试入口显式覆盖 D=32/64/128/256。

编译失败时先读上下文和完整错误，再改代码。需要截取错误时可使用：

```bash
pip install ... 2>&1 | grep -E "error:|note:|static_assert|undefined" | head -60
```

## 验证

日志命名规范：

- 准确性日志：`1test_results/NNN_<修改内容简述>.log`
- benchmark 日志：`2benchmark_results/NNN_<commit_short>_gpu<MHz>MHz_<描述>.log`
- profile 产物：`3profile_results/NNN_<描述>.*`
- SASS：`4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full.sass`

数值正确性：

```bash
HSTU_SWEEP_FP8_QUANT_MODE=2 python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/sweep_accuracy.py 2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/1test_results/NNN_xxx.log
```

判定标准：各配置 `fp8_gt_cos >= 0.995`，当前优良基线通常 `>= 0.9996`；BF16 路径应关注 `bf16_gt_cos` 和 `bf16_gt_max`，不要只看 `bf16_fp8_cos`。

所有 HSTU CUDA/CuTe 正确性验证都必须包含 repo 自带 pytest：

```bash
PYTHONPATH=/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu python -m pytest -q /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu/test/hstu_test.py 2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/1test_results/NNN_hstu_test.log
```

当前 SM120 口径下 `HSTUPagedKVTest` 是 BF16 paged KV 测试，class decorator 会跳过；FP8 paged KV correctness 已纳入 `HSTU8Test::test_sm120_fp8_blockscale_matrix`，并继续由 `run_hstu8_examples.sh`、`sweep_accuracy.py` 和 paged same-input/edge case 覆盖。最新全文件 pytest 日志为 `1test_results/phase30_hdim32_64_hstu_test_full.log`，结果 `3 passed, 1 skipped`。固定 SM120 FP8 `quant_mode=2` matrix 覆盖 `D=32/64/128/256 × seq=99/128/256 × full/causal/local/context/target/arbitrary × none/rab/drab × non-paged/paged`，审计日志见 `1test_results/phase30_hdim32_64_hstu8_matrix.log`。

BF16 正确性应优先使用 `hstu_test.py` 的 `HSTU16Test`，不要用 `run_hstu8_examples.sh` 作为 BF16 结论。当前 Phase 30 构建已启用 hdim32/64/256 编译，但 SM120 BF16 runtime 支持面仍是 hdim64/128/256；BF16 hdim32 不作为基线，BF16 hdim256 只作为 full/causal 基线，不应把 hdim256+RAB 或 arbitrary 当作 HSTU16 默认支持面：

```bash
PYTHONPATH=/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu/test/hstu_test.py 2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/1test_results/NNN_bf16_hstu16_single.log
```

完整 Hypothesis BF16 测试入口如下；如果当前构建禁用了 hdim64，先不要直接跑完整入口，除非同步调整测试参数或重新构建启用 hdim64：

```bash
PYTHONPATH=/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu python -m pytest -q /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu/test/hstu_test.py::HSTU16Test::test_hstu_attn 2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/1test_results/NNN_bf16_hstu16_hypothesis.log
```

BF16 WS/TMA 实验路径已验证 correctness 但性能低于默认 cp.async，不作为默认性能路径；当前 BF16 性能优化应优先针对默认 cp.async `kBlockN=128` 路径。

example cases：

```bash
bash /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/run_hstu8_examples.sh
```

该脚本覆盖原 14 个 HSTU8/FP8 block-scale non-paged 示例，并追加 paged KV mirror/full/edge。Phase 30 后同一套语义同时覆盖 D=32/D=64/D=128/D=256；额外覆盖 non-paged full/local/context/target/arbitrary + RAB/DRAB，并逐个镜像到 paged KV，local paged 应真实运行通过，不再标为 `PASS-UNSUPPORTED`。Phase 33 后额外覆盖 D128/D256、h=4 的 head-shared `rab_h1/drab_h1` non-paged 和 paged mirror。当前判定标准为全脚本通过，最新日志 `1test_results/630_hstu8_examples_rab_h1_scheduler.log` 为 `300/300 passed`。该脚本不替代 BF16 的 `HSTU16Test`。

benchmark：

benchmark 和性能对比必须先锁频，默认使用 `sudo nvidia-smi -lgc 2407`。如果当前环境无法锁频，日志文件名和结论必须明确标注 `unlocked`，且不能作为严格性能回归/提升结论。

```bash
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu/benchmark/bench_hstu_attn_sm120.py 2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/2benchmark_results/NNN_xxx.log
```

`bench_hstu_attn_sm120.py` 默认覆盖 `full/causal/local/context/target/arbitrary` × `none/rab/drab`，并对每个逻辑 case 输出 BF16、non-paged FP8、paged FP8 三列。可用 `--mask-configs`、`--bias-configs` 和 `--columns bf16 fp8 paged` 做子集筛选；`--rab-heads shared` 用于 head-shared RAB (`h_rab=1`) 同频对比，默认 `per-head`；旧 `--full-only` / `--causal-only` 仍保留为 alias。BF16、FP8、paged 三列独立计时，BF16 unsupported 不应阻塞 FP8/paged 结果。TFLOPS 按实际 valid attention pairs 计算，不再按 full 矩阵统一估算。

profile：

```bash
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu
```

```bash
./run_profile.sh NNN <描述>
```

SASS dump：

```bash
cd /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu
```

```bash
./dump_sass_fp8_ws.sh
```

SASS 中重点搜索 `LDL`/`STL` 以定位 register spill。

Phase 23 FP8 WS SASS 参考保留在 `PLANS.md` 的 Phase 23 验证记录中。当前仍以 `REG:168 STACK:0 LOCAL:0`、无 `LDL/STL` 作为 FP8 WS 稳定基线；任何新 specialization，尤其是 hdim32/64/256 和 RAB/DRAB，都必须重新 dump SASS 检查 spill。Phase 31 后 D32 full/paged full、D64 full/paged full 代表实例均无 `LDL/STL`；D64 paged full 最终 resource 为 `REG:168 STACK:0 LOCAL:0`。当前 hdim256 full、causal、paged full、paged causal 代表实例已达到无 spill；hdim256 paged causal+RAB 仍有 `STACK:8` 和 1 对 `STL/LDL`，hdim256 non-paged causal+RAB 仍有 `STACK:16` 和 12 条 `LDL/STL`，需要作为残余风险看待。

## 性能分析重点

- `launch__registers_per_thread`
- `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum`
- `l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum`
- No Eligible stall
- Long Scoreboard stall
- Math Throttle
- SM 利用率
- SMEM bank conflict
- Block Limit SMEM；当前 FP8 WS 路径受约 85KB SMEM/CTA 限制，通常为 1 CTA/SM。
- FP8 WS pure full 和 pure causal 当前都是 K/V stage persistent 的 mbar producer/consumer four-load-warp persistent kernel；若回到 hdim128 FP8 WS 性能优化，应跑 NCU 确认 No Eligible、Long Scoreboard、mbarrier wait、O-store 与下一 tile Q/SFA/K/V TMA overlap、TMA pipe 竞争，以及 tail-wave 的占比。
- 旧 static launch pair 仍可作为同频对照：当前 split persistent causal 已超过旧 static pair 记录，full 略低于旧 static pair 最好记录。

跑 benchmark、profile 或任何性能数据前必须锁频：

```bash
sudo nvidia-smi -lgc 2407
```

跑完后解锁：

```bash
sudo nvidia-smi -rgc
```

## 数值错误调试规程

如果代码修改后数值错误且静态阅读无法快速定位，不要继续盲目读代码，按以下顺序缩小范围：

1. `gemm1_only + all_ones`
   - 设置 `params.debug_gemm1_only=true`。
   - Q/K/SFA/SFB 全设为 1，e8m0 为 `0x3F`。
   - 预期应按当前 dtype 的 `kBlockN` 构造 reference：Phase 22 默认 BF16 no-RAB headDim128 已切到 `kBlockN=128`；FP8 no-RAB headDim128 通常也是 `kBlockN=128`。如果后续重新实验 BF16 `kBlockN=64`，必须同步更新 debug reference。
   - 失败则在 GEMM1 前后观察 Q/K/SFA/SFB fragment。

2. `gemm1_only` 真实数据
   - 使用真实 Q/K/SFA/SFB。
   - 对比 GEMM1 输出与 BF16 参考的 cosine similarity。
   - 若 `< 0.995`，问题在 GEMM1 路径；若 `>= 0.995`，问题转向 GEMM2、V/SFV 或 P staging。

3. 全链路 `all_ones`
   - Q/K/V/SFA/SFB/SFV 全设为 1。
   - 预期输出为 `actual_valid_k * kHeadDim * alpha * silu(alpha * kHeadDim)`；例如无 mask、seq=128、headDim=128、alpha=1 时为 `128 * 128 = 16384`。

插入 printf 时使用 guard，避免刷屏：

```cpp
if ((tidx_math & 31) == 0 && tidx_math < 64) {
  printf(...);
}
```

重点观察：

- 输入 fragment：`tXrQ(0)`、`tXrK(0)` 等。
- scale factor：`tCrSFA(0,0,0)`、`tCrSFB(0,nr,0)`、`tCrSFV(...)`。
- 输出 accumulator：`acc_s(0)`、`acc_o(0)`。

Illegal memory access 时使用：

```bash
HSTU_SWEEP_FP8_QUANT_MODE=2 compute-sanitizer --tool memcheck python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/sweep_accuracy.py 2>&1 | tee /tmp/claude/sanitizer_out.log
```
