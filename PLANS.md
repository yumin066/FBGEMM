# HSTU SM120 FP8 当前计划

来源：`CLAUDE.md`、`PLAN.md`、`memory.md`。仓库中没有找到小写 `plan.md`，实际存在的是 `PLAN.md`。`PLAN.md` 最后更新到 Phase 19 候选方向，`CLAUDE.md` 中包含更新的 Phase 20/21 状态，因此本文件以 `CLAUDE.md` 的 Phase 21 记录为最新状态。

## 当前状态

项目当前位于 SM120 FP8 WS kernel 优化后续阶段。Phase 21 已完成并验证，最新基线为 benchmark 056。Phase 22 已完成第一轮 BF16 迁移评估：当前提交只保留默认 BF16 路径的 `kBlockN=128` 性能改动；WS/TMA 实验因性能回退不进入待提交 diff。Phase 23 当前实现为 FP8 WS pure full 和 pure causal split persistent kernel；其他 mask 仍保持原始 3D grid。
下一阶段目标定为 Phase 24：支持 SM120 FP8 paged KV cache。

最新有效基线：

- Phase 21：Opt A + Opt C，完成于 2026-04-24。
- Phase 23：FP8 WS pure full 使用 branch-local grid-stride persistent loop，launch CTA 数为 `min(total_tiles, SM_count)`；pure causal 使用 branch-local paired persistent loop，launch CTA 数为 `min(total_tile_pairs, SM_count)`；其他 mask 保持原始 3D grid。
- Phase 23 split persistent 把 TMA/load persistent loop、math/softmax persistent loop 和 epilogue/O-store 所属路径拆开；O TMA store 由 active load warp 负责，math warps 通过 O ready/empty mbarrier handoff。
- 当前工作树在 split persistent 基础上已把 Q/K/V/O producer/consumer mbarrier 初始化和 CTA S1 提到 persistent loop 外，跨 tile 维护 mbarrier parity 和 K/V stage；`S_arb` 仍只属于 arbitrary/non-persistent 路径。
- 当前 S1-hoist 版本 bs=8 seq=4096 h=16 causal：FP8 1160.4 TFLOPS，BF16 653.3 TFLOPS，FP8 比 BF16 快 77.6%。
- 当前 S1-hoist 版本 bs=8 seq=4096 h=16 full：FP8 643.6 TFLOPS，BF16 375.9 TFLOPS，FP8 比 BF16 快 71.2%。
- 当前 K/V stage persistent four-load-warp O-stage N-tile parity 公式版 bs=8 seq=4096 h=16 causal：FP8 1194.6 TFLOPS，BF16 677.8 TFLOPS，FP8 比 BF16 快 76.2%。
- 当前 K/V stage persistent four-load-warp O-stage N-tile parity 公式版 bs=8 seq=4096 h=16 full：FP8 643.7 TFLOPS，BF16 369.2 TFLOPS，FP8 比 BF16 快 74.4%。
- 2407MHz locked full benchmark `157` 中，当前 persistent bs=8 seq=4096 h=16 full/causal 为 `627.8/1168.8 TFLOPS`；同环境临时关闭 persistent scheduler 的 current-code non-persistent benchmark `158` 为 `4.8/9.0 TFLOPS`，说明当前 four-load-warp/mbar 实现强依赖 persistent scheduler。历史 phase21 non-persistent 参考 `056` 为 `640.0/1131.8 TFLOPS`，不是同源码对照。
- Pre-hoist split persistent 最好记录仍是 benchmark `140`：full/causal `649.5/1162.5 TFLOPS`。S1-hoist correctness 通过且无 spill，但未形成性能提升。
- SASS：当前 K/V stage persistent four-load-warp full/causal 目标 kernel 均为 `REG:168 STACK:0 LOCAL:0`，无 `LDL/STL`。
- 准确性：`sweep_accuracy.py` 全部 `fp8_gt_cos >= 0.9996`。
- examples：`run_hstu8_examples.sh` 为 14/14 PASS。

## Phase 24：SM120 FP8 paged KV cache

目标：让 SM120 FP8 forward 支持 paged KV cache，并先在 `quant_mode=2` block-scale FP8 路径上打通 correctness，再评估性能。

当前结论：

- 已实现第一版 SM120 FP8 paged KV forward correctness 路径。
- `hstu_varlen_fwd_120` schema 已接入 `kv_cache/page_offsets/page_ids/last_page_lens` optional 参数。
- `set_params_fprop_sm120` 已设置 `kv_cache` stride、page metadata 和 `is_paged_kv`。
- SM120 FP8 traits 已支持 `Paged_KV` template bool；BF16 traits 仍固定 `Paged_KV=false`。
- Python wrapper 已能在 SM120 `quant_mode=2` 下量化 paged `kv_cache`，并传入 combined cache+contiguous scale table。
- 当前实现覆盖 no-RAB、full 和 causal/target、headDim128、`page_size=64`、forward。
- 定向验证已通过：`B=1,H=1` paged KV cos `0.99875`；`B=2,H=4` paged KV cos `0.99876`；partial last page cos `0.99881`；target length 0 cos `0.99886`；非 paged `sweep_accuracy.py` 仍全部 `fp8_gt_cos >= 0.9996`。
- profile 已支持 paged KV：`ncu_hstu_attn.py --target paged`，`run_profile.sh NNN desc paged` 或 `fp8,paged`。
- paged causal/target 已接入 paired persistent scheduler；host grid 从 per-tile `(num_m_block,h,b)` 改为 `min(total_tile_pairs, SM_count)`。
- paged K/V tile copy 已从同步 `LDG+STS` 改为 warp 内 `cp.async.cg.shared.global` 16B copy，再用 existing ready mbarrier 通知 math warp。
- paged history K/V 已实现 page-cache TMA 版本：`kv_cache` 按 `[page_size, d, h_k, total_pages]` 建 TMA descriptor，load warp 使用 runtime `page_id` 作为 TMA 坐标直接搬 `[64,128]` tile。
- 当前保留 `n_block_paged >= 16` 的 TMA 启发式；较小 history 和 paged target tail 继续用 guarded `cp.async` copy。history SFB/SFV 也使用现有 scale-factor TMA descriptor，scale tile id 为 `page_id * page_size / kBlockN`。
- profile 结论：`BS=4,SEQ=2048,H=16,D=128 causal` 中 paged duration 从 `467.1us` 降到 persistent-only `350.4us`，再降到 persistent+cp.async `200.2us`；同轮 non-paged FP8 为 `169.6us`。
- profile 指标：paged grid `1024 -> 188`，waves/SM `5.45 -> 1`，local spilling requests `261632 -> 30928 -> 0`，No Eligible `77.6% -> 73.6% -> 61.1%`。
- 全量 kernel benchmark `176`：36 个 paged causal case 平均 latency 相比旧 benchmark `175` 提升约 `62%`；`seq>=1024` 平均提升约 `110%`；paged 相对 non-paged FP8 的差距从约 `-58.8%` 收敛到约 `-14.2%`。
- 全量 kernel benchmark `181`：paged TMA + SF TMA 版相对 `176` cp.async paged，`seq>=1024` 平均 paged TFLOPS `+4.42%`，`seq>=4096` 平均 `+6.27%`；`seq>=4096` paged 相对 non-paged FP8 平均差距约 `-8.52%`。
- full paged KV 已接入：`hstu_varlen_fwd_120` 允许 `num_targets=None` 且 `window_size=(-1,-1)` 的 paged KV full path，device 端复用同一 paged TMA history K/V/SFB/SFV load 逻辑。correctness 日志：build `1test_results/446_phase24_paged_full_guard_build.log`；sweep `1test_results/447_phase24_paged_full_guard_sweep.log`；examples `1test_results/448_phase24_paged_full_guard_examples.log`，其中 examples 为 `31/31 passed`。
- paged KV 公平误差对比已改为 dequantized FP8/e8m0 reference：先手动量化 Q/K/V/kv_cache，再用同一批 FP8 tensor 和 e8m0 scale 重建 Python reference。`1test_results/449_phase24_paged_fair_ref_sweep.log` 中 paged causal/full `cos=0.9996~0.9997`；`1test_results/450_phase24_paged_fair_ref_edges.log` 中 partial page/target/full edge `cos>=0.99962`。
- paged vs non-paged same-input 对照已加入 `sweep_accuracy.py`：同一份 raw Q/K/V 生成 contiguous K/V 和 paged cache，两边都用 `block_size=64` 量化以保证 V scale 粒度一致。`1test_results/451_phase24_paged_same_input_sweep.log` 显示 full/causal、H=1/4、SEQ=128/256/512 全部 `max_err=0`，说明 aligned case 下 paged 与 non-paged 输出 bitwise 一致。
- examples 已按原 14 个 non-paged HSTU8 example 逐个生成 paged mirror：causal、context+causal、target+causal、arbitrary 真实运行并通过；RAB/DRAB/local 仍属于当前 SM120 FP8 paged guard 的不支持范围，脚本标记为 `PASS-UNSUPPORTED` 并验证会触发明确错误。最新日志 `1test_results/456_phase24_paged_mirror_examples_final.log` 为 `31/31 passed`。`sweep_accuracy.py` 的 `paged_f_*` 列覆盖每个 non-paged full sweep 行，最新日志 `1test_results/455_phase24_paged_mirror_sweep.log` 通过，same-input full/causal 继续 `max_err=0`。
- benchmark `182`：full paged 相对 non-paged FP8，全部 36 个 full case 平均 `-3.0%`，`seq>=1024` 平均 `-0.1%`，`seq>=4096` 平均 `+1.5%`；causal paged 平均约 `-8.2%`，`seq>=4096` 平均约 `-8.3%`。
- SASS：paged causal TMA 实例含 `UTMALDG.4D`，当前 SF TMA 版资源 `REG:168 STACK:0 LOCAL:0`，无 `LDL/STL` 命中；`UTMALDG` 计数为 `10`。

第一阶段范围：

- forward only。
- SM120 FP8 `quant_mode=2`。
- no-RAB、full 和 causal/target paged KV 优先。
- headDim128 优先。
- 初始只支持 `page_size == kBlockN`，优先匹配当前 BN64 路径；不先支持跨 page 的单个 N tile。
- 先不做 backward、RAB、arbitrary mask、local window 的完整支持。

实施计划：

1. API 和参数接线
   - 给 `hstu_varlen_fwd_120` 增加 `kv_cache/page_offsets/page_ids/last_page_lens` optional 参数。
   - Python SM120 分支传递 paged KV 参数。
   - 在 `set_params_fprop_sm120` 中设置 `kv_cache_ptr`、stride、`page_size`、`total_pages`、page metadata 和 `is_paged_kv`。
   - 加 runtime guard：只有当前支持的 FP8 paged KV shape 进入新路径，不支持的组合直接 `TORCH_CHECK` 报错。

2. Kernel traits 和 dispatch
   - 给 SM120 FP8 traits 加 `Paged_KV` template bool，不再固定为 false。
   - dispatch 增加 paged KV specialization。
   - 初始限制 `Page_Size == kBlockN`，避免 TMA tile 跨物理 page gather。
   - 保持非 paged 路径生成代码不变，避免影响当前性能基线。

3. Paged K/V load
   - 参考 Ampere `Paged_KV` 逻辑：`n_block < n_block_paged` 走 paged cache，后续 target/new tokens 走普通 contiguous K/V。
   - 当前已为 physical paged cache 建 TMA descriptor：形状 `[page_size, d, h_k, total_pages]`，`page_id` 是 TMA 坐标而不是手工 gather base list。
   - history page block 数达到 `16` 时，paged K/V 直接 TMA 到原 WS SW128 SMEM layout。
   - history SFB/SFV 也用现有 scale-factor TMA descriptor 搬运，scale tile id 为 `page_id * page_size / kBlockN`，并与 K/V 使用同一个 ready mbarrier。
   - history 较短时保留 `cp.async.cg.shared.global` 16B copy，避免小 seq 被 page TMA descriptor/控制流固定开销拖慢。
   - K load 用 `page_ids[page_offsets[bidb] + n_block]` 映射到物理 page；V load 使用同一个 page id，K/V tensor 维度分别取 0/1。
   - contiguous target K/V 也在 paged path 中由 load warp 手动 copy，避免 target 起点不按 `kBlockN` 对齐时 TMA tile index 不清。
   - 保持 Q/O 路径不变。

4. FP8 scale-factor 支持
   - 明确 paged K/V cache 的 scale-factor layout。
   - K cache scale 需要按 physical page id 可寻址；target/new K 仍按 contiguous token 可寻址。
   - V cache scale 需要与 physical page/block 对齐；target/new V 仍沿 N block 对齐。
   - 第一版使用 combined scale table：`[physical cache pages, contiguous q/k/v tokens]`，paged tile 用 `page_id * page_size` 索引，target tile 用 `total_pages * page_size + target_token_start` 索引。
   - Python quantization 已支持 paged `kv_cache` 的 FP8 block-scale 打包。

5. Mask 和 block info
   - 在 SM120 `HstuBlockInfo` 中启用 `page_offsets/page_ids/last_page_lens`。
   - 对齐 Ampere 的 `n_block_history`、`n_block_target`、`last_page_offset` 语义。
   - 重点保证 full no-target、target rows 对 paged history 的 causal mask 正确。
   - 初始不支持 RAB/arbitrary/local，降低 mask 组合复杂度。

6. Correctness 验证
   - 基于 `HSTUPagedKVTest` 增加 SM120 FP8 paged KV 定向测试。
   - 先覆盖：`page_size=64`、headDim128、full、causal、target length 为 0 和非 0。
   - 对比 BF16 reference 或 FP8 Python reference，要求 cosine >= 0.995，理想 >= 0.999。
   - 小 shape 先跑 compute-sanitizer，重点查 page 边界和 last page。
   - 当前已完成 aligned directed、last page 非满、target length 为 0；还需要补 varlen batch 的正式测试。
   - 当前日志：`1test_results/401_phase24_sm120_fp8_paged_kv_directed.log`、`402_phase24_sm120_fp8_paged_kv_b2h4.log`、`403_phase24_sm120_fp8_paged_kv_partial_last_page.log`、`404_phase24_sm120_fp8_paged_kv_target0.log`、`405_phase24_sm120_fp8_paged_kv_sweep.log`。

7. 性能验证
   - 增加 paged KV benchmark case，记录 kernel-only latency 和 corrected TFLOPS。
   - 与“手动 gather 成 contiguous K/V 后的 SM120 FP8”做同 shape 对照。
   - dump SASS 检查 `LDL/STL`，目标仍保持无 register spill。
   - NCU 重点看 TMA pipe、page-id gather 开销、mbarrier wait、No Eligible 和 L2 hit。
   - 当前日志：profile `176/177/178_phase24_*paged*_profile.*`，benchmark `2benchmark_results/176_bbb84710_gpu2407MHz_phase24_paged_persistent_cpasync_kernel.log`，correctness `1test_results/412_phase24_paged_persistent_cpasync_sweep.log`、`413_phase24_paged_persistent_cpasync_examples.log`。
   - paged TMA + SF TMA 日志：build `1test_results/437_phase24_paged_tma_sf_tma_build.log`；correctness `1test_results/438_phase24_paged_tma_sf_tma_sweep.log`、`439_phase24_paged_tma_sf_tma_examples.log`；benchmark `2benchmark_results/181_bbb84710_gpu2407MHz_phase24_paged_tma_sf_tma_kernel.log`；SASS `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_paged_causal_tma_sf_tma.sass`。
   - full paged 日志：build `1test_results/446_phase24_paged_full_guard_build.log`；correctness `1test_results/447_phase24_paged_full_guard_sweep.log`、`448_phase24_paged_full_guard_examples.log`；fair-reference correctness `1test_results/449_phase24_paged_fair_ref_sweep.log`、`450_phase24_paged_fair_ref_edges.log`；same-input output parity `1test_results/451_phase24_paged_same_input_sweep.log`；paged mirror examples/sweep `1test_results/456_phase24_paged_mirror_examples_final.log`、`455_phase24_paged_mirror_sweep.log`；benchmark `2benchmark_results/182_fcb69d8a_gpu2407MHz_phase24_paged_full_kernel.log`。

风险点：

- TMA 不支持一个 tile 内跨多个非连续 physical page gather；因此必须先限制 `page_size == kBlockN`。在该限制下，paged history K/V 可以把 physical `page_id` 当作 TMA 坐标直接搬运。
- FP8 block-scale 的 K/V cache scale-factor 索引比 BF16 paged KV 更复杂，是本阶段最大 correctness 风险。
- paged target mask 中 `last_page_lens` 和 `last_page_offset` 容易出现 off-by-one。
- full paged 的 Python `quant_mode=2` wrapper 仍要求 N 按 128 对齐；`seq=160` 会在量化阶段失败，尚未覆盖 full partial-last-page。
- 新 specialization 可能增加 register pressure；persistent+cp.async profile 当前显示 local spilling requests 为 0，但每个保留版本仍必须检查 SASS/profile。
- persistent scheduler 当前按 logical tile decode；paged KV 需要保证 load/math 两边 page-id decode 完全一致。
- 当前 full paged 大 seq 已与 non-paged FP8 基本持平；causal paged 大 seq 仍约落后 non-paged FP8 `8%`。主要剩余差距来自 page-id/descriptor 控制流、target tail 手工 copy，以及 page TMA descriptor 固定开销。若要小 seq 和大 seq 都最优，下一步应考虑拆成独立 cp.async-paged 与 TMA-paged dispatch specialization，避免小 seq 也携带 TMA descriptor。

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

## Phase 23：FP8 WS split persistent kernel

目标：按用户要求把 FP8 WS pure full 和 pure causal no-RAB kernel 改为真正 persistent kernel，同时利用 causal tile 工作量从重到轻的结构做首尾配对负载拉平。

当前实现：

- pure full no-RAB WS kernel 使用 branch-local grid-stride persistent loop，launch 为 `grid.x = min(total_tiles, SM_count)`。
- pure causal no-RAB WS kernel 使用 branch-local paired persistent loop，launch 为 `grid.x = min(total_tile_pairs, SM_count)`，其中 `total_tile_pairs=(total_tiles+1)/2`。
- causal 每个 CTA 从 `blockIdx.x` 开始，以 `gridDim.x` 为 stride 遍历 tile-pair；每个 tile-pair 处理 `tile_pair` 和 `total_tiles - 1 - tile_pair`，保持 heavy/light 配对。
- local、context、target、arbitrary 等其他 WS 实例保持原始三维 grid：`dim3(num_m_block, h, b)`。
- `hstu_compute_attn_1rowblock_sm120_fp8_ws` 内部按 WS 分支拆成 load persistent loop 和 math persistent loop；kernel wrapper 只调用一次，不再把整块 monolithic body 包进外层 persistent loop。
- tile id 使用静态本地 decode：load/math 两边运行同一确定性 scheduler，不使用 dynamic work queue、atomic counter 或 shared tile-id state。
- epilogue 中 math warps 只负责写 O 到 SMEM；active load warp 等 O ready mbarrier 后负责 `tma_o` store 并等待完成，随后释放 O empty mbarrier。
- persistent full/causal 路径已把 Q/K/V/O producer/consumer mbarrier 初始化和 CTA S1 hoist 到 static scheduler loop 外；load/math 分支跨 tile 维护各自 parity 和 K/V stage。非 persistent 路径仍保持 per-tile S1。
- four-load-warp 实验版把 load WG2 的 4 个 warp 分为 Q/SFA load、K/SFB load、V/SFV load、O store；Q/K/V/O 使用 ready/empty mbarrier。SMEM 不新增 Q/O double buffer，O 复用当前 tile 最后被 math 消费的 K/V stage；math epilogue 直接使用 `math_stage ^ 1`，O-store warp 使用同一 tile 参数和当前起始 stage 推导该 stage。
- 当前 full 和 causal persistent 实例已消除 wrapper-loop register spill。当前 mbar producer/consumer four-load-warp full/causal SASS 均无 `LDL/STL`。

已拒绝的中间实现：

- 纯 static grid-stride persistent correctness 可过，但 causal 性能从同环境基线 `1105.1 TFLOPS` 降到 `1037.8 TFLOPS`，主要原因是 tile 分配相位固定导致 causal 重 tile 分布不均。
- dynamic persistent work queue correctness 可过，但收益小且 tile 间必须做 CTA sync/atomic，同环境 100 iters full/causal 约 `644.0/1111.7 TFLOPS`，200 iters repeat causal 回落到 `1098.1 TFLOPS`。
- 在 wrapper 中新增普通或 padded static `__shared__` 保存 next tile 会破坏 WS dynamic SMEM 绝对对齐，触发 `misaligned address`。
- noinline callee 隔离 wrapper loop 和 WS body 的尝试已放弃：causal FP8 WS 实例的 `ptxas` 超过 7 分钟未完成，编译成本不可接受。
- 无 persistent loop 的静态首尾配对 launch queue 更稳：`grid.x=total_tiles`，`blockIdx.x` 映射为 `0,total_tiles-1,1,total_tiles-2,...`，SASS 无 spill，benchmark 最好记录 full/causal 为 `654.1/1135.5 TFLOPS`。但当前代码按用户要求保留真正 persistent 版本。

验证记录：

- `1test_results/320_phase23_fp8_ws_paired_persistent_accuracy.log`：FP8 sweep 通过，`fp8_gt_cos >= 0.9996`。
- `1test_results/321_phase23_fp8_ws_paired_persistent_examples.log`：`run_hstu8_examples.sh` 为 14/14 PASS。
- `2benchmark_results/137_27011a7f_gpu_unlocked_phase23_fp8_ws_paired_persistent_kernel200.log`：bs=8 seq=4096 h=16 full FP8 `650.9 TFLOPS`，causal FP8 `1110.1 TFLOPS`。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full_paired_persistent.sass`：full 实例 `REG:168 STACK:0 LOCAL:0`，`LDL/STL=0`。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_causal_paired_persistent.sass`：causal 实例 `REG:168 STACK:56 LOCAL:0`，`LDL=45`、`STL=27`。
- `1test_results/323_phase23_fp8_ws_full_causal_persistent_accuracy.log`：full+causal persistent sweep 通过，`fp8_gt_cos >= 0.9996`。
- `1test_results/324_phase23_fp8_ws_full_causal_persistent_examples.log`：`run_hstu8_examples.sh` 为 14/14 PASS。
- `2benchmark_results/138_27011a7f_gpu_unlocked_phase23_fp8_ws_full_causal_persistent_kernel200.log`：bs=8 seq=4096 h=16 full FP8 `630.9 TFLOPS`，causal FP8 `1114.6 TFLOPS`。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full_full_causal_persistent.sass`：full 实例 `REG:168 STACK:48 LOCAL:0`，`LDL=22`、`STL=16`。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_causal_full_causal_persistent.sass`：causal 实例 `REG:168 STACK:56 LOCAL:0`，`LDL=45`、`STL=27`。
- 外层 persistent wrapper loop 前加 `#pragma unroll 1` 后，correctness 仍通过（`1test_results/326_phase23_fp8_ws_full_causal_persistent_unroll1_accuracy.log`、`327_phase23_fp8_ws_full_causal_persistent_unroll1_examples.log`），但 SASS spill 不变：full 仍 `STACK:48`、`LDL=22`、`STL=16`，causal 仍 `STACK:56`、`LDL=45`、`STL=27`。
- `2benchmark_results/139_27011a7f_gpu_unlocked_phase23_fp8_ws_full_causal_persistent_unroll1_kernel200.log`：外层 `unroll 1` 版本 full FP8 `627.5 TFLOPS`，causal FP8 `1097.6 TFLOPS`，不作为性能改进。
- `1test_results/334_phase23_fp8_ws_split_static_decode_accuracy.log`：split persistent + static decode sweep 通过，`fp8_gt_cos >= 0.9996`。
- `1test_results/335_phase23_fp8_ws_split_static_decode_examples.log`：`run_hstu8_examples.sh` 为 14/14 PASS。
- `2benchmark_results/140_27011a7f_gpu_unlocked_phase23_fp8_ws_split_static_decode_kernel200.log`：bs=8 seq=4096 h=16 full FP8 `649.5 TFLOPS`，causal FP8 `1162.5 TFLOPS`。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full_split_static_decode.sass`：full 实例 `REG:168 STACK:0 LOCAL:0`，`LDL/STL=0`。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_causal_split_static_decode.sass`：causal 实例 `REG:168 STACK:0 LOCAL:0`，`LDL/STL=0`。
- `1test_results/336_phase23_fp8_ws_persistent_hoist_s1_accuracy.log`：S1-hoist split persistent sweep 通过，`fp8_gt_cos >= 0.9996`。
- `1test_results/337_phase23_fp8_ws_persistent_hoist_s1_examples.log` / `1test_results/338_hstu8_examples_qm2.log`：`run_hstu8_examples.sh` 为 14/14 PASS。
- `2benchmark_results/142_27011a7f_gpu_unlocked_phase23_fp8_ws_persistent_hoist_s1_kernel200_target.log`：同配置 bs=8 seq=4096 h=16 full FP8 `643.6 TFLOPS`，causal FP8 `1160.4 TFLOPS`；相对 benchmark `140`，full 略降、causal 基本持平，不作为性能提升结论。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full_persistent_hoist_s1.sass`：full 实例无 `LDL/STL`。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_causal_persistent_hoist_s1.sass`：causal 实例无 `LDL/STL`。
- `1test_results/339_phase23_fp8_ws_four_load_warp_accuracy.log`：four-load-warp sweep 通过，`fp8_gt_cos >= 0.9996`。
- `1test_results/340_phase23_fp8_ws_four_load_warp_examples.log` / `1test_results/341_hstu8_examples_qm2.log`：`run_hstu8_examples.sh` 为 14/14 PASS。
- `2benchmark_results/143_27011a7f_gpu_unlocked_phase23_fp8_ws_four_load_warp_kernel200_target.log`：同配置 bs=8 seq=4096 h=16 full FP8 `625.2 TFLOPS`，causal FP8 `1166.0 TFLOPS`；causal 略高于 `140/142`，full 明显回退。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full_four_load_warp.sass`：full 实例无 `LDL/STL`。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_causal_four_load_warp.sass`：causal 实例无 `LDL/STL`。
- `1test_results/346_phase23_fp8_ws_dynamic_o_stage_mathstage_accuracy.log`：dynamic O-stage sweep 通过，`fp8_gt_cos >= 0.9996`。
- `1test_results/347_phase23_fp8_ws_dynamic_o_stage_mathstage_examples.log` / `1test_results/348_hstu8_examples_qm2.log`：`run_hstu8_examples.sh` 为 14/14 PASS。
- `2benchmark_results/145_27011a7f_gpu_unlocked_phase23_fp8_ws_dynamic_o_stage_mathstage_kernel200_target.log`：同配置 bs=8 seq=4096 h=16 full FP8 `630.1 TFLOPS`，causal FP8 `1169.1 TFLOPS`；相对固定 stage0 four-load-warp 版本，full/causal 均略升。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full_dynamic_o_stage_mathstage.sass`：full 实例无 `LDL/STL`。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_causal_dynamic_o_stage_mathstage.sass`：causal 实例无 `LDL/STL`。
- `1test_results/352_phase23_fp8_ws_dynamic_o_stage_final_accuracy.log`：最终 dynamic O-stage sweep 通过，`fp8_gt_cos >= 0.9996`。
- `1test_results/353_phase23_fp8_ws_dynamic_o_stage_final_examples.log` / `1test_results/354_hstu8_examples_qm2.log`：`run_hstu8_examples.sh` 为 14/14 PASS。
- `2benchmark_results/147_27011a7f_gpu_unlocked_phase23_fp8_ws_dynamic_o_stage_final_kernel200_target.log`：同配置 bs=8 seq=4096 h=16 full FP8 `630.3 TFLOPS`，causal FP8 `1170.3 TFLOPS`。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full_dynamic_o_stage_final.sass`：full 实例无 `LDL/STL`。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_causal_dynamic_o_stage_final.sass`：causal 实例无 `LDL/STL`。
- `q_consumed_mbar + bar.sync 4,96` 实验已拒绝：correctness 通过（`1test_results/349_phase23_fp8_ws_q_consumed_bar4_96_accuracy.log`、`350_phase23_fp8_ws_q_consumed_bar4_96_examples.log`），但 benchmark `146` full/causal 只有 `621.6/1155.0 TFLOPS`，低于 dynamic O-stage 保留版本。
- `o_empty` pre-arrive 版本已拒绝：sweep 可过（`1test_results/362_phase23_fp8_ws_mbar_cnt_sync_oready8_helper_accuracy.log`），但 examples `363/364` 只有 12/14 PASS，pure causal `seq=128/256` 出现 NaN。根因是 K/V 首次不会 wait `o_empty`，pre-arrive 会让第一次 pending wait 吃到 stale initial phase，导致 K/V 在真实 O-store 完成前覆盖 O SMEM。
- `1test_results/365_phase23_fp8_ws_mbar_cnt_sync_oempty_real_accuracy.log`：移除 `o_empty` pre-arrive 后 sweep 通过，`fp8_gt_cos >= 0.9996`。
- `1test_results/366_phase23_fp8_ws_mbar_cnt_sync_oempty_real_examples.log`：`run_hstu8_examples.sh` 为 14/14 PASS。
- `2benchmark_results/148_98cfc5b8_gpu_unlocked_phase23_fp8_ws_mbar_cnt_sync_oempty_real_kernel200_target.log`：同配置 bs=8 seq=4096 h=16 full FP8 `646.5 TFLOPS`，causal FP8 `1177.6 TFLOPS`。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full_mbar_cnt_sync_oempty_real.sass`：full 实例 `REG:168 STACK:0 LOCAL:0`，无 `LDL/STL`。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_causal_mbar_cnt_sync_oempty_real.sass`：causal 实例 `REG:168 STACK:0 LOCAL:0`，无 `LDL/STL`。
- `1test_results/377_phase23_fp8_ws_kv_stage_persist_accuracy.log`：K/V stage persistent 版本 sweep 通过，`fp8_gt_cos >= 0.9996`。
- `1test_results/378_phase23_fp8_ws_kv_stage_persist_examples.log` / `1test_results/379_hstu8_examples_qm2.log`：`run_hstu8_examples.sh` 为 14/14 PASS。
- `2benchmark_results/155_98cfc5b8_gpu_unlocked_phase23_fp8_ws_kv_stage_persist_kernel200_target.log`：同配置 bs=8 seq=4096 h=16 full FP8 `642.2 TFLOPS`，causal FP8 `1204.1 TFLOPS`。causal 明显超过 benchmark `148`，full 低于 benchmark `148/140`。
- `1test_results/380_phase23_fp8_ws_o_stage_ntiles_formula_accuracy.log`：O-stage N-tile parity 公式版 sweep 通过，`fp8_gt_cos >= 0.9996`。
- `1test_results/381_phase23_fp8_ws_o_stage_ntiles_formula_examples.log` / `1test_results/382_hstu8_examples_qm2.log`：`run_hstu8_examples.sh` 为 14/14 PASS。
- `2benchmark_results/156_98cfc5b8_gpu_unlocked_phase23_fp8_ws_o_stage_ntiles_formula_kernel200_target.log`：同配置 bs=8 seq=4096 h=16 full FP8 `643.7 TFLOPS`，causal FP8 `1194.6 TFLOPS`。
- `2benchmark_results/157_98cfc5b8_gpu2407MHz_phase23_fp8_ws_o_stage_ntiles_formula_full_benchmark.log`：2407MHz locked 默认全量 benchmark；bs=8 seq=4096 h=16 kernel-only full/causal FP8 `627.8/1168.8 TFLOPS`。
- `2benchmark_results/158_98cfc5b8_gpu2407MHz_phase23_fp8_ws_nonpersistent_full_benchmark.log`：同环境临时关闭 full/causal persistent scheduler 的 current-code non-persistent 对照；bs=8 seq=4096 h=16 kernel-only full/causal FP8 `4.8/9.0 TFLOPS`。该数据不是旧 phase21 non-persistent 源码，只用于证明当前 four-load-warp/mbar 代码不能按 per-tile non-persistent 方式运行。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full_kv_stage_persist.sass` 和 `..._causal_kv_stage_persist.sass`：full/causal 均为 `REG:168 STACK:0 LOCAL:0`，无 `LDL/STL`。

下一步：

- 对当前 K/V stage persistent four-load-warp 版本跑 NCU，确认 full/causal 的 No Eligible、Long Scoreboard、Math Throttle、SMEM stall、mbarrier wait、TMA pipe 竞争和 tail-wave 分布。
- full 仍略低于 benchmark `140/148`，需要确认跨 tile stage overlap 对 full 无收益或控制流成本抵消收益；causal 已形成当前最好记录，可以重点看 paired persistent tail-wave 和 O-store overlap 是否改善。
- 下一步若继续，应评估是否能让 Q warp 在 math 加载当前 Q 到 register 后更早预取 next Q。
- 和旧 static launch pair 做同频对照：当前 split persistent causal 已超过旧 static pair 记录，full 略低于旧 static pair 最好记录。

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
- Phase 23：FP8 WS pure full 和 pure causal split persistent kernel 已实现并通过 correctness；当前 S1-hoist full/causal 实例均无 register spill，但 S1-hoist 未带来性能提升。

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

1. Phase 24：支持 SM120 FP8 paged KV cache。先打通 API/参数/schema，再实现 paged K/V TMA load、paged scale-factor layout、causal/target mask，最后做 correctness 和 benchmark。
2. 对 Phase 23 当前 FP8 WS 版本做 NCU，确认 No Eligible、Long Scoreboard、SMEM、register、occupancy、mbarrier wait、TMA pipe 竞争和 tail-wave 分布。
3. 若继续性能优化，评估 `kBlockN=256` 的 SMEM 可行性和寄存器压力，重点检查是否超过 85KB 预算以及是否恶化 1 CTA/SM 限制。
4. 评估能否把 SMEM 降到约 50KB，以解除 1 CTA/SM 限制；这是高难度方向，但可能直接改善 No Eligible。
5. Phase 22 BF16：停止通过降低 BF16 causal tile 或单纯提高 launch bound 追求 occupancy；已测方案没有超过默认 `{128,128,8}`。后续若继续 BF16，应优先寻找减少指令/同步/冗余工作且不缩小主 tile 的方案，或做 runtime 多 kernel dispatch 但必须证明目标 shape 有稳定收益。

不建议立即继续：

- 不继续 `kBlockM=256`，除非先提出明确的寄存器活跃量削减方案。
- 不把 `tCrSFP` 提升到 N-loop 外；已有 race condition 记录。

## 验证清单

每次 CUDA/CuTe 内核改动后至少执行：

- 重新编译 HSTU extension。
- `HSTU_SWEEP_FP8_QUANT_MODE=2 python sweep_accuracy.py`，要求 `fp8_gt_cos >= 0.995`，理想值 `>= 0.9996`。
- BF16 相关改动运行 `hstu_test.py` 的 `HSTU16Test` 定向用例，至少覆盖 aligned WS 场景和不对齐 fallback 场景。
- FP8 相关改动运行 `bash run_hstu8_examples.sh`，要求 14/14 PASS。
- paged KV 相关改动运行 SM120 FP8 paged KV 定向测试，至少覆盖 `page_size == kBlockN`、last page 非满、target length 为 0 和非 0。

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
