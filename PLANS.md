# HSTU SM120 当前计划

来源：`CLAUDE.md`、`PLAN.md`、`memory.md` 和后续 Codex 协作记录。本文件只保留当前目标、阶段状态、验证方法和仍有价值的历史结论；长期协作规则放在 `AGENTS.md`。

## 当前状态

Phase 33 当前目标：并行推进两条性能问题。BF16 CuTe DSL 的目标是 GPU kernel-only 性能接近当前 BF16 C++，验收线为代表 aligned D128 full/causal 至少达到 C++ 的 95%；wrapper-call 端到端开销单独记录，不和 kernel-only 混算。FP8 WS 的目标聚焦 `seq=2048,h=4,d=256`：解释并降低 RAB/DRAB 带来的 TFLOPS 下降，解释各 mask 的 valid-pairs 与调度差异，分析 D256 causal 只有 full 约一半 TFLOPS 的原因，并对比 benchmark `596` 与 `173` 的 D128 full/causal regression。

Phase 32 当前目标：参考现有 CuTe DSL 路径实现 SM120 BF16 prototype，并与当前 SM120 C++ BF16 路径做性能对比。已确认 SM100 DSL `tcgen05` BF16 MMA 不能直接用于 SM120：CUTLASS DSL 4.5 的 `tcgen05.MmaF16BF16Op` 只接受 `sm_100a/sm_101a/sm_103a`，`sm_120a` 会在 JIT 阶段报 arch 不支持；强制按 `sm_100a` 编译后也会在 SM120 上报 `cudaErrorNoKernelImageForDevice`。因此 Phase 32 prototype 改用 Ampere-style warp-level `mma.sync` CuTe DSL，与当前 SM120 C++ BF16 的 “SM80 mma.sync on SM120” 模型一致。当前 SM120 CuTe DSL BF16 已补齐 D32/D64/D128/D256 × full、causal、local、context+causal、target+causal、context+target+causal、arbitrary × none/RAB/DRAB；覆盖 dense full-batch、compact varlen、`alpha=1.0/0.1`。BF16 paged KV 以 wrapper-side materialization 接入，先还原成 dense K/V 后复用同一个 DSL kernel。最新结论：早期 CUDA-event benchmark 的 8%~12% ratio 主要是把 Python/CuTe DSL wrapper 的 descriptor/from_dlpack/launch 开销计入了 event 区间，不代表 GPU kernel 本身；nsys GPU kernel summary 显示对齐 8-warps、fast_silu 和 in-kernel scaling 后，`bs=1,seq=1024,H=4,D=128,full` DSL kernel 为 `40.0us`，当前 C++ BF16 为 `35.1us`，约 `87.7%`。

Phase 31 当前状态：已修复 SM120 FP8 `D=64 paged full` register spill。修复方式是把 persistent scheduler 的 `num_m_block/total_tiles/total_tile_pairs` 从 top-level wrapper 参数和跨分支 live range 中移除，改为 load/math persistent 分支内局部计算。最终 D64 paged full resource 为 `REG:168 STACK:0 LOCAL:0`，SASS 无 `LDL/STL`。最终 correctness 通过 `hstu_test.py`、`sweep_accuracy.py` 和 `run_hstu8_examples.sh`；锁频全量 FP8+paged kernel benchmark 已完成，日志为 `2benchmark_results/phase31_gpu2407MHz_full_fp8_paged_kernel_final.log`。

Phase 30 当前状态：SM120 FP8 `quant_mode=2` forward 已接入 headDim32/headDim64，并按当前 headDim128/headDim256 的支持面覆盖 `D=32/64 × non-paged/paged × full/causal/local/context/target/arbitrary × none/RAB/DRAB`。Correctness 已通过 examples、sweep 和 `hstu_test.py`；D32/D64 全 mask/bias kernel-only benchmark 已锁频跑完。

Phase 29 已完成：repo 自带 `hstu_test.py` 的 SM120 FP8 `quant_mode=2` fixed matrix 已覆盖 irregular seqlen corner cases，并把 paged KV cache 纳入同一个 pytest 验收矩阵。`(99,99)` 等非 block 对齐输入现在真实运行通过，不再通过 alignment guard 跳过。

Phase 28 已完成 `headDim=128/256 + paged KV + RAB/DRAB` 全配置：full、pure causal、context+causal、target+causal、local、arbitrary 真实运行通过。

Phase 27 已完成 `headDim=128/256 + non-paged + RAB/DRAB` 全配置：full、pure causal、context+causal、target+causal、local、arbitrary 默认走 FP8 WS TMA；D=128 RAB 已从旧 BN128 fallback 切到 BN64 WS。

Phase 25 的 SM120 forward headDim256 已完成第一阶段支持。FP8 hdim256 correctness 路径已打通，pure non-paged causal 的 residual register spill 已清理；验收覆盖当前 hdim128 已支持的 FP8 non-paged 配置组合，以及当前 hdim128 paged KV 已支持的 full、causal、context+causal、target+causal、local、arbitrary 组合。

最新有效基线：

- Phase 21：Opt A + Opt C，benchmark 056 为历史 hdim128 FP8 non-paged 基线。
- Phase 22：BF16 默认 hdim128 路径保留 `kBlockN=128`，WS/TMA 实验因性能回退不进入默认路径。
- Phase 23：FP8 WS pure full 使用 branch-local grid-stride persistent loop，launch CTA 数为 `min(total_tiles, SM_count)`；pure causal 使用 branch-local paired persistent loop，launch CTA 数为 `min(total_tile_pairs, SM_count)`；其他 mask 保持原始 3D grid。
- Phase 23 split persistent 把 TMA/load persistent loop、math/softmax persistent loop 和 epilogue/O-store 所属路径拆开；O TMA store 由 active load warp 负责，math warps 通过 O ready/empty mbarrier handoff。
- 当前工作树在 split persistent 基础上已把 Q/K/V/O producer/consumer mbarrier 初始化和 CTA S1 提到 persistent loop 外，跨 tile 维护 mbarrier parity 和 K/V stage；`S_arb` 仍只属于 arbitrary/non-persistent 路径。
- 当前 K/V stage persistent four-load-warp O-stage N-tile parity 公式版 bs=8 seq=4096 h=16 causal：FP8 1194.6 TFLOPS，BF16 677.8 TFLOPS，FP8 比 BF16 快 76.2%。
- 当前 K/V stage persistent four-load-warp O-stage N-tile parity 公式版 bs=8 seq=4096 h=16 full：FP8 643.7 TFLOPS，BF16 369.2 TFLOPS，FP8 比 BF16 快 74.4%。
- Pre-hoist split persistent 最好记录仍是 benchmark `140`：full/causal `649.5/1162.5 TFLOPS`。S1-hoist correctness 通过且无 spill，但未形成性能提升。
- SASS：当前 K/V stage persistent four-load-warp full/causal 目标 kernel 均为 `REG:168 STACK:0 LOCAL:0`，无 `LDL/STL`。
- Phase 24：full paged hdim128 相对 non-paged FP8，benchmark `182` 全部 36 个 full case 平均 `-3.0%`，`seq>=1024` 平均 `-0.1%`，`seq>=4096` 平均 `+1.5%`；causal paged 平均约 `-8.2%`。
- Phase 24：paged same-input 对照 `1test_results/451_phase24_paged_same_input_sweep.log` 显示 full/causal aligned cases 输出 `max_err=0`。
- Phase 24：`run_hstu8_examples.sh` 的 paged mirror 最新日志 `1test_results/456_phase24_paged_mirror_examples_final.log` 为 `31/31 passed`；`sweep_accuracy.py` paged mirror 最新日志 `1test_results/455_phase24_paged_mirror_sweep.log` 通过。
- Phase 25 hdim256 已接入：SM120 编译/dispatch/runtime guard 支持 hdim256，FP8 WS 使用 `{kBlockM=128,kBlockN=64,kHeadDim=256}`；correctness 和 SASS 日志见 Phase 25 记录。
- Phase 28 paged RAB/DRAB complex coverage 已接入：D=128/D=256 的 full、pure causal、context+causal、target+causal、local、arbitrary + paged KV + RAB/DRAB examples 全部真实通过。RAB bias 仍在 WS math fragment 中从 global RAB tensor 直接加到 `acc_s`，不新增 RAB SMEM tile。D=128 paged causal+RAB SASS 为 `REG:168 STACK:0 LOCAL:0` 且无 `LDL/STL`；D=256 paged causal+RAB 为 `REG:168 STACK:8 LOCAL:0`，有 1 个 `STL` 和 1 个 `LDL`，作为残余风险记录。
- Phase 27 non-paged RAB/DRAB WS 当前版已接入：D=128/D=256 full、pure causal、context+causal、target+causal、local、arbitrary non-paged RAB/DRAB 走 WS TMA，correctness 通过。`bs=2,seq=1024,h=4,d=256` kernel-only latency 相比 fallback：full `0.0693ms vs 0.1111ms`，local `0.0583ms vs 0.0825ms`，context `0.1264ms vs 0.2118ms`，target `0.0798ms vs 0.1246ms`；arbitrary WS 与 fallback 基本持平，用于统一路径。
- `bench_hstu_attn_sm120.py` 已扩展为默认覆盖 `full/causal/local/context/target/arbitrary` × `none/rab/drab`，每个逻辑 case 同时输出 BF16、non-paged FP8、paged FP8。三列独立计时，BF16 unsupported 不再阻塞 FP8/paged 结果；可用 `--columns fp8 paged` 只跑目标列。TFLOPS 按 `generate_input` 产生的实际 valid attention pairs 计算，避免 causal/local/context/target 被 full FLOPs 高估。
- repo 自带 `hstu_test.py` 已纳入当前 correctness 目标；Phase 31 最新全文件日志 `1test_results/phase31_final_hstu_test.log` 为 `3 passed, 1 skipped`。其中 `HSTU8Test::test_sm120_fp8_blockscale_matrix` 已扩展到 D=32/D=64/D=128/D=256、seq=99/128/256、full/causal/local/context/target/arbitrary、none/RAB/DRAB、non-paged/paged。后续 HSTU CUDA/CuTe 改动必须保持该 pytest 入口通过。

## Phase 33：BF16 CuTe DSL 与 FP8 WS 性能优化

目标：

- BF16 CuTe DSL：先把 GPU kernel-only 性能从当前约 87-88% 提到接近 BF16 C++，代表 aligned D128 full/causal 验收线为 `>=95%`。wrapper-call 端到端开销单独作为 Python/CuTe DSL launch 问题，不用 CUDA-event wrapper-call 口径判断 kernel 本体。
- FP8 WS：以 `bs=1,seq=2048,h=4,d=256` 为第一代表形状，降低 RAB/DRAB 性能损失，解释 full/causal/local/context/target/arbitrary 的 FLOP 口径和调度差异，并复查 D128 benchmark `596` 相对 `173` 的 full/causal regression。

当前证据：

- BF16 CuTe DSL kernel-only 差距不是 tile size 或 MMA atom 不一致。D128 no-RAB 两边都使用 `{kBlockM=128,kBlockN=128}` 和 SM80-style `mma.sync`，但 DSL 固定分配 `sQ/sK/sV`，每个 N block 重新从 SMEM 读 Q；C++ BF16 对 `kHeadDim<=128` 使用 `Q-in-reg + Share_Q_K_smem`。nsys 代表 case 显示 DSL dynamic SMEM 约 `99KB`，C++ 约 `64KB`。
- BF16 CuTe DSL wrapper-call 仍由 Python/CuTe descriptor/from_dlpack/launch 固定开销主导。当前默认 CUDA-event geomean DSL/C++ 约 `0.203`，不能代表 GPU kernel-only；后续 benchmark 必须明确区分 wrapper-call 与 kernel-only。
- FP8 WS D256 RAB/DRAB 的主要瓶颈是 `add_rab_bs` 在 math warp critical path 中对 BF16 RAB tensor 做 scalar global load。NCU `full -> full+RAB` 显示 duration 约 `109.5us -> 193.9us`，DRAM throughput `3.69% -> 12.93%`，L2 hit `91.4% -> 65.4%`，No Eligible `67.3% -> 76.1%`；full+RAB 无 local spill，因此主因不是 register spill。
- FP8 WS D256 causal no-RAB 的 latency 与 full 接近，但 valid pairs 只有 full 的约 50%，所以 TFLOPS 约为 full 一半。该形状只有 `16 M-blocks * 4 heads = 64 CTAs`，且 D256 non-paged causal 当前不走 paired persistent scheduler，仍使用 runtime mask 单体 N-loop。
- benchmark `596` 相比 `173`：D128 no-bias full kernel-only 无 regression，long-seq geomean 约 `+3.4%`；D128 no-bias causal 有小幅回退，long-seq geomean 约 `-6.1%`，需作为后续 FP8 WS causal 调度检查项。

本轮已完成：

- 已启动两个只读子 agent 分别分析 BF16 CuTe DSL 与 FP8 WS；子 agent 未修改文件。
- 已在锁频 `2407MHz` 下完成 FP8 WS D256 `bs=1,seq=2048,h=4` 的 NCU 对照，并已解锁 GPU clocks。profile 产物：
  - `3profile_results/phase33_fp8_d256_s2048_h4_full_none_ncu.ncu-rep`
  - `3profile_results/phase33_fp8_d256_s2048_h4_full_rab_ncu.ncu-rep`
  - `3profile_results/phase33_fp8_d256_s2048_h4_causal_none_ncu.ncu-rep`
  - `3profile_results/phase33_fp8_d256_s2048_h4_causal_rab_ncu.ncu-rep`
  - 同名 `.csv` 已导出，便于下一台机器继续解析。
- FP8 WS 已完成第一轮 direct-global RAB 优化：在 selected masked paths 中先 apply mask，再让 `add_rab_bs_skip_masked` 跳过已为 `-inf` 的 accumulator 元素，避免无效 RAB global load。未采用“对所有 masked paths 复制完整 mask predicate 到 RAB add 内”的版本；该版本 correctness 通过但 context/target/arbitrary 明显回退。
- 当前保留的 selective 版本 correctness 已通过：
  - `1test_results/phase33_fp8_ws_rab_selective_maskbefore_hstu_test.log`
  - `1test_results/phase33_fp8_ws_rab_selective_maskbefore_sweep.log`
  - `1test_results/phase33_fp8_ws_rab_selective_maskbefore_examples.log`
- 锁频 D256 `bs=1,seq=2048,h=4` 子集 benchmark：`2benchmark_results/600_gpu2407MHz_phase33_fp8_ws_rab_selective_maskbefore_d256_s2048_h4_kernel.log`。相对原始同机基线 `2benchmark_results/597_gpu2407MHz_phase33_fp8_ws_original_d256_s2048_h4_kernel_after_revert.log`，FP8 全部 case geomean `+1.95%`，FP8 RAB/DRAB geomean `+2.87%`；paged 全部 case geomean `+3.43%`，paged RAB/DRAB geomean `+5.24%`。最大收益来自 local+RAB/DRAB：non-paged 约 `+12%`，paged 约 `+26%`；full/context 等少数 case 有 `<1%` 波动或小回退。
- D256 resource 扫描日志：`1test_results/phase33_fp8_ws_rab_selective_resource_d256.log`。当前 D256 WS 仍为 `REG:168 LOCAL:0`，stack 分布没有比既有 Phase 27/28 记录恶化。完成 benchmark 后已执行 `sudo nvidia-smi -rgc` 解锁。
- FP8 WS 第二轮 RAB 优化已把 RAB tile 改为由 K load warp 发起 TMA，先落到独立 RAB SMEM buffer，再由 math warp 从 SMEM 加到 `acc_s`。RAB SMEM 当前为单 stage；double-buffer 版本需要牺牲 D128 independent-O buffer，实测更慢，不保留。主要修改文件为 `kernel_traits.h`、`hstu_fwd_kernel.h`、`hstu_fwd_kernel_fp8_ws.h`。
- 第二轮 correctness 已通过；清理单-stage RAB 残留后重新编译并验证：`1test_results/619_hstu_test_rab_smem_cleanup.log` 为 `3 passed, 1 skipped`，`1test_results/620_sweep_rab_smem_cleanup.log` 通过，`1test_results/621_hstu8_examples_rab_smem_cleanup.log` 为 `284/284 passed`。D128 full+RAB SASS 代表实例 `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full_rab_cleanup.sass`，dump 日志 `1test_results/622_dump_sass_i128_full_rab_cleanup.log`；resource 为 `REG:168 STACK:0 LOCAL:0`，无 `LDL/STL`。
- 第二轮锁频全量 kernel benchmark：`2benchmark_results/607_6098fb7a_gpu2407MHz_rab_smem_cleanup_full_kernel.log`。代表 `bs=8,seq=2048,h=16,d=128`：full no-RAB `656.6 TFLOPS`，full+RAB `320.6 TFLOPS`，causal no-RAB `483.8 TFLOPS`，causal+RAB `229.6 TFLOPS`；D256 full no-RAB `566.5 TFLOPS`，full+RAB `444.7 TFLOPS`。相比旧 direct-global RAB，D128 full+RAB 从约 `120.7` 提到 `320.6 TFLOPS`，但仍未达到 no-RAB 的 `95%`。
- 已排除三条继续优化方向：RAB double-buffer focus benchmark `2benchmark_results/604_6098fb7a_gpu2407MHz_rab_double_buffer_focus_kernel.log` 更慢；direct pointer SMEM indexing `2benchmark_results/605_6098fb7a_gpu2407MHz_rab_direct_smem_focus_kernel.log` 略慢；把 RAB add 与 alpha/silu 融合的版本 `2benchmark_results/606_6098fb7a_gpu2407MHz_rab_fused_silu_focus_kernel.log` 明显更慢。
- D128 full no-RAB vs full+RAB 的 NCU 对照说明剩余差距是 dense BF16 RAB 带宽瓶颈：`3profile_results/616_fp8_full_none_d128.ncu-rep` 到 `3profile_results/616_fp8_full_rab_d128.ncu-rep`，duration `504.7us -> 914.6us`，DRAM throughput `15.08% -> 84.85%`，memory throughput `240.6GB/s -> 1354.2GB/s`，L2 hit `85.87% -> 45.48%`，local spill 仍为 0。`bs=8,seq=2048,h=16,d=128` 的 dense per-head BF16 RAB 读量约 `1.07GB`；若要达到 no-RAB latency 的 `95%`，即使把整个目标时间都给 RAB 读，也需要约 `2.4TB/s` 有效带宽，实际还要同时承担 no-RAB 的 Q/K/V/O 工作。因此在当前 dense per-head BF16 RAB 语义下，full+RAB 达到 no-RAB `95%` 不现实。
- 第三轮 head-shared RAB 已实现：当 `Has_rab && h_rab == 1 && h > 1` 时，persistent scheduler 改用 head-fast tile decode，使同一 `(b,m)` 的不同 head 相邻执行；non-persistent grid 也改为 `dim3(h, num_m_block, b)` 并在 kernel 内按 head-fast 解释。`h=1` 仍走旧路径，避免无收益场景引入额外调度分支。benchmark 脚本新增 `--rab-heads shared`，`run_hstu8_examples.sh` 增加 D128/D256、h=4 的 `rab_h1/drab_h1` non-paged 和 paged mirror correctness。
- 第三轮 correctness：最终重编后 `1test_results/638_hstu_test_rab_h1_scheduler_final_retry.log` 为 `3 passed, 1 skipped`；`1test_results/629_sweep_rab_h1_scheduler.log` 通过；`1test_results/630_hstu8_examples_rab_h1_scheduler.log` 为 `300/300 passed`。`1test_results/632_dump_sass_i128_full_rab_h1_scheduler.log` 对应 `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full_rab_h1_scheduler.sass`，D128 full+RAB resource 为 `REG:168 STACK:0 LOCAL:0`，无 `LDL/STL`。
- 第三轮锁频 focused benchmark：per-head RAB `2benchmark_results/633_gpu2407MHz_rab_h1_scheduler_per_head_focus.log`，head-shared RAB `2benchmark_results/634_gpu2407MHz_rab_h1_scheduler_shared_focus.log`，no-RAB 对照 `2benchmark_results/635_gpu2407MHz_rab_h1_scheduler_no_rab_focus.log`。`rab_h=1` 相比 per-head RAB 的 TFLOPS geomean `+8.3%`；full `+9.3%`、causal `+7.3%`、D128 `+9.5%`、D256 `+7.1%`、h=16 `+11.5%`。典型收益：`bs=8,h=16,d=128 full+rab` `317.2 -> 391.4 TFLOPS`，`bs=1,h=16,d=256 causal+rab` `194.0 -> 247.2 TFLOPS`。
- `rab_h=1` 仍未达到 no-RAB 95%。同形状对 no-RAB 的 geomean 只有约 `66.6%`；D256 full 约 `82%~84%`，D128 full 约 `62%~64%`，D128 causal 约 `50%`。原因是 head-shared layout 主要改善 L2/DRAM reuse，但每个 head 的 CTA 仍要各自 TMA 同一 RAB tile 到 SMEM，并执行相同的 SMEM read + RAB add。
- 已尝试并拒绝 `h_rab==1` direct-global RAB add：实验日志 `2benchmark_results/636_gpu2407MHz_rab_h1_direct_global_focus.log`。它消除了每 head RAB TMA/SMEM，但 full 全面回退；只在少数 causal 小形状略好，因此不保留。masked tile-level 额外 predicate 也不保留：当前 N-loop 的 `n_block_min/max` 和 arbitrary valid-block list 已跳过整块无效 N tile；额外 predicate 会扰动 RAB mbarrier 热路径且没有新增可跳过 tile。
- BF16 CuTe DSL 已完成 D64/D128 no-RAB 的 `Q-in-reg + Share_Q_K_smem` 第一版实现，修改位置为 `fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120_cutedsl/hstu_attention_warp.py`。实现方式是在 no-RAB 且 `head_dim<=128` 时，把 Q 从 SMEM 读入寄存器后跨 N-loop 复用，并让 K 复用 Q 的 SMEM storage。D128 full 代表 case dynamic SMEM 已从约 `99KB` 降到 `66.56KB`，接近 C++ BF16 的 `65.54KB`；register/thread 为 DSL `210`、C++ `216`，未引入 local spill。
- BF16 CuTe DSL correctness smoke 已通过 D64/D128 full/causal no-RAB：`1test_results/phase33_cutedsl_bf16_qinreg_final_smoke.log`。整体验证在 FP8 WS 与 BF16 DSL 两项改动后也通过：`1test_results/phase33_after_fp8_bf16dsl_hstu_test.log`、`1test_results/phase33_after_fp8_bf16dsl_sweep.log`、`1test_results/phase33_after_fp8_bf16dsl_hstu8_examples.log`。
- BF16 CuTe DSL 当前未达到 `>=95%` kernel-only 目标。锁频 nsys kernel-only 代表 case `bs=4,seq=1024,h=16,d=128,full`：DSL kernel avg `124.2us`，C++ avg `113.5us`，约为 C++ 的 `91.4%`。对应 stats：`3profile_results/phase33_bf16_cutedsl_qinreg_d128_full_stats.log`。CUDA-event wrapper-call benchmark `2benchmark_results/601_gpu2407MHz_phase33_bf16_cutedsl_qinreg_event_d64_d128.log` 仍受 Python/CuTe DSL wrapper 开销污染，不能作为 kernel-only 结论。
- BF16 CuTe DSL 已排除两条错误方向：Q-in-reg 但不 share Q/K SMEM 的版本更慢，D128 full DSL kernel avg 约 `127.7us`；把 DSL swizzle 从 `cute.make_swizzle(swizzle_bits, 4, 3)` 改为 `(swizzle_bits, 3, 3)` 会破坏 correctness，说明 Python DSL swizzle 参数不能直接按 C++ `Swizzle<kSwizzle,3,3>` 等价替换。
- BF16 CuTe DSL 下一步瓶颈更可能是 SMEM layout/copy schedule。NCU instruction 对照显示 DSL 与 C++ 的 `ldsm` 指令数相同，且 local ld/st 为 0，但 DSL shared bank conflicts 约 `16,935,305`，C++ 约 `11,139`；shared data bytes DSL `2.46GB`，C++ `2.34GB`。日志：`3profile_results/phase33_bf16_cutedsl_qkshare_d128_full_ncu_inst.log` 与 `3profile_results/phase33_bf16_cpp_d128_full_ncu_inst.log`。

实施顺序：

1. 先做 FP8 WS RAB/DRAB 第一轮优化：
   - 固定代表形状 `bs=1,seq=2048,h=4,d=256`，先用已有 `596` 与 Phase 33 NCU 作为基线。
   - 直接优化 `add_rab_bs` direct-global path：对 aligned full/context 等路径减少 per-element 分支和整数计算；对 causal/local/arbitrary/target 把 mask predicate 提前用于 RAB load skip，避免后续会被置为 `-inf` 的元素仍然发起 global RAB load。
   - 不重复 Phase 31 的单 K warp 串行 RAB copy；如果 direct-global 优化不够，再评估独立 RAB TMA、RAB layout/packing，或 D256 中复用已消费 K/V stage。
   - 当前 selective mask-before-RAB 版本已完成 D256 `seq=2048,h=4` 全 mask/bias 子集和三项 correctness。下一步若继续 FP8，应补全量锁频 benchmark，并用 NCU 对 local+RAB、paged local+RAB、target+RAB 检查 RAB load sectors 与 No Eligible 是否按预期下降。
   - 第二轮 K-warp RAB TMA/SMEM 已实现并完成 correctness、SASS、锁频全量 benchmark 和 NCU。它显著优于 direct-global RAB，但 dense full+RAB 仍达不到 no-RAB 95%。继续追 95% 必须改变 RAB 字节量或复用方式，例如 `h_rab==1` 跨 head 复用、低精度/压缩 RAB、相对位置表在线生成，或 masked paths 避免搬 masked-out RAB tile。
   - `h_rab == 1` / `h_rab < h` 第一版已完成：FP8 WS scheduler 对 head-shared RAB 使用 head-fast tile 顺序，同一 `(b,m)` 的不同 head 相邻执行；benchmark 已增加 `--rab-heads shared`。它带来约 `+8.3%` geomean，但仍达不到 no-RAB `95%`，原因是每个 head 仍各自执行 RAB TMA/SMEM read/add。
   - masked path 的额外 tile-level predicate 不保留：当前 N-loop 的 `n_block_min/max` 和 arbitrary valid-block list 已跳过整块无效 N tile；partial tile 继续用当前 mask-before-RAB / skip-masked-element 逻辑。
   - FP8 RAB + scale 暂不进入本轮实现。Ampere/Hopper 当前没有 `rab_descale` 语义，Hopper FP8 明确把 RAB typed as `OutputType`；若后续做，需要作为新的 SM120 API/数值标准单独立项。
2. FP8 WS D256 causal 单独处理：
   - 先用 SASS/resource 和 NCU 确认当前 no-RAB / RAB 的 spill、runtime mask loop、tail-wave 和 No Eligible 代价。
   - 若尝试 paired persistent 或 split masked/unmasked loop，保留条件是无 `LDL/STL`、latency 下降、且 D128 causal 不继续 regression。
3. 再做 BF16 CuTe DSL kernel-body 优化：
   - D64/D128 no-RAB 的 `Q-in-reg + Share_Q_K_smem` 第一版已实现，dynamic SMEM footprint 已基本对齐 C++，但 D128 full kernel-only 仍只有 C++ 的约 `91.4%`。
   - 验证必须用 nsys/ncu kernel-only 口径；wrapper-call CUDA-event geomean 只能作为 Python launch overhead 参考。
   - 下一步优先查 DSL SMEM bank conflict 和 shared copy schedule；保留 correctness 约束为 BF16 DSL smoke + repo `hstu_test.py`，性能目标仍为代表 aligned D128 full/causal 达到 C++ BF16 的 `>=95%`。

## Phase 32：SM120 BF16 CuTe DSL Prototype

目标：

- 用 CuTe DSL 实现一版可在 SM120 上运行的 BF16 HSTU forward prototype。
- 不改变当前默认 `torch.ops.fbgemm.hstu_varlen_fwd_120` BF16 dispatch；prototype 只通过专用 Python wrapper/benchmark 调用。
- 与当前 SM120 C++ BF16 做同输入 correctness 和锁频 kernel-only 性能对比。

方案：

- 不直接复用 SM100 DSL `HSTUAttentionForwardSm100` 的 kernel body，因为它依赖 `tcgen05` BF16 MMA/TMEM，当前 CUTLASS DSL 不支持 `sm_120a`。
- 采用 `external/cutlass/examples/python/CuTeDSL/ampere/hstu_attention.py` 的 warp-level MMA HSTU 结构作为基础；SM120 BF16 C++ 路径同样使用 SM80 `mma.sync` atom，所以这条路线硬件模型一致。SM120 专用代码中的类/API 命名使用 `Sm120CuteDsl`，不再把 Ampere 写进 API 名称。
- 给 SM120 CuTe DSL kernel 增加 compile-time `has_rab` 分支，no-RAB 时跳过 RAB 工作；RAB/DRAB forward bias 改为 GEMM1 后按 accumulator 坐标从 global RAB tensor 直接加到 `acc_S`，不再分配 RAB SMEM tile，因此 D256+RAB/DRAB 可以保留 `{kBlockM=64,kBlockN=64}`。
- 当前 DSL prototype 默认使用 256 threads / 8 warps，对齐 SM120 C++ BF16 D128 no-RAB 的 `{kBlockM=128,kBlockN=128,kNWarps=8}`；默认启用 fast sigmoid，并把 `1 / seqlen_q` output scaling 放入 DSL kernel epilogue，避免 wrapper 额外发 PyTorch `mul_` kernel。
- 新增 `hstu_blackwell_sm120_cutedsl.hstu_varlen_fwd_120_bf16_cutedsl` wrapper：dense full-batch 直接 view 成 `[B,S,H,D]`；compact varlen 先按 Ampere 语义 pad 到 dense max-seqlen，target tail 放到 dense 末尾，RAB/func 同步重映射，kernel 返回后再 gather 回 compact 输出。当前 wrapper 支持 local、context、target、context+target、arbitrary mask，并按 headDim/bias 选择 tile policy。paged KV 通过 wrapper-side materialization 支持：按 `kv_cache/page_offsets/page_ids/last_page_lens` 还原 history K/V，并把 target tail 从 contiguous K/V 放到 dense 末尾。
- 新增 `bench_hstu_attn_sm120_bf16_cutedsl.py`：专门对比 current BF16 与 CuTe DSL BF16，可通过 `--masks`、`--biases`、`--headdims` 覆盖当前 DSL 支持面，输出 latency、TFLOPS、DSL/current ratio 和 correctness 误差。

功能补齐验收：

- correctness：`1test_results/phase32_cutedsl_bf16_ampere_fwd_matrix.log` 覆盖 Ampere forward 全配置面：D64/D128 × full、causal、local、context、target、context_target、arbitrary × none/RAB/DRAB × full_batch true/false × alpha 1.0/0.1，日志结尾 `PASS`。`1test_results/phase32_cutedsl_bf16_d32_d256arb_matrix.log` 覆盖 SM120 DSL 额外 D32 全 mask/bias 和早期 D256 no-RAB。`1test_results/phase32_cutedsl_bf16_paged_d256_rab.log` 覆盖 D256 × full/causal/local/context/target/arbitrary × RAB/DRAB，并验证 D128 full paged、HSTUPagedKVTest-style D128 target paged、D256 target paged+DRAB 对照。
- benchmark：锁频运行专用 benchmark，记录到 `2benchmark_results/`，给出 DSL/current TFLOPS ratio；CuTe DSL Python wrapper 的 CUDA-event 数字只代表 wrapper-call latency，GPU kernel-only 结论必须以 nsys/ncu kernel summary 为准。
- 结论：明确 CuTe DSL prototype 的性能是否有继续优化价值；若低于当前 C++ BF16，下一步优先看是否是 DSL wrapper overhead、epilogue scale 额外 torch kernel、tile scheduler 或 cp.async/mainloop 差异。

当前实现与验证：

- 已新增 `hstu_blackwell_sm120_cutedsl.hstu_varlen_fwd_120_bf16_cutedsl`，默认 dispatch 不变。
- 已新增专用 benchmark：`fbgemm_gpu/experimental/hstu/benchmark/bench_hstu_attn_sm120_bf16_cutedsl.py`；RAB smoke 日志 `1test_results/phase32_cutedsl_bf16_benchmark_rab_smoke.log` 通过。
- 功能补齐验证：`1test_results/phase32_cutedsl_bf16_ampere_fwd_matrix.log` 结尾为 `PASS`。当前支持面为 D32/D64/D128/D256 全 mask/bias、dense/compact varlen、alpha 1.0/0.1；paged KV 通过 wrapper-side dense materialization 覆盖 full、delta-q target 和 target+RAB smoke。
- D256+RAB/DRAB 已保留：旧方案失败的原因是 RAB 使用 SMEM tile；当前改为 direct global RAB add 后不再额外占用 RAB SMEM。定向日志 `1test_results/phase32_cutedsl_bf16_paged_d256_rab.log` 中 D256 全 mask × RAB/DRAB cosine 均约 `0.999996`，paged 对照 `max=0`。
- 默认 op 回归：`1test_results/phase32_bf16_cutedsl_hstu_test.log`，结果 `3 passed, 1 skipped`。
- 早期锁频 CUDA-event smoke 日志：`2benchmark_results/phase32_gpu2407MHz_bf16_cutedsl_vs_current_smoke_final.log`，geomean DSL/current TFLOPS ratio 为 `0.082`；定向日志 `2benchmark_results/phase32_gpu2407MHz_bf16_cutedsl_vs_current_seq1024_final.log` 为 `0.115`。该口径后来确认包含 DSL Python wrapper 开销，不能作为 GPU kernel-only 结论。
- 去掉 wrapper D2H 校验后的 CUDA-event 日志：`2benchmark_results/phase32_gpu2407MHz_bf16_cutedsl_after_remove_cpu_sync.log`，geomean ratio `0.127`；启用 8 warps、fast_silu、in-kernel scaling 后 `2benchmark_results/phase32_gpu2407MHz_bf16_cutedsl_fast_silu_scale_in_kernel.log`，geomean ratio `0.148`。这些仍是 wrapper-call latency，仍被 descriptor/from_dlpack/launch 开销压低。
- nsys GPU kernel-only 结论：`3profile_results/phase32_cutedsl_fast_silu_once_nsys.nsys-rep` / `.sqlite` 中，`bs=1,seq=1024,H=4,D=128,full` DSL kernel avg `40.0us`，当前 C++ BF16 kernel avg `35.1us`，DSL 达到约 `87.7%`。这才是当前 prototype 与 C++ BF16 的合理 kernel-only 对比。
- wrapper profile/优化：`3profile_results/phase32_bf16_cutedsl_wrapper_cprofile_before.log` 显示 `Tensor.dim_order()` 及其 memory-format 检查链是最大 per-call Python 热点；因 wrapper 已将 Q/K/V/O/RAB canonicalize 为 contiguous dense tensor，已改为直接传 contiguous stride order 常量。cProfile wall 从 `0.428ms/call` 降到 `0.195ms/call`；锁频默认 CUDA-event benchmark 从 `2benchmark_results/6483184e_gpu2407MHz_phase32_bf16_cutedsl_vs_cpp_default.log` 的 geomean `0.129` 提到 `2benchmark_results/6483184e_gpu2407MHz_phase32_bf16_cutedsl_dimorder_default.log` 的 `0.203`。代表 case `bs=4,seq=1024,H=16,D=128,full` 从 DSL/current `0.582` 提到 `0.874`。
- 当前结论：prototype correctness 没问题，GPU kernel 本体性能有继续优化价值；但默认 72-case wrapper-call geomean `0.203` 仍然很差，不能作为生产端到端替代方案。`0.203` 主要由 Python/CuTe DSL wrapper 固定开销拉低，小 case 更严重；按规模拆分后 `seq>=1024,H>=16,D=128` geomean 约 `0.489`，而代表大 case kernel-only 约 `0.87`。后续需要分两条线：kernel 本体用 `nsys/ncu` 分析剩余约 `10%~15%` 差距；端到端若要可用，需要把 DSL launch 从 Python wrapper 下沉到 C++/extension 层或等价低开销路径。

## Phase 31：D64 paged full spill 修复与全量性能分析

目标：

- 解决 `D=64 paged full` 的 residual register spill。
- 锁频全量 benchmark，确认 D128/D256 没有性能回退。
- 以 FP8 non-paged D128 pure full/causal TFLOPS 为同 shape baseline，分析低于 baseline 的配置组合。

实现状态：

- 已从 `hstu_compute_attn_1rowblock_sm120_fp8_ws` 的参数和 top-level kernel wrapper 中移除 persistent scheduler count 参数，避免这些值跨 load/math 分支延长 live range。
- load persistent loop 和 math persistent loop 内部分别局部计算 `num_m_block_persistent`、`total_tiles_persistent`、`total_tile_pairs_persistent`，保持 scheduler 语义不变。
- `ncu_hstu_attn.py` 已支持 `--mask-config` 和 `--bias-config`，可直接 profile RAB/DRAB、local/context/target/arbitrary 等低 TFLOPS case。

验证结果：

- build：`1test_results/phase31_final_rebuild_after_rab_revert.log`。
- pytest：`1test_results/phase31_final_hstu_test.log`，结果 `3 passed, 1 skipped`。
- sweep：`1test_results/phase31_final_sweep.log`，通过；paged same-input full/causal 继续 `max_err=0`。
- examples：`1test_results/phase31_final_examples.log`，内部日志 `1test_results/690_hstu8_examples_qm2.log`，结果 `284/284 passed`。
- SASS：`4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I64_paged_full_final.sass`；resource 为 `REG:168 STACK:0 LOCAL:0`，无 `LDL/STL`。
- 锁频全量 benchmark：`2benchmark_results/phase31_gpu2407MHz_full_fp8_paged_kernel_final.log`，覆盖 D=32/64/128/256、bs=1/4/8、seq=128/256/512/1024/2048/4096、H=4/16、full/causal/local/context/target/arbitrary、none/RAB/DRAB、FP8+paged 共 2592 行。

D128/D256 regression 结论：

- 对比 `2benchmark_results/phase31_gpu2407MHz_full_fp8_paged_kernel_after_d64_spillfix.log`，D128/D256 pure full/causal 平均无回退。
- D128 pure full/causal：FP8 平均 `+0.61%`，paged 平均 `+0.49%`；最差分别约 `-1.42%`、`-1.55%`，属于小幅噪声。
- D256 pure full/causal：FP8 平均 `+0.60%`；paged 平均 `+0.18%`。最差点为 `bs=8 seq=4096 h=16 d=256 causal paged`，`480.5 -> 453.4 TFLOPS`，需要后续单独复测确认是否为测量波动或 D256 paged causal 真实回退。

低于 D128 pure baseline 的主要原因：

- D32/D64 + RAB/DRAB 是最低 TFLOPS 主类。NCU 对比显示 D32 full no-RAB 计算侧仍有约 `42.9%` issue slots busy；D32 full+RAB 降到约 `10.3%`，No Eligible 升到约 `88.4%`，DRAM/L2 命中明显变差，且无 local spill。瓶颈是 math warp 在 `add_rab_bs` 中对 BF16 RAB 的 scalar global load 延迟，而不是 register spill。
- RAB/DRAB 的额外 BF16 global load 不计入 HSTU GEMM FLOPs；D 越小，tensor-core 工作越少，RAB load 越容易主导 wall time，因此 D32/D64 的 TFLOPS 相对 D128 pure baseline 最低。
- context/target/local/arbitrary 的 valid-pairs FLOP 分母低于 full/causal，但 tile-level overhead、mask/control flow、paged target tail 处理仍存在，TFLOPS 会自然低于 D128 pure full/causal baseline。
- paged target+RAB/DRAB 额外叠加 page-id/tail path 和 RAB global load，是当前低 TFLOPS 的最差组合。

已拒绝的优化实验：

- 尝试把 D32/D64 RAB tile 用 K load warp 预取到 SMEM，correctness 通过，但锁频 smoke benchmark 严重回退：例如 `bs=1 seq=4096 h=16 d=32 full+rab` FP8 从约 `30.6 TFLOPS` 降到约 `6.0 TFLOPS`，D64 full+rab 从约 `60.5 TFLOPS` 降到约 `12.1 TFLOPS`。
- 回退原因：单个 K load warp 串行搬 16KB RAB tile，延迟了 K-stage producer/consumer 节奏并破坏主流水。该方案不保留。

后续可行方向：

- 若继续优化 RAB/DRAB，不能再用单 warp 同步 SMEM copy。应评估独立 RAB TMA/异步 bulk copy，或把 RAB 访问改成更 cache-friendly 的布局/打包，并保证不会延迟 K/V stage release。
- D256 paged causal 的单点回退需先复测同 shape，再决定是否 profile；若稳定回退，重点看 paged scheduler、page descriptor 控制流和 tail-wave。

## Phase 30：SM120 FP8 headDim32/headDim64 full coverage

目标：把 SM120 FP8 `quant_mode=2` forward 从当前 headDim128/headDim256 扩展到 headDim32/headDim64，并覆盖当前 hdim128/256 已支持的所有配置组合。

范围：

- `D=32/64` 都必须支持 non-paged FP8 和 paged KV FP8。
- 每个 D 都必须覆盖 `full`、`causal`、`local`、`context+causal`、`target+causal`、`arbitrary`。
- 每个 mask 组合都必须覆盖 `none`、`RAB`、`DRAB`。
- irregular seqlen 继续沿用 Phase 29 actual/padded seqlen 方案，不能重新引入 block-alignment skip。
- backward、BF16 paged KV、block-scale quant modes 1/3/4/5 不进入本阶段。

实现状态：

- SM120 FP8 runtime guard、kernel generation 和 setup source filtering 已允许 D=32/D=64/D=128/D=256；BF16 仍保持 D=64/D=128/D=256。
- Python wrapper 与 C++ tile-size 已统一到 FP8 `{kBlockM=128,kBlockN=64,kNWarps=8}`，包括 RAB/DRAB 和 paged KV。
- Q/K e8m0 scale 构造已支持 `D < 128`：D32/D64 都生成 1 个 128-D scale chunk 并按现有 int32 packing 传给 kernel。
- WS TMA layout 现在按 headDim 选择 SW32/SW64/SW128；paged cp.async fallback 也改为通过 layout 计算 SMEM 地址，不再写死 SW128。
- GEMM2 的 V fragment placement 对 D64 使用 linear 64-wide N-tile 公式；D32 在 GEMM2 内部使用 64-wide TileN 并只 store 前 32 列，额外 V fragment 清零。

已完成：

- 统一 FP8 tile-size：SM120 FP8 所有 headDim 优先使用 `{kBlockM=128,kBlockN=64,kNWarps=8}`，包括 RAB/DRAB；这样 paged KV 的 `page_size == kBlockN` 继续成立，WS warp 数也保持与现有 kernel invariant 一致。
- 先启用 hdim64：移除 dispatch guard，使用 `Kernel_traits` 中的 SW64 TMA layout 替代 WS 文件里硬编码的 SW128 Q/K layout，修正 Python BN 和 scale chunk 逻辑，然后编译 `HSTU_DISABLE_HDIM64=FALSE`。
- 再启用 hdim32：切到 SW32 或必要的等价 SMEM layout，处理 TMA/cp.async fallback 和 ldmatrix 地址计算，并编译 `HSTU_DISABLE_HDIM32=FALSE`。
- 扩展测试矩阵：`run_hstu8_examples.sh`、`sweep_accuracy.py`、`hstu_test.py::HSTU8Test::test_sm120_fp8_blockscale_matrix` 都改为覆盖 D=32/64/128/256。
- 扩展 benchmark：`bench_hstu_attn_sm120.py` 支持按 D=32/64/128/256 输出 FP8/paged 列；D32/D64 不再被旧 unsupported guard 跳过。全量 benchmark 必须锁频后运行。
- SASS 验收：至少 dump D32/D64 的 full、causal、paged causal、RAB/DRAB 代表实例，检查 `LDL/STL` 和 `STACK`，避免把 register spill 引入默认路径。

验证结果：

- 构建命令使用 `HSTU_DISABLE_HDIM32=FALSE HSTU_DISABLE_HDIM64=FALSE HSTU_DISABLE_HDIM256=FALSE` 且通过。
- `run_hstu8_examples.sh` 覆盖 D=32/64/128/256，日志 `1test_results/670_hstu8_examples_qm2.log` 为 `284/284 passed`。
- `HSTU_SWEEP_FP8_QUANT_MODE=2 python sweep_accuracy.py` 日志 `1test_results/phase30_hdim32_64_sweep.log` 通过；D32/D64 的 `fp8_gt_cos >= 0.9996`，paged same-input full/causal 继续 `max_err=0`。
- `HSTU8Test::test_sm120_fp8_blockscale_matrix` 日志 `1test_results/phase30_hdim32_64_hstu8_matrix.log` 通过；全文件 pytest `1test_results/phase30_hdim32_64_hstu_test_full.log` 为 `3 passed, 1 skipped`。
- Resource/SASS：`1test_results/phase30_hdim32_64_resource_usage.log`；D32 代表项为 `REG:168 STACK:0 LOCAL:0`，D64 non-paged 代表项为 `REG:168 STACK:0 LOCAL:0`。Phase 31 已修复 D64 paged full residual spill；最终 SASS `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I64_paged_full_final.sass` 为 `REG:168 STACK:0 LOCAL:0`，无 `LDL/STL`。
- 锁频 benchmark：`2benchmark_results/phase30_gpu2407MHz_hdim32_64_kernel_only_after_guard.log`，覆盖 D=32/D=64、batch=1/4/8、seq=128/256/512/1024/2048/4096、H=4/16、full/causal/local/context/target/arbitrary、none/RAB/DRAB、FP8+paged 共 1296 行；startup summary 为 `Unsupported selected cases: none`，无 `ERR`/`ERROR`。
- D32/D64 benchmark 摘要：D32 平均 FP8/Paged 为 `50.8/47.4 TFLOPS`，paged 相对 non-paged 平均 `-9.3%`；D64 平均 FP8/Paged 为 `84.5/79.7 TFLOPS`，paged 相对 non-paged 平均 `-7.1%`。`seq>=4096` 时 paged 差距收敛到 D32 `-4.4%`、D64 `-4.0%`。
- 已补全同频 D=128/D=256 全量 benchmark：`2benchmark_results/phase31_gpu2407MHz_full_fp8_paged_kernel_final.log`。

## Phase 29：SM120 FP8 hstu_test.py irregular seqlen + paged KV

目标：让 `hstu_test.py` 中 SM120 FP8 `quant_mode=2` 支持当前主路径的所有 forward case，包括 non-paged、paged KV、RAB/DRAB、headDim128/headDim256，以及 `(99,99)` 等非 block 对齐 seqlen。测试应真实执行，不再因为 `total_q % kBlockM`、`total_k % kBlockN` 或 `seq_k % 128` 跳过。

实施方案：

- 在 SM120 Python wrapper 中增加 irregular-length padding 层。仅当 `major_version >= 12 && quant_mode == 2` 且发现 Q/K physical offset 不满足 kernel tile 对齐时启用。
- padding 策略采用 actual/padded seqlen 模型：Q 按 `kBlockM=128` padding，K/V 按当前 `kBlockN` padding；传入 kernel 的 `cu_seqlens_q/k` 为 padded offset，`seqused_q/k` 为 actual total length。
- Q/K/V physical tensor 先 padding，再做 FP8 block-scale 量化。padded token 填 0；scale factor 使用 e8m0 neutral/真实 block scale。aligned case 继续保留现有 fast path。
- 任意 mask 的 `func` 也要按 padded Q offset 重排，否则 kernel 侧 `params.func_ptr + binfo.sum_s_q` 会按 padded offset 读错。
- paged KV history cache 保持 page-aligned，不重排 page cache；仅处理新 query/new-token/tail 的 Q/K/V/SF 和 compact output unpad。
- SM120 kernel 返回 padded output 后，wrapper 按 actual `seqused_q` unpad 回 compact output，保持 `hstu_attn_varlen_func` API 不变。
- `hstu_test.py` 增加 SM120 FP8 paged KV fixed matrix，并把 irregular seqlen 加入 fixed matrix。固定 matrix 至少覆盖 aligned/irregular × headDim128/headDim256 × full、causal、context、target、local、arbitrary × none/RAB/DRAB × non-paged/paged。

范围边界：

- Phase 29 目标是 SM120 FP8 block-scale `quant_mode=2` forward。当前 C++ SM120 只支持 FP8 quant modes 0/2；Hypothesis 中 quant modes 1/3/4/5 属于 Hopper FP8 语义，不纳入本阶段。
- headDim32/headDim64 已在 Phase 30 接入；Phase 29 的 irregular/paged padding 方案继续复用。
- backward 不进入 Phase 29；SM120 backward 仍按现有测试逻辑跳过。

验证计划：

- 先跑 `python -m py_compile fbgemm_gpu/experimental/hstu/test/hstu_test.py fbgemm_gpu/experimental/hstu/hstu/cuda_hstu_attention.py`。
- 跑 `PYTHONPATH=... python -m pytest -q -s fbgemm_gpu/experimental/hstu/test/hstu_test.py::HSTU8Test::test_sm120_fp8_blockscale_matrix`，检查 `[SM120_FP8_CASE]` 是否包含 irregular 和 paged KV。
- 跑 `PYTHONPATH=... python -m pytest -q fbgemm_gpu/experimental/hstu/test/hstu_test.py --tb=short`，要求 SM120 FP8 支持面所有 case 真实运行；HSTUPagedKVTest 若仍作为 BF16 legacy test，可保留 skip，但 FP8 paged coverage 必须在 HSTU8Test 中出现。
- 跑 `bash run_hstu8_examples.sh` 和 `HSTU_SWEEP_FP8_QUANT_MODE=2 python sweep_accuracy.py`，防止 wrapper padding 影响 examples/sweep。
- 最后跑全量 `bench_hstu_attn_sm120.py` 并把日志写入 `2benchmark_results/NNN_<commit>_phase29_irregular_hstu_test_full_benchmark.log`。aligned benchmark 不应因 padding path 出现性能回退；irregular case 不作为性能主指标。

验证结果：

- build：`1test_results/592_phase29_rebuild_paged_k_offsets.log` 通过。
- py_compile：`cuda_hstu_attention.py` 和 `hstu_test.py` 通过。
- fixed matrix：`1test_results/592_phase29_hstu8_matrix_k_offsets.log`，`1 passed`，覆盖 216 个 SM120 FP8 subcases。
- full pytest：`1test_results/592_phase29_hstu_test_full_after_k_offsets.log`，`3 passed, 1 skipped`。
- examples：`1test_results/592_phase29_examples_after_k_offsets.log`，`142/142 passed`。
- sweep：`1test_results/592_phase29_sweep.log` 通过；same-input paged vs non-paged full/causal 继续 `max_err=0`。
- benchmark：`2benchmark_results/592_282d811e_phase29_irregular_paged_hstu_test_full_benchmark.log` 完成。该全量 benchmark 中 BF16 hdim256+RAB/DRAB/arbitrary baseline 会出现 unsupported/invalid-argument 行，FP8 和 paged 列仍正常输出；这些 BF16 行不作为 SM120 FP8 correctness 失败。

实现结论：

- SM120 FP8 `quant_mode=2` wrapper 使用 actual/padded seqlen：Q 按 `kBlockM=128` padding，non-paged K/V 按当前 `kBlockN` padding，arbitrary `func` 按 padded Q offset 重排，kernel 输出再 unpad 回 compact Q layout。
- paged target tail 必须按 K/V physical layout 处理。actual K 可能包含 page cache 中已有 history，而 contiguous K/V 只包含 new history + target；wrapper 在 K/V physical layout 中插入 last-page gap，使 target tail 对齐 `kBlockN`，kernel target load 使用 K physical offset，而不是 Q offset。
- RAB/DRAB bias 坐标保持 logical K col；paged target block 减去 `last_page_offset`，history page padding col 跳过。

## Phase 28：paged KV + RAB/DRAB complex coverage

目标：补齐 SM120 FP8 `quant_mode=2` paged KV cache 与 RAB/DRAB 在 context、target、arbitrary、local 上的 coverage，同时不破坏 Phase 24/26/27 已有路径。

当前状态：

- 已完成 D=128/D=256 的 full、pure causal、context+causal、target+causal、local、arbitrary + paged KV + RAB/DRAB correctness 覆盖。
- `hstu_varlen_fwd_120` 的 paged KV host guard 已从“no-target full only”放宽为 no-target full/local/arbitrary/context window；带 `num_targets` 的 paged KV 仍限制为 causal/target 语义，即 `window_size_left < 0 && window_size_right == 0`。
- local paged 不再在 `run_hstu8_examples.sh` 中标为 `PASS-UNSUPPORTED`。当前支持的是 no-target local window；target + local-window 仍不属于当前支持范围。
- `run_hstu8_examples.sh` 已把 D=128/D=256 non-paged RAB/DRAB extra cases 逐个镜像到 paged KV，新增覆盖 full/local/context+causal/target+causal/arbitrary + RAB/DRAB。
- Scheduler 口径：paged full RAB/DRAB 走 full persistent；D=128 paged pure causal/target+causal RAB/DRAB 走 paired persistent；D=256 paged RAB/DRAB 和 context/local/arbitrary 走 3D grid。

验证记录：

- build：`1test_results/591_phase28_paged_rab_complex_build.log`。
- examples：`1test_results/591_phase28_paged_rab_complex_examples.log`，`142/142 passed`。
- sweep：`1test_results/591_phase28_paged_rab_complex_sweep.log` 通过；no-RAB paged/non-paged same-input full/causal 继续 `max_err=0`。

剩余风险：

- 本阶段没有重新 dump 新的 SASS 资源日志；因为 kernel body/template 已在 Phase 26/27 接入，本次主要是 host guard 和测试覆盖放开。提交前若要把资源状态写死，应补 paged local/context/arbitrary RAB 代表实例的 `LDL/STL` scan。
- block-scale irregular length、`page_size=32` 和 backward 仍不在本阶段范围。

## Phase 27：non-paged RAB/DRAB 逐步切到 WS

目标：在不破坏现有 no-RAB 和 paged RAB 路径的前提下，把 non-paged FP8 RAB/DRAB 从 cp.async fallback 逐步迁移到 WS TMA。

当前状态：

- 已完成 D=128/D=256 全配置：full、pure causal、context+causal、target+causal、local、arbitrary + non-paged + RAB/DRAB 走 FP8 WS TMA。
- `headDim=128 + non-paged + RAB/DRAB` 已从旧 BN128 cp.async fallback 切到 BN64 WS。Python `get_bm_and_bn_block_size_fwd(rab, 128)`、C++ `get_tile_size_fwd_sm120<128, Has_rab, fp8>` 和 WS specialization 都必须保持 `{128,64,8}`。
- `headDim=256 + arbitrary + non-paged + RAB/DRAB` 也走 WS；同频 kernel-only 对照显示 WS 与 fallback 基本持平，因此该项属于路径统一，不作为性能收益来源。
- WS RAB 实现复用 Phase 26 的 direct global RAB add：GEMM1 后、mask/activation 前，根据 fragment identity 坐标从 global BF16 RAB 加到 `acc_s`，不新增 RAB SMEM tile。
- scheduler 与 WS 条件一致：full RAB 对 D=128/D=256 走 full persistent；pure causal RAB 只有 D=128 走 paired persistent；D=256 pure causal RAB 和 context/target/local/arbitrary RAB 走 3D grid。

验证记录：

- final build：`1test_results/590_phase27_d128_d256_all_rab_ws_build.log`。
- examples：`1test_results/590_phase27_d128_d256_all_rab_ws_examples.log`，`102/102 passed`。
- sweep：`1test_results/590_phase27_d128_d256_all_rab_ws_sweep.log` 通过。
- resource usage：`1test_results/590_phase27_d128_d256_all_rab_ws_resource.log` 和 `1test_results/590_phase27_d128_rab_ws_resource.log`。D=128 non-paged RAB WS 代表实例为 `REG:168 LOCAL:0`，当前扫描到的 non-paged 代表实例 `STACK:0`；D=256 non-paged RAB WS 代表实例均为 `REG:168 LOCAL:0`，full `STACK:0`，pure causal `STACK:16`，context `STACK:8`，target `STACK:24`，local `STACK:32`，arbitrary `STACK:16`。
- kernel-only 对照：
  - WS：`2benchmark_results/586_phase27_d256_nonarbitrary_rab_ws_kernel_only.log`
  - fallback：`2benchmark_results/587_phase27_d256_rab_fallback_kernel_only.log`
  - `bs=2,seq=1024,h=4,d=256 full`：WS `0.0693ms`，fallback `0.1111ms`。
  - `bs=2,seq=1024,h=4,d=256 local`：WS `0.0583ms`，fallback `0.0825ms`。
  - `bs=2,seq=1024,h=4,d=256 context`：WS `0.1264ms`，fallback `0.2118ms`。
  - `bs=2,seq=1024,h=4,d=256 target`：WS `0.0798ms`，fallback `0.1246ms`。
  - `bs=2,seq=1024,h=4,d=256 arbitrary`：此前 fallback 约 `0.0897ms`；当前切到 WS 是路径统一，非主要性能收益来源。

剩余风险：

- 新 WS specialization 仍有 stack frame，尤其 target/local；需要后续继续压 live range，并补 SASS `LDL/STL` dump 做最终提交前检查。
- e2e quick timing 对 full/local 会被 Python wrapper 和量化开销稀释，不应用来否定 kernel-only 收益；性能判断以 kernel-only 对照为准。
- hdim128 non-paged RAB/DRAB 已切到 WS；后续若关注性能，需要补 D=128 RAB kernel-only 同频对照。

## Phase 26：SM120 FP8 paged KV + RAB/DRAB

目标：让 SM120 FP8 `quant_mode=2` paged KV cache 支持 RAB/DRAB 配置。Phase 26 先覆盖 causal+RAB/DRAB；Phase 28 已扩展到 full、context、target、arbitrary、local。

当前状态：

- 已移除 Python wrapper、`hstu_varlen_fwd_120` host guard 和 FP8 dispatch 中对 paged+RAB 的拒绝。
- paged+RAB/DRAB 走 FP8 WS TMA kernel；Phase 27 后 D=128/D=256 non-paged RAB/DRAB 全配置也已切到 WS。
- hdim128 RAB 需要强制使用 `kBlockN=64`，不能沿用旧 non-paged hdim128 RAB fallback 的 `kBlockN=128`。Python block-scale wrapper 对 `rab is not None && dim == 128` 返回 BN64；paged KV 下继续以 `kv_cache.shape[2]` 作为 BN。
- WS math 路径新增 `add_rab_bs`：GEMM1 后、mask/activation 前，用 fragment identity 坐标把 BF16 RAB bias 直接从 global tensor 加到 `acc_s`。该方案不占用额外 SMEM，避免 hdim256 paged 路径超过 SM120 dynamic SMEM limit。
- hdim256 paged+RAB/DRAB 不走 paired persistent scheduler，保持 3D-grid WS，避免 persistent wrapper 进一步扩大 live range；host 和 device 两侧 `Use_paired_persistent` 条件必须保持一致。
- `run_hstu8_examples.sh` 不再把 paged RAB/DRAB 标记为 unsupported；Phase 28 后 local paged 也必须真实运行通过。

验证记录：

- build：`1test_results/578_phase26_paged_rab_final_rebuild.log`。
- examples：`1test_results/578_phase26_paged_rab_examples.log`，`62/62 passed`。其中 paged `causal+rab/drab` 对 D=128/D=256、seq=128/256 均为真实 `PASS`。
- sweep：`1test_results/578_phase26_paged_rab_sweep.log` 通过；no-RAB paged/non-paged same-input full/causal 继续 `max_err=0`。
- SASS：`4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_paged_causal_rab.sass` 为 `REG:168 STACK:0 LOCAL:0`，无 `LDL/STL`；`4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I256_paged_causal_rab.sass` 为 `REG:168 STACK:8 LOCAL:0`，当前有 1 个 `STL` 和 1 个 `LDL`。

剩余风险：

- hdim256 paged+RAB 的 8B stack/local load-store 还没有完全消除。尝试把 hdim256 RAB-add loop 改成 `#pragma unroll 1` 会恶化到 `STACK:128`，已回退。
- Phase 28 已补齐 target/context/arbitrary/local + RAB/DRAB paged examples。残余风险转为资源检查而不是 correctness coverage。

## Phase 24：SM120 FP8 paged KV cache

状态：SM120 FP8 hdim128 forward paged KV cache 已完成第一阶段支持，并已纳入 examples、sweep、benchmark 和 profile。

当前结论：

- 已实现第一版 SM120 FP8 paged KV forward correctness 路径。
- `hstu_varlen_fwd_120` schema 已接入 `kv_cache/page_offsets/page_ids/last_page_lens` optional 参数。
- `set_params_fprop_sm120` 已设置 `kv_cache` stride、page metadata 和 `is_paged_kv`。
- SM120 FP8 traits 已支持 `Paged_KV` template bool；BF16 traits 仍固定 `Paged_KV=false`。
- Python wrapper 已能在 SM120 `quant_mode=2` 下量化 paged `kv_cache`，并传入 combined cache+contiguous scale table。
- Phase 24 初始实现覆盖 headDim128、`page_size=64`、forward；full、causal、context+causal、target+causal、arbitrary paged mirror 已通过。Phase 26 后 causal+RAB/DRAB paged mirror 真实运行通过；Phase 28 后 full、context+causal、target+causal、local、arbitrary + RAB/DRAB paged mirror 也已通过。
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
- examples 已按原 14 个 non-paged HSTU8 example 逐个生成 paged mirror。Phase 28 后 D=128/D=256 RAB/DRAB extra cases 也逐个生成 paged mirror，local paged 由 unsupported 改为真实运行并通过。
- benchmark `182`：full paged 相对 non-paged FP8，全部 36 个 full case 平均 `-3.0%`，`seq>=1024` 平均 `-0.1%`，`seq>=4096` 平均 `+1.5%`；causal paged 平均约 `-8.2%`，`seq>=4096` 平均约 `-8.3%`。
- SASS：paged causal TMA 实例含 `UTMALDG.4D`，当前 SF TMA 版资源 `REG:168 STACK:0 LOCAL:0`，无 `LDL/STL` 命中；`UTMALDG` 计数为 `10`。

当前支持范围：

- forward only。
- SM120 FP8 `quant_mode=2`。
- full、causal、context+causal、target+causal、local、arbitrary paged KV 已通过；RAB/DRAB 对这些语义也已通过。
- Phase 24 初始范围为 headDim128；Phase 25/26/28 已扩展到 headDim256 paged no-RAB 与 RAB/DRAB full、causal、context、target、local、arbitrary。
- 初始只支持 `page_size == kBlockN`，优先匹配当前 BN64 路径；不先支持跨 page 的单个 N tile。
- 先不做 backward、block-scale irregular length 和 `page_size=32`；target + local-window 组合仍不在当前 paged KV 支持范围。

已实施路径：

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

## Phase 25：SM120 headDim256 支持

目标：让 SM120 forward 支持 headDim256，并把 FP8 `quant_mode=2` 当前 hdim128 已支持的所有配置组合都纳入验收范围。

当前状态：

- 已打开 SM120 hdim256 编译、dispatch 和 runtime guard；当前构建使用 `HSTU_DISABLE_HDIM256=FALSE`。
- FP8 WS hdim256 采用 `{kBlockM=128,kBlockN=64,kHeadDim=256,kNWarps=8}`；Q/K 的 e8m0 scale 改为每 token 一个 int32，按 lane 打包连续 128-D chunk，hdim128 layout 保持兼容。
- FP8 hdim256 GEMM1 按两个 128-D chunk 分段执行并累加到同一个 `acc_s`；`k_empty` release 延后到两个 chunk 的 K 从 SMEM 消费完成之后，避免 paged/non-paged 多 N tile 时 K stage 被 load warp 提前覆盖。
- FP8 hdim256 paged KV 支持 full、causal、context+causal、target+causal、local、arbitrary；Phase 28 后 RAB/DRAB 的这些 paged mirror 也真实运行并通过。
- paged hdim256 history K/V/SFB/SFV 已走 page-cache TMA；aligned full-block target tail 也走 contiguous TMA，非满或不对齐 target tail 保留 guarded copy fallback。
- hdim256 arbitrary 为了控制 dynamic SMEM，使用 single K/V stage；否则 hdim256 WS buffer 加 `ValidBlockIds` 会超过 SM120 opt-in shared-memory limit。
- `run_hstu8_examples.sh` 已把原 14 个 non-paged example 扩展为 D=128/D=256，并为 paged mirror/full/edge 同步覆盖 D=128/D=256。最新 spill-fix 后日志 `1test_results/553_phase25_hdim256_spill_final_examples.log` 为 `62/62 passed`。
- `sweep_accuracy.py` 已扩展 D=128/D=256 主表和 same-input paged vs non-paged 对照。最新 spill-fix 后日志 `1test_results/553_phase25_hdim256_spill_final_sweep.log` 通过；D=256 各项 `fp8_gt_cos >= 0.9996`，paged full/causal same-input 均为 `max_err=0`。
- hdim256 锁频全量 kernel benchmark 日志 `2benchmark_results/537_c9985651_gpu2407MHz_phase25_hdim128_vs_256_full_kernel.log` 已覆盖 D=128/D=256、full/causal、BF16/FP8/paged；逐 case 对比表在 `2benchmark_results/537_c9985651_gpu2407MHz_phase25_hdim128_vs_256_compare.md`。D=256 vs D=128 geomean：全部 cases FP8 TFLOPS `+4.3%`、paged TFLOPS `+9.2%`；`seq>=1024` FP8 TFLOPS `-11.8%`、paged TFLOPS `-6.4%`。注意 hdim256 flops 翻倍，latency geomean 分别约为 FP8 `1.91x`、paged `1.84x`。
- BF16 hdim256 定向 full/causal correctness 日志 `1test_results/542_phase25_bf16_hdim256_directed.log` 通过；`hstu_test.py` 的 SM120 attn_dim guard 已扩展到 `64/128/256`。
- hdim256 spill fix：pure non-paged causal 不再使用 paired persistent scheduler，改回普通 3D grid；该实例使用 runtime mask 单体 N-loop，避免同时实例化 masked/unmasked 两套 hdim256 causal body。hdim256 也不再启用 in-mainloop O-store。
- SASS 检查：`4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I256_full.sass`、`I256_causal.sass`、`I256_paged_full.sass`、`I256_paged_causal.sass` 均为 `REG:168 STACK:0 LOCAL:0`，且 `LDL=0/STL=0`。spill-fix 重编日志为 `1test_results/553_phase25_hdim256_spill_final_rebuild.log`。
- 锁频 causal kernel-only 子集日志 `2benchmark_results/553_gpu2407MHz_phase25_hdim256_spill_final_causal_kernel_subset.log`：D=256 non-paged causal FP8 TFLOPS 与前一版 nonpersistent 结果一致；相对 persistent+spill 版 geomean 约 `+13%`，大多数 case 提升，`bs=1/4,seq=2048,h=16` 两个高并发中等 seq case 略低。

范围：

- 先做 forward；backward 不进第一阶段。
- BF16 headDim256 先作为 correctness 基线，用于验证 tile/SMEM/dispatch 基础能力。
- FP8 headDim256 必须覆盖当前 hdim128 FP8 non-paged 已支持的全部配置组合；最小集合是 `run_hstu8_examples.sh` 原 14 个 non-paged example 的同语义 hdim256 版本。
- FP8 headDim256 paged KV 必须覆盖当前 hdim128 paged KV 已支持的全部配置组合；Phase 28 后 full、causal、context、target、local、arbitrary + RAB/DRAB 都属于强制支持项，不允许静默漏测。
- paged KV 继续以 `page_size == kBlockN` 为第一约束，避免单个 N tile 跨物理 page。

实施计划：

1. 打开 SM120 hdim256 编译和 dispatch
   - 构建去掉 `HSTU_DISABLE_HDIM256=TRUE`。
   - `hstu_varlen_fwd_120` runtime guard 从 `64/128` 扩展到 `64/128/256`，但 paged KV guard 单独控制支持范围。
   - `run_hstu_fwd_headdim_sm120` 增加 hdim256 specialization。
   - 测试入口允许 SM120 `attn_dim=256`，但只打开 Phase 25 支持范围内的 cases。

2. 先打通 BF16 headDim256
   - 修 `get_tile_size_fwd_sm120` 中 BF16 hdim256 的 tile/warp 配置，确保满足 `16*kNWarps <= kBlockM`。
   - 优先选择较保守的 `{kBlockM=64,kBlockN=64,kNWarps=4}` 或等价配置，先保证 correctness。
   - 跑 BF16 full/causal correctness，并 dump SASS 查 `LDL/STL`、`STACK`、`LOCAL`。

3. 打通 FP8 non-paged headDim256 全配置组合
   - 泛化 Q/K block-scale：headDim256 下每个 token/head 有 2 个 128-D scale chunk。
   - kernel GEMM1 必须按 D chunk 读取对应 Q/K e8m0 scale，不能沿用 hdim128 的单 scale 假设。
   - 修 FP8 WS 中 hard-coded 128 stride、O epilogue 128 列覆盖、SFA/SFB SMEM 大小和 `kHeadDim/32` 相关手写寻址。
   - 先用 correctness 证明 full/causal、context/target、arbitrary、RAB/DRAB/local 等当前 hdim128 non-paged 已支持组合可用，再决定是否进入性能优化。

4. 打通 FP8 headDim256 paged KV cache 全支持组合
   - paged `kv_cache` shape 扩展到 `[total_pages, 2, page_size, heads, 256]`。
   - paged K cache scale-factor 需要按 physical page id 和 D chunk 可寻址；V cache scale 仍按 page/N block 维度可寻址。
   - page-cache TMA descriptor 从 `[page_size, 128, h, pages]` 泛化到 `[page_size, 256, h, pages]`。
   - `copy_fp8_tile_rowmajor_to_sw128` 等 paged fallback helper 不能再 hard-code `kHeadDim=128`，必须模板化。
   - paged mirror 要按 hdim128 当前可运行语义覆盖 full、causal、context+causal、target+causal、local、arbitrary、RAB/DRAB 和 partial/edge。
   - same-input paged vs non-paged 对照要用同一份 raw Q/K/V，并保证 V block-scale 粒度一致。

验收标准：

- BF16 hdim256 non-paged full/causal correctness 通过。
- FP8 hdim256 non-paged correctness 覆盖当前 hdim128 已支持的全部 non-paged 配置组合，`fp8_gt_cos >= 0.995`，理想 `>= 0.999`。
- FP8 hdim256 paged KV cache correctness 覆盖当前 hdim128 已支持的全部 paged 配置组合；RAB/DRAB full、causal、context、target、local、arbitrary 必须真实通过，不能成为 hdim256-only failure 或静默跳过。
- FP8 hdim256 same-input 对照通过：non-paged 与 paged 在 hdim128 已支持且语义对齐的 paged cases 下输出应 bitwise 一致或误差可解释。
- SASS 需要持续查 `LDL/STL`、`STACK`、`LOCAL`；当前 hdim256 full、causal、paged full、paged causal 代表实例均已达到 `STACK:0 LOCAL:0 LDL=0 STL=0`。
- benchmark 单独输出 hdim256 表，并覆盖当前 hdim128 benchmark 中 FP8 non-paged 与 paged KV 已支持的配置组合。

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

1. Phase 25 性能：继续分析 hdim256 大 seq normalized TFLOPS 回退，尤其 `seq>=1024` causal。锁频全量 benchmark `537` 显示 D=256 vs D=128 的 FP8 geomean TFLOPS 在大 seq 为 `-11.8%`。
2. Phase 25 补充验证：如要提交，可按最终 commit 跑完整 BF16 Hypothesis 子集；当前 spill-fix 已重新记录 SASS/resource summary、examples、sweep 和 causal benchmark 子集。
3. Phase 25 后续：评估是否给 hdim256 non-paged causal 重新设计不会 spill 的 load/math persistent 版本；当前为消除 spill，只有 hdim256 non-paged causal 回到普通 3D grid，hdim128 causal 和 hdim256 paged causal 仍保留 persistent。
4. Phase 24 后续性能优化：若回到 hdim128 paged KV，优先分析 causal paged 相对 non-paged 约 `8%` 差距，重点看 page-id/descriptor 控制流、target tail copy 和小 seq TMA 固定开销。
5. Phase 23 后续性能分析：如继续 FP8 WS hdim128 优化，再跑 NCU 确认 No Eligible、Long Scoreboard、SMEM、register、occupancy、mbarrier wait、TMA pipe 竞争和 tail-wave 分布。
6. Phase 22 BF16：停止通过降低 BF16 causal tile 或单纯提高 launch bound 追求 occupancy；已测方案没有超过默认 `{128,128,8}`。后续若继续 BF16，应优先寻找减少指令/同步/冗余工作且不缩小主 tile 的方案，或做 runtime 多 kernel dispatch 但必须证明目标 shape 有稳定收益。

不建议立即继续：

- 不继续 `kBlockM=256`，除非先提出明确的寄存器活跃量削减方案。
- 不把 `tCrSFP` 提升到 N-loop 外；已有 race condition 记录。

## 验证清单

每次 CUDA/CuTe 内核改动后至少执行：

- 重新编译 HSTU extension。
- `PYTHONPATH=... python -m pytest -q fbgemm_gpu/experimental/hstu/test/hstu_test.py`，要求 repo 自带测试通过；当前 SM120 预期为 `3 passed, 1 skipped`，其中固定 FP8 matrix 必须真实执行并覆盖 non-paged/paged、D=128/D=256、seq=99/128/256。
- `HSTU_SWEEP_FP8_QUANT_MODE=2 python sweep_accuracy.py`，要求 `fp8_gt_cos >= 0.995`，理想值 `>= 0.9996`。
- BF16 相关改动运行 `hstu_test.py` 的 `HSTU16Test` 定向用例，至少覆盖 aligned WS 场景和不对齐 fallback 场景。
- FP8 相关改动运行 `bash run_hstu8_examples.sh`，要求全脚本通过；Phase 29 当前口径为 `142/142 passed`，其中 D=128/D=256 non-paged 与 paged RAB/DRAB extra cases、partial-last-page/target tail 必须通过，local paged mirror 必须真实运行通过。
- paged KV 相关改动运行 SM120 FP8 paged KV 定向测试，至少覆盖 `page_size == kBlockN`、full、causal、last page 非满、target length 为 0 和非 0。
- Phase 25 hdim256 相关改动必须额外覆盖 BF16 hdim256 full/causal、FP8 hdim256 当前 hdim128 已支持的全部 non-paged/paged 配置组合，以及 same-input paged vs non-paged 对照。

性能相关改动还需要：

- 运行 `bench_hstu_attn_sm120.py`，记录到 `2benchmark_results/NNN_xxx.log`。默认口径覆盖 18 个逻辑配置组合；需要缩小范围时使用 `--mask-configs` / `--bias-configs` / `--columns`，不要再假设 benchmark 只包含 full/causal no-RAB。
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
