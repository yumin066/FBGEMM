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
- headDim 主要为 128；`kBlockN` 必须整除 128。
- Q/K 用于 GEMM1，K 方向为 headDim=128；V 用于 GEMM2，K 方向为 `kBlockN=128`。
- scale factor 为 e8m0，每 128 个 K 元素对应 1 个 scale；4 个连续块打包为 1 个 `int32`。

当前 FP8 WS Phase 23 状态：

- FP8 WS pure causal no-RAB 路径使用静态首尾配对 launch queue：`blockIdx.x` 映射为 `0, total_tiles-1, 1, total_tiles-2, ...`，用于拉平 causal heavy/light tile 的 wave/tail。
- FP8 WS full、local、context、target、arbitrary 等其他实例保持原始三维 grid：`dim3(num_m_block, h, b)`。
- 不保留 dynamic persistent work queue：correctness 可过，但 tile 间需要 CTA sync/atomic，性能收益小且波动。
- 不保留 paired persistent wrapper loop：causal 有单次高分，但会让 inline WS body 出现 `STACK`/`LDL`/`STL` spill，full 明显回退。
- 若后续继续做真正 persistent kernel，应先把 `hstu_compute_attn_1rowblock_sm120_fp8_ws` 拆成 producer/consumer/store 三段，而不是在 wrapper 中循环调用整个 inline body。

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
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user HSTU_ARCH_LIST="12.0" HSTU_DISABLE_BACKWARD=TRUE HSTU_DISABLE_DETERMINISTIC=FALSE HSTU_DISABLE_HDIM32=TRUE HSTU_DISABLE_HDIM64=TRUE HSTU_DISABLE_HDIM256=TRUE MAX_JOBS=32 pip install --no-build-isolation --config-settings editable_mode=compat -e .
```

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

BF16 正确性应优先使用 `hstu_test.py` 的 `HSTU16Test`，不要用 `run_hstu8_examples.sh` 作为 BF16 结论。当前常用构建关闭了 hdim32/64/256，因此 BF16 定向测试应先选 `attn_hidden_dims=(128, 128)`：

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

该脚本覆盖 HSTU8/FP8 block-scale 示例，判定标准：`14/14 passed`；它不替代 BF16 的 `HSTU16Test`。

benchmark：

```bash
PYTHONUSERBASE=/home/scratch.minyu_gpu/project/.cache/pip-user python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu/benchmark/bench_hstu_attn_sm120.py 2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/2benchmark_results/NNN_xxx.log
```

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

Phase 23 FP8 WS SASS 参考：

- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_full_static_launch_pair.sass`：full 实例 `REG:168 STACK:0 LOCAL:0`，`LDL/STL=0`。
- `4sass_dump_ws/hstu_fwd_kernel_sm120_fp8_ws_tma_I128_causal_static_launch_pair.sass`：causal 实例 `REG:168 STACK:0 LOCAL:0`，`LDL/STL=0`。

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
- FP8 WS static launch pair 下一步应跑 NCU，重点确认 tail wave 是否改善，以及 No Eligible/Long Scoreboard 是否变化。

跑性能数据前可锁频：

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
