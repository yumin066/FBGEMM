# Phase 6: SM120 FP8 Warp-Specialized Kernel (Producer/Consumer Pipeline)

## Goal Description

Transform the SM120 FP8 forward attention kernel from a serialized all-threads pattern (Phase 5) into a warp-specialized producer/consumer pipeline. Warp 0 (load warp) exclusively issues TMA loads for K, V^T, and signals readiness via `load_mbar`. Warps 1–8 (math warps, 8 warps) wait on `load_mbar`, execute FP8 block-scale QMMA (GEMM1 + silu + GEMM2), then signal via `math_mbar`. This enables true TMA/MMA overlap: TMA hardware executes asynchronously while math warps compute the previous tile.

Target scope: FP8 block-scale path only (`hstu_fwd_kernel_sm120_fp8_tma` and traits), forward pass only. BF16 path and backward pass are unchanged.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Kernel compiles without errors for SM120 (arch 12.0)
  - Positive Tests (expected to PASS):
    - `pip install` with `HSTU_ARCH_LIST="12.0" HSTU_DISABLE_BACKWARD=TRUE ...` exits 0
    - No CUDA `static_assert` or type-mismatch errors in build output
    - `python -c "import fbgemm_gpu.experimental.hstu"` executes without error
  - Negative Tests (expected to FAIL):
    - Building with `HSTU_ARCH_LIST="9.0"` does not compile `hstu_blackwell_sm120/` sources
    - Kernel with kNMathWarps * 16 ≠ kBlockM triggers a `static_assert` at compile time

- AC-2: Numerical accuracy preserved after warp specialization
  - Positive Tests (expected to PASS):
    - `sweep_accuracy.py` with `HSTU_SWEEP_FP8_QUANT_MODE=2` produces `fp8_gt_cos ≥ 0.95` for all (h, seq, d) configs
    - `hstu_test.py` exits 0 with max diff < 0.01
    - Results match Phase 5 FP8 output within tolerance (fp8_gt_cos ≥ 0.9990)
  - Negative Tests (expected to FAIL):
    - Running with malformed SF pointers (nullptr sf_q) still runs (unit SF fallback), does not crash
    - If load warp and math warp access the same SMEM buffer without correct barrier sequencing, accuracy degrades to fp8_gt_cos < 0.5 — this must NOT happen in the final implementation

- AC-3: True TMA/MMA overlap visible in nsys profile
  - Positive Tests (expected to PASS):
    - `nsys stats cuda_gpu_kern_sum` shows the warp-specialized kernel executing
    - nsys timeline shows TMA data-movement ops overlapping with QMMA compute ops within a single kernel invocation
    - Load warp and math warp execution time slots do not fully serialize (as observed in profile)
  - Negative Tests (expected to FAIL):
    - Phase 5 nsys profile does NOT show TMA/MMA overlap (serialized pattern — this is the baseline to beat)

- AC-4: Benchmark performance improvement over Phase 5 (direction check)
  - Positive Tests (expected to PASS):
    - `bench_hstu_attn_sm120.py` runs to completion without errors for FP8 quant_mode=2
    - FP8 kernel latency at seq≥1024 shows measurable reduction compared to Phase 5 benchmark
    - BF16 path (unchanged) shows no regression vs Phase 5 baseline
  - Negative Tests (expected to FAIL):
    - FP8 kernel latency must NOT be significantly slower than Phase 5 (warp specialization must not add overhead that outweighs overlap gains)

- AC-5: Thread count and SMEM layout are self-consistent
  - Positive Tests (expected to PASS):
    - `kNThreads == (kNWarps + kNLoadWarps) * 32 == 9 * 32 == 288` in the warp-specialized kernel traits or launch function
    - `kSmemMbarSize == 16` (two 8-byte mbarriers: load_mbar + math_mbar)
    - `kSmemSize` includes `kSmemMbarSize` and does not exceed GPU SMEM limit (≤ 99 KB for SM120)
    - `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, kSmemSize)` succeeds at runtime
  - Negative Tests (expected to FAIL):
    - Launching kernel with 256 threads (8 warps) when the implementation expects 288 threads (9 warps) must fail or produce wrong results

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
The warp-specialized FP8 forward kernel is implemented with 9 warps (warp 0 = load warp, warps 1–8 = math warps), dual mbarrier synchronization (load_mbar + math_mbar), and a single-stage SMEM pipeline. SFQ is loaded in the preamble by math warp 1 (replaces existing warp_id==0 preamble logic). SFV (V scale) is loaded per-tile by math warps after the load_mbar wait. A new `Hstu_fwd_kernel_traits_sm120_fp8_ws` traits struct and new kernel function `hstu_fwd_kernel_sm120_fp8_ws` are added alongside existing Phase 5 code. The Phase 5 TMA kernel is retained as a fallback/reference. The new kernel is invoked from `run_hstu_fwd_sm120`.

### Lower Bound (Minimum Acceptable Scope)
The minimum acceptable implementation has warp 0 issuing TMA (K+V^T) while math warps 1–8 execute QMMA (using `get_thread_slice(threadIdx.x - 32)` to correctly cover M-tiles 0–7), with correct dual mbarrier synchronization ensuring no SMEM race conditions. Accuracy (AC-2) and compilation (AC-1) pass.

### Allowed Choices
- Can use: raw `uint64_t*` PTX mbarriers (same pattern as Phase 5), SM90_TMA_LOAD, SM89 FP8 MMA atom, any number of `__syncthreads()` between warp roles
- Can use: single-stage SMEM (no extra ping-pong buffer needed for minimum viable implementation)
- Load warp is warp 0 (as specified in CLAUDE.md); math warps are 1–8
- Cannot use: Hopper WGMMA, SM120A tcgen05, ClusterTransactionBarrier (Phase 5 already established raw uint64_t* pattern)
- Cannot use: warp count that makes `kNMathWarps * 16 ≠ kBlockM` (would break TiledMma partition)

> **Note on Deterministic Designs**: The warp assignment (warp 0 = load, warps 1–8 = math), warp count (9 warps = 288 threads), and dual mbarrier design are fixed per CLAUDE.md specification. TiledMma must use `get_thread_slice(threadIdx.x - 32)` in math warps to offset warp-1 to M-tile 0.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

**Kernel traits changes (kernel_traits.h):**
```cpp
// Add to Hstu_fwd_kernel_traits_sm120_fp8 (or new derived struct):
static constexpr int kNLoadWarps = 1;
static constexpr int kNMathWarps = kNWarps;  // kNWarps=8 for math
static constexpr int kNThreads = (kNLoadWarps + kNMathWarps) * cutlass::NumThreadsPerWarp;
static constexpr int kLoadWarpIdx = 0;       // warp 0 = load warp (per CLAUDE.md)
// Two mbarriers: load_mbar + math_mbar, 8 bytes each
static constexpr int kSmemMbarSize = 16;
```

**Key TiledMma offset**: math warps are threads 32–287 (warps 1–8). To correctly map these to M-tiles 0–7, math warps must call `tiled_mma.get_thread_slice(threadIdx.x - 32)` (subtract one warp = 32 threads). This ensures warp-1 covers M-tile 0 (rows 0–15), warp-2 covers M-tile 1 (rows 16–31), ..., warp-8 covers M-tile 7 (rows 112–127).

**Kernel function structure:**
```cpp
template <typename Kernel_traits, typename TMA_K_t, typename TMA_Vt_t>
__global__ __launch_bounds__(Kernel_traits::kNThreads)
void hstu_fwd_kernel_sm120_fp8_ws(Hstu_fwd_params_fp8_tma<TMA_K_t, TMA_Vt_t> params) {
    const int warp_idx = threadIdx.x / 32;
    if (warp_idx == 0) {
        // Producer: issue TMA, wait for math to consume
        hstu_load_warp_loop_sm120(params, bidb, bidh, m_block);
    } else {
        // Consumer: wait for TMA, execute QMMA with offset thread slice
        hstu_math_warp_loop_sm120(params, bidb, bidh, m_block);
    }
}
```

**Mbarrier init (in preamble, before divergence):**
```cpp
uint64_t* load_mbar = &smem_mbar[0];  // load→math: TMA done
uint64_t* math_mbar = &smem_mbar[1];  // math→load: SMEM free
if (threadIdx.x == 0) {
    mbarrier_init(load_mbar, 1);   // 1 arrival: load warp via arrive_and_expect_tx
    mbarrier_init(math_mbar, 8);   // 8 arrivals: 1 per math warp (warps 1-8)
}
__syncthreads();
```

**Producer loop (load warp, warp 0):**
```cpp
// Preamble: issue first tile TMA
issue_tma_kv(nb_first);   // K[nb_first] + V^T[nb_first]
arrive_load_mbar_with_tx(load_mbar, K_bytes + Vt_bytes, load_phase);

for (int nb = nb_first; nb >= nb_min; nb--) {
    int nb_next = nb - 1;
    wait(math_mbar, math_phase);    // math warps consumed previous SMEM
    math_phase ^= 1;
    if (nb_next >= nb_min) {
        issue_tma_kv(nb_next);
        arrive_load_mbar_with_tx(load_mbar, K_bytes + Vt_bytes, load_phase);
        load_phase ^= 1;
    } else {
        arrive_load_mbar_no_tx(load_mbar);  // signal: no more tiles
        load_phase ^= 1;
    }
}
```

**Consumer loop (math warps, warps 0–7):**
```cpp
for (int nb = nb_first; nb >= nb_min; nb--) {
    wait(load_mbar, load_phase);    // TMA for K[nb]+V^T[nb] complete
    load_phase ^= 1;
    // load SFV from GMEM (scalar, no TMA needed)
    // execute QMMA: GEMM1(Q,K) + silu + convert to FP8 + GEMM2(P,V^T)
    arrive(math_mbar);              // signal: SMEM can be overwritten
}
```

**Note on last tile for load warp**: When `nb_next < nb_min`, the load warp still needs to signal load_mbar with 0 tx bytes (or the math warps for the final tile will never get the signal). Use `mbarrier.arrive.shared::cta.b64` without `expect_tx` for this case.

### Relevant References
- `fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/hstu_fwd_kernel.h` — Phase 5 TMA implementation (mbarrier PTX patterns, issue_tma_kv lambda, fwd_step_fp8bs lambda)
- `fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/kernel_traits.h` — SmemLayoutK_TMA, SmemLayoutVt_TMA, kSmemMbarSize, kSmemSFSize
- `fbgemm_gpu/experimental/hstu/src/hstu_blackwell/hstu_fwd.py` — SM100 warp-specialized reference (load_warp_id=9, mma_warp_id=8, silu_warp_ids, dual pipeline barriers)
- `6KD_fp8_block_scale/kernels/include/sm120_blockscaled_gemm/sm120_blockscaled_utils.cuh` — SM120BlockScaledBuilder (kNumMathThreads, SmemLayoutSFA/SFB, partition_fragment_*)

## Dependencies and Sequence

### Milestones
1. **Traits and SMEM layout update**: Modify kernel_traits.h to add kNLoadWarps, update kNThreads and kSmemMbarSize for the warp-specialized path
   - Step A: Add `kNLoadWarps=1`, `kLoadWarpIdx=kNWarps`, update `kNThreads`
   - Step B: Change `kSmemMbarSize` from 8 to 16 (two 8-byte mbarriers)
   - Step C: Verify kSmemSize stays within SM120 SMEM limit

2. **New kernel function**: Write `hstu_fwd_kernel_sm120_fp8_ws` with warp divergence
   - Step A: Extract producer (load warp) preamble + loop from current `hstu_compute_attn_1rowblock_sm120` TMA path
   - Step B: Extract consumer (math warp) preamble + loop, adding load_mbar wait
   - Step C: Add math_mbar arrive at end of each math iteration
   - Step D: Add preamble mbarrier init for both load_mbar and math_mbar

3. **Launch template update**: Wire `run_hstu_fwd_sm120_fp8_ws_impl` function (9-warp launch), connect to `run_hstu_fwd_sm120` dispatch

4. **Accuracy validation**: Run sweep_accuracy.py, verify fp8_gt_cos ≥ 0.95

5. **Performance validation**: Run bench_hstu_attn_sm120.py + nsys profile, verify TMA/MMA overlap visible

Milestone 3 depends on Milestones 1–2 being complete. Milestones 4–5 depend on Milestone 3.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Investigate SM120BlockScaledBuilder third parameter and kNumMathThreads derivation to confirm 9-warp (288-thread) compatibility | AC-5 | analyze | - |
| task2 | Update `Hstu_fwd_kernel_traits_sm120_fp8` in kernel_traits.h: add kNLoadWarps, kLoadWarpIdx, update kNThreads and kSmemMbarSize to 16 | AC-1, AC-5 | coding | task1 |
| task3 | Write preamble mbarrier init for load_mbar (1 arrival) and math_mbar (8 arrivals) in `hstu_compute_attn_1rowblock_sm120` TMA path | AC-2, AC-5 | coding | task2 |
| task4 | Implement producer loop in `hstu_fwd_kernel_sm120_fp8_ws`: issue TMA per tile, arrive load_mbar_with_tx, wait math_mbar | AC-2, AC-3 | coding | task3 |
| task5 | Implement consumer loop in `hstu_fwd_kernel_sm120_fp8_ws`: wait load_mbar per tile, execute QMMA (reuse fwd_step_fp8bs logic), arrive math_mbar | AC-2, AC-3 | coding | task3 |
| task6 | Write `run_hstu_fwd_sm120_fp8_ws_impl` launch function (kNThreads=288, correct SMEM size) and wire into `run_hstu_fwd_sm120` FP8 dispatch | AC-1, AC-5 | coding | task4, task5 |
| task7 | Compile and fix any type/layout errors. Run `sweep_accuracy.py` and verify AC-2 | AC-1, AC-2 | coding | task6 |
| task8 | Run `bench_hstu_attn_sm120.py` and `nsys profile`. Verify TMA/MMA overlap in timeline and no BF16 regression | AC-3, AC-4 | coding | task7 |

## Claude-Codex Deliberation

### Agreements
- The only valid math warp count is 8 (since kBlockM=128 = 8 × 16, enforced by TiledMma layout)
- Total thread count must be 9 × 32 = 288 to accommodate 1 load warp + 8 math warps
- Single-stage SMEM pipeline (no double-buffer) is the correct starting point to establish correctness before attempting multi-stage optimization
- Phase 5 TMA mbarrier PTX pattern (uint64_t* + inline PTX) is the right basis for Phase 6 mbarrier implementation
- Performance target is directional (overlap visible in nsys), not a hard percentage threshold

### Resolved Disagreements
- **Load warp index**: CLAUDE.md specifies warp 0 as load warp. This requires: (1) existing `if (warp_id == 0)` preamble logic (Is_arbitrary func loading) must move to a math warp (warp 1); (2) math warps (1–8) must call `get_thread_slice(threadIdx.x - 32)` to offset TiledMma tile assignment and correctly cover M-tiles 0–7. Chosen: warp 0 per CLAUDE.md specification.
- **SFB (K scale) loading**: Claude position = math warps load SFB after load_mbar wait (lower risk, avoids modifying load warp to handle cp.async alongside TMA). Chosen: math warps.

### Convergence Status
- Final Status: `partially_converged`
- Note: Codex CLI available but api.openai.com unreachable in current sandbox. Claude-only analysis. Reduced cross-review confidence. User confirmed to proceed.

## Pending User Decisions

- DEC-1: SM120BlockScaledBuilder 9-warp compatibility + TiledMma thread offset
  - Claude Position: `SM120BlockScaledBuilder<kBlockM, kBlockN, 4>` likely derives kNumMathThreads from M/N tile sizes (256 = kBlockM threads). With math warps 1–8, the MMA calls use `threadIdx.x - 32` as the slice index, so BS1/BS2 builder types are still built with `kNWarps=8` math warps. task1 (analyze) must confirm `kNumMathThreads` derivation and whether the `partition_fragment_*` / `partition_S` APIs accept an offset thread index correctly.
  - Codex Position: N/A - open question (Codex unavailable)
  - Tradeoff Summary: If SM120BlockScaledBuilder hardcodes 256 threads in its tile partition, passing `threadIdx.x - 32` (range 0–255 for math warps) is correct. If it uses `threadIdx.x` directly (range 32–287), the partition breaks. task1 must resolve this.
  - Decision Status: PENDING (to be resolved by task1 analyze result)

- DEC-2: Fallback if warp 0 / threadIdx.x-32 offset causes SM120BlockScaledBuilder incompatibility
  - Claude Position: Fall back to warp 8 as load warp (warps 0-7 as math warps). No threadIdx.x offset needed for TiledMma. Requires changing warp_id==0 preamble logic to be compatible with load warp role (warp 8 does NOT execute the warp_id==0 branch — it only does TMA).
  - Codex Position: N/A - open question (Codex unavailable)
  - Tradeoff Summary: Warp-0-as-load is the CLAUDE.md-specified design; warp-8-as-load is simpler to integrate with existing code. Choose based on task1 findings.
  - Decision Status: PENDING (contingent on DEC-1 outcome)

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", or similar workflow markers
- Use descriptive, domain-appropriate naming in code: `load_warp`, `math_warp`, `kv_load_mbar`, `math_done_mbar`, `producer_loop`, `consumer_loop`
- Kernel function names should follow existing conventions: `hstu_fwd_kernel_sm120_fp8_ws` (ws = warp-specialized)
- Add `// Phase 6` comments sparingly, only at the top-level function/struct to mark the new addition

---

--- Original Design Draft Start ---

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
- [ ] **Phase 6**：Warp-specialized kernel（load warp 专职 TMA，math warps 专职 MMA，producer/consumer pipeline）

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

## Phase 6 任务：前向 Warp-Specialized Kernel（当前目标）

**目标**：将 `hstu_blackwell_sm120/hstu_fwd_kernel.h` 改造为 warp-specialized 前向设计，通过 producer/consumer pipeline 实现 TMA 搬运与 MMA 计算真正重叠。**暂不实现后向。**

### Phase 5 现状 vs Phase 6 目标

**Phase 5 现状**：所有线程 spin-wait mbarrier → 全部线程一起做 MMA。TMA 与 MMA **串行**，无真正 overlap。

**Phase 6 目标**：
- **warp 0（load warp）**：专职发 TMA（K[nb]+V^T[nb]+SF[nb]），等 math warps 消费完毕后发下一个
- **warp 1-7（math warps）**：等待 load warp 完成 TMA，执行 GEMM1+silu+GEMM2，通知 load warp 可以发下一个
- TMA 搬运与 MMA 计算**真正 overlap**

### 参考文件（必读）
- **SM100 前向参考**：`src/hstu_blackwell/hstu_fwd.py`
  - 前向 warp 分工：`load_warp_id=9`（TMA issue）、`mma_warp_id=8`（tcgen05 MMA）、`silu0_warp_ids=(0-3)`、`silu1_warp_ids=(4-7)`，共 12 warps
  - pipeline barrier：`load_mma_Q/K/V_mbar_ptr` 管理 TMA→MMA 同步；`mma_compute_S_mbar_ptr` 管理 MMA→silu 同步
  - 多 stage pipeline：`kv_stage=4`（FP8）/`kv_stage=3`（BF16），`q_stage=2`

### SM120 与 SM100 的关键差异

| 特性 | SM100（参考） | SM120（目标） |
|------|--------------|--------------|
| Tensor Core 指令 | `tcgen05`（全 CTA 共享 TMEM） | `mma.sync`（per-warp，block-scale QMMA） |
| MMA 调用 | 单 mma_warp 代理全 CTA | 所有 math warps 各自执行 QMMA |
| TMA API | 相同（SM90+） | 相同 |
| SF 处理 | 无（非 block-scale） | SFA/SFB/SFV 需随 K/V tile 预取 |
| V transpose | TMA 预转置 GMEM layout | Phase 5 已完成 TMA 预转置 ✅ |

### Phase 6 Warp 分工方案（SM120 适配）

SM120 无 TMEM，mma.sync 是 per-warp 的，因此所有 math warps 都要执行 MMA：

```
warp 0      : load warp（专职 TMA issue：K/V^T + SF）
warp 1-7    : math warps（QMMA block-scale + silu + softmax）
warp 8-X    : 可选 epilogue / empty warps
```

**线程数**：FP8 路径维持 `BS1::kNumMathThreads=256`（8 warps），load warp 额外 +1 → 共 9 warps = 288 threads（待确认与 `SM120BlockScaledBuilder` 的兼容性）。

### Phase 6 核心实现步骤

1. **双 barrier 设计**：
   - `load_mbar`（load→math）：load warp 发完 TMA 后 arrive，math warps wait
   - `math_mbar`（math→load）：math warps 消费完 SMEM 后 arrive，load warp wait

2. **Load warp 职责**：
   ```cpp
   if (warp_idx == 0) {
       // preamble: 发出 K[0]+V^T[0]+SF[0] TMA
       for (nb = n_block_max-1; nb >= 0; nb--) {
           math_mbar.wait(phase);          // 等 math warps 消费完上一 tile
           issue_tma_kv(nb);               // 发 K[nb]+V^T[nb] TMA
           load_mbar.arrive_and_expect_tx(...); // 通知 math warps
       }
   }
   ```

3. **Math warp 职责**：
   ```cpp
   else {
       for (nb = n_block_max-1; nb >= 0; nb--) {
           load_mbar.wait(phase);          // 等 TMA 完成
           // GEMM1: Q×K → acc_s
           // silu + softmax
           // GEMM2: P×V^T → acc_o
           math_mbar.arrive();             // 通知 load warp 可发下一个
       }
   }
   ```

4. **SF prefetch 与 MMA overlap**：SFA/SFB/SFV 在上一轮 MMA 执行期间由 load warp 预取到 SMEM。

### 验证目标
- 编译通过，`sweep_accuracy.py` fp8_gt_cos > 0.95
- bench 性能显著优于 Phase 5（预期：TMA 与 MMA 真正 overlap，消除 GMEM 读延迟气泡）

---

## 核心文件

### SM120 原生内核（当前主目录）
`fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/`
- `hstu_fwd_kernel.h` — 前向内核（BF16 + FP8 block-scale 路径）★核心
- `kernel_traits.h` — BF16 + FP8 内核 traits
- `hstu_ops_gpu.cpp` — PyTorch 入口（`hstu_varlen_fwd_120`）
- `hstu.h` — Params 结构体（含 FP8 descale 字段）
- `utils.h` — tile 大小、类型转换、silu 辅助
- `hstu_fwd_launch_template.h` — 启动模板

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

**运行准确度测试**：
```bash
HSTU_SWEEP_FP8_QUANT_MODE=2 python /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/sweep_accuracy.py \
  2>&1 | tee /home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/test_results/NNN_xxx.log
```

**测试日志命名规范**：文件名格式 `NNN_<修改内容简述>.log`，NNN 为三位数字顺序编号。

**环境版本**：PyTorch 2.10.0（nvcr.io/nvidia/pytorch:26.01-py3）；CUDA 13.1。

---

## 性能分析流程（每轮优化后必须执行）

每次跑完 benchmark 之后，必须按以下步骤做 nsys profile 分析，找到性能瓶颈并制定下一步优化方向。

**文件命名规范**：所有 benchmark 产物（bench log、nsys trace、stats log）统一存放在 `benchmark_results/` 目录，文件名格式与 `test_results/` 相同：`NNN_<内容简述>.<ext>`，NNN 为三位数字顺序编号。

### 步骤 1：运行 benchmark
```bash
REPO=/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu
python ${REPO}/fbgemm_gpu/experimental/hstu/benchmark/bench_hstu_attn_sm120.py \
  2>&1 | tee ${REPO}/benchmark_results/NNN_bench.log
```

### 步骤 2：nsys profile 抓取 trace
```bash
REPO=/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu
TRACE=${REPO}/benchmark_results/NNN_profile
nsys profile \
  --output ${TRACE} \
  --trace cuda,nvtx \
  --force-overwrite true \
  python ${REPO}/fbgemm_gpu/experimental/hstu/benchmark/profile_hstu_attn.py
```

### 步骤 3：nsys stats 分析 trace
```bash
nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_pushpop_sum \
  --force-export true \
  --format csv \
  ${TRACE}.nsys-rep \
  2>&1 | tee ${REPO}/benchmark_results/NNN_stats.log
```

注：nsys 2025.6.1 的 report 名称为 `cuda_gpu_kern_sum,cuda_api_sum,nvtx_pushpop_sum`（非旧版的 `gputrace,cudaapisum,nvtxsum`）。

### 步骤 4：分析瓶颈并制定优化方向

看 `cuda_gpu_kern_sum` 输出，重点关注：
- **HSTU kernel 耗时占比**：是否 compute-bound 还是 memory-bound
- **kernel 内部 stall**：通过 duration 与理论 FLOP/s 对比判断
- **量化 kernel 耗时**：`quant_mode=2` 时 Python 侧量化 kernel 的开销（出现在 CUDA API 调用中）
- **SM 利用率**：kernel duration × SM count vs 理论峰值

分析完成后，根据瓶颈类型制定下一步优化方向：
- 若 compute-bound → 考虑 instruction-level 优化（减少 SF 加载、register spill）
- 若 memory-bound → 考虑增加 prefetch stage 数、TMA multicast
- 若量化开销主导 → 考虑将量化融合进 kernel

---

## 目录结构

```
fbgemm_gpu/experimental/hstu/
├── benchmark/
│   ├── bench_hstu_attn.py               # 吞吐量基准测试（通用）
│   ├── bench_hstu_attn_sm120.py         # SM120 专用 benchmark ★
│   └── profile_hstu_attn.py             # nsys 性能分析脚本
├── hstu/
│   ├── cuda_hstu_attention.py           # Python 入口，SM 版本分发
│   └── library.py                       # 命名空间包检测（已修复）
├── src/
│   ├── generate_kernels.py              # 生成 Hopper/Blackwell .cu 文件
│   ├── hstu_ampere/                     # Ampere (SM80) 原生内核（upstream）
│   ├── hstu_blackwell/                  # SM100 原生内核（upstream，CuTe-DSL 实现）
│   ├── hstu_blackwell_sm120/            # SM120 最终版本（Phase 4，当前主目录）★
│   │   ├── hstu_fwd_kernel.h            # 前向内核（BF16 + FP8 block-scale + K/V 预取）★
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
├── sweep_accuracy.py                    # FP8 vs BF16 准确度扫描 ★
├── memory.md                            # Claude 对话记忆（跨会话）
└── test_results/
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

---

## 语言要求

**任何时候只用中文回答，不要显示韩文或日文。**

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

--- Original Design Draft End ---
