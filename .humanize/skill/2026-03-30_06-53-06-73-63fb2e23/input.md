# Ask Codex Input

## Question

You are a GPU kernel performance engineer with deep expertise in CUDA warp specialization, TMA, and FP8 MMA pipelines.

## Repository Context

This is the HSTU (Hierarchical Sequential Transduction Unit) CUDA attention kernel project targeting NVIDIA Blackwell SM120 GPU (RTX PRO 6000). The codebase implements a custom attention mechanism with FP8 block-scale quantization.

## Current State (Phase 5 - COMPLETE)

The Phase 5 kernel (`hstu_fwd_kernel_sm120_fp8_tma`) works as follows:
- ALL 8 warps (kNWarps=8, kNThreads=256) execute together
- preamble: thread-0 issues TMA for K[nb=n_block_max-1] + V^T[nb] → mbarrier arrive_expect_tx
- ALL threads spin-wait on mbarrier → ALL threads execute QMMA (GEMM1 + silu + GEMM2)
- end of iteration: ALL threads sync → thread-0 issues TMA for next K+V^T → repeat
- RESULT: TMA and MMA are SERIALIZED — no overlap

Key files:
- `fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/hstu_fwd_kernel.h` — main kernel (~1618 lines)
- `fbgemm_gpu/experimental/hstu/src/hstu_blackwell_sm120/kernel_traits.h` — kernel traits (SmemLayout types, kSmemSize)
- `6KD_fp8_block_scale/kernels/include/sm120_blockscaled_gemm/sm120_blockscaled_utils.cuh` — SM120BlockScaledBuilder (FP8 QMMA types)

Key constants for FP8 hdim=128 path:
- kBlockM=128, kBlockN=128, kHeadDim=128, kNWarps=8, kNThreads=256
- SmemLayoutK_TMA, SmemLayoutVt_TMA: SW128-swizzled SMEM layouts for TMA destinations
- kSmemMbarSize=8 (8 bytes reserved at end of SMEM for mbarrier)
- Phase 5 uses 1 mbarrier (load_mbar only) — math→load signaling doesn't exist yet

## Phase 6 Design (from draft CLAUDE.md)

Goal: Warp-specialized producer/consumer pipeline:
- **warp 0 (load warp)**: exclusively issues TMA (K[nb]+V^T[nb]+SF[nb]), waits for math warps to signal "consumed"
- **warps 1-7 (math warps, 7 warps)**: wait for TMA complete, execute QMMA block-scale GEMM1+silu+GEMM2, signal "consumed"
- Two mbarriers: `load_mbar` (load→math: TMA done) + `math_mbar` (math→load: SMEM consumed/free)
- True TMA/MMA overlap: while math warps compute tile[nb], load warp issues TMA for tile[nb-1]

Warp count question: draft says "warp 0 + warps 1-7 = 8 warps = 256 threads" OR "9 warps = 288 threads with optional empty warp". SM100 uses 12 warps (1 load + 1 mma + 8 silu + 2 empty). SM120 uses per-warp mma.sync so all 7 math warps do QMMA independently.

## Request

Analyze this Phase 6 design and provide:

CORE_RISKS: What are the 3 highest-risk assumptions or failure modes?
  - Be specific about SM120 hardware constraints (mbarrier semantics, per-warp mma.sync, SMEM access)

MISSING_REQUIREMENTS: What requirements or edge cases are NOT mentioned in the draft?
  - Consider: preamble synchronization, SFV (V scale factor) loading by load warp vs math warps, sf_q loading, epilogue write conflicts, arbitrary/causal/masking interaction

TECHNICAL_GAPS: What technical gaps exist in the current Phase 5 → Phase 6 transition plan?
  - Specifically: how does the load warp know when math warps have finished consuming SMEM? How does math_mbar get initialized? How many arrivals does each mbarrier expect?

ALTERNATIVE_DIRECTIONS: Are there simpler/safer alternatives to full warp specialization?
  - e.g., double-buffering with 1-step lookahead but without dedicated load warp

QUESTIONS_FOR_USER: What explicit decisions need to be made before implementation?
  - Thread count (8 warps=256 vs 9 warps=288), single-stage vs double-buffer SMEM, SF loading strategy

CANDIDATE_CRITERIA: Suggest 5 concrete acceptance criteria for Phase 6 completion
  - Include both functional (accuracy) and performance (kernel speedup vs Phase 5) criteria

## Configuration

- Model: gpt-5.4
- Effort: high
- Timeout: 3600s
- Timestamp: 2026-03-30_06-53-06
