/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SM120 (Blackwell consumer) kernel traits for HSTU forward pass.
// Uses per-warp mma.sync (same model as Ampere), NOT warpgroup MMA (Hopper).
// Provides BF16 and FP8 MMA atom selection.

#pragma once

#include <cutlass/numeric_types.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"

// SM89 FP8 MMA: mma.m16n8k32.e4m3.e4m3.f32 (supported on SM89+, including SM120 consumer)
// Note: SM120 consumer (RTX Pro) uses SM89-style FP8 MMA (works on __CUDA_ARCH__ >= 890).
//       SM120A (datacenter B100/B200) supports a different 'kind::f8f6f4' variant,
//       but that variant is NOT available on SM120 consumer hardware.
#include "cute/arch/mma_sm89.hpp"
#include "cute/atom/mma_traits_sm89.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"  // for GMMA::Layout_K_SW128_Atom (Phase 5 TMA)

using namespace cute;

// Base kernel traits for SM120.
// For BF16: uses SM80 mma.sync atom (available on SM120 via backward compat).
// For FP8: uses SM120 f8f6f4 mma.sync atom when CUTE_ARCH_F8F6F4_MMA_ENABLED.
template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNWarps_,
    typename elem_type = cutlass::bfloat16_t>
struct Flash_kernel_traits_sm120 {
  using Element = elem_type;
  using ElementAccum = float;
  using index_t = int64_t;

  using MMA_Atom_Arch = std::conditional_t<
      std::is_same_v<elem_type, cutlass::half_t>,
      MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
};

// SM120 BF16 forward kernel traits — identical structure to Ampere.
// Compiled with -gencode arch=compute_120,code=sm_120 for native Blackwell execution.
template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNWarps_,
    bool Is_causal_,
    bool Is_target_,
    bool Is_context_,
    bool Is_local_,
    bool Is_arbitrary_,
    int kNFunc_,
    bool Has_rab_,
    bool Is_Q_in_regs_ = false,
    bool Share_Q_K_smem_ = false,
    typename elem_type = cutlass::bfloat16_t,
    typename Base =
        Flash_kernel_traits_sm120<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type>>
struct Hstu_fwd_kernel_traits_sm120 : public Base {
  static constexpr bool Is_causal = Is_causal_;
  static constexpr bool Is_target = Is_target_;
  static constexpr bool Is_context = Is_context_;
  static constexpr bool Is_local = Is_local_;
  static constexpr bool Is_arbitrary = Is_arbitrary_;
  static constexpr int kNFunc = Is_arbitrary_ ? kNFunc_ : 0;
  static constexpr bool Has_rab = Has_rab_;
  static constexpr bool Paged_KV = false;  // Not supported in first Blackwell impl
  static constexpr bool Is_fp8 = false;

  using Element = typename Base::Element;
  // For BF16 path: ElementSmem == Element (no conversion needed)
  using ElementSmem = Element;
  // For BF16 path: OutputType == Element (output same dtype as input)
  using OutputType = Element;
  using ElementAccum = typename Base::ElementAccum;
  using index_t = typename Base::index_t;
  using SmemCopyAtom = typename Base::SmemCopyAtom;
  using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

  static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
  static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static_assert(kHeadDim % 32 == 0);
  static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
  static constexpr int kBlockKSmemRab = kBlockN % 64 == 0 ? 64 : 32;
  static constexpr int kBlockKGmem =
      kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
  static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
  static constexpr int kSwizzleRab = kBlockKSmemRab == 32 ? 2 : 3;
  static constexpr int kStages = 1;

  using TiledMma = TiledMMA<
      typename Base::MMA_Atom_Arch,
      Layout<Shape<Int<kNWarps>, _1, _1>>,
      Tile<Int<16 * kNWarps>, _16, _16>>;
  static_assert(16 * kNWarps <= kBlockM);

  using SmemLayoutAtomQ = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

  using SmemLayoutAtomRab = decltype(composition(
      Swizzle<kSwizzleRab, 3, 3>{},
      Layout<Shape<_8, Int<kBlockKSmemRab>>, Stride<Int<kBlockKSmemRab>, _1>>{}));

  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtomQ{},
      Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutRab = decltype(tile_to_shape(
      SmemLayoutAtomRab{},
      Shape<Int<kBlockM>, Int<kBlockN>, Int<kStages>>{}));
  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtomQ{},
      Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages>>{}));

  using SmemLayoutVtransposed = decltype(composition(
      SmemLayoutKV{},
      make_layout(Shape<Int<kHeadDim>, Int<kBlockN>, Int<kStages>>{},
          Stride<Int<kBlockN>, _1, Int<kHeadDim * kBlockN>>{})));
  using SmemLayoutVtransposedNoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

  using SmemLayoutAtomO = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutO = decltype(tile_to_shape(
      SmemLayoutAtomO{},
      Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopy, Element>;

  static constexpr int MaxSeqLenK = 64 * 1024;
  static constexpr int MaxValidBlock = MaxSeqLenK / kBlockN;
  using SmemLayoutValidBlockIds = Layout<Shape<Int<MaxValidBlock>>, Stride<_1>>;
  using SmemLayoutMaxFunc = Layout<Shape<Int<kNFunc/2 + 1>>, Stride<_1>>;
  using SmemLayoutMinFunc = Layout<Shape<Int<kNFunc/2 + 1>>, Stride<_1>>;

  static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
  static constexpr int kSmemRabSize =
      Has_rab ? size(SmemLayoutRab{}) * sizeof(Element) : 0;
  static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
  static constexpr int kSmemSizeQKV = Share_Q_K_smem
      ? std::max(kSmemQSize, kSmemKVSize)
      : kSmemQSize + kSmemKVSize;
  static constexpr int kSmemSizeQKVRab = kSmemSizeQKV + kSmemRabSize;
  static constexpr int kSmemSizeQKVRabValidBlockIds = kSmemSizeQKVRab +
      (Is_arbitrary ? size(SmemLayoutValidBlockIds{}) * sizeof(int) : 0);
  static constexpr int kSmemSize = kSmemSizeQKVRabValidBlockIds +
      (Is_arbitrary ? (size(SmemLayoutMaxFunc{}) + size(SmemLayoutMinFunc{}) + 1) * sizeof(int) : 0);

  static constexpr int kGmemElemsPerLoad =
      sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(kHeadDim % kGmemElemsPerLoad == 0);
  static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
  static constexpr int kGmemThreadsPerRowRab = kBlockKSmemRab / kGmemElemsPerLoad;
  static_assert(kNThreads % kGmemThreadsPerRow == 0);
  static_assert(kNThreads % kGmemThreadsPerRowRab == 0);

  using GmemLayoutAtom = Layout<
      Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemLayoutAtomRab = Layout<
      Shape<Int<kNThreads / kGmemThreadsPerRowRab>, Int<kGmemThreadsPerRowRab>>,
      Stride<Int<kGmemThreadsPerRowRab>, _1>>;

  using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using GmemTiledCopyQKV = decltype(make_tiled_copy(
      Copy_Atom<Gmem_copy_struct, Element>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, _8>>{}));

  // rab_row_size: M-values per thread per atom call; must cover exactly kBlockM rows total.
  static constexpr int rab_row_size = Has_rab
      ? kBlockM * kGmemThreadsPerRowRab / kNThreads
      : 1;
  using GmemTiledCopyRab = decltype(make_tiled_copy(
      Copy_Atom<Gmem_copy_struct, Element>{},
      GmemLayoutAtomRab{},
      Layout<Shape<Int<rab_row_size>, _8>, Stride<_8, _1>>{}));

  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, Element>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, _8>>{}));
};

// SM120 FP8 forward kernel traits (Phase 2).
// Q, K, V are FP8 (e4m3) in both GMEM and SMEM.
// GEMM1 (Q×K): uses SM89_16x8x32_F32E4M3E4M3F32_TN (mma.m16n8k32, K=32 per step).
//   SM120 consumer supports SM89-style FP8 but NOT SM120A 'kind::f8f6f4' variant.
// GEMM2 (P×V): uses the same FP8 MMA after converting float acc_s → FP8.
// Descale factors applied after GEMM1 (S *= descale_q * descale_k) and epilogue (O *= descale_v).
// FP8 SMEM halves SMEM usage vs BF16 → allows kNWarps=8 (vs Phase 1 kNWarps=4).
template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNWarps_,
    bool Is_causal_,
    bool Is_target_,
    bool Is_context_,
    bool Is_local_,
    bool Is_arbitrary_,
    int kNFunc_,
    bool Has_rab_,
    bool Is_Q_in_regs_ = false,
    bool Share_Q_K_smem_ = false,
    typename out_type = cutlass::bfloat16_t>
struct Hstu_fwd_kernel_traits_sm120_fp8 {
  // FP8 input type (Q, K, V in GMEM and SMEM)
  using Element = cutlass::float_e4m3_t;
  // Phase 2: FP8 stays in SMEM (no BF16 conversion)
  using ElementSmem = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using OutputType = out_type;
  using index_t = int64_t;

  static constexpr bool Is_causal = Is_causal_;
  static constexpr bool Is_target = Is_target_;
  static constexpr bool Is_context = Is_context_;
  static constexpr bool Is_local = Is_local_;
  static constexpr bool Is_arbitrary = Is_arbitrary_;
  static constexpr int kNFunc = Is_arbitrary_ ? kNFunc_ : 0;
  static constexpr bool Has_rab = Has_rab_;
  static constexpr bool Paged_KV = false;
  static constexpr bool Is_fp8 = true;

  // FP8 MMA: M16×N8×K32, FP8 inputs, FP32 accumulator.
  // Uses SM89_16x8x32_F32E4M3E4M3F32_TN which is supported on all SM89+ including SM120.
  // SM120 consumer (RTX Pro) does NOT support the SM120A 'kind::f8f6f4' variant.
  // SM89 FP8 MMA uses: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
  using MMA_Atom_Arch = MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>;

  // For FP8 SMEM→regs: use DefaultCopy (element-by-element) for both Q/K and V^T.
  // SM75_U32x4_LDSM_N uses ldmatrix.x4 in b16-mode (16-bit words). With FP8 (8-bit elements),
  // two FP8 are packed per b16 slot, but the resulting register arrangement does NOT match what
  // SM89_16x8x32_F32E4M3E4M3F32_TN expects for the A operand → wrong GEMM1 → cos_sim ≈ 0.36.
  // DefaultCopy loads elements one-by-one, always producing the correct per-thread register
  // layout as dictated by make_tiled_copy_A/B + TiledMma, at the cost of vectorization.
  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
  // For V-transposed (B operand of GEMM2): same reasoning — use DefaultCopy.
  using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, Element>;

  static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
  static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static_assert(kHeadDim % 32 == 0);
  // FP8 SMEM: flat 16×32 atom, aligned with one MMA A-tile (m16×k32).
  // kBlockKSmem=32 ensures each MMA K-step sub-tile is self-contained: the 32-column atom
  // maps exactly to the k=32 dimension of SM89_16x8x32 FP8 MMA. With kBlockKSmem=64
  // (8×64 atom), one 16×32 MMA A-tile spans two atom rows causing incorrect element
  // mapping when DefaultCopy partitions the SMEM via make_tiled_copy_A + TiledMma.
  static constexpr int kBlockKSmem = 32;
  static constexpr int kBlockKSmemRab = kBlockN % 64 == 0 ? 64 : 32;
  static constexpr int kBlockKGmem =
      kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
  static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
  // RAB is always BF16: use 64-BF16 strip with Swizzle<3,3,3> (8 BF16 contiguous = 128 bits)
  static constexpr int kSwizzleRab = kBlockKSmemRab == 32 ? 2 : 3;
  static constexpr int kStages = 1;

  // TiledMma: K=32 per step for FP8 MMA (vs K=16 for BF16).
  using TiledMma = TiledMMA<
      MMA_Atom_Arch,
      Layout<Shape<Int<kNWarps>, _1, _1>>,
      Tile<Int<16 * kNWarps>, _16, _32>>;
  static_assert(16 * kNWarps <= kBlockM);

  // SMEM layout for Q/K/V: flat 16×32 atom, aligned with one SM89_16x8x32 MMA A-tile.
  // A single 16-row × 32-col FP8 tile is 512 bytes; with 32 threads per warp each loading
  // 16 bytes (128 bits), the warp covers the tile at stride T×16 as expected by DefaultCopy.
  // The 8×64 atom (kBlockKSmem=64) spans TWO atom rows per MMA tile, breaking this alignment
  // and causing incorrect element assignment during make_tiled_copy_A partitioning.
  using SmemLayoutAtomQ =
      Layout<Shape<_16, _32>, Stride<_32, _1>>;

  using SmemLayoutAtomRab = decltype(composition(
      Swizzle<kSwizzleRab, 3, 3>{},
      Layout<Shape<_8, Int<kBlockKSmemRab>>, Stride<Int<kBlockKSmemRab>, _1>>{}));

  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtomQ{},
      Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutRab = decltype(tile_to_shape(
      SmemLayoutAtomRab{},
      Shape<Int<kBlockM>, Int<kBlockN>, Int<kStages>>{}));
  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtomQ{},
      Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages>>{}));

  using SmemLayoutVtransposed = decltype(composition(
      SmemLayoutKV{},
      make_layout(Shape<Int<kHeadDim>, Int<kBlockN>, Int<kStages>>{},
          Stride<Int<kBlockN>, _1, Int<kHeadDim * kBlockN>>{})));
  using SmemLayoutVtransposedNoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

  // Phase 5 TMA layouts: SW128 swizzle (= SM120BlockScaledBuilder::SmemLayoutAtomA/B).
  // Used for make_tma_copy in run_hstu_fwd_sm120_impl (host-side) and for the
  // Phase 5 compute function (device-side).  These match the local SW128 types
  // defined inside hstu_compute_attn_1rowblock_sm120 via BS1/BS2::SmemLayoutAtomA/B.
  using SmemLayoutAtomSW128 = GMMA::Layout_K_SW128_Atom<Element>;  // 8-row × 128-FP8 atom
  using SmemLayoutQ_TMA  = decltype(tile_to_shape(SmemLayoutAtomSW128{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutK_TMA  = decltype(tile_to_shape(SmemLayoutAtomSW128{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
  // V^T [kHeadDim, kBlockN] in SMEM, loaded via TMA from GMEM [d, total_k] (stride-1 on d=kHeadDim axis).
  // TMA requires SMEM inner dimension → GMEM stride-1 dimension.
  // V^T: N=kHeadDim (d, stride 1) is fast → use MN_SW128 (N/kHeadDim inner) so TMA dim0=kHeadDim→d (stride=1). ✓
  // LDSM_N (K inner) cannot be used with MN_SW128; use partition_B + cute::copy (element-by-element) instead.
  using SmemLayoutVt_TMA = decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<Element>{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));

  // Output layout: BF16 written to sO (reuses smem_ base), then copied to GMEM.
  // SM120 QMMA output uses PermMmaTileN = Layout<_8,_4,_4>, Stride<_1,_32,_8>, which
  // permutes the N-axis of the C fragment. The interleaved atom below matches this permutation:
  // within each 8-element group (cols 0..7, 8..15, etc.), elements are consecutive in physical
  // SMEM (stride-1 within group, stride-64 between groups). This enables AutoVectorizingCopy
  // to perform 8-element (128-bit) stores correctly.
  // Reference: sm120_blockscaled_moe_gemm_impl.cuh SmemAtomLayoutO.
  // NOTE: kBlockKSmem=32 is for FP8 INPUT; output is BF16 with 64-wide SW128 atom.
  using SmemLayoutAtomO = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, Int<64>>>>{}));
  using SmemLayoutO = decltype(tile_to_shape(
      SmemLayoutAtomO{},
      Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopy, OutputType>;

  static constexpr int MaxSeqLenK = 64 * 1024;
  static constexpr int MaxValidBlock = MaxSeqLenK / kBlockN;
  using SmemLayoutValidBlockIds = Layout<Shape<Int<MaxValidBlock>>, Stride<_1>>;
  using SmemLayoutMaxFunc = Layout<Shape<Int<kNFunc/2 + 1>>, Stride<_1>>;
  using SmemLayoutMinFunc = Layout<Shape<Int<kNFunc/2 + 1>>, Stride<_1>>;

  // SMEM sizes: FP8 is 1 byte each (half of BF16), allowing kNWarps=8
  static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
  static constexpr int kSmemRabSize =
      Has_rab ? size(SmemLayoutRab{}) * sizeof(cutlass::bfloat16_t) : 0;  // RAB is always BF16
  static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
  // FP8 epilogue writes BF16 output into sO which reuses smem_ base.
  // sO needs size(SmemLayoutO) * sizeof(OutputType) bytes (BF16 = 2x FP8).
  // kSmemSizeQKV must be >= kSmemOSize to avoid OOB SMEM access in epilogue.
  static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(OutputType);
  // Phase 3 single-buffer V layout (when Share_Q_K_smem=true):
  //   [0, kSmemQSize)       Q = K  (shared, single buffer)
  //   [kSmemQSize, 2x)      V (single buffer)
  // kSmemOSize (BF16 epilogue) = kBlockM*kHeadDim*2 = 128*128*2 = 32768 = 2*kSmemQSize,
  // so output can reuse smem_q safely.
  static constexpr int kSmemSizeQKV = Share_Q_K_smem
      ? std::max(2 * kSmemQSize, kSmemOSize)   // Q=K(0) + V(1), single-buf V
      : std::max(kSmemQSize + kSmemKVSize, kSmemOSize);
  static constexpr int kSmemSizeQKVRab = kSmemSizeQKV + kSmemRabSize;
  static constexpr int kSmemSizeQKVRabValidBlockIds = kSmemSizeQKVRab +
      (Is_arbitrary ? size(SmemLayoutValidBlockIds{}) * sizeof(int) : 0);
  // Extra SMEM for block-scale SF: SFA (kBlockM int32 = 512B) + SFB (kBlockN int32 = 512B)
  static constexpr int kSmemSFSize = 1024;
  // Extra SMEM for TMA barrier(s).
  // Phase 5: 1 barrier (load_mbar) = 8 bytes.
  // Phase 6 WS: 2 barriers (load_mbar + math_mbar) = 16 bytes.
  static constexpr int kSmemMbarSize = 8;
  static constexpr int kSmemSize = kSmemSizeQKVRabValidBlockIds +
      (Is_arbitrary ? (size(SmemLayoutMaxFunc{}) + size(SmemLayoutMinFunc{}) + 1) * sizeof(int) : 0) +
      kSmemSFSize + kSmemMbarSize;

  // GMEM copy: cp.async FP8 elements directly into flat SMEM (no conversion).
  // Uses SM80_CP_ASYNC_CACHEGLOBAL<uint128_t> (16 bytes = 16 FP8 per thread per load).
  // Flat SMEM layout has stride-1 in K → 16 consecutive FP8 are physically contiguous. ✓
  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);  // = 16 for FP8
  static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;  // = 32/16 = 2 for FP8
  // RAB is always BF16: 8 BF16 per uint128_t (16 bytes)
  static constexpr int kGmemElemsPerLoadRab = sizeof(cute::uint128_t) / sizeof(cutlass::bfloat16_t);
  static constexpr int kGmemThreadsPerRowRab = kBlockKSmemRab / kGmemElemsPerLoadRab;
  static_assert(kHeadDim % kGmemElemsPerLoad == 0);
  static_assert(kNThreads % kGmemThreadsPerRow == 0);
  static_assert(kNThreads % kGmemThreadsPerRowRab == 0);

  using GmemLayoutAtom = Layout<
      Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemLayoutAtomRab = Layout<
      Shape<Int<kNThreads / kGmemThreadsPerRowRab>, Int<kGmemThreadsPerRowRab>>,
      Stride<Int<kGmemThreadsPerRowRab>, _1>>;

  // cp.async for FP8 Q/K/V: loads 16 FP8 (128-bit) directly into flat FP8 SMEM per thread.
  // CACHEGLOBAL<uint128_t> issues `cp.async.cg` (L1-bypass), accepts 16-byte transfers.
  using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using GmemTiledCopyQKV = decltype(make_tiled_copy(
      Copy_Atom<Gmem_copy_struct, Element>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // 16 FP8 per thread

  // RAB is BF16 (not quantized)
  using Gmem_copy_struct_rab = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  // rab_row_size: M-values per thread per atom call in the RAB copy.
  // Must satisfy: (kNThreads / kGmemThreadsPerRowRab) * rab_row_size == kBlockM
  // so the tiled copy covers exactly [kBlockM, kBlockN] BF16 elements without
  // threads writing beyond kBlockM rows (which would corrupt SMEM past sRab).
  // Formula: kBlockM * kGmemThreadsPerRowRab / kNThreads
  // For kBlockM=128, kNThreads=256, kGmemThreadsPerRowRab=8: 128*8/256 = 4. ✓
  static constexpr int rab_row_size = Has_rab
      ? kBlockM * kGmemThreadsPerRowRab / kNThreads
      : 1;
  using GmemTiledCopyRab = decltype(make_tiled_copy(
      Copy_Atom<Gmem_copy_struct_rab, cutlass::bfloat16_t>{},
      GmemLayoutAtomRab{},
      Layout<Shape<Int<rab_row_size>, _8>, Stride<_8, _1>>{}));

  // Output copy (BF16 or FP16): uses OutputType-sized loads (not FP8-sized).
  // SmemLayoutAtomO has 64-wide BF16 row (128-byte SW128 atom).
  // kGmemThreadsPerRowO must be 64/8=8 for BF16 output — NOT kBlockKSmem=32 (FP8 input).
  static constexpr int kGmemElemsPerLoadO =
      sizeof(cute::uint128_t) / sizeof(OutputType);  // = 8 for BF16
  static constexpr int kGmemThreadsPerRowO = 64 / kGmemElemsPerLoadO;  // = 8
  using GmemLayoutAtomO = Layout<
      Shape<Int<kNThreads / kGmemThreadsPerRowO>, Int<kGmemThreadsPerRowO>>,
      Stride<Int<kGmemThreadsPerRowO>, _1>>;
  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, OutputType>{},
      GmemLayoutAtomO{},
      Layout<Shape<_1, Int<kGmemElemsPerLoadO>>>{}));
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Phase 6 warp-specialized FP8 kernel traits.
// Extends the Phase 5 FP8 traits by designating warp 0 as a dedicated load warp (TMA-only)
// and warps 1–8 as math warps (QMMA-only).
// kNThreads = 288 (9 warps × 32), kNMathThreads = 256 (8 math warps × 32).
// Math warps use tidx_math = tidx - 32 ∈ [0, 255].
// All GMEM/SMEM layout arithmetic (SmemLayoutQ, GmemTiledCopyQKV, etc.) is inherited from
// the base FP8 traits where kNThreads=256, so those layouts remain correct for 256 math threads.
template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNWarps_,   // number of MATH warps (= 8 for kBlockM=128)
    bool Is_causal_,
    bool Is_target_,
    bool Is_context_,
    bool Is_local_,
    bool Is_arbitrary_,
    int kNFunc_,
    bool Has_rab_,
    bool Is_Q_in_regs_ = false,
    bool Share_Q_K_smem_ = false,
    typename out_type = cutlass::bfloat16_t>
struct Hstu_fwd_kernel_traits_sm120_fp8_ws
    : public Hstu_fwd_kernel_traits_sm120_fp8<
          kHeadDim_, kBlockM_, kBlockN_, kNWarps_,
          Is_causal_, Is_target_, Is_context_, Is_local_, Is_arbitrary_, kNFunc_, Has_rab_,
          Is_Q_in_regs_, Share_Q_K_smem_, out_type> {
  using Base = Hstu_fwd_kernel_traits_sm120_fp8<
      kHeadDim_, kBlockM_, kBlockN_, kNWarps_,
      Is_causal_, Is_target_, Is_context_, Is_local_, Is_arbitrary_, kNFunc_, Has_rab_,
      Is_Q_in_regs_, Share_Q_K_smem_, out_type>;

  // Warp roles: warps 0..kNWarps_-1 = math warps; warp kNWarps_ = load warp.
  static constexpr int kNMathWarps   = kNWarps_;     // 8 math warps
  static constexpr int kNLoadWarps   = 1;            // 1 load warp
  static constexpr int kLoadWarpIdx  = kNWarps_;     // warp 8 is the load warp

  // kNThreads overrides Base::kNThreads: total = math warps + load warp = 9 × 32 = 288.
  static constexpr int kNThreads     = (kNMathWarps + kNLoadWarps) * cutlass::NumThreadsPerWarp;  // 288
  // kNMathThreads: math-warp-only thread count used for per-warp layout arithmetic.
  static constexpr int kNMathThreads = kNMathWarps * cutlass::NumThreadsPerWarp;  // 256

  // Double-buffer pipeline: 4 barriers (2 per role), no load_mbar needed.
  //   tma_mbar[0], tma_mbar[1]: TMA completion per stage (math warps wait these)
  //   math_mbar[0], math_mbar[1]: SMEM consumed per stage (load warp waits these)
  // Each barrier is 8 bytes; total = 32 bytes.  Placed at the last 32 bytes of kSmemSize.
  static constexpr int kSmemMbarSize = 32;

  // Double-buffer KV SMEM layout.
  // Each K or Vt tile is kBlockN * kHeadDim FP8 bytes (1 byte each).
  static constexpr int kSmemKVBytes =
      Base::kBlockN * Base::kHeadDim * (int)sizeof(typename Base::Element);
  // 2 stages × (K + Vt) = 4 × kSmemKVBytes (e.g. 4 × 16384 = 65536 for kBlockN=kHeadDim=128)
  static constexpr int kSmemWsKVTotalBytes = 4 * kSmemKVBytes;

  // WS SMEM region offsets (double-buffer layout):
  //   [0 .. kSmemWsKVTotalBytes)          : K[0], Vt[0], K[1], Vt[1]
  //   [kSmemWsKVTotalBytes .. +ValidBl)   : ValidBlockIds (Is_arbitrary only)
  //   [.. + func region)                  : func arrays (Is_arbitrary only)
  //   [padded to 8B)                      : SFA(512B) + SFB(512B) = 1024B
  //   [last kSmemMbarSize bytes)          : 4 mbarriers × 8B
  static constexpr int kSmemWsValidBlockIdsOffset = kSmemWsKVTotalBytes;
  static constexpr int kSmemWsFuncOffset = kSmemWsValidBlockIdsOffset +
      (Is_arbitrary_ ? (int)(size(typename Base::SmemLayoutValidBlockIds{}) * sizeof(int)) : 0);
  // Arbitrary func data: sn_valid_block_max (1 int) + sFunc_min (kNFunc/2+1 ints) + sFunc_max (kNFunc/2+1 ints)
  static constexpr int kSmemWsFuncEnd = kSmemWsFuncOffset +
      (Is_arbitrary_ ? (int)((Base::kNFunc/2 + 1 + Base::kNFunc/2 + 1 + 1) * (int)sizeof(int)) : 0);

  // Override kSmemSize: WS data (padded to 128B for TMA alignment) + SF + 4 mbarriers.
  // TMA destination SMEM address must be 128-byte aligned.
  static constexpr int kSmemWsDataSizePadded = ((kSmemWsFuncEnd + 127) / 128) * 128;
  // WS SF SMEM: SFA(512B) + SFB[0](512B) + SFB[1](512B) + SFV[0](512B) + SFV[1](512B) = 2560B.
  // SFB and SFV are both double-buffered, each loaded by TMA in load warp.
  static constexpr int kSmemWsSFSize = Base::kSmemSFSize + 3 * Base::kBlockN * (int)sizeof(int32_t);
  static constexpr int kSmemSize = kSmemWsDataSizePadded + kSmemWsSFSize + kSmemMbarSize;

  // Invariant checks
  static_assert(kNMathWarps * 16 == Base::kBlockM,
      "kNMathWarps * 16 must equal kBlockM (8 warps × 16 rows = 128)");
  static_assert(kNThreads == (kNMathWarps + kNLoadWarps) * 32,
      "kNThreads == (kNMathWarps + kNLoadWarps) * 32");
  static_assert(kSmemMbarSize == 32,
      "kSmemMbarSize must be 32 (four 8-byte mbarriers: tma_mbar[2] + math_mbar[2])");
};
