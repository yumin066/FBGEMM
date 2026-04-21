/*
 * Copyright (c) 2023, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SM120 (Blackwell consumer) HSTU forward kernel.
// Adapted from Ampere (SM80) kernel.  Supports BF16 and FP8 (quant_mode=0).
//
// FP8 path (quant_mode=0, Phase 2):
//   Q, K, V read from GMEM as FP8 (e4m3), stored directly as FP8 in SMEM.
//   GEMM1 (Q×K): SM120_16x8x32_TN native FP8 MMA, K=32 per step.
//   After GEMM1: S *= descale_q * descale_k  (dequantize)
//   GEMM2 (P×V): float acc_s → FP8 (clamped), then FP8 MMA.
//   After GEMM2: O *= descale_v / scaling_seqlen (dequantize + normalize)
//   Uses kNWarps=8 (halved SMEM footprint enables double the warps).

#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "hstu.h"

#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>

// Force SM120-specific TMA path in cute/arch/copy_sm90_tma.hpp.
// SM90_TMA_LOAD_xD::copy has two paths:
//   - CUTE_ARCH_TMA_SM120_ENABLED defined → cp.async.bulk.tensor.Nd.shared::cta  (SM120 correct)
//   - otherwise                            → cp.async.bulk.tensor.Nd.shared::cluster (SM90 clusters, illegal on SM120 consumer)
// CUTE_ARCH_TMA_SM120_ENABLED is normally set by cute/arch/config.hpp when CUTLASS_ARCH_MMA_SM120_ENABLED
// is defined (i.e., when __CUDA_ARCH__==1200). Due to include ordering, it may not be set before
// SM90_TMA_LOAD_xD::copy is compiled. Force it here for device code only (host doesn't need it).
// Note: CUTE_ARCH_TMA_SM90_ENABLED is NOT forced here — it's __device__-only and must come
// from the normal config path to avoid calling synclog_emit_tma_load from __host__.
#if defined(__CUDA_ARCH__)
#ifndef CUTE_ARCH_TMA_SM120_ENABLED
#define CUTE_ARCH_TMA_SM120_ENABLED
#endif
#endif

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>  // SM90_TMA_LOAD (also available on SM120)

#include "block_info.h"
#include "kernel_traits.h"
#include "static_switch.h"
#include "sm120_qmma_builder.h"
#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Main HSTU forward attention kernel for SM120.
// Supports both BF16 (Is_fp8=false) and FP8 (Is_fp8=true).
template <typename Kernel_traits, typename Params>
inline __device__ void hstu_compute_attn_1rowblock_sm120(
    const Params& params,
    const int bidb,
    const int bidh,
    int m_block) {
  using ElementSmem = std::conditional_t<
      Kernel_traits::Is_fp8,
      typename Kernel_traits::ElementSmem,
      typename Kernel_traits::Element>;
  // RAB is always BF16 (never quantized), even in FP8 mode.
  using RabElement = std::conditional_t<
      Kernel_traits::Is_fp8,
      cutlass::bfloat16_t,
      ElementSmem>;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;

  extern __shared__ char smem_[];

  const auto tidx = threadIdx.x;
  constexpr bool Is_causal = Kernel_traits::Is_causal;
  constexpr bool Is_target = Kernel_traits::Is_target;
  constexpr bool Is_context = Kernel_traits::Is_context;
  constexpr bool Is_arbitrary = Kernel_traits::Is_arbitrary;
  constexpr int kNFunc = Kernel_traits::kNFunc;
  constexpr bool Is_local = Kernel_traits::Is_local;
  constexpr bool Has_rab = Kernel_traits::Has_rab;
  constexpr bool Is_fp8 = Kernel_traits::Is_fp8;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

  const HstuBlockInfo<Kernel_traits, Params> binfo(params, bidb);
  if (m_block * kBlockM >= binfo.actual_seqlen_q_padded) {
    return;
  }

  char* smem_q = reinterpret_cast<char*>(smem_);
  char* smem_func = reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemSizeQKVRabValidBlockIds;

  int* sn_valid_block_max = reinterpret_cast<int*>(smem_func);
  int* sf_min = reinterpret_cast<int*>(sn_valid_block_max) + 1;
  int* sf_max = reinterpret_cast<int*>(sf_min) + (kNFunc/2 + 1);

  const int actual_seqlen_q = binfo.actual_seqlen_q;
  const int actual_seqlen_k = binfo.actual_seqlen_k;
  const int actual_seqlen_q_padded = binfo.actual_seqlen_q_padded;
  const int actual_seqlen_t = Is_target ? binfo.actual_seqlen_t : 0;
  const int actual_seqlen_c = Is_context ? binfo.actual_seqlen_c : 0;
  const int actual_seqlen_h =
      Is_target ? actual_seqlen_k - actual_seqlen_t : actual_seqlen_k;
  const int actual_seqlen_offset = actual_seqlen_k - actual_seqlen_q;

  const bool is_jump =
      Is_target && m_block * kBlockM + actual_seqlen_offset > actual_seqlen_h;
  const bool is_in_target =
      Is_target && (m_block + 1) * kBlockM + actual_seqlen_offset > actual_seqlen_h;
  (void)is_in_target;
  const bool is_in_context =
      Is_context && (m_block + 1) * kBlockM <= actual_seqlen_c;
  const bool is_in_mixed_context = Is_context &&
      (m_block + 1) * kBlockM > actual_seqlen_c &&
      m_block * kBlockM < actual_seqlen_c;

  const int n_block_history = cute::ceil_div(actual_seqlen_h, kBlockN);
  const int target_index =
      (m_block * kBlockM - actual_seqlen_h) / params.target_group_size;
  const int n_block_target = cute::ceil_div(actual_seqlen_t, kBlockN);

  int n_block_min = !Is_local ? 0
                              : std::max(
                                    0,
                                    (m_block * kBlockM + actual_seqlen_offset -
                                     params.window_size_left) /
                                        kBlockN);
  int n_block_max = cute::ceil_div(actual_seqlen_k, kBlockN);
  if constexpr (Is_causal || Is_local) {
    int offset = (m_block + 1) * kBlockM + actual_seqlen_offset + params.window_size_right;
    n_block_max = std::min(n_block_max, cute::ceil_div(offset, kBlockN));
  }
  if constexpr (Is_context) {
    n_block_min = (is_in_context || is_in_mixed_context) ? 0 : n_block_min;
    n_block_max = (is_in_context || is_in_mixed_context)
        ? std::max(n_block_history, n_block_max)
        : n_block_max;
  }

  int n_masking_block_max = cute::ceil_div(
      std::min(actual_seqlen_k, (m_block + 1) * kBlockM + actual_seqlen_offset),
      kBlockN);
  int n_masking_block_min =
      (m_block * kBlockM + actual_seqlen_offset) / kBlockN;
  if constexpr (Is_target) {
    n_masking_block_min = is_jump ? (actual_seqlen_h + actual_seqlen_offset +
                                     target_index * params.target_group_size) /
            kBlockN
                                  : n_masking_block_min;
  }
  if constexpr (Is_context) {
    n_masking_block_min = is_in_mixed_context ? n_block_min : n_masking_block_min;
    n_masking_block_max = is_in_mixed_context ? n_block_max : n_masking_block_max;
  }

  const int n_masking_steps = (!Is_causal || is_in_context)
      ? 0
      : n_masking_block_max - n_masking_block_min;

  // arbitrary func
  Tensor mMaxFunc = make_tensor(
      make_gmem_ptr(reinterpret_cast<int*>(params.func_ptr) + binfo.sum_s_q),
      make_shape(Int<1>{}, Int<kNFunc/2 + 1>{}, actual_seqlen_q),
      make_stride(params.func_head_stride, 2 * params.func_ids_stride, _1{}));
  Tensor mMinFunc = make_tensor(
      make_gmem_ptr(reinterpret_cast<int*>(params.func_ptr) + binfo.sum_s_q + params.func_ids_stride),
      make_shape(Int<1>{}, Int<kNFunc/2>{}, actual_seqlen_q),
      make_stride(params.func_head_stride, 2 * params.func_ids_stride, _1{}));
  Tensor gMaxFunc = local_tile(mMaxFunc(Int<0>{}, _, _),
                               make_shape(Int<kNFunc/2 + 1>{}, Int<kBlockM>{}),
                               make_coord(Int<0>{}, m_block));
  Tensor gMinFunc = local_tile(mMinFunc(Int<0>{}, _, _),
                               make_shape(Int<kNFunc/2>{}, Int<kBlockM>{}),
                               make_coord(Int<0>{}, m_block));

  // Global memory tensors (typed as FP8 for FP8 mode, or BF16/FP16 for BF16 mode)
  using GmemElement = std::conditional_t<Is_fp8, typename Kernel_traits::Element, ElementSmem>;

  Tensor mQ = make_tensor(
      make_gmem_ptr(reinterpret_cast<GmemElement*>(params.q_ptr) +
          binfo.q_offset(params.q_row_stride)),
      make_shape(actual_seqlen_q, params.h, params.d),
      make_stride(params.q_row_stride, params.q_head_stride, _1{}));
  Tensor gQ = local_tile(
      mQ(_, bidh, _),
      Shape<Int<kBlockM>, Int<kHeadDim>>{},
      make_coord(m_block, 0));
  Tensor mK = make_tensor(
      make_gmem_ptr(reinterpret_cast<GmemElement*>(params.k_ptr) +
          binfo.k_offset(params.k_row_stride)),
      make_shape(actual_seqlen_k, params.h_k, params.d),
      make_stride(params.k_row_stride, params.k_head_stride, _1{}));
  Tensor gK = local_tile(
      mK(_, bidh / params.h_h_k_ratio, _),
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      make_coord(_, 0));
  Tensor mV = make_tensor(
      make_gmem_ptr(reinterpret_cast<GmemElement*>(params.v_ptr) +
          binfo.k_offset(params.v_row_stride)),
      make_shape(actual_seqlen_k, params.h_k, params.d),
      make_stride(params.v_row_stride, params.v_head_stride, _1{}));
  Tensor gV = local_tile(
      mV(_, bidh / params.h_h_k_ratio, _),
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      make_coord(_, 0));

  const int bidh_rab = (params.h_rab > 1) ? bidh : 0;
  size_t rab_qkv_not_equal_offset = bidb * params.rab_seqlen_qk_stride +
      bidh_rab * params.rab_seqlen_q_stride +
      params.seqlen_k_rounded * actual_seqlen_offset;
  // RAB is always BF16/FP16 (not quantized), typed as RabElement
  auto mRab = make_tensor(
      make_gmem_ptr(reinterpret_cast<RabElement*>(params.rab_ptr) +
          rab_qkv_not_equal_offset),
      make_shape(actual_seqlen_q, params.seqlen_k_rounded),
      make_stride(params.rab_seqlen_k_stride, _1{}));
  auto gRab = local_tile(
      mRab(_, _),
      make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
      make_coord(m_block, _));

  // SMEM tensors: Q/K/V typed as ElementSmem (FP8 in Phase 2, BF16 otherwise).
  // RAB SMEM is typed as RabElement (always BF16).
  Tensor sQ = make_tensor(
      make_smem_ptr(reinterpret_cast<ElementSmem*>(smem_q)),
      typename Kernel_traits::SmemLayoutQ{});
  Tensor sK = make_tensor(
      sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
      typename Kernel_traits::SmemLayoutKV{});
  Tensor sV =
      make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
  Tensor sVt =
      make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle = make_tensor(
      sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
  Tensor sRab = make_tensor(
      make_smem_ptr(reinterpret_cast<RabElement*>(smem_ + Kernel_traits::kSmemSizeQKV)),
      typename Kernel_traits::SmemLayoutRab{});
  Tensor sValidBlockIds = make_tensor(
      make_smem_ptr(reinterpret_cast<int*>(smem_ + Kernel_traits::kSmemSizeQKVRab)),
      typename Kernel_traits::SmemLayoutValidBlockIds{});
  Tensor sFunc_min = make_tensor(
      make_smem_ptr(reinterpret_cast<int*>(sf_min)),
      typename Kernel_traits::SmemLayoutMinFunc{});
  Tensor sFunc_max = make_tensor(
      make_smem_ptr(reinterpret_cast<int*>(sf_max)),
      typename Kernel_traits::SmemLayoutMaxFunc{});

  if constexpr (Is_arbitrary) {
    const int lane_id = cutlass::canonical_lane_idx();
    const int warp_id = cutlass::canonical_warp_idx_sync();
    if (warp_id == 0) {
      *sn_valid_block_max = 0;
      sFunc_min[0] = 0;
      __syncwarp();
      int f_min = INT_MAX;
      int f_max = INT_MIN;
      const int base_row = m_block * kBlockM;
      for (int i = 0; i < size<0>(gMinFunc); i++) {
        for (int j = lane_id; j < size<1>(gMinFunc); j += 32) {
          const int row = base_row + j;
          if (row < actual_seqlen_q) {
            if (f_min > gMinFunc(i, j)) f_min = gMinFunc(i, j);
          }
        }
        warpReduce(f_min, MinOp<int>());
        if (lane_id == 0) sFunc_min[i+1] = f_min;
        f_min = INT_MAX;
      }
      for (int i = 0; i < size<0>(gMaxFunc); i++) {
        for (int j = lane_id; j < size<1>(gMaxFunc); j += 32) {
          const int row = base_row + j;
          if (row < actual_seqlen_q) {
            if (f_max < gMaxFunc(i, j)) f_max = gMaxFunc(i, j);
          }
        }
        warpReduce(f_max, MaxOp<int>());
        if (lane_id == 0) sFunc_max[i] = f_max;
        f_max = INT_MIN;
      }
      if (lane_id == 0) {
        for (int n_block = n_block_min; n_block < n_block_max; n_block++) {
          int b_max = (n_block + 1) * kBlockN;
          int b_min = n_block * kBlockN;
          for (int i = 0; i < (kNFunc + 1)/2; i++) {
            int f_min = sFunc_min[i];
            int f_max = sFunc_max[i];
            if (f_max <= f_min) continue;
            bool case1 = (f_min <= b_min && f_max > b_min);
            bool case2 = (f_min >= b_min && b_max > f_min);
            bool case3 = (f_min >= b_min && f_max < b_max);
            if (case1 || case2 || case3) {
              sValidBlockIds[*sn_valid_block_max] = n_block;
              (*sn_valid_block_max)++;
              break;
            }
          }
        }
      }
    }
    __syncthreads();
    n_block_max = *sn_valid_block_max;
    n_block_min = 0;
  }

  // Early exit: write zeros if no valid blocks
  if (((Is_causal || Is_local || Is_arbitrary) && n_block_max <= n_block_min) ||
      m_block * kBlockM >= actual_seqlen_q) {
    // BF16/FP16 output element type
    using OutElement = std::conditional_t<Is_fp8,
        typename Kernel_traits::OutputType,
        typename Kernel_traits::Element>;
    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<OutElement*>(params.o_ptr) +
            binfo.q_offset(params.o_row_stride)),
        make_shape(actual_seqlen_q, params.h, params.d),
        make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(
        mO(_, bidh, _),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_coord(m_block, 0));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO = make_tensor<OutElement>(shape(tOgO));
    clear(tOrO);
    Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    flash::copy<false, false, false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO,
        actual_seqlen_q_padded - m_block * kBlockM);
    return;
  }

  // -----------------------------------------------------------------------
  // GMEM → SMEM copies (both BF16 and FP8 use cp.async in Phase 2)
  // -----------------------------------------------------------------------

  // RAB copy (always BF16, use cp.async)
  typename Kernel_traits::GmemTiledCopyRab gmem_tiled_copy_Rab;
  auto gmem_thr_copy_Rab = gmem_tiled_copy_Rab.get_thread_slice(tidx);
  auto tQgRab = gmem_thr_copy_Rab.partition_S(gRab);
  auto tQsRab = gmem_thr_copy_Rab.partition_D(sRab);
  auto cRab = make_identity_tensor(make_shape(size<0>(sRab), size<1>(sRab)));
  auto tQcRab = gmem_thr_copy_Rab.partition_S(cRab);

  auto copy_if_g2s_rab = [&](int n_block_id, int buffer_stage) {
    auto ctQgRab_view = tQgRab(_, _, _, n_block_id);
#pragma unroll
    for (int m = 0; m < size<1>(ctQgRab_view); ++m) {
      if (get<0>(tQcRab(0, m, 0)) < (actual_seqlen_q - m_block * kBlockM)) {
#pragma unroll
        for (int k = 0; k < size<2>(ctQgRab_view); ++k) {
          if (get<1>(tQcRab(0, m, k)) < (actual_seqlen_k - n_block_id * kBlockN)) {
            cute::copy(gmem_tiled_copy_Rab, ctQgRab_view(_, m, k), tQsRab(_, m, k, buffer_stage));
          }
        }
      }
    }
  };

  auto copy_g2s_rab = [&](int n_block_id, int buffer_stage) {
    auto ctQgRab_view = tQgRab(_, _, _, n_block_id);
#pragma unroll
    for (int m = 0; m < size<1>(ctQgRab_view); ++m) {
#pragma unroll
      for (int k = 0; k < size<2>(ctQgRab_view); ++k) {
        if (get<0>(tQcRab(0, m, k)) < (actual_seqlen_q - m_block * kBlockM)) {
          cute::copy(gmem_tiled_copy_Rab, ctQgRab_view(_, m, k), tQsRab(_, m, k, buffer_stage));
        }
      }
    }
  };

  int n_valid_block_max = Is_arbitrary ? *sn_valid_block_max - 1 : 0;
  int n_block = !Is_arbitrary ? n_block_max - 1 : sValidBlockIds[n_valid_block_max];
  int buffer_stage = 0;
  // Runtime GEMM1 bypass: set HSTU_DEBUG_GEMM1_ONLY=1 to skip GEMM2
  // and output raw Q×K^T tiles (summed over n_blocks) instead.
  const bool kGemm1Bypass = params.debug_gemm1_only;

  if constexpr (Has_rab) {
    copy_if_g2s_rab(n_block, buffer_stage);
  }

  if constexpr (!Is_fp8) {
    // BF16/FP16 path: use cp.async for Q/K loads
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK(_, _, _0{}));
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle(_, _, _0{}));
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    auto smem_tiled_copy_rab = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_rab = smem_tiled_copy_rab.get_thread_slice(tidx);
    auto tSsRab = smem_thr_copy_rab.partition_S(sRab);

    Tensor cQ2 = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor tQcQ2 = gmem_thr_copy_QKV.partition_S(cQ2);

    // Prefill Q (non-async for Is_Q_in_regs, async otherwise)
    flash::copy<false>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ,
        actual_seqlen_q - m_block * kBlockM);
    if constexpr (Kernel_traits::Is_Q_in_regs) {
      cute::cp_async_fence();
    }
    if constexpr (Kernel_traits::Share_Q_K_smem) {
      flash::cp_async_wait<0>();
      __syncthreads();
      Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
      cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
      __syncthreads();
    }

    // Prefill K
    auto tKsK_stage0 = tKsK(_, _, _, buffer_stage);
    flash::copy<false, false>(gmem_tiled_copy_QKV,
        tKgK(_, _, _, n_block), tKsK_stage0, tKVcKV,
        actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    if constexpr (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
      flash::cp_async_wait<1>();
      __syncthreads();
      Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
      cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    auto col_limit_right = [&](int row) {
      return std::min(actual_seqlen_k, row + 1 + params.window_size_right);
    };
    auto col_limit_left = [&](int row) {
      return std::max(0, row - params.window_size_left);
    };

    auto apply_mask = [&](auto& tSrS, int n_block) {
      static constexpr int Row = 0, Col = 1;
      Tensor cS   = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
      Tensor tScS = thr_mma.partition_C(cS);
      const int base_row = m_block * kBlockM + actual_seqlen_offset;
      const int base_col = n_block * kBlockN;
      // Direct flat indexing: tScS(flat)/tSrS(flat) share the same element ordering
      // produced by partition_C, giving correct (row,col) coords for any AtomLayout.
      Tensor col_min = make_tensor<int>(make_shape(size<0>(gMinFunc)));
      Tensor col_max = make_tensor<int>(make_shape(size<0>(gMaxFunc)));
      int prev_block_row    = -1;
      int row               = 0;
      [[maybe_unused]] int target_col_limit_left = 0;
#pragma unroll
      for (int flat = 0; flat < size(tSrS); ++flat) {
        const auto coord    = tScS(flat);
        const int block_row = int(get<Row>(coord));
        if (block_row != prev_block_row) {
          row            = block_row + base_row;
          prev_block_row = block_row;
          if constexpr (Is_target) {
            const int target_index = (row - actual_seqlen_h) / params.target_group_size;
            target_col_limit_left = actual_seqlen_h + target_index * params.target_group_size;
          }
          if constexpr (Is_arbitrary) {
            col_max(0) = gMaxFunc(0, block_row);
#pragma unroll
            for (int j = 0; j < size<0>(gMinFunc); ++j) {
              col_min(j)   = gMinFunc(j, block_row);
              col_max(j+1) = gMaxFunc(j+1, block_row);
            }
          }
        }
        const int block_col = int(get<Col>(coord));
        const int col       = block_col + base_col;
        if constexpr (!Is_causal && !Is_local && !Is_arbitrary) {
          if (col >= actual_seqlen_k) { tSrS(flat) = -INFINITY; continue; }
        } else {
          if constexpr (Is_context) {
            if (row < actual_seqlen_c && col < actual_seqlen_h) continue;
          }
          if (col >= col_limit_right(row)) { tSrS(flat) = -INFINITY; continue; }
          if constexpr (Is_local) {
            if (col < col_limit_left(row)) { tSrS(flat) = -INFINITY; continue; }
          }
          if constexpr (Is_target) {
            if (row >= actual_seqlen_h && col >= actual_seqlen_h && col < target_col_limit_left)
              tSrS(flat) = -INFINITY;
          }
        }
        if constexpr (Is_arbitrary) {
          bool non_mask = (0 <= col) && (col < col_max(0));
          if (non_mask) continue;
#pragma unroll
          for (int j = 0; j < size<0>(gMinFunc); ++j) {
            non_mask = (col_min(j) <= col) && (col < col_max(j+1));
            if (non_mask) break;
          }
          if (!non_mask) tSrS(flat) = -INFINITY;
        }
      }
    };

    auto fwd_step = [&](int n_valid_block, int masking_step) {
      int n_block = n_valid_block;
      if constexpr (Is_arbitrary) {
        n_block = sValidBlockIds[n_valid_block];
      }
      const bool is_masking = masking_step < n_masking_steps ||
          (n_block + 1) * kBlockN > actual_seqlen_h;
      Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
      flash::cp_async_wait<0>();
      __syncthreads();

      // async load V (skipped when GEMM1 bypass is enabled)
      if (!kGemm1Bypass) {
        auto tVsV_stage = tVsV(_, _, _, buffer_stage);
        if (masking_step > 0) {
          flash::copy<true>(gmem_tiled_copy_QKV,
              tVgV(_, _, _, n_block), tVsV_stage, tKVcKV);
        } else {
          flash::copy<false, true>(gmem_tiled_copy_QKV,
              tVgV(_, _, _, n_block), tVsV_stage, tKVcKV,
              actual_seqlen_k - n_block * kBlockN);
        }
        cute::cp_async_fence();
      }

      // GEMM1: Q × K^T
      if constexpr (Has_rab) {
        Tensor rRab = make_tensor<ElementSmem>(
            partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
        auto tSrRab_view = smem_thr_copy_rab.retile_D(rRab);
        cute::copy(smem_tiled_copy_rab, tSsRab(_, _, _, buffer_stage), tSrRab_view(_, _, _));
        flash::convert_type_safe(rRab, acc_s);
        if (n_valid_block > n_block_min) {
          int n_block_next = n_block - 1;
          if constexpr (Is_arbitrary) {
            n_block_next = sValidBlockIds[n_valid_block - 1];
          }
          if (is_jump && masking_step == n_masking_steps - 1)
            n_block_next = std::min(n_block, n_block_history) - 1;
          if (n_block_next >= n_block_min) copy_g2s_rab(n_block_next, buffer_stage);
        }
      } else {
        clear(acc_s);
      }
      flash::gemm<Kernel_traits::Is_Q_in_regs>(
          acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, buffer_stage),
          tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
          smem_thr_copy_Q, smem_thr_copy_K);

      if (kGemm1Bypass) {
        // Stage-A: bypass right after GEMM1.
        Tensor acc_s_view = make_tensor(
            acc_s.data(),
            group<1, 3>(group<0, 2>(select<1, 2, 0, 3>(flatten(acc_s.layout())))));
        Tensor acc_o_view = make_tensor(
            acc_o.data(),
            group<1, 3>(group<0, 2>(select<1, 2, 0, 3>(flatten(acc_o.layout())))));
#pragma unroll
        for (int r = 0; r < size<0>(acc_o_view); ++r) {
#pragma unroll
          for (int c = 0; c < size<1>(acc_o_view); ++c) {
            if (c < size<1>(acc_s_view)) {
              acc_o_view(r, c) += acc_s_view(r, c);
            }
          }
        }
        flash::cp_async_wait<0>();
        __syncthreads();
        // Keep K pipeline moving so seq>=128 iterates across distinct n_blocks.
        if (n_valid_block > n_block_min) {
          int n_block_next = n_block - 1;
          if constexpr (Is_arbitrary) {
            n_block_next = sValidBlockIds[n_valid_block - 1];
          }
          if (is_jump && masking_step == n_masking_steps - 1)
            n_block_next = std::min(n_block, n_block_history) - 1;
          if (n_block_next >= n_block_min) {
            auto tKsK_stage_next = tKsK(_, _, _, buffer_stage);
            flash::copy<true>(gmem_tiled_copy_QKV,
                tKgK(_, _, _, n_block_next), tKsK_stage_next, tKVcKV);
            cute::cp_async_fence();
          }
        }
        return;
      }

      if (Is_arbitrary || Is_local || is_masking) {
        apply_mask(acc_s, n_block);
      }

      flash::cp_async_wait<0>();
      __syncthreads();

      if (n_valid_block > n_block_min) {
        int n_block_next = n_block - 1;
        if constexpr (Is_arbitrary) {
          n_block_next = sValidBlockIds[n_valid_block - 1];
        }
        if (is_jump && masking_step == n_masking_steps - 1)
          n_block_next = std::min(n_block, n_block_history) - 1;
        if (n_block_next >= n_block_min) {
          auto tKsK_stage_next = tKsK(_, _, _, buffer_stage);
          flash::copy<true>(gmem_tiled_copy_QKV,
              tKgK(_, _, _, n_block_next), tKsK_stage_next, tKVcKV);
          cute::cp_async_fence();
        }
      }

      for (int i = 0; i < size(acc_s); ++i) acc_s(i) *= params.alpha;
      fast_silu(acc_s);

      Tensor rP = make_tensor_like<ElementSmem>(acc_s);
      flash::convert_type_safe(acc_s, rP);
      Tensor tOrP = make_tensor(
          rP.data(),
          flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));

      flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt(_, _, _, buffer_stage),
          tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    };

    for (int n_block = n_block_max - 1, masking_step = 0; n_block >= n_block_min; ++masking_step, --n_block) {
      fwd_step(n_block, masking_step);
      if (is_jump && masking_step == n_masking_steps - 1)
        n_block = std::min(n_block, n_block_history);
    }

    // Scale output
    for (int i = 0; i < size(acc_o); ++i) acc_o(i) /= params.scaling_seqlen;

    // Epilogue: FP32 → BF16/FP16
    using OutElement = typename Kernel_traits::Element;
    Tensor rO = make_tensor_like<OutElement>(acc_o);
    flash::convert_type_safe(acc_o, rO);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
    if constexpr (Kernel_traits::Share_Q_K_smem) __syncthreads();
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<OutElement*>(params.o_ptr) +
            binfo.q_offset(params.o_row_stride)),
        make_shape(actual_seqlen_q, params.h, params.d),
        make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    __syncthreads();
    Tensor tOrO = make_tensor<OutElement>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    for (int m = 0; m < size<1>(tOgO); m++) {
      if (get<0>(tOcO(0, m, 0)) >= actual_seqlen_q - m_block * kBlockM) cute::clear(tOrO(_, m, _));
    }
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    flash::copy<false, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO,
        actual_seqlen_q_padded - m_block * kBlockM);

  } else {
    // Phase 1: FP8 block-scale (quant_mode=2), cp.async + unit SF (0x7f7f7f7f).
    // Uses SM120 QMMA (SM120_16x8x32_TN_VS block-scaled FP8 MMA).
    using BS1 = hstu::SM120QmmaBuilder<kBlockM, kBlockN, 4>;
    using BS2 = hstu::SM120QmmaBuilder<kBlockM, kHeadDim, 4>;

    // SW128 SMEM layouts (required by SM120 block-scaled ldmatrix).
    using SmemLayoutQ_SW128 = decltype(tile_to_shape(
        typename BS1::SmemLayoutAtomA{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutK_SW128 = decltype(tile_to_shape(
        typename BS1::SmemLayoutAtomB{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutV_SW128 = decltype(tile_to_shape(
        typename BS2::SmemLayoutAtomB{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    // V^T for GEMM2 B operand: GEMM2 needs B in [N=kHeadDim, K=kBlockN] form.
    // SmemLayoutVt_SW128 is a fresh tile_to_shape with swapped dims [kHeadDim, kBlockN],
    // compatible with ldmatrix (SmemCopyAtomB expects SW128 with K=kBlockN as inner dim).
    using SmemLayoutVt_SW128 = decltype(tile_to_shape(
        typename BS2::SmemLayoutAtomB{},
        Shape<Int<kHeadDim>, Int<kBlockN>>{}));

    // In the FP8 path, ElementSmem == Element == cutlass::float_e4m3_t.
    using FP8Elem = ElementSmem;

    // Q and K share SMEM (Share_Q_K_smem=true): both at smem_q base.
    // V is at smem_q + size(SmemLayoutK_SW128) elements.
    Tensor sQ_sw128 = make_tensor(
        make_smem_ptr(reinterpret_cast<FP8Elem*>(smem_q)),
        SmemLayoutQ_SW128{});
    Tensor sK_sw128 = make_tensor(
        make_smem_ptr(reinterpret_cast<FP8Elem*>(smem_q)),
        SmemLayoutK_SW128{});
    Tensor sV_sw128 = make_tensor(
        make_smem_ptr(reinterpret_cast<FP8Elem*>(smem_q) + size(SmemLayoutK_SW128{})),
        SmemLayoutV_SW128{});
    // Phase 3: single V buffer at [kSmemQSize, 2*kSmemQSize).
    // sQ=sK at [0, kSmemQSize), sV_sw128 at [kSmemQSize, 2*kSmemQSize).

    // SF SMEM layout: [DATA...][SFA(512B)][SFB(512B)][mbar(8B)]
    // SF starts at kSmemSize - kSmemMbarSize - kSmemSFSize to avoid overlapping the mbarrier.
    static constexpr int kSmemSFOffset = Kernel_traits::kSmemSize - Kernel_traits::kSmemMbarSize - Kernel_traits::kSmemSFSize;
    int32_t* smem_sfa_ptr = reinterpret_cast<int32_t*>(smem_ + kSmemSFOffset);
    int32_t* smem_sfb_ptr = smem_sfa_ptr + kBlockM;  // 128 int32 = 512 bytes gap

    // Load Q SF (SFA): per-row loading — one int32 per token in this Q-tile.
    // Q is quantized along D (per-token scale, 1 scale per token per head).
    // q_block_descale_head_stride = total_q_tokens per head (one scale-entry per token).
    // Tile [m_block*kBlockM : (m_block+1)*kBlockM]: load each row's own scale.
    if (params.sf_q_packed_ptr != nullptr && params.cu_seqlens_q_block_descale != nullptr) {
      const int64_t q_tile_base = static_cast<int64_t>(bidh) * params.q_block_descale_head_stride
          + params.cu_seqlens_q_block_descale[bidb] + m_block * kBlockM;
      for (int i = tidx; i < kBlockM; i += Kernel_traits::kNThreads)
        smem_sfa_ptr[i] = params.sf_q_packed_ptr[q_tile_base + i];
    } else {
      for (int i = tidx; i < kBlockM; i += Kernel_traits::kNThreads)
        smem_sfa_ptr[i] = 0x7f7f7f7f;
    }
    // SFB (K scale) is loaded per n_block inside fwd_step_fp8bs.
    __syncthreads();

    // Build SMEM SF tensors (shape [N, 1, SF_Stages=1]).
    using SmemLayoutSFA = typename BS1::SmemLayoutSFA;
    using SmemLayoutSFB = typename BS1::SmemLayoutSFB;
    Tensor sSFA_ = make_tensor(make_smem_ptr(smem_sfa_ptr), SmemLayoutSFA{});
    Tensor sSFB_ = make_tensor(make_smem_ptr(smem_sfb_ptr), SmemLayoutSFB{});
    auto sSFA = as_position_independent_swizzle_tensor(sSFA_);
    auto sSFB = as_position_independent_swizzle_tensor(sSFB_);


    // GMEM→SMEM copy for SW128 FP8 SMEM:
    // 8 threads/row × 16 FP8/thread = 128 FP8 per row (128-byte row = SW128 atom).
    using GmemLayoutAtom_SW128 = Layout<
        Shape<Int<Kernel_traits::kNThreads / 8>, _8>,
        Stride<_8, _1>>;
    auto gmem_tiled_copy_sw128 = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, FP8Elem>{},
        GmemLayoutAtom_SW128{},
        Layout<Shape<_1, _16>>{});  // 16 FP8 per thread per load
    auto gmem_thr_copy = gmem_tiled_copy_sw128.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy.partition_D(sQ_sw128);
    Tensor tKgK = gmem_thr_copy.partition_S(gK);
    Tensor tKsK = gmem_thr_copy.partition_D(sK_sw128);
    Tensor tVgV = gmem_thr_copy.partition_S(gV);
    Tensor tVsV = gmem_thr_copy.partition_D(sV_sw128);   // V (single buffer)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK_sw128), size<1>(sK_sw128)));
    Tensor tKVcKV = gmem_thr_copy.partition_S(cKV);

    // MMA setup for GEMM1 (Q×K) and GEMM2 (P×V).
    typename BS1::TiledMma tiled_mma_g1;
    auto thr_mma_g1 = tiled_mma_g1.get_thread_slice(tidx);
    typename BS2::TiledMma tiled_mma_g2;
    auto thr_mma_g2 = tiled_mma_g2.get_thread_slice(tidx);

    // Output accumulator (GEMM2 result: kBlockM × kHeadDim).
    Tensor acc_o = partition_fragment_C(tiled_mma_g2, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_o);

    // s2r copy atoms — A operands use Z-pattern uint32 loads (see load_a_z_pattern below).
    auto s2r_copy_B  = make_tiled_copy_B(typename BS1::SmemCopyAtomB{}, tiled_mma_g1);
    auto s2r_thr_copy_B  = s2r_copy_B.get_thread_slice(tidx);
    auto s2r_copy_B2 = make_tiled_copy_B(typename BS2::SmemCopyAtomB{}, tiled_mma_g2);
    auto s2r_thr_copy_B2 = s2r_copy_B2.get_thread_slice(tidx);

    // A-operand Z-pattern loader for SM120_16x8x32_TN AtomLayout <_8,_1,_1>.
    // See matching implementation in hstu_fwd_kernel_fp8_ws.h for full comment.
    // Uses individual uint32 loads (not ldmatrix.x4) to correctly implement Z-pattern
    // from K_SW128 SMEM (ldmatrix.x4 fails for odd rows due to Swizzle<3,4,3> non-contiguity).
    auto load_a_z_pattern = [&](auto&& sA_pi, auto& tCrA, int k_block_base, int k_block_count) {
      const int lane   = tidx & 31;
      const int warp_m = tidx / 32;
      const int m_row0 = warp_m * 16 + (lane >> 2);
      const int m_row1 = m_row0 + 8;
      const int k_col  = (lane & 3) * 4;
      auto tXrA = recast<uint32_t>(tCrA);
      CUTE_UNROLL
      for (int kb = 0; kb < k_block_count; ++kb) {
        const int K0 = k_col + (k_block_base + kb) * 32;
        const int K1 = K0 + 16;
        tXrA(4*kb+0) = *reinterpret_cast<const uint32_t*>(&sA_pi(m_row0, K0));
        tXrA(4*kb+1) = *reinterpret_cast<const uint32_t*>(&sA_pi(m_row1, K0));
        tXrA(4*kb+2) = *reinterpret_cast<const uint32_t*>(&sA_pi(m_row0, K1));
        tXrA(4*kb+3) = *reinterpret_cast<const uint32_t*>(&sA_pi(m_row1, K1));
      }
    };

    // B-operand Z-pattern loader.  Mirrors load_b_z_pattern in hstu_fwd_kernel_fp8_ws.h.
    // make_tiled_copy_B(SM75_U32x4_LDSM_N) is incompatible with AtomLayout <_8,_1,_1>
    // (ThrN=1) because ldmatrix.x4 expects 16-byte-aligned addresses derived from the
    // N-partition, which the new layout cannot guarantee.  Direct uint32 reads bypass this.
    // PermMmaTileN = Layout<Shape<_8,_4,_4>, Stride<_1,_32,_8>>:
    //   N_base[nr] = (nr % 4)*32 + (nr / 4)*8; linear_idx = 2*nr + 32*kb.
    auto load_b_z_pattern = [&](auto&& sB_pi, auto& tCrB, int k_block_base, int k_block_count) {
      const int lane  = tidx & 31;
      const int n_row = lane >> 2;
      const int k_col = (lane & 3) * 4;
      auto tXrB = recast<uint32_t>(tCrB);
      constexpr int kNReps = kBlockN / 8;
      CUTE_UNROLL
      for (int kb = 0; kb < k_block_count; ++kb) {
        const int K0 = k_col + (k_block_base + kb) * 32;
        const int K1 = K0 + 16;
        CUTE_UNROLL
        for (int nr = 0; nr < kNReps; ++nr) {
          const int N    = (nr % 4) * 32 + (nr / 4) * 8 + n_row;
          const int base = 32 * kb + 2 * nr;
          tXrB(base + 0) = *reinterpret_cast<const uint32_t*>(&sB_pi(N, K0));
          tXrB(base + 1) = *reinterpret_cast<const uint32_t*>(&sB_pi(N, K1));
        }
      }
    };

    // s2r copy for SF operands.
    auto s2r_copy_SFA = make_tiled_copy_impl(
        typename BS1::SmemCopyAtomSF{},
        BS1::get_layoutSFA_TV(tiled_mma_g1),
        make_shape(size<0>(tile_shape(tiled_mma_g1)), _1{}));
    auto s2r_thr_copy_SFA = s2r_copy_SFA.get_thread_slice(tidx);
    auto s2r_copy_SFB = make_tiled_copy_impl(
        typename BS1::SmemCopyAtomSF{},
        BS1::get_layoutSFB_TV(tiled_mma_g1),
        make_shape(size<1>(tile_shape(tiled_mma_g1)), _1{}));
    auto s2r_thr_copy_SFB = s2r_copy_SFB.get_thread_slice(tidx);

    auto s2r_copy_SFP = make_tiled_copy_impl(
        typename BS2::SmemCopyAtomSF{},
        BS2::get_layoutSFA_TV(tiled_mma_g2),
        make_shape(size<0>(tile_shape(tiled_mma_g2)), _1{}));
    auto s2r_thr_copy_SFP = s2r_copy_SFP.get_thread_slice(tidx);
    auto s2r_copy_SFV = make_tiled_copy_impl(
        typename BS2::SmemCopyAtomSF{},
        BS2::get_layoutSFB_TV(tiled_mma_g2),
        make_shape(size<1>(tile_shape(tiled_mma_g2)), _1{}));
    auto s2r_thr_copy_SFV = s2r_copy_SFV.get_thread_slice(tidx);

    // Load Q: cp.async → SW128 SMEM → s2r → registers.
    {
      Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ_sw128), size<1>(sQ_sw128)));
      Tensor tQcQ = gmem_thr_copy.partition_S(cQ);
      flash::copy<false>(gmem_tiled_copy_sw128, tQgQ, tQsQ, tQcQ,
          actual_seqlen_q - m_block * kBlockM);
    }
    cute::cp_async_fence();
    flash::cp_async_wait<0>();
    __syncthreads();

    auto sQ_pi = as_position_independent_swizzle_tensor(sQ_sw128);
    Tensor tCrQ = thr_mma_g1.partition_fragment_A(sQ_pi);
    load_a_z_pattern(sQ_pi, tCrQ, 0, kHeadDim / 32);

    // Load SFA once (unit scale stays constant throughout).
    Tensor tCrSFA = BS1::partition_fragment_SFA(sSFA(_,_,_0{}), thr_mma_g1);
    {
      auto tXsSFA = s2r_thr_copy_SFA.partition_S(sSFA);
      auto tXrSFA = s2r_thr_copy_SFA.retile_D(tCrSFA);
      cute::copy(s2r_copy_SFA, tXsSFA(_,_,_,_0{}), tXrSFA);
    }
    auto tCrSFA_frg = BS1::transform_fragment_for_qmma(tCrSFA);

    // mask lambdas (identical logic to BF16 path).
    auto col_limit_right = [&](int row) {
      return std::min(actual_seqlen_k, row + 1 + params.window_size_right);
    };
    auto col_limit_left = [&](int row) {
      return std::max(0, row - params.window_size_left);
    };
    // apply_mask_bs: Full masking logic for FP8 block-scale path (mirrors BF16 apply_mask).
    // Handles Is_causal, Is_local, Is_context, Is_target, Is_arbitrary, and seqlen bounds.
    auto apply_mask_bs = [&](auto& tSrS, int nb) {
      static constexpr int Row = 0, Col = 1;
      Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
      Tensor tScS = thr_mma_g1.partition_C(cS);
      const int base_row = m_block * kBlockM + actual_seqlen_offset;
      const int base_col = nb * kBlockN;

      Tensor col_min = make_tensor<int>(make_shape(size<0>(gMinFunc)));
      Tensor col_max = make_tensor<int>(make_shape(size<0>(gMaxFunc)));
      int prev_block_row         = -1;
      int row                    = 0;
      [[maybe_unused]] int target_col_limit_left = 0;
#pragma unroll
      for (int flat = 0; flat < size(tSrS); ++flat) {
        const auto coord    = tScS(flat);
        const int block_row = int(get<Row>(coord));
        if (block_row != prev_block_row) {
          row            = block_row + base_row;
          prev_block_row = block_row;
          if constexpr (Is_target) {
            const int target_index = (row - actual_seqlen_h) / params.target_group_size;
            target_col_limit_left  = actual_seqlen_h + target_index * params.target_group_size;
          }
          if constexpr (Is_arbitrary) {
            col_max(0) = gMaxFunc(0, block_row);
#pragma unroll
            for (int j = 0; j < size<0>(gMinFunc); ++j) {
              col_min(j)   = gMinFunc(j, block_row);
              col_max(j+1) = gMaxFunc(j+1, block_row);
            }
          }
        }
        const int block_col = int(get<Col>(coord));
        const int col       = block_col + base_col;
        if constexpr (!Is_causal && !Is_local && !Is_arbitrary) {
          if (col >= actual_seqlen_k) { tSrS(flat) = -INFINITY; continue; }
        } else {
          if constexpr (Is_context) {
            if (row < actual_seqlen_c && col < actual_seqlen_h) continue;
          }
          if (col >= col_limit_right(row)) { tSrS(flat) = -INFINITY; continue; }
          if constexpr (Is_local) {
            if (col < col_limit_left(row)) { tSrS(flat) = -INFINITY; continue; }
          }
          if constexpr (Is_target) {
            if (row >= actual_seqlen_h && col >= actual_seqlen_h && col < target_col_limit_left) {
              tSrS(flat) = -INFINITY;
            }
          }
        }
        if constexpr (Is_arbitrary) {
          bool non_mask = (0 <= col) && (col < col_max(0));
          if (non_mask) continue;
#pragma unroll
          for (int j = 0; j < size<0>(gMinFunc); ++j) {
            non_mask = (col_min(j) <= col) && (col < col_max(j+1));
            if (non_mask) break;
          }
          if (!non_mask) tSrS(flat) = -INFINITY;
        }
      }
    };

    // Preamble: prefetch K[n0] + V[n0] (V transposed in fwd_step) and RAB[n0] if needed.
    flash::copy<false, true>(gmem_tiled_copy_sw128,
        tKgK(_,_,_, n_block), tKsK, tKVcKV,
        actual_seqlen_k - n_block * kBlockN);
    flash::copy<false, true>(gmem_tiled_copy_sw128,
        tVgV(_,_,_, n_block), tVsV, tKVcKV,
        actual_seqlen_k - n_block * kBlockN);
    if constexpr (Has_rab) {
        copy_g2s_rab(n_block, 0);
    }
    cute::cp_async_fence();

    // nb       = actual K-block index for this iteration
    // nb_next  = actual K-block index to prefetch at end (-1 = no prefetch)
    auto fwd_step_fp8bs = [&](int nb, int nb_next, int masking_step) {
      const bool is_masking = masking_step < n_masking_steps ||
          (nb + 1) * kBlockN > actual_seqlen_h;

      // Load K SF for this n_block BEFORE cp_async_wait+sync so the __syncthreads
      // ensures visibility of both K SMEM and K SF with no extra barrier.
      // K: per-row loading — one int32 per token in this K-tile.
      // K is quantized along D (per-token scale), so tile [nb*kBlockN:(nb+1)*kBlockN]
      // has kBlockN=128 different scale entries in sf_k_packed.
      if (params.sf_k_packed_ptr != nullptr && params.cu_seqlens_kv_block_descale != nullptr) {
        const int64_t k_tile_base = static_cast<int64_t>(bidh) * params.kv_block_descale_head_stride
            + params.cu_seqlens_kv_block_descale[bidb] + nb * kBlockN;
        for (int i = tidx; i < kBlockN; i += Kernel_traits::kNThreads)
          smem_sfb_ptr[i] = params.sf_k_packed_ptr[k_tile_base + i];
      } else {
        for (int i = tidx; i < kBlockN; i += Kernel_traits::kNThreads)
          smem_sfb_ptr[i] = 0x7f7f7f7f;
      }

      // K[nb] and V[nb] were prefetched at end of previous iteration (or in preamble).
      // Wait for cp.async completion then sync so SMEM is visible to all threads.
      flash::cp_async_wait<0>();
      __syncthreads();
      auto sK_pi = as_position_independent_swizzle_tensor(sK_sw128);
      Tensor tCrK = thr_mma_g1.partition_fragment_B(sK_pi);
      load_b_z_pattern(sK_pi, tCrK, 0, kHeadDim / 32);

      // Load SFB (unit scale).
      Tensor tCrSFB = BS1::partition_fragment_SFB(sSFB(_,_,_0{}), thr_mma_g1);
      {
        auto tXsSFB = s2r_thr_copy_SFB.partition_S(sSFB);
        auto tXrSFB = s2r_thr_copy_SFB.retile_D(tCrSFB);
        cute::copy(s2r_copy_SFB, tXsSFB(_,_,_,_0{}), tXrSFB);
      }
      auto tCrSFB_frg = BS1::transform_fragment_for_qmma(tCrSFB);

      // GEMM1: acc_s += Q × K^T (block-scaled).
      // tCrSFA_frg / tCrSFB_frg are rank-4 (32, MMA_M, 4, AB_Stages).
      // Index the last dim with _0{} to obtain a rank-3 slice matching tCrQ/tCrK.
      // For unit scale (Phase 1) all AB_Stages carry the same 1.0 value, so
      // using stage 0 is correct for the full MMA_K=4 iteration in cute::gemm.
      Tensor acc_s = partition_fragment_C(tiled_mma_g1, Shape<Int<kBlockM>, Int<kBlockN>>{});
      clear(acc_s);
      cute::gemm(tiled_mma_g1,
          make_zip_tensor(tCrQ, tCrSFA_frg(_,_,_,_0{})),
          make_zip_tensor(tCrK, tCrSFB_frg(_,_,_,_0{})),
          acc_s);

      // Apply RAB (relative attention bias) to acc_s after GEMM1.
      // sRab[kBlockM, kBlockN, 1] was preloaded from GMEM (BF16) in the preamble
      // (copy_if_g2s_rab) and fenced with the K+V cp.async.  cp_async_wait<0> at
      // the top of this function guarantees sRab is fully written to SMEM.
      // We use thr_mma_g1.partition_C to get the logical (m, n) coordinate for each
      // register element, then read sRab(m,n,0) as float and add to acc_s.
      if constexpr (Has_rab) {
        Tensor cRab_id = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScRab  = thr_mma_g1.partition_C(cRab_id);
        CUTE_UNROLL
        for (int flat = 0; flat < size(acc_s); ++flat) {
          const auto coord = tScRab(flat);
          const int m      = int(get<0>(coord));
          const int n      = int(get<1>(coord));
          acc_s(flat) += float(sRab(m, n, 0));
        }
      }

      // GEMM1 bypass: accumulate raw Q×K^T into acc_o, skip mask/silu/GEMM2.
      // Issue K+V prefetch for next iter so the pipeline keeps moving, then return.
      if (params.debug_gemm1_only) {
        for (int i = 0; i < size(acc_s); ++i) acc_o(i) += acc_s(i);
        __syncthreads();
        if (nb_next >= 0) {
          flash::copy<false, true>(gmem_tiled_copy_sw128,
              tKgK(_,_,_, nb_next), tKsK, tKVcKV,
              actual_seqlen_k - nb_next * kBlockN);
          flash::copy<false, true>(gmem_tiled_copy_sw128,
              tVgV(_,_,_, nb_next), tVsV, tKVcKV,
              actual_seqlen_k - nb_next * kBlockN);
          cute::cp_async_fence();
        }
        return;
      }

      if (Is_arbitrary || Is_local || is_masking)
        apply_mask_bs(acc_s, nb);

      for (int i = 0; i < size(acc_s); ++i) acc_s(i) *= params.alpha;
      fast_silu(acc_s);

      // Convert float acc_s → FP8 rP.
      Tensor rP = make_tensor_like<FP8Elem>(acc_s);
      flash::convert_type_safe(acc_s, rP);

      // V[nb] is already in SMEM (loaded at the start of this iteration).
      // P SMEM roundtrip: write rP to sQ/sK buffer (Q is in regs; K is in regs).
      // IMPORTANT: must do this BEFORE issuing the K prefetch, because sPbuf shares
      // SMEM with sK_sw128.  If we issue cp.async K first, it may overwrite sPbuf
      // before we finish reading P back → race condition → exploding norms at seq≥256.
      //
      // Fix: use explicit logical (m,k) coordinate writes to sPbuf to bypass the
      // PermMmaTileN permutation in make_tiled_copy_C.  The SM120 QMMA C-matrix has
      // PermMmaTileN interleaving, so make_tiled_copy_C(DefaultCopy) maps register
      // element (r,c) to a permuted SMEM position, not its logical (m,k) position.
      // When seqlen<kBlockN (e.g. SEQ=64), only P[m,k<64] are non-zero; the permutation
      // scatters these non-zero bytes to wrong SMEM positions, causing GEMM2 to read
      // zeros for K=0..63 in some output tiles → acc_o[m,d>=64]=0.
      //
      // Solution: use partition_C to get logical (m,k) for each register element, then
      // write directly to sPbuf_pi(m,k).  The SW128 swizzle in sPbuf_pi maps logical
      // (m,k) to the correct physical SMEM address for ldmatrix reads.
      __syncthreads();

      Tensor sPbuf = make_tensor(sQ_sw128.data(), SmemLayoutQ_SW128{});
      auto sPbuf_pi = as_position_independent_swizzle_tensor(sPbuf);
      {
        Tensor cP_id = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tPcP  = thr_mma_g1.partition_C(cP_id);
        CUTE_UNROLL
        for (int flat = 0; flat < size(rP); ++flat) {
          const auto coord  = tPcP(flat);
          const int m_rel   = int(get<0>(coord));
          const int k_pos   = int(get<1>(coord));
          sPbuf_pi(m_rel, k_pos) = rP(flat);
        }
      }
      __syncthreads();

      // s2r load P (A-operand for GEMM2) via Z-pattern uint32 loads.
      Tensor tCrP = thr_mma_g2.partition_fragment_A(sPbuf_pi);
      load_a_z_pattern(sPbuf_pi, tCrP, 0, kBlockN / 32);

      // Fill SFP (P scale, always unit) and load SFV (V scale).
      // These writes to end-of-SMEM (smem_sfa/sfb_ptr) happen before V DMA, no conflict.
      for (int i = tidx; i < kBlockM; i += Kernel_traits::kNThreads)
        smem_sfa_ptr[i] = 0x7f7f7f7f;  // SFP = unit (P not explicitly quantized)
      if (params.sf_v_packed_ptr != nullptr && params.cu_seqlens_v_block_descale != nullptr) {
        // V SF: scalar broadcast — V is quantized along N (1 scale per 128-token tile).
        // v_block_descale_head_stride = SEQ/128 (from descale_v.stride(0)).
        // cu_seqlens_v_block_descale[bidb] = cumulative block offset for V (in blocks).
        const int64_t v_sf_idx = static_cast<int64_t>(bidh) * params.v_block_descale_head_stride
            + params.cu_seqlens_v_block_descale[bidb] + nb;
        const int32_t vsf_val = params.sf_v_packed_ptr[v_sf_idx];
        for (int i = tidx; i < kBlockN; i += Kernel_traits::kNThreads)
          smem_sfb_ptr[i] = vsf_val;
      } else {
        for (int i = tidx; i < kBlockN; i += Kernel_traits::kNThreads)
          smem_sfb_ptr[i] = 0x7f7f7f7f;
      }

      // V^T transpose: cooperatively transpose V from SMEM into sVt_buf (smem_q[0]).
      // Sync to protect smem_q[0] between s2r P done and Vt transpose writes.
      __syncthreads();
      Tensor sVt_buf = make_tensor(
          make_smem_ptr(reinterpret_cast<FP8Elem*>(smem_q)),
          SmemLayoutVt_SW128{});
      {
          FP8Elem* sV_curr_ptr = reinterpret_cast<FP8Elem*>(smem_q) + size(SmemLayoutK_SW128{});
          Tensor sV_curr = make_tensor(make_smem_ptr(sV_curr_ptr), SmemLayoutV_SW128{});
          for (int idx = tidx; idx < kBlockN * kHeadDim; idx += Kernel_traits::kNThreads) {
              int k = idx / kHeadDim;
              int d = idx % kHeadDim;
              sVt_buf(d, k) = sV_curr(k, d);
          }
      }
      __syncthreads();

      // s2r Vt: [kHeadDim, kBlockN] → tCrV registers (GEMM2 B operand).
      // Use Z-pattern uint32 reads (same as WS kernel) — ldmatrix.x4 is incompatible
      // with AtomLayout <_8,_1,_1> ThrN=1, producing misaligned SMEM addresses.
      FP8Elem* sVt_read_ptr = reinterpret_cast<FP8Elem*>(smem_q);
      Tensor sVt_read = make_tensor(make_smem_ptr(sVt_read_ptr), SmemLayoutVt_SW128{});
      auto sVt_pi = as_position_independent_swizzle_tensor(sVt_read);
      Tensor tCrV = thr_mma_g2.partition_fragment_B(sVt_pi);
      load_b_z_pattern(sVt_pi, tCrV, 0, kBlockN / 32);

      // Prefetch K[nb_next] + V[nb_next] for next iteration.
      if (nb_next >= 0) {
          flash::copy<false, true>(gmem_tiled_copy_sw128,
              tKgK(_,_,_, nb_next), tKsK, tKVcKV,
              actual_seqlen_k - nb_next * kBlockN);
          flash::copy<false, true>(gmem_tiled_copy_sw128,
              tVgV(_,_,_, nb_next), tVsV, tKVcKV,
              actual_seqlen_k - nb_next * kBlockN);
          if constexpr (Has_rab) {
              copy_g2s_rab(nb_next, 0);
          }
          cute::cp_async_fence();
      }

      // Load SFP (P scale) and SFV (V scale): both unit, reuse sSFA/sSFB.
      Tensor tCrSFP = BS2::partition_fragment_SFA(sSFA(_,_,_0{}), thr_mma_g2);
      {
        auto tXsSFP = s2r_thr_copy_SFP.partition_S(sSFA);
        auto tXrSFP = s2r_thr_copy_SFP.retile_D(tCrSFP);
        cute::copy(s2r_copy_SFP, tXsSFP(_,_,_,_0{}), tXrSFP);
      }
      auto tCrSFP_frg = BS2::transform_fragment_for_qmma(tCrSFP);

      Tensor tCrSFV = BS2::partition_fragment_SFB(sSFB(_,_,_0{}), thr_mma_g2);
      {
        auto tXsSFV = s2r_thr_copy_SFV.partition_S(sSFB);
        auto tXrSFV = s2r_thr_copy_SFV.retile_D(tCrSFV);
        cute::copy(s2r_copy_SFV, tXsSFV(_,_,_,_0{}), tXrSFV);
      }
      auto tCrSFV_frg = BS2::transform_fragment_for_qmma(tCrSFV);

      // GEMM2: acc_o += P × V (block-scaled).
      // Same rank-3 slice pattern as GEMM1.
      cute::gemm(tiled_mma_g2,
          make_zip_tensor(tCrP, tCrSFP_frg(_,_,_,_0{})),
          make_zip_tensor(tCrV, tCrSFV_frg(_,_,_,_0{})),
          acc_o);

    };

    // Main loop over n_blocks (descending order, same as BF16 path).
    // For Is_arbitrary: n_valid is the index into sValidBlockIds[] (0..n_block_max-1),
    //   nb = sValidBlockIds[n_valid] is the actual K-block index.
    // For !Is_arbitrary: n_valid == nb (the actual K-block index directly).
    for (int n_valid = n_block_max - 1, masking_step = 0; n_valid >= n_block_min;
         ++masking_step, --n_valid) {
      int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;
      // Adjust nb_next for is_jump (mirrors BF16 path's n_block_next adjustment):
      // when is_jump triggers, the actual next valid block jumps to n_block_history-1,
      // not n_valid-1.  Prefetch the correct block so the next iteration reads from SMEM.
      int nb_next_valid = (is_jump && masking_step == n_masking_steps - 1)
          ? std::min(n_valid, n_block_history) - 1
          : n_valid - 1;
      int nb_next = (nb_next_valid >= n_block_min)
          ? (Is_arbitrary ? int(sValidBlockIds[nb_next_valid]) : nb_next_valid)
          : -1;
      fwd_step_fp8bs(nb, nb_next, masking_step);
      if (is_jump && masking_step == n_masking_steps - 1)
        n_valid = std::min(n_valid, n_block_history);
    }

    // Scale output: divide by scaling_seqlen.
    for (int i = 0; i < size(acc_o); ++i) acc_o(i) /= params.scaling_seqlen;

    // Epilogue: float → OutputType → SMEM (flat row-major) → GMEM.
    //
    // Using make_tiled_copy_C(AutoVectorizingCopy, tiled_mma_g2) directly causes
    // an M-axis permutation in the SMEM write: the SM120 QMMA's PermMmaTileN
    // interleaving makes partition_D map some valid M rows (0..63) to SMEM positions
    // that belong to invalid M rows (64..127), and vice versa.  For uniform acc_o
    // (SEQ≥128, all rows equal) this is invisible; for non-uniform acc_o (SEQ=64,
    // valid rows vs zero rows) it produces cos_sim=0.7071 (half output zero).
    //
    // Fix: use thr_mma_g2.partition_C to get logical (M,N) coordinates for each
    // register element, then write directly to a flat row-major SMEM at (M,N).
    // This bypasses the QMMA permutation entirely.  The S→G copy then reads from
    // the flat SMEM straightforwardly.
    using OutElement = typename Kernel_traits::OutputType;
    Tensor rO = make_tensor_like<OutElement>(acc_o);
    flash::convert_type_safe(acc_o, rO);

    // Flat row-major BF16 SMEM: sO_flat(m,n) at byte offset (m*kHeadDim+n)*2.
    // kHeadDim=128: each row = 256 bytes (16-byte aligned), so 128-bit S→G loads
    // are correctly aligned for any starting column that is a multiple of 8.
    Tensor sO_flat = make_tensor(
        make_smem_ptr(reinterpret_cast<OutElement*>(smem_q)),
        Layout<Shape<Int<kBlockM>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>{});

    // Write each register element to its logical (M,N) position in flat SMEM.
    // partition_C(identity_tensor) gives the logical output coordinate for each
    // fragment position — same flatten/select pattern as apply_mask_bs.
    {
      Tensor cO_id = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
      Tensor tOcO  = thr_mma_g2.partition_C(cO_id);
      CUTE_UNROLL
      for (int flat = 0; flat < size(rO); ++flat) {
        const auto coord  = tOcO(flat);
        const int m_rel   = int(get<0>(coord));
        const int n_pos   = int(get<1>(coord));
        sO_flat(m_rel, n_pos) = rO(flat);
      }
    }
    __syncthreads();

    // Write sO_flat → GMEM using GmemTiledCopyO.
    // Flat row-major SMEM has consecutive elements in each row at stride-1 BF16,
    // which is compatible with the 128-bit vectorized GMEM stores in GmemTiledCopyO.
    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<OutElement*>(params.o_ptr) +
            binfo.q_offset(params.o_row_stride)),
        make_shape(actual_seqlen_q, params.h, params.d),
        make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO_bs = local_tile(mO(_,bidh,_),
        Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));
    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO_flat);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO_bs);
    __syncthreads();
    Tensor tOrO = make_tensor<OutElement>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
    Tensor cO_bs = make_identity_tensor(make_shape(size<0>(sO_flat), size<1>(sO_flat)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO_bs);
    for (int m = 0; m < size<1>(tOgO); m++) {
      if (get<0>(tOcO(0,m,0)) >= actual_seqlen_q - m_block * kBlockM)
        cute::clear(tOrO(_,m,_));
    }
    flash::copy<false,false,false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO,
        actual_seqlen_q_padded - m_block * kBlockM);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, typename Params>
__global__ void hstu_fwd_kernel_sm120(Params params) {
  int m_block = gridDim.x - blockIdx.x - 1;
  int bidh = blockIdx.y;
  int bidb = blockIdx.z;
  hstu_compute_attn_1rowblock_sm120<Kernel_traits>(params, bidb, bidh, m_block);
}

} // temporarily close flash namespace to define global struct

// Phase 6 WS TMA params: extends Hstu_fwd_params with TMA Q, K, V^T, and SF descriptors.
// TMA_Q_t / TMA_K_t / TMA_Vt_t are TMA copy atoms for FP8 data.
// TMA_SFA_t / TMA_SFB_t are TMA copy atoms for int32 SF data.
template <typename TMA_Q_t, typename TMA_K_t, typename TMA_Vt_t,
          typename TMA_SFA_t, typename TMA_SFB_t, typename TMA_SFV_t>
struct Hstu_fwd_params_fp8_ws_tma : public Hstu_fwd_params {
    TMA_Q_t   tma_q;   // TMA descriptor for Q:        [total_q, d, h]
    TMA_K_t   tma_k;   // TMA descriptor for K:        [total_k, d, h_k]
    TMA_Vt_t  tma_vt;  // TMA descriptor for V (row-major): [total_k, d, h_k]
    TMA_SFA_t tma_sfa; // TMA descriptor for Q scale:  [q_block_descale_head_stride, 1, h]
    TMA_SFB_t tma_sfb; // TMA descriptor for K scale:  [kv_block_descale_head_stride, 1, h_k]
    TMA_SFV_t tma_sfv; // TMA descriptor for V scale:  [v_block_descale_head_stride, 1, h_k]
};

namespace flash {  // reopen flash namespace for kernel

#include "hstu_fwd_kernel_fp8_ws.h"  // Phase 6 WS kernel

} // namespace flash

////////////////////////////////////////////////////////////////////////////////////////////////////

// Phase 6 WS TMA impl: creates TMA Q, K, and V^T descriptors on host, launches WS TMA kernel.
template <
    typename elem_type,
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    bool Is_causal,
    bool Is_target,
    bool Is_context,
    bool Is_local,
    bool Is_arbitrary,
    int kNFunc,
    bool Has_rab,
    bool Is_Q_in_regs = false,
    bool Share_Q_K_smem = false>
void run_hstu_fwd_sm120_fp8_ws_tma_impl(Hstu_fwd_params& params, cudaStream_t stream) {
  using Kernel_traits = Hstu_fwd_kernel_traits_sm120_fp8_ws<
      kHeadDim, kBlockM, kBlockN, kNWarps,
      Is_causal, Is_target, Is_context, Is_local, Is_arbitrary, kNFunc, Has_rab,
      Is_Q_in_regs, Share_Q_K_smem, cutlass::half_t>;

  using FP8Elem = typename Kernel_traits::Element;
  using SmemLayoutQ_TMA  = typename Kernel_traits::SmemLayoutQ_TMA;
  using SmemLayoutK_TMA  = typename Kernel_traits::SmemLayoutK_TMA;
  using SmemLayoutVt_TMA = typename Kernel_traits::SmemLayoutVt_TMA;

  // Q TMA descriptor: Q described as [total_q, d, h] with strides [q_row_stride, 1, q_head_stride].
  auto tensor_Q_full = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<FP8Elem*>(params.q_ptr)),
      cute::make_layout(
          cute::make_shape(params.total_q, params.d, params.h),
          cute::make_stride(params.q_row_stride, cute::_1{}, params.q_head_stride)));
  auto tma_q = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_Q_full,
      SmemLayoutQ_TMA{},
      cute::make_shape(cute::Int<kBlockM>{}, cute::Int<kHeadDim>{}),
      cute::_1{});

  // K TMA descriptor: K described as [total_k, d, h_k] with strides [k_row_stride, 1, k_head_stride].
  auto tensor_K_full = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<FP8Elem*>(params.k_ptr)),
      cute::make_layout(
          cute::make_shape(params.total_k, params.d, params.h_k),
          cute::make_stride(params.k_row_stride, cute::_1{}, params.k_head_stride)));
  auto tma_k = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_K_full,
      SmemLayoutK_TMA{},
      cute::make_shape(cute::Int<kBlockN>{}, cute::Int<kHeadDim>{}),
      cute::_1{});

  // V TMA: GMEM described as [total_k, d, h_k] strides [v_row_stride, 1, v_head_stride].
  // Row-major V: v_row_stride=d (token stride), d-stride=1.
  // Tile = [kBlockN, kHeadDim]: dim 0 = n_k (kBlockN), dim 1 = d (kHeadDim).
  // SmemLayoutVt_TMA uses K_SW128 [kBlockN, kHeadDim] → d is SMEM fast axis (K_SW128 convention).
  auto tensor_Vt_full = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<FP8Elem*>(params.v_ptr)),
      cute::make_layout(
          cute::make_shape(params.total_k, params.d, params.h_k),
          cute::make_stride(params.v_row_stride, cute::_1{}, params.v_head_stride)));
  auto tma_vt = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_Vt_full,
      SmemLayoutVt_TMA{},
      cute::make_shape(cute::Int<kBlockN>{}, cute::Int<kHeadDim>{}),
      cute::_1{});

  // SFA: Q scale factors — int32 [q_block_descale_head_stride, 1, h] with strides [1, stride, stride].
  // Tile [kBlockM, 1]: kBlockM int32 (= 512B) per head per Q tile.
  // SmemLayoutSFA_TMA = Layout<[kBlockM, 1], [1, kBlockM]> matches SmemLayoutSFA{}(_,_,_0{}) from BS1.
  using SmemLayoutSFA_TMA_t = cute::Layout<cute::Shape<cute::Int<kBlockM>, cute::Int<1>>,
                                           cute::Stride<cute::_1, cute::Int<kBlockM>>>;
  // Only build SFA TMA descriptor when SF pointer is valid (always true in WS FP8 block-scale path).
  TORCH_CHECK(params.sf_q_packed_ptr != nullptr,
              "run_hstu_fwd_sm120_fp8_ws_tma_impl: sf_q_packed_ptr must be non-null");
  auto tensor_SFA_full = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<int32_t*>(params.sf_q_packed_ptr)),
      cute::make_layout(
          cute::make_shape((int64_t)params.q_block_descale_head_stride, cute::Int<1>{}, params.h),
          cute::make_stride(cute::_1{}, (int64_t)params.q_block_descale_head_stride,
                            (int64_t)params.q_block_descale_head_stride)));
  auto tma_sfa = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_SFA_full,
      SmemLayoutSFA_TMA_t{},
      cute::make_shape(cute::Int<kBlockM>{}, cute::Int<1>{}),
      cute::_1{});

  // SFB: K scale factors — int32 [kv_block_descale_head_stride, 1, h_k] with strides [1, stride, stride].
  // Tile [kBlockN, 1]: kBlockN int32 (= 512B) per KV head per K tile.
  // SmemLayoutSFB_TMA = Layout<[kBlockN, 1], [1, kBlockN]> matches SmemLayoutSFB{}(_,_,_0{}) from BS1.
  using SmemLayoutSFB_TMA_t = cute::Layout<cute::Shape<cute::Int<kBlockN>, cute::Int<1>>,
                                           cute::Stride<cute::_1, cute::Int<kBlockN>>>;
  TORCH_CHECK(params.sf_k_packed_ptr != nullptr,
              "run_hstu_fwd_sm120_fp8_ws_tma_impl: sf_k_packed_ptr must be non-null");
  auto tensor_SFB_full = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<int32_t*>(params.sf_k_packed_ptr)),
      cute::make_layout(
          cute::make_shape((int64_t)params.kv_block_descale_head_stride, cute::Int<1>{}, params.h_k),
          cute::make_stride(cute::_1{}, (int64_t)params.kv_block_descale_head_stride,
                            (int64_t)params.kv_block_descale_head_stride)));
  auto tma_sfb = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_SFB_full,
      SmemLayoutSFB_TMA_t{},
      cute::make_shape(cute::Int<kBlockN>{}, cute::Int<1>{}),
      cute::_1{});

  // SFV: V scale factors — same layout as SFB but using v_block_descale_head_stride and sf_v_packed_ptr.
  // After Python-side expansion, sf_v_packed has shape [H, total_tokens] with same kBlockN tile layout.
  // SmemLayoutSFV_TMA = Layout<[kBlockN, 1], [1, kBlockN]> (same as SFB).
  using SmemLayoutSFV_TMA_t = cute::Layout<cute::Shape<cute::Int<kBlockN>, cute::Int<1>>,
                                           cute::Stride<cute::_1, cute::Int<kBlockN>>>;
  TORCH_CHECK(params.sf_v_packed_ptr != nullptr,
              "run_hstu_fwd_sm120_fp8_ws_tma_impl: sf_v_packed_ptr must be non-null");
  auto tensor_SFV_full = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<int32_t*>(params.sf_v_packed_ptr)),
      cute::make_layout(
          cute::make_shape((int64_t)params.v_block_descale_head_stride, cute::Int<1>{}, params.h_k),
          cute::make_stride(cute::_1{}, (int64_t)params.v_block_descale_head_stride,
                            (int64_t)params.v_block_descale_head_stride)));
  auto tma_sfv = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_SFV_full,
      SmemLayoutSFV_TMA_t{},
      cute::make_shape(cute::Int<kBlockN>{}, cute::Int<1>{}),
      cute::_1{});

  using TMA_Q_t   = decltype(tma_q);
  using TMA_K_t   = decltype(tma_k);
  using TMA_Vt_t  = decltype(tma_vt);
  using TMA_SFA_t = decltype(tma_sfa);
  using TMA_SFB_t = decltype(tma_sfb);
  using TMA_SFV_t = decltype(tma_sfv);

  Hstu_fwd_params_fp8_ws_tma<TMA_Q_t, TMA_K_t, TMA_Vt_t, TMA_SFA_t, TMA_SFB_t, TMA_SFV_t> tma_params;
  static_cast<Hstu_fwd_params&>(tma_params) = params;
  tma_params.tma_q   = tma_q;
  tma_params.tma_k   = tma_k;
  tma_params.tma_vt  = tma_vt;
  tma_params.tma_sfa = tma_sfa;
  tma_params.tma_sfb = tma_sfb;
  tma_params.tma_sfv = tma_sfv;

  size_t smem_size = Kernel_traits::kSmemSize;
  const int num_m_block = (params.seqlen_q + kBlockM - 1) / kBlockM;
  dim3 grid = dim3(num_m_block, params.h, params.b);
  auto kernel = &flash::hstu_fwd_kernel_sm120_fp8_ws_tma<
      Kernel_traits, TMA_Q_t, TMA_K_t, TMA_Vt_t, TMA_SFA_t, TMA_SFB_t, TMA_SFV_t>;

  if (smem_size >= 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(tma_params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Arch,
    typename elem_type,
    int kHeadDim,
    bool Has_rab,
    bool Is_local,
    bool Is_causal,
    bool Is_context,
    bool Is_target,
    bool Is_arbitrary,
    int kNFunc>
void run_hstu_fwd_sm120(Hstu_fwd_params& params, cudaStream_t stream) {
  constexpr bool Is_fp8_type = std::is_same_v<elem_type, cutlass::float_e4m3_t>;

  static constexpr auto tile_size = flash::get_tile_size_fwd_sm120<kHeadDim, Has_rab, Is_fp8_type>();
  static constexpr int kBlockM = std::get<0>(tile_size);
  static constexpr int kBlockN = std::get<1>(tile_size);
  static constexpr int kNWarps = std::get<2>(tile_size);
  static constexpr bool Is_Q_in_regs = kHeadDim <= 128;
  static constexpr bool Share_Q_K_smem = kHeadDim <= 128;

  const int num_m_block = (params.seqlen_q + kBlockM - 1) / kBlockM;
  dim3 grid = dim3(num_m_block, params.h, params.b);

  // Shared launch helper: sets smem attribute if needed, launches kernel, checks error.
  auto launch_kernel = [&](auto kernel, int n_threads, size_t smem_size) {
    if (smem_size >= 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<grid, n_threads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };

  if constexpr (Is_fp8_type) {
    if constexpr ((kBlockN % 128) == 0) {
      if constexpr (!Has_rab) {
        // WS TMA kernel: load warp issues TMA for Q/K/V^T/SFB/SFV; math warps do QMMA.
        // Has_rab=true is not supported in WS kernel; handled by cp.async fallback below.
        run_hstu_fwd_sm120_fp8_ws_tma_impl<
            elem_type, kHeadDim, kBlockM, kBlockN, kNWarps,
            Is_causal, Is_target, Is_context, Is_local, Is_arbitrary, kNFunc, Has_rab,
            Is_Q_in_regs, Share_Q_K_smem>(params, stream);
      } else {
        // cp.async fallback for Has_rab=true (WS kernel does not support RAB).
        using Kernel_traits = Hstu_fwd_kernel_traits_sm120_fp8<
            kHeadDim, kBlockM, kBlockN, kNWarps,
            Is_causal, Is_target, Is_context, Is_local, Is_arbitrary, kNFunc, Has_rab,
            Is_Q_in_regs, Share_Q_K_smem, cutlass::half_t>;
        auto kernel = &flash::hstu_fwd_kernel_sm120<Kernel_traits, Hstu_fwd_params>;
        launch_kernel(kernel, Kernel_traits::kNThreads, Kernel_traits::kSmemSize);
      }
    } else {
      // Compile-time gate: do not instantiate FP8 blockscaled kernel for unsupported N tiles.
      TORCH_CHECK(
          false,
          "SM120 FP8 blockscaled path currently requires kBlockN divisible by 128, got kBlockN=",
          kBlockN);
    }
  } else {
    using Kernel_traits = Hstu_fwd_kernel_traits_sm120<
        kHeadDim, kBlockM, kBlockN, kNWarps,
        Is_causal, Is_target, Is_context, Is_local, Is_arbitrary, kNFunc, Has_rab,
        Is_Q_in_regs, Share_Q_K_smem, elem_type>;
    auto kernel = &flash::hstu_fwd_kernel_sm120<Kernel_traits, Hstu_fwd_params>;
    launch_kernel(kernel, Kernel_traits::kNThreads, Kernel_traits::kSmemSize);
  }
}
