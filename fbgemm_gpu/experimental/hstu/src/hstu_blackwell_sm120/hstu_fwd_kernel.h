/*
 * Copyright (c) 2023, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SM120 (Blackwell consumer) HSTU forward kernels.
// BF16/FP16 uses the per-CTA cp.async kernel. FP8 block-scale quant_mode=2
// uses the warp-specialized TMA kernel in hstu_fwd_kernel_fp8_ws.h.

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

// Main HSTU forward attention kernel for SM120 BF16/FP16.
// SM120 FP8 forward dispatch uses the WS TMA kernel below.
template <typename Kernel_traits, typename Params>
inline __device__ void hstu_compute_attn_1rowblock_sm120(
    const Params& params,
    const int bidb,
    const int bidh,
    int m_block) {
  static_assert(
      !Kernel_traits::Is_fp8,
      "SM120 FP8 forward uses the WS TMA kernel; non-WS FP8 dispatch is removed.");
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

  // SMEM tensors for the BF16/FP16 per-CTA path.
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
  // GMEM → SMEM copies for the BF16/FP16 per-CTA path.
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

// WS TMA params: extends Hstu_fwd_params with TMA Q, K, V^T, and SF descriptors.
// TMA_Q_t / TMA_K_t / TMA_Vt_t are TMA copy atoms for FP8 data.
// TMA_SFA_t / TMA_SFB_t are TMA copy atoms for int32 SF data.
template <typename TMA_Q_t, typename TMA_K_t, typename TMA_Vt_t,
          typename TMA_SFA_t, typename TMA_SFB_t, typename TMA_SFV_t,
          typename TMA_O_t, typename TMA_Rab_t>
struct Hstu_fwd_params_fp8_ws_tma : public Hstu_fwd_params {
    TMA_Q_t   tma_q;   // TMA descriptor for Q:        [total_q, d, h]
    TMA_K_t   tma_k;   // TMA descriptor for K:        [total_k, d, h_k]
    TMA_Vt_t  tma_vt;  // TMA descriptor for V (row-major): [total_k, d, h_k]
    TMA_SFA_t tma_sfa; // TMA descriptor for Q scale:  [q_block_descale_head_stride, 1, h]
    TMA_SFB_t tma_sfb; // TMA descriptor for K scale:  [kv_block_descale_head_stride, 1, h_k]
    TMA_SFV_t tma_sfv; // TMA descriptor for V scale:  [v_block_descale_head_stride, 1, h_k]
    TMA_O_t   tma_o;   // TMA descriptor for O store:  [total_q, d, h] (same dim ordering as tma_q)
    TMA_Rab_t tma_rab; // TMA descriptor for RAB:      [seqlen_k_rounded, seqlen_k_rounded, h_rab, b]
};

template <typename TMA_Q_t, typename TMA_K_t, typename TMA_Vt_t,
          typename TMA_SFA_t, typename TMA_SFB_t, typename TMA_SFV_t,
          typename TMA_O_t, typename TMA_Rab_t,
          typename TMA_KPage_t, typename TMA_VtPage_t>
struct Hstu_fwd_params_fp8_ws_tma_paged
    : public Hstu_fwd_params_fp8_ws_tma<
          TMA_Q_t, TMA_K_t, TMA_Vt_t, TMA_SFA_t, TMA_SFB_t, TMA_SFV_t,
          TMA_O_t, TMA_Rab_t> {
    TMA_KPage_t  tma_k_page;   // TMA descriptor for paged K: [page_size, d, h_k, total_pages]
    TMA_VtPage_t tma_vt_page;  // TMA descriptor for paged V: [page_size, d, h_k, total_pages]
};

namespace flash {  // reopen flash namespace for kernel

#include "hstu_fwd_kernel_fp8_ws.h"  // WS kernel

} // namespace flash

////////////////////////////////////////////////////////////////////////////////////////////////////

// WS TMA impl: creates TMA Q, K, and V^T descriptors on host, launches WS TMA kernel.
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
    bool Paged_KV = false,
    bool Is_Q_in_regs = false,
    bool Share_Q_K_smem = false>
void run_hstu_fwd_sm120_fp8_ws_tma_impl(Hstu_fwd_params& params, cudaStream_t stream) {
  using Kernel_traits = Hstu_fwd_kernel_traits_sm120_fp8_ws<
      kHeadDim, kBlockM, kBlockN, kNWarps,
      Is_causal, Is_target, Is_context, Is_local, Is_arbitrary, kNFunc, Has_rab,
      Paged_KV, Is_Q_in_regs, Share_Q_K_smem, cutlass::half_t>;

  using FP8Elem = typename Kernel_traits::Element;
  using RabElement = cutlass::bfloat16_t;
  using SmemLayoutQ_TMA  = typename Kernel_traits::SmemLayoutQ_TMA;
  using SmemLayoutK_TMA  = typename Kernel_traits::SmemLayoutK_TMA;
  using SmemLayoutVt_TMA = typename Kernel_traits::SmemLayoutVt_TMA;
  using SmemLayoutRab_TMA = typename Kernel_traits::SmemLayoutRab_TMA;

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

  // O TMA STORE descriptor: BF16 output described as [total_q, d, h] with strides
  // [o_row_stride, 1, o_head_stride] — same dimension ordering as the Q/K/V TMA LOADs.
  // Tile = [kBlockM, kHeadDim] over the (total_q, d) plane; head is indexed via get_tma_tensor.
  using OutElement = typename Kernel_traits::OutputType;
  using SmemLayoutO_TMA_t = cute::Layout<cute::Shape<cute::Int<kBlockM>, cute::Int<kHeadDim>>,
                                         cute::Stride<cute::Int<kHeadDim>, cute::_1>>;
  auto tensor_O_full = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<OutElement*>(params.o_ptr)),
      cute::make_layout(
          cute::make_shape(params.total_q, params.d, params.h),
          cute::make_stride(params.o_row_stride, cute::_1{}, params.o_head_stride)));
  auto tma_o = cute::make_tma_copy(
      cute::SM90_TMA_STORE{},
      tensor_O_full,
      SmemLayoutO_TMA_t{},
      cute::make_shape(cute::Int<kBlockM>{}, cute::Int<kHeadDim>{}),
      cute::_1{});

  // RAB TMA descriptor.  Has_rab=false still builds a dummy descriptor with
  // valid box dimensions; the kernel never issues it in that specialization.
  RabElement* rab_base = Has_rab
      ? static_cast<RabElement*>(params.rab_ptr)
      : reinterpret_cast<RabElement*>(params.q_ptr);
  const int64_t rab_dim0 = Has_rab ? (int64_t)params.seqlen_k_rounded : (int64_t)kBlockM;
  const int64_t rab_dim1 = Has_rab ? (int64_t)params.seqlen_k_rounded : (int64_t)kBlockN;
  const int64_t rab_heads = Has_rab ? (int64_t)params.h_rab : 1;
  const int64_t rab_batch = Has_rab ? (int64_t)params.b : 1;
  const int64_t rab_stride_m = Has_rab ? (int64_t)params.rab_seqlen_k_stride : (int64_t)kBlockN;
  const int64_t rab_stride_h = Has_rab ? (int64_t)params.rab_seqlen_q_stride
                                       : (int64_t)kBlockM * kBlockN;
  const int64_t rab_stride_b = Has_rab ? (int64_t)params.rab_seqlen_qk_stride
                                       : (int64_t)kBlockM * kBlockN;
  auto tensor_Rab_full = cute::make_tensor(
      cute::make_gmem_ptr(rab_base),
      cute::make_layout(
          cute::make_shape(rab_dim0, rab_dim1, rab_heads, rab_batch),
          cute::make_stride(rab_stride_m, cute::_1{}, rab_stride_h, rab_stride_b)));
  auto tma_rab = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_Rab_full,
      SmemLayoutRab_TMA{}(_, _, _0{}),
      cute::make_shape(cute::Int<kBlockM>{}, cute::Int<kBlockN>{}),
      cute::_1{});

  using TMA_Q_t   = decltype(tma_q);
  using TMA_K_t   = decltype(tma_k);
  using TMA_Vt_t  = decltype(tma_vt);
  using TMA_SFA_t = decltype(tma_sfa);
  using TMA_SFB_t = decltype(tma_sfb);
  using TMA_SFV_t = decltype(tma_sfv);
  using TMA_O_t   = decltype(tma_o);
  using TMA_Rab_t = decltype(tma_rab);

  size_t smem_size = Kernel_traits::kSmemSize;
  const int num_m_block = (params.seqlen_q + kBlockM - 1) / kBlockM;
  const int total_tiles = num_m_block * params.h * params.b;
  const int total_tile_pairs = (total_tiles + 1) / 2;
  static constexpr bool Use_full_persistent =
      !Is_causal && !Is_target && !Is_context && !Is_local && !Is_arbitrary;
  static constexpr bool Use_paired_persistent =
      Is_causal && !Is_context && !Is_local && !Is_arbitrary &&
      (!Is_target || Paged_KV) &&
      (kHeadDim <= 128 || (Paged_KV && !Has_rab));
  static constexpr bool Use_persistent = Use_full_persistent || Use_paired_persistent;
  const int persistent_work_units = Use_paired_persistent ? total_tile_pairs : total_tiles;
  int sm_count = 0;
  if constexpr (Use_persistent) {
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  }
  dim3 grid = Use_persistent
      ? dim3(std::min(persistent_work_units, sm_count))
      : (Has_rab && params.h_rab == 1 && params.h > 1
          ? dim3(params.h, num_m_block, params.b)
          : dim3(num_m_block, params.h, params.b));

  auto launch_tma_kernel = [&](auto& tma_params) {
    using TmaParamsT = std::decay_t<decltype(tma_params)>;
    auto kernel = &flash::hstu_fwd_kernel_sm120_fp8_ws_tma<Kernel_traits, TmaParamsT>;
    if (smem_size >= 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(tma_params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };

  if constexpr (Paged_KV) {
    // Paged K/V TMA descriptors.  The physical page id is a regular TMA coordinate:
    // kv_cache layout is [total_pages, 2, page_size, h_k, d], but the descriptor
    // is exposed as [page_size, d, h_k, total_pages] so each paged history N tile
    // is one TMA box at (page_row_tile=0, d_tile=0, head, page_id).
    auto tensor_K_page = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<FP8Elem*>(params.kv_cache_ptr)),
        cute::make_layout(
            cute::make_shape(params.page_size, params.d, params.h_k, params.total_pages),
            cute::make_stride(
                params.kv_cache_head_stride, cute::_1{},
                params.kv_cache_row_stride, params.kv_cache_kvtensor_stride)));
    auto tma_k_page = cute::make_tma_copy(
        cute::SM90_TMA_LOAD{},
        tensor_K_page,
        SmemLayoutK_TMA{},
        cute::make_shape(cute::Int<kBlockN>{}, cute::Int<kHeadDim>{}),
        cute::_1{});
    auto tensor_Vt_page = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<FP8Elem*>(params.kv_cache_ptr) + params.kv_cache_page_stride),
        cute::make_layout(
            cute::make_shape(params.page_size, params.d, params.h_k, params.total_pages),
            cute::make_stride(
                params.kv_cache_head_stride, cute::_1{},
                params.kv_cache_row_stride, params.kv_cache_kvtensor_stride)));
    auto tma_vt_page = cute::make_tma_copy(
        cute::SM90_TMA_LOAD{},
        tensor_Vt_page,
        SmemLayoutVt_TMA{},
        cute::make_shape(cute::Int<kBlockN>{}, cute::Int<kHeadDim>{}),
        cute::_1{});

    using TMA_KPage_t  = decltype(tma_k_page);
    using TMA_VtPage_t = decltype(tma_vt_page);
    Hstu_fwd_params_fp8_ws_tma_paged<
        TMA_Q_t, TMA_K_t, TMA_Vt_t, TMA_SFA_t, TMA_SFB_t, TMA_SFV_t, TMA_O_t,
        TMA_Rab_t, TMA_KPage_t, TMA_VtPage_t> tma_params;
    static_cast<Hstu_fwd_params&>(tma_params) = params;
    tma_params.tma_q   = tma_q;
    tma_params.tma_k   = tma_k;
    tma_params.tma_vt  = tma_vt;
    tma_params.tma_sfa = tma_sfa;
    tma_params.tma_sfb = tma_sfb;
    tma_params.tma_sfv = tma_sfv;
    tma_params.tma_o   = tma_o;
    tma_params.tma_rab = tma_rab;
    tma_params.tma_k_page = tma_k_page;
    tma_params.tma_vt_page = tma_vt_page;
    launch_tma_kernel(tma_params);
  } else {
    Hstu_fwd_params_fp8_ws_tma<
        TMA_Q_t, TMA_K_t, TMA_Vt_t, TMA_SFA_t, TMA_SFB_t, TMA_SFV_t, TMA_O_t,
        TMA_Rab_t> tma_params;
    static_cast<Hstu_fwd_params&>(tma_params) = params;
    tma_params.tma_q   = tma_q;
    tma_params.tma_k   = tma_k;
    tma_params.tma_vt  = tma_vt;
    tma_params.tma_sfa = tma_sfa;
    tma_params.tma_sfb = tma_sfb;
    tma_params.tma_sfv = tma_sfv;
    tma_params.tma_o   = tma_o;
    tma_params.tma_rab = tma_rab;
    launch_tma_kernel(tma_params);
  }
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

  static constexpr auto tile_size =
      flash::get_tile_size_fwd_sm120<kHeadDim, Has_rab, Is_fp8_type, Is_arbitrary>();
  static constexpr int kBlockM = std::get<0>(tile_size);
  static constexpr int kBlockN = std::get<1>(tile_size);
  static constexpr int kNWarps = std::get<2>(tile_size);
  static constexpr bool Is_Q_in_regs = kHeadDim <= 128;
  static constexpr bool Share_Q_K_smem = kHeadDim <= 128;

  if constexpr (Is_fp8_type) {
    static_assert(
        kBlockN == 64,
        "SM120 FP8 forward now routes through the WS TMA path, which expects kBlockN=64.");
    BOOL_SWITCH(params.is_paged_kv, Paged_KV, [&] {
      if constexpr (Paged_KV) {
        TORCH_CHECK(
            params.page_size == kBlockN,
            "SM120 FP8 paged KV WS path requires page_size == kBlockN, got page_size=",
            params.page_size,
            ", kBlockN=",
            kBlockN);
      }
      run_hstu_fwd_sm120_fp8_ws_tma_impl<
          elem_type, kHeadDim, kBlockM, kBlockN, kNWarps,
          Is_causal, Is_target, Is_context, Is_local, Is_arbitrary, kNFunc, Has_rab,
          Paged_KV, Is_Q_in_regs, Share_Q_K_smem>(params, stream);
    });
  } else {
    const int num_m_block = (params.seqlen_q + kBlockM - 1) / kBlockM;
    dim3 grid = dim3(num_m_block, params.h, params.b);

    using Kernel_traits = Hstu_fwd_kernel_traits_sm120<
        kHeadDim, kBlockM, kBlockN, kNWarps,
        Is_causal, Is_target, Is_context, Is_local, Is_arbitrary, kNFunc, Has_rab,
        Is_Q_in_regs, Share_Q_K_smem, elem_type>;
    auto kernel = &flash::hstu_fwd_kernel_sm120<Kernel_traits, Hstu_fwd_params>;
    size_t smem_size = Kernel_traits::kSmemSize;
    if (smem_size >= 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
