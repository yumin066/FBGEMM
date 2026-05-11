#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <type_traits>

#include <cutlass/numeric_conversion.h>

#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/tensor.hpp>

#include "block_info.h"
#include "hstu.h"
#include "hstu_fwd_kernel_bf16.h"
#include "kernel_traits.h"
#include "static_switch.h"
#include "sm120_qmma_builder.h"
#include "utils.h"

template <typename TMA_Q_t, typename TMA_K_t, typename TMA_Vt_t,
          typename TMA_SFA_t, typename TMA_SFB_t, typename TMA_SFV_t,
          typename TMA_O_t, typename TMA_Rab_t>
struct Hstu_fwd_params_fp8_ws_tma : public Hstu_fwd_params {
    TMA_Q_t   tma_q;
    TMA_K_t   tma_k;
    TMA_Vt_t  tma_vt;
    TMA_SFA_t tma_sfa;
    TMA_SFB_t tma_sfb;
    TMA_SFV_t tma_sfv;
    TMA_O_t   tma_o;
    TMA_Rab_t tma_rab;
};

template <typename TMA_Q_t, typename TMA_K_t, typename TMA_Vt_t,
          typename TMA_SFA_t, typename TMA_SFB_t, typename TMA_SFV_t,
          typename TMA_O_t, typename TMA_Rab_t,
          typename TMA_KPage_t, typename TMA_VtPage_t>
struct Hstu_fwd_params_fp8_ws_tma_paged
    : public Hstu_fwd_params_fp8_ws_tma<
          TMA_Q_t, TMA_K_t, TMA_Vt_t, TMA_SFA_t, TMA_SFB_t, TMA_SFV_t,
          TMA_O_t, TMA_Rab_t> {
    TMA_KPage_t  tma_k_page;
    TMA_VtPage_t tma_vt_page;
};

namespace flash {
using namespace cute;
#include "hstu_fwd_kernel_fp8_ws.h"
} // namespace flash

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

  using SmemLayoutSFA_TMA_t = cute::Layout<cute::Shape<cute::Int<kBlockM>, cute::Int<1>>,
                                           cute::Stride<cute::_1, cute::Int<kBlockM>>>;
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

  using OutElement = typename Kernel_traits::OutputType;
  using SmemLayoutO_TMA_t = typename Kernel_traits::SmemLayoutWsO_TMA;
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
      SmemLayoutRab_TMA{}(cute::_, cute::_, cute::_0{}),
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
  static constexpr bool Use_paired_persistent =
      Is_causal &&
      !Is_context &&
      !Is_local &&
      !Is_arbitrary &&
      (!Is_target || Paged_KV) &&
      (kHeadDim <= 128 || (Paged_KV && !Has_rab));
  static constexpr bool Use_full_persistent =
      !Is_causal && !Is_target && !Is_context && !Is_local && !Is_arbitrary;
  static constexpr bool Use_persistent = Use_full_persistent || Use_paired_persistent;
  const int persistent_work_units = Use_paired_persistent ? total_tile_pairs : total_tiles;
  int device = 0;
  int sm_count = 0;
  if constexpr (Use_persistent) {
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  }
  dim3 grid = Use_persistent
      ? dim3(std::min(persistent_work_units, sm_count))
      : (Has_rab && params.h_rab == 1 && params.h > 1
          ? dim3(params.h, num_m_block, params.b)
          : dim3(num_m_block, params.h, params.b));
  static constexpr bool Can_use_target_group_size_1 =
      kHeadDim > 128 &&
      Is_causal &&
      Is_target &&
      !Is_context &&
      !Is_local &&
      !Is_arbitrary &&
      !Has_rab &&
      !Paged_KV;

  auto launch_tma_kernel = [&](auto& tma_params) {
    using TmaParamsT = std::decay_t<decltype(tma_params)>;
    if constexpr (Can_use_target_group_size_1) {
      if (params.target_group_size == 1) {
        auto kernel =
            &flash::hstu_fwd_kernel_sm120_fp8_ws_tma<Kernel_traits, TmaParamsT, true>;
        if (smem_size >= 48 * 1024) {
          static std::atomic<bool> smem_attr_set_g1{false};
          if (!smem_attr_set_g1.load(std::memory_order_acquire)) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            smem_attr_set_g1.store(true, std::memory_order_release);
          }
        }
        kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(tma_params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
      }
    }
    auto kernel =
        &flash::hstu_fwd_kernel_sm120_fp8_ws_tma<Kernel_traits, TmaParamsT, false>;
    if (smem_size >= 48 * 1024) {
      static std::atomic<bool> smem_attr_set{false};
      if (!smem_attr_set.load(std::memory_order_acquire)) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        smem_attr_set.store(true, std::memory_order_release);
      }
    }
    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(tma_params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };

  if constexpr (Paged_KV) {
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
    auto kernel = &flash::hstu_fwd_kernel_sm120_bf16<Kernel_traits, Hstu_fwd_params>;
    size_t smem_size = Kernel_traits::kSmemSize;
    if (smem_size >= 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
