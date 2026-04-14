/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SM120 (Blackwell consumer) HSTU forward attention ops.
// Registered as the hstu_varlen_fwd_120 operator.
// Supports BF16 and FP8 (quant_mode >= 0) forward pass.

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/nn/functional.h>

#include <cstdlib>
#include <optional>

#include "hstu.h"
#include "static_switch.h"

namespace fbgemm_gpu::hstu {

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                           \
  TORCH_CHECK(                                        \
      x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
      #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

////////////////////////////////////////////////////////////////////////////////////////////////////

void set_params_fprop_sm120(
    Hstu_fwd_params* params,
    const size_t b,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t scaling_seqlen,
    const size_t target_group_size,
    const size_t seqlen_q_rounded,
    const size_t seqlen_k_rounded,
    const size_t h,
    const size_t h_k,
    const size_t h_rab,
    const size_t d,
    const float alpha,
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    const at::Tensor rab,
    at::Tensor out,
    void* num_contexts_d,
    void* cu_seqlens_q_d,
    void* cu_seqlens_k_d,
    void* seqused_q_d,
    void* seqused_k_d,
    void* num_targets_d,
    bool has_rab,
    int quant_mode,
    const std::optional<at::Tensor>& func,
    int window_size_left,
    int window_size_right,
    // FP8 descale factors (optional, required when quant_mode >= 0)
    const std::optional<at::Tensor>& descale_q,
    const std::optional<at::Tensor>& descale_k,
    const std::optional<at::Tensor>& descale_v) {
  *params = {};

  params->arch = at::cuda::getCurrentDeviceProperties()->major * 10 +
      at::cuda::getCurrentDeviceProperties()->minor;

  params->q_ptr = q.data_ptr();
  params->k_ptr = k.data_ptr();
  params->v_ptr = v.data_ptr();
  params->q_row_stride = q.stride(-3);
  params->k_row_stride = k.stride(-3);
  params->v_row_stride = v.stride(-3);
  params->q_head_stride = q.stride(-2);
  params->k_head_stride = k.stride(-2);
  params->v_head_stride = v.stride(-2);

  if (out.numel() > 0) {
    params->o_ptr = out.data_ptr();
    params->o_row_stride = out.stride(-3);
    params->o_head_stride = out.stride(-2);
  }

  params->has_rab = has_rab;
  if (has_rab) {
    params->rab_ptr = rab.data_ptr();
    params->rab_seqlen_qk_stride = rab.stride(-4);
    params->rab_seqlen_q_stride = rab.stride(-3);
    params->rab_seqlen_k_stride = rab.stride(-2);
    params->h_rab = h_rab;
  } else {
    params->rab_ptr = nullptr;
    params->rab_seqlen_qk_stride = 0;
    params->rab_seqlen_q_stride = 0;
    params->rab_seqlen_k_stride = 0;
    params->h_rab = 0;
  }

  params->num_contexts = static_cast<int*>(num_contexts_d);
  params->cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params->cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);
  params->num_targets = static_cast<int*>(num_targets_d);
  params->seqused_q = static_cast<int*>(seqused_q_d);
  params->seqused_k = static_cast<int*>(seqused_k_d);

  // FP8 mode: quant_mode >= 0
  params->quant_mode = quant_mode;
  params->is_bf16 = (quant_mode < 0) ? (q.dtype() == at::kBFloat16) : true;  // output is BF16
  params->is_e4m3 = (quant_mode >= 0);
  params->is_e5m2 = false;
  params->output_dtype = 0;  // 0 = BF16

  if (quant_mode >= 0) {
    // FP8 descale factors (per-tensor, quant_mode=0)
    if (descale_q.has_value()) {
      params->descale_q_ptr = static_cast<float*>(descale_q.value().data_ptr());
      params->descale_q_head_stride = descale_q.value().numel() > 1 ? 1 : 0;
    }
    if (descale_k.has_value()) {
      params->descale_k_ptr = static_cast<float*>(descale_k.value().data_ptr());
      params->descale_k_head_stride = descale_k.value().numel() > 1 ? 1 : 0;
    }
    if (descale_v.has_value()) {
      params->descale_v_ptr = static_cast<float*>(descale_v.value().data_ptr());
      params->descale_v_head_stride = descale_v.value().numel() > 1 ? 1 : 0;
    }
  }

  params->b = b;
  params->h = h;
  params->h_k = h_k;
  params->h_h_k_ratio = h / h_k;
  params->seqlen_q = seqlen_q;
  params->seqlen_k = seqlen_k;
  params->seqlen_q_rounded = seqlen_q_rounded;
  params->seqlen_k_rounded = seqlen_k_rounded;
  params->scaling_seqlen = (scaling_seqlen == 0 || scaling_seqlen == (size_t)-1) ? seqlen_q : scaling_seqlen;
  params->d = d;
  params->alpha = alpha;
  params->is_target = num_targets_d != nullptr;
  params->target_group_size = target_group_size;
  if (params->is_target) {
    TORCH_CHECK(target_group_size > 0, "target_group_size must be positive");
  }
  params->is_context = num_contexts_d != nullptr;

  if (window_size_left < 0 || window_size_left > (int)seqlen_k) window_size_left = seqlen_k;
  if (window_size_right < 0 || window_size_right > (int)seqlen_k) window_size_right = seqlen_k;
  params->window_size_left = window_size_left;
  params->window_size_right = window_size_right;
  params->is_causal = params->window_size_left == (int)seqlen_k && params->window_size_right == 0;
  params->is_local = (window_size_left < (int)seqlen_k || window_size_right < (int)seqlen_k) &&
      !params->is_causal;

  params->is_arbitrary_mask = func.has_value() && func.value().defined();
  if (params->is_arbitrary_mask) {
    TORCH_CHECK(func.value().dtype() == torch::kInt32, "func must have dtype int32");
    CHECK_DEVICE(func.value());
    params->func_ptr = func.value().data_ptr();
    params->func_ids_stride = func.value().stride(-2);
    params->func_head_stride = func.value().stride(-3);
    params->n_func = func.value().size(-2);
    TORCH_CHECK(params->n_func == HSTU_ARBITRARY_NFUNC, "n_func mismatch");
  }

  // No paged KV support in first Blackwell impl
  params->is_paged_kv = false;

  // Runtime EXP-A switch (no recompile needed):
  // export HSTU_EXP_A_SYNC_K=1 to disable K prefetch and reload K synchronously.
  // Backward-compatible alias: HSTU_EXP_A_SYNC_K_NO_PREFETCH.
  if (const char* exp_a = std::getenv("HSTU_EXP_A_SYNC_K")) {
    params->exp_a_sync_k = std::atoi(exp_a) != 0 ? 1 : 0;
  } else if (const char* exp_a_legacy = std::getenv("HSTU_EXP_A_SYNC_K_NO_PREFETCH")) {
    params->exp_a_sync_k = std::atoi(exp_a_legacy) != 0 ? 1 : 0;
  } else {
    params->exp_a_sync_k = 0;
  }

  // Runtime EXP-B switch (no recompile needed):
  // export HSTU_EXP_B_PBUF_IN_SK=1 to route P-roundtrip buffer to sK.
  if (const char* exp_b = std::getenv("HSTU_EXP_B_PBUF_IN_SK")) {
    params->exp_b_pbuf_in_sk = std::atoi(exp_b) != 0 ? 1 : 0;
  } else {
    params->exp_b_pbuf_in_sk = 0;
  }

  // Debug GEMM1-only bypass switch:
  // export HSTU_DEBUG_GEMM1_ONLY=1 to output raw GEMM1 scores instead of GEMM2 result.
  // Both BF16 and FP8 paths honour this flag, enabling direct kernel GEMM1 comparison.
  if (const char* g1only = std::getenv("HSTU_DEBUG_GEMM1_ONLY")) {
    params->debug_gemm1_only = std::atoi(g1only) != 0;
  } else {
    params->debug_gemm1_only = false;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype, bool Has_rab, bool Is_local, bool Is_causal,
          bool Is_context, bool Is_target, bool Is_arbitrary, int kNFunc>
void run_hstu_fwd_headdim_sm120(Hstu_fwd_params& params, cudaStream_t stream) {
  constexpr int Arch = 120;
#ifndef HSTU_DISABLE_HDIM64
  if (params.d == 64) {
    run_hstu_fwd_sm120<Arch, Dtype, 64, Has_rab, Is_local, Is_causal,
        Is_context, Is_target, Is_arbitrary, kNFunc>(params, stream);
    return;
  }
#endif
#ifndef HSTU_DISABLE_HDIM128
  if (params.d == 128) {
    run_hstu_fwd_sm120<Arch, Dtype, 128, Has_rab, Is_local, Is_causal,
        Is_context, Is_target, Is_arbitrary, kNFunc>(params, stream);
    return;
  }
#endif
  TORCH_CHECK(false, "Unsupported head dim: ", params.d, " (SM120 supports 64, 128)");
}

void run_hstu_fwd_blackwell(Hstu_fwd_params& params, cudaStream_t stream) {
  RAB_SWITCH(params.has_rab, Has_rab, [&] {
    // Dispatch on dtype: FP8 (quant_mode >= 0) or BF16/FP16
    if (params.quant_mode >= 0) {
      // FP8 mode: use float_e4m3_t as Element type
      using FP8Type = cutlass::float_e4m3_t;
#ifndef HSTU_DISABLE_ARBITRARY
      if (params.is_arbitrary_mask) {
        run_hstu_fwd_headdim_sm120<FP8Type, Has_rab, false, false, false, false, true, HSTU_ARBITRARY_NFUNC>(params, stream);
        return;
      }
#endif
#ifndef HSTU_DISABLE_LOCAL
      if (params.is_local) {
        run_hstu_fwd_headdim_sm120<FP8Type, Has_rab, true, false, false, false, false, 0>(params, stream);
        return;
      }
#endif
      if (!params.is_causal) {
        run_hstu_fwd_headdim_sm120<FP8Type, Has_rab, false, false, false, false, false, 0>(params, stream);
        return;
      }
#ifndef HSTU_DISABLE_CAUSAL
      CONTEXT_SWITCH(params.is_context, Is_context, [&] {
        TARGET_SWITCH(params.is_target, Is_target, [&] {
          run_hstu_fwd_headdim_sm120<FP8Type, Has_rab, false, true, Is_context, Is_target, false, 0>(params, stream);
        });
      });
#endif
    } else {
      // BF16/FP16 mode
      FP16_BF16_SWITCH(params.is_bf16, [&] {
#ifndef HSTU_DISABLE_ARBITRARY
        if (params.is_arbitrary_mask) {
          run_hstu_fwd_headdim_sm120<Dtype, Has_rab, false, false, false, false, true, HSTU_ARBITRARY_NFUNC>(params, stream);
          return;
        }
#endif
#ifndef HSTU_DISABLE_LOCAL
        if (params.is_local) {
          run_hstu_fwd_headdim_sm120<Dtype, Has_rab, true, false, false, false, false, 0>(params, stream);
          return;
        }
#endif
        if (!params.is_causal) {
          run_hstu_fwd_headdim_sm120<Dtype, Has_rab, false, false, false, false, false, 0>(params, stream);
          return;
        }
#ifndef HSTU_DISABLE_CAUSAL
        CONTEXT_SWITCH(params.is_context, Is_context, [&] {
          TARGET_SWITCH(params.is_target, Is_target, [&] {
            run_hstu_fwd_headdim_sm120<Dtype, Has_rab, false, true, Is_context, Is_target, false, 0>(params, stream);
          });
        });
#endif
      });
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Main entry point: varlen forward for SM120 (BF16 and FP8).
std::tuple<at::Tensor, at::Tensor> hstu_varlen_fwd_120(
    const at::Tensor& q,   // total_q x num_heads x head_size
    const at::Tensor& k,   // total_k x num_heads_k x head_size
    const at::Tensor& v,   // total_k x num_heads_k x head_size
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const std::optional<at::Tensor>& seqused_q,
    const std::optional<at::Tensor>& seqused_k,
    const int64_t max_seqlen_q,
    const int64_t max_seqlen_k,
    const int64_t scaling_seqlen,
    const std::optional<at::Tensor>& num_contexts,
    const std::optional<at::Tensor>& num_targets,
    const int64_t target_group_size,
    int64_t window_size_left,
    int64_t window_size_right,
    const double alpha,
    std::optional<at::Tensor> rab,
    const std::optional<at::Tensor>& func,
    // FP8 arguments (optional; if present, quant_mode >= 0)
    int64_t quant_mode,  // -1=BF16/FP16, 0=per-tensor FP8, 2=blockwise FP8
    const std::optional<at::Tensor>& descale_q,
    const std::optional<at::Tensor>& descale_k,
    const std::optional<at::Tensor>& descale_v,
    const std::optional<at::Tensor>& sf_q_packed,
    const std::optional<at::Tensor>& sf_k_packed,
    const std::optional<at::Tensor>& sf_v_packed,
    // Block-wise descale cumulative seqlens (quant_mode=2 only)
    const std::optional<at::Tensor>& cu_seqlens_q_block_descale,
    const std::optional<at::Tensor>& cu_seqlens_kv_block_descale,
    const std::optional<at::Tensor>& cu_seqlens_v_block_descale) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  const int arch = dprops->major * 10 + dprops->minor;
  TORCH_CHECK(arch >= 120, "hstu_varlen_fwd_120 requires SM120+ (Blackwell) GPU, got SM", arch);

  auto q_dtype = q.dtype();
  if (quant_mode >= 0) {
    TORCH_CHECK(q_dtype == at::kFloat8_e4m3fn,
        "FP8 mode requires float8_e4m3fn input, got ", q_dtype);
    TORCH_CHECK(k.dtype() == at::kFloat8_e4m3fn, "k must be float8_e4m3fn in FP8 mode");
    TORCH_CHECK(v.dtype() == at::kFloat8_e4m3fn, "v must be float8_e4m3fn in FP8 mode");
    TORCH_CHECK(quant_mode == 0 || quant_mode == 2,
        "SM120 supports quant_mode=0 (per-tensor FP8) or quant_mode=2 (blockwise FP8), got ", quant_mode);
    TORCH_CHECK(descale_q.has_value(), "descale_q required for FP8 mode");
    TORCH_CHECK(descale_k.has_value(), "descale_k required for FP8 mode");
    TORCH_CHECK(descale_v.has_value(), "descale_v required for FP8 mode");
    if (quant_mode == 2) {
      // Phase 1: cu_seqlens_q/kv_block_descale are optional (kernel uses unit SF).
      if (sf_q_packed.has_value()) {
        TORCH_CHECK(sf_q_packed.value().dtype() == at::kInt, "sf_q_packed must be int32");
      }
      if (sf_k_packed.has_value()) {
        TORCH_CHECK(sf_k_packed.value().dtype() == at::kInt, "sf_k_packed must be int32");
      }
      if (sf_v_packed.has_value()) {
        TORCH_CHECK(sf_v_packed.value().dtype() == at::kInt, "sf_v_packed must be int32");
      }
    }
  } else {
    TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
        "BF16/FP16 mode requires fp16 or bf16 input");
    TORCH_CHECK(k.dtype() == q_dtype, "k must have same dtype as q");
    TORCH_CHECK(v.dtype() == q_dtype, "v must have same dtype as q");
  }

  TORCH_CHECK(cu_seqlens_q.dtype() == at::kInt, "cu_seqlens_q must be int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == at::kInt, "cu_seqlens_k must be int32");

  CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
  CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);
  CHECK_CONTIGUOUS(cu_seqlens_q); CHECK_CONTIGUOUS(cu_seqlens_k);

  const int batch_size = cu_seqlens_q.numel() - 1;
  const int num_heads = q.size(1);
  const int head_size = q.size(2);
  const int total_k = k.size(0);
  const int num_heads_k = k.size(1);

  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
  CHECK_SHAPE(k, total_k, num_heads_k, head_size);
  CHECK_SHAPE(v, total_k, num_heads_k, head_size);

  TORCH_CHECK(batch_size > 0, "batch_size must be positive");
  TORCH_CHECK(num_heads == num_heads_k, "num_heads_k must equal num_heads");
  TORCH_CHECK(head_size == 64 || head_size == 128,
      "SM120 supports head_size 64 or 128, got ", head_size);
  TORCH_CHECK(q.stride(-1) == 1, "q must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "k must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "v must have contiguous last dimension");

  // FP8 mode: output is float16 (matches q_raw dtype from which q was quantized).
  // float16 has 10 mantissa bits (step=0.0625 at 64) vs BF16's 7 (step=0.5 at 64),
  // so float16 can represent values like 64.25 that BF16 would round to 64.0.
  at::Tensor out;
  if (quant_mode >= 0) {
    out = torch::empty({q.size(0), num_heads, head_size},
        q.options().dtype(at::kHalf));
  } else {
    out = torch::empty_like(q);
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  // Round to 16 for alignment (works for both FP8 and BF16)
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, 16);
  const int seqlen_k_rounded = round_multiple(max_seqlen_k, 16);

  bool has_rab = rab.has_value();
  int num_heads_rab = num_heads;
  if (has_rab) {
    num_heads_rab = rab.value().size(1);
    CHECK_DEVICE(rab.value());
    TORCH_CHECK(rab.value().stride(-1) == 1, "rab must have contiguous last dimension");
    TORCH_CHECK(num_heads == num_heads_rab || num_heads_rab == 1, "rab num_heads mismatch");
    CHECK_SHAPE(rab.value(), batch_size, num_heads_rab, max_seqlen_k, max_seqlen_k);
    if (seqlen_k_rounded != max_seqlen_k) {
      rab = torch::nn::functional::pad(
          rab.value(),
          torch::nn::functional::PadFuncOptions({0, seqlen_k_rounded - max_seqlen_k}));
    }
  }

  if (seqused_q.has_value()) {
    CHECK_DEVICE(seqused_q.value());
    CHECK_CONTIGUOUS(seqused_q.value());
    CHECK_SHAPE(seqused_q.value(), batch_size);
  }
  if (seqused_k.has_value()) {
    CHECK_DEVICE(seqused_k.value());
    CHECK_CONTIGUOUS(seqused_k.value());
    CHECK_SHAPE(seqused_k.value(), batch_size);
  }

  Hstu_fwd_params params;
  set_params_fprop_sm120(
      &params,
      batch_size, max_seqlen_q, max_seqlen_k,
      scaling_seqlen, target_group_size,
      seqlen_q_rounded, seqlen_k_rounded,
      num_heads, num_heads_k, num_heads_rab,
      head_size, static_cast<float>(alpha),
      q, k, v,
      has_rab ? rab.value() : at::Tensor(),
      out,
      num_contexts.has_value() ? num_contexts.value().data_ptr() : nullptr,
      cu_seqlens_q.data_ptr(),
      cu_seqlens_k.data_ptr(),
      seqused_q.has_value() ? seqused_q.value().data_ptr() : nullptr,
      seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
      num_targets.has_value() ? num_targets.value().data_ptr() : nullptr,
      has_rab,
      static_cast<int>(quant_mode),
      func,
      static_cast<int>(window_size_left),
      static_cast<int>(window_size_right),
      descale_q, descale_k, descale_v);

  // Block-wise descale parameters (quant_mode=2 only)
  if (quant_mode == 2) {
    if (cu_seqlens_q_block_descale.has_value()) {
      params.cu_seqlens_q_block_descale =
          static_cast<int*>(cu_seqlens_q_block_descale.value().data_ptr());
    }
    if (cu_seqlens_kv_block_descale.has_value()) {
      params.cu_seqlens_kv_block_descale =
          static_cast<int*>(cu_seqlens_kv_block_descale.value().data_ptr());
    }
    // Head strides: descale_q/k/v are [head, total_blocks], stride(0) = total_blocks
    if (descale_q.has_value() && descale_q.value().dim() >= 2) {
      params.q_block_descale_head_stride = descale_q.value().stride(0);
    }
    if (descale_k.has_value() && descale_k.value().dim() >= 2) {
      params.kv_block_descale_head_stride = descale_k.value().stride(0);
    }
    if (descale_v.has_value() && descale_v.value().dim() >= 2) {
      params.v_block_descale_head_stride = descale_v.value().stride(0);
    }
    if (cu_seqlens_v_block_descale.has_value()) {
      params.cu_seqlens_v_block_descale =
          static_cast<int*>(cu_seqlens_v_block_descale.value().data_ptr());
    }
    if (sf_q_packed.has_value()) {
      params.sf_q_packed_ptr = static_cast<int32_t*>(sf_q_packed.value().data_ptr());
      // For TMA: use sf_q_packed.size(1) as head stride (PyTorch sets stride(0)=1 for H=1,
      // which would make TMA globalDim[0]=1 < boxDim=kBlockM → fail).
      if (sf_q_packed.value().dim() >= 2) {
        params.q_block_descale_head_stride = sf_q_packed.value().size(1);
      }
    }
    if (sf_k_packed.has_value()) {
      params.sf_k_packed_ptr = static_cast<int32_t*>(sf_k_packed.value().data_ptr());
      if (sf_k_packed.value().dim() >= 2) {
        params.kv_block_descale_head_stride = sf_k_packed.value().size(1);
      }
    }
    if (sf_v_packed.has_value()) {
      params.sf_v_packed_ptr = static_cast<int32_t*>(sf_v_packed.value().data_ptr());
      if (sf_v_packed.value().dim() >= 2) {
        params.v_block_descale_head_stride = sf_v_packed.value().size(1);
      }
    }
  }

  params.total_k = total_k;  // Phase 5: needed for TMA descriptor covering full concat K/V
  params.total_q = q.size(0);  // Phase 6 WS TMA: needed for TMA Q descriptor

  if (total_k > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_hstu_fwd_blackwell(params, stream);
  } else {
    out.zero_();
  }

  return {out, has_rab ? rab.value() : at::Tensor()};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "hstu_varlen_fwd_120("
      "Tensor q, Tensor k, Tensor v, "
      "Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
      "Tensor? seqused_q, Tensor? seqused_k, "
      "int max_seqlen_q, int max_seqlen_k, int scaling_seqlen, "
      "Tensor? num_contexts, Tensor? num_targets, int target_group_size, "
      "int window_size_left, int window_size_right, float alpha, "
      "Tensor? rab, Tensor? func, "
      "int quant_mode, "
      "Tensor? descale_q, Tensor? descale_k, Tensor? descale_v, "
      "Tensor? sf_q_packed, Tensor? sf_k_packed, Tensor? sf_v_packed, "
      "Tensor? cu_seqlens_q_block_descale, Tensor? cu_seqlens_kv_block_descale, "
      "Tensor? cu_seqlens_v_block_descale"
      ") -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("hstu_varlen_fwd_120", hstu_varlen_fwd_120);
}

} // namespace fbgemm_gpu::hstu
