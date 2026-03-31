// Phase 6 warp-specialized FP8 kernel body.
// Included inside namespace flash from hstu_fwd_kernel.h.
// Do not include directly; use hstu_fwd_kernel.h.

////////////////////////////////////////////////////////////////////////////////////////////////////
// Phase 6: warp-specialized FP8 compute function.
// Warp 0 (threads 0-31):   dedicated TMA load warp — issues K + V^T TMA each iteration.
// Warps 1-8 (threads 32-287): math warps — QMMA block-scale GEMM1+silu+GEMM2.
// Dual mbarrier: load_mbar (TMA→math) + math_mbar (math→load).
// SMEM layout: [Q/K/Vt/ValidBlockIds/...][SFA(512B)][SFB(512B)][load_mbar(8B)][math_mbar(8B)]
template <typename Kernel_traits, typename Params>
inline __device__ void hstu_compute_attn_1rowblock_sm120_fp8_ws(
    const Params& params,
    const int bidb,
    const int bidh,
    int m_block) {

  static_assert(Kernel_traits::Is_fp8, "Phase 6 WS: FP8 path only");
  static_assert(!Kernel_traits::Has_rab, "Phase 6 WS: Has_rab not yet supported");

  using BS1 = sm120_blockscaled_gemm::SM120BlockScaledBuilder<
      Kernel_traits::kBlockM, Kernel_traits::kBlockN, 4>;
  using BS2 = sm120_blockscaled_gemm::SM120BlockScaledBuilder<
      Kernel_traits::kBlockM, Kernel_traits::kHeadDim, 4>;
  using FP8Elem = typename Kernel_traits::Element;

  extern __shared__ char smem_[];

  const int tidx = threadIdx.x;
  // Warp 0 = load warp (threads 0-31); warps 1-8 = math warps (threads 32-287).
  const bool is_load_warp = (tidx < 32);
  // Math warp thread index in [0,255]; load warp slot 0 is unused for compute.
  const int tidx_math = is_load_warp ? 0 : (tidx - 32);
  constexpr int kNMathThreads = Kernel_traits::kNMathThreads;  // 256

  constexpr bool Is_causal   = Kernel_traits::Is_causal;
  constexpr bool Is_target   = Kernel_traits::Is_target;
  constexpr bool Is_context  = Kernel_traits::Is_context;
  constexpr bool Is_arbitrary= Kernel_traits::Is_arbitrary;
  constexpr int  kNFunc      = Kernel_traits::kNFunc;
  constexpr bool Is_local    = Kernel_traits::Is_local;
  constexpr int  kBlockM     = Kernel_traits::kBlockM;
  constexpr int  kBlockN     = Kernel_traits::kBlockN;
  constexpr int  kHeadDim    = Kernel_traits::kHeadDim;

  const HstuBlockInfo<Kernel_traits, Params> binfo(params, bidb);
  if (m_block * kBlockM >= binfo.actual_seqlen_q_padded) return;

  char* smem_q    = reinterpret_cast<char*>(smem_);
  char* smem_func = reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemSizeQKVRabValidBlockIds;
  int* sn_valid_block_max = reinterpret_cast<int*>(smem_func);
  int* sf_min_ptr = reinterpret_cast<int*>(sn_valid_block_max) + 1;
  int* sf_max_ptr = sf_min_ptr + (kNFunc/2 + 1);

  const int actual_seqlen_q        = binfo.actual_seqlen_q;
  const int actual_seqlen_k        = binfo.actual_seqlen_k;
  const int actual_seqlen_q_padded = binfo.actual_seqlen_q_padded;
  const int actual_seqlen_t        = Is_target  ? binfo.actual_seqlen_t : 0;
  const int actual_seqlen_c        = Is_context ? binfo.actual_seqlen_c : 0;
  const int actual_seqlen_h        = Is_target  ? actual_seqlen_k - actual_seqlen_t : actual_seqlen_k;
  const int actual_seqlen_offset   = actual_seqlen_k - actual_seqlen_q;

  const bool is_jump = Is_target && m_block * kBlockM + actual_seqlen_offset > actual_seqlen_h;
  const bool is_in_context = Is_context && (m_block + 1) * kBlockM <= actual_seqlen_c;
  const bool is_in_mixed_context = Is_context &&
      (m_block + 1) * kBlockM > actual_seqlen_c && m_block * kBlockM < actual_seqlen_c;

  const int n_block_history = cute::ceil_div(actual_seqlen_h, kBlockN);
  const int target_index = (m_block * kBlockM - actual_seqlen_h) / params.target_group_size;

  int n_block_min = !Is_local ? 0
      : std::max(0, (m_block * kBlockM + actual_seqlen_offset - params.window_size_left) / kBlockN);
  int n_block_max = cute::ceil_div(actual_seqlen_k, kBlockN);
  if constexpr (Is_causal || Is_local) {
    int offset = (m_block + 1) * kBlockM + actual_seqlen_offset + params.window_size_right;
    n_block_max = std::min(n_block_max, cute::ceil_div(offset, kBlockN));
  }
  if constexpr (Is_context) {
    n_block_min = (is_in_context || is_in_mixed_context) ? 0 : n_block_min;
    n_block_max = (is_in_context || is_in_mixed_context)
        ? std::max(n_block_history, n_block_max) : n_block_max;
  }

  int n_masking_block_max = cute::ceil_div(
      std::min(actual_seqlen_k, (m_block + 1) * kBlockM + actual_seqlen_offset), kBlockN);
  int n_masking_block_min = (m_block * kBlockM + actual_seqlen_offset) / kBlockN;
  if constexpr (Is_target) {
    n_masking_block_min = is_jump
        ? (actual_seqlen_h + actual_seqlen_offset + target_index * params.target_group_size) / kBlockN
        : n_masking_block_min;
  }
  if constexpr (Is_context) {
    n_masking_block_min = is_in_mixed_context ? n_block_min : n_masking_block_min;
    n_masking_block_max = is_in_mixed_context ? n_block_max : n_masking_block_max;
  }
  const int n_masking_steps = (!Is_causal || is_in_context)
      ? 0 : n_masking_block_max - n_masking_block_min;

  // GMEM tensors
  Tensor mQ = make_tensor(
      make_gmem_ptr(reinterpret_cast<FP8Elem*>(params.q_ptr) + binfo.q_offset(params.q_row_stride)),
      make_shape(actual_seqlen_q, params.h, params.d),
      make_stride(params.q_row_stride, params.q_head_stride, _1{}));
  Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

  // Arbitrary func GMEM tensors
  Tensor mMaxFunc = make_tensor(
      make_gmem_ptr(reinterpret_cast<int*>(params.func_ptr) + binfo.sum_s_q),
      make_shape(Int<1>{}, Int<kNFunc/2 + 1>{}, actual_seqlen_q),
      make_stride(params.func_head_stride, 2 * params.func_ids_stride, _1{}));
  Tensor mMinFunc = make_tensor(
      make_gmem_ptr(reinterpret_cast<int*>(params.func_ptr) + binfo.sum_s_q + params.func_ids_stride),
      make_shape(Int<1>{}, Int<kNFunc/2>{}, actual_seqlen_q),
      make_stride(params.func_head_stride, 2 * params.func_ids_stride, _1{}));
  Tensor gMaxFunc = local_tile(mMaxFunc(Int<0>{}, _, _),
      make_shape(Int<kNFunc/2 + 1>{}, Int<kBlockM>{}), make_coord(Int<0>{}, m_block));
  Tensor gMinFunc = local_tile(mMinFunc(Int<0>{}, _, _),
      make_shape(Int<kNFunc/2>{}, Int<kBlockM>{}), make_coord(Int<0>{}, m_block));

  // SMEM tensors
  Tensor sValidBlockIds = make_tensor(
      make_smem_ptr(reinterpret_cast<int*>(smem_ + Kernel_traits::kSmemSizeQKVRab)),
      typename Kernel_traits::SmemLayoutValidBlockIds{});
  Tensor sFunc_min = make_tensor(make_smem_ptr(sf_min_ptr), typename Kernel_traits::SmemLayoutMinFunc{});
  Tensor sFunc_max = make_tensor(make_smem_ptr(sf_max_ptr), typename Kernel_traits::SmemLayoutMaxFunc{});

  // Is_arbitrary setup: warp 1 (first math warp) handles this before the WS split.
  // Warp 0 is the dedicated load warp and must not execute non-TMA preamble work.
  if constexpr (Is_arbitrary) {
    const int lane_id = cutlass::canonical_lane_idx();
    const int warp_id = cutlass::canonical_warp_idx_sync();
    if (warp_id == 1) {
      *sn_valid_block_max = 0;
      sFunc_min[0] = 0;
      __syncwarp();
      int f_min = INT_MAX, f_max = INT_MIN;
      const int base_row = m_block * kBlockM;
      for (int i = 0; i < size<0>(gMinFunc); i++) {
        for (int j = lane_id; j < size<1>(gMinFunc); j += 32) {
          const int row = base_row + j;
          if (row < actual_seqlen_q) if (f_min > gMinFunc(i, j)) f_min = gMinFunc(i, j);
        }
        warpReduce(f_min, MinOp<int>());
        if (lane_id == 0) sFunc_min[i+1] = f_min;
        f_min = INT_MAX;
      }
      for (int i = 0; i < size<0>(gMaxFunc); i++) {
        for (int j = lane_id; j < size<1>(gMaxFunc); j += 32) {
          const int row = base_row + j;
          if (row < actual_seqlen_q) if (f_max < gMaxFunc(i, j)) f_max = gMaxFunc(i, j);
        }
        warpReduce(f_max, MaxOp<int>());
        if (lane_id == 0) sFunc_max[i] = f_max;
        f_max = INT_MIN;
      }
      if (lane_id == 0) {
        for (int n_block = n_block_min; n_block < n_block_max; n_block++) {
          int b_max = (n_block + 1) * kBlockN, b_min = n_block * kBlockN;
          for (int i = 0; i < (kNFunc + 1)/2; i++) {
            int fmin = sFunc_min[i], fmax = sFunc_max[i];
            if (fmax <= fmin) continue;
            bool c1 = fmin <= b_min && fmax > b_min;
            bool c2 = fmin >= b_min && b_max > fmin;
            bool c3 = fmin >= b_min && fmax < b_max;
            if (c1 || c2 || c3) { sValidBlockIds[*sn_valid_block_max] = n_block; (*sn_valid_block_max)++; break; }
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
    using OutElement = typename Kernel_traits::OutputType;
    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<OutElement*>(params.o_ptr) + binfo.q_offset(params.o_row_stride)),
        make_shape(actual_seqlen_q, params.h, params.d),
        make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));
    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx_math);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO = make_tensor<OutElement>(shape(tOgO));
    clear(tOrO);
    Tensor cO_zr = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO_zr);
    if (!is_load_warp) {
      flash::copy<false, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO,
          actual_seqlen_q_padded - m_block * kBlockM);
    }
    return;
  }

  // SW128 SMEM layouts (same as Phase 5 FP8)
  using SmemLayoutQ_SW128  = decltype(tile_to_shape(typename BS1::SmemLayoutAtomA{},
      Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutK_SW128  = decltype(tile_to_shape(typename BS1::SmemLayoutAtomB{},
      Shape<Int<kBlockN>, Int<kHeadDim>>{}));
  using SmemLayoutVt_SW128 = decltype(tile_to_shape(typename BS2::SmemLayoutAtomB{},
      Shape<Int<kHeadDim>, Int<kBlockN>>{}));
  using SmemLayoutVt_TMA_t = typename Kernel_traits::SmemLayoutVt_TMA;

  Tensor sQ_sw128 = make_tensor(make_smem_ptr(reinterpret_cast<FP8Elem*>(smem_q)), SmemLayoutQ_SW128{});
  Tensor sK_sw128 = make_tensor(make_smem_ptr(reinterpret_cast<FP8Elem*>(smem_q)), SmemLayoutK_SW128{});
  Tensor sVt_tma  = make_tensor(
      make_smem_ptr(reinterpret_cast<FP8Elem*>(smem_q) + size(SmemLayoutK_SW128{})),
      SmemLayoutVt_TMA_t{});

  // WS SMEM layout: [DATA...][SFA(512B)][SFB(512B)][load_mbar(8B)][math_mbar(8B)]
  // SF is placed at kSmemSize - kSmemMbarSize(16) - kSmemSFSize(1024) — no overlap with mbars.
  static constexpr int kSmemSFOffset_WS =
      Kernel_traits::kSmemSize - Kernel_traits::kSmemMbarSize - Kernel_traits::kSmemSFSize;
  int32_t* smem_sfa_ptr = reinterpret_cast<int32_t*>(smem_ + kSmemSFOffset_WS);
  int32_t* smem_sfb_ptr = smem_sfa_ptr + kBlockM;  // SFB at +512B after SFA

  // Two WS barriers at the last 16 bytes of SMEM.
  static constexpr int kSmemLoadMbarOffset =
      Kernel_traits::kSmemSize - Kernel_traits::kSmemMbarSize;
  static constexpr int kSmemMathMbarOffset =
      Kernel_traits::kSmemSize - Kernel_traits::kSmemMbarSize / 2;
  uint64_t* load_mbar_ptr = reinterpret_cast<uint64_t*>(smem_ + kSmemLoadMbarOffset);
  uint64_t* math_mbar_ptr = reinterpret_cast<uint64_t*>(smem_ + kSmemMathMbarOffset);

  // Thread 0 inits both barriers and pre-satisfies math_mbar phase 0 (8 fake arrives)
  // so load warp can issue the first TMA without waiting.
  if (tidx == 0) {
    uint32_t laddr = static_cast<uint32_t>(__cvta_generic_to_shared(load_mbar_ptr));
    uint32_t maddr = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(laddr), "r"(1));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(maddr), "r"(8));
    for (int i = 0; i < 8; i++)
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" : : "r"(maddr));
  }
  asm volatile("fence.proxy.async.shared::cta;\n" : : : "memory");
  __syncthreads();

  // SF SMEM tensors (shared between load and math warps, but only math warps write)
  using SmemLayoutSFA = typename BS1::SmemLayoutSFA;
  using SmemLayoutSFB = typename BS1::SmemLayoutSFB;
  Tensor sSFA_ = make_tensor(make_smem_ptr(smem_sfa_ptr), SmemLayoutSFA{});
  Tensor sSFB_ = make_tensor(make_smem_ptr(smem_sfb_ptr), SmemLayoutSFB{});
  auto sSFA = as_position_independent_swizzle_tensor(sSFA_);
  auto sSFB = as_position_independent_swizzle_tensor(sSFB_);

  // GMEM copy atom for SW128 FP8 (based on kNMathThreads=256, math warps only)
  using GmemLayoutAtom_SW128 = Layout<Shape<Int<kNMathThreads / 8>, _8>, Stride<_8, _1>>;
  auto gmem_tiled_copy_sw128 = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, FP8Elem>{},
      GmemLayoutAtom_SW128{},
      Layout<Shape<_1, _16>>{});
  auto gmem_thr_copy_math = gmem_tiled_copy_sw128.get_thread_slice(tidx_math);
  Tensor tQgQ = gmem_thr_copy_math.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_math.partition_D(sQ_sw128);

  // MMA tiled objects (math warps use tidx_math)
  typename BS1::TiledMma tiled_mma_g1;
  auto thr_mma_g1 = tiled_mma_g1.get_thread_slice(tidx_math);
  typename BS2::TiledMma tiled_mma_g2;
  auto thr_mma_g2 = tiled_mma_g2.get_thread_slice(tidx_math);

  Tensor acc_o = partition_fragment_C(tiled_mma_g2, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  clear(acc_o);

  // s2r copy atoms (using tidx_math)
  auto s2r_copy_A   = make_tiled_copy_A(typename BS1::SmemCopyAtomA{}, tiled_mma_g1);
  auto s2r_thr_A    = s2r_copy_A.get_thread_slice(tidx_math);
  auto s2r_copy_B   = make_tiled_copy_B(typename BS1::SmemCopyAtomB{}, tiled_mma_g1);
  auto s2r_thr_B    = s2r_copy_B.get_thread_slice(tidx_math);
  auto s2r_copy_A2  = make_tiled_copy_A(typename BS2::SmemCopyAtomA{}, tiled_mma_g2);
  auto s2r_thr_A2   = s2r_copy_A2.get_thread_slice(tidx_math);
  auto s2r_copy_B2  = make_tiled_copy_B(typename BS2::SmemCopyAtomB{}, tiled_mma_g2);
  auto s2r_thr_B2   = s2r_copy_B2.get_thread_slice(tidx_math);
  auto s2r_copy_SFA = make_tiled_copy_impl(typename BS1::SmemCopyAtomSF{},
      BS1::get_layoutSFA_TV(tiled_mma_g1), make_shape(size<0>(tile_shape(tiled_mma_g1)), _1{}));
  auto s2r_thr_SFA  = s2r_copy_SFA.get_thread_slice(tidx_math);
  auto s2r_copy_SFB = make_tiled_copy_impl(typename BS1::SmemCopyAtomSF{},
      BS1::get_layoutSFB_TV(tiled_mma_g1), make_shape(size<1>(tile_shape(tiled_mma_g1)), _1{}));
  auto s2r_thr_SFB  = s2r_copy_SFB.get_thread_slice(tidx_math);
  auto s2r_copy_SFP = make_tiled_copy_impl(typename BS2::SmemCopyAtomSF{},
      BS2::get_layoutSFA_TV(tiled_mma_g2), make_shape(size<0>(tile_shape(tiled_mma_g2)), _1{}));
  auto s2r_thr_SFP  = s2r_copy_SFP.get_thread_slice(tidx_math);
  auto s2r_copy_SFV = make_tiled_copy_impl(typename BS2::SmemCopyAtomSF{},
      BS2::get_layoutSFB_TV(tiled_mma_g2), make_shape(size<1>(tile_shape(tiled_mma_g2)), _1{}));
  auto s2r_thr_SFV  = s2r_copy_SFV.get_thread_slice(tidx_math);

  // Load Q (math warps only; load warp idles through cp_async_fence/wait/__syncthreads)
  if (!is_load_warp) {
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ_sw128), size<1>(sQ_sw128)));
    Tensor tQcQ = gmem_thr_copy_math.partition_S(cQ);
    flash::copy<false>(gmem_tiled_copy_sw128, tQgQ, tQsQ, tQcQ,
        actual_seqlen_q - m_block * kBlockM);
  }
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  // s2r Q (math warps only)
  auto sQ_pi = as_position_independent_swizzle_tensor(sQ_sw128);
  Tensor tCrQ = thr_mma_g1.partition_fragment_A(sQ_pi);
  if (!is_load_warp) {
    auto tXsQ = s2r_thr_A.partition_S(sQ_pi);
    auto tXrQ = s2r_thr_A.retile_D(tCrQ);
    cute::copy(s2r_copy_A, tXsQ, tXrQ);
  }

  // Load SFA (Q scale) into SMEM (math warps only), then s2r SFA
  if (!is_load_warp) {
    if (params.sf_q_packed_ptr != nullptr && params.cu_seqlens_q_block_descale != nullptr) {
      const int64_t q_tile_base = static_cast<int64_t>(bidh) * params.q_block_descale_head_stride
          + params.cu_seqlens_q_block_descale[bidb] + m_block * kBlockM;
      for (int i = tidx_math; i < kBlockM; i += kNMathThreads)
        smem_sfa_ptr[i] = params.sf_q_packed_ptr[q_tile_base + i];
    } else {
      for (int i = tidx_math; i < kBlockM; i += kNMathThreads)
        smem_sfa_ptr[i] = 0x7f7f7f7f;
    }
  }
  // bar.sync 1, 256: sync math warps only (SFA SMEM visible before s2r SFA).
  // MUST be inside if(!is_load_warp): bar.sync count must exactly equal the number of
  // threads that execute the instruction (288 total, 256 math warps). Calling with count=256
  // from all 288 threads is undefined behavior → illegal memory access.
  if (!is_load_warp) {
    asm volatile("bar.sync 1, 256;\n" : : : "memory");
  }

  Tensor tCrSFA = BS1::partition_fragment_SFA(sSFA(_,_,_0{}), thr_mma_g1);
  if (!is_load_warp) {
    auto tXsSFA = s2r_thr_SFA.partition_S(sSFA);
    auto tXrSFA = s2r_thr_SFA.retile_D(tCrSFA);
    cute::copy(s2r_copy_SFA, tXsSFA(_,_,_,_0{}), tXrSFA);
  }
  auto tCrSFA_frg = BS1::transform_fragment_for_qmma(tCrSFA);

  // TMA descriptors for K and V^T (same as Phase 5)
  const int bidh_kv = bidh / params.h_h_k_ratio;
  auto mK_tma   = params.tma_k.get_tma_tensor(make_shape(params.total_k, params.d, params.h_k));
  auto gK_head  = mK_tma(_, _, bidh_kv);
  auto gK_tiles = local_tile(gK_head, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _));
  auto tma_slice_K  = params.tma_k.get_slice(0);
  auto tKsK_d       = tma_slice_K.partition_D(sK_sw128);
  auto tKgK_tma     = tma_slice_K.partition_S(gK_tiles(_, _, _, Int<0>{}));

  // Vt: same (total_k, d, h_k) ordering as K — matches the TMA descriptor created on host.
  auto mVt_tma   = params.tma_vt.get_tma_tensor(make_shape(params.total_k, params.d, params.h_k));
  auto gVt_head  = mVt_tma(_, _, bidh_kv);
  auto gVt_tiles = local_tile(gVt_head, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _));
  auto tma_slice_Vt = params.tma_vt.get_slice(0);
  auto tVtsVt_d     = tma_slice_Vt.partition_D(sVt_tma);
  auto tVtgVt_tma   = tma_slice_Vt.partition_S(gVt_tiles(_, _, _, Int<0>{}));

  constexpr int kKBytes  = kBlockN * kHeadDim * sizeof(FP8Elem);
  constexpr int kVtBytes = kHeadDim * kBlockN * sizeof(FP8Elem);

  // Mask lambda (uses thr_mma_g1 partitioned with tidx_math → math warps only)
  auto col_limit_right = [&](int row) {
    return std::min(actual_seqlen_k, row + 1 + params.window_size_right);
  };
  auto col_limit_left = [&](int row) {
    return std::max(0, row - params.window_size_left);
  };
  auto apply_mask_bs = [&](auto& tSrS, int nb) {
    static constexpr int Row = 0, Col = 1;
    Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thr_mma_g1.partition_C(cS);
    const int base_row = m_block * kBlockM + actual_seqlen_offset;
    const int base_col = nb * kBlockN;
    Tensor tSrS_v = make_tensor(tSrS.data(),
        group<1,3>(group<0,2>(select<1,2,0,3>(flatten(tSrS.layout())))));
    Tensor tScS_v = make_tensor(tScS.data(),
        group<1,3>(group<0,2>(select<1,2,0,3>(flatten(tScS.layout())))));
    Tensor col_min = make_tensor<int>(make_shape(size<0>(gMinFunc)));
    Tensor col_max = make_tensor<int>(make_shape(size<0>(gMaxFunc)));
#pragma unroll
    for (int r = 0; r < size<0>(tSrS_v); r++) {
      const int block_row = int(get<Row>(tScS_v(r,0)));
      const int row = block_row + base_row;
      [[maybe_unused]] const int tgt_idx =
          Is_target ? (row - actual_seqlen_h) / params.target_group_size : 0;
      [[maybe_unused]] const int tgt_col_lft =
          Is_target ? actual_seqlen_h + tgt_idx * params.target_group_size : 0;
      if constexpr (Is_arbitrary) {
        col_max(0) = gMaxFunc(0, block_row);
#pragma unroll
        for (int j = 0; j < size<0>(gMinFunc); ++j) {
          col_min(j) = gMinFunc(j, block_row);
          col_max(j+1) = gMaxFunc(j+1, block_row);
        }
      }
#pragma unroll
      for (int c = 0; c < size<1>(tSrS_v); c++) {
        const int block_col = int(get<Col>(tScS_v(r,c)));
        const int col = block_col + base_col;
        if constexpr (!Is_causal && !Is_local && !Is_arbitrary) {
          if (col >= actual_seqlen_k) { tSrS_v(r,c) = -INFINITY; continue; }
        } else {
          if constexpr (Is_context) {
            if (row < actual_seqlen_c && col < actual_seqlen_h) continue;
          }
          if (col >= col_limit_right(row)) { tSrS_v(r,c) = -INFINITY; continue; }
          if constexpr (Is_local) {
            if (col < col_limit_left(row)) { tSrS_v(r,c) = -INFINITY; continue; }
          }
          if constexpr (Is_target) {
            if (row >= actual_seqlen_h && col >= actual_seqlen_h && col < tgt_col_lft)
              tSrS_v(r,c) = -INFINITY;
          }
          if constexpr (Is_arbitrary) {
            bool non_mask = (0 <= col) && (col < col_max(0));
            if (non_mask) continue;
#pragma unroll
            for (int j = 0; j < size<0>(gMinFunc); ++j) {
              non_mask = (col_min(j) <= col) && (col < col_max(j+1));
              if (non_mask) break;
            }
            if (!non_mask) tSrS_v(r,c) = -INFINITY;
          }
        }
      }
    }
  };

  // Sync all 288 threads before entering WS loop.
  // Ensures math warps finish s2r SFA before load warp starts issuing TMA.
  __syncthreads();

  // ===== WS MAIN LOOP =====
  // load_phase: expected parity for load_mbar waits (math warps use this).
  //   test_wait.parity returns TRUE when current_phase != parity.
  //   Fresh mbarrier phase=0 → test_wait(addr,0) blocks (0==0 → FALSE).
  //   After first TMA completion, phase=1 → test_wait(addr,0) passes (1≠0).
  // math_phase: parity for math_mbar waits (load warp uses this).
  //   After pre-satisfaction, phase=1 → test_wait(addr,0) passes (1≠0) immediately.
  int load_phase = 0;
  int math_phase = 0;

  if (is_load_warp) {
    // ===== LOAD WARP: TMA producer loop =====
    for (int n_valid = n_block_max - 1, masking_step = 0; n_valid >= n_block_min;
         ++masking_step, --n_valid) {
      const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;
      const int nb_abs = binfo.sum_s_k / kBlockN + nb;

      // Wait for math warps to consume previous tile (phase 0 pre-satisfied on first iter)
      {
        uint32_t maddr = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr));
        uint32_t done = 0;
        do {
          asm volatile("{.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;}\n"
                       : "=r"(done) : "r"(maddr), "r"((uint32_t)math_phase));
        } while (!done);
      }
      math_phase ^= 1;

      // Issue TMA K[nb_abs] + V^T[nb_abs]: copy first, then arrive.expect_tx.
      // This matches the Phase 5 proven ordering (copy issues the TMA descriptor to HW,
      // arrive.expect_tx sets the expected byte count so math warps can wait correctly).
      if (tidx == 0) {
        uint32_t laddr = static_cast<uint32_t>(__cvta_generic_to_shared(load_mbar_ptr));
        cute::copy(params.tma_k.with(*load_mbar_ptr),  tKgK_tma(_, _, _, nb_abs), tKsK_d);
        cute::copy(params.tma_vt.with(*load_mbar_ptr), tVtgVt_tma(_, _, _, nb_abs), tVtsVt_d);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                     : : "r"(laddr), "r"(kKBytes + kVtBytes));
      }

      if (is_jump && masking_step == n_masking_steps - 1)
        n_valid = std::min(n_valid, n_block_history);
    }
    // Load warp done; falls through to __syncthreads() below.

  } else {
    // ===== MATH WARPS: QMMA consumer loop =====
    for (int n_valid = n_block_max - 1, masking_step = 0; n_valid >= n_block_min;
         ++masking_step, --n_valid) {
      const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;
      const bool is_masking = masking_step < n_masking_steps ||
          (nb + 1) * kBlockN > actual_seqlen_h;

      // Load SFB (K scale) into SMEM — can happen before waiting on load_mbar
      if (params.sf_k_packed_ptr != nullptr && params.cu_seqlens_kv_block_descale != nullptr) {
        const int64_t k_tile_base = static_cast<int64_t>(bidh) * params.kv_block_descale_head_stride
            + params.cu_seqlens_kv_block_descale[bidb] + nb * kBlockN;
        for (int i = tidx_math; i < kBlockN; i += kNMathThreads)
          smem_sfb_ptr[i] = params.sf_k_packed_ptr[k_tile_base + i];
      } else {
        for (int i = tidx_math; i < kBlockN; i += kNMathThreads)
          smem_sfb_ptr[i] = 0x7f7f7f7f;
      }

      // Wait for TMA K[nb]+V^T[nb] to land in SMEM
      {
        uint32_t laddr = static_cast<uint32_t>(__cvta_generic_to_shared(load_mbar_ptr));
        uint32_t done = 0;
        do {
          asm volatile("{.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;}\n"
                       : "=r"(done) : "r"(laddr), "r"((uint32_t)load_phase));
        } while (!done);
      }
      load_phase ^= 1;
      // Sync math warps: SFB + K/Vt SMEM visible before s2r
      asm volatile("bar.sync 1, 256;\n" : : : "memory");

      // s2r K
      auto sK_pi = as_position_independent_swizzle_tensor(sK_sw128);
      Tensor tCrK = thr_mma_g1.partition_fragment_B(sK_pi);
      {
        auto tXsK = s2r_thr_B.partition_S(sK_pi);
        auto tXrK = s2r_thr_B.retile_D(tCrK);
        cute::copy(s2r_copy_B, tXsK, tXrK);
      }
      // s2r SFB
      Tensor tCrSFB = BS1::partition_fragment_SFB(sSFB(_,_,_0{}), thr_mma_g1);
      {
        auto tXsSFB = s2r_thr_SFB.partition_S(sSFB);
        auto tXrSFB = s2r_thr_SFB.retile_D(tCrSFB);
        cute::copy(s2r_copy_SFB, tXsSFB(_,_,_,_0{}), tXrSFB);
      }
      auto tCrSFB_frg = BS1::transform_fragment_for_qmma(tCrSFB);

      // GEMM1: acc_s += Q × K^T (block-scaled)
      Tensor acc_s = partition_fragment_C(tiled_mma_g1, Shape<Int<kBlockM>, Int<kBlockN>>{});
      clear(acc_s);
      cute::gemm(tiled_mma_g1,
          make_zip_tensor(tCrQ, tCrSFA_frg(_,_,_,_0{})),
          make_zip_tensor(tCrK, tCrSFB_frg(_,_,_,_0{})),
          acc_s);

      if (params.debug_gemm1_only) {
        for (int i = 0; i < size(acc_s); ++i) acc_o(i) += acc_s(i);
        if ((tidx & 31) == 0) {
          uint32_t maddr = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr));
          asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" : : "r"(maddr));
        }
        if (is_jump && masking_step == n_masking_steps - 1)
          n_valid = std::min(n_valid, n_block_history);
        continue;
      }

      if (Is_arbitrary || Is_local || is_masking) apply_mask_bs(acc_s, nb);
      for (int i = 0; i < size(acc_s); ++i) acc_s(i) *= params.alpha;
      fast_silu(acc_s);

      // Convert acc_s → FP8 rP
      Tensor rP = make_tensor_like<FP8Elem>(acc_s);
      flash::convert_type_safe(acc_s, rP);

      // Write SFP (unit) and SFV (V scale) into SF SMEM (before P SMEM roundtrip)
      for (int i = tidx_math; i < kBlockM; i += kNMathThreads)
        smem_sfa_ptr[i] = 0x7f7f7f7f;
      if (params.sf_v_packed_ptr != nullptr && params.cu_seqlens_v_block_descale != nullptr) {
        const int64_t v_sf_idx = static_cast<int64_t>(bidh) * params.v_block_descale_head_stride
            + params.cu_seqlens_v_block_descale[bidb] + nb;
        const int32_t vsf_val = params.sf_v_packed_ptr[v_sf_idx];
        for (int i = tidx_math; i < kBlockN; i += kNMathThreads)
          smem_sfb_ptr[i] = vsf_val;
      } else {
        for (int i = tidx_math; i < kBlockN; i += kNMathThreads)
          smem_sfb_ptr[i] = 0x7f7f7f7f;
      }

      // P SMEM roundtrip: write rP to sPbuf (= sQ_sw128 = sK_sw128 shared buffer)
      // bar.sync before write: ensure all warps done s2r K (so overwriting sPbuf is safe)
      asm volatile("bar.sync 1, 256;\n" : : : "memory");
      Tensor sPbuf = make_tensor(sQ_sw128.data(), SmemLayoutQ_SW128{});
      auto sPbuf_pi = as_position_independent_swizzle_tensor(sPbuf);
      {
        Tensor cP_id = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tPcP_raw = thr_mma_g1.partition_C(cP_id);
        Tensor tPcP_v = make_tensor(tPcP_raw.data(),
            group<1,3>(group<0,2>(select<1,2,0,3>(flatten(tPcP_raw.layout())))));
        Tensor rP_v = make_tensor(rP.data(),
            group<1,3>(group<0,2>(select<1,2,0,3>(flatten(rP.layout())))));
        CUTE_UNROLL
        for (int r = 0; r < size<0>(rP_v); ++r) {
          CUTE_UNROLL
          for (int c = 0; c < size<1>(rP_v); ++c) {
            sPbuf_pi(int(get<0>(tPcP_v(r,c))), int(get<1>(tPcP_v(r,c)))) = rP_v(r,c);
          }
        }
      }
      // bar.sync after write: P SMEM visible to all warps before s2r P
      asm volatile("bar.sync 1, 256;\n" : : : "memory");

      // s2r P (GEMM2 A-operand)
      Tensor tCrP = thr_mma_g2.partition_fragment_A(sPbuf_pi);
      {
        auto tXsP = s2r_thr_A2.partition_S(sPbuf_pi);
        auto tXrP = s2r_thr_A2.retile_D(tCrP);
        cute::copy(s2r_copy_A2, tXsP, tXrP);
      }

      // s2r V^T (GEMM2 B-operand) from sVt_tma
      Tensor sVt_read = make_tensor(
          make_smem_ptr(reinterpret_cast<FP8Elem*>(smem_q) + size(SmemLayoutK_SW128{})),
          SmemLayoutVt_SW128{});
      auto sVt_pi = as_position_independent_swizzle_tensor(sVt_read);
      Tensor tCrV = thr_mma_g2.partition_fragment_B(sVt_pi);
      {
        auto tXsVt = s2r_thr_B2.partition_S(sVt_pi);
        auto tXrV  = s2r_thr_B2.retile_D(tCrV);
        cute::copy(s2r_copy_B2, tXsVt, tXrV);
      }

      // Notify load warp: K+Vt SMEM no longer needed (s2r K and s2r Vt complete)
      if ((tidx & 31) == 0) {
        uint32_t maddr = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr));
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" : : "r"(maddr));
      }

      // s2r SFP and SFV (from SF SMEM, unaffected by load warp TMA for next tile)
      Tensor tCrSFP = BS2::partition_fragment_SFA(sSFA(_,_,_0{}), thr_mma_g2);
      {
        auto tXsSFP = s2r_thr_SFP.partition_S(sSFA);
        auto tXrSFP = s2r_thr_SFP.retile_D(tCrSFP);
        cute::copy(s2r_copy_SFP, tXsSFP(_,_,_,_0{}), tXrSFP);
      }
      auto tCrSFP_frg = BS2::transform_fragment_for_qmma(tCrSFP);
      Tensor tCrSFV = BS2::partition_fragment_SFB(sSFB(_,_,_0{}), thr_mma_g2);
      {
        auto tXsSFV = s2r_thr_SFV.partition_S(sSFB);
        auto tXrSFV = s2r_thr_SFV.retile_D(tCrSFV);
        cute::copy(s2r_copy_SFV, tXsSFV(_,_,_,_0{}), tXrSFV);
      }
      auto tCrSFV_frg = BS2::transform_fragment_for_qmma(tCrSFV);

      // GEMM2: acc_o += P × V^T (block-scaled)
      cute::gemm(tiled_mma_g2,
          make_zip_tensor(tCrP, tCrSFP_frg(_,_,_,_0{})),
          make_zip_tensor(tCrV, tCrSFV_frg(_,_,_,_0{})),
          acc_o);

      if (is_jump && masking_step == n_masking_steps - 1)
        n_valid = std::min(n_valid, n_block_history);
    }
  }

  // ===== EPILOGUE =====
  // Math warps write acc_o to flat SMEM; load warp idles.
  if (!is_load_warp) {
    for (int i = 0; i < size(acc_o); ++i) acc_o(i) /= params.scaling_seqlen;
    using OutElement = typename Kernel_traits::OutputType;
    Tensor rO = make_tensor_like<OutElement>(acc_o);
    flash::convert_type_safe(acc_o, rO);

    Tensor sO_flat = make_tensor(
        make_smem_ptr(reinterpret_cast<OutElement*>(smem_q)),
        Layout<Shape<Int<kBlockM>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>{});
    {
      Tensor cO_id = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
      Tensor tOcO_raw = thr_mma_g2.partition_C(cO_id);
      Tensor tOcO_v = make_tensor(tOcO_raw.data(),
          group<1,3>(group<0,2>(select<1,2,0,3>(flatten(tOcO_raw.layout())))));
      Tensor rO_v = make_tensor(rO.data(),
          group<1,3>(group<0,2>(select<1,2,0,3>(flatten(rO.layout())))));
      CUTE_UNROLL
      for (int r = 0; r < size<0>(rO_v); ++r) {
        CUTE_UNROLL
        for (int c = 0; c < size<1>(rO_v); ++c) {
          sO_flat(int(get<0>(tOcO_v(r,c))), int(get<1>(tOcO_v(r,c)))) = rO_v(r,c);
        }
      }
    }
  }
  // All 288 threads sync: load warp arrives here, math warps' SMEM write visible
  __syncthreads();

  if (!is_load_warp) {
    using OutElement = typename Kernel_traits::OutputType;
    Tensor sO_flat = make_tensor(
        make_smem_ptr(reinterpret_cast<OutElement*>(smem_q)),
        Layout<Shape<Int<kBlockM>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>{});
    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<OutElement*>(params.o_ptr) + binfo.q_offset(params.o_row_stride)),
        make_shape(actual_seqlen_q, params.h, params.d),
        make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO_bs = local_tile(mO(_, bidh, _),
        Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));
    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx_math);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO_flat);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO_bs);
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

// Phase 6 WS kernel entry: launched with kNThreadsTotal=288.
template <typename Kernel_traits, typename TMA_K_t, typename TMA_Vt_t>
__global__ void __launch_bounds__(Kernel_traits::kNThreads)
hstu_fwd_kernel_sm120_fp8_ws(
    Hstu_fwd_params_fp8_tma<TMA_K_t, TMA_Vt_t> params) {
  int m_block = gridDim.x - blockIdx.x - 1;
  int bidh    = blockIdx.y;
  int bidb    = blockIdx.z;
  hstu_compute_attn_1rowblock_sm120_fp8_ws<Kernel_traits>(params, bidb, bidh, m_block);
}

