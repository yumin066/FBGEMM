// Phase 6 warp-specialized FP8 kernel body — Q, K, and V^T via TMA.
// Included inside namespace flash from hstu_fwd_kernel.h.
// Do not include directly; use hstu_fwd_kernel.h.

////////////////////////////////////////////////////////////////////////////////////////////////////

// Spin-wait helper: blocks until mbarrier phase parity != expected parity.
__device__ inline void wait_mbar_parity(uint64_t* mbar, uint32_t parity) {
  uint32_t maddr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  uint32_t done = 0;
  do {
    asm volatile(
        "{.reg .pred p;\n"
        "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;}\n"
        : "=r"(done) : "r"(maddr), "r"(parity));
  } while (!done);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Phase 6: warp-specialized FP8 compute function — Q, K, V^T all via TMA.
// Warp 0 (threads 0-31):   load warp — Q TMA preamble (thread 0), then K+Vt TMA per iter (thread 0).
// Warps 1-8 (threads 32-287): math warps — QMMA block-scale GEMM1+silu+GEMM2.
// Three mbarriers: tma_k_mbar (TMA Q/K/Vt completion), load_mbar (K+Vt→math), math_mbar (math→load).
// SMEM: [Q/K/Vt/ValidBlockIds/...][SFA(512B)][SFB(512B)][tma_k_mbar(8B)][load_mbar(8B)][math_mbar(8B)]
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
  // WS double-buffer: func/validblockids start after 4 KV stage buffers.
  char* smem_func = reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsFuncOffset;
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
      make_smem_ptr(reinterpret_cast<int*>(smem_ + Kernel_traits::kSmemWsValidBlockIdsOffset)),
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

  // SW128 SMEM layouts (Phase 6 TMA: Q, K, V^T all via TMA)
  using SmemLayoutQ_SW128  = decltype(tile_to_shape(typename BS1::SmemLayoutAtomA{},
      Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutK_SW128  = decltype(tile_to_shape(typename BS1::SmemLayoutAtomB{},
      Shape<Int<kBlockN>, Int<kHeadDim>>{}));
  // V^T layout: [kHeadDim, kBlockN] in SMEM, loaded directly by TMA (no kernel-side transpose).
  // Must match SmemLayoutVt_TMA (MN_SW128): TMA requires SMEM inner=kHeadDim → GMEM d (stride=1).
  using SmemLayoutVt_SW128 = typename Kernel_traits::SmemLayoutVt_TMA;

  // Double-buffer SMEM layout:
  //   Stage 0: sK[0] at smem_q + 0,              sVt[0] at smem_q + kSmemKVElems
  //   Stage 1: sK[1] at smem_q + 2*kSmemKVElems, sVt[1] at smem_q + 3*kSmemKVElems
  constexpr int kSmemKVElems = kBlockN * kHeadDim;  // FP8 elements per K or Vt tile (= 16384)
  FP8Elem* const sK_base[2] = {
      reinterpret_cast<FP8Elem*>(smem_q),
      reinterpret_cast<FP8Elem*>(smem_q) + 2 * kSmemKVElems
  };
  FP8Elem* const sVt_base[2] = {
      reinterpret_cast<FP8Elem*>(smem_q) + kSmemKVElems,
      reinterpret_cast<FP8Elem*>(smem_q) + 3 * kSmemKVElems
  };
  // sQ shares stage-0's K buffer (Q preamble happens before the K loop overwrites it)
  Tensor sQ_sw128 = make_tensor(make_smem_ptr(sK_base[0]), SmemLayoutQ_SW128{});

  // WS SMEM layout:
  //   [DATA...][SFA(512B)][SFB(512B)][tma_mbar×2][math_mbar×2]
  // SF region starts at kSmemWsDataSizePadded.
  static constexpr int kSmemSFOffset_WS = Kernel_traits::kSmemWsDataSizePadded;
  int32_t* smem_sfa_ptr = reinterpret_cast<int32_t*>(smem_ + kSmemSFOffset_WS);
  // SFB is double-buffered (TMA loaded by load warp, stage-indexed).
  int32_t* const smem_sfb_ptr[2] = {
      smem_sfa_ptr + kBlockM,
      smem_sfa_ptr + kBlockM + kBlockN
  };
  // SFV is double-buffered (TMA loaded by load warp, stage-indexed), placed after SFB[0]+SFB[1].
  int32_t* const smem_sfv_ptr[2] = {
      smem_sfa_ptr + kBlockM + 2 * kBlockN,
      smem_sfa_ptr + kBlockM + 3 * kBlockN
  };

  // Four WS barriers (double-buffer pipeline) at the last 32 bytes of SMEM.
  //   tma_mbar[2]: TMA completion per stage; math warps wait these.
  //   math_mbar[2]: SMEM consumed per stage; load warp waits these.
  // Both tma_mbar[] are also shared with Q TMA (tma_mbar[0] used for Q in preamble).
  static constexpr int kSmemMbar0Offset =
      Kernel_traits::kSmemSize - Kernel_traits::kSmemMbarSize;  // tma_mbar[0]
  uint64_t* tma_mbar_ptr[2] = {
      reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset),
      reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 8)
  };
  uint64_t* math_mbar_ptr[2] = {
      reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 16),
      reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 24)
  };

  // Thread 0 inits all 4 barriers.
  //   tma_mbar[0], tma_mbar[1]: count=1 (thread 0 will arrive.expect_tx each time)
  //   math_mbar[0], math_mbar[1]: count=8 (one arrive per math warp leader)
  // Pre-satisfy both math_mbar[0] and math_mbar[1] so load warp starts immediately.
  if (tidx == 0) {
    uint32_t tm0 = static_cast<uint32_t>(__cvta_generic_to_shared(tma_mbar_ptr[0]));
    uint32_t tm1 = static_cast<uint32_t>(__cvta_generic_to_shared(tma_mbar_ptr[1]));
    uint32_t mm0 = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr[0]));
    uint32_t mm1 = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr[1]));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(tm0), "r"(1));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(tm1), "r"(1));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(mm0), "r"(8));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(mm1), "r"(8));
    for (int i = 0; i < 8; i++) {
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" : : "r"(mm0));
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" : : "r"(mm1));
    }
  }
  asm volatile("fence.proxy.async.shared::cta;\n" : : : "memory");
  __syncthreads();
  // SF SMEM tensors: SFA is single-buffered; sSFB is created inside the math warp loop (stage-indexed).
  using SmemLayoutSFA = typename BS1::SmemLayoutSFA;
  using SmemLayoutSFB = typename BS1::SmemLayoutSFB;
  Tensor sSFA_ = make_tensor(make_smem_ptr(smem_sfa_ptr), SmemLayoutSFA{});
  auto sSFA = as_position_independent_swizzle_tensor(sSFA_);

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

  // Q+SFA TMA preamble: thread 0 issues Q and SFA TMA together on tma_mbar[0].
  {
    constexpr uint32_t kSmemQBytes   = kBlockM * kHeadDim * (uint32_t)sizeof(FP8Elem);
    constexpr uint32_t kSmemSFABytes = kBlockM * (uint32_t)sizeof(int32_t);
    using SmemLayoutSFA_TMA_t = cute::Layout<cute::Shape<cute::Int<kBlockM>, cute::Int<1>>,
                                             cute::Stride<cute::_1, cute::Int<kBlockM>>>;
    const int m_abs = binfo.sum_s_q / kBlockM + m_block;
    if (tidx == 0) {
      // Q TMA
      auto mQ_tma   = params.tma_q.get_tma_tensor(make_shape(params.total_q, params.d, params.h));
      auto gQ_head  = mQ_tma(_, _, bidh);
      auto gQ_tiles = local_tile(gQ_head, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(_, _));
      auto tma_slice_Q = params.tma_q.get_slice(0);
      auto tQsQ_d      = tma_slice_Q.partition_D(sQ_sw128);
      auto tQgQ_tma    = tma_slice_Q.partition_S(gQ_tiles(_, _, _, Int<0>{}));
      // SFA TMA
      auto mSFA_tma   = params.tma_sfa.get_tma_tensor(
          make_shape((int64_t)params.q_block_descale_head_stride, cute::Int<1>{}, params.h));
      auto gSFA_head  = mSFA_tma(_, _, bidh);
      auto gSFA_tiles = local_tile(gSFA_head, Shape<Int<kBlockM>, Int<1>>{}, make_coord(_, _));
      auto tma_slice_SFA = params.tma_sfa.get_slice(0);
      auto tSFAsSFA_d    = tma_slice_SFA.partition_D(
          make_tensor(make_smem_ptr(smem_sfa_ptr), SmemLayoutSFA_TMA_t{}));
      auto tSFAgSFA_tma  = tma_slice_SFA.partition_S(gSFA_tiles(_, _, _, Int<0>{}));
      // Issue Q and SFA together on tma_mbar[0]
      uint32_t taddr = static_cast<uint32_t>(__cvta_generic_to_shared(tma_mbar_ptr[0]));
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                   : : "r"(taddr), "r"(kSmemQBytes + kSmemSFABytes));
      cute::copy(params.tma_q.with(*tma_mbar_ptr[0]),   tQgQ_tma(_, _, _, m_abs),    tQsQ_d);
      cute::copy(params.tma_sfa.with(*tma_mbar_ptr[0]), tSFAgSFA_tma(_, _, _, m_abs), tSFAsSFA_d);
      wait_mbar_parity(tma_mbar_ptr[0], 0);
    }
  }
  __syncthreads();  // Q and SFA visible to all threads; tma_mbar_ptr[0] now at phase 1

  // s2r Q (math warps only)
  auto sQ_pi = as_position_independent_swizzle_tensor(sQ_sw128);
  Tensor tCrQ = thr_mma_g1.partition_fragment_A(sQ_pi);
  if (!is_load_warp) {
    auto tXsQ = s2r_thr_A.partition_S(sQ_pi);
    auto tXrQ = s2r_thr_A.retile_D(tCrQ);
    cute::copy(s2r_copy_A, tXsQ, tXrQ);
  }

  // SFA is now in smem_sfa_ptr (TMA-loaded in preamble, __syncthreads ensures visibility).

  Tensor tCrSFA = BS1::partition_fragment_SFA(sSFA(_,_,_0{}), thr_mma_g1);
  if (!is_load_warp) {
    auto tXsSFA = s2r_thr_SFA.partition_S(sSFA);
    auto tXrSFA = s2r_thr_SFA.retile_D(tCrSFA);
    cute::copy(s2r_copy_SFA, tXsSFA(_,_,_,_0{}), tXrSFA);
  }
  auto tCrSFA_frg = BS1::transform_fragment_for_qmma(tCrSFA);

  // K, V^T, and SFB TMA tensors — issued together per tile (single mbarrier).
  const int bidh_kv = bidh / params.h_h_k_ratio;
  constexpr int kSmemKBytes   = kBlockN * kHeadDim * (int)sizeof(FP8Elem);
  constexpr int kSmemVtBytes  = kHeadDim * kBlockN * (int)sizeof(FP8Elem);
  constexpr int kSmemSFBBytes = kBlockN * (int)sizeof(int32_t);
  constexpr int kSmemSFVBytes = kBlockN * (int)sizeof(int32_t);
  // Combined expected TX per stage: K + Vt + SFB + SFV
  constexpr uint32_t kSmemKVtSFBBytes = (uint32_t)(kSmemKBytes + kSmemVtBytes + kSmemSFBBytes + kSmemSFVBytes);
  // K TMA: [total_k, d, h_k]
  auto mK_tma   = params.tma_k.get_tma_tensor(make_shape(params.total_k, params.d, params.h_k));
  auto gK_head  = mK_tma(_, _, bidh_kv);
  auto gK_tiles = local_tile(gK_head, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _));
  auto tma_slice_K = params.tma_k.get_slice(0);
  auto tKgK_tma    = tma_slice_K.partition_S(gK_tiles(_, _, _, Int<0>{}));
  // Per-stage partition_D for K
  auto tKsK_d_0 = tma_slice_K.partition_D(
      make_tensor(make_smem_ptr(sK_base[0]), SmemLayoutK_SW128{}));
  auto tKsK_d_1 = tma_slice_K.partition_D(
      make_tensor(make_smem_ptr(sK_base[1]), SmemLayoutK_SW128{}));

  // V^T TMA: [d, total_k, h_k], tile [kHeadDim, kBlockN]
  auto mVt_tma   = params.tma_vt.get_tma_tensor(make_shape(params.d, params.total_k, params.h_k));
  auto gVt_head  = mVt_tma(_, _, bidh_kv);
  auto gVt_tiles = local_tile(gVt_head, Shape<Int<kHeadDim>, Int<kBlockN>>{}, make_coord(_, _));
  auto tma_slice_Vt = params.tma_vt.get_slice(0);
  auto tVtgVt_tma   = tma_slice_Vt.partition_S(gVt_tiles(_, _, Int<0>{}, _));
  // Per-stage partition_D for Vt
  auto tVtsVt_d_0 = tma_slice_Vt.partition_D(
      make_tensor(make_smem_ptr(sVt_base[0]), SmemLayoutVt_SW128{}));
  auto tVtsVt_d_1 = tma_slice_Vt.partition_D(
      make_tensor(make_smem_ptr(sVt_base[1]), SmemLayoutVt_SW128{}));

  // SFB TMA: [kv_block_descale_head_stride, 1, h_k], tile [kBlockN, 1], double-buffered.
  using SmemLayoutSFB_TMA_t = cute::Layout<cute::Shape<cute::Int<kBlockN>, cute::Int<1>>,
                                           cute::Stride<cute::_1, cute::Int<kBlockN>>>;
  auto mSFB_tma     = params.tma_sfb.get_tma_tensor(
      make_shape((int64_t)params.kv_block_descale_head_stride, cute::Int<1>{}, params.h_k));
  auto gSFB_head_k  = mSFB_tma(_, _, bidh_kv);
  auto gSFB_tiles_k = local_tile(gSFB_head_k, Shape<Int<kBlockN>, Int<1>>{}, make_coord(_, _));
  auto tma_slice_SFB  = params.tma_sfb.get_slice(0);
  auto tSFBgSFB_tma   = tma_slice_SFB.partition_S(gSFB_tiles_k(_, _, _, Int<0>{}));
  auto tSFBsSFB_d_0   = tma_slice_SFB.partition_D(
      make_tensor(make_smem_ptr(smem_sfb_ptr[0]), SmemLayoutSFB_TMA_t{}));
  auto tSFBsSFB_d_1   = tma_slice_SFB.partition_D(
      make_tensor(make_smem_ptr(smem_sfb_ptr[1]), SmemLayoutSFB_TMA_t{}));

  // SFV TMA: [v_block_descale_head_stride, 1, h_k], tile [kBlockN, 1], double-buffered.
  // sf_v_packed has been expanded Python-side to [H, total_tokens] (same structure as sf_k_packed).
  // Same SMEM layout as SFB; indexed by nb_abs = binfo.sum_s_k / kBlockN + nb.
  using SmemLayoutSFV_TMA_t = cute::Layout<cute::Shape<cute::Int<kBlockN>, cute::Int<1>>,
                                           cute::Stride<cute::_1, cute::Int<kBlockN>>>;
  auto mSFV_tma     = params.tma_sfv.get_tma_tensor(
      make_shape((int64_t)params.v_block_descale_head_stride, cute::Int<1>{}, params.h_k));
  auto gSFV_head_k  = mSFV_tma(_, _, bidh_kv);
  auto gSFV_tiles_k = local_tile(gSFV_head_k, Shape<Int<kBlockN>, Int<1>>{}, make_coord(_, _));
  auto tma_slice_SFV  = params.tma_sfv.get_slice(0);
  auto tSFVgSFV_tma   = tma_slice_SFV.partition_S(gSFV_tiles_k(_, _, _, Int<0>{}));
  auto tSFVsSFV_d_0   = tma_slice_SFV.partition_D(
      make_tensor(make_smem_ptr(smem_sfv_ptr[0]), SmemLayoutSFV_TMA_t{}));
  auto tSFVsSFV_d_1   = tma_slice_SFV.partition_D(
      make_tensor(make_smem_ptr(smem_sfv_ptr[1]), SmemLayoutSFV_TMA_t{}));

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

  // ===== DOUBLE-BUFFER WS PIPELINE =====
  // Phase tracking for tma_mbar[2]:
  //   tma_mbar[0] starts at phase=1 (after Q TMA), so first K stage-0 TMA advances it 1→0.
  //   Math warps wait parity=1 for stage-0 first tile (passes when phase=0).
  //   tma_mbar[1] starts at phase=0, so first K stage-1 TMA advances it 0→1.
  //   Math warps wait parity=0 for stage-1 first tile (passes when phase=1).
  //   After each wait: toggle the parity for that stage.
  // Phase tracking for math_mbar[2]:
  //   Both pre-satisfied → phase=1. Load warp waits parity=0 (passes when phase=1 ≠ 0).
  //   After wait: toggle to parity=1. Next wait (parity=1) blocks until math warps arrive.
  //
  // Layout of K+Vt SMEM helper lambda (stage s → base pointer):
  using SmemLayoutVt_K_SW128 = decltype(tile_to_shape(typename BS2::SmemLayoutAtomB{},
      Shape<Int<kHeadDim>, Int<kBlockN>>{}));

  if (is_load_warp) {
    // ===== LOAD WARP: double-buffer TMA producer =====
    // Phase state per stage: initially tma_wait = {1,0}, math_wait = {0,0} (both pre-satisfied).
    int math_wait_parity[2] = {0, 0};  // load warp waits math_mbar[s] with this parity

    // --- PREAMBLE: issue TMA for tile n_block_max-1 into stage 0 ---
    // Wait math_mbar[0] parity=0 (pre-satisfied → phase=1 → passes immediately)
    {
      uint32_t maddr = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr[0]));
      uint32_t done = 0;
      do {
        asm volatile("{.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;}\n"
                     : "=r"(done) : "r"(maddr), "r"((uint32_t)math_wait_parity[0]));
      } while (!done);
    }
    math_wait_parity[0] ^= 1;

    // Thread 0: issue K+Vt TMA for tile n_block_max-1 into stage 0 (tma_mbar[0])
    if (tidx == 0) {
      const int nb0 = Is_arbitrary ? int(sValidBlockIds[n_block_max - 1]) : (n_block_max - 1);
      const int nb_abs0 = binfo.sum_s_k / kBlockN + nb0;
      uint32_t taddr0 = static_cast<uint32_t>(__cvta_generic_to_shared(tma_mbar_ptr[0]));
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                   : : "r"(taddr0), "r"(kSmemKVtSFBBytes));
      cute::copy(params.tma_k.with(*tma_mbar_ptr[0]),   tKgK_tma(_, _, _, nb_abs0),    tKsK_d_0);
      cute::copy(params.tma_vt.with(*tma_mbar_ptr[0]),  tVtgVt_tma(_, _, _, nb_abs0),  tVtsVt_d_0);
      cute::copy(params.tma_sfb.with(*tma_mbar_ptr[0]), tSFBgSFB_tma(_, _, _, nb_abs0), tSFBsSFB_d_0);
      cute::copy(params.tma_sfv.with(*tma_mbar_ptr[0]), tSFVgSFV_tma(_, _, _, nb_abs0), tSFVsSFV_d_0);
      // Do NOT wait for TMA completion here — math warps wait on tma_mbar[0] directly.
    }

    // Compute load warp loop's starting n_valid by mirroring math warp's masking_step=0 update.
    // Math warp at masking_step=0: is_jump adjusts n_valid = min(n_block_max-1, n_block_history),
    // then loop --n_valid gives the n_valid for masking_step=1.
    // Load warp's loop starts at masking_step=1 and must use the SAME n_valid sequence as math warp.
    int load_n_valid_init = n_block_max - 1;
    if (is_jump && (n_masking_steps - 1 == 0))  // is_jump triggers at masking_step=0?
      load_n_valid_init = std::min(load_n_valid_init, n_block_history);
    load_n_valid_init -= 1;  // loop --n_valid after masking_step=0

    // --- LOAD LOOP: tiles for math warp's masking_step=1,2,... alternating stages 1→0→1... ---
    int load_stage = 1;
    for (int n_valid = load_n_valid_init, masking_step_load = 1; n_valid >= n_block_min;
         ++masking_step_load, --n_valid) {
      const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;

      // Wait for math warps to consume load_stage's SMEM before overwriting it
      {
        uint32_t maddr = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr[load_stage]));
        uint32_t done = 0;
        do {
          asm volatile("{.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;}\n"
                       : "=r"(done) : "r"(maddr), "r"((uint32_t)math_wait_parity[load_stage]));
        } while (!done);
      }
      math_wait_parity[load_stage] ^= 1;

      // Thread 0: issue K+Vt TMA for this tile into load_stage — do NOT wait for completion
      if (tidx == 0) {
        const int nb_abs = binfo.sum_s_k / kBlockN + nb;
        uint32_t taddr = static_cast<uint32_t>(__cvta_generic_to_shared(tma_mbar_ptr[load_stage]));
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                     : : "r"(taddr), "r"(kSmemKVtSFBBytes));
        if (load_stage == 0) {
          cute::copy(params.tma_k.with(*tma_mbar_ptr[0]),   tKgK_tma(_, _, _, nb_abs),    tKsK_d_0);
          cute::copy(params.tma_vt.with(*tma_mbar_ptr[0]),  tVtgVt_tma(_, _, _, nb_abs),  tVtsVt_d_0);
          cute::copy(params.tma_sfb.with(*tma_mbar_ptr[0]), tSFBgSFB_tma(_, _, _, nb_abs), tSFBsSFB_d_0);
          cute::copy(params.tma_sfv.with(*tma_mbar_ptr[0]), tSFVgSFV_tma(_, _, _, nb_abs), tSFVsSFV_d_0);
        } else {
          cute::copy(params.tma_k.with(*tma_mbar_ptr[1]),   tKgK_tma(_, _, _, nb_abs),    tKsK_d_1);
          cute::copy(params.tma_vt.with(*tma_mbar_ptr[1]),  tVtgVt_tma(_, _, _, nb_abs),  tVtsVt_d_1);
          cute::copy(params.tma_sfb.with(*tma_mbar_ptr[1]), tSFBgSFB_tma(_, _, _, nb_abs), tSFBsSFB_d_1);
          cute::copy(params.tma_sfv.with(*tma_mbar_ptr[1]), tSFVgSFV_tma(_, _, _, nb_abs), tSFVsSFV_d_1);
        }
        // TMA runs asynchronously — math warps will wait on tma_mbar[load_stage].
      }

      // Mirror math warp's is_jump at the same masking_step index
      if (is_jump && masking_step_load == n_masking_steps - 1)
        n_valid = std::min(n_valid, n_block_history);

      load_stage ^= 1;
    }
    // Load warp done; falls through to __syncthreads() below.

  } else {
    // ===== MATH WARPS: double-buffer QMMA consumer loop =====
    // Math warps wait tma_mbar[stage] directly (no load_mbar intermediary).
    // tma_wait_parity[0]=1: after Q TMA phase=1, K stage-0 TMA: phase 1→0; wait parity=1 passes.
    // tma_wait_parity[1]=0: tma_mbar[1] fresh phase=0, K stage-1 TMA: phase 0→1; wait parity=0 passes.
    int tma_wait_parity[2] = {1, 0};
    int math_stage = 0;  // starts with stage 0 (preamble loaded tile N-1 there)

    // SFA (unit scale for P) is constant across all N-block tiles — initialize once.
    for (int i = tidx_math; i < kBlockM; i += kNMathThreads)
      smem_sfa_ptr[i] = 0x7f7f7f7f;

    for (int n_valid = n_block_max - 1, masking_step = 0; n_valid >= n_block_min;
         ++masking_step, --n_valid) {
      const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;
      const bool is_masking = masking_step < n_masking_steps ||
          (nb + 1) * kBlockN > actual_seqlen_h;

      // Wait for TMA K[nb]+Vt[nb]+SFB[nb] to land in SMEM for math_stage
      {
        uint32_t taddr = static_cast<uint32_t>(__cvta_generic_to_shared(tma_mbar_ptr[math_stage]));
        uint32_t done = 0;
        do {
          asm volatile("{.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;}\n"
                       : "=r"(done) : "r"(taddr), "r"((uint32_t)tma_wait_parity[math_stage]));
        } while (!done);
      }
      tma_wait_parity[math_stage] ^= 1;
      // Sync math warps: TMA K+Vt+SFB all visible
      asm volatile("bar.sync 1, 256;\n" : : : "memory");

      // SFB is in smem_sfb_ptr[math_stage] (TMA-loaded by load warp along with K+Vt)
      Tensor sSFB_ = make_tensor(make_smem_ptr(smem_sfb_ptr[math_stage]), SmemLayoutSFB{});
      auto sSFB = as_position_independent_swizzle_tensor(sSFB_);

      // Stage-specific SMEM pointers for K and Vt
      FP8Elem* sK_cur  = sK_base[math_stage];
      FP8Elem* sVt_cur = sVt_base[math_stage];

      // s2r K from sK[math_stage]
      auto sK_cur_pi = as_position_independent_swizzle_tensor(
          make_tensor(make_smem_ptr(sK_cur), SmemLayoutK_SW128{}));
      Tensor tCrK = thr_mma_g1.partition_fragment_B(sK_cur_pi);
      {
        auto tXsK = s2r_thr_B.partition_S(sK_cur_pi);
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
          uint32_t maddr = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr[math_stage]));
          asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" : : "r"(maddr));
        }
        if (is_jump && masking_step == n_masking_steps - 1)
          n_valid = std::min(n_valid, n_block_history);
        math_stage ^= 1;
        continue;
      }

      if (Is_arbitrary || Is_local || is_masking) apply_mask_bs(acc_s, nb);
      for (int i = 0; i < size(acc_s); ++i) acc_s(i) *= params.alpha;
      fast_silu(acc_s);

      // Convert acc_s → FP8 rP
      Tensor rP = make_tensor_like<FP8Elem>(acc_s);
      flash::convert_type_safe(acc_s, rP);

      // P SMEM roundtrip: write rP to sPbuf = sK[math_stage] (K already in registers, buffer free)
      // bar.sync before write: ensure all warps done s2r K
      asm volatile("bar.sync 1, 256;\n" : : : "memory");
      Tensor sPbuf = make_tensor(make_smem_ptr(sK_cur), SmemLayoutQ_SW128{});
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

      // s2r P (GEMM2 A-operand) from sK[math_stage] (now used as P buffer)
      Tensor tCrP = thr_mma_g2.partition_fragment_A(sPbuf_pi);
      {
        auto tXsP = s2r_thr_A2.partition_S(sPbuf_pi);
        auto tXrP = s2r_thr_A2.retile_D(tCrP);
        cute::copy(s2r_copy_A2, tXsP, tXrP);
      }
      // bar.sync: all threads done reading P; safe to reuse sK[math_stage] for V^T K_SW128.
      asm volatile("bar.sync 1, 256;\n" : : : "memory");

      // Cooperative SMEM transpose: Vt[math_stage] MN_SW128 → sK[math_stage] K_SW128.
      // Root cause: partition_B with MN_SW128 gives N-first register ordering; QMMA needs K-first.
      // Fix: copy Vt to sK[math_stage] in K_SW128 layout so standard LDSM_N gives correct ordering.
      {
        Tensor sVt_mn = make_tensor(make_smem_ptr(sVt_cur), SmemLayoutVt_SW128{});
        Tensor sVt_k  = make_tensor(make_smem_ptr(sK_cur),  SmemLayoutVt_K_SW128{});
        for (int i = tidx_math; i < kHeadDim * kBlockN; i += kNMathThreads) {
          sVt_k(i / kBlockN, i % kBlockN) = sVt_mn(i / kBlockN, i % kBlockN);
        }
      }
      // bar.sync: K_SW128 V^T visible to all math warps before s2r.
      asm volatile("bar.sync 1, 256;\n" : : : "memory");

      // s2r V^T (GEMM2 B-operand) from K_SW128 SMEM in sK[math_stage]
      auto sVt_k_pi = as_position_independent_swizzle_tensor(
          make_tensor(make_smem_ptr(sK_cur), SmemLayoutVt_K_SW128{}));
      Tensor tCrV = thr_mma_g2.partition_fragment_B(sVt_k_pi);
      {
        auto tXsVt = s2r_thr_B2.partition_S(sVt_k_pi);
        auto tXrV  = s2r_thr_B2.retile_D(tCrV);
        cute::copy(s2r_copy_B2, tXsVt, tXrV);
      }

      // s2r SFP and SFV (from SF SMEM).
      // SFP (unit) is in smem_sfa_ptr (written above); SFV is in smem_sfv_ptr[math_stage] (TMA-loaded).
      Tensor tCrSFP = BS2::partition_fragment_SFA(sSFA(_,_,_0{}), thr_mma_g2);
      {
        auto tXsSFP = s2r_thr_SFP.partition_S(sSFA);
        auto tXrSFP = s2r_thr_SFP.retile_D(tCrSFP);
        cute::copy(s2r_copy_SFP, tXsSFP(_,_,_,_0{}), tXrSFP);
      }
      auto tCrSFP_frg = BS2::transform_fragment_for_qmma(tCrSFP);
      Tensor sSFV_ = make_tensor(make_smem_ptr(smem_sfv_ptr[math_stage]), SmemLayoutSFB{});
      auto sSFV = as_position_independent_swizzle_tensor(sSFV_);
      Tensor tCrSFV = BS2::partition_fragment_SFB(sSFV(_,_,_0{}), thr_mma_g2);
      {
        auto tXsSFV = s2r_thr_SFV.partition_S(sSFV);
        auto tXrSFV = s2r_thr_SFV.retile_D(tCrSFV);
        cute::copy(s2r_copy_SFV, tXsSFV(_,_,_,_0{}), tXrSFV);
      }
      auto tCrSFV_frg = BS2::transform_fragment_for_qmma(tCrSFV);

      // Notify load warp: sK[math_stage] and sVt[math_stage] fully consumed.
      // Load warp can now overwrite this stage's SMEM with the next K+Vt tile.
      // GEMM2 below uses only registers, so overwriting SMEM is safe.
      if ((tidx & 31) == 0) {
        uint32_t maddr = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr[math_stage]));
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" : : "r"(maddr));
      }

      // GEMM2: acc_o += P × V^T (block-scaled) — runs while load warp issues next TMA
      cute::gemm(tiled_mma_g2,
          make_zip_tensor(tCrP, tCrSFP_frg(_,_,_,_0{})),
          make_zip_tensor(tCrV, tCrSFV_frg(_,_,_,_0{})),
          acc_o);

      if (is_jump && masking_step == n_masking_steps - 1)
        n_valid = std::min(n_valid, n_block_history);

      math_stage ^= 1;
    }
  }

  // ===== EPILOGUE =====
  // Math warps write acc_o to flat SMEM; load warp idles.
  if (!is_load_warp) {
    for (int i = 0; i < size(acc_o); ++i) acc_o(i) /= params.scaling_seqlen;
    using OutElement = typename Kernel_traits::OutputType;
    Tensor rO = make_tensor_like<OutElement>(acc_o);
    flash::convert_type_safe(acc_o, rO);

    // bar.sync: all 256 math warps must exit the loop (finish s2r V^T from smem_q)
    // before any warp overwrites smem_q with the output. Without this, for n_block_max=1
    // (SEQ=128), a fast warp can write acc_o to sK_base[0]=smem_q while a slow warp
    // is still reading V^T from sK_base[0] (math_stage=0 → sK_cur=sK_base[0]).
    // For n_block_max≥2, the last iteration uses math_stage=1 → sK_base[1] (no conflict).
    asm volatile("bar.sync 1, 256;\n" : : : "memory");
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

// Phase 6 WS TMA kernel entry: launched with kNThreadsTotal=288.
// Q, K, V^T, Q-SF, K-SF, and V-SF all via TMA.
template <typename Kernel_traits, typename TMA_Q_t, typename TMA_K_t, typename TMA_Vt_t,
          typename TMA_SFA_t, typename TMA_SFB_t, typename TMA_SFV_t>
__global__ void __launch_bounds__(Kernel_traits::kNThreads)
hstu_fwd_kernel_sm120_fp8_ws_tma(
    __grid_constant__ Hstu_fwd_params_fp8_ws_tma<TMA_Q_t, TMA_K_t, TMA_Vt_t,
                                                  TMA_SFA_t, TMA_SFB_t, TMA_SFV_t> const params) {
  int m_block = gridDim.x - blockIdx.x - 1;
  int bidh    = blockIdx.y;
  int bidb    = blockIdx.z;
  hstu_compute_attn_1rowblock_sm120_fp8_ws<Kernel_traits>(params, bidb, bidh, m_block);
}

