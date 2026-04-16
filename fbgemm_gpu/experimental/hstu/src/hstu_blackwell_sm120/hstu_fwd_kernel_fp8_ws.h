// Phase 6 warp-specialized FP8 kernel body — Q, K, and V^T via TMA.
// Included inside namespace flash from hstu_fwd_kernel.h.
// Do not include directly; use hstu_fwd_kernel.h.
//
// Warp layout (12 warps = 3 complete warpgroups, kNThreads=384):
//   WG0: warps 0-3  (math)
//   WG1: warps 4-7  (math)
//   WG2: warps 8-11 (load warpgroup)
//     warp 8  = active load warp: does TMA, barrier init, mbarrier spin
//     warps 9-11 = idle load warps: dec registers, participate in syncs, then idle
// Complete WG2 ensures setmaxnreg TRY_ALLOC WARPSYNC.ALL retry loop works correctly.
// With __launch_bounds__(384,1): compiler budget = 65536/384 ≈ 168 regs → TRY_ALLOC(inc) succeeds
// on first attempt (already at budget), no retry needed.
//
// CTA-level __syncthreads__ map (must match between branches):
//   S_arb : Is_arbitrary only — math warp 1 writes sValidBlockIds; load warp just syncs
//   S1    : after load warp inits barriers + fence
//   S2    : after load warp issues Q+SFA TMA and waits (Q visible to math warps)
//   S3    : before main WS loop
//   S5    : epilogue — after math warps write acc_o to SMEM (sO visible for GMEM copy)

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
  constexpr int kNMathThreads = Kernel_traits::kNMathThreads;  // 256
  const bool is_load_warp = (tidx >= kNMathThreads);

  // CTA-level sync before setmaxnreg: all warps start from clean state.
  __syncthreads();

  // ============================================================
  // LOAD WARPGROUP PATH  (warps 8-11, threads 256-383)
  //   warp 8  (tidx 256-287): active load warp — TMA + barrier init
  //   warps 9-11 (tidx 288-383): idle — dec registers, sync, then idle after S3
  // ============================================================
  if (is_load_warp) {
    //printf("load warp, befor setmaxnreg, %d\n", tidx);
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(64));
    //printf("load warp, after setmaxnreg, %d\n", tidx);
    // is_active_load: true only for warp 8 (threads 256-287)
    const bool is_active_load = (tidx < kNMathThreads + 32);

    constexpr bool Is_causal    = Kernel_traits::Is_causal;
    constexpr bool Is_target    = Kernel_traits::Is_target;
    constexpr bool Is_context   = Kernel_traits::Is_context;
    constexpr bool Is_arbitrary = Kernel_traits::Is_arbitrary;
    constexpr int  kNFunc       = Kernel_traits::kNFunc;
    constexpr bool Is_local     = Kernel_traits::Is_local;
    constexpr int  kBlockM      = Kernel_traits::kBlockM;
    constexpr int  kBlockN      = Kernel_traits::kBlockN;
    constexpr int  kHeadDim     = Kernel_traits::kHeadDim;

    const HstuBlockInfo<Kernel_traits, Params> binfo(params, bidb);
    // Early exit 1: before any sync — both branches exit simultaneously.
    if (m_block * kBlockM >= binfo.actual_seqlen_q_padded) return;

    char* smem_q    = reinterpret_cast<char*>(smem_);
    char* smem_func = reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsFuncOffset;
    int* sn_valid_block_max = reinterpret_cast<int*>(smem_func);

    const int actual_seqlen_q        = binfo.actual_seqlen_q;
    const int actual_seqlen_k        = binfo.actual_seqlen_k;
    const int actual_seqlen_q_padded = binfo.actual_seqlen_q_padded;
    const int actual_seqlen_t        = Is_target  ? binfo.actual_seqlen_t : 0;
    const int actual_seqlen_c        = Is_context ? binfo.actual_seqlen_c : 0;
    const int actual_seqlen_h        = Is_target  ? actual_seqlen_k - actual_seqlen_t : actual_seqlen_k;
    const int actual_seqlen_offset   = actual_seqlen_k - actual_seqlen_q;

    const bool is_jump             = Is_target && m_block * kBlockM + actual_seqlen_offset > actual_seqlen_h;
    const bool is_in_context       = Is_context && (m_block + 1) * kBlockM <= actual_seqlen_c;
    const bool is_in_mixed_context = Is_context &&
        (m_block + 1) * kBlockM > actual_seqlen_c && m_block * kBlockM < actual_seqlen_c;

    const int n_block_history = cute::ceil_div(actual_seqlen_h, kBlockN);
    const int target_index    = (m_block * kBlockM - actual_seqlen_h) / params.target_group_size;

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

    // Is_arbitrary: load warp only participates in __syncthreads__; math warp 1 does the work.
    if constexpr (Is_arbitrary) {
      __syncthreads();  // S_arb: wait for math warp 1 to write sValidBlockIds
      n_block_max = *sn_valid_block_max;
      n_block_min = 0;
    }

    // Early exit 2 (after Is_arbitrary): both branches exit with 4 syncs ahead.
    if (((Is_causal || Is_local || Is_arbitrary) && n_block_max <= n_block_min) ||
        m_block * kBlockM >= actual_seqlen_q) {
      __syncthreads();  // S1
      __syncthreads();  // S2
      __syncthreads();  // S3
      __syncthreads();  // S5
      return;
    }

    // SMEM layout types for TMA partition_D.
    using SmemLayoutK_SW128  = decltype(tile_to_shape(typename BS1::SmemLayoutAtomB{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutVt_SW128 = typename Kernel_traits::SmemLayoutVt_TMA;
    using SmemLayoutQ_SW128  = decltype(tile_to_shape(typename BS1::SmemLayoutAtomA{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    constexpr int kSmemKVElems = kBlockN * kHeadDim;
    FP8Elem* const sK_base[2] = {
        reinterpret_cast<FP8Elem*>(smem_q),
        reinterpret_cast<FP8Elem*>(smem_q) + 2 * kSmemKVElems
    };
    FP8Elem* const sVt_base[2] = {
        reinterpret_cast<FP8Elem*>(smem_q) + kSmemKVElems,
        reinterpret_cast<FP8Elem*>(smem_q) + 3 * kSmemKVElems
    };

    // Barrier pointers.
    static constexpr int kSmemMbar0Offset =
        Kernel_traits::kSmemSize - Kernel_traits::kSmemMbarSize;
    uint64_t* tma_mbar_ptr[2] = {
        reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset),
        reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 8)
    };
    uint64_t* math_mbar_ptr[2] = {
        reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 16),
        reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 24)
    };

    // SF SMEM pointers.
    static constexpr int kSmemSFOffset_WS = Kernel_traits::kSmemWsDataSizePadded;
    int32_t* smem_sfa_ptr = reinterpret_cast<int32_t*>(smem_ + kSmemSFOffset_WS);
    int32_t* smem_sfp_ptr = smem_sfa_ptr + kBlockM;
    int32_t* const smem_sfb_ptr[2] = {
        smem_sfa_ptr + 2 * kBlockM,
        smem_sfa_ptr + 2 * kBlockM + kBlockN
    };
    int32_t* const smem_sfv_ptr[2] = {
        smem_sfa_ptr + 2 * kBlockM + 2 * kBlockN,
        smem_sfa_ptr + 2 * kBlockM + 3 * kBlockN
    };

    // Barrier init: only thread kNMathThreads (load warp's first thread).
    if (tidx == kNMathThreads) {
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
    __syncthreads();  // S1: barriers initialized and visible to all warps.

    // Q+SFA TMA preamble: only thread kNMathThreads issues TMA and waits.
    {
      constexpr uint32_t kSmemQBytes   = kBlockM * kHeadDim * (uint32_t)sizeof(FP8Elem);
      constexpr uint32_t kSmemSFABytes = kBlockM * (uint32_t)sizeof(int32_t);
      using SmemLayoutSFA_TMA_t = cute::Layout<cute::Shape<cute::Int<kBlockM>, cute::Int<1>>,
                                               cute::Stride<cute::_1, cute::Int<kBlockM>>>;
      const int m_abs = binfo.sum_s_q / kBlockM + m_block;
      if (tidx == kNMathThreads) {
        auto mQ_tma   = params.tma_q.get_tma_tensor(make_shape(params.total_q, params.d, params.h));
        auto gQ_head  = mQ_tma(_, _, bidh);
        auto gQ_tiles = local_tile(gQ_head, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(_, _));
        auto tma_slice_Q = params.tma_q.get_slice(0);
        auto sQ_buf      = make_tensor(make_smem_ptr(sK_base[0]), SmemLayoutQ_SW128{});
        auto tQsQ_d      = tma_slice_Q.partition_D(sQ_buf);
        auto tQgQ_tma    = tma_slice_Q.partition_S(gQ_tiles(_, _, _, Int<0>{}));
        auto mSFA_tma    = params.tma_sfa.get_tma_tensor(
            make_shape((int64_t)params.q_block_descale_head_stride, cute::Int<1>{}, params.h));
        auto gSFA_head   = mSFA_tma(_, _, bidh);
        auto gSFA_tiles  = local_tile(gSFA_head, Shape<Int<kBlockM>, Int<1>>{}, make_coord(_, _));
        auto tma_slice_SFA  = params.tma_sfa.get_slice(0);
        auto tSFAsSFA_d     = tma_slice_SFA.partition_D(
            make_tensor(make_smem_ptr(smem_sfa_ptr), SmemLayoutSFA_TMA_t{}));
        auto tSFAgSFA_tma   = tma_slice_SFA.partition_S(gSFA_tiles(_, _, _, Int<0>{}));
        uint32_t taddr = static_cast<uint32_t>(__cvta_generic_to_shared(tma_mbar_ptr[0]));
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                     : : "r"(taddr), "r"(kSmemQBytes + kSmemSFABytes));
        cute::copy(params.tma_q.with(*tma_mbar_ptr[0]),   tQgQ_tma(_, _, _, m_abs),     tQsQ_d);
        cute::copy(params.tma_sfa.with(*tma_mbar_ptr[0]), tSFAgSFA_tma(_, _, _, m_abs), tSFAsSFA_d);
        wait_mbar_parity(tma_mbar_ptr[0], 0);  // spin until Q+SFA TMA complete
      }
    }
    __syncthreads();  // S2: Q and SFA visible to math warps; tma_mbar[0] now at phase 1.

    // TMA tensor setup for K, V^T, SFB, SFV (load warp only).
    const int bidh_kv = bidh / params.h_h_k_ratio;
    constexpr int kSmemKBytes    = kBlockN * kHeadDim * (int)sizeof(FP8Elem);
    constexpr int kSmemVtBytes   = kHeadDim * kBlockN * (int)sizeof(FP8Elem);
    constexpr int kSmemSFBBytes  = kBlockN * (int)sizeof(int32_t);
    constexpr int kSmemSFVBytes  = kBlockN * (int)sizeof(int32_t);
    constexpr uint32_t kSmemKVtSFBBytes =
        (uint32_t)(kSmemKBytes + kSmemVtBytes + kSmemSFBBytes + kSmemSFVBytes);

    auto mK_tma    = params.tma_k.get_tma_tensor(make_shape(params.total_k, params.d, params.h_k));
    auto gK_head   = mK_tma(_, _, bidh_kv);
    auto gK_tiles  = local_tile(gK_head, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _));
    auto tma_slice_K  = params.tma_k.get_slice(0);
    auto tKgK_tma     = tma_slice_K.partition_S(gK_tiles(_, _, _, Int<0>{}));
    auto tKsK_d_0     = tma_slice_K.partition_D(
        make_tensor(make_smem_ptr(sK_base[0]), SmemLayoutK_SW128{}));
    auto tKsK_d_1     = tma_slice_K.partition_D(
        make_tensor(make_smem_ptr(sK_base[1]), SmemLayoutK_SW128{}));

    auto mVt_tma   = params.tma_vt.get_tma_tensor(make_shape(params.d, params.total_k, params.h_k));
    auto gVt_head  = mVt_tma(_, _, bidh_kv);
    auto gVt_tiles = local_tile(gVt_head, Shape<Int<kHeadDim>, Int<kBlockN>>{}, make_coord(_, _));
    auto tma_slice_Vt  = params.tma_vt.get_slice(0);
    auto tVtgVt_tma    = tma_slice_Vt.partition_S(gVt_tiles(_, _, Int<0>{}, _));
    auto tVtsVt_d_0    = tma_slice_Vt.partition_D(
        make_tensor(make_smem_ptr(sVt_base[0]), SmemLayoutVt_SW128{}));
    auto tVtsVt_d_1    = tma_slice_Vt.partition_D(
        make_tensor(make_smem_ptr(sVt_base[1]), SmemLayoutVt_SW128{}));

    using SmemLayoutSFB_TMA_t = cute::Layout<cute::Shape<cute::Int<kBlockN>, cute::Int<1>>,
                                             cute::Stride<cute::_1, cute::Int<kBlockN>>>;
    auto mSFB_tma      = params.tma_sfb.get_tma_tensor(
        make_shape((int64_t)params.kv_block_descale_head_stride, cute::Int<1>{}, params.h_k));
    auto gSFB_head_k   = mSFB_tma(_, _, bidh_kv);
    auto gSFB_tiles_k  = local_tile(gSFB_head_k, Shape<Int<kBlockN>, Int<1>>{}, make_coord(_, _));
    auto tma_slice_SFB = params.tma_sfb.get_slice(0);
    auto tSFBgSFB_tma  = tma_slice_SFB.partition_S(gSFB_tiles_k(_, _, _, Int<0>{}));
    auto tSFBsSFB_d_0  = tma_slice_SFB.partition_D(
        make_tensor(make_smem_ptr(smem_sfb_ptr[0]), SmemLayoutSFB_TMA_t{}));
    auto tSFBsSFB_d_1  = tma_slice_SFB.partition_D(
        make_tensor(make_smem_ptr(smem_sfb_ptr[1]), SmemLayoutSFB_TMA_t{}));

    using SmemLayoutSFV_TMA_t = cute::Layout<cute::Shape<cute::Int<kBlockN>, cute::Int<1>>,
                                             cute::Stride<cute::_1, cute::Int<kBlockN>>>;
    auto mSFV_tma      = params.tma_sfv.get_tma_tensor(
        make_shape((int64_t)params.v_block_descale_head_stride, cute::Int<1>{}, params.h_k));
    auto gSFV_head_k   = mSFV_tma(_, _, bidh_kv);
    auto gSFV_tiles_k  = local_tile(gSFV_head_k, Shape<Int<kBlockN>, Int<1>>{}, make_coord(_, _));
    auto tma_slice_SFV = params.tma_sfv.get_slice(0);
    auto tSFVgSFV_tma  = tma_slice_SFV.partition_S(gSFV_tiles_k(_, _, _, Int<0>{}));
    auto tSFVsSFV_d_0  = tma_slice_SFV.partition_D(
        make_tensor(make_smem_ptr(smem_sfv_ptr[0]), SmemLayoutSFV_TMA_t{}));
    auto tSFVsSFV_d_1  = tma_slice_SFV.partition_D(
        make_tensor(make_smem_ptr(smem_sfv_ptr[1]), SmemLayoutSFV_TMA_t{}));

    Tensor sValidBlockIds = make_tensor(
        make_smem_ptr(reinterpret_cast<int*>(smem_ + Kernel_traits::kSmemWsValidBlockIdsOffset)),
        typename Kernel_traits::SmemLayoutValidBlockIds{});

    __syncthreads();  // S3: before main loop; math warps finished Q→persist SMEM copy.

    // ===== LOAD WARP DOUBLE-BUFFER TMA PRODUCER =====
    // Only active load warp (warp 8) participates.
    // Idle load warps (9-11) skip this entire section and go directly to S5.
    if (is_active_load) {
    int math_wait_parity[2] = {0, 0};

    // Preamble: wait math_mbar[0] (pre-satisfied), then issue tile n_block_max-1 into stage 0.
    {
      uint32_t maddr = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr[0]));
      uint32_t done = 0;
      do {
        asm volatile("{.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;}\n"
                     : "=r"(done) : "r"(maddr), "r"((uint32_t)math_wait_parity[0]));
      } while (!done);
    }
    math_wait_parity[0] ^= 1;

    if (tidx == kNMathThreads) {
      const int nb0     = Is_arbitrary ? int(sValidBlockIds[n_block_max - 1]) : (n_block_max - 1);
      const int nb_abs0 = binfo.sum_s_k / kBlockN + nb0;
      uint32_t taddr0   = static_cast<uint32_t>(__cvta_generic_to_shared(tma_mbar_ptr[0]));
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                   : : "r"(taddr0), "r"(kSmemKVtSFBBytes));
      cute::copy(params.tma_k.with(*tma_mbar_ptr[0]),   tKgK_tma(_, _, _, nb_abs0),    tKsK_d_0);
      cute::copy(params.tma_vt.with(*tma_mbar_ptr[0]),  tVtgVt_tma(_, _, _, nb_abs0),  tVtsVt_d_0);
      cute::copy(params.tma_sfb.with(*tma_mbar_ptr[0]), tSFBgSFB_tma(_, _, _, nb_abs0), tSFBsSFB_d_0);
      cute::copy(params.tma_sfv.with(*tma_mbar_ptr[0]), tSFVgSFV_tma(_, _, _, nb_abs0), tSFVsSFV_d_0);
    }

    int load_n_valid_init = n_block_max - 1;
    if (is_jump && (n_masking_steps - 1 == 0))
      load_n_valid_init = std::min(load_n_valid_init, n_block_history);
    load_n_valid_init -= 1;

    int load_stage = 1;
    for (int n_valid = load_n_valid_init, masking_step_load = 1; n_valid >= n_block_min;
         ++masking_step_load, --n_valid) {
      const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;

      {
        uint32_t maddr = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr[load_stage]));
        uint32_t done = 0;
        do {
          asm volatile("{.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;}\n"
                       : "=r"(done) : "r"(maddr), "r"((uint32_t)math_wait_parity[load_stage]));
        } while (!done);
      }
      math_wait_parity[load_stage] ^= 1;

      if (tidx == kNMathThreads) {
        const int nb_abs = binfo.sum_s_k / kBlockN + nb;
        uint32_t taddr   = static_cast<uint32_t>(__cvta_generic_to_shared(tma_mbar_ptr[load_stage]));
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
      }

      if (is_jump && masking_step_load == n_masking_steps - 1)
        n_valid = std::min(n_valid, n_block_history);

      load_stage ^= 1;
    }
    }  // end if (is_active_load)
    // Idle load warps (9-11) reach here immediately after S3.
    // All 4 load warps (+ math warps) meet at S5 when everyone is done.
    __syncthreads();  // S5: epilogue — wait for math warps to write acc_o to SMEM.
    // All load warps idle while math warps copy sO to GMEM.

  // ============================================================
  } else {
  // MATH WARP PATH  (warps 0-7, threads 0-255)
  // ============================================================
    //printf("math warp, befor setmaxnreg, %d\n", tidx);
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" : : "n"(216));
    //printf("math warp, after setmaxnreg, %d\n", tidx);

    constexpr bool Is_causal    = Kernel_traits::Is_causal;
    constexpr bool Is_target    = Kernel_traits::Is_target;
    constexpr bool Is_context   = Kernel_traits::Is_context;
    constexpr bool Is_arbitrary = Kernel_traits::Is_arbitrary;
    constexpr int  kNFunc       = Kernel_traits::kNFunc;
    constexpr bool Is_local     = Kernel_traits::Is_local;
    constexpr int  kBlockM      = Kernel_traits::kBlockM;
    constexpr int  kBlockN      = Kernel_traits::kBlockN;
    constexpr int  kHeadDim     = Kernel_traits::kHeadDim;

    const int tidx_math = tidx;  // math warps: tidx_math == tidx (0-255)

    const HstuBlockInfo<Kernel_traits, Params> binfo(params, bidb);
    // Early exit 1: before any sync — both branches exit simultaneously.
    if (m_block * kBlockM >= binfo.actual_seqlen_q_padded) return;

    char* smem_q    = reinterpret_cast<char*>(smem_);
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

    const bool is_jump             = Is_target && m_block * kBlockM + actual_seqlen_offset > actual_seqlen_h;
    const bool is_in_context       = Is_context && (m_block + 1) * kBlockM <= actual_seqlen_c;
    const bool is_in_mixed_context = Is_context &&
        (m_block + 1) * kBlockM > actual_seqlen_c && m_block * kBlockM < actual_seqlen_c;

    const int n_block_history = cute::ceil_div(actual_seqlen_h, kBlockN);
    const int target_index    = (m_block * kBlockM - actual_seqlen_h) / params.target_group_size;

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

    // Arbitrary func GMEM tensors (only needed when Is_arbitrary, but declared always for lambda).
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

    // SMEM tensors for Is_arbitrary.
    Tensor sValidBlockIds = make_tensor(
        make_smem_ptr(reinterpret_cast<int*>(smem_ + Kernel_traits::kSmemWsValidBlockIdsOffset)),
        typename Kernel_traits::SmemLayoutValidBlockIds{});
    Tensor sFunc_min = make_tensor(make_smem_ptr(sf_min_ptr), typename Kernel_traits::SmemLayoutMinFunc{});
    Tensor sFunc_max = make_tensor(make_smem_ptr(sf_max_ptr), typename Kernel_traits::SmemLayoutMaxFunc{});

    // Is_arbitrary setup: warp 1 (threads 32-63) does the computation; load warp just syncs.
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
              if (c1 || c2 || c3) {
                sValidBlockIds[*sn_valid_block_max] = n_block;
                (*sn_valid_block_max)++;
                break;
              }
            }
          }
        }
      }
      __syncthreads();  // S_arb: sValidBlockIds written by warp 1, visible to all (incl. load warp).
      n_block_max = *sn_valid_block_max;
      n_block_min = 0;
    }

    // Early exit 2 (after Is_arbitrary): math warps write zeros; both branches have 4 syncs ahead.
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
      flash::copy<false, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO,
          actual_seqlen_q_padded - m_block * kBlockM);
      __syncthreads();  // S1
      __syncthreads();  // S2
      __syncthreads();  // S3
      __syncthreads();  // S5
      return;
    }

    // SW128 SMEM layouts.
    using SmemLayoutQ_SW128  = decltype(tile_to_shape(typename BS1::SmemLayoutAtomA{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutK_SW128  = decltype(tile_to_shape(typename BS1::SmemLayoutAtomB{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutVt_SW128 = typename Kernel_traits::SmemLayoutVt_TMA;

    constexpr int kSmemKVElems = kBlockN * kHeadDim;
    FP8Elem* const sK_base[2] = {
        reinterpret_cast<FP8Elem*>(smem_q),
        reinterpret_cast<FP8Elem*>(smem_q) + 2 * kSmemKVElems
    };
    FP8Elem* const sVt_base[2] = {
        reinterpret_cast<FP8Elem*>(smem_q) + kSmemKVElems,
        reinterpret_cast<FP8Elem*>(smem_q) + 3 * kSmemKVElems
    };
    Tensor sQ_sw128 = make_tensor(make_smem_ptr(sK_base[0]), SmemLayoutQ_SW128{});

    // Barrier pointers (same SMEM addresses as load warp).
    static constexpr int kSmemMbar0Offset =
        Kernel_traits::kSmemSize - Kernel_traits::kSmemMbarSize;
    uint64_t* tma_mbar_ptr[2] = {
        reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset),
        reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 8)
    };
    uint64_t* math_mbar_ptr[2] = {
        reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 16),
        reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 24)
    };

    // SF SMEM pointers.
    static constexpr int kSmemSFOffset_WS = Kernel_traits::kSmemWsDataSizePadded;
    int32_t* smem_sfa_ptr = reinterpret_cast<int32_t*>(smem_ + kSmemSFOffset_WS);
    int32_t* smem_sfp_ptr = smem_sfa_ptr + kBlockM;
    int32_t* const smem_sfb_ptr[2] = {
        smem_sfa_ptr + 2 * kBlockM,
        smem_sfa_ptr + 2 * kBlockM + kBlockN
    };
    int32_t* const smem_sfv_ptr[2] = {
        smem_sfa_ptr + 2 * kBlockM + 2 * kBlockN,
        smem_sfa_ptr + 2 * kBlockM + 3 * kBlockN
    };

    using SmemLayoutSFA = typename BS1::SmemLayoutSFA;
    using SmemLayoutSFB = typename BS1::SmemLayoutSFB;
    Tensor sSFA_ = make_tensor(make_smem_ptr(smem_sfa_ptr), SmemLayoutSFA{});
    auto sSFA = as_position_independent_swizzle_tensor(sSFA_);
    Tensor sSFP_ = make_tensor(make_smem_ptr(smem_sfp_ptr), SmemLayoutSFA{});
    auto sSFP = as_position_independent_swizzle_tensor(sSFP_);

    // MMA tiled objects (use tidx_math = tidx for math warps).
    typename BS1::TiledMma tiled_mma_g1;
    auto thr_mma_g1 = tiled_mma_g1.get_thread_slice(tidx_math);
    typename BS2::TiledMma tiled_mma_g2;
    auto thr_mma_g2 = tiled_mma_g2.get_thread_slice(tidx_math);

    Tensor acc_o = partition_fragment_C(tiled_mma_g2, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_o);

    // s2r copy atoms.
    auto s2r_copy_A   = make_tiled_copy_A(typename BS1::SmemCopyAtomA{}, tiled_mma_g1);
    auto s2r_thr_A    = s2r_copy_A.get_thread_slice(tidx_math);
    auto s2r_copy_B   = make_tiled_copy_B(typename BS1::SmemCopyAtomB{}, tiled_mma_g1);
    auto s2r_thr_B    = s2r_copy_B.get_thread_slice(tidx_math);
    auto s2r_copy_A2  = make_tiled_copy_A(typename BS2::SmemCopyAtomA{}, tiled_mma_g2);
    auto s2r_thr_A2   = s2r_copy_A2.get_thread_slice(tidx_math);
    auto s2r_copy_B2  = make_tiled_copy_B(typename BS2::SmemCopyAtomB{}, tiled_mma_g2);
    auto s2r_thr_B2   = s2r_copy_B2.get_thread_slice(tidx_math);
    // LDSM_T tiled copy: construction succeeds; cute::copy is bypassed (static_assert in copy_unpack
    // trips on MN_SW128 source). We use partition_S only to compute per-thread SMEM addresses, then
    // issue the PTX and byte-reorder manually — matching SM100_U8x16_LDSM_T::copy() exactly.
    auto s2r_copy_Vt  = make_tiled_copy_B(Copy_Atom<SM100_U8x16_LDSM_T, FP8Elem>{}, tiled_mma_g2);
    auto s2r_thr_Vt   = s2r_copy_Vt.get_thread_slice(tidx_math);
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

    __syncthreads();  // S1: barriers initialized by load warp; math warps can now use them.

    Tensor sQ_persist = make_tensor(
        make_smem_ptr(reinterpret_cast<FP8Elem*>(
            reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsQPersistOffset)),
        SmemLayoutQ_SW128{});
    auto sQ_persist_pi = as_position_independent_swizzle_tensor(sQ_persist);

    __syncthreads();  // S2: Q+SFA TMA complete (load warp waited); Q visible at sK_base[0].

    // Copy Q out of the K/V double-buffer slot before load warp overwrites it with K TMA.
    // IMPORTANT: write via sQ_persist_pi (position-independent swizzle) to match the
    // physical addresses that LDSM will read via sQ_persist_pi in the main loop.
    // sQ_persist base (67712 for Is_arbitrary) is NOT aligned to the SW128 swizzle period
    // (2048B), so sQ_persist(r,c) and sQ_persist_pi(r,c) map to different physical locations.
    // Using sQ_persist (non-PI) here while LDSM reads sQ_persist_pi causes element misplacement
    // for Is_arbitrary but not Is_causal (65536 % 2048 == 0, so PI == non-PI for Is_causal).
    {
      for (int i = tidx_math; i < kBlockM * kHeadDim; i += kNMathThreads) {
        const int r = i / kHeadDim;
        const int c = i % kHeadDim;
        sQ_persist_pi(r, c) = sQ_sw128(r, c);
      }
    }
    // Math-only rendezvous; do NOT use __syncthreads() here (would desync load path's S3/S5).
    asm volatile("bar.sync 1, 256;\n" : : : "memory");

    // Mask lambda (captures variables from math warp scope).
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

    __syncthreads();  // S3: before main loop; load warp ready to issue K/Vt TMA preamble.

    // SFP (unit scale for P) — separate buffer so real SFA SMEM stays valid for per-tile GEMM1 s2r.
    for (int i = tidx_math; i < kBlockM; i += kNMathThreads)
      smem_sfp_ptr[i] = 0x7f7f7f7f;

    // ===== MATH WARP DOUBLE-BUFFER QMMA CONSUMER LOOP =====
    // Phase 11: s2r V^T uses ldmatrix.m16n16.x2.trans.b8 (LDSM_T) directly from non-swizzled
    // D-major SMEM (SmemLayoutVt_TMA = Layout<[kHeadDim,kBlockN],[1,kHeadDim]>).
    // Source addressing: CuTe partition_S on s2r_thr_Vt gives 16B-aligned row pointers because
    // every row start = n*kHeadDim + d_base with kHeadDim=128 (128-aligned) and d_base a multiple of 16.
    static_assert(kHeadDim % 32 == 0 && kBlockN % 16 == 0,
        "LDSM_T requires kHeadDim divisible by 32 and kBlockN divisible by 16.");
    int tma_wait_parity[2] = {1, 0};
    int math_stage = 0;

    for (int n_valid = n_block_max - 1, masking_step = 0; n_valid >= n_block_min;
         ++masking_step, --n_valid) {
      const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;
      const bool is_masking = masking_step < n_masking_steps ||
          (nb + 1) * kBlockN > actual_seqlen_h;

      // Wait for TMA K[nb]+Vt[nb]+SFB[nb]+SFV[nb] to land.
      {
        uint32_t taddr = static_cast<uint32_t>(__cvta_generic_to_shared(tma_mbar_ptr[math_stage]));
        uint32_t done = 0;
        do {
          asm volatile("{.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;}\n"
                       : "=r"(done) : "r"(taddr), "r"((uint32_t)tma_wait_parity[math_stage]));
        } while (!done);
      }
      tma_wait_parity[math_stage] ^= 1;
      asm volatile("bar.sync 1, 256;\n" : : : "memory");

      Tensor sSFB_ = make_tensor(make_smem_ptr(smem_sfb_ptr[math_stage]), SmemLayoutSFB{});
      auto sSFB = as_position_independent_swizzle_tensor(sSFB_);

      FP8Elem* sK_cur  = sK_base[math_stage];
      FP8Elem* sVt_cur = sVt_base[math_stage];

      // GEMM1: acc_s += Q × K^T (block-scaled).
      // acc_s declared outside the K/SFB scope so it outlives tCrK/tCrSFB.
      Tensor acc_s = partition_fragment_C(tiled_mma_g1, Shape<Int<kBlockM>, Int<kBlockN>>{});
      clear(acc_s);
      { // Per-tile s2r Q+SFA (short lifetime) + K/SFB + GEMM1.
      Tensor tCrQ = thr_mma_g1.partition_fragment_A(sQ_persist_pi);
      {
        auto tXsQ = s2r_thr_A.partition_S(sQ_persist_pi);
        auto tXrQ = s2r_thr_A.retile_D(tCrQ);
        cute::copy(s2r_copy_A, tXsQ, tXrQ);
      }
      Tensor tCrSFA = BS1::partition_fragment_SFA(sSFA(_,_,_0{}), thr_mma_g1);
      {
        auto tXsSFA = s2r_thr_SFA.partition_S(sSFA);
        auto tXrSFA = s2r_thr_SFA.retile_D(tCrSFA);
        cute::copy(s2r_copy_SFA, tXsSFA(_,_,_,_0{}), tXrSFA);
      }
      auto tCrSFA_frg = BS1::transform_fragment_for_qmma(tCrSFA);

      { // tCrK and tCrSFB scoped: freed before silu / P path.
      auto sK_cur_pi = as_position_independent_swizzle_tensor(
          make_tensor(make_smem_ptr(sK_cur), SmemLayoutK_SW128{}));
      Tensor tCrK = thr_mma_g1.partition_fragment_B(sK_cur_pi);
      {
        auto tXsK = s2r_thr_B.partition_S(sK_cur_pi);
        auto tXrK = s2r_thr_B.retile_D(tCrK);
        cute::copy(s2r_copy_B, tXsK, tXrK);
      }

      Tensor tCrSFB = BS1::partition_fragment_SFB(sSFB(_,_,_0{}), thr_mma_g1);
      {
        auto tXsSFB = s2r_thr_SFB.partition_S(sSFB);
        auto tXrSFB = s2r_thr_SFB.retile_D(tCrSFB);
        cute::copy(s2r_copy_SFB, tXsSFB(_,_,_,_0{}), tXrSFB);
      }
      auto tCrSFB_frg = BS1::transform_fragment_for_qmma(tCrSFB);

      cute::gemm(tiled_mma_g1,
          make_zip_tensor(tCrQ, tCrSFA_frg(_,_,_,_0{})),
          make_zip_tensor(tCrK, tCrSFB_frg(_,_,_,_0{})),
          acc_s);
      } // tCrK, tCrSFB
      } // tCrQ, tCrSFA

      if (params.debug_gemm1_only) {
        for (int i = 0; i < size(acc_s); ++i) acc_o(i) += acc_s(i);
        if ((tidx_math & 31) == 0) {
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

      // ── P pre-conversion: F32 → packed FP8 ───────────────────────────────
      // Problem: during the P-write loop below, both acc_s (64 F32 = 64 regs)
      // and acc_o (64 F32 = 64 regs) must be live, and the SW128 swizzle address
      // LOP3 computation needs ~20 temp regs.  These overlap with acc_o's lower
      // register range (R4–R23), causing the compiler to spill acc_o (10×STL.64 +
      // 10×LDL.LU.64 = 20 local-memory accesses per loop iteration).
      //
      // Fix: pre-convert acc_s (64 F32) → acc_s_packed (16 uint32, 4 FP8/reg)
      // BEFORE the bar.sync + P-write loop.  After this block acc_s is dead
      // (last read is here), so its 64 registers are freed and available to the
      // LOP3 address computation, eliminating the acc_o spill.
      //
      // kAccSElems = 64 for kBlockM=kBlockN=128 / kNMathThreads=256.
      // Packing order: flat index f = r*kPWriteC + c (same as P-write (r,c) order),
      // acc_s_packed[f/4] bits [8*(f%4)+7 : 8*(f%4)] = FP8(acc_s_v(r,c)).
      constexpr int kAccSElems = kBlockM * kBlockN / kNMathThreads;  // = 64
      static_assert(kAccSElems % 4 == 0, "kAccSElems must be a multiple of 4");
      uint32_t acc_s_packed[kAccSElems / 4];  // 16 registers (vs. 64 for F32)
      {
        cutlass::NumericConverter<FP8Elem, float> fp32_to_fp8;
        auto acc_s_v_pre = make_tensor(acc_s.data(),
            group<1,3>(group<0,2>(select<1,2,0,3>(flatten(acc_s.layout())))));
        // size<1>(acc_s_v_pre) is a compile-time Int<C>; with CUTE_UNROLL all
        // loop variables are compile-time so flat/4 and flat%4 fold to constants.
        constexpr int kC = decltype(size<1>(acc_s_v_pre))::value;
        static_assert(kAccSElems % kC == 0, "kAccSElems must be divisible by kC");
        CUTE_UNROLL
        for (int flat = 0; flat < kAccSElems; flat += 4) {
          auto get_fp8 = [&](int f) -> uint8_t {
            FP8Elem v = fp32_to_fp8(static_cast<float>(acc_s_v_pre(f / kC, f % kC)));
            return *reinterpret_cast<const uint8_t*>(&v);
          };
          acc_s_packed[flat / 4] =
              (uint32_t)get_fp8(flat)            |
              ((uint32_t)get_fp8(flat + 1) <<  8) |
              ((uint32_t)get_fp8(flat + 2) << 16) |
              ((uint32_t)get_fp8(flat + 3) << 24);
        }
      }
      // acc_s (64 F32 regs) is now DEAD — freed for LOP3 in P-write below.

      // Convert/store P: write packed FP8 (from acc_s_packed) into sPbuf via CuTe API.
      // acc_s (64 F32 regs) was already converted → acc_s_packed (16 uint32) above,
      // freeing ~48 registers before this write.  We use the CuTe position-independent
      // API (guaranteed correct address) instead of a manual swizzle formula.
      asm volatile("bar.sync 1, 256;\n" : : : "memory");
      Tensor sPbuf    = make_tensor(make_smem_ptr(sK_cur), SmemLayoutQ_SW128{});
      auto sPbuf_pi   = as_position_independent_swizzle_tensor(sPbuf);
      {
        Tensor cP_id    = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tPcP_raw = thr_mma_g1.partition_C(cP_id);
        Tensor tPcP_v   = make_tensor(tPcP_raw.data(),
            group<1,3>(group<0,2>(select<1,2,0,3>(flatten(tPcP_raw.layout())))));
        constexpr int kC2 = decltype(size<1>(tPcP_v))::value;
        constexpr int kR = kAccSElems / kC2;
        CUTE_UNROLL
        for (int r = 0; r < kR; ++r) {
          CUTE_UNROLL
          for (int c = 0; c < kC2; ++c) {
            int flat = r * kC2 + c;  // compile-time with CUTE_UNROLL
            uint8_t fp8_byte = (acc_s_packed[flat / 4] >> ((flat % 4) * 8)) & 0xFF;
            FP8Elem fp8_val;
            *reinterpret_cast<uint8_t*>(&fp8_val) = fp8_byte;
            sPbuf_pi(get<0>(tPcP_v(r, c)), get<1>(tPcP_v(r, c))) = fp8_val;
          }
        }
      }
      asm volatile("bar.sync 1, 256;\n" : : : "memory");

      // s2r P (GEMM2 A-operand).
      Tensor tCrP = thr_mma_g2.partition_fragment_A(sPbuf_pi);
      {
        auto tXsP = s2r_thr_A2.partition_S(sPbuf_pi);
        auto tXrP = s2r_thr_A2.retile_D(tCrP);
        cute::copy(s2r_copy_A2, tXsP, tXrP);
      }

      { // tCrV, tCrSFV, tCrSFP scoped here: compiler can reuse registers freed by tCrK/tCrSFB.
      // s2r V^T — Phase 11 LDSM_T from MN_SW128-swizzled SMEM.
      //
      // sVt_cur layout: SmemLayoutVt_TMA = tile_to_shape(MN_SW128_Atom<Element>, [kHeadDim, kBlockN]).
      //   Swizzle<3,4,3>: physical_byte(d, n_k) = n_k * kHeadDim + (d ^ ((n_k & 7) << 4)).
      //   TMA writes swizzled bytes automatically via the descriptor's swizzle mode.
      //
      // BS2 TiledMMA: AtomLayout<_2,_4,_1>, PermMmaTileN=Layout<_8,_4,_4, Stride<_1,_32,_8>>.
      // → 4 N-warps (ThrN=0..3), each covering 32 N-columns (d-direction): d ∈ [ThrN*32, ThrN*32+32).
      // n_warp = (tidx_math / 32) % 4  (= ThrN = warp_id % 4 from Stride<_4,_1,_0> AtomLayout).
      //
      // Fragment tXrV = recast<uint32_t>(tCrV): shape (2, 4, 4) = (reg, N_atom_in_warp, K_step).
      //   linear_idx = reg + 2*N_atom + 8*K_step.  Total = 32 uint32 per thread.
      //
      // ldmatrix.m16n16.x2.trans.b8 at iteration ni (ni=0..7):
      //   Covers n_k ∈ [ni*16, ni*16+16) and d ∈ [n_warp*32, n_warp*32+32).
      //   Thread lane provides row address for M0 (lanes 0..15, d_mat=0) or M1 (lanes 16..31, d_mat=16).
      //   n_k_row = ni*16 + n_off; d_start = n_warp*32 + d_mat.
      //   swizzle_xor = (n_k_row & 7) << 4 = (n_off & 7) << 4  (ni*16 contributes 0 mod 8).
      //   Physical address = vt_base + n_k_row * kHeadDim + (d_start ^ swizzle_xor).
      //   16B alignment: d_start ∈ {0,16,32,...,112}, swizzle_xor ∈ {0,16,...,112}
      //   → (d_start ^ swizzle_xor) is always a multiple of 16. ✓
      //
      //   After .trans, thread t gets (same logical element mapping as non-swizzled):
      //     r0: M0[4*(t&3)..+3][t>>2]   → d = n_warp*32      + (t>>2), K = ni*16+{4*(t&3),..,+3}
      //     r1: M0[4*(t&3)..+3][t>>2+8] → d = n_warp*32 +  8 + (t>>2), same K
      //     r2: M1[4*(t&3)..+3][t>>2]   → d = n_warp*32 + 16 + (t>>2), same K
      //     r3: M1[4*(t&3)..+3][t>>2+8] → d = n_warp*32 + 24 + (t>>2), same K
      //   → r0..r3 fill N_atoms 0..3 of this warp (8 d-values each).
      //   K mapping: K_step = ni>>1, reg = ni&1 → base = (ni&1) + 8*(ni>>1).
      auto sVt_ns = make_tensor(make_smem_ptr(sVt_cur), SmemLayoutVt_SW128{});
      Tensor tCrV = thr_mma_g2.partition_fragment_B(sVt_ns);
      {
        const uint32_t vt_base = static_cast<uint32_t>(__cvta_generic_to_shared(sVt_cur));
        const int lane   = tidx_math & 31;
        const int n_off  = lane & 15;          // row index within 16-row group (for M0 or M1)
        const int d_mat  = (lane >> 4) << 4;  // d-offset: 0 for M0 (lanes 0..15), 16 for M1 (16..31)
        // n_warp: this warp's N-tile index (0..3). AtomLayout Stride<_4,_1,_0> → ThrN = warp_id % 4.
        const int n_warp = (tidx_math / 32) % 4;

        auto tXrV = recast<uint32_t>(tCrV);
        constexpr int kNTiles = kBlockN / 16;    // 8 (K-axis: 16 n_k values per LDSM_T call)
        CUTE_UNROLL
        for (int ni = 0; ni < kNTiles; ++ni) {
          // MN_SW128 swizzle: physical_byte(d, n_k) = n_k * kHeadDim + (d ^ ((n_k & 7) << 4)).
          // n_k_row & 7 == n_off & 7 (since ni * 16 contributes 0 mod 8).
          const int n_k_row     = ni * 16 + n_off;
          const int d_start     = n_warp * 32 + d_mat;
          const int swizzle_xor = (n_off & 7) << 4;
          uint32_t addr = vt_base + (uint32_t)(n_k_row * kHeadDim + (d_start ^ swizzle_xor));
          uint32_t r0, r1, r2, r3;
          asm volatile(
              "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];\n"
              : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
              : "r"(addr));
          // K_step = ni>>1, reg = ni&1 → base = (ni&1) + 8*(ni>>1)
          const int base = (ni & 1) + (8 * (ni >> 1));
          tXrV(base + 0) = r0;   // N_atom=0: d ∈ [n_warp*32,    n_warp*32+8)
          tXrV(base + 2) = r1;   // N_atom=1: d ∈ [n_warp*32+8,  n_warp*32+16)
          tXrV(base + 4) = r2;   // N_atom=2: d ∈ [n_warp*32+16, n_warp*32+24)
          tXrV(base + 6) = r3;   // N_atom=3: d ∈ [n_warp*32+24, n_warp*32+32)
        }
      }

      // s2r SFP (unit) and SFV.
      Tensor tCrSFP = BS2::partition_fragment_SFA(sSFP(_,_,_0{}), thr_mma_g2);
      {
        auto tXsSFP = s2r_thr_SFP.partition_S(sSFP);
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

      // Signal load warp: sK[math_stage] and sVt[math_stage] consumed; load warp may overwrite.
      if ((tidx_math & 31) == 0) {
        uint32_t maddr = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr[math_stage]));
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" : : "r"(maddr));
      }

      // GEMM2: acc_o += P × V^T (block-scaled) — overlaps with load warp's next TMA.
      cute::gemm(tiled_mma_g2,
          make_zip_tensor(tCrP, tCrSFP_frg(_,_,_,_0{})),
          make_zip_tensor(tCrV, tCrSFV_frg(_,_,_,_0{})),
          acc_o);
      } // tCrV, tCrSFV, tCrSFP freed here.

      if (is_jump && masking_step == n_masking_steps - 1)
        n_valid = std::min(n_valid, n_block_history);

      math_stage ^= 1;
    }

    // ===== EPILOGUE =====
    for (int i = 0; i < size(acc_o); ++i) acc_o(i) /= params.scaling_seqlen;
    using OutElement = typename Kernel_traits::OutputType;
    Tensor rO = make_tensor_like<OutElement>(acc_o);
    flash::convert_type_safe(acc_o, rO);

    // bar.sync among 256 math warps: all must exit loop before any warp overwrites smem_q.
    asm volatile("bar.sync 1, 256;\n" : : : "memory");
    Tensor sO_flat = make_tensor(
        make_smem_ptr(reinterpret_cast<OutElement*>(smem_q)),
        Layout<Shape<Int<kBlockM>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>{});
    {
      Tensor cO_id    = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
      Tensor tOcO_raw = thr_mma_g2.partition_C(cO_id);
      Tensor tOcO_v   = make_tensor(tOcO_raw.data(),
          group<1,3>(group<0,2>(select<1,2,0,3>(flatten(tOcO_raw.layout())))));
      Tensor rO_v     = make_tensor(rO.data(),
          group<1,3>(group<0,2>(select<1,2,0,3>(flatten(rO.layout())))));
      CUTE_UNROLL
      for (int r = 0; r < size<0>(rO_v); ++r) {
        CUTE_UNROLL
        for (int c = 0; c < size<1>(rO_v); ++c) {
          sO_flat(int(get<0>(tOcO_v(r,c))), int(get<1>(tOcO_v(r,c)))) = rO_v(r,c);
        }
      }
    }

    __syncthreads();  // S5: sO visible to all; load warp arrives here after its loop ends.

    // Copy sO to GMEM.
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
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1)
hstu_fwd_kernel_sm120_fp8_ws_tma(
    __grid_constant__ Hstu_fwd_params_fp8_ws_tma<TMA_Q_t, TMA_K_t, TMA_Vt_t,
                                                  TMA_SFA_t, TMA_SFB_t, TMA_SFV_t> const params) {
  int m_block = gridDim.x - blockIdx.x - 1;
  int bidh    = blockIdx.y;
  int bidb    = blockIdx.z;
  hstu_compute_attn_1rowblock_sm120_fp8_ws<Kernel_traits>(params, bidb, bidh, m_block);
}
