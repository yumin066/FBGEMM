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
//   S1    : after load warp inits tma_mbar/math_mbar + fence; q_tma_mbar is now math-warp-local
//   S2→   : replaced by wait_mbar_parity(q_tma_mbar_ptr,0) for all 256 math threads (no CTA sync)
//   S3    : removed (was a no-op rendezvous with no real data dependency)
//   S5    : replaced by bar.sync 1,256 (math-warp-only); load warps exit after main loop

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

// Rearrange GEMM1 C-fragment (acc_s_packed[16]) to GEMM2 A-fragment (tCrP[16]) layout
// using warp shuffle only — no SMEM staging, no bar.sync required.
//
// With AtomLayout <_8,_1,_1> (1 N-warp), each warp holds all 128 N-values of P in its
// own registers after GEMM1. No cross-warp communication is needed: the entire D→A
// rearrangement is intra-warp, performed by SHFL.IDX within each 4-thread quad.
//
// QMMA.SF.16832 coordinate mappings (SPA ISA):
//   C-fragment: acc_s_packed[nr] = N-atom nr, bytes {mr=0,i=0},{mr=0,i=1},{mr=1,i=0},{mr=1,i=1}
//     where N_base(nr) = (nr%4)*32 + (nr/4)*8  (from PermMmaTileN strides (1,32,8))
//     and   col = N_base(nr) + (lane&3)*2 + i
//   A-fragment: tCrP[4*kb + 2*c + mr] at row=8*mr+(lane>>2), col=32*kb+16*c+(lane&3)*4+{0..3}
//
// For each (kb,c), the 4 K-values needed by one thread span two consecutive C-fragment
// atoms: atom_lo covers col[0..1] (from src_lane0) and atom_hi covers col[2..3] (from src_lane1).
//   src_lane0 = (lane & ~3u) | ((lane&1u) << 1)  — lower pair of the quad
//   src_lane1 = src_lane0 + 1                     — upper pair of the quad
// kAtomLut[kb][c][h]: C-fragment atom index for K-half h (h = (lane&3)>>1).
//   kAtomLut[kb][c][h] = kb + 4*(2*c + h)
// Byte assembly:
//   mr=0 (row=quad):   __byte_perm(a, b, 0x5410) = {a.b0,a.b1,b.b0,b.b1}
//   mr=1 (row=8+quad): __byte_perm(a, b, 0x7632) = {a.b2,a.b3,b.b2,b.b3}
//
// Cost: 4 kb × 2 c × 4 SHFL + 2 __byte_perm = 32 SHFL + 16 BYTE_PERM per thread.
// Savings: eliminates 2 × bar.sync (256-thread) + 16KB SMEM write + LDSM load-back.
__device__ __forceinline__ void permute_acc_s_packed_to_tCrP(
    uint32_t* __restrict__ tCrP,
    const uint32_t* __restrict__ acc_s_packed) {
  const unsigned lane      = threadIdx.x & 31u;
  const unsigned tiq       = lane & 3u;
  const unsigned quad_base = lane & ~3u;
  const unsigned src_lane0 = quad_base | ((tiq & 1u) << 1u);
  const unsigned src_lane1 = src_lane0 + 1u;

  constexpr uint8_t kLut0[4][2] = {{ 0, 8}, { 1, 9}, { 2, 10}, { 3, 11}};
  constexpr uint8_t kLut1[4][2] = {{ 4,12}, { 5,13}, { 6, 14}, { 7, 15}};

  CUTE_UNROLL
  for (int kb = 0; kb < 4; ++kb) {
    CUTE_UNROLL
    for (int c = 0; c < 2; ++c) {
      // Shuffle BOTH halves first, then select: src_lane's tiq determines which
      // N-atom register IT holds, which may differ from the destination's h-half.
      const uint32_t pk0 = acc_s_packed[kLut0[kb][c]];
      const uint32_t pk1 = acc_s_packed[kLut1[kb][c]];
      const uint32_t a0  = __shfl_sync(0xFFFFFFFFu, pk0, src_lane0);
      const uint32_t b0  = __shfl_sync(0xFFFFFFFFu, pk0, src_lane1);
      const uint32_t a1  = __shfl_sync(0xFFFFFFFFu, pk1, src_lane0);
      const uint32_t b1  = __shfl_sync(0xFFFFFFFFu, pk1, src_lane1);
      const uint32_t a   = (tiq >> 1u) ? a1 : a0;
      const uint32_t b   = (tiq >> 1u) ? b1 : b0;
      tCrP[4 * kb + 2 * c + 0] = __byte_perm(a, b, 0x5410u);
      tCrP[4 * kb + 2 * c + 1] = __byte_perm(a, b, 0x7632u);
    }
  }
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

  using BS1 = hstu::SM120QmmaBuilder<
      Kernel_traits::kBlockM, Kernel_traits::kBlockN, 4>;
  using BS2 = hstu::SM120QmmaBuilder<
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
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(56));
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

    // Early exit 2 (after Is_arbitrary): both branches exit with 1 sync ahead (S1).
    if (((Is_causal || Is_local || Is_arbitrary) && n_block_max <= n_block_min) ||
        m_block * kBlockM >= actual_seqlen_q) {
      __syncthreads();  // S1
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
    uint64_t* tma_mbar_ptr0  = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset);
    uint64_t* tma_mbar_ptr1  = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 8);
    uint64_t* math_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 16);
    uint64_t* math_mbar_ptr1 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 24);
    // q_tma_mbar_ptr is now initialized by math warp thread 0, not here.

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
    // Initializes tma_mbar0/1 and math_mbar0/1 only; q_tma_mbar is initialized by math warp thread 0.
    if (tidx == kNMathThreads) {
      uint32_t tm0 = static_cast<uint32_t>(__cvta_generic_to_shared(tma_mbar_ptr0));
      uint32_t tm1 = static_cast<uint32_t>(__cvta_generic_to_shared(tma_mbar_ptr1));
      uint32_t mm0 = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr0));
      uint32_t mm1 = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr1));
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
    __syncthreads();  // S1: tma_mbar0/1 and math_mbar0/1 initialized and visible to all warps.

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

    auto mVt_tma   = params.tma_vt.get_tma_tensor(make_shape(params.total_k, params.d, params.h_k));
    auto gVt_head  = mVt_tma(_, _, bidh_kv);
    auto gVt_tiles = local_tile(gVt_head, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _));
    auto tma_slice_Vt  = params.tma_vt.get_slice(0);
    auto tVtgVt_tma    = tma_slice_Vt.partition_S(gVt_tiles(_, _, _, Int<0>{}));
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

    // ===== LOAD WARP DOUBLE-BUFFER TMA PRODUCER =====
    // Only active load warp (warp 8) participates.
    // Idle load warps (9-11) skip this entire section and go directly to S5.
    if (is_active_load) {
      int math_wait_parity0 = 0;
      int math_wait_parity1 = 0;

      // Preamble: wait math_mbar[0] (pre-satisfied), then issue tile n_block_max-1 into stage 0.
      {
        uint32_t maddr = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr0));
        uint32_t done = 0;
        do {
          asm volatile("{.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;}\n"
                       : "=r"(done) : "r"(maddr), "r"((uint32_t)math_wait_parity0));
        } while (!done);
      }
      math_wait_parity0 ^= 1;

      if (tidx == kNMathThreads) {
        const int nb0     = Is_arbitrary ? int(sValidBlockIds[n_block_max - 1]) : (n_block_max - 1);
        const int nb_abs0 = binfo.sum_s_k / kBlockN + nb0;
        uint32_t taddr0   = static_cast<uint32_t>(__cvta_generic_to_shared(tma_mbar_ptr0));
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                     : : "r"(taddr0), "r"(kSmemKVtSFBBytes));
        cute::copy(params.tma_k.with(*tma_mbar_ptr0),   tKgK_tma(_, _, _, nb_abs0),    tKsK_d_0);
        cute::copy(params.tma_vt.with(*tma_mbar_ptr0),  tVtgVt_tma(_, _, _, nb_abs0),  tVtsVt_d_0);
        cute::copy(params.tma_sfb.with(*tma_mbar_ptr0), tSFBgSFB_tma(_, _, _, nb_abs0), tSFBsSFB_d_0);
        cute::copy(params.tma_sfv.with(*tma_mbar_ptr0), tSFVgSFV_tma(_, _, _, nb_abs0), tSFVsSFV_d_0);
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
          const int cur_math_parity = load_stage ? math_wait_parity1 : math_wait_parity0;
          uint32_t maddr = static_cast<uint32_t>(__cvta_generic_to_shared(math_mbar_ptr0)) + (uint32_t)(load_stage * 8);
          uint32_t done = 0;
          do {
            asm volatile("{.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;}\n"
                         : "=r"(done) : "r"(maddr), "r"((uint32_t)cur_math_parity));
          } while (!done);
        }
        if (load_stage) { math_wait_parity1 ^= 1; } else { math_wait_parity0 ^= 1; }

        if (tidx == kNMathThreads) {
          const int nb_abs = binfo.sum_s_k / kBlockN + nb;
          uint32_t taddr   = static_cast<uint32_t>(__cvta_generic_to_shared(tma_mbar_ptr0)) + (uint32_t)(load_stage * 8);
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                       : : "r"(taddr), "r"(kSmemKVtSFBBytes));
          if (load_stage == 0) {
            cute::copy(params.tma_k.with(*tma_mbar_ptr0),   tKgK_tma(_, _, _, nb_abs),    tKsK_d_0);
            cute::copy(params.tma_vt.with(*tma_mbar_ptr0),  tVtgVt_tma(_, _, _, nb_abs),  tVtsVt_d_0);
            cute::copy(params.tma_sfb.with(*tma_mbar_ptr0), tSFBgSFB_tma(_, _, _, nb_abs), tSFBsSFB_d_0);
            cute::copy(params.tma_sfv.with(*tma_mbar_ptr0), tSFVgSFV_tma(_, _, _, nb_abs), tSFVsSFV_d_0);
          } else {
            cute::copy(params.tma_k.with(*tma_mbar_ptr1),   tKgK_tma(_, _, _, nb_abs),    tKsK_d_1);
            cute::copy(params.tma_vt.with(*tma_mbar_ptr1),  tVtgVt_tma(_, _, _, nb_abs),  tVtsVt_d_1);
            cute::copy(params.tma_sfb.with(*tma_mbar_ptr1), tSFBgSFB_tma(_, _, _, nb_abs), tSFBsSFB_d_1);
            cute::copy(params.tma_sfv.with(*tma_mbar_ptr1), tSFVgSFV_tma(_, _, _, nb_abs), tSFVsSFV_d_1);
          }
        }

        if (is_jump && masking_step_load == n_masking_steps - 1)
          n_valid = std::min(n_valid, n_block_history);

        load_stage ^= 1;
      }
    }  // end if (is_active_load)
    // Load warps exit here; epilogue is math-warp-only (bar.sync 1, 256).

  // ============================================================
  } else {
  // MATH WARP PATH  (warps 0-7, threads 0-255)
  // ============================================================
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" : : "n"(224));

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

    // Early exit 2 (after Is_arbitrary): math warps write zeros; both branches have 1 sync ahead (S1).
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
      return;
    }

    // SW128 SMEM layouts.
    using SmemLayoutQ_SW128  = decltype(tile_to_shape(typename BS1::SmemLayoutAtomA{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutK_SW128  = decltype(tile_to_shape(typename BS1::SmemLayoutAtomB{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutVt_SW128 = typename Kernel_traits::SmemLayoutVt_TMA;

    constexpr int kSmemKVElems = kBlockN * kHeadDim;

    // smem_base32: shared-memory base address as uint32 for on-demand barrier/KV
    // pointer computation in the main loop — replaces 6 persistent pointer arrays.
    static constexpr int kSmemMbar0Offset =
        Kernel_traits::kSmemSize - Kernel_traits::kSmemMbarSize;
    const uint32_t smem_base32 =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_));
    uint64_t* q_tma_mbar_ptr = reinterpret_cast<uint64_t*>(
        reinterpret_cast<char*>(smem_) + kSmemMbar0Offset + 32);

    // SF SMEM pointers.
    static constexpr int kSmemSFOffset_WS = Kernel_traits::kSmemWsDataSizePadded;
    int32_t* smem_sfa_ptr = reinterpret_cast<int32_t*>(smem_ + kSmemSFOffset_WS);
    int32_t* smem_sfp_ptr = smem_sfa_ptr + kBlockM;

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

    // s2r copy atoms — A operands (Q, P) loaded via load_a_z_pattern (inline PTX Z-pattern).
    // B operands (K) loaded via load_b_z_pattern (direct uint32 reads, bypassing ldmatrix.x4
    // which is incompatible with AtomLayout <_8,_1,_1> ThrN=1).
    // B operand (V^T) loaded via LDSM_T inline PTX.
    auto s2r_copy_SFA = make_tiled_copy_impl(typename BS1::SmemCopyAtomSF{},
        BS1::get_layoutSFA_TV(tiled_mma_g1), make_shape(size<0>(tile_shape(tiled_mma_g1)), _1{}));
    auto s2r_thr_SFA  = s2r_copy_SFA.get_thread_slice(tidx_math);
    // SFB/SFV: get_layoutSFB_TV degenerates under AtomLayout <_8,_1,_1> (ThrN=1 → all
    // threads map to SMEM position 0). Use direct SMEM reads at the call site instead.
    auto s2r_copy_SFP = make_tiled_copy_impl(typename BS2::SmemCopyAtomSF{},
        BS2::get_layoutSFA_TV(tiled_mma_g2), make_shape(size<0>(tile_shape(tiled_mma_g2)), _1{}));
    auto s2r_thr_SFP  = s2r_copy_SFP.get_thread_slice(tidx_math);

    // A-operand loader using ldmatrix.sync.aligned.m8n8.x4.shared.b16.
    //
    // The 4-matrix 2×2 arrangement covers exactly 16M × 32K FP8 = one QMMA A-tile:
    //   mat 0: M[warp_m*16+0..7 ], K[kb*32+0..15 ]  → r0 → Ra+0
    //   mat 1: M[warp_m*16+8..15], K[kb*32+0..15 ]  → r1 → Ra+1
    //   mat 2: M[warp_m*16+0..7 ], K[kb*32+16..31]  → r2 → Ra+2
    //   mat 3: M[warp_m*16+8..15], K[kb*32+16..31]  → r3 → Ra+3
    // Thread (mat_num=lane>>3) provides addr for row (mat_row=lane&7) of its matrix.
    //
    // K_SW128 swizzle (Swizzle<3,4,3>): physical_k = K_start ^ ((M_abs & 7) << 4).
    // M_abs & 7 == mat_row since warp_m*16 and (mat_num&1)*8 are multiples of 8.
    auto load_a_z_pattern = [&](auto&& sA_pi, auto& tCrA, int k_block_base, int k_block_count) {
      const int lane    = tidx_math & 31;
      const int warp_m  = tidx_math / 32;
      const int mat_num = lane >> 3;    // matrix index (0..3) this lane provides addr for
      const int mat_row = lane & 7;     // row within that matrix (0..7)
      const int M_abs   = warp_m * 16 + ((mat_num & 1) << 3) + mat_row;
      const int K_half  = mat_num >> 1; // 0 = K low 16, 1 = K high 16
      const uint32_t smem_base =
          static_cast<uint32_t>(__cvta_generic_to_shared(&sA_pi(0, 0)));
      const uint32_t row_base = smem_base + static_cast<uint32_t>(M_abs * 128);
      auto tXrA = recast<uint32_t>(tCrA);
      CUTE_UNROLL
      for (int kb = 0; kb < k_block_count; ++kb) {
        const int K_start = (k_block_base + kb) * 32 + (K_half << 4);
        const uint32_t addr =
            row_base + static_cast<uint32_t>(K_start ^ (mat_row << 4));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
            : "=r"(tXrA(4*kb+0)), "=r"(tXrA(4*kb+1)),
              "=r"(tXrA(4*kb+2)), "=r"(tXrA(4*kb+3))
            : "r"(addr));
      }
    };

    // B-operand loader using ldmatrix.sync.aligned.m8n8.x4.shared.b16.
    //
    // PermMmaTileN = Layout<Shape<_8,_4,_4>, Stride<_1,_32,_8>>:
    //   N_base[nr] = (nr%4)*32 + (nr/4)*8.  Group by g = nr%4:
    //   g=0: nr={0,4,8,12}  → N=[0,8,16,24]    (32 consecutive SMEM rows)
    //   g=1: nr={1,5,9,13}  → N=[32,40,48,56]
    //   g=2: nr={2,6,10,14} → N=[64,72,80,88]
    //   g=3: nr={3,7,11,15} → N=[96,104,112,120]
    // One x4 call covers 4 N-atoms (32 rows) × 16 K = one (g, K_half) slice.
    //   frag_base = 32*kb + 2*g
    //   r0→frag_base+K_half+0, r1→+8, r2→+16, r3→+24
    // K_SW128 swizzle: physical_k = K_start ^ ((N_addr&7)<<4) = K_start ^ (mat_row<<4).
    auto load_b_z_pattern = [&](auto&& sB_pi, auto& tCrB, int k_block_base, int k_block_count) {
      const int lane    = tidx_math & 31;
      const int mat_num = lane >> 3;
      const int mat_row = lane & 7;
      const uint32_t smem_base =
          static_cast<uint32_t>(__cvta_generic_to_shared(&sB_pi(0, 0)));
      auto tXrB = recast<uint32_t>(tCrB);
      constexpr int kNGroups = kBlockN / 32;  // 4
      CUTE_UNROLL
      for (int kb = 0; kb < k_block_count; ++kb) {
        CUTE_UNROLL
        for (int g = 0; g < kNGroups; ++g) {
          const uint32_t N_row   = static_cast<uint32_t>(g * 32 + mat_num * 8 + mat_row);
          const int frag_base    = 32 * kb + 2 * g;
          CUTE_UNROLL
          for (int K_half = 0; K_half < 2; ++K_half) {
            const int K_start  = (k_block_base + kb) * 32 + K_half * 16;
            const uint32_t addr =
                smem_base + N_row * 128 + static_cast<uint32_t>(K_start ^ (mat_row << 4));
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                : "=r"(tXrB(frag_base + K_half +  0)),
                  "=r"(tXrB(frag_base + K_half +  8)),
                  "=r"(tXrB(frag_base + K_half + 16)),
                  "=r"(tXrB(frag_base + K_half + 24))
                : "r"(addr));
          }
        }
      }
    };

    __syncthreads();  // S1: tma_mbar0/1 and math_mbar0/1 visible to all warps; math warps can use them.

    Tensor sQ_persist = make_tensor(
        make_smem_ptr(reinterpret_cast<FP8Elem*>(
            reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsQPersistOffset)),
        SmemLayoutQ_SW128{});
    auto sQ_persist_pi = as_position_independent_swizzle_tensor(sQ_persist);

    // Q+SFA TMA preamble: math warp thread 0 initializes q_tma_mbar and issues TMA directly into
    // sQ_persist and smem_sfa_ptr.  q_tma_mbar is math-warp-local; no cross-warp sync needed.
    // kSmemWsQPersistOffset is 2048B-aligned, so TMA's absolute SW128 write addresses and
    // PI-swizzled LDSM read addresses are identical for all mask patterns (PI == non-PI).
    {
      constexpr uint32_t kSmemQBytes   = kBlockM * kHeadDim * (uint32_t)sizeof(FP8Elem);
      constexpr uint32_t kSmemSFABytes = kBlockM * (uint32_t)sizeof(int32_t);
      using SmemLayoutSFA_TMA_t = cute::Layout<cute::Shape<cute::Int<kBlockM>, cute::Int<1>>,
                                               cute::Stride<cute::_1, cute::Int<kBlockM>>>;
      const int m_abs = binfo.sum_s_q / kBlockM + m_block;
      if (tidx_math == 0) {
        auto mQ_tma   = params.tma_q.get_tma_tensor(make_shape(params.total_q, params.d, params.h));
        auto gQ_head  = mQ_tma(_, _, bidh);
        auto gQ_tiles = local_tile(gQ_head, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(_, _));
        auto tma_slice_Q = params.tma_q.get_slice(0);
        auto sQ_buf      = make_tensor(make_smem_ptr(sQ_persist.data()), SmemLayoutQ_SW128{});
        auto tQsQ_d      = tma_slice_Q.partition_D(sQ_buf);
        auto tQgQ_tma    = tma_slice_Q.partition_S(gQ_tiles(_, _, _, Int<0>{}));
        auto mSFA_tma    = params.tma_sfa.get_tma_tensor(
            make_shape((int64_t)params.q_block_descale_head_stride, cute::Int<1>{}, params.h));
        auto gSFA_head   = mSFA_tma(_, _, bidh);
        auto gSFA_tiles  = local_tile(gSFA_head, Shape<Int<kBlockM>, Int<1>>{}, make_coord(_, _));
        auto tma_slice_SFA = params.tma_sfa.get_slice(0);
        auto tSFAsSFA_d    = tma_slice_SFA.partition_D(
            make_tensor(make_smem_ptr(smem_sfa_ptr), SmemLayoutSFA_TMA_t{}));
        auto tSFAgSFA_tma  = tma_slice_SFA.partition_S(gSFA_tiles(_, _, _, Int<0>{}));
        uint32_t qaddr = static_cast<uint32_t>(__cvta_generic_to_shared(q_tma_mbar_ptr));
        // Init q_tma_mbar here (math-warp-local); fence ensures TMA proxy sees the write.
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(qaddr), "r"(1));
        asm volatile("fence.proxy.async.shared::cta;\n" : : : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                     : : "r"(qaddr), "r"(kSmemQBytes + kSmemSFABytes));
        cute::copy(params.tma_q.with(*q_tma_mbar_ptr),   tQgQ_tma(_, _, _, m_abs),     tQsQ_d);
        cute::copy(params.tma_sfa.with(*q_tma_mbar_ptr), tSFAgSFA_tma(_, _, _, m_abs), tSFAsSFA_d);
      }
    }
    // bar.sync 2,256: make thread 0's mbarrier.init write visible to all 256 math threads
    // before they spin on wait_mbar_parity. Without this, threads 1-255 may see stale/garbage
    // barrier state (SMEM is not implicitly coherent across threads without a barrier).
    asm volatile("bar.sync 2, 256;\n" : : : "memory");
    // All 256 math threads wait for Q+SFA TMA completion. mbarrier.test_wait is non-destructive
    // and provides acquire semantics: SMEM data written by TMA is visible to all waiters.
    wait_mbar_parity(q_tma_mbar_ptr, 0);

    // Mask lambda (captures variables from math warp scope).
    auto col_limit_right = [&](int row) {
      return std::min(actual_seqlen_k, row + 1 + params.window_size_right);
    };
    auto col_limit_left = [&](int row) {
      return std::max(0, row - params.window_size_left);
    };
    auto apply_mask_bs = [&](auto& tSrS, int nb) {
      static constexpr int Row = 0, Col = 1;
      Tensor cS   = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
      Tensor tScS = thr_mma_g1.partition_C(cS);
      const int base_row = m_block * kBlockM + actual_seqlen_offset;
      const int base_col = nb * kBlockN;
      // Under AtomLayout <_8,_1,_1>, select<1,2,0,3> is invalid.
      // Use direct flat indexing: tScS(flat) and tSrS(flat) share the same
      // element ordering produced by partition_C, giving correct (row,col) coords.
      Tensor col_min = make_tensor<int>(make_shape(size<0>(gMinFunc)));
      Tensor col_max = make_tensor<int>(make_shape(size<0>(gMaxFunc)));
      int prev_block_row    = -1;
      int row               = 0;
      [[maybe_unused]] int tgt_col_lft = 0;
#pragma unroll
      for (int flat = 0; flat < size(tSrS); ++flat) {
        const auto coord    = tScS(flat);
        const int block_row = int(get<Row>(coord));
        // Lazily recompute per-row values when block_row changes.
        // Each thread has exactly 2 distinct block_rows (lane>>2 and lane>>2+8),
        // alternating every 2 elements in the MMA D-register flat ordering.
        if (block_row != prev_block_row) {
          row           = block_row + base_row;
          prev_block_row = block_row;
          if constexpr (Is_target) {
            const int tgt_idx = (row - actual_seqlen_h) / params.target_group_size;
            tgt_col_lft = actual_seqlen_h + tgt_idx * params.target_group_size;
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
            if (row >= actual_seqlen_h && col >= actual_seqlen_h && col < tgt_col_lft)
              tSrS(flat) = -INFINITY;
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
      }
    };

    // SFP (unit scale for P) — separate buffer so real SFA SMEM stays valid for per-tile GEMM1 s2r.
    for (int i = tidx_math; i < kBlockM; i += kNMathThreads)
      smem_sfp_ptr[i] = 0x7f7f7f7f;

    // ===== MATH WARP DOUBLE-BUFFER QMMA CONSUMER LOOP =====
    // Phase 11: s2r V^T uses ldmatrix.m16n16.x2.trans.b8 (LDSM_T) directly from MN_SW128 SMEM.
    // Under AtomLayout <_8,_1,_1>, all 8 warps load all 4 N-slabs (nw=0..3 outer loop);
    // 16B alignment: d_start = nw*32+d_mat ∈ multiples of 16, XOR swizzle_xor also multiple of 16.
    static_assert(kHeadDim % 32 == 0 && kBlockN % 16 == 0,
        "LDSM_T requires kHeadDim divisible by 32 and kBlockN divisible by 16.");
    // ── N-loop tile-invariants (Opt A): hoisted from per-tile load ───────────────────────
    // Q (sQ_persist) and SFA are written by TMA before the loop (guaranteed visible after
    // wait_mbar_parity at S2) and never change.  Hoisting eliminates 4 ldmatrix + 1 LDS
    // per N-tile.
    // NOTE: tCrSFP is NOT hoisted — smem_sfp_ptr is written by distributed thread writes
    // (not TMA) and requires the long per-tile gap (mbarrier wait + GEMM1, ~500 cycles)
    // to guarantee visibility.  Hoisting would introduce a race with no explicit barrier.
    Tensor tCrSFA = BS1::partition_fragment_SFA(sSFA(_,_,_0{}), thr_mma_g1);
    {
      const int warp_m  = tidx_math / 32;
      const int lane    = tidx_math & 31;
      const int sfa_row = warp_m * 16 + 8 * (lane & 1) + (lane >> 2);
      tCrSFA(0, 0, 0)  = smem_sfa_ptr[sfa_row];
    }
    auto tCrSFA_frg = BS1::transform_fragment_for_qmma(tCrSFA);

    Tensor tCrQ = thr_mma_g1.partition_fragment_A(sQ_persist_pi);
    load_a_z_pattern(sQ_persist_pi, tCrQ, 0, kHeadDim / 32);

    int tma_parity0 = 0;  // parity for mbar[0]: first K TMA preamble arrives at parity 0
    int tma_parity1 = 0;  // parity for mbar[1]: first V TMA preamble arrives at parity 0
    int math_stage  = 0;

    // Per-tile lambda (Opt C): instantiated with kIsMasking=true (Phase 1, causal diagonal)
    // and kIsMasking=false (Phase 2, steady-state unmasked) to allow compile-time dead-code
    // elimination of apply_mask_bs in the hot path.  n_valid_ref is modified in-place by
    // the is_jump adjustment; math_stage ^= 1 executes at the end of every tile path.
    auto run_n_tile = [&](int nb, int& n_valid_ref, int masking_step, auto kIsMasking_c) {
      constexpr bool kIsMasking = decltype(kIsMasking_c)::value;

      // Stage-dependent SMEM pointers — pure pointer arithmetic, no TMA dependency.
      int32_t* smem_sfb_cur = smem_sfa_ptr + 2 * kBlockM + math_stage * kBlockN;
      FP8Elem* sK_cur  = reinterpret_cast<FP8Elem*>(smem_q) + math_stage * 2 * kSmemKVElems;
      FP8Elem* sVt_cur = reinterpret_cast<FP8Elem*>(smem_q) + kSmemKVElems + math_stage * 2 * kSmemKVElems;

      // Wait for TMA K[nb]+Vt[nb]+SFB[nb]+SFV[nb] to land.
      {
        const int cur_parity = math_stage ? tma_parity1 : tma_parity0;
        uint32_t taddr = smem_base32 + (uint32_t)kSmemMbar0Offset + (uint32_t)(math_stage * 8);
        uint32_t done = 0;
        do {
          asm volatile("{.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;}\n"
                       : "=r"(done) : "r"(taddr), "r"((uint32_t)cur_parity));
        } while (!done);
      }
      if (math_stage) { tma_parity1 ^= 1; } else { tma_parity0 ^= 1; }
      // mbarrier.test_wait already guarantees TMA data visibility in SMEM.
      // No cross-warp SMEM write->read dependency exists before GEMM1; math_mbar
      // (count=8, one arrive per warp-leader after V s2r) protects stage buffer
      // reuse by the load warp.  Compiler barrier suffices.
      asm volatile("" ::: "memory");

      Tensor sSFB_ = make_tensor(make_smem_ptr(smem_sfb_cur), SmemLayoutSFB{});
      auto sSFB = as_position_independent_swizzle_tensor(sSFB_);

      // GEMM1: acc_s += Q × K^T (block-scaled QMMA).
      // Full-K=128 copy: make_tiled_copy_A/B expects the full TiledMMA tile;
      // cute::gemm internally loops 4×K=32 when the fragment covers K=128.
      Tensor acc_s = partition_fragment_C(tiled_mma_g1, Shape<Int<kBlockM>, Int<kBlockN>>{});
      clear(acc_s);

      auto sK_cur_pi = as_position_independent_swizzle_tensor(
          make_tensor(make_smem_ptr(sK_cur), SmemLayoutK_SW128{}));

      Tensor tCrSFB = BS1::partition_fragment_SFB(sSFB(_,_,_0{}), thr_mma_g1);
      {
        // get_layoutSFB_TV degenerates under <_8,_1,_1>: ThrN=1 causes all threads to map
        // to SMEM position 0 via the stride-0 btile, leaving tCrSFB[1..15] uninitialized.
        // Direct load from smem_sfb_ptr (LINEAR N-row order [0..kBlockN-1]).
        // partition_fragment_SFB->make_fragment_like compacts the (4,4) N-rep sub-modes in
        // column-major order (first dim fastest), giving fragment nr -> N_base = 8*nr (LINEAR).
        // n_row_sfb: this thread's N-offset within the 8-wide N-atom (lane >> 2 = 0..7).
        const int n_row_sfb = (tidx_math & 31) >> 2;
        constexpr int kNAtomsSFB = kBlockN / 8;
        CUTE_UNROLL
        for (int nr = 0; nr < kNAtomsSFB; ++nr) {
          const int N_base = nr * 8;  // LINEAR: 0, 8, 16, ..., 120
          tCrSFB(0, nr, 0) = smem_sfb_cur[N_base + n_row_sfb];
        }
      }
      auto tCrSFB_frg = BS1::transform_fragment_for_qmma(tCrSFB);

      // GEMM1: tCrQ and tCrSFA_frg are N-loop invariants hoisted above the loop.
      Tensor tCrK = thr_mma_g1.partition_fragment_B(sK_cur_pi);
      load_b_z_pattern(sK_cur_pi, tCrK, 0, kHeadDim / 32);
      cute::gemm(tiled_mma_g1,
          make_zip_tensor(tCrQ, tCrSFA_frg(_,_,_,_0{})),
          make_zip_tensor(tCrK, tCrSFB_frg(_,_,_,_0{})),
          acc_s);

      if (params.debug_gemm1_only) {
        for (int i = 0; i < size(acc_s); ++i) acc_o(i) += acc_s(i);
        if ((tidx_math & 31) == 0) {
          uint32_t maddr = smem_base32 + (uint32_t)kSmemMbar0Offset + 16u + (uint32_t)(math_stage * 8);
          asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" : : "r"(maddr));
        }
        if (is_jump && masking_step == n_masking_steps - 1)
          n_valid_ref = std::min(n_valid_ref, n_block_history);
        math_stage ^= 1;
        return;
      }

      // Masking (Opt C: compile-time specialized).
      // kIsMasking=true  (Phase 1): always apply_mask_bs (causal diagonal or Is_arbitrary/Is_local).
      // kIsMasking=false (Phase 2): steady-state; skip diagonal masking; varlen-end check only.
      if constexpr (Is_arbitrary || Is_local || kIsMasking) {
        apply_mask_bs(acc_s, nb);
      } else {
        if ((nb + 1) * kBlockN > actual_seqlen_h) apply_mask_bs(acc_s, nb);
      }
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
      // Packing order matches the flat C-fragment order of thr_mma_g1.partition_C(...).
      // Under AtomLayout <_8,_1,_1>, the old select<1,2,0,3>-based regrouping is not a
      // valid (M,N) mapping; use acc_s(flat) directly to preserve C-fragment order.
      constexpr int kAccSElems = kBlockM * kBlockN / kNMathThreads;  // = 64
      static_assert(kAccSElems % 4 == 0, "kAccSElems must be a multiple of 4");
      uint32_t acc_s_packed[kAccSElems / 4];  // 16 registers (vs. 64 for F32)
      {
        // Use cvt.e4m3x2: 2 F32→FP8 per instruction (vs. 4 scalar conversions).
        // Each cvt packs a pair into 16 bits; mov.b32 combines two pairs into uint32.
        // PTX operand order: cvt d, srcHi, srcLo → d[15:8]=FP8(srcHi), d[7:0]=FP8(srcLo)
        CUTE_UNROLL
        for (int flat = 0; flat < kAccSElems; flat += 4) {
          uint32_t out;
          asm volatile(
              "{\n"
              ".reg .b16 lo, hi;\n"
              "cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n"
              "cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
              "mov.b32 %0, {lo, hi};\n"
              "}\n"
              : "=r"(out)
              : "f"(static_cast<float>(acc_s(flat + 0))),
                "f"(static_cast<float>(acc_s(flat + 1))),
                "f"(static_cast<float>(acc_s(flat + 2))),
                "f"(static_cast<float>(acc_s(flat + 3))));
          acc_s_packed[flat / 4] = out;
        }
      }
      // acc_s (64 F32 regs) is now DEAD.
      //
      // Build GEMM2 A-fragment P directly in registers via warp shuffle.
      // With AtomLayout <_8,_1,_1> (1 N-warp), all 128 N-columns of P live in each
      // warp's own registers — no cross-warp SMEM staging or bar.sync is needed.
      // sPbuf_pi is defined for partition_fragment_A shape inference only; sK_cur is
      // not written and remains available for the load warp's next TMA fill.
      Tensor sPbuf  = make_tensor(make_smem_ptr(sK_cur), SmemLayoutQ_SW128{});
      auto sPbuf_pi = as_position_independent_swizzle_tensor(sPbuf);
      Tensor tCrP = thr_mma_g2.partition_fragment_A(sPbuf_pi);
      {
        // Warp shuffle: rearrange acc_s_packed (C-fragment, 16 uint32) into
        // tCrP (A-fragment, 16 uint32) without SMEM staging or bar.sync.
        //
        // Each (kb,c) pair selects TWO compile-time acc_s_packed slots (pk0 for
        // tiq∈{0,2}, pk1 for tiq∈{1,3}), shuffles them from the correct quad-lane
        // pair, and byte-permutes them into the mr=0/mr=1 A-fragment slots.
        //
        // acc_s_packed[] MUST stay in registers for __shfl_sync to exchange real
        // register values.  All indices into acc_s_packed are compile-time constants
        // after CUTE_UNROLL expansion so the compiler keeps the array in registers.
        auto tXrP = recast<uint32_t>(tCrP);
        const unsigned lane      = (unsigned)tidx_math & 31u;
        const unsigned tiq       = lane & 3u;
        const unsigned quad_base = lane & ~3u;
        // Lower pair (src_lane0) and upper pair (src_lane1) within the quad:
        //   tiq=0,2 → src_lane0=quad+0, src_lane1=quad+1
        //   tiq=1,3 → src_lane0=quad+2, src_lane1=quad+3
        const unsigned src_lane0 = quad_base | ((tiq & 1u) << 1u);
        const unsigned src_lane1 = src_lane0 + 1u;
        // kLut{0,1}[kb][c]: compile-time acc_s_packed indices for K-half h=0 and h=1.
        //   kLut0[kb][c] = kb + 4*(2*c + 0) = kb + 8*c
        //   kLut1[kb][c] = kb + 4*(2*c + 1) = kb + 8*c + 4
        constexpr uint8_t kLut0[4][2] = {{ 0, 8}, { 1, 9}, { 2, 10}, { 3, 11}};
        constexpr uint8_t kLut1[4][2] = {{ 4,12}, { 5,13}, { 6, 14}, { 7, 15}};
        CUTE_UNROLL
        for (int kb = 0; kb < 4; ++kb) {
          CUTE_UNROLL
          for (int c = 0; c < 2; ++c) {
            // Both indices are compile-time constants after CUTE_UNROLL expansion;
            // NVCC keeps acc_s_packed in registers and emits a predicated SHFL pair.
            const uint32_t pk0 = acc_s_packed[kLut0[kb][c]];
            const uint32_t pk1 = acc_s_packed[kLut1[kb][c]];
            // Shuffle BOTH halves first, then select: src_lane's tiq determines which
            // N-atom register IT holds, which may differ from the destination's h-half.
            const uint32_t a0  = __shfl_sync(0xFFFFFFFFu, pk0, src_lane0);
            const uint32_t b0  = __shfl_sync(0xFFFFFFFFu, pk0, src_lane1);
            const uint32_t a1  = __shfl_sync(0xFFFFFFFFu, pk1, src_lane0);
            const uint32_t b1  = __shfl_sync(0xFFFFFFFFu, pk1, src_lane1);
            const uint32_t a   = (tiq >> 1u) ? a1 : a0;
            const uint32_t b   = (tiq >> 1u) ? b1 : b0;
            // byte_perm 0x5410 = {a.b0,a.b1,b.b0,b.b1} → mr=0 (row = quad)
            // byte_perm 0x7632 = {a.b2,a.b3,b.b2,b.b3} → mr=1 (row = 8+quad)
            tXrP(4 * kb + 2 * c + 0) = __byte_perm(a, b, 0x5410u);
            tXrP(4 * kb + 2 * c + 1) = __byte_perm(a, b, 0x7632u);
          }
        }
      }

      { // tCrV, tCrSFV, tCrSFP scoped here: compiler can reuse registers freed by tCrK/tCrSFB.
      // s2r V row-major — LDSM_T from K_SW128-swizzled SMEM [kBlockN, kHeadDim].
      //
      // SmemLayoutVt_SW128 = K_SW128 on [kBlockN, kHeadDim]:
      //   physical_byte(n_k, d) = n_k * kHeadDim + (d ^ ((n_k & 7) << 4)).
      //   TMA loads row-major V directly into this layout.
      //
      // LDSM_T (ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8): transposes [16 n_k × 32 d]
      // from SMEM into registers. Lane t provides address for n_k row (ni*16 + t%16),
      // d_start = dg*32 + (t>>4)*16 (matrix 0 or 1). After transpose, thread t receives
      // 16 n_k values at its specific d-column, split into 4 registers r0..r3.
      //
      // Fragment placement: PermMmaTileN = Layout<Shape<_8,_4,_4>, Stride<_1,_32,_8>>
      //   maps d-group dg (d ∈ [dg*32,(dg+1)*32)) to N-atoms nr ∈ {dg, dg+4, dg+8, dg+12}.
      //   ni maps to K-block kb=ni>>1, K-half k_h=ni&1.
      //   base = 32*(ni>>1) + 2*dg + (ni&1):
      //     r0 → slot base+0  (nr=dg,    k_h)
      //     r1 → slot base+8  (nr=dg+4,  k_h)
      //     r2 → slot base+16 (nr=dg+8,  k_h)
      //     r3 → slot base+24 (nr=dg+12, k_h)
      //   Covers all 128 uint32 slots (16 N-atoms × 4 K-blocks × 2 regs). ✓
      auto sVt_ns = make_tensor(make_smem_ptr(sVt_cur), SmemLayoutVt_SW128{});
      Tensor tCrV = thr_mma_g2.partition_fragment_B(sVt_ns);
      {
        auto tXrV = recast<uint32_t>(tCrV);
        const uint32_t v_smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(sVt_cur));
        const int lane  = tidx_math & 31;
        const int n_off = lane & 15;          // row within 16-row n_k group
        const int d_mat = (lane >> 4) << 4;  // 0 for lanes 0-15, 16 for lanes 16-31
        CUTE_UNROLL
        for (int dg = 0; dg < kHeadDim / 32; ++dg) {   // 4 d-groups of 32 d-values each
          const int d_start = dg * 32 + d_mat;
          CUTE_UNROLL
          for (int ni = 0; ni < kBlockN / 16; ++ni) {   // 8 n_k tiles of 16 rows each
            const int n_k_row     = ni * 16 + n_off;
            const int swizzle_xor = (n_k_row & 7) << 4;
            const uint32_t addr   = v_smem_base + (uint32_t)(n_k_row * kHeadDim + (d_start ^ swizzle_xor));
            // r0..r3 cover 4 n_k-groups (4 rows each) at d = dg*32+lane, placed into
            // N-atoms {dg, dg+4, dg+8, dg+12} × K-half {ni&1} of K-block {ni>>1}.
            const int base = 32 * (ni >> 1) + 2 * dg + (ni & 1);
            asm volatile(
                "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];\n"
                : "=r"(tXrV(base + 0)), "=r"(tXrV(base + 8)),
                  "=r"(tXrV(base + 16)), "=r"(tXrV(base + 24))
                : "r"(addr));
          }
        }
      }

      // s2r SFP (unit scale) and SFV.
      // tCrSFP is per-tile: smem_sfp_ptr is thread-written (not TMA); the mbarrier wait +
      // GEMM1 above (~500 cycles) ensure all write-thread SFP commits are visible here.
      Tensor tCrSFP = BS2::partition_fragment_SFA(sSFP(_,_,_0{}), thr_mma_g2);
      {
        auto tXsSFP = s2r_thr_SFP.partition_S(sSFP);
        auto tXrSFP = s2r_thr_SFP.retile_D(tCrSFP);
        cute::copy(s2r_copy_SFP, tXsSFP(_,_,_,_0{}), tXrSFP);
      }
      auto tCrSFP_frg = BS2::transform_fragment_for_qmma(tCrSFP);
      int32_t* smem_sfv_cur = smem_sfa_ptr + 2 * kBlockM + 2 * kBlockN + math_stage * kBlockN;
      Tensor sSFV_ = make_tensor(make_smem_ptr(smem_sfv_cur), SmemLayoutSFB{});
      auto sSFV = as_position_independent_swizzle_tensor(sSFV_);
      Tensor tCrSFV = BS2::partition_fragment_SFB(sSFV(_,_,_0{}), thr_mma_g2);
      {
        // Same degeneracy as SFB: ThrN=1 → all threads → position 0 via stride-0 btile.
        // Direct load from smem_sfv_ptr; N-atoms span the head-dim (d) direction for V^T.
        // Same scale-fragment compaction rule as SFB: fragment nr -> N_base = 8*nr (LINEAR).
        // AtomLayoutSFB_TV = (4,8):(0,1): lane>>2 indexes the within-atom N-column offset.
        const int n_row_sfv = (tidx_math & 31) >> 2;
        constexpr int kNAtomsSFV = kHeadDim / 8;
        CUTE_UNROLL
        for (int nr = 0; nr < kNAtomsSFV; ++nr) {
          const int N_base = nr * 8;  // LINEAR: 0, 8, 16, ..., 120
          tCrSFV(0, nr, 0) = smem_sfv_cur[N_base + n_row_sfv];
        }
      }
      auto tCrSFV_frg = BS2::transform_fragment_for_qmma(tCrSFV);

      // Signal load warp: sK[math_stage] and sVt[math_stage] consumed; load warp may overwrite.
      if ((tidx_math & 31) == 0) {
        uint32_t maddr = smem_base32 + (uint32_t)kSmemMbar0Offset + 16u + (uint32_t)(math_stage * 8);
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" : : "r"(maddr));
      }

      // GEMM2: acc_o += P × V^T (block-scaled, tCrSFP_frg hoisted).
      cute::gemm(tiled_mma_g2,
          make_zip_tensor(tCrP, tCrSFP_frg(_,_,_,_0{})),
          make_zip_tensor(tCrV, tCrSFV_frg(_,_,_,_0{})),
          acc_o);
      } // tCrV, tCrSFV freed here.

      // End-of-tile cleanup: is_jump adjustment and double-buffer flip.
      if (is_jump && masking_step == n_masking_steps - 1)
        n_valid_ref = std::min(n_valid_ref, n_block_history);
      math_stage ^= 1;
    };  // end run_n_tile lambda

    // N-loop (Opt C): for causal (not arbitrary, not local), split into masked Phase 1
    // (n_masking_steps tiles) and unmasked Phase 2 (steady-state) to allow compile-time
    // dead-code elimination of apply_mask_bs in the hot path.
    if constexpr (Is_causal && !Is_arbitrary && !Is_local) {
      int n_valid    = n_block_max - 1;
      int masking_step = 0;
      // Phase 1: causal diagonal tiles — apply_mask_bs compiled in (kIsMasking=true).
      for (; n_valid >= n_block_min && masking_step < n_masking_steps; ++masking_step, --n_valid)
        run_n_tile(n_valid, n_valid, masking_step, std::true_type{});
      // Phase 2: steady-state tiles — apply_mask_bs compile-time eliminated (kIsMasking=false).
      for (; n_valid >= n_block_min; ++masking_step, --n_valid)
        run_n_tile(n_valid, n_valid, masking_step, std::false_type{});
    } else if constexpr (Is_arbitrary || Is_local) {
      // Every tile needs masking: pass true_type so apply_mask_bs is always compiled in.
      for (int n_valid = n_block_max - 1, masking_step = 0; n_valid >= n_block_min;
           ++masking_step, --n_valid) {
        const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;
        run_n_tile(nb, n_valid, masking_step, std::true_type{});
      }
    } else {
      // Full attention (!Is_causal, !Is_arbitrary, !Is_local): varlen-end check only.
      for (int n_valid = n_block_max - 1, masking_step = 0; n_valid >= n_block_min;
           ++masking_step, --n_valid)
        run_n_tile(n_valid, n_valid, masking_step, std::false_type{});
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
      // stmatrix.sync.aligned.x4.m8n8.shared.b16: 8 warp-cooperative stores replace 32 STS.32.
      //
      // Layout invariants this asm relies on:
      //   AtomLayout <_8,_1,_1>  → 8 math warps all in M direction, warp covers rows [warp_m*16, warp_m*16+15]
      //   PermMmaTileN <_8,_4,_4>:<_1,_32,_8>  → N_sorted[j] = 32*(j%4) + 8*(j/4) for j=0..15
      //   C-atom SM80_16x8_Row  → rO_u32[2*j + m_grp] covers (M=warp_m*16+m_grp*8+lq, N=N_sorted[j]+lqt*2)
      //   OutElement is 2-byte (BF16); sO_flat row-stride = kHeadDim * sizeof(OutElement) = 256 bytes.
      static_assert(sizeof(OutElement) == 2, "stmatrix.m8n8.b16 requires 2-byte elements");

      auto rO_u32 = recast<uint32_t>(rO);

      const int lane        = tidx_math & 31;
      const int lq          = lane >> 2;   // lane-quad (0..7): selects data row within 8×8 matrix
      const int lqt         = lane & 3;    // lane-quad-thread (0..3): selects data col-group
      const int warp_m      = tidx_math >> 5;  // M-warp index (0..7)
      const int addr_row    = ((lq & 1) << 2) | lqt;  // STSM address-row within matrix (0..7)
      const int mat_in_lane = lq >> 1;                 // which of 4 matrices this thread addresses

      const uint32_t smem_base  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_q));
      constexpr uint32_t row_bytes = kHeadDim * (uint32_t)sizeof(OutElement);  // 256

      CUTE_UNROLL
      for (int m_grp = 0; m_grp < 2; ++m_grp) {
        const uint32_t m_bytes = (uint32_t)(warp_m * 16 + m_grp * 8 + addr_row) * row_bytes;

        CUTE_UNROLL
        for (int n_grp = 0; n_grp < 4; ++n_grp) {
          // This thread addresses row addr_row of matrix mat_in_lane,
          // starting at N-column (n_grp*32 + mat_in_lane*8).
          uint32_t stsm_addr = smem_base
              + m_bytes
              + (uint32_t)(n_grp * 32 + mat_in_lane * 8) * (uint32_t)sizeof(OutElement);

          // Data: rb_k carries the uint32 whose N_sorted position is n_grp*32 + k*8.
          // N_sorted[n_grp + k*4] = 32*(n_grp+k*4)%4 + 8*(n_grp+k*4)/4 = 32*n_grp + 8*k ✓
          uint32_t rb0 = rO_u32[2 * (n_grp     ) + m_grp];
          uint32_t rb1 = rO_u32[2 * (n_grp +  4) + m_grp];
          uint32_t rb2 = rO_u32[2 * (n_grp +  8) + m_grp];
          uint32_t rb3 = rO_u32[2 * (n_grp + 12) + m_grp];

          asm volatile(
              "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
              : : "r"(stsm_addr), "r"(rb0), "r"(rb1), "r"(rb2), "r"(rb3) : "memory");
        }
      }
    }

    asm volatile("bar.sync 1, 256;\n" : : : "memory");  // S5: all 256 math threads see sO_flat writes.

    // For partial tiles (last tile of a varlen sequence): zero SMEM rows [valid_rows, kBlockM)
    // so TMA bulk store does not write garbage to GMEM beyond actual_seqlen_q.
    const int valid_rows = actual_seqlen_q - m_block * kBlockM;
    if (valid_rows < kBlockM) {
      const int oob_elems = (kBlockM - valid_rows) * kHeadDim;
      OutElement* sO_raw = reinterpret_cast<OutElement*>(smem_q) + valid_rows * kHeadDim;
      for (int i = tidx_math; i < oob_elems; i += kNMathThreads)
        sO_raw[i] = OutElement(0);
      asm volatile("bar.sync 1, 256;\n" : : : "memory");  // OOB zeros visible before TMA.
    }

    // Threads 1-255 (7 math warps) exit early — TMA does not require their participation.
    if (tidx_math != 0) { return; }

    // Thread 0: issue TMA async bulk store (sO_flat → GMEM O tile) and wait for completion.
    // fence.proxy.async.shared::cta: make all SMEM writes (STSM outputs + OOB zeros)
    // visible to the TMA proxy before the store is issued.
    asm volatile("fence.proxy.async.shared::cta;\n" : : : "memory");
    {
      using SmemLayoutO_TMA_t = cute::Layout<
          cute::Shape<cute::Int<kBlockM>, cute::Int<kHeadDim>>,
          cute::Stride<cute::Int<kHeadDim>, cute::_1>>;
      Tensor sO_tma = make_tensor(
          make_smem_ptr(reinterpret_cast<OutElement*>(smem_q)),
          SmemLayoutO_TMA_t{});
      // mO_tma: full output tensor [total_q, d, h] — matches descriptor dim ordering.
      // Slice off the head dimension (dim 2) at bidh to get a (total_q, d) per-head view.
      auto mO_tma   = params.tma_o.get_tma_tensor(make_shape(params.total_q, params.d, params.h));
      auto gO_head  = mO_tma(_, _, bidh);
      auto gO_tiles = local_tile(gO_head, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(_, _));
      const int m_abs = binfo.sum_s_q / kBlockM + m_block;
      auto tma_slice_O = params.tma_o.get_slice(0);
      Tensor tOsO     = tma_slice_O.partition_S(sO_tma);                   // SMEM source
      Tensor tOgO_all = tma_slice_O.partition_D(gO_tiles(_, _, _, Int<0>{}));  // GMEM dest tiles
      cute::copy(params.tma_o, tOsO, tOgO_all(_, _, _, m_abs));
      cute::tma_store_arrive();
      cute::tma_store_wait<0>();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Phase 6 WS TMA kernel entry: launched with kNThreadsTotal=288.
// Q, K, V^T, Q-SF, K-SF, and V-SF all via TMA.
template <typename Kernel_traits, typename TMA_Q_t, typename TMA_K_t, typename TMA_Vt_t,
          typename TMA_SFA_t, typename TMA_SFB_t, typename TMA_SFV_t, typename TMA_O_t>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1)
hstu_fwd_kernel_sm120_fp8_ws_tma(
    __grid_constant__ Hstu_fwd_params_fp8_ws_tma<TMA_Q_t, TMA_K_t, TMA_Vt_t,
                                                  TMA_SFA_t, TMA_SFB_t, TMA_SFV_t, TMA_O_t> const params) {
  int m_block = gridDim.x - blockIdx.x - 1;
  int bidh    = blockIdx.y;
  int bidb    = blockIdx.z;
  hstu_compute_attn_1rowblock_sm120_fp8_ws<Kernel_traits>(params, bidb, bidh, m_block);
}
