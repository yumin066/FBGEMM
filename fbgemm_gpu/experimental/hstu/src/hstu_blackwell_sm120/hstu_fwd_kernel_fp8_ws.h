// Phase 6 warp-specialized FP8 kernel body — Q, K, and V^T via TMA.
// Included inside namespace flash from hstu_fwd_kernel.h.
// Do not include directly; use hstu_fwd_kernel.h.
//
// Warp layout (12 warps = 3 complete warpgroups, kNThreads=384):
//   WG0: warps 0-3  (math)
//   WG1: warps 4-7  (math)
//   WG2: warps 8-11 (load warpgroup)
//     warp 8  = Q/SFA TMA load
//     warp 9  = K/SFB TMA load
//     warp 10 = V/SFV TMA load
//     warp 11 = O TMA store
// Complete WG2 ensures setmaxnreg TRY_ALLOC WARPSYNC.ALL retry loop works correctly.
// With __launch_bounds__(384,1): compiler budget = 65536/384 ≈ 168 regs → TRY_ALLOC(inc) succeeds
// on first attempt (already at budget), no retry needed.
//
// CTA-level __syncthreads__ map (must match between branches):
//   S_arb : Is_arbitrary only — math warp 1 writes sValidBlockIds; load warp just syncs
//   S1    : after load warp inits producer/consumer mbarriers + fence; persistent full/causal does this
//           once before the static scheduler loop, non-persistent paths keep the per-tile S1
//   S2→   : replaced by wait_mbar_parity(q_ready_mbar_ptr, parity) for all 256 math threads
//   S3    : removed (was a no-op rendezvous with no real data dependency)
//   S5    : partial-tile-only math bar.sync; full O-store handoff uses o_ready/o_empty.

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ uint32_t mbar_smem_addr(uint64_t* mbar) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
}

__device__ __forceinline__ void arrive_mbar(uint32_t maddr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" : : "r"(maddr));
}

__device__ __forceinline__ void arrive_mbar(uint64_t* mbar) {
  arrive_mbar(mbar_smem_addr(mbar));
}

__device__ __forceinline__ void arrive_expect_tx_mbar(
    uint32_t maddr,
    uint32_t expected_tx_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               : : "r"(maddr), "r"(expected_tx_bytes));
}

__device__ __forceinline__ void arrive_expect_tx_mbar(
    uint64_t* mbar,
    uint32_t expected_tx_bytes) {
  arrive_expect_tx_mbar(mbar_smem_addr(mbar), expected_tx_bytes);
}

// Spin-wait helper: blocks until the expected mbarrier phase completes.
__device__ inline void wait_mbar_parity(uint32_t maddr, uint32_t parity) {
  uint32_t done = 0;
  do {
    asm volatile(
        "{.reg .pred p;\n"
        "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;}\n"
        : "=r"(done) : "r"(maddr), "r"(parity));
  } while (!done);
}

__device__ inline void wait_mbar_parity(uint64_t* mbar, uint32_t parity) {
  wait_mbar_parity(mbar_smem_addr(mbar), parity);
}

__device__ __forceinline__ void cp_async_cg_16B(
    void* __restrict__ dst_smem,
    const void* __restrict__ src_gmem) {
  const uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
  const uint64_t gmem_addr = reinterpret_cast<uint64_t>(src_gmem);
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      : : "r"(smem_addr), "l"(gmem_addr) : "memory");
}

template <typename FP8Elem, int kHeadDim, typename SmemLayout>
__device__ __forceinline__ void copy_fp8_tile_rowmajor_to_smem(
    FP8Elem* __restrict__ dst_smem,
    const FP8Elem* __restrict__ src_gmem,
    int64_t src_row_stride,
    int rows_valid,
    int lane,
    SmemLayout const& layout) {
  constexpr int kVecElems = 16;
  constexpr int kVecsPerRow = kHeadDim / kVecElems;
  static_assert(sizeof(FP8Elem) == 1, "FP8 paged copy expects 1-byte elements");
  using Vec = uint4;
  const Vec zero = make_uint4(0, 0, 0, 0);

  int issued_in_group = 0;
  for (int vec = lane; vec < rows_valid * kVecsPerRow; vec += 32) {
    const int row = vec / kVecsPerRow;
    const int d = (vec - row * kVecsPerRow) * kVecElems;
    cp_async_cg_16B(
        dst_smem + static_cast<int>(layout(row, d)),
        src_gmem + row * src_row_stride + d);
    if (++issued_in_group == 8) {
      asm volatile("cp.async.commit_group;\n" : : : "memory");
      asm volatile("cp.async.wait_group 0;\n" : : : "memory");
      issued_in_group = 0;
    }
  }
  if (issued_in_group != 0) {
    asm volatile("cp.async.commit_group;\n" : : : "memory");
    asm volatile("cp.async.wait_group 0;\n" : : : "memory");
  }
  asm volatile("fence.proxy.async.shared::cta;\n" : : : "memory");

  for (int vec = lane + rows_valid * kVecsPerRow; vec < 64 * kVecsPerRow; vec += 32) {
    const int row = vec / kVecsPerRow;
    const int d = (vec - row * kVecsPerRow) * kVecElems;
    *reinterpret_cast<Vec*>(dst_smem + static_cast<int>(layout(row, d))) = zero;
  }
}

__device__ __forceinline__ void copy_packed_sf_tile(
    int32_t* __restrict__ dst_smem,
    const int32_t* __restrict__ src_gmem,
    int64_t src_head_stride,
    int head,
    int token_base,
    int rows_valid,
    int lane) {
  for (int row = lane; row < 64; row += 32) {
    dst_smem[row] = row < rows_valid
        ? src_gmem[(int64_t)head * src_head_stride + token_base + row]
        : 0x7f7f7f7f;
  }
}

struct HstuWsTileCoord {
  int bidb;
  int bidh;
  int m_block;
};

__device__ __forceinline__ HstuWsTileCoord hstu_ws_decode_tile(
    const int tile,
    const int num_m_block,
    const int num_heads,
    const bool head_shared_rab) {
  if (head_shared_rab) {
    const int bidh = tile % num_heads;
    const int bm = tile / num_heads;
    const int m_linear = bm % num_m_block;
    return HstuWsTileCoord{
        bm / num_m_block,
        bidh,
        num_m_block - 1 - m_linear};
  } else {
    const int m_linear = tile % num_m_block;
    const int bh = tile / num_m_block;
    return HstuWsTileCoord{
        bh / num_heads,
        bh % num_heads,
        num_m_block - 1 - m_linear};
  }
}

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

template <
    typename Kernel_traits,
    bool Use_full_persistent = false,
    bool Use_paired_persistent = false,
    typename Params>
inline __device__ void hstu_compute_attn_1rowblock_sm120_fp8_ws(
    const Params& params,
    const int bidb_arg,
    const int bidh_arg,
    int m_block_arg) {

  static_assert(Kernel_traits::Is_fp8, "Phase 6 WS: FP8 path only");
  static_assert(
      !(Use_full_persistent && Use_paired_persistent),
      "Only one FP8 WS persistent scheduler can be enabled");
  constexpr bool Use_persistent = Use_full_persistent || Use_paired_persistent;

  using BS1 = hstu::SM120QmmaBuilder<
      Kernel_traits::kBlockM, Kernel_traits::kBlockN, 4>;
  static constexpr int kHeadDimGemm2 =
      Kernel_traits::kHeadDim < 64 ? 64 : Kernel_traits::kHeadDim;
  using BS2 = hstu::SM120QmmaBuilder<
      Kernel_traits::kBlockM, kHeadDimGemm2, 4>;
  using FP8Elem = typename Kernel_traits::Element;

  extern __shared__ char smem_[];
  static constexpr int kSmemMbar0Offset =
      Kernel_traits::kSmemSize - Kernel_traits::kSmemMbarSize;

  const int tidx = threadIdx.x;
  constexpr int kNMathThreads = Kernel_traits::kNMathThreads;  // 256
  const bool is_load_warp = (tidx >= kNMathThreads);

  // CTA-level sync before setmaxnreg: all warps start from clean state.
  __syncthreads();

  // ============================================================
  // LOAD WARPGROUP PATH  (warps 8-11, threads 256-383)
  //   warp 8  (tidx 256-287): Q/SFA load
  //   warp 9  (tidx 288-319): K/SFB load
  //   warp 10 (tidx 320-351): V/SFV load
  //   warp 11 (tidx 352-383): O store
  // ============================================================
  if (is_load_warp) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(56));
    // Four load warps are role-specialized:
    //   warp 8  : Q + SFA TMA load
    //   warp 9  : K + SFB TMA load
    //   warp 10 : V + SFV TMA load
    //   warp 11 : O TMA store
    const int load_warp_id = (tidx - kNMathThreads) >> 5;
    const bool is_q_load_warp = load_warp_id == 0;
    const bool is_k_load_warp = load_warp_id == 1;
    const bool is_v_load_warp = load_warp_id == 2;
    const bool is_o_store_warp = load_warp_id == 3;

    constexpr bool Is_causal    = Kernel_traits::Is_causal;
    constexpr bool Is_target    = Kernel_traits::Is_target;
    constexpr bool Is_context   = Kernel_traits::Is_context;
    constexpr bool Is_arbitrary = Kernel_traits::Is_arbitrary;
    constexpr int  kNFunc       = Kernel_traits::kNFunc;
    constexpr bool Is_local     = Kernel_traits::Is_local;
    constexpr bool Has_rab      = Kernel_traits::Has_rab;
    constexpr bool Paged_KV     = Kernel_traits::Paged_KV;
    constexpr int  kBlockM      = Kernel_traits::kBlockM;
    constexpr int  kBlockN      = Kernel_traits::kBlockN;
    constexpr int  kHeadDim     = Kernel_traits::kHeadDim;
    using OutElement = typename Kernel_traits::OutputType;

    uint64_t* k_ready_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset);
    uint64_t* k_ready_mbar_ptr1 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 8);
    uint64_t* v_ready_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 16);
    uint64_t* v_ready_mbar_ptr1 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 24);
    uint64_t* k_empty_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 32);
    uint64_t* k_empty_mbar_ptr1 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 40);
    uint64_t* v_empty_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 48);
    uint64_t* v_empty_mbar_ptr1 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 56);
    uint64_t* q_ready_mbar_ptr  = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 64);
    uint64_t* q_empty_mbar_ptr  = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 72);
    uint64_t* o_ready_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 80);
    uint64_t* o_ready_mbar_ptr1 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 88);
    uint64_t* o_empty_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 96);
    uint64_t* o_empty_mbar_ptr1 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 104);
    uint64_t* rab_ready_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 112);
    uint64_t* rab_empty_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 120);

    auto init_ws_mbarriers = [&]() {
      if (tidx == kNMathThreads) {
        uint32_t kr0 = static_cast<uint32_t>(__cvta_generic_to_shared(k_ready_mbar_ptr0));
        uint32_t kr1 = static_cast<uint32_t>(__cvta_generic_to_shared(k_ready_mbar_ptr1));
        uint32_t vr0 = static_cast<uint32_t>(__cvta_generic_to_shared(v_ready_mbar_ptr0));
        uint32_t vr1 = static_cast<uint32_t>(__cvta_generic_to_shared(v_ready_mbar_ptr1));
        uint32_t ke0 = static_cast<uint32_t>(__cvta_generic_to_shared(k_empty_mbar_ptr0));
        uint32_t ke1 = static_cast<uint32_t>(__cvta_generic_to_shared(k_empty_mbar_ptr1));
        uint32_t ve0 = static_cast<uint32_t>(__cvta_generic_to_shared(v_empty_mbar_ptr0));
        uint32_t ve1 = static_cast<uint32_t>(__cvta_generic_to_shared(v_empty_mbar_ptr1));
        uint32_t qr  = static_cast<uint32_t>(__cvta_generic_to_shared(q_ready_mbar_ptr));
        uint32_t qe  = static_cast<uint32_t>(__cvta_generic_to_shared(q_empty_mbar_ptr));
        uint32_t or0 = static_cast<uint32_t>(__cvta_generic_to_shared(o_ready_mbar_ptr0));
        uint32_t or1 = static_cast<uint32_t>(__cvta_generic_to_shared(o_ready_mbar_ptr1));
        uint32_t oe0 = static_cast<uint32_t>(__cvta_generic_to_shared(o_empty_mbar_ptr0));
        uint32_t oe1 = static_cast<uint32_t>(__cvta_generic_to_shared(o_empty_mbar_ptr1));
        uint32_t rr0 = static_cast<uint32_t>(__cvta_generic_to_shared(rab_ready_mbar_ptr0));
        uint32_t re0 = static_cast<uint32_t>(__cvta_generic_to_shared(rab_empty_mbar_ptr0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(kr0), "r"(1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(kr1), "r"(1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(vr0), "r"(1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(vr1), "r"(1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(ke0), "r"(8));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(ke1), "r"(8));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(ve0), "r"(8));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(ve1), "r"(8));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(qr), "r"(1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(qe), "r"(8));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(or0), "r"(8));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(or1), "r"(8));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(oe0), "r"(1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(oe1), "r"(1));
        if constexpr (Has_rab) {
          asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(rr0), "r"(1));
          asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(re0), "r"(8));
        }
        for (int i = 0; i < 8; i++) {
          arrive_mbar(ke0);
          arrive_mbar(ke1);
          arrive_mbar(ve0);
          arrive_mbar(ve1);
          arrive_mbar(qe);
          if constexpr (Has_rab) {
            arrive_mbar(re0);
          }
        }
        // Independent O buffer starts empty; math epilogue waits on this phase before
        // writing O, and the O-store warp releases the next phase after TMA store wait.
        arrive_mbar(oe0);
      }
      asm volatile("fence.proxy.async.shared::cta;\n" : : : "memory");
    };

    int k_empty_wait_parity0 = 0;
    int k_empty_wait_parity1 = 0;
    int v_empty_wait_parity0 = 0;
    int v_empty_wait_parity1 = 0;
    int q_empty_wait_parity = 0;
    int o_ready_wait_parity0 = 0;
    int rab_empty_wait_parity0 = 0;
    int kv_load_stage = 0;
    if constexpr (Use_persistent) {
      init_ws_mbarriers();
      __syncthreads();  // One-time S1 for persistent: WS mbarriers initialized before scheduler loop.
    }

    auto run_load_tile = [&](const int bidb, const int bidh, int m_block) {
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
      const int last_page_seqlen       = Paged_KV ? binfo.last_page_seqlen : kBlockN;
      const int page_offset            = Paged_KV ? binfo.sum_s_page : 0;

      const bool is_jump             = Is_target && m_block * kBlockM + actual_seqlen_offset > actual_seqlen_h;
      const bool is_in_target        = Is_target && (m_block + 1) * kBlockM + actual_seqlen_offset > actual_seqlen_h;
      const bool is_in_context       = Is_context && (m_block + 1) * kBlockM <= actual_seqlen_c;
      const bool is_in_mixed_context = Is_context &&
          (m_block + 1) * kBlockM > actual_seqlen_c && m_block * kBlockM < actual_seqlen_c;
      const bool is_in_paged_target  = is_in_target && Paged_KV;
      const int last_page_offset     = is_in_paged_target ? kBlockN - last_page_seqlen : 0;

      const int n_block_history = cute::ceil_div(actual_seqlen_h, kBlockN);
      const int target_index    = (m_block * kBlockM - actual_seqlen_h) / params.target_group_size;
      const int n_block_paged   = Paged_KV ? n_block_history : 0;
      const int n_block_target  = cute::ceil_div(actual_seqlen_t, kBlockN);

      int n_block_min = !Is_local ? 0
          : std::max(0, (m_block * kBlockM + actual_seqlen_offset - params.window_size_left) / kBlockN);
      int n_block_max = Paged_KV ? n_block_history + n_block_target : cute::ceil_div(actual_seqlen_k, kBlockN);
      if constexpr (Is_causal || Is_local) {
        int offset = (m_block + 1) * kBlockM + actual_seqlen_offset + params.window_size_right;
        if (is_in_paged_target) offset += last_page_offset;
        n_block_max = std::min(n_block_max, cute::ceil_div(offset, kBlockN));
      }
      if constexpr (Is_context) {
        n_block_min = (is_in_context || is_in_mixed_context) ? 0 : n_block_min;
        n_block_max = (is_in_context || is_in_mixed_context)
            ? std::max(n_block_history, n_block_max) : n_block_max;
      }

      int n_masking_block_max = cute::ceil_div(
          std::min(actual_seqlen_k + last_page_offset,
                   (m_block + 1) * kBlockM + actual_seqlen_offset + last_page_offset),
          kBlockN);
      int n_masking_block_min = (m_block * kBlockM + actual_seqlen_offset) / kBlockN;
      if constexpr (Is_target) {
        n_masking_block_min = is_jump
            ? (actual_seqlen_h + actual_seqlen_offset + target_index * params.target_group_size + last_page_offset) / kBlockN
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

      // Early exit 2 (after Is_arbitrary): non-persistent paths still consume the per-tile S1.
      if (((Is_causal || Is_local || Is_arbitrary) && n_block_max <= n_block_min) ||
          m_block * kBlockM >= actual_seqlen_q) {
        if constexpr (!Use_persistent) {
          __syncthreads();  // S1
        }
        return;
      }

      // SMEM layout types for TMA partition_D.
      using SmemLayoutK_SW128  = typename Kernel_traits::SmemLayoutK_TMA;
      using SmemLayoutVt_SW128 = typename Kernel_traits::SmemLayoutVt_TMA;
      using SmemLayoutQ_SW128  = typename Kernel_traits::SmemLayoutQ_TMA;
      using SmemLayoutRab_TMA  = typename Kernel_traits::SmemLayoutRab_TMA;
      using RabElement = cutlass::bfloat16_t;

      constexpr int kSmemKVElems = kBlockN * kHeadDim;
      FP8Elem* const sK_base[2] = {
          reinterpret_cast<FP8Elem*>(smem_q),
          Kernel_traits::kUseSingleKVStage
              ? reinterpret_cast<FP8Elem*>(smem_q)
              : reinterpret_cast<FP8Elem*>(smem_q) + 2 * kSmemKVElems
      };
      FP8Elem* const sVt_base[2] = {
          reinterpret_cast<FP8Elem*>(smem_q) + kSmemKVElems,
          Kernel_traits::kUseSingleKVStage
              ? reinterpret_cast<FP8Elem*>(smem_q) + kSmemKVElems
              : reinterpret_cast<FP8Elem*>(smem_q) + 3 * kSmemKVElems
      };

      // q_tma_mbar_ptr is initialized with the other WS mbarriers above.

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

      if constexpr (!Use_persistent) {
        init_ws_mbarriers();
        __syncthreads();  // S1: WS mbarriers visible to all warps.
      }

      // TMA tensor setup for K, V^T, SFB, SFV (load warps only).
      const int bidh_kv = bidh / params.h_h_k_ratio;
      constexpr int kSmemKBytes    = kBlockN * kHeadDim * (int)sizeof(FP8Elem);
      constexpr int kSmemVtBytes   = kHeadDim * kBlockN * (int)sizeof(FP8Elem);
      constexpr int kSmemSFBBytes  = kBlockN * (int)sizeof(int32_t);
      constexpr int kSmemSFVBytes  = kBlockN * (int)sizeof(int32_t);
      constexpr uint32_t kSmemKSFBBytes =
          (uint32_t)(kSmemKBytes + kSmemSFBBytes);
      constexpr uint32_t kSmemVtSFVBytes =
          (uint32_t)(kSmemVtBytes + kSmemSFVBytes);
      constexpr uint32_t kSmemRabBytes =
          (uint32_t)(kBlockM * kBlockN * (int)sizeof(RabElement));

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

      RabElement* smem_rab = reinterpret_cast<RabElement*>(
          reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsRabOffset);
      Tensor sRab_tma = make_tensor(make_smem_ptr(smem_rab), SmemLayoutRab_TMA{});
      const int bidh_rab = Has_rab && params.h_rab > 1 ? bidh : 0;
      auto mRab_tma = params.tma_rab.get_tma_tensor(
          make_shape(params.seqlen_k_rounded, params.seqlen_k_rounded,
                     Has_rab ? params.h_rab : 1, params.b))(_, _, bidh_rab, bidb);
      auto gRab_tiles = local_tile(
          domain_offset(make_coord(actual_seqlen_offset, _0{}), mRab_tma),
          Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block, _));
      auto tma_slice_Rab = params.tma_rab.get_slice(0);
      auto tRabgRab_tma = group_modes<0, 3>(tma_slice_Rab.partition_S(gRab_tiles));
      auto tRabsRab_d = group_modes<0, 3>(tma_slice_Rab.partition_D(sRab_tma));

      auto use_rab_smem_for_nb = [&](int nb) {
        if constexpr (!Has_rab) {
          return false;
        } else {
          bool aligned = (actual_seqlen_offset % kBlockM) == 0;
          if constexpr (Paged_KV && Is_target) {
            if (nb >= n_block_paged && last_page_offset != 0) {
              aligned = false;
            }
          }
          return aligned;
        }
      };

      auto load_rab_tma = [&](int nb) {
        if constexpr (Has_rab) {
          if (use_rab_smem_for_nb(nb)) {
            wait_mbar_parity(rab_empty_mbar_ptr0, (uint32_t)rab_empty_wait_parity0);
            rab_empty_wait_parity0 ^= 1;
            if (tidx == kNMathThreads + 32) {
              arrive_expect_tx_mbar(rab_ready_mbar_ptr0, kSmemRabBytes);
              cute::copy(params.tma_rab.with(*rab_ready_mbar_ptr0),
                         tRabgRab_tma(_, nb), tRabsRab_d(_, _0{}));
            }
          }
        }
      };

      Tensor sValidBlockIds = make_tensor(
          make_smem_ptr(reinterpret_cast<int*>(smem_ + Kernel_traits::kSmemWsValidBlockIdsOffset)),
          typename Kernel_traits::SmemLayoutValidBlockIds{});

      // ===== ROLE-SPECIALIZED LOAD WARPS =====
      // Q/SFA warp issues the per-row-tile Q preamble. K/SFB and V/SFV warps independently
      // feed the double-buffered N loop. O-store warp owns the final TMA store.  Q and O do
      // not get extra buffers.  Producer/consumer mbarriers replace the old tile-end load
      // rendezvous: load warps wait empty, math waits ready, and O-store waits O ready.
      if (is_q_load_warp) {
        wait_mbar_parity(q_empty_mbar_ptr, (uint32_t)q_empty_wait_parity);
        q_empty_wait_parity ^= 1;
        if (tidx == kNMathThreads) {
          constexpr uint32_t kSmemQBytes   = kBlockM * kHeadDim * (uint32_t)sizeof(FP8Elem);
          constexpr uint32_t kSmemSFABytes = kBlockM * (uint32_t)sizeof(int32_t);
          using SmemLayoutSFA_TMA_t = cute::Layout<cute::Shape<cute::Int<kBlockM>, cute::Int<1>>,
                                                   cute::Stride<cute::_1, cute::Int<kBlockM>>>;
          auto mQ_tma   = params.tma_q.get_tma_tensor(make_shape(params.total_q, params.d, params.h));
          auto gQ_head  = mQ_tma(_, _, bidh);
          auto gQ_tiles = local_tile(gQ_head, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(_, _));
          auto tma_slice_Q = params.tma_q.get_slice(0);
          auto sQ_buf      = make_tensor(
              make_smem_ptr(reinterpret_cast<FP8Elem*>(
                  reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsQPersistOffset)),
              SmemLayoutQ_SW128{});
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
          arrive_expect_tx_mbar(q_ready_mbar_ptr, kSmemQBytes + kSmemSFABytes);
          const int m_abs = binfo.sum_s_q / kBlockM + m_block;
          cute::copy(params.tma_q.with(*q_ready_mbar_ptr),   tQgQ_tma(_, _, _, m_abs),     tQsQ_d);
          cute::copy(params.tma_sfa.with(*q_ready_mbar_ptr), tSFAgSFA_tma(_, _, _, m_abs), tSFAsSFA_d);
        }
      }

      if (is_k_load_warp) {
        if constexpr (Paged_KV) {
          auto copy_paged_or_target_k = [&](int nb, int stage) {
            FP8Elem* dst = stage ? sK_base[1] : sK_base[0];
            int32_t* sf_dst = stage ? smem_sfb_ptr[1] : smem_sfb_ptr[0];
            const int lane = tidx & 31;
            if (nb < n_block_paged) {
              const int page_id = params.page_ids[page_offset + nb];
              const FP8Elem* src = reinterpret_cast<const FP8Elem*>(params.kv_cache_ptr)
                  + (int64_t)page_id * params.kv_cache_kvtensor_stride
                  + (int64_t)bidh_kv * params.kv_cache_row_stride;
              copy_fp8_tile_rowmajor_to_smem<FP8Elem, kHeadDim>(
                  dst, src, params.kv_cache_head_stride, kBlockN, lane,
                  SmemLayoutK_SW128{});
              copy_packed_sf_tile(
                  sf_dst, params.sf_k_packed_ptr, params.kv_block_descale_head_stride,
                  bidh_kv, page_id * params.page_size, kBlockN, lane);
            } else {
              const int target_block = nb - n_block_paged;
              const int target_start = binfo.sum_s_k + actual_seqlen_k - actual_seqlen_t
                  + last_page_offset + target_block * kBlockN;
              const int rows_valid = std::max(0, std::min(kBlockN, actual_seqlen_t - target_block * kBlockN));
              const FP8Elem* src = reinterpret_cast<const FP8Elem*>(params.k_ptr)
                  + (int64_t)target_start * params.k_row_stride
                  + (int64_t)bidh_kv * params.k_head_stride;
              copy_fp8_tile_rowmajor_to_smem<FP8Elem, kHeadDim>(
                  dst, src, params.k_row_stride, rows_valid, lane,
                  SmemLayoutK_SW128{});
              copy_packed_sf_tile(
                  sf_dst, params.sf_k_packed_ptr, params.kv_block_descale_head_stride,
                  bidh_kv, params.total_pages * params.page_size + target_start,
                  rows_valid, lane);
            }
            __syncwarp();
          };

          auto mK_page_tma = params.tma_k_page.get_tma_tensor(
              make_shape(params.page_size, params.d, params.h_k, params.total_pages));
          auto gK_page_head = mK_page_tma(_, _, bidh_kv, _);
          auto gK_page_tiles = local_tile(
              gK_page_head, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _));
          auto tma_slice_K_page = params.tma_k_page.get_slice(0);
          auto tKPgK_tma = tma_slice_K_page.partition_S(
              gK_page_tiles(_, _, _, Int<0>{}, _));
          auto tKPsK_d_0 = tma_slice_K_page.partition_D(
              make_tensor(make_smem_ptr(sK_base[0]), SmemLayoutK_SW128{}));
          auto tKPsK_d_1 = tma_slice_K_page.partition_D(
              make_tensor(make_smem_ptr(sK_base[1]), SmemLayoutK_SW128{}));

          auto load_paged_tma_or_target_k =
              [&](int nb, int stage, uint64_t* k_ready_mbar_ptr) {
            const bool use_paged_history_tma = (kHeadDim > 128) || n_block_paged >= 16;
            if (nb < n_block_paged && use_paged_history_tma) {
              const int page_id = params.page_ids[page_offset + nb];
              const int sf_page_block = (page_id * params.page_size) / kBlockN;
              if (tidx == kNMathThreads + 32) {
                arrive_expect_tx_mbar(k_ready_mbar_ptr, kSmemKSFBBytes);
                if (stage == 0) {
                  cute::copy(params.tma_k_page.with(*k_ready_mbar_ptr),
                             tKPgK_tma(_, _, _, Int<0>{}, page_id), tKPsK_d_0);
                  cute::copy(params.tma_sfb.with(*k_ready_mbar_ptr),
                             tSFBgSFB_tma(_, _, _, sf_page_block), tSFBsSFB_d_0);
                } else {
                  cute::copy(params.tma_k_page.with(*k_ready_mbar_ptr),
                             tKPgK_tma(_, _, _, Int<0>{}, page_id), tKPsK_d_1);
                  cute::copy(params.tma_sfb.with(*k_ready_mbar_ptr),
                             tSFBgSFB_tma(_, _, _, sf_page_block), tSFBsSFB_d_1);
                }
              }
            } else if (nb >= n_block_paged) {
              const int target_block = nb - n_block_paged;
              const int target_start = binfo.sum_s_k + actual_seqlen_k - actual_seqlen_t
                  + last_page_offset + target_block * kBlockN;
              const int rows_valid = std::max(0, std::min(kBlockN, actual_seqlen_t - target_block * kBlockN));
              const bool use_target_tma =
                  (rows_valid == kBlockN) && (target_start % kBlockN == 0);
              if (use_target_tma) {
                const int nb_abs = target_start / kBlockN;
                const int sf_abs = (params.total_pages * params.page_size + target_start) / kBlockN;
                if (tidx == kNMathThreads + 32) {
                  arrive_expect_tx_mbar(k_ready_mbar_ptr, kSmemKSFBBytes);
                  if (stage == 0) {
                    cute::copy(params.tma_k.with(*k_ready_mbar_ptr),
                               tKgK_tma(_, _, _, nb_abs), tKsK_d_0);
                    cute::copy(params.tma_sfb.with(*k_ready_mbar_ptr),
                               tSFBgSFB_tma(_, _, _, sf_abs), tSFBsSFB_d_0);
                  } else {
                    cute::copy(params.tma_k.with(*k_ready_mbar_ptr),
                               tKgK_tma(_, _, _, nb_abs), tKsK_d_1);
                    cute::copy(params.tma_sfb.with(*k_ready_mbar_ptr),
                               tSFBgSFB_tma(_, _, _, sf_abs), tSFBsSFB_d_1);
                  }
                }
              } else {
                copy_paged_or_target_k(nb, stage);
                if (tidx == kNMathThreads + 32) {
                  arrive_mbar(k_ready_mbar_ptr);
                }
              }
            } else {
              copy_paged_or_target_k(nb, stage);
              if (tidx == kNMathThreads + 32) {
                arrive_mbar(k_ready_mbar_ptr);
              }
            }
          };

          for (int n_valid = n_block_max - 1, masking_step_load = 0; n_valid >= n_block_min;
               ++masking_step_load, --n_valid) {
            const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;

            uint64_t* k_empty_mbar_ptr = kv_load_stage ? k_empty_mbar_ptr1 : k_empty_mbar_ptr0;
            int& k_empty_wait_parity = kv_load_stage ? k_empty_wait_parity1 : k_empty_wait_parity0;
            wait_mbar_parity(k_empty_mbar_ptr, (uint32_t)k_empty_wait_parity);
            k_empty_wait_parity ^= 1;

            load_paged_tma_or_target_k(
                nb, kv_load_stage, kv_load_stage ? k_ready_mbar_ptr1 : k_ready_mbar_ptr0);
            load_rab_tma(nb);

            if (is_jump && masking_step_load == n_masking_steps - 1)
              n_valid = std::min(n_valid, n_block_history);

            if constexpr (!Kernel_traits::kUseSingleKVStage) {
              kv_load_stage ^= 1;
            }
          }
        } else {
          for (int n_valid = n_block_max - 1, masking_step_load = 0; n_valid >= n_block_min;
               ++masking_step_load, --n_valid) {
            const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;

            uint64_t* k_empty_mbar_ptr = kv_load_stage ? k_empty_mbar_ptr1 : k_empty_mbar_ptr0;
            int& k_empty_wait_parity = kv_load_stage ? k_empty_wait_parity1 : k_empty_wait_parity0;
            wait_mbar_parity(k_empty_mbar_ptr, (uint32_t)k_empty_wait_parity);
            k_empty_wait_parity ^= 1;

            if (tidx == kNMathThreads + 32) {
              const int nb_abs = binfo.sum_s_k / kBlockN + nb;
              if (kv_load_stage == 0) {
                arrive_expect_tx_mbar(k_ready_mbar_ptr0, kSmemKSFBBytes);
                cute::copy(params.tma_k.with(*k_ready_mbar_ptr0),   tKgK_tma(_, _, _, nb_abs),    tKsK_d_0);
                cute::copy(params.tma_sfb.with(*k_ready_mbar_ptr0), tSFBgSFB_tma(_, _, _, nb_abs), tSFBsSFB_d_0);
              } else {
                arrive_expect_tx_mbar(k_ready_mbar_ptr1, kSmemKSFBBytes);
                cute::copy(params.tma_k.with(*k_ready_mbar_ptr1),   tKgK_tma(_, _, _, nb_abs),    tKsK_d_1);
                cute::copy(params.tma_sfb.with(*k_ready_mbar_ptr1), tSFBgSFB_tma(_, _, _, nb_abs), tSFBsSFB_d_1);
              }
            }
            load_rab_tma(nb);

            if (is_jump && masking_step_load == n_masking_steps - 1)
              n_valid = std::min(n_valid, n_block_history);

            if constexpr (!Kernel_traits::kUseSingleKVStage) {
              kv_load_stage ^= 1;
            }
          }
        }
      }

      if (is_v_load_warp) {
        if constexpr (Paged_KV) {
          auto copy_paged_or_target_v = [&](int nb, int stage) {
            FP8Elem* dst = stage ? sVt_base[1] : sVt_base[0];
            int32_t* sf_dst = stage ? smem_sfv_ptr[1] : smem_sfv_ptr[0];
            const int lane = tidx & 31;
            if (nb < n_block_paged) {
              const int page_id = params.page_ids[page_offset + nb];
              const FP8Elem* src = reinterpret_cast<const FP8Elem*>(params.kv_cache_ptr)
                  + (int64_t)page_id * params.kv_cache_kvtensor_stride
                  + params.kv_cache_page_stride
                  + (int64_t)bidh_kv * params.kv_cache_row_stride;
              copy_fp8_tile_rowmajor_to_smem<FP8Elem, kHeadDim>(
                  dst, src, params.kv_cache_head_stride, kBlockN, lane,
                  SmemLayoutVt_SW128{});
              copy_packed_sf_tile(
                  sf_dst, params.sf_v_packed_ptr, params.v_block_descale_head_stride,
                  bidh_kv, page_id * params.page_size, kBlockN, lane);
            } else {
              const int target_block = nb - n_block_paged;
              const int target_start = binfo.sum_s_k + actual_seqlen_k - actual_seqlen_t
                  + last_page_offset + target_block * kBlockN;
              const int rows_valid = std::max(0, std::min(kBlockN, actual_seqlen_t - target_block * kBlockN));
              const FP8Elem* src = reinterpret_cast<const FP8Elem*>(params.v_ptr)
                  + (int64_t)target_start * params.v_row_stride
                  + (int64_t)bidh_kv * params.v_head_stride;
              copy_fp8_tile_rowmajor_to_smem<FP8Elem, kHeadDim>(
                  dst, src, params.v_row_stride, rows_valid, lane,
                  SmemLayoutVt_SW128{});
              copy_packed_sf_tile(
                  sf_dst, params.sf_v_packed_ptr, params.v_block_descale_head_stride,
                  bidh_kv, params.total_pages * params.page_size + target_start,
                  rows_valid, lane);
            }
            __syncwarp();
          };

          auto mVt_page_tma = params.tma_vt_page.get_tma_tensor(
              make_shape(params.page_size, params.d, params.h_k, params.total_pages));
          auto gVt_page_head = mVt_page_tma(_, _, bidh_kv, _);
          auto gVt_page_tiles = local_tile(
              gVt_page_head, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _));
          auto tma_slice_Vt_page = params.tma_vt_page.get_slice(0);
          auto tVPgVt_tma = tma_slice_Vt_page.partition_S(
              gVt_page_tiles(_, _, _, Int<0>{}, _));
          auto tVPsVt_d_0 = tma_slice_Vt_page.partition_D(
              make_tensor(make_smem_ptr(sVt_base[0]), SmemLayoutVt_SW128{}));
          auto tVPsVt_d_1 = tma_slice_Vt_page.partition_D(
              make_tensor(make_smem_ptr(sVt_base[1]), SmemLayoutVt_SW128{}));

          auto load_paged_tma_or_target_v =
              [&](int nb, int stage, uint64_t* v_ready_mbar_ptr) {
            const bool use_paged_history_tma = (kHeadDim > 128) || n_block_paged >= 16;
            if (nb < n_block_paged && use_paged_history_tma) {
              const int page_id = params.page_ids[page_offset + nb];
              const int sf_page_block = (page_id * params.page_size) / kBlockN;
              if (tidx == kNMathThreads + 64) {
                arrive_expect_tx_mbar(v_ready_mbar_ptr, kSmemVtSFVBytes);
                if (stage == 0) {
                  cute::copy(params.tma_vt_page.with(*v_ready_mbar_ptr),
                             tVPgVt_tma(_, _, _, Int<0>{}, page_id), tVPsVt_d_0);
                  cute::copy(params.tma_sfv.with(*v_ready_mbar_ptr),
                             tSFVgSFV_tma(_, _, _, sf_page_block), tSFVsSFV_d_0);
                } else {
                  cute::copy(params.tma_vt_page.with(*v_ready_mbar_ptr),
                             tVPgVt_tma(_, _, _, Int<0>{}, page_id), tVPsVt_d_1);
                  cute::copy(params.tma_sfv.with(*v_ready_mbar_ptr),
                             tSFVgSFV_tma(_, _, _, sf_page_block), tSFVsSFV_d_1);
                }
              }
            } else if (nb >= n_block_paged) {
              const int target_block = nb - n_block_paged;
              const int target_start = binfo.sum_s_k + actual_seqlen_k - actual_seqlen_t
                  + last_page_offset + target_block * kBlockN;
              const int rows_valid = std::max(0, std::min(kBlockN, actual_seqlen_t - target_block * kBlockN));
              const bool use_target_tma =
                  (rows_valid == kBlockN) && (target_start % kBlockN == 0);
              if (use_target_tma) {
                const int nb_abs = target_start / kBlockN;
                const int sf_abs = (params.total_pages * params.page_size + target_start) / kBlockN;
                if (tidx == kNMathThreads + 64) {
                  arrive_expect_tx_mbar(v_ready_mbar_ptr, kSmemVtSFVBytes);
                  if (stage == 0) {
                    cute::copy(params.tma_vt.with(*v_ready_mbar_ptr),
                               tVtgVt_tma(_, _, _, nb_abs), tVtsVt_d_0);
                    cute::copy(params.tma_sfv.with(*v_ready_mbar_ptr),
                               tSFVgSFV_tma(_, _, _, sf_abs), tSFVsSFV_d_0);
                  } else {
                    cute::copy(params.tma_vt.with(*v_ready_mbar_ptr),
                               tVtgVt_tma(_, _, _, nb_abs), tVtsVt_d_1);
                    cute::copy(params.tma_sfv.with(*v_ready_mbar_ptr),
                               tSFVgSFV_tma(_, _, _, sf_abs), tSFVsSFV_d_1);
                  }
                }
              } else {
                copy_paged_or_target_v(nb, stage);
                if (tidx == kNMathThreads + 64) {
                  arrive_mbar(v_ready_mbar_ptr);
                }
              }
            } else {
              copy_paged_or_target_v(nb, stage);
              if (tidx == kNMathThreads + 64) {
                arrive_mbar(v_ready_mbar_ptr);
              }
            }
          };

          for (int n_valid = n_block_max - 1, masking_step_load = 0; n_valid >= n_block_min;
               ++masking_step_load, --n_valid) {
            const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;

            uint64_t* v_empty_mbar_ptr = kv_load_stage ? v_empty_mbar_ptr1 : v_empty_mbar_ptr0;
            int& v_empty_wait_parity = kv_load_stage ? v_empty_wait_parity1 : v_empty_wait_parity0;
            wait_mbar_parity(v_empty_mbar_ptr, (uint32_t)v_empty_wait_parity);
            v_empty_wait_parity ^= 1;

            load_paged_tma_or_target_v(
                nb, kv_load_stage, kv_load_stage ? v_ready_mbar_ptr1 : v_ready_mbar_ptr0);

            if (is_jump && masking_step_load == n_masking_steps - 1)
              n_valid = std::min(n_valid, n_block_history);

            if constexpr (!Kernel_traits::kUseSingleKVStage) {
              kv_load_stage ^= 1;
            }
          }
        } else {
          for (int n_valid = n_block_max - 1, masking_step_load = 0; n_valid >= n_block_min;
               ++masking_step_load, --n_valid) {
            const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;

            uint64_t* v_empty_mbar_ptr = kv_load_stage ? v_empty_mbar_ptr1 : v_empty_mbar_ptr0;
            int& v_empty_wait_parity = kv_load_stage ? v_empty_wait_parity1 : v_empty_wait_parity0;
            wait_mbar_parity(v_empty_mbar_ptr, (uint32_t)v_empty_wait_parity);
            v_empty_wait_parity ^= 1;

            if (tidx == kNMathThreads + 64) {
              const int nb_abs = binfo.sum_s_k / kBlockN + nb;
              if (kv_load_stage == 0) {
                arrive_expect_tx_mbar(v_ready_mbar_ptr0, kSmemVtSFVBytes);
                cute::copy(params.tma_vt.with(*v_ready_mbar_ptr0),  tVtgVt_tma(_, _, _, nb_abs),  tVtsVt_d_0);
                cute::copy(params.tma_sfv.with(*v_ready_mbar_ptr0), tSFVgSFV_tma(_, _, _, nb_abs), tSFVsSFV_d_0);
              } else {
                arrive_expect_tx_mbar(v_ready_mbar_ptr1, kSmemVtSFVBytes);
                cute::copy(params.tma_vt.with(*v_ready_mbar_ptr1),  tVtgVt_tma(_, _, _, nb_abs),  tVtsVt_d_1);
                cute::copy(params.tma_sfv.with(*v_ready_mbar_ptr1), tSFVgSFV_tma(_, _, _, nb_abs), tSFVsSFV_d_1);
              }
            }

            if (is_jump && masking_step_load == n_masking_steps - 1)
              n_valid = std::min(n_valid, n_block_history);

            if constexpr (!Kernel_traits::kUseSingleKVStage) {
              kv_load_stage ^= 1;
            }
          }
        }
      }

      if (is_o_store_warp) {
        if constexpr (Kernel_traits::kUseIndependentOBuffer) {
        // Math warps own compute/softmax and write O to independent SMEM; O-store waits
        // for o_ready[0], then releases o_empty[0] after the TMA store completes.
        wait_mbar_parity(o_ready_mbar_ptr0, (uint32_t)o_ready_wait_parity0);
        o_ready_wait_parity0 ^= 1;
        if (tidx == kNMathThreads + 96) {
          asm volatile("fence.proxy.async.shared::cta;\n" : : : "memory");
          static_assert(
              Kernel_traits::kSmemWsOBytes >= kBlockM * kHeadDim * (int)sizeof(OutElement),
              "Independent O SMEM buffer is too small.");
          char* smem_o = reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsOOffset;
          using SmemLayoutO_TMA_t = cute::Layout<
              cute::Shape<cute::Int<kBlockM>, cute::Int<kHeadDim>>,
              cute::Stride<cute::Int<kHeadDim>, cute::_1>>;
          Tensor sO_tma = make_tensor(
              make_smem_ptr(reinterpret_cast<OutElement*>(smem_o)),
              SmemLayoutO_TMA_t{});
          auto mO_tma   = params.tma_o.get_tma_tensor(make_shape(params.total_q, params.d, params.h));
          auto gO_head  = mO_tma(_, _, bidh);
          auto gO_tiles = local_tile(gO_head, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(_, _));
          const int m_abs = binfo.sum_s_q / kBlockM + m_block;
          auto tma_slice_O = params.tma_o.get_slice(0);
          Tensor tOsO     = tma_slice_O.partition_S(sO_tma);
          Tensor tOgO_all = tma_slice_O.partition_D(gO_tiles(_, _, _, Int<0>{}));
          cute::copy(params.tma_o, tOsO, tOgO_all(_, _, _, m_abs));
          cute::tma_store_arrive();
          cute::tma_store_wait<0>();
          arrive_mbar(o_empty_mbar_ptr0);
        }
        }
      }
    };

    // Static tile-id broadcast: load and math paths run the same deterministic
    // scheduler and decode the same tile ids locally.  There is no dynamic work
    // queue, atomic counter, or shared tile-id sync on this path.
    const bool head_shared_rab = Has_rab && params.h_rab == 1 && params.h > 1;
    if constexpr (Use_paired_persistent) {
      const int num_m_block_persistent = (params.seqlen_q + kBlockM - 1) / kBlockM;
      const int total_tiles_persistent = num_m_block_persistent * params.h * params.b;
      const int total_tile_pairs_persistent = (total_tiles_persistent + 1) / 2;
      #pragma unroll 1
      for (int tile_pair = int(blockIdx.x); tile_pair < total_tile_pairs_persistent; tile_pair += int(gridDim.x)) {
        const int paired_tile = total_tiles_persistent - 1 - tile_pair;
        const int tiles_this_pair = paired_tile == tile_pair ? 1 : 2;

        #pragma unroll 1
        for (int pair_slot = 0; pair_slot < tiles_this_pair; ++pair_slot) {
          const int tile = pair_slot == 0 ? tile_pair : paired_tile;
          const HstuWsTileCoord coord =
              hstu_ws_decode_tile(tile, num_m_block_persistent, params.h, head_shared_rab);
          run_load_tile(coord.bidb, coord.bidh, coord.m_block);
        }
      }
    } else if constexpr (Use_full_persistent) {
      const int num_m_block_persistent = (params.seqlen_q + kBlockM - 1) / kBlockM;
      const int total_tiles_persistent = num_m_block_persistent * params.h * params.b;
      #pragma unroll 1
      for (int tile = int(blockIdx.x); tile < total_tiles_persistent; tile += int(gridDim.x)) {
        const HstuWsTileCoord coord =
            hstu_ws_decode_tile(tile, num_m_block_persistent, params.h, head_shared_rab);
        run_load_tile(coord.bidb, coord.bidh, coord.m_block);
      }
    } else {
      run_load_tile(bidb_arg, bidh_arg, m_block_arg);
    }
    // Load warp path exits here.  Active load warp has also completed O TMA-store.

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
    constexpr bool Has_rab      = Kernel_traits::Has_rab;
    constexpr bool Paged_KV     = Kernel_traits::Paged_KV;
    constexpr int  kBlockM      = Kernel_traits::kBlockM;
    constexpr int  kBlockN      = Kernel_traits::kBlockN;
    constexpr int  kHeadDim     = Kernel_traits::kHeadDim;
    using RabElement = cutlass::bfloat16_t;

    const int tidx_math = tidx;  // math warps: tidx_math == tidx (0-255)
    int tma_parity0 = 0;  // Persistent K/V mbarrier parity for stage 0.
    int tma_parity1 = 0;  // Persistent K/V mbarrier parity for stage 1.
    int q_tma_parity = 0;
    int o_empty_wait_parity = 0;
    int rab_ready_wait_parity0 = 0;
    int math_stage  = 0;

    if constexpr (Use_persistent) {
      __syncthreads();  // One-time S1: load warp has initialized K/V mbarriers.
    }

    auto run_math_tile = [&](const int bidb, const int bidh, int m_block) {
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
      const int last_page_seqlen       = Paged_KV ? binfo.last_page_seqlen : kBlockN;

      const bool is_jump             = Is_target && m_block * kBlockM + actual_seqlen_offset > actual_seqlen_h;
      const bool is_in_target        = Is_target && (m_block + 1) * kBlockM + actual_seqlen_offset > actual_seqlen_h;
      const bool is_in_context       = Is_context && (m_block + 1) * kBlockM <= actual_seqlen_c;
      const bool is_in_mixed_context = Is_context &&
          (m_block + 1) * kBlockM > actual_seqlen_c && m_block * kBlockM < actual_seqlen_c;
      const bool is_in_paged_target  = is_in_target && Paged_KV;
      const int last_page_offset     = is_in_paged_target ? kBlockN - last_page_seqlen : 0;

      const int n_block_history = cute::ceil_div(actual_seqlen_h, kBlockN);
      const int target_index    = (m_block * kBlockM - actual_seqlen_h) / params.target_group_size;
      const int n_block_paged   = Paged_KV ? n_block_history : 0;
      const int n_block_target  = cute::ceil_div(actual_seqlen_t, kBlockN);

      int n_block_min = !Is_local ? 0
          : std::max(0, (m_block * kBlockM + actual_seqlen_offset - params.window_size_left) / kBlockN);
      int n_block_max = Paged_KV ? n_block_history + n_block_target : cute::ceil_div(actual_seqlen_k, kBlockN);
      if constexpr (Is_causal || Is_local) {
        int offset = (m_block + 1) * kBlockM + actual_seqlen_offset + params.window_size_right;
        if (is_in_paged_target) offset += last_page_offset;
        n_block_max = std::min(n_block_max, cute::ceil_div(offset, kBlockN));
      }
      if constexpr (Is_context) {
        n_block_min = (is_in_context || is_in_mixed_context) ? 0 : n_block_min;
        n_block_max = (is_in_context || is_in_mixed_context)
            ? std::max(n_block_history, n_block_max) : n_block_max;
      }

      int n_masking_block_max = cute::ceil_div(
          std::min(actual_seqlen_k + last_page_offset,
                   (m_block + 1) * kBlockM + actual_seqlen_offset + last_page_offset),
          kBlockN);
      int n_masking_block_min = (m_block * kBlockM + actual_seqlen_offset) / kBlockN;
      if constexpr (Is_target) {
        n_masking_block_min = is_jump
            ? (actual_seqlen_h + actual_seqlen_offset + target_index * params.target_group_size + last_page_offset) / kBlockN
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

      // Early exit 2 (after Is_arbitrary): math warps write zeros; non-persistent paths still
      // consume the per-tile S1.
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
        if constexpr (!Use_persistent) {
          __syncthreads();  // S1
        }
        return;
      }

      // WS SMEM layouts. Kernel_traits selects SW32/SW64/SW128 by headDim.
      using SmemLayoutQ_SW128  = typename Kernel_traits::SmemLayoutQ_TMA;
      using SmemLayoutK_SW128  = typename Kernel_traits::SmemLayoutK_TMA;

      constexpr int kSmemKVElems = kBlockN * kHeadDim;

      // smem_base32: shared-memory base address as uint32 for on-demand barrier/KV
      // pointer computation in the main loop — replaces 6 persistent pointer arrays.
      const uint32_t smem_base32 =
          static_cast<uint32_t>(__cvta_generic_to_shared(smem_));
      uint64_t* q_ready_mbar_ptr = reinterpret_cast<uint64_t*>(
          reinterpret_cast<char*>(smem_) + kSmemMbar0Offset + 64);
      uint64_t* q_empty_mbar_ptr = reinterpret_cast<uint64_t*>(
          reinterpret_cast<char*>(smem_) + kSmemMbar0Offset + 72);

      // SF SMEM pointers.
      static constexpr int kSmemSFOffset_WS = Kernel_traits::kSmemWsDataSizePadded;
      int32_t* smem_sfa_ptr = reinterpret_cast<int32_t*>(smem_ + kSmemSFOffset_WS);
      int32_t* smem_sfp_ptr = smem_sfa_ptr + kBlockM;

      using SmemLayoutSFA = typename BS1::SmemLayoutSFA;
      using SmemLayoutSFB = typename BS1::SmemLayoutSFB;
      using SmemLayoutSFV = typename BS2::SmemLayoutSFB;
      Tensor sSFA_ = make_tensor(make_smem_ptr(smem_sfa_ptr), SmemLayoutSFA{});
      auto sSFA = as_position_independent_swizzle_tensor(sSFA_);
      Tensor sSFP_ = make_tensor(make_smem_ptr(smem_sfp_ptr), SmemLayoutSFA{});
      auto sSFP = as_position_independent_swizzle_tensor(sSFP_);
      using SmemLayoutRab_TMA = typename Kernel_traits::SmemLayoutRab_TMA;
      RabElement* smem_rab = reinterpret_cast<RabElement*>(
          reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsRabOffset);
      Tensor sRab = make_tensor(make_smem_ptr(smem_rab), SmemLayoutRab_TMA{});

      // MMA tiled objects (use tidx_math = tidx for math warps).
      typename BS1::TiledMma tiled_mma_g1;
      auto thr_mma_g1 = tiled_mma_g1.get_thread_slice(tidx_math);
      typename BS2::TiledMma tiled_mma_g2;
      auto thr_mma_g2 = tiled_mma_g2.get_thread_slice(tidx_math);

      Tensor acc_o = partition_fragment_C(tiled_mma_g2, Shape<Int<kBlockM>, Int<kHeadDimGemm2>>{});
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
      auto s2r_copy_B_g1 = make_tiled_copy_B(typename BS1::SmemCopyAtomB{}, tiled_mma_g1);
      auto s2r_thr_B_g1  = s2r_copy_B_g1.get_thread_slice(tidx_math);

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
      auto load_a_z_pattern = [&](auto&& sA_pi, auto& tCrA, int k_block_base, int k_block_count, int row_stride) {
        const int lane    = tidx_math & 31;
        const int warp_m  = tidx_math / 32;
        const int mat_num = lane >> 3;    // matrix index (0..3) this lane provides addr for
        const int mat_row = lane & 7;     // row within that matrix (0..7)
        const int M_abs   = warp_m * 16 + ((mat_num & 1) << 3) + mat_row;
        const int K_half  = mat_num >> 1; // 0 = K low 16, 1 = K high 16
        const uint32_t smem_base =
            static_cast<uint32_t>(__cvta_generic_to_shared(&sA_pi(0, 0)));
        const uint32_t row_base = smem_base + static_cast<uint32_t>(M_abs * row_stride);
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
      auto load_b_z_pattern = [&](auto&& sB_pi, auto& tCrB, int k_block_base, int k_block_count, int row_stride) {
        const int lane    = tidx_math & 31;
        const int mat_num = lane >> 3;
        const int mat_row = lane & 7;
        const uint32_t smem_base =
            static_cast<uint32_t>(__cvta_generic_to_shared(&sB_pi(0, 0)));
        auto tXrB = recast<uint32_t>(tCrB);
        constexpr int kNGroups = kBlockN / 32;  // 2 for BN64, 4 for BN128
        CUTE_UNROLL
        for (int kb = 0; kb < k_block_count; ++kb) {
          CUTE_UNROLL
          for (int g = 0; g < kNGroups; ++g) {
            const uint32_t N_row = static_cast<uint32_t>(g * 32 + mat_num * 8 + mat_row);
            CUTE_UNROLL
            for (int K_half = 0; K_half < 2; ++K_half) {
              const int K_start  = (k_block_base + kb) * 32 + K_half * 16;
              const uint32_t addr =
                  smem_base + N_row * row_stride + static_cast<uint32_t>(K_start ^ (mat_row << 4));
              if constexpr (kBlockN == 64) {
                const int frag_base64 = 16 * kb + 8 * g;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                    : "=r"(tXrB(frag_base64 + K_half + 0)),
                      "=r"(tXrB(frag_base64 + K_half + 2)),
                      "=r"(tXrB(frag_base64 + K_half + 4)),
                      "=r"(tXrB(frag_base64 + K_half + 6))
                    : "r"(addr));
              } else {
                const int frag_base = 32 * kb + 2 * g;
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
        }
      };

      auto load_a_z_pattern_layout = [&](FP8Elem* smem_ptr, auto layout, auto& tCrA,
                                         int k_block_base, int k_block_count) {
        const int lane    = tidx_math & 31;
        const int warp_m  = tidx_math / 32;
        const int mat_num = lane >> 3;
        const int mat_row = lane & 7;
        const int M_abs   = warp_m * 16 + ((mat_num & 1) << 3) + mat_row;
        const int K_half  = mat_num >> 1;
        const uint32_t smem_base =
            static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
        auto tXrA = recast<uint32_t>(tCrA);
        CUTE_UNROLL
        for (int kb = 0; kb < k_block_count; ++kb) {
          const int K_start = (k_block_base + kb) * 32 + (K_half << 4);
          const uint32_t addr =
              smem_base + (uint32_t)layout(M_abs, K_start);
          asm volatile(
              "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
              : "=r"(tXrA(4*kb+0)), "=r"(tXrA(4*kb+1)),
                "=r"(tXrA(4*kb+2)), "=r"(tXrA(4*kb+3))
              : "r"(addr));
        }
      };

      auto load_b_z_pattern_layout = [&](FP8Elem* smem_ptr, auto layout, auto& tCrB,
                                         int k_block_base, int k_block_count) {
        const int lane    = tidx_math & 31;
        const int mat_num = lane >> 3;
        const int mat_row = lane & 7;
        const uint32_t smem_base =
            static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
        auto tXrB = recast<uint32_t>(tCrB);
        constexpr int kNGroups = kBlockN / 32;
        CUTE_UNROLL
        for (int kb = 0; kb < k_block_count; ++kb) {
          CUTE_UNROLL
          for (int g = 0; g < kNGroups; ++g) {
            const uint32_t N_row = static_cast<uint32_t>(g * 32 + mat_num * 8 + mat_row);
            CUTE_UNROLL
            for (int K_half = 0; K_half < 2; ++K_half) {
              const int K_start = (k_block_base + kb) * 32 + K_half * 16;
              const uint32_t addr =
                  smem_base + (uint32_t)layout((int)N_row, K_start);
              if constexpr (kBlockN == 64) {
                const int frag_base64 = 16 * kb + 8 * g;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                    : "=r"(tXrB(frag_base64 + K_half + 0)),
                      "=r"(tXrB(frag_base64 + K_half + 2)),
                      "=r"(tXrB(frag_base64 + K_half + 4)),
                      "=r"(tXrB(frag_base64 + K_half + 6))
                    : "r"(addr));
              } else {
                const int frag_base = 32 * kb + 2 * g;
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
        }
      };

      if constexpr (!Use_persistent) {
        __syncthreads();  // S1: tma_mbar0/1 and math_mbar0/1 visible to all warps.
      }

      FP8Elem* q_persist_base = reinterpret_cast<FP8Elem*>(
          reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsQPersistOffset);
      Tensor sQ_persist = make_tensor(
          make_smem_ptr(q_persist_base),
          SmemLayoutQ_SW128{});
      auto sQ_persist_pi = as_position_independent_swizzle_tensor(sQ_persist);

      // Q+SFA TMA is issued by load warp 8 into sQ_persist and smem_sfa_ptr.
      // Math waits q_ready, loads Q/SFA into registers, then releases q_empty.
      wait_mbar_parity(q_ready_mbar_ptr, (uint32_t)q_tma_parity);
      q_tma_parity ^= 1;

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
          int col             = block_col + base_col;
          if (Paged_KV && row >= actual_seqlen_h) {
            col -= last_page_offset;
          }
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
              if (row >= actual_seqlen_h &&
                  (col + (Paged_KV ? last_page_offset : 0)) >= actual_seqlen_h &&
                  col < tgt_col_lft)
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

      auto add_rab_bs = [&](auto& tSrS, int nb) {
        if constexpr (Has_rab) {
          static constexpr int Row = 0, Col = 1;
          Tensor cS   = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
          Tensor tScS = thr_mma_g1.partition_C(cS);
          const int bidh_rab = (params.h_rab > 1) ? bidh : 0;
          const size_t rab_offset = bidb * params.rab_seqlen_qk_stride
              + bidh_rab * params.rab_seqlen_q_stride
              + params.seqlen_k_rounded * actual_seqlen_offset;
          const RabElement* rab_ptr =
              reinterpret_cast<const RabElement*>(params.rab_ptr) + rab_offset;
          const int base_col = nb * kBlockN;

          CUTE_UNROLL
          for (int flat = 0; flat < size(tSrS); ++flat) {
            const auto coord = tScS(flat);
            const int block_row = int(get<Row>(coord));
            const int q_idx = m_block * kBlockM + block_row;
            if (q_idx >= actual_seqlen_q) {
              continue;
            }
            const int block_col = int(get<Col>(coord));
            int col = block_col + base_col;
            if constexpr (Paged_KV && Is_target) {
              if (nb >= n_block_paged) {
                col -= last_page_offset;
              }
            }
            if constexpr (Paged_KV && Is_target) {
              if (nb < n_block_paged && col >= actual_seqlen_h) {
                continue;
              }
            }
            if (0 <= col && col < actual_seqlen_k) {
              tSrS(flat) += static_cast<float>(
                  rab_ptr[q_idx * params.rab_seqlen_k_stride + col]);
            }
          }
        }
      };

      auto add_rab_bs_skip_masked = [&](auto& tSrS, int nb) {
        if constexpr (Has_rab) {
          static constexpr int Row = 0, Col = 1;
          Tensor cS   = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
          Tensor tScS = thr_mma_g1.partition_C(cS);
          const int bidh_rab = (params.h_rab > 1) ? bidh : 0;
          const size_t rab_offset = bidb * params.rab_seqlen_qk_stride
              + bidh_rab * params.rab_seqlen_q_stride
              + params.seqlen_k_rounded * actual_seqlen_offset;
          const RabElement* rab_ptr =
              reinterpret_cast<const RabElement*>(params.rab_ptr) + rab_offset;
          const int base_col = nb * kBlockN;

          CUTE_UNROLL
          for (int flat = 0; flat < size(tSrS); ++flat) {
            if (tSrS(flat) == -INFINITY) {
              continue;
            }
            const auto coord = tScS(flat);
            const int block_row = int(get<Row>(coord));
            const int q_idx = m_block * kBlockM + block_row;
            if (q_idx >= actual_seqlen_q) {
              continue;
            }
            const int block_col = int(get<Col>(coord));
            int col = block_col + base_col;
            if constexpr (Paged_KV && Is_target) {
              if (nb >= n_block_paged) {
                col -= last_page_offset;
              }
            }
            if constexpr (Paged_KV && Is_target) {
              if (nb < n_block_paged && col >= actual_seqlen_h) {
                continue;
              }
            }
            if (0 <= col && col < actual_seqlen_k) {
              tSrS(flat) += static_cast<float>(
                  rab_ptr[q_idx * params.rab_seqlen_k_stride + col]);
            }
          }
        }
      };

      auto use_rab_smem_for_nb = [&](int nb) {
        if constexpr (!Has_rab) {
          return false;
        } else {
          bool aligned = (actual_seqlen_offset % kBlockM) == 0;
          if constexpr (Paged_KV && Is_target) {
            if (nb >= n_block_paged && last_page_offset != 0) {
              aligned = false;
            }
          }
          return aligned;
        }
      };

      auto add_rab_smem_bs = [&](auto& tSrS, int nb, bool skip_masked, int stage) {
        if constexpr (Has_rab) {
          static constexpr int Row = 0, Col = 1;
          Tensor cS   = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
          Tensor tScS = thr_mma_g1.partition_C(cS);
          const int base_col = nb * kBlockN;

          CUTE_UNROLL
          for (int flat = 0; flat < size(tSrS); ++flat) {
            if (skip_masked && tSrS(flat) == -INFINITY) {
              continue;
            }
            const auto coord = tScS(flat);
            const int block_row = int(get<Row>(coord));
            const int q_idx = m_block * kBlockM + block_row;
            if (q_idx >= actual_seqlen_q) {
              continue;
            }
            const int block_col = int(get<Col>(coord));
            int col = block_col + base_col;
            if constexpr (Paged_KV && Is_target) {
              if (nb >= n_block_paged) {
                col -= last_page_offset;
              }
            }
            if constexpr (Paged_KV && Is_target) {
              if (nb < n_block_paged && col >= actual_seqlen_h) {
                continue;
              }
            }
            if (0 <= col && col < actual_seqlen_k) {
              tSrS(flat) += static_cast<float>(sRab(block_row, block_col, stage));
            }
          }
        }
      };

      auto consume_rab_bs = [&](auto& tSrS, int nb, bool skip_masked) {
        if constexpr (Has_rab) {
          if (use_rab_smem_for_nb(nb)) {
            wait_mbar_parity(
                smem_base32 + (uint32_t)kSmemMbar0Offset + 112u,
                (uint32_t)rab_ready_wait_parity0);
            rab_ready_wait_parity0 ^= 1;
            asm volatile("" ::: "memory");
            add_rab_smem_bs(tSrS, nb, skip_masked, 0);
            if ((tidx_math & 31) == 0) {
              arrive_mbar(smem_base32 + (uint32_t)kSmemMbar0Offset + 120u);
            }
          } else if (skip_masked) {
            add_rab_bs_skip_masked(tSrS, nb);
          } else {
            add_rab_bs(tSrS, nb);
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
      const int warp_m  = tidx_math / 32;
      const int lane    = tidx_math & 31;
      const int sfa_row = warp_m * 16 + 8 * (lane & 1) + (lane >> 2);
      tCrSFA(0, 0, 0)  = smem_sfa_ptr[sfa_row];
      auto tCrSFA_frg = BS1::transform_fragment_for_qmma(tCrSFA);

      Tensor tCrQ = thr_mma_g1.partition_fragment_A(sQ_persist_pi);
      if constexpr (kHeadDim <= 128) {
        clear(tCrQ);
        load_a_z_pattern_layout(q_persist_base, SmemLayoutQ_SW128{}, tCrQ, 0, kHeadDim / 32);
      }
      if constexpr (kHeadDim <= 128) {
        if ((tidx_math & 31) == 0) {
          arrive_mbar(q_empty_mbar_ptr);
        }
      }

      // Per-tile lambda (Opt C): instantiated with kIsMasking=true (Phase 1, causal diagonal)
      // and kIsMasking=false (Phase 2, steady-state unmasked) to allow compile-time dead-code
      // elimination of apply_mask_bs in the hot path.  n_valid_ref is modified in-place by
      // the is_jump adjustment; math_stage persists across tiles and flips after every N tile.
      constexpr bool kStoreOInMainloop =
          kHeadDim <= 128 &&
          Is_causal && !Is_target && !Is_context && !Is_arbitrary && !Is_local;
      bool o_epilogue_done = false;
      auto store_o_epilogue = [&]() {
        for (int i = 0; i < size(acc_o); ++i) acc_o(i) /= params.scaling_seqlen;
        using OutElement = typename Kernel_traits::OutputType;
        Tensor rO = make_tensor_like<OutElement>(acc_o);
        flash::convert_type_safe(acc_o, rO);

        if constexpr (!Kernel_traits::kUseIndependentOBuffer) {
          Tensor cO_id = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimGemm2>>{});
          Tensor tOcO = thr_mma_g2.partition_C(cO_id);
          OutElement* gO_ptr = reinterpret_cast<OutElement*>(params.o_ptr)
              + binfo.q_offset(params.o_row_stride)
              + bidh * params.o_head_stride;
          const int valid_rows = actual_seqlen_q - m_block * kBlockM;
          CUTE_UNROLL
          for (int flat = 0; flat < size(rO); ++flat) {
            const auto coord = tOcO(flat);
            const int m_rel = int(get<0>(coord));
            const int n_pos = int(get<1>(coord));
            if (m_rel < valid_rows && n_pos < kHeadDim) {
              gO_ptr[(m_block * kBlockM + m_rel) * params.o_row_stride + n_pos] = rO(flat);
            }
          }
        } else {
          static_assert(
              Kernel_traits::kSmemWsOBytes >= kBlockM * kHeadDim * (int)sizeof(OutElement),
              "Independent O SMEM buffer is too small.");
          char* smem_o = reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsOOffset;

          // Wait until the independent O buffer is no longer owned by the previous TMA store.
          wait_mbar_parity(
              smem_base32 + (uint32_t)kSmemMbar0Offset + 96u,
              (uint32_t)o_empty_wait_parity);
          o_empty_wait_parity ^= 1;
          Tensor sO_flat = make_tensor(
              make_smem_ptr(reinterpret_cast<OutElement*>(smem_o)),
              Layout<Shape<Int<kBlockM>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>{});
          {
          // stmatrix.sync.aligned.x4.m8n8.shared.b16: 8 warp-cooperative stores replace 32 STS.32.
          //
          // Layout invariants this asm relies on:
          //   AtomLayout <_8,_1,_1>  -> 8 math warps all in M direction, warp covers rows [warp_m*16, warp_m*16+15]
          //   PermMmaTileN <_8,_4,_4>:<_1,_32,_8>  -> N_sorted[j] = 32*(j%4) + 8*(j/4) for j=0..15
          //   C-atom SM80_16x8_Row  -> rO_u32[2*j + m_grp] covers (M=warp_m*16+m_grp*8+lq, N=N_sorted[j]+lqt*2)
          //   OutElement is 2-byte (BF16); sO_flat row-stride = kHeadDim * sizeof(OutElement) = 256 bytes.
          static_assert(sizeof(OutElement) == 2, "stmatrix.m8n8.b16 requires 2-byte elements");

          auto rO_u32 = recast<uint32_t>(rO);

          const int lane        = tidx_math & 31;
          const int lq          = lane >> 2;   // lane-quad (0..7): selects data row within 8x8 matrix
          const int lqt         = lane & 3;    // lane-quad-thread (0..3): selects data col-group
          const int warp_m      = tidx_math >> 5;  // M-warp index (0..7)
          const int addr_row    = ((lq & 1) << 2) | lqt;  // STSM address-row within matrix (0..7)
          const int mat_in_lane = lq >> 1;                 // which of 4 matrices this thread addresses

          const uint32_t smem_base  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_o));
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
              // N_sorted[n_grp + k*4] = 32*(n_grp+k*4)%4 + 8*(n_grp+k*4)/4 = 32*n_grp + 8*k.
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

          // For partial tiles (last tile of a varlen sequence): zero SMEM rows [valid_rows, kBlockM)
          // so TMA bulk store does not write garbage to GMEM beyond actual_seqlen_q.
          const int valid_rows = actual_seqlen_q - m_block * kBlockM;
          if (valid_rows < kBlockM) {
            // Ensure all STSM stores are complete before any warp zeros OOB rows.
            // Full tiles skip this: o_ready is initialized with count=8, so the O-store warp
            // cannot proceed until each math warp has arrived after its own STSM stores.
            asm volatile("bar.sync 1, 256;\n" : : : "memory");
            const int oob_elems = (kBlockM - valid_rows) * kHeadDim;
            OutElement* sO_raw = reinterpret_cast<OutElement*>(smem_o) + valid_rows * kHeadDim;
            for (int i = tidx_math; i < oob_elems; i += kNMathThreads)
              sO_raw[i] = OutElement(0);
            asm volatile("bar.sync 1, 256;\n" : : : "memory");  // OOB zeros visible before TMA.
          }

          // Produce o_ready[0]. O-store warp owns the TMA store and releases
          // o_empty[0] after the store completes.
          if ((tidx_math & 31) == 0) {
            const uint32_t o_ready_addr =
                smem_base32 + (uint32_t)kSmemMbar0Offset + 80u;
            arrive_mbar(o_ready_addr);
          }
        }
      };

      auto run_n_tile = [&](int nb, int n_valid_ref, int masking_step, bool do_mask_runtime,
                            bool store_o_after, auto kMaskMode_c) {
        constexpr int kMaskMode = decltype(kMaskMode_c)::value;
        constexpr bool kIsMasking = kMaskMode == 1;
        constexpr bool kRuntimeMasking = kMaskMode == 2;

        // Stage-dependent SMEM pointers — pure pointer arithmetic, no TMA dependency.
        int32_t* smem_sfb_cur = smem_sfa_ptr + 2 * kBlockM + math_stage * kBlockN;
        FP8Elem* sK_cur  = reinterpret_cast<FP8Elem*>(smem_q) + math_stage * 2 * kSmemKVElems;
        FP8Elem* sVt_cur = reinterpret_cast<FP8Elem*>(smem_q) + kSmemKVElems + math_stage * 2 * kSmemKVElems;

        // GEMM1 only needs K/SFB. Wait V/SFV later, right before GEMM2.
        {
          const int cur_parity = math_stage ? tma_parity1 : tma_parity0;
          uint32_t kaddr = smem_base32 + (uint32_t)kSmemMbar0Offset + (uint32_t)(math_stage * 8);
          wait_mbar_parity(kaddr, (uint32_t)cur_parity);
        }
        // mbarrier.test_wait already guarantees K/SFB data visibility in SMEM.
        // K and V have separate empty counters: K is released after K/SFB s2r,
        // V is waited/released later around V/SFV s2r.
        asm volatile("" ::: "memory");

        Tensor sSFB_ = make_tensor(make_smem_ptr(smem_sfb_cur), SmemLayoutSFB{});
        auto sSFB = as_position_independent_swizzle_tensor(sSFB_);

        // GEMM1: acc_s += Q × K^T (block-scaled QMMA).
        // Full-K=128 copy: make_tiled_copy_A/B expects the full TiledMMA tile;
        // cute::gemm internally loops 4×K=32 when the fragment covers K=128.
        Tensor acc_s = partition_fragment_C(tiled_mma_g1, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);

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
        if constexpr (kHeadDim <= 128) {
          auto sK_cur_pi = as_position_independent_swizzle_tensor(
              make_tensor(make_smem_ptr(sK_cur), SmemLayoutK_SW128{}));
          auto tCrSFB_frg = BS1::transform_fragment_for_qmma(tCrSFB);
          Tensor tCrK = thr_mma_g1.partition_fragment_B(sK_cur_pi);
          clear(tCrK);
          // GEMM1: tCrQ and tCrSFA_frg are N-loop invariants hoisted above the loop.
          if constexpr (kBlockN == 64) {
            auto tXsK = s2r_thr_B_g1.partition_S(sK_cur_pi);
            auto tXrK = s2r_thr_B_g1.retile_D(tCrK);
            cute::copy(s2r_copy_B_g1, tXsK, tXrK);
          } else {
            load_b_z_pattern(sK_cur_pi, tCrK, 0, kHeadDim / 32, kHeadDim);
          }
          if ((tidx_math & 31) == 0) {
            uint32_t k_empty_addr =
                smem_base32 + (uint32_t)kSmemMbar0Offset + 32u + (uint32_t)(math_stage * 8);
            arrive_mbar(k_empty_addr);
          }
          cute::gemm(tiled_mma_g1,
              make_zip_tensor(tCrQ, tCrSFA_frg(_,_,_,_0{})),
              make_zip_tensor(tCrK, tCrSFB_frg(_,_,_,_0{})),
              acc_s);
        } else if constexpr (kHeadDim == 256) {
          auto replicate_e8m0_lane = [](int32_t packed, int lane) {
            const uint32_t byte = (static_cast<uint32_t>(packed) >> (8 * lane)) & 0xffu;
            return static_cast<int32_t>(
                byte | (byte << 8) | (byte << 16) | (byte << 24));
          };

          using SmemLayoutQChunk_SW128 = decltype(tile_to_shape(
              GMMA::Layout_K_SW128_Atom<FP8Elem>{},
              Shape<Int<kBlockM>, Int<128>>{}));
          using SmemLayoutKChunk_SW128 = decltype(tile_to_shape(
              GMMA::Layout_K_SW128_Atom<FP8Elem>{},
              Shape<Int<kBlockN>, Int<128>>{}));
          FP8Elem* q_persist_base = reinterpret_cast<FP8Elem*>(
              reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsQPersistOffset);

          {
            Tensor tCrSFA0 = make_tensor_like<int32_t>(tCrSFA);
            Tensor tCrSFB0 = make_tensor_like<int32_t>(tCrSFB);
            CUTE_UNROLL
            for (int i = 0; i < size(tCrSFA); ++i) {
              tCrSFA0(i) = replicate_e8m0_lane(tCrSFA(i), 0);
            }
            CUTE_UNROLL
            for (int i = 0; i < size(tCrSFB); ++i) {
              tCrSFB0(i) = replicate_e8m0_lane(tCrSFB(i), 0);
            }
            auto tCrSFA0_frg = BS1::transform_fragment_for_qmma(tCrSFA0);
            auto tCrSFB0_frg = BS1::transform_fragment_for_qmma(tCrSFB0);

            Tensor sQ_chunk0 = make_tensor(
                make_smem_ptr(q_persist_base),
                SmemLayoutQChunk_SW128{});
            auto sQ_chunk0_pi = as_position_independent_swizzle_tensor(sQ_chunk0);
            Tensor tCrQ0 = thr_mma_g1.partition_fragment_A(sQ_chunk0_pi);
            clear(tCrQ0);
            load_a_z_pattern_layout(q_persist_base, SmemLayoutQ_SW128{}, tCrQ0, 0, 4);

            Tensor sK_chunk0 = make_tensor(
                make_smem_ptr(sK_cur),
                SmemLayoutKChunk_SW128{});
            auto sK_chunk0_pi = as_position_independent_swizzle_tensor(sK_chunk0);
            Tensor tCrK0 = thr_mma_g1.partition_fragment_B(sK_chunk0_pi);
            clear(tCrK0);
            load_b_z_pattern_layout(sK_cur, SmemLayoutK_SW128{}, tCrK0, 0, 4);

            cute::gemm(tiled_mma_g1,
                make_zip_tensor(tCrQ0, tCrSFA0_frg(_,_,_,_0{})),
                make_zip_tensor(tCrK0, tCrSFB0_frg(_,_,_,_0{})),
                acc_s);
          }

          {
            Tensor tCrSFA1 = make_tensor_like<int32_t>(tCrSFA);
            Tensor tCrSFB1 = make_tensor_like<int32_t>(tCrSFB);
            CUTE_UNROLL
            for (int i = 0; i < size(tCrSFA); ++i) {
              tCrSFA1(i) = replicate_e8m0_lane(tCrSFA(i), 1);
            }
            CUTE_UNROLL
            for (int i = 0; i < size(tCrSFB); ++i) {
              tCrSFB1(i) = replicate_e8m0_lane(tCrSFB(i), 1);
            }
            auto tCrSFA1_frg = BS1::transform_fragment_for_qmma(tCrSFA1);
            auto tCrSFB1_frg = BS1::transform_fragment_for_qmma(tCrSFB1);

            Tensor sQ_chunk1 = make_tensor(
                make_smem_ptr(q_persist_base),
                SmemLayoutQChunk_SW128{});
            auto sQ_chunk1_pi = as_position_independent_swizzle_tensor(sQ_chunk1);
            Tensor tCrQ1 = thr_mma_g1.partition_fragment_A(sQ_chunk1_pi);
            clear(tCrQ1);
            load_a_z_pattern_layout(q_persist_base, SmemLayoutQ_SW128{}, tCrQ1, 4, 4);

            Tensor sK_chunk1 = make_tensor(
                make_smem_ptr(sK_cur),
                SmemLayoutKChunk_SW128{});
            auto sK_chunk1_pi = as_position_independent_swizzle_tensor(sK_chunk1);
            Tensor tCrK1 = thr_mma_g1.partition_fragment_B(sK_chunk1_pi);
            clear(tCrK1);
            load_b_z_pattern_layout(sK_cur, SmemLayoutK_SW128{}, tCrK1, 4, 4);

            cute::gemm(tiled_mma_g1,
                make_zip_tensor(tCrQ1, tCrSFA1_frg(_,_,_,_0{})),
                make_zip_tensor(tCrK1, tCrSFB1_frg(_,_,_,_0{})),
                acc_s);
          }

          if ((tidx_math & 31) == 0) {
            uint32_t k_empty_addr =
                smem_base32 + (uint32_t)kSmemMbar0Offset + 32u + (uint32_t)(math_stage * 8);
            arrive_mbar(k_empty_addr);
          }
        }
        constexpr bool kUseRabSkipMasked =
            (Is_causal && !Is_context && !Is_target && !Is_local && !Is_arbitrary) ||
            Is_local || Is_arbitrary || (Is_context && !Paged_KV) || (Is_target && Paged_KV);
        if constexpr (!(Has_rab && kUseRabSkipMasked)) {
          consume_rab_bs(acc_s, nb, false);
        }
        if (params.debug_gemm1_only) {
          if constexpr (Has_rab && kUseRabSkipMasked) {
            consume_rab_bs(acc_s, nb, false);
          }
          for (int i = 0; i < size(acc_s); ++i) acc_o(i) += acc_s(i);
          {
            const int cur_parity = math_stage ? tma_parity1 : tma_parity0;
            uint32_t vaddr =
                smem_base32 + (uint32_t)kSmemMbar0Offset + 16u + (uint32_t)(math_stage * 8);
            wait_mbar_parity(vaddr, (uint32_t)cur_parity);
          }
          if (math_stage) { tma_parity1 ^= 1; } else { tma_parity0 ^= 1; }
          asm volatile("" ::: "memory");
          if ((tidx_math & 31) == 0) {
            uint32_t v_empty_addr =
                smem_base32 + (uint32_t)kSmemMbar0Offset + 48u + (uint32_t)(math_stage * 8);
            arrive_mbar(v_empty_addr);
          }
          if (is_jump && masking_step == n_masking_steps - 1)
            n_valid_ref = std::min(n_valid_ref, n_block_history);
          if constexpr (!Kernel_traits::kUseSingleKVStage) {
            math_stage ^= 1;
          }
          return n_valid_ref;
        }

        // Masking (Opt C: compile-time specialized).  For RAB, apply the mask
        // first on masked tiles so add_rab_bs can skip -inf entries without
        // recomputing the mask predicate on the math critical path.
        // kIsMasking=true  (Phase 1): always apply_mask_bs (causal diagonal or Is_arbitrary/Is_local).
        // kIsMasking=false (Phase 2): steady-state; skip diagonal masking; varlen-end check only.
        bool mask_applied = false;
        if constexpr (Is_arbitrary || Is_local || kIsMasking) {
          apply_mask_bs(acc_s, nb);
          mask_applied = true;
        } else if constexpr (kRuntimeMasking) {
          if (do_mask_runtime) {
            apply_mask_bs(acc_s, nb);
            mask_applied = true;
          } else {
            if ((nb + 1) * kBlockN > actual_seqlen_h) {
              apply_mask_bs(acc_s, nb);
              mask_applied = true;
            }
          }
        } else {
          if ((nb + 1) * kBlockN > actual_seqlen_h) {
            apply_mask_bs(acc_s, nb);
            mask_applied = true;
          }
        }
        if constexpr (Has_rab && kUseRabSkipMasked) {
          consume_rab_bs(acc_s, nb, mask_applied);
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
        // kAccSElems = 64 for BM128/BN128 and 32 for BM128/BN64 at 256 math threads.
        // Packing order matches the flat C-fragment order of thr_mma_g1.partition_C(...).
        // Under AtomLayout <_8,_1,_1>, the old select<1,2,0,3>-based regrouping is not a
        // valid (M,N) mapping; use acc_s(flat) directly to preserve C-fragment order.
        constexpr int kAccSElems = kBlockM * kBlockN / kNMathThreads;
        static_assert(kAccSElems % 4 == 0, "kAccSElems must be a multiple of 4");
        uint32_t acc_s_packed[kAccSElems / 4];
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
        // acc_s is now DEAD.
        //
        // Build GEMM2 A-fragment P directly in registers via warp shuffle.
        // With AtomLayout <_8,_1,_1> (1 N-warp), all N-columns of P live in each
        // warp's own registers — no cross-warp SMEM staging or bar.sync is needed.
        // sPbuf_pi is defined for partition_fragment_A shape inference only; sK_cur is
        // not written and remains available for the load warp's next TMA fill.
        auto sPbuf_pi = [&]() {
          if constexpr (kBlockN == 64) {
            using SmemLayoutP_SW64 = decltype(tile_to_shape(
                GMMA::Layout_K_SW64_Atom<FP8Elem>{},
                Shape<Int<kBlockM>, Int<kBlockN>>{}));
            Tensor sPbuf = make_tensor(make_smem_ptr(sK_cur), SmemLayoutP_SW64{});
            return as_position_independent_swizzle_tensor(sPbuf);
          } else {
            using SmemLayoutP_SW128 = decltype(tile_to_shape(
                typename BS2::SmemLayoutAtomA{},
                Shape<Int<kBlockM>, Int<kBlockN>>{}));
            Tensor sPbuf = make_tensor(make_smem_ptr(sK_cur), SmemLayoutP_SW128{});
            return as_position_independent_swizzle_tensor(sPbuf);
          }
        }();
        Tensor tCrP = thr_mma_g2.partition_fragment_A(sPbuf_pi);
        {
          // Warp shuffle: rearrange acc_s_packed (C-fragment) into
          // tCrP (A-fragment) without SMEM staging or bar.sync.
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
          if constexpr (kBlockN == 64) {
            CUTE_UNROLL
            for (int kb = 0; kb < 2; ++kb) {
              CUTE_UNROLL
              for (int c = 0; c < 2; ++c) {
                // Linear 64-wide PermMmaTileN: N-atom order is 0,8,16,...,56.
                // For a 4-column A-fragment slice, tiq selects lower/upper atom pair.
                const uint32_t pk0 = acc_s_packed[4 * kb + 2 * c + 0];
                const uint32_t pk1 = acc_s_packed[4 * kb + 2 * c + 1];
                const uint32_t a0  = __shfl_sync(0xFFFFFFFFu, pk0, src_lane0);
                const uint32_t b0  = __shfl_sync(0xFFFFFFFFu, pk0, src_lane1);
                const uint32_t a1  = __shfl_sync(0xFFFFFFFFu, pk1, src_lane0);
                const uint32_t b1  = __shfl_sync(0xFFFFFFFFu, pk1, src_lane1);
                const uint32_t a   = (tiq >> 1u) ? a1 : a0;
                const uint32_t b   = (tiq >> 1u) ? b1 : b0;
                tXrP(4 * kb + 2 * c + 0) = __byte_perm(a, b, 0x5410u);
                tXrP(4 * kb + 2 * c + 1) = __byte_perm(a, b, 0x7632u);
              }
            }
            if constexpr (kHeadDim == 256) {
              CUTE_UNROLL
              for (int i = 8; i < size(tXrP); ++i) {
                tXrP(i) = tXrP(i & 7);
              }
            }
          } else {
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
        }

        { // tCrV, tCrSFV, tCrSFP scoped here: compiler can reuse registers freed by tCrK/tCrSFB.
          {
            const int cur_parity = math_stage ? tma_parity1 : tma_parity0;
            uint32_t vaddr =
                smem_base32 + (uint32_t)kSmemMbar0Offset + 16u + (uint32_t)(math_stage * 8);
            wait_mbar_parity(vaddr, (uint32_t)cur_parity);
          }
          if (math_stage) { tma_parity1 ^= 1; } else { tma_parity0 ^= 1; }
          asm volatile("" ::: "memory");

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
          auto sVt_ns = [&]() {
            if constexpr (kBlockN == 64) {
              using SmemLayoutVt_SW64 = decltype(tile_to_shape(
                  GMMA::Layout_K_SW64_Atom<FP8Elem>{},
                  Shape<Int<kHeadDimGemm2>, Int<kBlockN>>{}));
              return make_tensor(make_smem_ptr(sVt_cur), SmemLayoutVt_SW64{});
            } else {
              using SmemLayoutVt_SW128 = decltype(tile_to_shape(
                  typename BS2::SmemLayoutAtomB{},
                  Shape<Int<kHeadDim>, Int<kBlockN>>{}));
              return make_tensor(make_smem_ptr(sVt_cur), SmemLayoutVt_SW128{});
            }
          }();
          Tensor tCrV = thr_mma_g2.partition_fragment_B(sVt_ns);
          {
            auto tXrV = recast<uint32_t>(tCrV);
            if constexpr (kHeadDim < kHeadDimGemm2) {
              clear(tCrV);
            }
            const uint32_t v_smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(sVt_cur));
            typename Kernel_traits::SmemLayoutVt_TMA smem_layout_vt;
            const int lane  = tidx_math & 31;
            const int n_off = lane & 15;          // row within 16-row n_k group
            const int d_mat = (lane >> 4) << 4;  // 0 for lanes 0-15, 16 for lanes 16-31
            CUTE_UNROLL
            for (int dg = 0; dg < kHeadDim / 32; ++dg) {   // 4 d-groups per 128-D slab
              CUTE_UNROLL
              for (int ni = 0; ni < kBlockN / 16; ++ni) {
                const int n_k_row     = ni * 16 + n_off;
                // r0..r3 cover 4 n_k-groups (4 rows each) at d = dg*32+lane, placed into
                // N-atoms {dg, dg+4, dg+8, dg+12} × K-half {ni&1} of K-block {ni>>1}.
                const int d_start = dg * 32 + d_mat;
                const uint32_t addr_slab =
                    v_smem_base + (uint32_t)smem_layout_vt(n_k_row, d_start);
                if constexpr (kHeadDim <= 64) {
                  // GEMM2 TileN is linear for D=32/64.  Each 32-D group has four
                  // 8-wide N-atoms, and ni supplies the K-half within a 32-token
                  // K block.  This mirrors GEMM1's BN64 B-fragment placement.
                  const int base = 16 * (ni >> 1) + 8 * dg + (ni & 1);
                  asm volatile(
                      "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];\n"
                      : "=r"(tXrV(base + 0)), "=r"(tXrV(base + 2)),
                        "=r"(tXrV(base + 4)), "=r"(tXrV(base + 6))
                      : "r"(addr_slab));
                } else {
                  const int dg_slab = dg >> 2;
                  const int dg_in_slab = dg & 3;
                  const int base =
                      (kHeadDim / 4) * (ni >> 1) + 32 * dg_slab + 2 * dg_in_slab + (ni & 1);
                  asm volatile(
                      "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];\n"
                      : "=r"(tXrV(base + 0)), "=r"(tXrV(base + 8)),
                        "=r"(tXrV(base + 16)), "=r"(tXrV(base + 24))
                      : "r"(addr_slab));
                }
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
          Tensor sSFV_ = make_tensor(make_smem_ptr(smem_sfv_cur), SmemLayoutSFV{});
          auto sSFV = as_position_independent_swizzle_tensor(sSFV_);
          Tensor tCrSFV = BS2::partition_fragment_SFB(sSFV(_,_,_0{}), thr_mma_g2);
          {
            // SFV TMA loads one scale vector per K/V tile stage (kBlockN entries).
            // GEMM2's B operand is V^T, so BS2 expects SFV across the head-dim N atoms.
            // The HSTU FP8 path stores the same V scale across the tile entries; for BN64,
            // reading nr*8 would run past the 64-entry stage. Broadcast a valid entry.
            // AtomLayoutSFB_TV = (4,8):(0,1): lane>>2 indexes the within-atom N-column offset.
            const int n_row_sfv = (tidx_math & 31) >> 2;
            const int sfv = smem_sfv_cur[n_row_sfv];
            constexpr int kNAtomsSFV = kHeadDimGemm2 / 8;
            CUTE_UNROLL
            for (int nr = 0; nr < kNAtomsSFV; ++nr) {
              tCrSFV(0, nr, 0) = sfv;
            }
          }
          auto tCrSFV_frg = BS2::transform_fragment_for_qmma(tCrSFV);

          // Signal load warp: sVt[math_stage]/SFV consumed; V load warp may overwrite.
          if ((tidx_math & 31) == 0) {
            uint32_t v_empty_addr =
                smem_base32 + (uint32_t)kSmemMbar0Offset + 48u + (uint32_t)(math_stage * 8);
            arrive_mbar(v_empty_addr);
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
        if constexpr (!Kernel_traits::kUseSingleKVStage) {
          math_stage ^= 1;
        }
        if constexpr (kStoreOInMainloop) {
          if (store_o_after) {
            store_o_epilogue();
            o_epilogue_done = true;
          }
        }
        return n_valid_ref;
      };  // end run_n_tile lambda

      // N-loop (Opt C): for causal (not arbitrary, not local), split into masked Phase 1
      // (n_masking_steps tiles) and unmasked Phase 2 (steady-state) to allow compile-time
      // dead-code elimination of apply_mask_bs in the hot path.
      if constexpr (Is_causal && !Is_arbitrary && !Is_local) {
        if constexpr (kHeadDim == 256 && !Is_target && !Is_context) {
          for (int n_valid = n_block_max - 1, masking_step = 0; n_valid >= n_block_min;
               ++masking_step, --n_valid) {
            n_valid = run_n_tile(n_valid, n_valid, masking_step,
                masking_step < n_masking_steps,
                false,
                std::integral_constant<int, 2>{});
          }
        } else {
          int n_valid    = n_block_max - 1;
          int masking_step = 0;
          // Phase 1: causal diagonal tiles — apply_mask_bs compiled in (kIsMasking=true).
          for (; n_valid >= n_block_min && masking_step < n_masking_steps; ++masking_step, --n_valid)
            n_valid = run_n_tile(n_valid, n_valid, masking_step,
                false,
                (!Is_target && !Is_context && n_valid == n_block_min),
                std::integral_constant<int, 1>{});
          // Phase 2: steady-state tiles — apply_mask_bs compile-time eliminated (kIsMasking=false).
          for (; n_valid >= n_block_min; ++masking_step, --n_valid)
            n_valid = run_n_tile(n_valid, n_valid, masking_step,
                false,
                (!Is_target && !Is_context && n_valid == n_block_min),
                std::integral_constant<int, 0>{});
        }
      } else if constexpr (Is_arbitrary || Is_local) {
        // Every tile needs masking: pass true_type so apply_mask_bs is always compiled in.
        for (int n_valid = n_block_max - 1, masking_step = 0; n_valid >= n_block_min;
             ++masking_step, --n_valid) {
          const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;
          n_valid = run_n_tile(nb, n_valid, masking_step,
              false, false, std::integral_constant<int, 1>{});
        }
      } else {
        // Full attention (!Is_causal, !Is_arbitrary, !Is_local): varlen-end check only.
        for (int n_valid = n_block_max - 1, masking_step = 0; n_valid >= n_block_min;
             ++masking_step, --n_valid)
          n_valid = run_n_tile(n_valid, n_valid, masking_step,
              false, false, std::integral_constant<int, 0>{});
      }

      if constexpr (kHeadDim > 128) {
        if ((tidx_math & 31) == 0) {
          arrive_mbar(q_empty_mbar_ptr);
        }
      }

      // Complex/full paths and debug_gemm1_only keep the old post-loop epilogue fallback.
      if constexpr (kStoreOInMainloop) {
        if (!o_epilogue_done) {
          store_o_epilogue();
        }
      } else {
        store_o_epilogue();
      }
    };

    // Static tile-id broadcast mirrors the load path exactly.
    const bool head_shared_rab = Has_rab && params.h_rab == 1 && params.h > 1;
    if constexpr (Use_paired_persistent) {
      const int num_m_block_persistent = (params.seqlen_q + kBlockM - 1) / kBlockM;
      const int total_tiles_persistent = num_m_block_persistent * params.h * params.b;
      const int total_tile_pairs_persistent = (total_tiles_persistent + 1) / 2;
      #pragma unroll 1
      for (int tile_pair = int(blockIdx.x); tile_pair < total_tile_pairs_persistent; tile_pair += int(gridDim.x)) {
        const int paired_tile = total_tiles_persistent - 1 - tile_pair;
        const int tiles_this_pair = paired_tile == tile_pair ? 1 : 2;

        #pragma unroll 1
        for (int pair_slot = 0; pair_slot < tiles_this_pair; ++pair_slot) {
          const int tile = pair_slot == 0 ? tile_pair : paired_tile;
          const HstuWsTileCoord coord =
              hstu_ws_decode_tile(tile, num_m_block_persistent, params.h, head_shared_rab);
          run_math_tile(coord.bidb, coord.bidh, coord.m_block);
        }
      }
    } else if constexpr (Use_full_persistent) {
      const int num_m_block_persistent = (params.seqlen_q + kBlockM - 1) / kBlockM;
      const int total_tiles_persistent = num_m_block_persistent * params.h * params.b;
      #pragma unroll 1
      for (int tile = int(blockIdx.x); tile < total_tiles_persistent; tile += int(gridDim.x)) {
        const HstuWsTileCoord coord =
            hstu_ws_decode_tile(tile, num_m_block_persistent, params.h, head_shared_rab);
        run_math_tile(coord.bidb, coord.bidh, coord.m_block);
      }
    } else {
      run_math_tile(bidb_arg, bidh_arg, m_block_arg);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Phase 6 WS TMA kernel entry: launched with kNThreads=384.
// Q, K, V^T, Q-SF, K-SF, and V-SF all via TMA.
template <typename Kernel_traits, typename Params>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1)
hstu_fwd_kernel_sm120_fp8_ws_tma(
    __grid_constant__ Params const params) {
  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr bool Use_full_persistent =
      !Kernel_traits::Is_causal &&
      !Kernel_traits::Is_target &&
      !Kernel_traits::Is_context &&
      !Kernel_traits::Is_local &&
      !Kernel_traits::Is_arbitrary;
  constexpr bool Use_paired_persistent =
      Kernel_traits::Is_causal &&
      !Kernel_traits::Is_context &&
      !Kernel_traits::Is_local &&
      !Kernel_traits::Is_arbitrary &&
      (!Kernel_traits::Is_target || Kernel_traits::Paged_KV) &&
      (Kernel_traits::kHeadDim <= 128 || (Kernel_traits::Paged_KV && !Kernel_traits::Has_rab));
  constexpr bool Use_persistent = Use_full_persistent || Use_paired_persistent;

  if constexpr (!Use_persistent) {
    int m_block;
    int bidh;
    int bidb = blockIdx.z;
    if constexpr (Kernel_traits::Has_rab) {
      if (params.h_rab == 1 && params.h > 1) {
        m_block = gridDim.y - blockIdx.y - 1;
        bidh    = blockIdx.x;
      } else {
        m_block = gridDim.x - blockIdx.x - 1;
        bidh    = blockIdx.y;
      }
    } else {
      m_block = gridDim.x - blockIdx.x - 1;
      bidh    = blockIdx.y;
    }
    hstu_compute_attn_1rowblock_sm120_fp8_ws<Kernel_traits>(params, bidb, bidh, m_block);
    return;
  }

  hstu_compute_attn_1rowblock_sm120_fp8_ws<
      Kernel_traits,
      Use_full_persistent,
      Use_paired_persistent>(
      params,
      0,
      0,
      0);
}
