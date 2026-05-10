// Warp-specialized SM120 FP8 forward kernel body.
// Included inside namespace flash from hstu_fwd_kernel_launch.h.
//
// Warps 0-7 are math warps. Warps 8-11 are specialized load/store warps:
// Q/SFA load, K/SFB/RAB load, V/SFV load, and O TMA store.
//
// CTA syncs are intentionally sparse:
//   S_arb: arbitrary valid-block handoff.
//   S1   : one-time mbarrier initialization before the persistent scheduler.
//   S5   : partial-O rows only; normal Q/K/V/O handoff uses mbarriers.

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

__device__ __forceinline__ void init_mbar(uint64_t* mbar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               : : "r"(mbar_smem_addr(mbar)), "r"(count));
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

__device__ __forceinline__ uint64_t* hstu_ws_stage_mbar(
    uint64_t* stage0,
    uint64_t* stage1,
    int stage) {
  return stage ? stage1 : stage0;
}

__device__ __forceinline__ void hstu_ws_wait_empty_stage(
    uint64_t* empty0,
    uint64_t* empty1,
    int& parity0,
    int& parity1,
    int stage) {
  uint64_t* mbar = hstu_ws_stage_mbar(empty0, empty1, stage);
  int& parity = stage ? parity1 : parity0;
  wait_mbar_parity(mbar, static_cast<uint32_t>(parity));
  parity ^= 1;
}

template <bool UseSingleKVStage>
__device__ __forceinline__ void hstu_ws_advance_kv_stage(int& stage) {
  if constexpr (!UseSingleKVStage) {
    stage ^= 1;
  }
}

enum class HstuWsLoadWarpRole : int {
  Q = 0,
  K = 1,
  V = 2,
  O = 3,
};

__device__ __forceinline__ HstuWsLoadWarpRole hstu_ws_load_warp_role(
    int tidx,
    int n_math_threads) {
  return static_cast<HstuWsLoadWarpRole>((tidx - n_math_threads) >> 5);
}

template <bool UseRabSmem>
__device__ __forceinline__ void hstu_ws_init_mbarriers(
    int tidx,
    int init_tidx,
    uint64_t* k_ready_mbar_ptr0,
    uint64_t* k_ready_mbar_ptr1,
    uint64_t* v_ready_mbar_ptr0,
    uint64_t* v_ready_mbar_ptr1,
    uint64_t* k_empty_mbar_ptr0,
    uint64_t* k_empty_mbar_ptr1,
    uint64_t* v_empty_mbar_ptr0,
    uint64_t* v_empty_mbar_ptr1,
    uint64_t* q_ready_mbar_ptr,
    uint64_t* q_empty_mbar_ptr,
    uint64_t* o_ready_mbar_ptr0,
    uint64_t* o_ready_mbar_ptr1,
    uint64_t* o_empty_mbar_ptr0,
    uint64_t* o_empty_mbar_ptr1,
    uint64_t* rab_ready_mbar_ptr0,
    uint64_t* rab_empty_mbar_ptr0) {
  if (tidx == init_tidx) {
    init_mbar(k_ready_mbar_ptr0, 1);
    init_mbar(k_ready_mbar_ptr1, 1);
    init_mbar(v_ready_mbar_ptr0, 1);
    init_mbar(v_ready_mbar_ptr1, 1);
    init_mbar(k_empty_mbar_ptr0, 8);
    init_mbar(k_empty_mbar_ptr1, 8);
    init_mbar(v_empty_mbar_ptr0, 8);
    init_mbar(v_empty_mbar_ptr1, 8);
    init_mbar(q_ready_mbar_ptr, 1);
    init_mbar(q_empty_mbar_ptr, 8);
    init_mbar(o_ready_mbar_ptr0, 8);
    init_mbar(o_ready_mbar_ptr1, 8);
    init_mbar(o_empty_mbar_ptr0, 1);
    init_mbar(o_empty_mbar_ptr1, 1);
    if constexpr (UseRabSmem) {
      init_mbar(rab_ready_mbar_ptr0, 1);
      init_mbar(rab_empty_mbar_ptr0, 8);
    }

    CUTE_UNROLL
    for (int i = 0; i < 8; i++) {
      arrive_mbar(k_empty_mbar_ptr0);
      arrive_mbar(k_empty_mbar_ptr1);
      arrive_mbar(v_empty_mbar_ptr0);
      arrive_mbar(v_empty_mbar_ptr1);
      arrive_mbar(q_empty_mbar_ptr);
      if constexpr (UseRabSmem) {
        arrive_mbar(rab_empty_mbar_ptr0);
      }
    }
    arrive_mbar(o_empty_mbar_ptr0);
  }
  asm volatile("fence.proxy.async.shared::cta;\n" : : : "memory");
}

template <typename Kernel_traits, typename Params>
__device__ __forceinline__ bool hstu_ws_use_rab_smem_for_nb(
    const Params& params,
    int nb,
    int actual_seqlen_offset,
    int n_block_paged,
    int last_page_offset) {
  if constexpr (!Kernel_traits::kUseRabSmem) {
    return false;
  } else {
    if constexpr (Kernel_traits::kHeadDim > 128 &&
        Kernel_traits::Is_causal &&
        !Kernel_traits::Is_target &&
        !Kernel_traits::Is_context &&
        !Kernel_traits::Is_local &&
        !Kernel_traits::Is_arbitrary) {
      if (params.b * params.h <= 4) {
        return false;
      }
    }
    bool aligned = (actual_seqlen_offset % Kernel_traits::kBlockM) == 0;
    if constexpr (Kernel_traits::Paged_KV && Kernel_traits::Is_target) {
      if (nb >= n_block_paged && last_page_offset != 0) {
        aligned = false;
      }
    }
    return aligned;
  }
}

template <int kHeadDim, typename OutElement, typename TensorO>
__device__ __forceinline__ void hstu_ws_store_o_stmatrix(
    char* smem_o,
    TensorO& rO,
    int tidx_math) {
  static_assert(sizeof(OutElement) == 2, "stmatrix.m8n8.b16 requires 2-byte elements");

  auto rO_u32 = recast<uint32_t>(rO);
  const int lane        = tidx_math & 31;
  const int lq          = lane >> 2;
  const int lqt         = lane & 3;
  const int warp_m      = tidx_math >> 5;
  const int addr_row    = ((lq & 1) << 2) | lqt;
  const int mat_in_lane = lq >> 1;

  const uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem_o));
  constexpr uint32_t row_bytes = kHeadDim * (uint32_t)sizeof(OutElement);

  CUTE_UNROLL
  for (int m_grp = 0; m_grp < 2; ++m_grp) {
    const uint32_t m_bytes =
        (uint32_t)(warp_m * 16 + m_grp * 8 + addr_row) * row_bytes;

    CUTE_UNROLL
    for (int n_grp = 0; n_grp < 4; ++n_grp) {
      const uint32_t stsm_addr = smem_base
          + m_bytes
          + (uint32_t)(n_grp * 32 + mat_in_lane * 8) * (uint32_t)sizeof(OutElement);

      const uint32_t rb0 = rO_u32[2 * (n_grp     ) + m_grp];
      const uint32_t rb1 = rO_u32[2 * (n_grp +  4) + m_grp];
      const uint32_t rb2 = rO_u32[2 * (n_grp +  8) + m_grp];
      const uint32_t rb3 = rO_u32[2 * (n_grp + 12) + m_grp];

      asm volatile(
          "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
          : : "r"(stsm_addr), "r"(rb0), "r"(rb1), "r"(rb2), "r"(rb3) : "memory");
    }
  }
}

template <int kBlockM, int kHeadDim, int kNMathThreads, typename OutElement>
__device__ __forceinline__ void hstu_ws_zero_oob_o_rows(
    char* smem_o,
    int valid_rows,
    int tidx_math) {
  const int oob_elems = (kBlockM - valid_rows) * kHeadDim;
  OutElement* sO_raw = reinterpret_cast<OutElement*>(smem_o) + valid_rows * kHeadDim;
  for (int i = tidx_math; i < oob_elems; i += kNMathThreads) {
    sO_raw[i] = OutElement(0);
  }
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

template <
    bool IsV,
    int kHeadDim,
    int kBlockN,
    typename Params,
    typename FP8Elem,
    typename SmemLayout>
__device__ __forceinline__ void hstu_ws_copy_paged_or_target_tile(
    FP8Elem* __restrict__ dst_smem,
    int32_t* __restrict__ sf_dst_smem,
    const Params& params,
    int nb,
    int n_block_paged,
    int page_offset,
    int bidh_kv,
    int target_start_base,
    int actual_seqlen_t,
    int lane,
    SmemLayout const& layout) {
  if (nb < n_block_paged) {
    const int page_id = params.page_ids[page_offset + nb];
    const FP8Elem* src = reinterpret_cast<const FP8Elem*>(params.kv_cache_ptr)
        + (int64_t)page_id * params.kv_cache_kvtensor_stride
        + (IsV ? params.kv_cache_page_stride : 0)
        + (int64_t)bidh_kv * params.kv_cache_row_stride;
    copy_fp8_tile_rowmajor_to_smem<FP8Elem, kHeadDim>(
        dst_smem,
        src,
        params.kv_cache_head_stride,
        kBlockN,
        lane,
        layout);
    copy_packed_sf_tile(
        sf_dst_smem,
        IsV ? params.sf_v_packed_ptr : params.sf_k_packed_ptr,
        IsV ? params.v_block_descale_head_stride : params.kv_block_descale_head_stride,
        bidh_kv,
        page_id * params.page_size,
        kBlockN,
        lane);
  } else {
    const int target_block = nb - n_block_paged;
    const int target_start = target_start_base + target_block * kBlockN;
    const int rows_valid = std::max(
        0,
        std::min(kBlockN, actual_seqlen_t - target_block * kBlockN));
    const FP8Elem* src;
    int64_t src_row_stride;
    if constexpr (IsV) {
      src = reinterpret_cast<const FP8Elem*>(params.v_ptr)
          + (int64_t)target_start * params.v_row_stride
          + (int64_t)bidh_kv * params.v_head_stride;
      src_row_stride = params.v_row_stride;
    } else {
      src = reinterpret_cast<const FP8Elem*>(params.k_ptr)
          + (int64_t)target_start * params.k_row_stride
          + (int64_t)bidh_kv * params.k_head_stride;
      src_row_stride = params.k_row_stride;
    }
    copy_fp8_tile_rowmajor_to_smem<FP8Elem, kHeadDim>(
        dst_smem,
        src,
        src_row_stride,
        rows_valid,
        lane,
        layout);
    copy_packed_sf_tile(
        sf_dst_smem,
        IsV ? params.sf_v_packed_ptr : params.sf_k_packed_ptr,
        IsV ? params.v_block_descale_head_stride : params.kv_block_descale_head_stride,
        bidh_kv,
        params.total_pages * params.page_size + target_start,
        rows_valid,
        lane);
  }
  __syncwarp();
}

struct HstuWsTileCoord {
  int bidb;
  int bidh;
  int m_block;
};

struct HstuWsTileInfo {
  int actual_seqlen_q;
  int actual_seqlen_k;
  int actual_seqlen_q_padded;
  int actual_seqlen_t;
  int actual_seqlen_c;
  int actual_seqlen_h;
  int actual_seqlen_offset;
  int page_offset;
  int last_page_offset;
  int n_block_history;
  int n_block_paged;
  int n_block_min;
  int n_block_max;
  int n_masking_steps;
  bool is_jump;
  bool is_in_context;
  bool is_in_paged_target;
};

template <typename Kernel_traits, typename Params>
__device__ __forceinline__ HstuWsTileInfo hstu_ws_make_tile_info(
    const Params& params,
    const HstuBlockInfo<Kernel_traits, Params>& binfo,
    int m_block) {
  constexpr bool Is_causal    = Kernel_traits::Is_causal;
  constexpr bool Is_target    = Kernel_traits::Is_target;
  constexpr bool Is_context   = Kernel_traits::Is_context;
  constexpr bool Is_local     = Kernel_traits::Is_local;
  constexpr bool Paged_KV     = Kernel_traits::Paged_KV;
  constexpr int  kBlockM      = Kernel_traits::kBlockM;
  constexpr int  kBlockN      = Kernel_traits::kBlockN;

  HstuWsTileInfo info;
  info.actual_seqlen_q        = binfo.actual_seqlen_q;
  info.actual_seqlen_k        = binfo.actual_seqlen_k;
  info.actual_seqlen_q_padded = binfo.actual_seqlen_q_padded;
  info.actual_seqlen_t        = Is_target  ? binfo.actual_seqlen_t : 0;
  info.actual_seqlen_c        = Is_context ? binfo.actual_seqlen_c : 0;
  info.actual_seqlen_h        = Is_target
      ? info.actual_seqlen_k - info.actual_seqlen_t
      : info.actual_seqlen_k;
  info.actual_seqlen_offset   = info.actual_seqlen_k - info.actual_seqlen_q;
  const int last_page_seqlen  = Paged_KV ? binfo.last_page_seqlen : kBlockN;
  info.page_offset            = Paged_KV ? binfo.sum_s_page : 0;

  info.is_jump = Is_target &&
      m_block * kBlockM + info.actual_seqlen_offset > info.actual_seqlen_h;
  const bool is_in_target = Is_target &&
      (m_block + 1) * kBlockM + info.actual_seqlen_offset > info.actual_seqlen_h;
  info.is_in_context = Is_context &&
      (m_block + 1) * kBlockM <= info.actual_seqlen_c;
  const bool is_in_mixed_context = Is_context &&
      (m_block + 1) * kBlockM > info.actual_seqlen_c &&
      m_block * kBlockM < info.actual_seqlen_c;
  info.is_in_paged_target = is_in_target && Paged_KV;
  info.last_page_offset = info.is_in_paged_target ? kBlockN - last_page_seqlen : 0;

  info.n_block_history = cute::ceil_div(info.actual_seqlen_h, kBlockN);
  const int target_index =
      (m_block * kBlockM - info.actual_seqlen_h) / params.target_group_size;
  info.n_block_paged = Paged_KV ? info.n_block_history : 0;
  const int n_block_target = cute::ceil_div(info.actual_seqlen_t, kBlockN);

  info.n_block_min = !Is_local ? 0
      : std::max(
            0,
            (m_block * kBlockM + info.actual_seqlen_offset - params.window_size_left) / kBlockN);
  info.n_block_max = Paged_KV
      ? info.n_block_history + n_block_target
      : cute::ceil_div(info.actual_seqlen_k, kBlockN);
  if constexpr (Is_causal || Is_local) {
    int offset = (m_block + 1) * kBlockM
        + info.actual_seqlen_offset
        + params.window_size_right;
    if (info.is_in_paged_target) {
      offset += info.last_page_offset;
    }
    info.n_block_max = std::min(info.n_block_max, cute::ceil_div(offset, kBlockN));
  }
  if constexpr (Is_context) {
    info.n_block_min = (info.is_in_context || is_in_mixed_context) ? 0 : info.n_block_min;
    info.n_block_max = (info.is_in_context || is_in_mixed_context)
        ? std::max(info.n_block_history, info.n_block_max)
        : info.n_block_max;
  }

  int n_masking_block_max = cute::ceil_div(
      std::min(
          info.actual_seqlen_k + info.last_page_offset,
          (m_block + 1) * kBlockM
              + info.actual_seqlen_offset
              + info.last_page_offset),
      kBlockN);
  int n_masking_block_min =
      (m_block * kBlockM + info.actual_seqlen_offset) / kBlockN;
  if constexpr (Is_target) {
    n_masking_block_min = info.is_jump
        ? (info.actual_seqlen_h + info.actual_seqlen_offset
              + target_index * params.target_group_size
              + info.last_page_offset) / kBlockN
        : n_masking_block_min;
  }
  if constexpr (Is_context) {
    n_masking_block_min = is_in_mixed_context ? info.n_block_min : n_masking_block_min;
    n_masking_block_max = is_in_mixed_context ? info.n_block_max : n_masking_block_max;
  }
  info.n_masking_steps = (!Is_causal || info.is_in_context)
      ? 0
      : n_masking_block_max - n_masking_block_min;

  return info;
}

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

template <
    bool Use_full_persistent,
    bool Use_paired_persistent,
    bool Sync_between_tiles,
    typename TileFn>
__device__ __forceinline__ void hstu_ws_run_static_schedule(
    const int num_m_block,
    const int num_heads,
    const int num_batches,
    const bool head_shared_rab,
    TileFn const& run_tile) {
  const int total_tiles = num_m_block * num_heads * num_batches;
  static_assert(
      Use_full_persistent || Use_paired_persistent,
      "FP8 WS static scheduler requires a persistent mode");

  if constexpr (Use_paired_persistent) {
    const int total_tile_pairs = (total_tiles + 1) / 2;
    #pragma unroll 1
    for (int tile_pair = int(blockIdx.x); tile_pair < total_tile_pairs; tile_pair += int(gridDim.x)) {
      const int paired_tile = total_tiles - 1 - tile_pair;
      const int tiles_this_pair = paired_tile == tile_pair ? 1 : 2;

      #pragma unroll 1
      for (int pair_slot = 0; pair_slot < tiles_this_pair; ++pair_slot) {
        const int tile = pair_slot == 0 ? tile_pair : paired_tile;
        run_tile(hstu_ws_decode_tile(tile, num_m_block, num_heads, head_shared_rab));
        if constexpr (Sync_between_tiles) {
          const bool has_more_pair_slots = pair_slot + 1 < tiles_this_pair;
          const bool has_more_grid_stride_pairs =
              tile_pair + int(gridDim.x) < total_tile_pairs;
          if (has_more_pair_slots || has_more_grid_stride_pairs) {
            __syncthreads();
          }
        }
      }
    }
  } else if constexpr (Use_full_persistent) {
    #pragma unroll 1
    for (int tile = int(blockIdx.x); tile < total_tiles; tile += int(gridDim.x)) {
      run_tile(hstu_ws_decode_tile(tile, num_m_block, num_heads, head_shared_rab));
      if constexpr (Sync_between_tiles) {
        if (tile + int(gridDim.x) < total_tiles) {
          __syncthreads();
        }
      }
    }
  }
}

// Rearrange GEMM1 C-fragment to GEMM2 A-fragment in registers.
// AtomLayout <8,1,1> keeps each warp's N-values local, so this is intra-warp
// shuffle/byte-permute only and avoids SMEM staging plus two CTA barriers.
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

__device__ __forceinline__ int32_t hstu_ws_replicate_e8m0_lane(
    int32_t packed,
    int lane) {
  const uint32_t byte = (static_cast<uint32_t>(packed) >> (8 * lane)) & 0xffu;
  return static_cast<int32_t>(byte | (byte << 8) | (byte << 16) | (byte << 24));
}

template <int kBlockN, typename TensorB, typename TensorFrag>
__device__ __forceinline__ void hstu_ws_load_b_z_pattern(
    TensorB&& sB_pi,
    TensorFrag& tCrB,
    int k_block_base,
    int k_block_count,
    int row_stride,
    int tidx_math) {
  const int lane    = tidx_math & 31;
  const int mat_num = lane >> 3;
  const int mat_row = lane & 7;
  const uint32_t smem_base =
      static_cast<uint32_t>(__cvta_generic_to_shared(&sB_pi(0, 0)));
  auto tXrB = recast<uint32_t>(tCrB);
  constexpr int kNGroups = kBlockN / 32;

  CUTE_UNROLL
  for (int kb = 0; kb < k_block_count; ++kb) {
    CUTE_UNROLL
    for (int g = 0; g < kNGroups; ++g) {
      const uint32_t n_row = static_cast<uint32_t>(g * 32 + mat_num * 8 + mat_row);
      CUTE_UNROLL
      for (int k_half = 0; k_half < 2; ++k_half) {
        const int k_start = (k_block_base + kb) * 32 + k_half * 16;
        const uint32_t addr =
            smem_base + n_row * row_stride + static_cast<uint32_t>(k_start ^ (mat_row << 4));
        if constexpr (kBlockN == 64) {
          const int frag_base64 = 16 * kb + 8 * g;
          asm volatile(
              "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
              : "=r"(tXrB(frag_base64 + k_half + 0)),
                "=r"(tXrB(frag_base64 + k_half + 2)),
                "=r"(tXrB(frag_base64 + k_half + 4)),
                "=r"(tXrB(frag_base64 + k_half + 6))
              : "r"(addr));
        } else {
          const int frag_base = 32 * kb + 2 * g;
          asm volatile(
              "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
              : "=r"(tXrB(frag_base + k_half +  0)),
                "=r"(tXrB(frag_base + k_half +  8)),
                "=r"(tXrB(frag_base + k_half + 16)),
                "=r"(tXrB(frag_base + k_half + 24))
              : "r"(addr));
        }
      }
    }
  }
}

template <typename FP8Elem, typename Layout, typename TensorFrag>
__device__ __forceinline__ void hstu_ws_load_a_z_pattern_layout(
    FP8Elem* smem_ptr,
    Layout const& layout,
    TensorFrag& tCrA,
    int k_block_base,
    int k_block_count,
    int tidx_math) {
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
    const int k_start = (k_block_base + kb) * 32 + (K_half << 4);
    const uint32_t addr = smem_base + (uint32_t)layout(M_abs, k_start);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
        : "=r"(tXrA(4 * kb + 0)), "=r"(tXrA(4 * kb + 1)),
          "=r"(tXrA(4 * kb + 2)), "=r"(tXrA(4 * kb + 3))
        : "r"(addr));
  }
}

template <int kBlockN, typename FP8Elem, typename Layout, typename TensorFrag>
__device__ __forceinline__ void hstu_ws_load_b_z_pattern_layout(
    FP8Elem* smem_ptr,
    Layout const& layout,
    TensorFrag& tCrB,
    int k_block_base,
    int k_block_count,
    int tidx_math) {
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
      const uint32_t n_row = static_cast<uint32_t>(g * 32 + mat_num * 8 + mat_row);
      CUTE_UNROLL
      for (int k_half = 0; k_half < 2; ++k_half) {
        const int k_start = (k_block_base + kb) * 32 + k_half * 16;
        const uint32_t addr = smem_base + (uint32_t)layout((int)n_row, k_start);
        if constexpr (kBlockN == 64) {
          const int frag_base64 = 16 * kb + 8 * g;
          asm volatile(
              "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
              : "=r"(tXrB(frag_base64 + k_half + 0)),
                "=r"(tXrB(frag_base64 + k_half + 2)),
                "=r"(tXrB(frag_base64 + k_half + 4)),
                "=r"(tXrB(frag_base64 + k_half + 6))
              : "r"(addr));
        } else {
          const int frag_base = 32 * kb + 2 * g;
          asm volatile(
              "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
              : "=r"(tXrB(frag_base + k_half +  0)),
                "=r"(tXrB(frag_base + k_half +  8)),
                "=r"(tXrB(frag_base + k_half + 16)),
                "=r"(tXrB(frag_base + k_half + 24))
              : "r"(addr));
        }
      }
    }
  }
}

template <
    typename Kernel_traits,
    typename Params,
    typename ThrMmaG1,
    typename TensorS>
__device__ __forceinline__ void hstu_ws_add_rab_global_bs(
    const Params& params,
    ThrMmaG1 const& thr_mma_g1,
    TensorS& tSrS,
    int bidb,
    int bidh,
    int m_block,
    int nb,
    bool skip_masked,
    int actual_seqlen_q,
    int actual_seqlen_k,
    int actual_seqlen_h,
    int actual_seqlen_offset,
    int last_page_offset,
    int n_block_paged) {
  if constexpr (Kernel_traits::Has_rab) {
    static constexpr int Row = 0, Col = 1;
    constexpr bool Is_target = Kernel_traits::Is_target;
    constexpr bool Paged_KV  = Kernel_traits::Paged_KV;
    constexpr int  kBlockM   = Kernel_traits::kBlockM;
    constexpr int  kBlockN   = Kernel_traits::kBlockN;
    using RabElement = cutlass::bfloat16_t;

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
}

template <
    typename Kernel_traits,
    typename ThrMmaG1,
    typename TensorS,
    typename TensorRab>
__device__ __forceinline__ void hstu_ws_add_rab_smem_bs(
    ThrMmaG1 const& thr_mma_g1,
    TensorS& tSrS,
    TensorRab const& sRab,
    int m_block,
    int nb,
    bool skip_masked,
    int stage,
    int actual_seqlen_q,
    int actual_seqlen_k,
    int actual_seqlen_h,
    int last_page_offset,
    int n_block_paged) {
  if constexpr (Kernel_traits::Has_rab) {
    static constexpr int Row = 0, Col = 1;
    constexpr bool Is_target = Kernel_traits::Is_target;
    constexpr bool Paged_KV  = Kernel_traits::Paged_KV;
    constexpr int  kBlockM   = Kernel_traits::kBlockM;
    constexpr int  kBlockN   = Kernel_traits::kBlockN;

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
        if (nb < n_block_paged && col >= actual_seqlen_h) {
          continue;
        }
      }
      if (0 <= col && col < actual_seqlen_k) {
        tSrS(flat) += static_cast<float>(sRab(block_row, block_col, stage));
      }
    }
  }
}

template <
    typename Kernel_traits,
    typename Params,
    typename ThrMmaG1,
    typename TensorS,
    typename TensorRab>
__device__ __forceinline__ void hstu_ws_consume_rab_bs(
    const Params& params,
    ThrMmaG1 const& thr_mma_g1,
    TensorS& tSrS,
    TensorRab const& sRab,
    uint32_t smem_base32,
    int& rab_ready_wait_parity0,
    int bidb,
    int bidh,
    int m_block,
    int nb,
    bool skip_masked,
    int actual_seqlen_q,
    int actual_seqlen_k,
    int actual_seqlen_h,
    int actual_seqlen_offset,
    int last_page_offset,
    int n_block_paged) {
  if constexpr (Kernel_traits::Has_rab) {
    constexpr uint32_t kSmemMbarOffset =
        (uint32_t)(Kernel_traits::kSmemSize - Kernel_traits::kSmemMbarSize);
    if (hstu_ws_use_rab_smem_for_nb<Kernel_traits>(
            params, nb, actual_seqlen_offset, n_block_paged, last_page_offset)) {
      wait_mbar_parity(
          smem_base32 + kSmemMbarOffset + 112u,
          (uint32_t)rab_ready_wait_parity0);
      rab_ready_wait_parity0 ^= 1;
      asm volatile("" ::: "memory");
      hstu_ws_add_rab_smem_bs<Kernel_traits>(
          thr_mma_g1, tSrS, sRab, m_block, nb, skip_masked, 0,
          actual_seqlen_q, actual_seqlen_k, actual_seqlen_h,
          last_page_offset, n_block_paged);
      if ((threadIdx.x & 31) == 0) {
        arrive_mbar(smem_base32 + kSmemMbarOffset + 120u);
      }
    } else {
      hstu_ws_add_rab_global_bs<Kernel_traits>(
          params, thr_mma_g1, tSrS, bidb, bidh, m_block, nb, skip_masked,
          actual_seqlen_q, actual_seqlen_k, actual_seqlen_h,
          actual_seqlen_offset, last_page_offset, n_block_paged);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename Kernel_traits,
    bool Use_full_persistent,
    bool Use_paired_persistent,
    typename Params>
inline __device__ void hstu_compute_attn_1rowblock_sm120_fp8_ws(
    const Params& params) {

  static_assert(Kernel_traits::Is_fp8, "FP8 WS path only");
  static_assert(
      !(Use_full_persistent && Use_paired_persistent),
      "Only one FP8 WS persistent scheduler can be enabled");
  static_assert(
      Use_full_persistent || Use_paired_persistent,
      "FP8 WS now always uses a persistent scheduler");

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
      static constexpr int kLoadWarpRegBudget =
          Kernel_traits::kHeadDim > 128 ? 40 : 56;
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(kLoadWarpRegBudget));
      const HstuWsLoadWarpRole load_warp_role =
          hstu_ws_load_warp_role(tidx, kNMathThreads);
      const bool is_q_load_warp = load_warp_role == HstuWsLoadWarpRole::Q;
      const bool is_k_load_warp = load_warp_role == HstuWsLoadWarpRole::K;
      const bool is_v_load_warp = load_warp_role == HstuWsLoadWarpRole::V;
      const bool is_o_store_warp = load_warp_role == HstuWsLoadWarpRole::O;

      constexpr bool Is_causal    = Kernel_traits::Is_causal;
    constexpr bool Is_target    = Kernel_traits::Is_target;
    constexpr bool Is_context   = Kernel_traits::Is_context;
    constexpr bool Is_arbitrary = Kernel_traits::Is_arbitrary;
    constexpr int  kNFunc       = Kernel_traits::kNFunc;
    constexpr bool Is_local     = Kernel_traits::Is_local;
    constexpr bool Has_rab      = Kernel_traits::Has_rab;
    constexpr bool Use_rab_smem = Kernel_traits::kUseRabSmem;
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

    int k_empty_wait_parity0 = 0;
    int k_empty_wait_parity1 = 0;
    int v_empty_wait_parity0 = 0;
    int v_empty_wait_parity1 = 0;
    int q_empty_wait_parity = 0;
    int o_ready_wait_parity0 = 0;
    int rab_empty_wait_parity0 = 0;
    int kv_load_stage = 0;
    hstu_ws_init_mbarriers<Use_rab_smem>(
        tidx,
        kNMathThreads,
        k_ready_mbar_ptr0,
        k_ready_mbar_ptr1,
        v_ready_mbar_ptr0,
        v_ready_mbar_ptr1,
        k_empty_mbar_ptr0,
        k_empty_mbar_ptr1,
        v_empty_mbar_ptr0,
        v_empty_mbar_ptr1,
        q_ready_mbar_ptr,
        q_empty_mbar_ptr,
        o_ready_mbar_ptr0,
        o_ready_mbar_ptr1,
        o_empty_mbar_ptr0,
        o_empty_mbar_ptr1,
        rab_ready_mbar_ptr0,
        rab_empty_mbar_ptr0);
    __syncthreads();  // One-time S1: WS mbarriers initialized before scheduler loop.

      auto run_load_tile = [&](const int bidb, const int bidh, int m_block) {
        const HstuBlockInfo<Kernel_traits, Params> binfo(params, bidb);
        // Early exit 1: before any sync — both branches exit simultaneously.
        if (m_block * kBlockM >= binfo.actual_seqlen_q_padded) return;

      char* smem_q    = reinterpret_cast<char*>(smem_);
      char* smem_func = reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsFuncOffset;
      int* sn_valid_block_max = reinterpret_cast<int*>(smem_func);

      const HstuWsTileInfo tile_info =
          hstu_ws_make_tile_info<Kernel_traits>(params, binfo, m_block);
      const int actual_seqlen_q      = tile_info.actual_seqlen_q;
      const int actual_seqlen_k      = tile_info.actual_seqlen_k;
      const int actual_seqlen_t      = tile_info.actual_seqlen_t;
      const int actual_seqlen_offset = tile_info.actual_seqlen_offset;
      const int page_offset          = tile_info.page_offset;
      const int last_page_offset     = tile_info.last_page_offset;
      const int n_block_history      = tile_info.n_block_history;
      const int n_block_paged        = tile_info.n_block_paged;
      int n_block_min                = tile_info.n_block_min;
      int n_block_max                = tile_info.n_block_max;
      const int n_masking_steps      = tile_info.n_masking_steps;
      const bool is_jump             = tile_info.is_jump;

      // Is_arbitrary: load warp only participates in __syncthreads__; math warp 1 does the work.
      if constexpr (Is_arbitrary) {
        __syncthreads();  // S_arb: wait for math warp 1 to write sValidBlockIds
        n_block_max = *sn_valid_block_max;
        n_block_min = 0;
      }

      if (((Is_causal || Is_local || Is_arbitrary) && n_block_max <= n_block_min) ||
          m_block * kBlockM >= actual_seqlen_q) {
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

      auto load_rab_tma = [&](int nb) {
        if constexpr (Has_rab) {
          if (hstu_ws_use_rab_smem_for_nb<Kernel_traits>(
                  params, nb, actual_seqlen_offset, n_block_paged, last_page_offset)) {
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
          const int target_start_base =
              binfo.sum_s_k + actual_seqlen_k - actual_seqlen_t + last_page_offset;

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
                hstu_ws_copy_paged_or_target_tile<false, kHeadDim, kBlockN>(
                    stage ? sK_base[1] : sK_base[0],
                    stage ? smem_sfb_ptr[1] : smem_sfb_ptr[0],
                    params,
                    nb,
                    n_block_paged,
                    page_offset,
                    bidh_kv,
                    target_start_base,
                    actual_seqlen_t,
                    tidx & 31,
                    SmemLayoutK_SW128{});
                if (tidx == kNMathThreads + 32) {
                  arrive_mbar(k_ready_mbar_ptr);
                }
              }
            } else {
              hstu_ws_copy_paged_or_target_tile<false, kHeadDim, kBlockN>(
                  stage ? sK_base[1] : sK_base[0],
                  stage ? smem_sfb_ptr[1] : smem_sfb_ptr[0],
                  params,
                  nb,
                  n_block_paged,
                  page_offset,
                  bidh_kv,
                  target_start_base,
                  actual_seqlen_t,
                  tidx & 31,
                  SmemLayoutK_SW128{});
              if (tidx == kNMathThreads + 32) {
                arrive_mbar(k_ready_mbar_ptr);
              }
            }
          };

          for (int n_valid = n_block_max - 1, masking_step_load = 0; n_valid >= n_block_min;
               ++masking_step_load, --n_valid) {
            const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;

            hstu_ws_wait_empty_stage(
                k_empty_mbar_ptr0, k_empty_mbar_ptr1,
                k_empty_wait_parity0, k_empty_wait_parity1,
                kv_load_stage);

            load_paged_tma_or_target_k(
                nb, kv_load_stage,
                hstu_ws_stage_mbar(k_ready_mbar_ptr0, k_ready_mbar_ptr1, kv_load_stage));
            load_rab_tma(nb);

            if (is_jump && masking_step_load == n_masking_steps - 1)
              n_valid = std::min(n_valid, n_block_history);

            hstu_ws_advance_kv_stage<Kernel_traits::kUseSingleKVStage>(kv_load_stage);
          }
        } else {
          for (int n_valid = n_block_max - 1, masking_step_load = 0; n_valid >= n_block_min;
               ++masking_step_load, --n_valid) {
            const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;

            hstu_ws_wait_empty_stage(
                k_empty_mbar_ptr0, k_empty_mbar_ptr1,
                k_empty_wait_parity0, k_empty_wait_parity1,
                kv_load_stage);

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

            hstu_ws_advance_kv_stage<Kernel_traits::kUseSingleKVStage>(kv_load_stage);
          }
        }
      }

        if (is_v_load_warp) {
          if constexpr (Paged_KV) {
          const int target_start_base =
              binfo.sum_s_k + actual_seqlen_k - actual_seqlen_t + last_page_offset;

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
                hstu_ws_copy_paged_or_target_tile<true, kHeadDim, kBlockN>(
                    stage ? sVt_base[1] : sVt_base[0],
                    stage ? smem_sfv_ptr[1] : smem_sfv_ptr[0],
                    params,
                    nb,
                    n_block_paged,
                    page_offset,
                    bidh_kv,
                    target_start_base,
                    actual_seqlen_t,
                    tidx & 31,
                    SmemLayoutVt_SW128{});
                if (tidx == kNMathThreads + 64) {
                  arrive_mbar(v_ready_mbar_ptr);
                }
              }
            } else {
              hstu_ws_copy_paged_or_target_tile<true, kHeadDim, kBlockN>(
                  stage ? sVt_base[1] : sVt_base[0],
                  stage ? smem_sfv_ptr[1] : smem_sfv_ptr[0],
                  params,
                  nb,
                  n_block_paged,
                  page_offset,
                  bidh_kv,
                  target_start_base,
                  actual_seqlen_t,
                  tidx & 31,
                  SmemLayoutVt_SW128{});
              if (tidx == kNMathThreads + 64) {
                arrive_mbar(v_ready_mbar_ptr);
              }
            }
          };

          for (int n_valid = n_block_max - 1, masking_step_load = 0; n_valid >= n_block_min;
               ++masking_step_load, --n_valid) {
            const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;

            hstu_ws_wait_empty_stage(
                v_empty_mbar_ptr0, v_empty_mbar_ptr1,
                v_empty_wait_parity0, v_empty_wait_parity1,
                kv_load_stage);

            load_paged_tma_or_target_v(
                nb, kv_load_stage,
                hstu_ws_stage_mbar(v_ready_mbar_ptr0, v_ready_mbar_ptr1, kv_load_stage));

            if (is_jump && masking_step_load == n_masking_steps - 1)
              n_valid = std::min(n_valid, n_block_history);

            hstu_ws_advance_kv_stage<Kernel_traits::kUseSingleKVStage>(kv_load_stage);
          }
        } else {
          for (int n_valid = n_block_max - 1, masking_step_load = 0; n_valid >= n_block_min;
               ++masking_step_load, --n_valid) {
            const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;

            hstu_ws_wait_empty_stage(
                v_empty_mbar_ptr0, v_empty_mbar_ptr1,
                v_empty_wait_parity0, v_empty_wait_parity1,
                kv_load_stage);

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

            hstu_ws_advance_kv_stage<Kernel_traits::kUseSingleKVStage>(kv_load_stage);
          }
        }
      }

        if (is_o_store_warp) {
          if constexpr (Kernel_traits::kUseIndependentOBuffer) {
          // Math writes O to SMEM; this warp waits on o_ready and performs TMA store.
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

    // Load and math branches run the same static scheduler; no dynamic tile queue.
    constexpr bool kSyncBetweenTiles = Has_rab;
    const bool head_shared_rab = Has_rab && params.h_rab == 1 && params.h > 1;
      hstu_ws_run_static_schedule<
          Use_full_persistent,
          Use_paired_persistent,
          kSyncBetweenTiles>(
          (params.seqlen_q + kBlockM - 1) / kBlockM,
          params.h,
          params.b,
          head_shared_rab,
          [&](HstuWsTileCoord const& coord) {
            run_load_tile(coord.bidb, coord.bidh, coord.m_block);
          });
    // Load warp path exits here.  Active load warp has also completed O TMA-store.

  // ============================================================
  } else {
  // MATH WARP PATH  (warps 0-7, threads 0-255)
  // ============================================================
    static constexpr int kMathWarpRegBudget =
        Kernel_traits::kHeadDim > 128 ? 232 : 224;
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" : : "n"(kMathWarpRegBudget));

    constexpr bool Is_causal    = Kernel_traits::Is_causal;
    constexpr bool Is_target    = Kernel_traits::Is_target;
    constexpr bool Is_context   = Kernel_traits::Is_context;
    constexpr bool Is_arbitrary = Kernel_traits::Is_arbitrary;
    constexpr int  kNFunc       = Kernel_traits::kNFunc;
    constexpr bool Is_local     = Kernel_traits::Is_local;
    constexpr bool Has_rab      = Kernel_traits::Has_rab;
    constexpr bool Use_rab_smem = Kernel_traits::kUseRabSmem;
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

    __syncthreads();  // One-time S1: load warp has initialized WS mbarriers.

    auto run_math_tile = [&](const int bidb, const int bidh, int m_block) {
      const HstuBlockInfo<Kernel_traits, Params> binfo(params, bidb);
      // Early exit 1: before any sync — both branches exit simultaneously.
      if (m_block * kBlockM >= binfo.actual_seqlen_q_padded) return;

      char* smem_q    = reinterpret_cast<char*>(smem_);
      char* smem_func = reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsFuncOffset;
      int* sn_valid_block_max = reinterpret_cast<int*>(smem_func);
      int* sf_min_ptr = reinterpret_cast<int*>(sn_valid_block_max) + 1;
      int* sf_max_ptr = sf_min_ptr + (kNFunc/2 + 1);

      const HstuWsTileInfo tile_info =
          hstu_ws_make_tile_info<Kernel_traits>(params, binfo, m_block);
      const int actual_seqlen_q        = tile_info.actual_seqlen_q;
      const int actual_seqlen_k        = tile_info.actual_seqlen_k;
      const int actual_seqlen_q_padded = tile_info.actual_seqlen_q_padded;
      const int actual_seqlen_t        = tile_info.actual_seqlen_t;
      const int actual_seqlen_c        = tile_info.actual_seqlen_c;
      const int actual_seqlen_h        = tile_info.actual_seqlen_h;
      const int actual_seqlen_offset   = tile_info.actual_seqlen_offset;
      const int last_page_offset       = tile_info.last_page_offset;
      const int n_block_history        = tile_info.n_block_history;
      const int n_block_paged          = tile_info.n_block_paged;
      int n_block_min                  = tile_info.n_block_min;
      int n_block_max                  = tile_info.n_block_max;
      const int n_masking_steps        = tile_info.n_masking_steps;
      const bool is_jump               = tile_info.is_jump;

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

      // Early exit 2 (after Is_arbitrary): math warps write zeros.
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
        Tensor col_min = make_tensor<int>(make_shape(size<0>(gMinFunc)));
        Tensor col_max = make_tensor<int>(make_shape(size<0>(gMaxFunc)));
        int prev_block_row    = -1;
        int row               = 0;
        [[maybe_unused]] int tgt_col_lft = 0;
  #pragma unroll
        for (int flat = 0; flat < size(tSrS); ++flat) {
          const auto coord    = tScS(flat);
          const int block_row = int(get<Row>(coord));
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

      // SFP (unit scale for P) — separate buffer so real SFA SMEM stays valid for per-tile GEMM1 s2r.
      for (int i = tidx_math; i < kBlockM; i += kNMathThreads)
        smem_sfp_ptr[i] = 0x7f7f7f7f;

      // ===== MATH WARP DOUBLE-BUFFER QMMA CONSUMER LOOP =====
      // V^T s2r uses LDSM_T directly from MN_SW128 SMEM.
      // Under AtomLayout <_8,_1,_1>, all 8 warps load all 4 N-slabs (nw=0..3 outer loop);
      // 16B alignment: d_start = nw*32+d_mat ∈ multiples of 16, XOR swizzle_xor also multiple of 16.
      static_assert(kHeadDim % 32 == 0 && kBlockN % 16 == 0,
          "LDSM_T requires kHeadDim divisible by 32 and kBlockN divisible by 16.");
      // Q (sQ_persist) and SFA are written by TMA before the loop (guaranteed visible after
      // wait_mbar_parity) and never change. Hoisting eliminates 4 ldmatrix + 1 LDS
      // per N-tile.
      // tCrSFP is not hoisted: smem_sfp_ptr is written by distributed thread writes
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
        const int lane_q    = tidx_math & 31;
        const int warp_m_q  = tidx_math / 32;
        const int mat_num_q = lane_q >> 3;
        const int mat_row_q = lane_q & 7;
        const int m_abs_q   = warp_m_q * 16 + ((mat_num_q & 1) << 3) + mat_row_q;
        const int k_half_q  = mat_num_q >> 1;
        const uint32_t smem_base_q =
            static_cast<uint32_t>(__cvta_generic_to_shared(q_persist_base));
        auto tXrQ = recast<uint32_t>(tCrQ);
        SmemLayoutQ_SW128 layout_q;
        CUTE_UNROLL
        for (int kb = 0; kb < kHeadDim / 32; ++kb) {
          const int k_start = kb * 32 + (k_half_q << 4);
          const uint32_t addr = smem_base_q + (uint32_t)layout_q(m_abs_q, k_start);
          asm volatile(
              "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
              : "=r"(tXrQ(4 * kb + 0)), "=r"(tXrQ(4 * kb + 1)),
                "=r"(tXrQ(4 * kb + 2)), "=r"(tXrQ(4 * kb + 3))
              : "r"(addr));
        }
      }
      if constexpr (kHeadDim <= 128) {
        if ((tidx_math & 31) == 0) {
          arrive_mbar(q_empty_mbar_ptr);
        }
      }

      // Per-tile lambda: instantiated with kIsMasking=true (causal diagonal)
      // and kIsMasking=false (steady-state unmasked) to allow compile-time dead-code
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

          wait_mbar_parity(
              smem_base32 + (uint32_t)kSmemMbar0Offset + 96u,
              (uint32_t)o_empty_wait_parity);
          o_empty_wait_parity ^= 1;
          hstu_ws_store_o_stmatrix<kHeadDim, OutElement>(smem_o, rO, tidx_math);

          const int valid_rows = actual_seqlen_q - m_block * kBlockM;
          if (valid_rows < kBlockM) {
            asm volatile("bar.sync 1, 256;\n" : : : "memory");
            hstu_ws_zero_oob_o_rows<kBlockM, kHeadDim, kNMathThreads, OutElement>(
                smem_o, valid_rows, tidx_math);
            asm volatile("bar.sync 1, 256;\n" : : : "memory");
          }

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
          const int n_row_sfb = (tidx_math & 31) >> 2;
          constexpr int kNAtomsSFB = kBlockN / 8;
          CUTE_UNROLL
          for (int nr = 0; nr < kNAtomsSFB; ++nr) {
            tCrSFB(0, nr, 0) = smem_sfb_cur[nr * 8 + n_row_sfb];
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
            hstu_ws_load_b_z_pattern<kBlockN>(
                sK_cur_pi, tCrK, 0, kHeadDim / 32, kHeadDim, tidx_math);
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
              tCrSFA0(i) = hstu_ws_replicate_e8m0_lane(tCrSFA(i), 0);
            }
            CUTE_UNROLL
            for (int i = 0; i < size(tCrSFB); ++i) {
              tCrSFB0(i) = hstu_ws_replicate_e8m0_lane(tCrSFB(i), 0);
            }
            auto tCrSFA0_frg = BS1::transform_fragment_for_qmma(tCrSFA0);
            auto tCrSFB0_frg = BS1::transform_fragment_for_qmma(tCrSFB0);

            Tensor sQ_chunk0 = make_tensor(
                make_smem_ptr(q_persist_base),
                SmemLayoutQChunk_SW128{});
            auto sQ_chunk0_pi = as_position_independent_swizzle_tensor(sQ_chunk0);
            Tensor tCrQ0 = thr_mma_g1.partition_fragment_A(sQ_chunk0_pi);
            clear(tCrQ0);
            hstu_ws_load_a_z_pattern_layout(
                q_persist_base, SmemLayoutQ_SW128{}, tCrQ0, 0, 4, tidx_math);

            Tensor sK_chunk0 = make_tensor(
                make_smem_ptr(sK_cur),
                SmemLayoutKChunk_SW128{});
            auto sK_chunk0_pi = as_position_independent_swizzle_tensor(sK_chunk0);
            Tensor tCrK0 = thr_mma_g1.partition_fragment_B(sK_chunk0_pi);
            clear(tCrK0);
            hstu_ws_load_b_z_pattern_layout<kBlockN>(
                sK_cur, SmemLayoutK_SW128{}, tCrK0, 0, 4, tidx_math);

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
              tCrSFA1(i) = hstu_ws_replicate_e8m0_lane(tCrSFA(i), 1);
            }
            CUTE_UNROLL
            for (int i = 0; i < size(tCrSFB); ++i) {
              tCrSFB1(i) = hstu_ws_replicate_e8m0_lane(tCrSFB(i), 1);
            }
            auto tCrSFA1_frg = BS1::transform_fragment_for_qmma(tCrSFA1);
            auto tCrSFB1_frg = BS1::transform_fragment_for_qmma(tCrSFB1);

            Tensor sQ_chunk1 = make_tensor(
                make_smem_ptr(q_persist_base),
                SmemLayoutQChunk_SW128{});
            auto sQ_chunk1_pi = as_position_independent_swizzle_tensor(sQ_chunk1);
            Tensor tCrQ1 = thr_mma_g1.partition_fragment_A(sQ_chunk1_pi);
            clear(tCrQ1);
            hstu_ws_load_a_z_pattern_layout(
                q_persist_base, SmemLayoutQ_SW128{}, tCrQ1, 4, 4, tidx_math);

            Tensor sK_chunk1 = make_tensor(
                make_smem_ptr(sK_cur),
                SmemLayoutKChunk_SW128{});
            auto sK_chunk1_pi = as_position_independent_swizzle_tensor(sK_chunk1);
            Tensor tCrK1 = thr_mma_g1.partition_fragment_B(sK_chunk1_pi);
            clear(tCrK1);
            hstu_ws_load_b_z_pattern_layout<kBlockN>(
                sK_cur, SmemLayoutK_SW128{}, tCrK1, 4, 4, tidx_math);

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
          hstu_ws_consume_rab_bs<Kernel_traits>(
              params, thr_mma_g1, acc_s, sRab, smem_base32, rab_ready_wait_parity0,
              bidb, bidh, m_block, nb, false,
              actual_seqlen_q, actual_seqlen_k, actual_seqlen_h,
              actual_seqlen_offset, last_page_offset, n_block_paged);
        }
        // Masking is compile-time specialized. For RAB, apply the mask
        // first on masked tiles so RAB add can skip -inf entries without
        // recomputing the mask predicate on the math critical path.
        // kIsMasking=true always applies apply_mask_bs; false keeps only varlen-end checks.
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
          hstu_ws_consume_rab_bs<Kernel_traits>(
              params, thr_mma_g1, acc_s, sRab, smem_base32, rab_ready_wait_parity0,
              bidb, bidh, m_block, nb, mask_applied,
              actual_seqlen_q, actual_seqlen_k, actual_seqlen_h,
              actual_seqlen_offset, last_page_offset, n_block_paged);
        }
        for (int i = 0; i < size(acc_s); ++i) acc_s(i) *= params.alpha;
        fast_silu(acc_s);

        // Pre-convert P to packed FP8 before the P-write address calculation.
        // This ends acc_s's live range early and avoids spilling acc_o.
        // Flat C-fragment order must be preserved for AtomLayout <8,1,1>.
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
          //   Covers all 128 uint32 slots (16 N-atoms x 4 K-blocks x 2 regs).
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

      // For causal (not arbitrary/local), split masked and steady-state tiles to allow compile-time
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
          // Causal diagonal tiles: apply_mask_bs compiled in.
          for (; n_valid >= n_block_min && masking_step < n_masking_steps; ++masking_step, --n_valid)
            n_valid = run_n_tile(n_valid, n_valid, masking_step,
                false,
                (!Is_target && !Is_context && n_valid == n_block_min),
                std::integral_constant<int, 1>{});
          // Steady-state tiles: apply_mask_bs compile-time eliminated.
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

      // Some paths still use the post-loop epilogue when in-mainloop store did not run.
      if constexpr (kStoreOInMainloop) {
        if (!o_epilogue_done) {
          store_o_epilogue();
        }
      } else {
        store_o_epilogue();
      }
    };

    // Same scheduler as the load branch.
    constexpr bool kSyncBetweenTiles = Has_rab;
    const bool head_shared_rab = Has_rab && params.h_rab == 1 && params.h > 1;
    hstu_ws_run_static_schedule<
        Use_full_persistent,
        Use_paired_persistent,
        kSyncBetweenTiles>(
        (params.seqlen_q + kBlockM - 1) / kBlockM,
        params.h,
        params.b,
        head_shared_rab,
        [&](HstuWsTileCoord const& coord) {
          run_math_tile(coord.bidb, coord.bidh, coord.m_block);
        });
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// WS TMA kernel entry: launched with kNThreads=384.
// Q, K, V^T, Q-SF, K-SF, and V-SF all via TMA.
template <typename Kernel_traits, typename Params>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1)
hstu_fwd_kernel_sm120_fp8_ws_tma(
    __grid_constant__ Params const params) {
  constexpr bool Use_paired_persistent =
      Kernel_traits::Is_causal &&
      !Kernel_traits::Is_context &&
      !Kernel_traits::Is_local &&
      !Kernel_traits::Is_arbitrary &&
      (!Kernel_traits::Is_target || Kernel_traits::Paged_KV) &&
      (Kernel_traits::kHeadDim <= 128 ||
       (Kernel_traits::Paged_KV && !Kernel_traits::Has_rab));
  constexpr bool Use_full_persistent = !Use_paired_persistent;
  static_assert(Use_full_persistent || Use_paired_persistent);

  hstu_compute_attn_1rowblock_sm120_fp8_ws<
      Kernel_traits,
      Use_full_persistent,
      Use_paired_persistent>(params);
}
