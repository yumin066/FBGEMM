// TMA B-operand unit test aligned with SM120BlockScaledKernel from sm120_blockscaled_gemm_impl.cuh.
// All types, barrier semantics, thread/warp structure come directly from the reference impl.
//
// Thread structure mirrors SM120BlockScaledKernel::operator():
//   warps [0, kNumMathWarps)          : math warps  → wait full_mbar, read sB, arrive empty_mbar
//   warp  kNumMathWarps               : epi warp     → dummy (no store needed in this unit test)
//   warp  kNumMathWarps + 1 (ab_warp) : AB TMA warp  → issue B TMA only (skip A), phase=1 start
//   warp  kNumMathWarps + 2           : sf warp       → idle
//   remaining TMA warps               : idle
//
// Barrier init exactly matches reference (is_tma_thread only, fence_barrier_init, NO pre-priming).
// The first k_tile_count=AB_Stages copies are issued without waiting (mbarrier.init satisfies wait(1)).

#if defined(__CUDA_ARCH__) && !defined(CUTE_ARCH_TMA_SM120_ENABLED)
#define CUTE_ARCH_TMA_SM120_ENABLED
#endif
#if defined(__CUDA_ARCH__) && !defined(CUTE_ARCH_TMA_SM90_ENABLED)
#define CUTE_ARCH_TMA_SM90_ENABLED
#endif

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "/home/minyu/project/shopee/fbgemm-hstu/6KD_fp8_block_scale/3rdparty/cutlass/include/cute/arch/copy_sm90_tma.hpp"
#include "/home/minyu/project/shopee/fbgemm-hstu/6KD_fp8_block_scale/3rdparty/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp"
#include "/home/minyu/project/shopee/fbgemm-hstu/6KD_fp8_block_scale/3rdparty/cutlass/include/cute/tensor.hpp"
#include "/home/minyu/project/shopee/fbgemm-hstu/6KD_fp8_block_scale/3rdparty/cutlass/include/cutlass/arch/barrier.h"
#include "/home/minyu/project/shopee/fbgemm-hstu/6KD_fp8_block_scale/3rdparty/cutlass/include/cutlass/gemm/gemm.h"
#include "/home/minyu/project/shopee/fbgemm-hstu/6KD_fp8_block_scale/3rdparty/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp"
#include "/home/minyu/project/shopee/fbgemm-hstu/6KD_fp8_block_scale/kernels/include/sm120_blockscaled_gemm/sm120_blockscaled_utils.cuh"
#include "/home/minyu/project/shopee/fbgemm-hstu/6KD_fp8_block_scale/kernels/include/sm120_blockscaled_gemm/sm120_blockscaled_gemm_impl.cuh"

using namespace cute;

namespace tma_b_ut {

// ---- KT types (identical to what SM120BlockScaledKernel uses) ----
using KT = sm120_blockscaled_gemm::SM120BlockScaledBuilder<64, 128, 4>;

// ---- Kernel types directly from the reference impl ----
using SM120Kernel = sm120_blockscaled_gemm::SM120BlockScaledKernel<KT>;
using SharedStorage = SM120Kernel::SharedStorage;

// Use all barrier/producer types from KT (same as SM120BlockScaledKernel)
using FullBarrier        = typename KT::FullBarrier;
using EmptyBarrier       = typename KT::EmptyBarrier;
using ProducerBarrierType = typename KT::ProducerBarrierType;

// Thread counts from reference
static constexpr int kNumTMAThreads    = SM120Kernel::kNumTMAThreads;   // 128
static constexpr int kNumMathThreads   = SM120Kernel::kNumMathThreads;
static constexpr int MaxThreadsPerBlock = SM120Kernel::MaxThreadsPerBlock;

// SMEM sizes
static constexpr int kSmemBytes        = SM120Kernel::kSmemSize;
// PSS reserves 1024 B from the front of the dynamic SMEM bank → user smem ptr = +0x400.
// Allocate 1024 extra bytes so BarrierStorage still fits within the hardware range.
static constexpr int kSmemBytesWithPSS = kSmemBytes + 1024;

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t _err = (expr);                                                  \
    if (_err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                  \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// Params struct for the kernel
template <typename TMA_B_t>
struct TmaCopyBParams {
  TMA_B_t tma_b;
  void*   out_ptr;   // device output buffer [kTileN * k_tile_count * kTileK] elements
  int     N;
  int     K;
  int     L;
  int     n_coord;
  int     l_coord;
};

template <typename TMA_B_t>
__global__
__launch_bounds__(MaxThreadsPerBlock, 1)
void tma_copy_b_kernel(CUTLASS_GRID_CONSTANT TmaCopyBParams<TMA_B_t> const params) {
  using X = Underscore;

  extern __shared__ char smem[];
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

  int thread_idx     = int(threadIdx.x);
  int warp_idx       = canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  bool is_tma_thread = (warp_idx == 0) && lane_predicate;

  // ---- prefetch TMA descriptor (same as SM120BlockScaledKernel) ----
  if (is_tma_thread) {
    cute::prefetch_tma_descriptor(params.tma_b.get_tma_descriptor());
  }
  __syncthreads();

  // ---- get barrier pointers via the reference helper ----
  auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar,
        store_full_mbar, store_empty_mbar] = SM120Kernel::get_mbarriers(shared_storage);

  // ---- barrier init: EXACTLY as in SM120BlockScaledKernel::operator() ----
  if (is_tma_thread) {
    #pragma unroll
    for (uint32_t i = 0; i < KT::SF_Stages; ++i) {
      sf_full_mbar[i].init(1);
      sf_empty_mbar[i].init(kNumMathThreads);
    }
    #pragma unroll
    for (uint32_t i = 0; i < KT::AB_Stages; ++i) {
      ab_full_mbar[i].init(1);
      ab_empty_mbar[i].init(kNumMathThreads);
    }
    store_full_mbar[0].init(kNumMathThreads);
    store_empty_mbar[0].init(1);
    cutlass::arch::fence_barrier_init();
    // NO pre-priming: after init, mbarrier satisfies wait(1) by PTX convention,
    // so producer phase=1 can issue the first AB_Stages TMAs without waiting.
  }
  __syncthreads();

  // ---- compute tile counts ----
  int32_t sf_tile_count = (params.K + 511) / 512;
  int32_t k_tile_count  = sf_tile_count * KT::kNumTileKPerSF;

  // ---- build TMA views (same as SM120BlockScaledKernel::load_ab) ----
  auto mB_nkl = params.tma_b.get_tma_tensor(make_shape(params.N, params.K, params.L));
  auto gB_nkl = local_tile(mB_nkl,
                            typename KT::TileShape{},
                            make_coord(_, _, _),
                            Step<X, _1, _1>{});            // (kTileN, kTileK, n, k, l)

  auto block_tma_b = params.tma_b.get_slice(0);
  auto gB  = gB_nkl(_, _, params.n_coord, _, params.l_coord);   // (kTileN, kTileK, k)
  auto tBgB = block_tma_b.partition_S(gB);                      // (TMA, TMA_N, TMA_K, k)

  auto sB_ = make_tensor(
      make_smem_ptr(shared_storage.tensors.load.smem_B.begin()),
      typename KT::SmemLayoutB{});
  auto sB   = as_position_independent_swizzle_tensor(sB_);       // (kTileN, kTileK, AB_Stages)
  auto tBsB = block_tma_b.partition_D(sB);                      // (TMA, TMA_N, TMA_K, AB_Stages)

  // ---- warp dispatch ----
  constexpr int epi_warp_idx = KT::kNumMathWarps;
  constexpr int ab_warp_idx  = KT::kNumMathWarps + 1;

  // ==================================================================
  // AB TMA warp: mirrors SM120BlockScaledKernel::load_ab (B only)
  // ==================================================================
  if (warp_idx == ab_warp_idx) {
    if (lane_predicate) {
      uint32_t phase       = 1;
      uint32_t store_phase = 1;

      // store_empty_mbar.init(1) satisfies wait(1) immediately → first block can proceed
      store_empty_mbar[0].wait(store_phase);
      store_phase ^= 1;

      for (int32_t k_tile_idx = 0; k_tile_idx < k_tile_count; k_tile_idx += KT::AB_Stages) {
        cute::for_each(cute::make_int_sequence<KT::AB_Stages>{}, [&](auto write_stage) {
          // ab_empty_mbar.init(kNumMathThreads) satisfies wait(1) immediately on first pass
          ab_empty_mbar[write_stage].wait(phase);
          auto& ab_full_barrier = ab_full_mbar[write_stage];
          auto tma_copy_b = params.tma_b.with(
              *recast_ptr<ProducerBarrierType>(&ab_full_barrier));
          cute::copy(tma_copy_b,
                     tBgB(_, _, _, k_tile_idx + write_stage),
                     tBsB(_, _, _, write_stage));
          // Declare expected transaction bytes for B only (no A in this unit test)
          ab_full_mbar[write_stage].arrive_and_expect_tx(KT::TmaTransactionBytesB);
        });
        phase ^= 1;
      }
    }
    __syncwarp();
  }

  // ==================================================================
  // Math warps: wait for TMA, copy sB stage to output, signal producer
  // ==================================================================
  else if (warp_idx < KT::kNumMathWarps) {
    uint32_t ab_phase = 0;

    // Output pointer: each k_tile writes kTileN*kTileK elements.
    // We collect only the first k_tile (stage 0) result for the correctness check.
    auto* out_ptr = reinterpret_cast<typename KT::ElementB*>(params.out_ptr);

    for (int32_t k_tile_idx = 0; k_tile_idx < k_tile_count; k_tile_idx += KT::AB_Stages) {
      cute::for_each(cute::make_int_sequence<KT::AB_Stages>{}, [&](auto read_stage) {
        ab_full_mbar[read_stage].wait(ab_phase);

        // Write sB[read_stage] → output, partitioned across all math threads
        int total = KT::kTileN * KT::kTileK;
        int threads_in_math = kNumMathThreads;
        int stage_offset = (k_tile_idx + read_stage) * total;
        for (int idx = thread_idx; idx < total; idx += threads_in_math) {
          int n = idx / KT::kTileK;
          int k = idx % KT::kTileK;
          out_ptr[stage_offset + idx] = sB(n, k, read_stage);
        }

        ab_empty_mbar[read_stage].arrive();
      });
      ab_phase ^= 1;
    }
  }

  // ==================================================================
  // Epi/store warp and remaining TMA warps: idle (no store in this test)
  // ==================================================================
  // (other warps simply fall through and exit)
}

}  // namespace tma_b_ut

int main() {
  using namespace tma_b_ut;

  std::cout << "[host] KT::kTileN=" << KT::kTileN
            << " KT::kTileK=" << KT::kTileK
            << " KT::AB_Stages=" << KT::AB_Stages
            << " kNumTileKPerSF=" << KT::kNumTileKPerSF << std::endl;
  std::cout << "[host] kNumMathThreads=" << kNumMathThreads
            << " kNumTMAThreads=" << kNumTMAThreads
            << " MaxThreadsPerBlock=" << MaxThreadsPerBlock << std::endl;
  std::cout << "[host] kSmemBytes=" << kSmemBytes
            << " kSmemBytesWithPSS=" << kSmemBytesWithPSS << std::endl;

#ifndef UNIT_TEST_B_K
#define UNIT_TEST_B_K 512
#endif
  constexpr int N       = KT::kTileN;   // 128
  constexpr int K       = UNIT_TEST_B_K;
  constexpr int L       = 1;
  constexpr int n_coord = 0;
  constexpr int l_coord = 0;

  int32_t sf_tile_count = (K + 511) / 512;
  int32_t k_tile_count  = sf_tile_count * KT::kNumTileKPerSF;
  int elem_count        = N * K * L;               // GMEM tensor size
  int out_elem_count    = N * KT::kTileK * k_tile_count;  // one stage per k_tile

  std::cout << "[host] K=" << K << " sf_tile_count=" << sf_tile_count
            << " k_tile_count=" << k_tile_count
            << " elem_count=" << elem_count
            << " out_elem_count=" << out_elem_count << std::endl;

  using ElementB = typename KT::ElementB;

  // Initialize host tensor B with a simple pattern
  std::vector<ElementB> h_b(elem_count);
  for (int i = 0; i < elem_count; ++i) {
    float v = static_cast<float>((i % 17) - 8) * 0.25f;
    h_b[i] = ElementB(v);
  }

  ElementB* d_b   = nullptr;
  ElementB* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_b,   elem_count     * sizeof(ElementB)));
  CUDA_CHECK(cudaMalloc(&d_out, out_elem_count * sizeof(ElementB)));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), elem_count * sizeof(ElementB), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, out_elem_count * sizeof(ElementB)));

  // Tensor B layout: (N, K, L) with row-major within each batch
  int64_t dB0 = K;          // stride between rows
  int64_t dB2 = (int64_t)N * K;  // stride between batches
  typename KT::StrideB dB = make_stride(dB0, Int<1>{}, dB2);

  auto tensor_B = make_tensor(make_gmem_ptr(d_b),
                               make_layout(make_shape(N, K, L), dB));

  // Build TMA copy for B (same call as SM120BlockScaledKernel::to_underlying_arguments)
  typename KT::TMA_B tma_b = make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_B,
      typename KT::SmemLayoutB{}(_, _, Int<0>{}),
      make_shape(shape<1>(typename KT::TileShape{}), shape<2>(typename KT::TileShape{})),
      _1{});

  std::cout << "[host] TMA_B descriptor built" << std::endl;

  using TMA_B_t = typename KT::TMA_B;
  TmaCopyBParams<TMA_B_t> params{tma_b, d_out, N, K, L, n_coord, l_coord};

  auto kernel = &tma_copy_b_kernel<TMA_B_t>;
  // PSS reserves 1024 bytes at the front of the SMEM bank (smem[] starts at +0x400).
  // Must request kSmemBytesWithPSS so SharedStorage + BarrierStorage fit within
  // the user-accessible region [0x400, 0x400+kSmemBytesWithPSS).
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytesWithPSS));

  // Launch with PSS (ProgrammaticStreamSerialization) exactly as the reference kernel
  cudaLaunchConfig_t    launch_config{};
  cudaLaunchAttribute   attrs[1]{};
  attrs[0].id                                             = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed     = 1;
  launch_config.gridDim         = dim3(1, 1, 1);
  launch_config.blockDim        = dim3(MaxThreadsPerBlock, 1, 1);
  launch_config.dynamicSmemBytes = kSmemBytesWithPSS;
  launch_config.stream           = 0;
  launch_config.attrs            = attrs;
  launch_config.numAttrs         = 1;

  std::cout << "[host] launching kernel with " << MaxThreadsPerBlock
            << " threads, smem=" << kSmemBytesWithPSS << " bytes" << std::endl;

  CUDA_CHECK(cudaLaunchKernelEx(&launch_config, kernel, params));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy output back
  std::vector<ElementB> h_out(out_elem_count);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                        out_elem_count * sizeof(ElementB), cudaMemcpyDeviceToHost));

  // Verify each k_tile against the reference
  int mismatch = 0;
  for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < KT::kTileK; ++k) {
        // GMEM source index: B[n_coord*N + n][k_tile*kTileK + k][l_coord]
        int src_k   = k_tile * KT::kTileK + k;
        int src_idx = (n_coord * N + n) * dB0 + src_k + l_coord * dB2;
        int dst_idx = k_tile * (N * KT::kTileK) + n * KT::kTileK + k;

        const auto* out_bytes = reinterpret_cast<const uint8_t*>(&h_out[dst_idx]);
        const auto* ref_bytes = reinterpret_cast<const uint8_t*>(&h_b[src_idx]);
        if (out_bytes[0] != ref_bytes[0]) {
          if (mismatch < 8) {
            std::cerr << "mismatch k_tile=" << k_tile
                      << " n=" << n << " k=" << k
                      << " dst=" << dst_idx << " got=" << (int)out_bytes[0]
                      << " exp=" << (int)ref_bytes[0] << std::endl;
          }
          ++mismatch;
        }
      }
    }
  }

  std::cout << "[TMA B unit test] mismatches=" << mismatch << std::endl;
  if (mismatch == 0) {
    std::cout << "[TMA B unit test] PASS" << std::endl;
  } else {
    std::cout << "[TMA B unit test] FAIL" << std::endl;
  }

  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_out));
  return mismatch == 0 ? 0 : 1;
}
