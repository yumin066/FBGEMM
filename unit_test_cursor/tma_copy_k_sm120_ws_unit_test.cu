#if defined(__CUDA_ARCH__) && !defined(CUTE_ARCH_TMA_SM120_ENABLED)
#define CUTE_ARCH_TMA_SM120_ENABLED
#endif
#if defined(__CUDA_ARCH__) && !defined(CUTE_ARCH_TMA_SM90_ENABLED)
#define CUTE_ARCH_TMA_SM90_ENABLED
#endif

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "../external/cutlass/include/cute/arch/copy_sm90_tma.hpp"
#include "../external/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp"
#include "../external/cutlass/include/cute/tensor.hpp"
#include "../external/cutlass/include/cutlass/numeric_types.h"

using namespace cute;

namespace tma_k_ut {

#ifndef UNIT_TEST_ENABLE_TMA_PREFETCH
#define UNIT_TEST_ENABLE_TMA_PREFETCH 0
#endif

using Element = cutlass::float_e4m3_t;
constexpr int kBlockN = 128;
constexpr int kHeadDim = 256;
using SmemLayoutAtomSW128 = GMMA::Layout_K_SW128_Atom<Element>;
using SmemLayoutK_TMA = decltype(tile_to_shape(
    SmemLayoutAtomSW128{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
constexpr int kSmemKBytes = size(SmemLayoutK_TMA{}) * sizeof(Element);
constexpr int kSmemBytes = kSmemKBytes + 8;  // +1 mbarrier (uint64_t)

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t _err = (expr);                                                  \
    if (_err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                  \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

template <typename TMA_K_t>
struct TmaCopyKParams {
  TMA_K_t tma_k;
  void* out_ptr;  // [kBlockN, kHeadDim] row-major
  int total_k;
  int d;
  int h_k;
  int head_idx;
  int nb_abs;
};

__device__ inline void wait_mbar_parity(uint64_t* mbar, uint32_t parity) {
  uint32_t maddr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  uint32_t done = 0;
  do {
    asm volatile(
        "{.reg .pred p;\n"
        "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;}\n"
        : "=r"(done)
        : "r"(maddr), "r"(parity));
  } while (!done);
}

template <typename TMA_K_t>
__global__ void tma_copy_k_only_kernel(__grid_constant__ TmaCopyKParams<TMA_K_t> const params) {
  extern __shared__ char smem[];
  auto sK = make_tensor(
      make_smem_ptr(reinterpret_cast<Element*>(smem)), SmemLayoutK_TMA{});
  uint64_t* load_mbar_ptr = reinterpret_cast<uint64_t*>(smem + kSmemKBytes);

  if (threadIdx.x == 0) {
// Optional prefetch path for debugging. Some environments assert here when
// CUTE_ARCH_TMA_SM90_ENABLED is not enabled in CuTe build config.
#if UNIT_TEST_ENABLE_TMA_PREFETCH
    cute::prefetch_tma_descriptor(params.tma_k.get_tma_descriptor());
#endif
    uint32_t laddr = static_cast<uint32_t>(__cvta_generic_to_shared(load_mbar_ptr));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(laddr), "r"(1));
  }
  asm volatile("fence.proxy.async.shared::cta;\n" : : : "memory");
  __syncthreads();

  // ----- Extracted TMA K copy path from hstu_fwd_kernel_fp8_ws.h -----
  auto mK_tma = params.tma_k.get_tma_tensor(
      make_shape(params.total_k, params.d, params.h_k));
  auto gK_head = mK_tma(_, _, params.head_idx);
  auto gK_tiles = local_tile(
      gK_head, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _));
  auto tma_slice_K = params.tma_k.get_slice(0);
  auto tKsK_d = tma_slice_K.partition_D(sK);
  auto tKgK_tma = tma_slice_K.partition_S(gK_tiles(_, _, _, Int<0>{}));

  if (threadIdx.x == 0) {
    cute::copy(
        params.tma_k.with(*load_mbar_ptr),
        tKgK_tma(_, _, _, params.nb_abs),
        tKsK_d);
    uint32_t laddr = static_cast<uint32_t>(__cvta_generic_to_shared(load_mbar_ptr));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :
                 : "r"(laddr), "r"(kSmemKBytes));
    wait_mbar_parity(load_mbar_ptr, 0);
  }
  __syncthreads();
  // ---------------------------------------------------------------

  Element* out_ptr = reinterpret_cast<Element*>(params.out_ptr);
  for (int idx = threadIdx.x; idx < kBlockN * kHeadDim; idx += blockDim.x) {
    int n = idx / kHeadDim;
    int d = idx % kHeadDim;
    out_ptr[idx] = sK(n, d);
  }
}

}  // namespace tma_k_ut

int main() {
  using namespace tma_k_ut;
  std::cout << "[TMA K unit test] UNIT_TEST_ENABLE_TMA_PREFETCH="
            << UNIT_TEST_ENABLE_TMA_PREFETCH << std::endl;
  constexpr int total_k = 128;
  constexpr int d = kHeadDim;
  constexpr int h_k = 1;
  constexpr int head_idx = 0;
  constexpr int nb_abs = 0;
  constexpr int k_row_stride = d * h_k;
  constexpr int k_head_stride = d;
  constexpr int elem_count = total_k * d * h_k;
  constexpr int out_elem_count = kBlockN * kHeadDim;

  std::vector<uint8_t> h_k_bytes(elem_count);
  for (int i = 0; i < elem_count; ++i) {
    h_k_bytes[i] = static_cast<uint8_t>((i * 13 + 7) & 0xFF);
  }

  Element* d_k = nullptr;
  Element* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_k, elem_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_out, out_elem_count * sizeof(Element)));
  CUDA_CHECK(cudaMemcpy(
      d_k, h_k_bytes.data(), elem_count * sizeof(Element), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, out_elem_count * sizeof(Element)));

  auto tensor_k = make_tensor(
      make_gmem_ptr(d_k),
      make_shape(total_k, d, h_k),
      make_stride(k_row_stride, _1{}, k_head_stride));

  auto tma_k = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_k,
      SmemLayoutK_TMA{},
      make_shape(Int<kBlockN>{}, Int<kHeadDim>{}),
      Int<1>{});

  using TMA_K_t = decltype(tma_k);
  TmaCopyKParams<TMA_K_t> params{
      tma_k, d_out, total_k, d, h_k, head_idx, nb_abs};

  auto kernel = &tma_copy_k_only_kernel<TMA_K_t>;
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  kernel<<<1, 128, kSmemBytes>>>(params);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<Element> h_out(out_elem_count);
  CUDA_CHECK(cudaMemcpy(
      h_out.data(), d_out, out_elem_count * sizeof(Element), cudaMemcpyDeviceToHost));

  const uint8_t* out_bytes = reinterpret_cast<const uint8_t*>(h_out.data());
  int mismatch = 0;
  for (int n = 0; n < kBlockN; ++n) {
    for (int kk = 0; kk < kHeadDim; ++kk) {
      int src_idx = (nb_abs * kBlockN + n) * k_row_stride + kk + head_idx * k_head_stride;
      int dst_idx = n * kHeadDim + kk;
      if (out_bytes[dst_idx] != h_k_bytes[src_idx]) {
        if (mismatch < 8) {
          std::cerr << "mismatch@" << dst_idx << " got="
                    << static_cast<int>(out_bytes[dst_idx]) << " exp="
                    << static_cast<int>(h_k_bytes[src_idx]) << std::endl;
        }
        ++mismatch;
      }
    }
  }

  std::cout << "[TMA K unit test] mismatches=" << mismatch << std::endl;
  if (mismatch == 0) {
    std::cout << "[TMA K unit test] PASS" << std::endl;
  } else {
    std::cout << "[TMA K unit test] FAIL" << std::endl;
  }

  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_out));
  return mismatch == 0 ? 0 : 1;
}
