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

namespace tma_vt_ut {

#ifndef UNIT_TEST_ENABLE_TMA_PREFETCH
#define UNIT_TEST_ENABLE_TMA_PREFETCH 0
#endif

using Element = cutlass::float_e4m3_t;
constexpr int kBlockN = 128;
constexpr int kHeadDim = 256;
using SmemLayoutAtomSW128 = GMMA::Layout_MN_SW128_Atom<Element>;
using SmemLayoutVt_TMA = decltype(tile_to_shape(
    SmemLayoutAtomSW128{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));
constexpr int kSmemVtBytes = size(SmemLayoutVt_TMA{}) * sizeof(Element);
constexpr int kSmemBytes = kSmemVtBytes + 8;  // +1 mbarrier (uint64_t)
constexpr int kDumpRows = 8;
constexpr int kDumpCols = 8;

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t _err = (expr);                                                  \
    if (_err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                  \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

template <typename TMA_V_t>
struct TmaCopyVtParams {
  TMA_V_t tma_vt;
  Element const* gmem_v_ptr;  // original V semantic: [k, d, h]
  void* out_ptr;  // [kHeadDim, kBlockN] row-major => out[d, n]
  int total_k;
  int d;
  int h_k;
  int v_row_stride;
  int v_head_stride;
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

__device__ inline int raw_u8(Element x) {
  return static_cast<int>(reinterpret_cast<unsigned char const*>(&x)[0]);
}

template <typename TMA_V_t>
__global__ void tma_copy_vt_only_kernel(__grid_constant__ TmaCopyVtParams<TMA_V_t> const params) {
  extern __shared__ char smem[];
  auto sVt = make_tensor(
      make_smem_ptr(reinterpret_cast<Element*>(smem)), SmemLayoutVt_TMA{});
  uint64_t* load_mbar_ptr = reinterpret_cast<uint64_t*>(smem + kSmemVtBytes);

  if (threadIdx.x == 0) {
#if UNIT_TEST_ENABLE_TMA_PREFETCH
    cute::prefetch_tma_descriptor(params.tma_vt.get_tma_descriptor());
#endif
    uint32_t laddr = static_cast<uint32_t>(__cvta_generic_to_shared(load_mbar_ptr));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(laddr), "r"(1));
  }
  asm volatile("fence.proxy.async.shared::cta;\n" : : : "memory");
  __syncthreads();

  // ----- TMA Vt copy path: GMEM (k,d,h) -> view as (d,k,h) -> SMEM (d,n) -----
  auto mVt_tma = params.tma_vt.get_tma_tensor(
      make_shape(params.d, params.total_k, params.h_k));
  auto gVt_head = mVt_tma(_, _, params.head_idx);
  auto gVt_tiles = local_tile(
      gVt_head, Shape<Int<kHeadDim>, Int<kBlockN>>{}, make_coord(_, _));
  auto tma_slice_Vt = params.tma_vt.get_slice(0);
  auto tVtsVt_d = tma_slice_Vt.partition_D(sVt);
  auto tVtgVt_tma = tma_slice_Vt.partition_S(gVt_tiles(_, _, Int<0>{}, _));

  if (threadIdx.x == 0) {
    cute::copy(
        params.tma_vt.with(*load_mbar_ptr),
        tVtgVt_tma(_, _, _, params.nb_abs),
        tVtsVt_d);
    uint32_t laddr = static_cast<uint32_t>(__cvta_generic_to_shared(load_mbar_ptr));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :
                 : "r"(laddr), "r"(kSmemVtBytes));
    wait_mbar_parity(load_mbar_ptr, 0);
  }
  __syncthreads();
  // ---------------------------------------------------------------------------

  if (threadIdx.x == 0) {
    int base_k = params.nb_abs * kBlockN;
    printf("[dump] gmem(V) original bytes [k_local, d], head=%d, base_k=%d\n",
           params.head_idx, base_k);
    for (int nn = 0; nn < kDumpCols; ++nn) {
      printf("  g k=%d :", nn);
      for (int dd = 0; dd < kDumpRows; ++dd) {
        int k = base_k + nn;
        int src_idx = k * params.v_row_stride + dd + params.head_idx * params.v_head_stride;
        Element gv = params.gmem_v_ptr[src_idx];
        printf(" %3d", raw_u8(gv));
      }
      printf("\n");
    }

    printf("[dump] smem(sVt) bytes [d, k_local]\n");
    for (int dd = 0; dd < kDumpRows; ++dd) {
      printf("  s d=%d :", dd);
      for (int nn = 0; nn < kDumpCols; ++nn) {
        Element sv = sVt(dd, nn);
        printf(" %3d", raw_u8(sv));
      }
      printf("\n");
    }

    int mini_mismatch = 0;
    for (int dd = 0; dd < kDumpRows; ++dd) {
      for (int nn = 0; nn < kDumpCols; ++nn) {
        int k = base_k + nn;
        int src_idx = k * params.v_row_stride + dd + params.head_idx * params.v_head_stride;
        int gv = raw_u8(params.gmem_v_ptr[src_idx]);  // gmem V[k,d]
        int sv = raw_u8(sVt(dd, nn));                 // smem Vt[d,k]
        if (gv != sv) {
          if (mini_mismatch < 8) {
            printf("  [mini_mismatch] d=%d k_local=%d gV[k,d]=%d sVt[d,k]=%d\n",
                   dd, nn, gv, sv);
          }
          ++mini_mismatch;
        }
      }
    }
    printf("[dump] transpose relation check on %dx%d window: mismatches=%d\n",
           kDumpRows, kDumpCols, mini_mismatch);
  }
  __syncthreads();

  Element* out_ptr = reinterpret_cast<Element*>(params.out_ptr);
  for (int idx = threadIdx.x; idx < kHeadDim * kBlockN; idx += blockDim.x) {
    int dd = idx / kBlockN;
    int nn = idx % kBlockN;
    out_ptr[idx] = sVt(dd, nn);
  }
}

}  // namespace tma_vt_ut

int main() {
  using namespace tma_vt_ut;
  std::cout << "[TMA Vt unit test] UNIT_TEST_ENABLE_TMA_PREFETCH="
            << UNIT_TEST_ENABLE_TMA_PREFETCH << std::endl;

  constexpr int total_k = 128;
  constexpr int d = kHeadDim;
  constexpr int h_k = 1;
  constexpr int head_idx = 0;
  constexpr int nb_abs = 0;
  constexpr int v_row_stride = d * h_k;  // original V semantic: [k, d, h]
  constexpr int v_head_stride = d;
  constexpr int elem_count = total_k * d * h_k;
  constexpr int out_elem_count = kHeadDim * kBlockN;

  std::vector<uint8_t> h_v_bytes(elem_count);
  for (int i = 0; i < elem_count; ++i) {
    h_v_bytes[i] = static_cast<uint8_t>((i * 7 + 11) & 0xFF);
  }

  Element* d_v = nullptr;
  Element* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_v, elem_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_out, out_elem_count * sizeof(Element)));
  CUDA_CHECK(cudaMemcpy(
      d_v, h_v_bytes.data(), elem_count * sizeof(Element), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, out_elem_count * sizeof(Element)));

  // Build transposed-view tensor for TMA:
  // original semantic is [k, d, h], but we expose [d, k, h] to TMA.
  auto tensor_vt = make_tensor(
      make_gmem_ptr(d_v),
      make_shape(d, total_k, h_k),
      make_stride(_1{}, v_row_stride, v_head_stride));

  auto tma_vt = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_vt,
      SmemLayoutVt_TMA{},
      make_shape(Int<kHeadDim>{}, Int<kBlockN>{}),
      Int<1>{});

  using TMA_V_t = decltype(tma_vt);
  TmaCopyVtParams<TMA_V_t> params{
      tma_vt, d_v, d_out, total_k, d, h_k, v_row_stride, v_head_stride, head_idx, nb_abs};

  auto kernel = &tma_copy_vt_only_kernel<TMA_V_t>;
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
  for (int dd = 0; dd < kHeadDim; ++dd) {
    for (int nn = 0; nn < kBlockN; ++nn) {
      int src_idx =
          (nb_abs * kBlockN + nn) * v_row_stride + dd + head_idx * v_head_stride;
      int dst_idx = dd * kBlockN + nn;  // out[d, n]
      if (out_bytes[dst_idx] != h_v_bytes[src_idx]) {
        if (mismatch < 8) {
          std::cerr << "mismatch@" << dst_idx << " got="
                    << static_cast<int>(out_bytes[dst_idx]) << " exp="
                    << static_cast<int>(h_v_bytes[src_idx]) << std::endl;
        }
        ++mismatch;
      }
    }
  }

  std::cout << "[TMA Vt unit test] mismatches=" << mismatch << std::endl;
  if (mismatch == 0) {
    std::cout << "[TMA Vt unit test] PASS" << std::endl;
  } else {
    std::cout << "[TMA Vt unit test] FAIL" << std::endl;
  }

  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_out));
  return mismatch == 0 ? 0 : 1;
}
