"""cuTile D256 FP8 block-scale HSTU forward kernels."""

import math
import importlib.util
import os
import sys

import numpy as np


def _preload_cuda_tile_cext() -> None:
    if os.environ.get("CUDA_TILE_PRELOAD_CEXT") != "1":
        return
    cext_path = os.environ.get("CUDA_TILE_CEXT_PATH")
    if not cext_path or "cuda.tile._cext" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location("cuda.tile._cext", cext_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load cuda.tile._cext from {cext_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cuda.tile._cext"] = module
    spec.loader.exec_module(module)


try:
    _preload_cuda_tile_cext()
    import cuda.tile as ct

    # The required operations are still in the development surface in the local
    # cuTile checkout. Patch them in when using that tree directly.
    if not hasattr(ct, "mma_scaled"):
        from cuda.tile._stub import mma_scaled

        ct.mma_scaled = mma_scaled
    if not hasattr(ct, "float8_e8m0fnu"):
        from cuda.tile._datatype import float8_e8m0fnu

        ct.float8_e8m0fnu = float8_e8m0fnu
except Exception as exc:  # pragma: no cover - exercised by availability checks.
    ct = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


TILE_M_D256 = 64
TILE_N_D256 = 64
HEAD_DIM_D256 = 256
QK_SCALE_BLOCKS_D256 = 8
PV_SCALE_BLOCKS_BN64 = 2


def import_error() -> Exception | None:
    return _IMPORT_ERROR


if ct is not None:
    ConstBool = ct.Constant[bool]

    @ct.kernel(occupancy=2)
    def hstu_fp8_d256_persistent_kernel(
        Q,
        K,
        V,
        QScale,
        KScale,
        VScale,
        QLengths,
        KLengths,
        Out,
        alpha: float,
        scaling_seqlen: int,
        causal: ConstBool,
    ):
        """Static grid-stride persistent D256 FP8 HSTU forward.

        Q/K/V are dense tensors:
          Q:      [B, max_q, H, 256], fp8_e4m3
          K/V:    [B, max_k, H, 256], fp8_e4m3
          QScale: [B, H, max_q, 8], e8m0, one scale per 32-D block
          KScale: [B, H, max_k, 8], e8m0, one scale per 32-D block
          VScale: [B, H, ceil(max_k / 64), 2, 256], e8m0
          Out:    [B, max_q, H, 256], fp16
        """

        bid = ct.bid(0)
        num_tile_blocks = ct.num_blocks(0)
        batch_size = Q.shape[0]
        max_q = Q.shape[1]
        max_k = K.shape[1]
        num_heads = Q.shape[2]
        num_m_blocks = ct.cdiv(max_q, TILE_M_D256)
        total_tiles = batch_size * num_heads * num_m_blocks

        for linear_tile in range(bid, total_tiles, num_tile_blocks):
            m_block = linear_tile % num_m_blocks
            tmp = linear_tile // num_m_blocks
            head_idx = tmp % num_heads
            batch_idx = tmp // num_heads

            q_len = ct.load(QLengths, index=batch_idx, shape=())
            k_len = ct.load(KLengths, index=batch_idx, shape=())
            qk_offset = k_len - q_len

            q = ct.load(
                Q,
                index=(batch_idx, m_block, head_idx, 0),
                shape=(1, TILE_M_D256, 1, HEAD_DIM_D256),
                padding_mode=ct.PaddingMode.ZERO,
                latency=4,
                allow_tma=True,
            ).reshape((TILE_M_D256, HEAD_DIM_D256))
            q_scale = ct.load(
                QScale,
                index=(batch_idx, head_idx, m_block, 0),
                shape=(1, 1, TILE_M_D256, QK_SCALE_BLOCKS_D256),
                padding_mode=ct.PaddingMode.ZERO,
                latency=2,
                allow_tma=True,
            ).reshape((TILE_M_D256, QK_SCALE_BLOCKS_D256))

            acc_o = ct.full((TILE_M_D256, HEAD_DIM_D256), 0.0, dtype=ct.float32)

            num_n_blocks = ct.cdiv(max_k, TILE_N_D256)
            offs_m = (
                m_block * TILE_M_D256 + ct.arange(TILE_M_D256, dtype=np.int32)
            )[:, None]
            offs_n_base = ct.arange(TILE_N_D256, dtype=np.int32)[None, :]

            for n_block in range(num_n_blocks):
                k_tile = ct.load(
                    K,
                    index=(batch_idx, n_block, head_idx, 0),
                    shape=(1, TILE_N_D256, 1, HEAD_DIM_D256),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=5,
                    allow_tma=True,
                ).reshape((TILE_N_D256, HEAD_DIM_D256))
                k_t = ct.transpose(k_tile)
                k_scale = ct.load(
                    KScale,
                    index=(batch_idx, head_idx, n_block, 0),
                    shape=(1, 1, TILE_N_D256, QK_SCALE_BLOCKS_D256),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=2,
                    allow_tma=True,
                ).reshape((TILE_N_D256, QK_SCALE_BLOCKS_D256))
                k_scale_t = ct.transpose(k_scale)

                acc_s = ct.full((TILE_M_D256, TILE_N_D256), 0.0, dtype=ct.float32)
                acc_s = ct.mma_scaled(q, q_scale, k_t, k_scale_t, acc_s)

                offs_n = n_block * tile_n + offs_n_base
                valid = (offs_m < q_len) & (offs_n < k_len)
                if causal:
                    valid = valid & (offs_n <= (qk_offset + offs_m))
                acc_s = ct.where(valid, acc_s, -math.inf)

                s = acc_s * alpha
                p = s / (ct.exp(-s) + 1.0)
                p = ct.where(valid, p, 0.0)
                p_fp8 = p.astype(Q.dtype)
                p_scale = ct.full(
                    (tile_m, PV_SCALE_BLOCKS_BN64),
                    1.0,
                    dtype=QScale.dtype,
                )

                v = ct.load(
                    V,
                    index=(batch_idx, n_block, head_idx, 0),
                    shape=(1, TILE_N_D256, 1, HEAD_DIM_D256),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=5,
                    allow_tma=True,
                ).reshape((TILE_N_D256, HEAD_DIM_D256))
                v_scale = ct.load(
                    VScale,
                    index=(batch_idx, head_idx, n_block, 0, 0),
                    shape=(1, 1, 1, PV_SCALE_BLOCKS_BN64, HEAD_DIM_D256),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=2,
                    allow_tma=True,
                ).reshape((PV_SCALE_BLOCKS_BN64, HEAD_DIM_D256))

                acc_o = ct.mma_scaled(p_fp8, p_scale, v, v_scale, acc_o)

            if scaling_seqlen > 1:
                acc_o = acc_o / scaling_seqlen

            out_tile = acc_o.astype(Out.dtype).reshape(
                (1, TILE_M_D256, 1, HEAD_DIM_D256)
            )
            ct.store(
                Out,
                index=(batch_idx, m_block, head_idx, 0),
                tile=out_tile,
                latency=2,
                allow_tma=True,
            )

else:
    hstu_fp8_d256_persistent_kernel = None
