"""cuTile D256 BF16 HSTU forward kernel."""

import math

import numpy as np

try:
    import cuda.tile as ct
except Exception as exc:  # pragma: no cover - availability path.
    ct = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


TILE_M_D256 = 64
TILE_N_D256 = 64
HEAD_DIM_D256 = 256


def import_error():
    return _IMPORT_ERROR


if ct is not None:
    ConstBool = ct.Constant[bool]

    @ct.kernel(occupancy=2)
    def hstu_bf16_d256_kernel(
        Q,
        K,
        V,
        QLengths,
        KLengths,
        Out,
        alpha: float,
        scaling_seqlen: int,
        causal: ConstBool,
    ):
        m_block = ct.bid(0)
        head_idx = ct.bid(1)
        batch_idx = ct.bid(2)

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

        acc_o = ct.full((TILE_M_D256, HEAD_DIM_D256), 0.0, dtype=ct.float32)
        num_n_blocks = ct.cdiv(K.shape[1], TILE_N_D256)
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
                latency=4,
                allow_tma=True,
            ).reshape((TILE_N_D256, HEAD_DIM_D256))
            k_t = ct.transpose(k_tile)

            acc_s = ct.full((TILE_M_D256, TILE_N_D256), 0.0, dtype=ct.float32)
            acc_s = ct.mma(q, k_t, acc_s)

            offs_n = n_block * TILE_N_D256 + offs_n_base
            valid = (offs_m < q_len) & (offs_n < k_len)
            if causal:
                valid = valid & (offs_n <= (qk_offset + offs_m))
            acc_s = ct.where(valid, acc_s, -math.inf)

            s = acc_s * alpha
            p = s / (ct.exp(-s) + 1.0)
            p = ct.where(valid, p, 0.0).astype(Q.dtype)

            v_tile = ct.load(
                V,
                index=(batch_idx, n_block, head_idx, 0),
                shape=(1, TILE_N_D256, 1, HEAD_DIM_D256),
                padding_mode=ct.PaddingMode.ZERO,
                latency=4,
                allow_tma=True,
            ).reshape((TILE_N_D256, HEAD_DIM_D256))
            acc_o = ct.mma(p, v_tile, acc_o)

        if scaling_seqlen > 1:
            acc_o = acc_o / scaling_seqlen

        out_tile = acc_o.astype(Out.dtype).reshape(
            (1, TILE_M_D256, 1, HEAD_DIM_D256)
        )
        ct.store(Out, index=(batch_idx, m_block, head_idx, 0), tile=out_tile)

else:
    hstu_bf16_d256_kernel = None

