"""Directed correctness checks for the isolated cuTile D256 FP8 prototype."""

from __future__ import annotations

import math

import torch

from .wrapper import TileToolchainError, hstu_fp8_d256_cudatile, require_available


def _make_inputs(batch: int, seqlen: int, heads: int):
    from hstu.cuda_hstu_attention import (
        pack_descale_to_e8m0x4_int32,
        quantize_for_block_scale_qk_along_d,
        quantize_for_block_scale_v_along_n,
    )

    q_raw = torch.randn((batch * seqlen, heads, 256), device="cuda", dtype=torch.bfloat16)
    k_raw = torch.randn_like(q_raw)
    v_raw = torch.randn_like(q_raw)
    cu = torch.arange(0, (batch + 1) * seqlen, seqlen, device="cuda", dtype=torch.int32)
    q, q_descale, cu_q_sf = quantize_for_block_scale_qk_along_d(q_raw, cu)
    k, k_descale, cu_k_sf = quantize_for_block_scale_qk_along_d(k_raw, cu)
    v, v_descale, cu_v_sf = quantize_for_block_scale_v_along_n(v_raw, cu, block_size=64)
    sf_q = pack_descale_to_e8m0x4_int32(q_descale)
    sf_k = pack_descale_to_e8m0x4_int32(k_descale)
    sf_v = pack_descale_to_e8m0x4_int32(v_descale).repeat_interleave(64, dim=1)
    return (
        q,
        k,
        v,
        cu,
        q_descale,
        k_descale,
        v_descale,
        cu_q_sf,
        cu_k_sf,
        cu_v_sf,
        sf_q,
        sf_k,
        sf_v,
    )


def _reference(
    q,
    k,
    v,
    cu,
    seqlen: int,
    q_descale,
    k_descale,
    v_descale,
    cu_q_sf,
    cu_k_sf,
    cu_v_sf,
    sf_q,
    sf_k,
    sf_v,
    causal: bool,
):
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        q,
        k,
        v,
        cu,
        cu,
        None,
        None,
        seqlen,
        seqlen,
        seqlen,
        None,
        None,
        1,
        -1 if causal else seqlen,
        0 if causal else seqlen,
        1.0,
        None,
        None,
        2,
        q_descale,
        k_descale,
        v_descale,
        sf_q,
        sf_k,
        sf_v,
        cu_q_sf,
        cu_k_sf,
        cu_v_sf,
        None,
        None,
        None,
        None,
    )
    return out


def run_case(*, causal: bool, batch: int = 1, seqlen: int = 128, heads: int = 1):
    (
        q,
        k,
        v,
        cu,
        q_descale,
        k_descale,
        v_descale,
        cu_q_sf,
        cu_k_sf,
        cu_v_sf,
        sf_q,
        sf_k,
        sf_v,
    ) = _make_inputs(batch, seqlen, heads)
    out = hstu_fp8_d256_cudatile(
        q,
        k,
        v,
        cu,
        cu,
        seqlen,
        seqlen,
        sf_q_packed=sf_q,
        sf_k_packed=sf_k,
        sf_v_packed=sf_v,
        causal=causal,
        scaling_seqlen=seqlen,
    )
    ref = _reference(
        q,
        k,
        v,
        cu,
        seqlen,
        q_descale,
        k_descale,
        v_descale,
        cu_q_sf,
        cu_k_sf,
        cu_v_sf,
        sf_q,
        sf_k,
        sf_v,
        causal,
    )
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    max_err = (out.float() - ref.float()).abs().max().item()
    print(
        f"causal={causal} batch={batch} seqlen={seqlen} heads={heads} "
        f"cos={cos:.6f} max_err={max_err:.6f}"
    )
    if not math.isfinite(cos) or cos < 0.995:
        raise AssertionError(f"cosine similarity too low: {cos}")
    return cos, max_err


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    try:
        require_available()
        run_case(causal=False)
        run_case(causal=True)
    except TileToolchainError as exc:
        print(f"SKIP: {exc}")
        return


if __name__ == "__main__":
    main()
