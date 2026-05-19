"""Scale layout conversion helpers for the cuTile D256 FP8 prototype."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DenseD256Inputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    q_lengths: torch.Tensor
    k_lengths: torch.Tensor


def _require_e8m0_dtype() -> torch.dtype:
    if not hasattr(torch, "float8_e8m0fnu"):
        raise RuntimeError("torch.float8_e8m0fnu is required for cuTile mma_scaled")
    return torch.float8_e8m0fnu


def _as_i32_packed(packed: torch.Tensor, name: str) -> torch.Tensor:
    if packed.dtype != torch.int32:
        raise TypeError(f"{name} must be int32 e8m0x4 packed scale, got {packed.dtype}")
    if packed.dim() != 2:
        raise ValueError(f"{name} must have shape [H, total_tokens], got {tuple(packed.shape)}")
    return packed.contiguous()


def unpack_e8m0x4_lanes(packed: torch.Tensor) -> torch.Tensor:
    """Return e8m0 lanes with shape packed.shape + (4,)."""

    e8m0 = _require_e8m0_dtype()
    raw = unpack_e8m0x4_lanes_u8(packed)
    return raw.view(e8m0)


def unpack_e8m0x4_lanes_u8(packed: torch.Tensor) -> torch.Tensor:
    """Return raw e8m0 lane bytes with shape packed.shape + (4,)."""

    packed = packed.contiguous()
    lanes_u8 = []
    for lane in range(4):
        lanes_u8.append(((packed >> (8 * lane)) & 0xFF).to(torch.uint8))
    return torch.stack(lanes_u8, dim=-1).contiguous()


def expand_qk_d256_scales_from_packed(packed: torch.Tensor) -> torch.Tensor:
    """Expand HSTU D256 Q/K packed scales to cuTile 32-wide scale blocks.

    Input:  [H, T] int32, lane 0 for D[0:128], lane 1 for D[128:256].
    Output: [H, T, 8] e8m0, one scale per 32-D block.
    """

    e8m0 = _require_e8m0_dtype()
    lanes = unpack_e8m0x4_lanes_u8(_as_i32_packed(packed, "packed"))
    first = lanes[..., 0:1].expand(*lanes.shape[:-1], 4)
    second = lanes[..., 1:2].expand(*lanes.shape[:-1], 4)
    raw = torch.cat([first, second], dim=-1).contiguous()
    return raw.view(e8m0)


def expand_v_d256_block_scale(scale_token: torch.Tensor) -> torch.Tensor:
    """Expand one `[H]` e8m0 V scale vector to `[H, 2, 256]`."""

    e8m0 = _require_e8m0_dtype()
    raw = scale_token.view(torch.uint8)
    expanded = raw[:, None, None].expand(raw.shape[0], 2, 256).contiguous()
    return expanded.view(e8m0)


def _lengths_from_cu(cu_seqlens: torch.Tensor) -> list[int]:
    cu = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    return [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]


def _copy_varlen_to_dense_d256(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    lengths = _lengths_from_cu(cu_seqlens)
    batch = len(lengths)
    dense = torch.zeros(
        (batch, max_len, x.shape[1], 256),
        dtype=x.dtype,
        device=x.device,
    )
    offsets = [int(v) for v in cu_seqlens.detach().cpu().tolist()]
    for b, length in enumerate(lengths):
        if length > 0:
            dense[b, :length].copy_(x[offsets[b] : offsets[b] + length])
    return dense.contiguous()


def _copy_qk_scale_to_dense_d256(
    packed: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    expanded = expand_qk_d256_scales_from_packed(packed)  # [H, total, 8]
    lengths = _lengths_from_cu(cu_seqlens)
    batch = len(lengths)
    heads = expanded.shape[0]
    dense = torch.zeros(
        (batch, heads, max_len, 8),
        dtype=expanded.dtype,
        device=expanded.device,
    )
    offsets = [int(v) for v in cu_seqlens.detach().cpu().tolist()]
    for b, length in enumerate(lengths):
        if length > 0:
            dense[b, :, :length, :].copy_(
                expanded[:, offsets[b] : offsets[b] + length, :]
            )
    return dense.contiguous()


def _copy_v_scale_to_dense_d256(
    sf_v_packed: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_len_k: int,
    tile_n: int,
) -> torch.Tensor:
    lanes = unpack_e8m0x4_lanes(_as_i32_packed(sf_v_packed, "sf_v_packed"))
    scale_by_token = lanes[..., 0]  # [H, total_tokens]
    lengths = _lengths_from_cu(cu_seqlens_k)
    batch = len(lengths)
    heads = scale_by_token.shape[0]
    num_n_blocks = (max_len_k + tile_n - 1) // tile_n
    dense = torch.zeros(
        (batch, heads, num_n_blocks, 2, 256),
        dtype=scale_by_token.dtype,
        device=scale_by_token.device,
    )
    offsets = [int(v) for v in cu_seqlens_k.detach().cpu().tolist()]
    for b, length in enumerate(lengths):
        if length <= 0:
            continue
        for n_block in range(num_n_blocks):
            token_in_batch = min(n_block * tile_n, length - 1)
            token = offsets[b] + token_in_batch
            dense[b, :, n_block, :, :].copy_(
                expand_v_d256_block_scale(scale_by_token[:, token])
            )
    return dense.contiguous()


def materialize_v_scale_d256(
    sf_v_packed: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_len_k: int,
    tile_n: int,
) -> torch.Tensor:
    return _copy_v_scale_to_dense_d256(
        sf_v_packed,
        cu_seqlens_k,
        max_len_k,
        tile_n,
    )


def materialize_dense_d256_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sf_q_packed: torch.Tensor,
    sf_k_packed: torch.Tensor,
    sf_v_packed: torch.Tensor,
    *,
    tile_n: int,
) -> DenseD256Inputs:
    """Materialize compact HSTU FP8 varlen inputs into dense cuTile tensors."""

    if q.dtype != torch.float8_e4m3fn or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q/k/v must all be torch.float8_e4m3fn")
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q/k/v must have shape [total, H, 256]")
    if q.shape[2] != 256 or k.shape[2] != 256 or v.shape[2] != 256:
        raise ValueError("this cuTile prototype is D256-only")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("q/k/v must have the same number of heads")
    for name, tensor in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be int32")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA")

    q_lengths = torch.diff(cu_seqlens_q).to(torch.int32).contiguous()
    k_lengths = torch.diff(cu_seqlens_k).to(torch.int32).contiguous()
    return DenseD256Inputs(
        q=_copy_varlen_to_dense_d256(q, cu_seqlens_q, max_seqlen_q),
        k=_copy_varlen_to_dense_d256(k, cu_seqlens_k, max_seqlen_k),
        v=_copy_varlen_to_dense_d256(v, cu_seqlens_k, max_seqlen_k),
        q_scale=_copy_qk_scale_to_dense_d256(sf_q_packed, cu_seqlens_q, max_seqlen_q),
        k_scale=_copy_qk_scale_to_dense_d256(sf_k_packed, cu_seqlens_k, max_seqlen_k),
        v_scale=_copy_v_scale_to_dense_d256(
            sf_v_packed, cu_seqlens_k, max_seqlen_k, tile_n
        ),
        q_lengths=q_lengths,
        k_lengths=k_lengths,
    )
