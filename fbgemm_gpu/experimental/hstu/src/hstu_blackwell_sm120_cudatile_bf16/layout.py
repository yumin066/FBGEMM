"""Layout conversion helpers for the cuTile BF16 prototype."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DenseD256Inputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_lengths: torch.Tensor
    k_lengths: torch.Tensor


def _lengths_from_cu(cu_seqlens: torch.Tensor) -> list[int]:
    cu = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    return [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]


def _copy_varlen_to_dense_d256(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    lengths = _lengths_from_cu(cu_seqlens)
    offsets = [int(v) for v in cu_seqlens.detach().cpu().tolist()]
    dense = torch.zeros(
        (len(lengths), max_len, x.shape[1], 256),
        dtype=x.dtype,
        device=x.device,
    )
    for b, length in enumerate(lengths):
        if length > 0:
            dense[b, :length].copy_(x[offsets[b] : offsets[b] + length])
    return dense.contiguous()


def materialize_dense_d256_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> DenseD256Inputs:
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q/k/v must all be torch.bfloat16")
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

    return DenseD256Inputs(
        q=_copy_varlen_to_dense_d256(q, cu_seqlens_q, max_seqlen_q),
        k=_copy_varlen_to_dense_d256(k, cu_seqlens_k, max_seqlen_k),
        v=_copy_varlen_to_dense_d256(v, cu_seqlens_k, max_seqlen_k),
        q_lengths=torch.diff(cu_seqlens_q).to(torch.int32).contiguous(),
        k_lengths=torch.diff(cu_seqlens_k).to(torch.int32).contiguous(),
    )

