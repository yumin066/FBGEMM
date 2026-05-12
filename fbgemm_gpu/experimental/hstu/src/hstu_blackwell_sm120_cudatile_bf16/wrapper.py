"""Python wrapper for the isolated cuTile D256 BF16 HSTU prototype."""

import math
from typing import Optional

import torch

from . import kernel_d256
from .kernel_d256 import TILE_M_D256
from .layout import materialize_dense_d256_inputs


class TileToolchainError(RuntimeError):
    pass


def _toolchain_error() -> str | None:
    if kernel_d256.import_error() is not None:
        return f"cuda.tile import failed: {kernel_d256.import_error()}"
    if kernel_d256.ct is None:
        return "cuda.tile is unavailable"
    if kernel_d256.hstu_bf16_d256_kernel is None:
        return "cuTile D256 BF16 kernel was not constructed"
    return None


def is_available() -> bool:
    return _toolchain_error() is None


def require_available() -> None:
    reason = _toolchain_error()
    if reason is not None:
        raise TileToolchainError(reason)


def hstu_bf16_d256_cudatile(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    alpha: float = 1.0,
    causal: bool = False,
    scaling_seqlen: Optional[int] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Run the experimental cuTile D256 BF16 dense HSTU forward."""

    require_available()
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("q/k/v must be CUDA tensors on the same device")

    dense = materialize_dense_d256_inputs(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )
    out_dense = torch.empty(
        (dense.q.shape[0], max_seqlen_q, dense.q.shape[2], 256),
        dtype=torch.bfloat16,
        device=q.device,
    )

    if scaling_seqlen is None or scaling_seqlen <= 0:
        scaling_seqlen = max_seqlen_q

    grid = (
        math.ceil(max_seqlen_q / TILE_M_D256),
        dense.q.shape[2],
        dense.q.shape[0],
    )
    launch_stream = stream if stream is not None else torch.cuda.current_stream(q.device)
    kernel_d256.ct.launch(
        launch_stream,
        grid,
        kernel_d256.hstu_bf16_d256_kernel,
        (
            dense.q,
            dense.k,
            dense.v,
            dense.q_lengths,
            dense.k_lengths,
            out_dense,
            float(alpha),
            int(scaling_seqlen),
            bool(causal),
        ),
    )

    return _compact_output(out_dense, cu_seqlens_q)


def _compact_output(out_dense: torch.Tensor, cu_seqlens_q: torch.Tensor) -> torch.Tensor:
    offsets = [int(v) for v in cu_seqlens_q.detach().cpu().tolist()]
    pieces = []
    for b in range(len(offsets) - 1):
        length = offsets[b + 1] - offsets[b]
        if length > 0:
            pieces.append(out_dense[b, :length])
    if not pieces:
        return out_dense.new_empty((0, out_dense.shape[2], out_dense.shape[3]))
    return torch.cat(pieces, dim=0).contiguous()

