"""Python wrapper for the isolated cuTile D256 FP8 HSTU prototype."""

from __future__ import annotations

import math
import tempfile
from functools import cache
from typing import Optional

import torch

from . import kernel_d256
from .kernel_d256 import TILE_M_D256, TILE_N_D256
from .scale_layout import materialize_dense_d256_inputs


class TileToolchainError(RuntimeError):
    pass


def _toolchain_error() -> str | None:
    if kernel_d256.import_error() is not None:
        return f"cuda.tile import failed: {kernel_d256.import_error()}"
    if kernel_d256.ct is None:
        return "cuda.tile is unavailable"
    if kernel_d256.hstu_fp8_d256_persistent_kernel is None:
        return "cuTile D256 kernel was not constructed"
    if not hasattr(torch, "float8_e8m0fnu"):
        return "torch.float8_e8m0fnu is unavailable"
    if not hasattr(kernel_d256.ct, "mma_scaled"):
        return "ct.mma_scaled is unavailable; tileiras 13.3/dev support is required"
    tileiras_error = _mma_scaled_tileiras_error()
    if tileiras_error is not None:
        return tileiras_error
    return None


@cache
def _mma_scaled_tileiras_error() -> str | None:
    try:
        from cuda.tile._bytecode.version import BytecodeVersion
        from cuda.tile._cext import dev_features_enabled
        from cuda.tile._compile import _get_max_supported_bytecode_version
    except Exception as exc:
        return f"cannot determine tileiras version for ct.mma_scaled: {exc}"

    current = _get_max_supported_bytecode_version(
        tempfile.gettempdir(),
        allow_dev=dev_features_enabled(),
    )
    required = BytecodeVersion.V_13_3
    if current < required:
        return (
            "ct.mma_scaled requires tileiras "
            f"{required.as_string()} or later, found {current.as_string()}"
        )
    return None


def is_available() -> bool:
    return _toolchain_error() is None


def require_available() -> None:
    reason = _toolchain_error()
    if reason is not None:
        raise TileToolchainError(reason)


def hstu_fp8_d256_cudatile(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    sf_q_packed: torch.Tensor,
    sf_k_packed: torch.Tensor,
    sf_v_packed: torch.Tensor,
    alpha: float = 1.0,
    causal: bool = False,
    scaling_seqlen: Optional[int] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Run the experimental cuTile D256 FP8 block-scale HSTU forward.

    This first implementation intentionally supports only dense/non-paged,
    no-RAB D256 full or pure-causal forward. Inputs use the production HSTU
    packed e8m0x4 scale layout and are materialized into cuTile-friendly dense
    scale tensors before launch.
    """

    require_available()
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("q/k/v must be CUDA tensors on the same device")
    if q.shape[2] != 256:
        raise ValueError("hstu_fp8_d256_cudatile is D256-only")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("q/k/v must have the same number of heads")
    if sf_q_packed.device != q.device or sf_k_packed.device != q.device or sf_v_packed.device != q.device:
        raise ValueError("scale tensors must be on the same CUDA device as q")

    dense = materialize_dense_d256_inputs(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        sf_q_packed,
        sf_k_packed,
        sf_v_packed,
        tile_n=TILE_N_D256,
    )
    out_dense = torch.empty(
        (dense.q.shape[0], max_seqlen_q, dense.q.shape[2], 256),
        dtype=torch.float16,
        device=q.device,
    )

    if scaling_seqlen is None or scaling_seqlen <= 0:
        scaling_seqlen = max_seqlen_q

    total_tiles = dense.q.shape[0] * dense.q.shape[2] * math.ceil(max_seqlen_q / TILE_M_D256)
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = (min(total_tiles, sms), 1, 1)
    launch_stream = stream if stream is not None else torch.cuda.current_stream(q.device)
    kernel_d256.ct.launch(
        launch_stream,
        grid,
        kernel_d256.hstu_fp8_d256_persistent_kernel,
        (
            dense.q,
            dense.k,
            dense.v,
            dense.q_scale,
            dense.k_scale,
            dense.v_scale,
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
