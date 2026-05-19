"""Python wrapper for the isolated cuTile D256 BF16 HSTU prototype."""

from contextlib import nullcontext
import math
from types import SimpleNamespace
from typing import Optional

import torch

from . import kernel_d256
from .layout import materialize_dense_d256_inputs

try:
    from cuda.tile.tune import exhaustive_search
except Exception as exc:  # pragma: no cover - availability path.
    exhaustive_search = None
    _TUNE_IMPORT_ERROR = exc
else:
    _TUNE_IMPORT_ERROR = None


class TileToolchainError(RuntimeError):
    pass


_AUTOTUNE_CACHE: dict[tuple[int, int, int, int, bool], SimpleNamespace] = {}


def _toolchain_error() -> str | None:
    if kernel_d256.import_error() is not None:
        return f"cuda.tile import failed: {kernel_d256.import_error()}"
    if kernel_d256.ct is None:
        return "cuda.tile is unavailable"
    if kernel_d256.hstu_bf16_d256_kernel is None:
        return "cuTile D256 BF16 kernel was not constructed"
    if exhaustive_search is None:
        return f"cuda.tile.tune import failed: {_TUNE_IMPORT_ERROR}"
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
    autotune: bool = True,
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

    launch_stream = stream if stream is not None else torch.cuda.current_stream(q.device)
    if autotune:
        cfg = _get_autotuned_config(
            dense.q,
            dense.k,
            dense.v,
            dense.q_lengths,
            dense.k_lengths,
            out_dense,
            float(alpha),
            int(scaling_seqlen),
            bool(causal),
            launch_stream,
        )
    else:
        cfg = SimpleNamespace(
            TILE_M=kernel_d256.DEFAULT_TILE_M_D256,
            TILE_N=kernel_d256.DEFAULT_TILE_N_D256,
            num_ctas=None,
            occupancy=2,
        )

    grid = (
        math.ceil(max_seqlen_q / cfg.TILE_M),
        dense.q.shape[2],
        dense.q.shape[0],
    )
    tuned_kernel = kernel_d256.hstu_bf16_d256_kernel.replace_hints(
        num_ctas=cfg.num_ctas,
        occupancy=cfg.occupancy,
    )
    kernel_d256.ct.launch(
        launch_stream,
        grid,
        tuned_kernel,
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
            int(cfg.TILE_M),
            int(cfg.TILE_N),
        ),
    )

    return _compact_output(out_dense, cu_seqlens_q)


def _get_autotuned_config(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lengths: torch.Tensor,
    k_lengths: torch.Tensor,
    out: torch.Tensor,
    alpha: float,
    scaling_seqlen: int,
    causal: bool,
    stream: torch.cuda.Stream,
) -> SimpleNamespace:
    key = (q.shape[0], q.shape[2], q.shape[1], k.shape[1], causal)
    cached = _AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    search_space = _autotune_search_space()
    with _compiler_timeout(10):
        result = exhaustive_search(
            search_space,
            stream,
            grid_fn=lambda cfg: (
                math.ceil(q.shape[1] / cfg.TILE_M),
                q.shape[2],
                q.shape[0],
            ),
            kernel=kernel_d256.hstu_bf16_d256_kernel,
            args_fn=lambda cfg: (
                q,
                k,
                v,
                q_lengths,
                k_lengths,
                out,
                alpha,
                scaling_seqlen,
                causal,
                cfg.TILE_M,
                cfg.TILE_N,
            ),
            hints_fn=lambda cfg: {
                "num_ctas": cfg.num_ctas,
                "occupancy": cfg.occupancy,
            },
        )
    cfg = result.best.config
    _AUTOTUNE_CACHE[key] = cfg
    return cfg


def _autotune_search_space() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=1),
        SimpleNamespace(TILE_M=128, TILE_N=64, num_ctas=1, occupancy=1),
        SimpleNamespace(TILE_M=64, TILE_N=128, num_ctas=1, occupancy=1),
        SimpleNamespace(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2),
        SimpleNamespace(TILE_M=64, TILE_N=32, num_ctas=1, occupancy=2),
        SimpleNamespace(TILE_M=32, TILE_N=128, num_ctas=1, occupancy=1),
        SimpleNamespace(TILE_M=32, TILE_N=64, num_ctas=1, occupancy=1),
        SimpleNamespace(TILE_M=32, TILE_N=32, num_ctas=1, occupancy=1),
    ]


def _compiler_timeout(seconds: int):
    if hasattr(kernel_d256.ct, "compiler_timeout"):
        return kernel_d256.ct.compiler_timeout(seconds)
    return nullcontext()


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
