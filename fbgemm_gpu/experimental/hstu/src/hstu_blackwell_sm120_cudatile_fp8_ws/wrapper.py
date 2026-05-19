"""Python wrapper for the isolated cuTile D256 FP8 HSTU prototype."""

from __future__ import annotations

from contextlib import nullcontext
import math
import tempfile
from functools import cache
from types import SimpleNamespace
from typing import Optional

import torch

from . import kernel_d256
from .scale_layout import materialize_dense_d256_inputs, materialize_v_scale_d256

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
    if kernel_d256.hstu_fp8_d256_persistent_kernel is None:
        return "cuTile D256 kernel was not constructed"
    if not hasattr(torch, "float8_e8m0fnu"):
        return "torch.float8_e8m0fnu is unavailable"
    if not hasattr(kernel_d256.ct, "mma_scaled"):
        return "ct.mma_scaled is unavailable; tileiras 13.3/dev support is required"
    if exhaustive_search is None:
        return f"cuda.tile.tune import failed: {_TUNE_IMPORT_ERROR}"
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
    autotune: bool = True,
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
        tile_n=kernel_d256.DEFAULT_TILE_N_D256,
    )
    out_dense = torch.empty(
        (dense.q.shape[0], max_seqlen_q, dense.q.shape[2], 256),
        dtype=torch.float16,
        device=q.device,
    )

    if scaling_seqlen is None or scaling_seqlen <= 0:
        scaling_seqlen = max_seqlen_q

    launch_stream = stream if stream is not None else torch.cuda.current_stream(q.device)
    v_scale_cache = {kernel_d256.DEFAULT_TILE_N_D256: dense.v_scale}

    def v_scale_for(tile_n: int) -> torch.Tensor:
        cached = v_scale_cache.get(tile_n)
        if cached is not None:
            return cached
        v_scale = materialize_v_scale_d256(
            sf_v_packed,
            cu_seqlens_k,
            max_seqlen_k,
            tile_n,
        )
        v_scale_cache[tile_n] = v_scale
        return v_scale

    if autotune:
        cfg = _get_autotuned_config(
            dense.q,
            dense.k,
            dense.v,
            dense.q_scale,
            dense.k_scale,
            dense.q_lengths,
            dense.k_lengths,
            out_dense,
            float(alpha),
            int(scaling_seqlen),
            bool(causal),
            launch_stream,
            v_scale_for,
        )
    else:
        cfg = SimpleNamespace(
            TILE_M=kernel_d256.DEFAULT_TILE_M_D256,
            TILE_N=kernel_d256.DEFAULT_TILE_N_D256,
            num_ctas=kernel_d256.KERNEL_NUM_CTAS_D256,
            occupancy=kernel_d256.KERNEL_OCCUPANCY_D256,
        )

    total_tiles = dense.q.shape[0] * dense.q.shape[2] * math.ceil(max_seqlen_q / cfg.TILE_M)
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = (min(total_tiles, sms), 1, 1)
    tuned_kernel = kernel_d256.hstu_fp8_d256_persistent_kernel.replace_hints(
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
            dense.q_scale,
            dense.k_scale,
            v_scale_for(cfg.TILE_N),
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
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    q_lengths: torch.Tensor,
    k_lengths: torch.Tensor,
    out: torch.Tensor,
    alpha: float,
    scaling_seqlen: int,
    causal: bool,
    stream: torch.cuda.Stream,
    v_scale_for,
) -> SimpleNamespace:
    key = (q.shape[0], q.shape[2], q.shape[1], k.shape[1], causal)
    cached = _AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    with _compiler_timeout(10):
        result = exhaustive_search(
            _autotune_search_space(),
            stream,
            grid_fn=lambda cfg: (
                min(
                    q.shape[0] * q.shape[2] * math.ceil(q.shape[1] / cfg.TILE_M),
                    torch.cuda.get_device_properties(q.device).multi_processor_count,
                ),
                1,
                1,
            ),
            kernel=kernel_d256.hstu_fp8_d256_persistent_kernel,
            args_fn=lambda cfg: (
                q,
                k,
                v,
                q_scale,
                k_scale,
                v_scale_for(cfg.TILE_N),
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
        SimpleNamespace(TILE_M=32, TILE_N=32, num_ctas=1, occupancy=1),
        SimpleNamespace(TILE_M=32, TILE_N=64, num_ctas=1, occupancy=1),
        SimpleNamespace(TILE_M=32, TILE_N=128, num_ctas=1, occupancy=1),
        SimpleNamespace(TILE_M=64, TILE_N=32, num_ctas=1, occupancy=1),
        SimpleNamespace(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2),
        SimpleNamespace(TILE_M=64, TILE_N=128, num_ctas=1, occupancy=1),
        SimpleNamespace(TILE_M=128, TILE_N=64, num_ctas=1, occupancy=1),
        SimpleNamespace(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=1),
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
