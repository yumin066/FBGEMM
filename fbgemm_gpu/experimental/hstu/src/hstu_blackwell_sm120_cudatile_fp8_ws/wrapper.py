"""Python wrapper for the isolated cuTile D256 FP8 HSTU prototype."""

from __future__ import annotations

from contextlib import nullcontext
import math
import os
import tempfile
from functools import cache
from itertools import product
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


_AUTOTUNE_CACHE: dict[tuple[int, int, int, int, bool, bool], SimpleNamespace] = {}


def _toolchain_error() -> str | None:
    if kernel_d256.import_error() is not None:
        return f"cuda.tile import failed: {kernel_d256.import_error()}"
    if kernel_d256.ct is None:
        return "cuda.tile is unavailable"
    if kernel_d256.hstu_fp8_d256_kernel is None:
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
    full_tiles = (
        not causal
        and max_seqlen_q == max_seqlen_k
        and bool(torch.all(dense.q_lengths == max_seqlen_q).item())
        and bool(torch.all(dense.k_lengths == max_seqlen_k).item())
    )
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
            full_tiles,
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

    total_tiles = dense.q.shape[0] * dense.q.shape[2] * math.ceil(
        max_seqlen_q / cfg.TILE_M
    )
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = (_num_launch_ctas(total_tiles, sms, bool(causal)), 1, 1)
    tuned_kernel = kernel_d256.hstu_fp8_d256_kernel.replace_hints(
        **_compiler_hints(cfg),
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
            bool(full_tiles),
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
    full_tiles: bool,
    stream: torch.cuda.Stream,
    v_scale_for,
) -> SimpleNamespace:
    key = (q.shape[0], q.shape[2], q.shape[1], k.shape[1], causal, full_tiles)
    cached = _AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    with _compiler_timeout(_autotune_compiler_timeout_seconds()):
        result = exhaustive_search(
            _autotune_search_space(),
            stream,
            grid_fn=lambda cfg: (
                _num_launch_ctas(
                    q.shape[0] * q.shape[2] * math.ceil(q.shape[1] / cfg.TILE_M),
                    torch.cuda.get_device_properties(q.device).multi_processor_count,
                    causal,
                ),
                1,
                1,
            ),
            kernel=kernel_d256.hstu_fp8_d256_kernel,
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
                full_tiles,
                cfg.TILE_M,
                cfg.TILE_N,
            ),
            hints_fn=lambda cfg: {
                **_compiler_hints(cfg),
            },
            quiet=_autotune_quiet(),
        )
    cfg = result.best.config
    _AUTOTUNE_CACHE[key] = cfg
    return cfg


def _autotune_search_space() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            TILE_M=tile_m,
            TILE_N=tile_n,
            num_ctas=num_ctas,
            occupancy=occupancy,
            num_worker_warps=num_worker_warps,
        )
        for tile_m, tile_n, num_ctas, occupancy, num_worker_warps in product(
            _autotune_tile_m_values(),
            _autotune_tile_n_values(),
            _autotune_num_ctas_values(),
            _autotune_occupancy_values(),
            _autotune_worker_warps_values(),
        )
    ]


def _autotune_tile_m_values() -> tuple[int, ...]:
    return _env_int_tuple("HSTU_CUTILE_FP8_AUTOTUNE_TILE_M", (16, 32, 64, 128, 256))


def _autotune_tile_n_values() -> tuple[int, ...]:
    return _env_int_tuple("HSTU_CUTILE_FP8_AUTOTUNE_TILE_N", (32, 64, 128, 256))


def _autotune_num_ctas_values() -> tuple[int | None, ...]:
    return _env_optional_int_tuple("HSTU_CUTILE_FP8_AUTOTUNE_NUM_CTAS", (None, 1))


def _autotune_occupancy_values() -> tuple[int, ...]:
    return _env_int_tuple("HSTU_CUTILE_FP8_AUTOTUNE_OCCUPANCY", (1, 2))


def _autotune_worker_warps_values() -> tuple[int | None, ...]:
    if not _compiler_hint_supported("num_worker_warps"):
        return (None,)
    return _env_optional_int_tuple("HSTU_CUTILE_FP8_AUTOTUNE_WORKER_WARPS", (None, 4, 8))


def _compiler_hints(cfg: SimpleNamespace) -> dict[str, int]:
    hints = {"occupancy": int(cfg.occupancy)}
    if cfg.num_ctas is not None:
        hints["num_ctas"] = int(cfg.num_ctas)
    if (
        getattr(cfg, "num_worker_warps", None) is not None
        and _compiler_hint_supported("num_worker_warps")
    ):
        hints["num_worker_warps"] = int(cfg.num_worker_warps)
    return hints


@cache
def _compiler_hint_supported(name: str) -> bool:
    options = getattr(kernel_d256.hstu_fp8_d256_kernel, "_compiler_options", None)
    return options is not None and hasattr(options, name)


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return tuple(int(part) for part in value.replace(",", " ").split())


def _env_optional_int_tuple(
    name: str,
    default: tuple[int | None, ...],
) -> tuple[int | None, ...]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    result: list[int | None] = []
    for part in value.replace(",", " ").split():
        if part.lower() in ("none", "null", "unset", "-"):
            result.append(None)
        else:
            result.append(int(part))
    return tuple(result)


def _autotune_compiler_timeout_seconds() -> int:
    return int(os.environ.get("HSTU_CUTILE_FP8_AUTOTUNE_COMPILER_TIMEOUT", "8"))


def _autotune_quiet() -> bool:
    return os.environ.get("HSTU_CUTILE_FP8_AUTOTUNE_VERBOSE", "0") not in ("1", "true", "TRUE")


def _num_launch_ctas(total_tiles: int, sms: int, causal: bool) -> int:
    if not kernel_d256.KERNEL_PERSISTENT_D256:
        return total_tiles
    if causal:
        return total_tiles
    return min(total_tiles, sms)


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
