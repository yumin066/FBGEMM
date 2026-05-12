"""Experimental cuTile FP8 block-scale HSTU forward for SM120.

This package is intentionally isolated from the production dispatcher.  It is
currently D256-first and dense/non-paged-first.
"""

from .wrapper import (
    TileToolchainError,
    hstu_fp8_d256_cudatile,
    is_available,
    require_available,
)

__all__ = [
    "TileToolchainError",
    "hstu_fp8_d256_cudatile",
    "is_available",
    "require_available",
]
