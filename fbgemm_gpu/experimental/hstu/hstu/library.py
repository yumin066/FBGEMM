#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024, NVIDIA Corporation & AFFILIATES.

# pyre-strict

import logging
import os

no_fbgemm_gpu: bool = False
try:
    import fbgemm_gpu  # noqa: F401
    # Namespace packages (created as a side-effect of editable install path hooks)
    # have __file__ == None and should not be treated as a real fbgemm_gpu install.
    if getattr(fbgemm_gpu, "__file__", None) is None:
        no_fbgemm_gpu = True
except ImportError:
    no_fbgemm_gpu = True

import torch

try:
    # pyre-ignore[21]
    # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
    from fbgemm_gpu import open_source

except Exception:
    open_source: bool = False

def _should_load_hstu_lib() -> bool:
    cap = torch.cuda.get_device_capability()
    # Load for SM < 10 (SM80/SM89/SM90) OR SM >= 12 (SM120 Blackwell consumer).
    # SM10x (GB200 etc.) is excluded until kernels are added.
    return cap < (10, 0) or cap[0] >= 12

if (
    torch.cuda.is_available()
    and torch.version.cuda is not None
    and torch.version.cuda >= "12.4"
):
    if open_source or no_fbgemm_gpu:
        if _should_load_hstu_lib():
            torch.ops.load_library(
                os.path.join(os.path.dirname(__file__), "fbgemm_gpu_experimental_hstu.so")
            )
            torch.classes.load_library(
                os.path.join(os.path.dirname(__file__), "fbgemm_gpu_experimental_hstu.so")
            )
    else:
        if _should_load_hstu_lib():
            torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_gpu")

            torch.ops.load_library(
                "//deeplearning/fbgemm/fbgemm_gpu/experimental/hstu/src:hstu_ops_gpu_sm80"
            )

            if torch.cuda.get_device_capability() >= (9, 0):
                torch.ops.load_library(
                    "//deeplearning/fbgemm/fbgemm_gpu/experimental/hstu/src:hstu_ops_gpu_sm90"
                )

else:
    logging.warning("CUDA is not available for FBGEMM HSTU")
