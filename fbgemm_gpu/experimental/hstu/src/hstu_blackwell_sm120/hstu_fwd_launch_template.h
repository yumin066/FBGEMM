/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SM120 HSTU forward launch template.
// Instantiation point for run_hstu_fwd_sm120<Arch, T, Hdim, ...>.

#pragma once

#include "hstu_fwd_kernel_launch.h"
#include "static_switch.h"
