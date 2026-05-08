#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SM120 BF16 CuTe DSL prototype.

This intentionally keeps the default SM120 C++ BF16 dispatcher unchanged.  It
uses the SM120 CuTe DSL HSTU forward implementation with Ampere-style
warp-level mma.sync, so benchmark scripts can compare the prototype
against the current BF16 CUDA path.
"""

import os
import sys
import types
from typing import Optional

import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from .hstu_attention_warp import HSTUAttentionForwardSm120CuteDsl


_CONTIGUOUS_STRIDE_ORDERS = {
    1: (0,),
    2: (0, 1),
    3: (0, 1, 2),
    4: (0, 1, 2, 3),
}
_BF16_DYNAMIC_DIVISIBILITY = 128 // cutlass.BFloat16.width


def _contiguous_stride_order(ndim: int) -> tuple[int, ...]:
    return _CONTIGUOUS_STRIDE_ORDERS.get(ndim, tuple(range(ndim)))


def _make_compact_dynamic_tensor(t: torch.Tensor):
    # The wrapper canonicalizes Q/K/V/O/RAB to contiguous dense tensors before
    # creating CuTe DSL descriptors. Calling Tensor.dim_order() here is expensive
    # because it revalidates the memory format; contiguous order is already known.
    return (
        from_dlpack(t.detach(), assumed_align=16)
        .mark_layout_dynamic(leading_dim=t.ndim - 1)
        .mark_compact_shape_dynamic(
            mode=t.ndim - 1,
            stride_order=_contiguous_stride_order(t.ndim),
            divisibility=_BF16_DYNAMIC_DIVISIBILITY,
        )
    )


if "cutlass.cute.experimental" not in sys.modules:
    # CUTLASS DSL 4.3 package discovery imports this optional module even though
    # this SM120 prototype does not use it.  The wheel raises from that import,
    # so keep compile-time package walking on the supported modules only.
    sys.modules["cutlass.cute.experimental"] = types.ModuleType(
        "cutlass.cute.experimental"
    )


def hstu_varlen_fwd_120_bf16_cutedsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int,
    window_size_left: int,
    window_size_right: int,
    alpha: float,
    rab: Optional[torch.Tensor],
    func: Optional[torch.Tensor],
    kv_cache: Optional[torch.Tensor] = None,
    page_offsets: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    last_page_lens: Optional[torch.Tensor] = None,
    *,
    paged_kv: Optional[torch.Tensor] = None,
    page_indptrs: Optional[torch.Tensor] = None,
    scale_output: bool = True,
):
    """Run the SM120 CuTe DSL BF16 prototype forward kernel.

    Support is intentionally still narrower than the production C++ path:
    BF16, compact varlen or dense full-batch Q/K/V, optional wrapper-side
    paged KV materialization, full/causal/local/context/target/arbitrary mask,
    and optional forward RAB/DRAB bias.  Compact varlen inputs are padded to the
    dense max-seqlen shape before the DSL kernel launch and gathered back
    afterwards.
    """

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    assert q.dtype == torch.bfloat16, "SM120 CuTe DSL prototype only supports BF16"
    assert k.dtype == q.dtype, "k and q must have the same dtype"
    assert v.dtype == q.dtype, "v and q must have the same dtype"
    assert cu_seqlens_q.dtype == torch.int32, "cu_seqlens_q must be int32"
    assert cu_seqlens_k.dtype == torch.int32, "cu_seqlens_k must be int32"
    assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3, "Q/K/V must be [total, H, D]"
    assert k.shape[1] == q.shape[1] and v.shape[1] == q.shape[1], (
        "SM120 CuTe DSL prototype expects Q/K/V to have the same number of heads"
    )

    head_dim = q.shape[2]
    head_dim_v = v.shape[2]
    assert head_dim == head_dim_v, "head_dim and head_dim_v must be equal"
    assert head_dim in (32, 64, 128, 256), (
        "SM120 CuTe DSL BF16 prototype supports headDim 32, 64, 128, and 256"
    )

    if kv_cache is None:
        kv_cache = paged_kv
    elif paged_kv is not None:
        assert kv_cache.data_ptr() == paged_kv.data_ptr(), (
            "kv_cache and paged_kv aliases must point to the same tensor"
        )
    if page_offsets is None:
        page_offsets = page_indptrs
    elif page_indptrs is not None:
        assert page_offsets.data_ptr() == page_indptrs.data_ptr(), (
            "page_offsets and page_indptrs aliases must point to the same tensor"
        )
    is_paged_kv = any(
        x is not None for x in (kv_cache, page_offsets, page_ids, last_page_lens)
    )
    if is_paged_kv:
        assert kv_cache is not None, "paged KV requires kv_cache"
        assert page_offsets is not None, "paged KV requires page_offsets"
        assert page_ids is not None, "paged KV requires page_ids"
        assert last_page_lens is not None, "paged KV requires last_page_lens"
        assert kv_cache.dtype == q.dtype, "kv_cache dtype must match q"
        assert kv_cache.ndim == 5 and kv_cache.shape[1] == 2, (
            "kv_cache must have shape [pages, 2, page_size, H, D]"
        )
        assert kv_cache.shape[3] == q.shape[1] and kv_cache.shape[4] == head_dim, (
            "kv_cache H/D must match Q"
        )
        assert page_offsets.dtype == torch.int32, "page_offsets must be int32"
        assert page_ids.dtype == torch.int32, "page_ids must be int32"
        assert last_page_lens.dtype == torch.int32, "last_page_lens must be int32"

    has_rab = rab is not None
    is_arbitrary = func is not None
    if has_rab:
        if head_dim <= 64:
            kBlockM = 128
            kBlockN = 64
            default_num_threads = 256
        else:
            kBlockM = 64
            kBlockN = 64
            default_num_threads = 128
    elif head_dim <= 64:
        kBlockM = 128
        kBlockN = 64
        default_num_threads = 256
    elif head_dim > 128:
        kBlockM = 64
        kBlockN = 64
        default_num_threads = 128
    else:
        kBlockM = 128
        kBlockN = 128
        default_num_threads = 256
    num_threads = int(os.environ.get("HSTU_CUTEDSL_NUM_THREADS", str(default_num_threads)))
    assert num_threads in (128, 256), "HSTU_CUTEDSL_NUM_THREADS must be 128 or 256"
    enable_fast_sigmoid = os.environ.get("HSTU_CUTEDSL_FAST_SIGMOID", "1") != "0"
    window_size_left = (
        max_seqlen_k
        if window_size_left < 0 or window_size_left > max_seqlen_k
        else window_size_left
    )
    window_size_right = (
        max_seqlen_k
        if window_size_right < 0 or window_size_right > max_seqlen_k
        else window_size_right
    )
    is_causal = window_size_left == max_seqlen_k and window_size_right == 0
    is_local = (
        window_size_left < max_seqlen_k or window_size_right < max_seqlen_k
    ) and not is_causal
    is_context = num_contexts is not None
    is_target = num_targets is not None
    assert not (is_context and is_local), (
        "SM120 CuTe DSL prototype does not support context + local mask"
    )
    assert not (is_target and not is_causal), (
        "SM120 CuTe DSL prototype only supports target mask with causal window"
    )

    batch_size = int(cu_seqlens_q.numel() - 1)
    assert batch_size > 0, "empty batch is unsupported"
    assert int(cu_seqlens_k.numel() - 1) == batch_size, "Q/K batch sizes must match"
    if is_target:
        assert num_targets.dtype == torch.int32, "num_targets must be int32"

    is_paged_delta_q = is_paged_kv and max_seqlen_k > max_seqlen_q
    q_dense_len = max_seqlen_k if is_paged_delta_q else max_seqlen_q

    q_is_dense = q.shape[0] == batch_size * max_seqlen_q
    k_is_dense = k.shape[0] == batch_size * max_seqlen_k

    def _target_len(b: int) -> int:
        return int(num_targets[b].item()) if is_target else 0

    def _copy_paged_delta_q_to_dense(dense: torch.Tensor) -> None:
        for b in range(batch_size):
            q_begin = int(cu_seqlens_q[b].item())
            q_end = int(cu_seqlens_q[b + 1].item())
            k_begin = int(cu_seqlens_k[b].item())
            k_end = int(cu_seqlens_k[b + 1].item())
            q_len = q_end - q_begin
            k_len = k_end - k_begin
            q_offset = k_len - q_len
            assert q_offset >= 0, "paged delta-q requires actual_seqlen_k >= actual_seqlen_q"
            dense[b, q_offset : q_offset + q_len].copy_(q[q_begin:q_end])

    def _copy_compact_to_dense(
        dense: torch.Tensor,
        compact: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_len: int,
    ) -> None:
        for b in range(batch_size):
            begin = int(cu_seqlens[b].item())
            end = int(cu_seqlens[b + 1].item())
            length = end - begin
            target_len = _target_len(b)
            if target_len > 0:
                prefix_len = length - target_len
                target_begin = max_len - target_len
                dense[b, :prefix_len].copy_(compact[begin : begin + prefix_len])
                dense[b, target_begin:max_len].copy_(compact[begin + prefix_len : end])
            else:
                dense[b, :length].copy_(compact[begin:end])

    if is_paged_delta_q:
        q_dense = q.new_zeros((batch_size, q_dense_len, q.shape[1], head_dim))
        _copy_paged_delta_q_to_dense(q_dense)
    elif q_is_dense:
        q_dense = q.view(batch_size, max_seqlen_q, q.shape[1], head_dim)
    else:
        q_dense = q.new_zeros((batch_size, max_seqlen_q, q.shape[1], head_dim))
        _copy_compact_to_dense(q_dense, q, cu_seqlens_q, max_seqlen_q)

    def _copy_paged_kv_to_dense(
        dense_k: torch.Tensor,
        dense_v: torch.Tensor,
    ) -> None:
        assert kv_cache is not None
        assert page_offsets is not None
        assert page_ids is not None
        assert last_page_lens is not None
        page_size = int(kv_cache.shape[2])
        use_k_offsets_for_tail = int(cu_seqlens_k[-1].item()) <= k.shape[0]
        use_q_offsets_for_tail = int(cu_seqlens_q[-1].item()) <= k.shape[0]
        for b in range(batch_size):
            logical_k_len = int(cu_seqlens_k[b + 1].item() - cu_seqlens_k[b].item())
            target_len = _target_len(b)
            cache_len = logical_k_len - target_len
            assert cache_len >= 0, "paged KV cache length must be non-negative"
            page_begin = int(page_offsets[b].item())
            page_end = int(page_offsets[b + 1].item())
            assert page_end > page_begin, "paged KV requires at least one page per batch"

            dst = 0
            remaining = cache_len
            for logical_page in range(page_begin, page_end):
                page_id = int(page_ids[logical_page].item())
                valid = page_size
                if logical_page == page_end - 1:
                    valid = int(last_page_lens[b].item())
                valid = min(valid, remaining)
                if valid > 0:
                    dense_k[b, dst : dst + valid].copy_(kv_cache[page_id, 0, :valid])
                    dense_v[b, dst : dst + valid].copy_(kv_cache[page_id, 1, :valid])
                    dst += valid
                    remaining -= valid
            assert remaining == 0, "paged KV metadata does not cover logical cache length"

            if target_len > 0:
                if use_k_offsets_for_tail:
                    src_end = int(cu_seqlens_k[b + 1].item())
                else:
                    assert use_q_offsets_for_tail, (
                        "paged KV target tail must be addressable through K offsets or Q offsets"
                    )
                    src_end = int(cu_seqlens_q[b + 1].item())
                src_begin = src_end - target_len
                dst_begin = max_seqlen_k - target_len
                dense_k[b, dst_begin:max_seqlen_k].copy_(k[src_begin:src_end])
                dense_v[b, dst_begin:max_seqlen_k].copy_(v[src_begin:src_end])

    if is_paged_kv:
        k_dense = k.new_zeros((batch_size, max_seqlen_k, k.shape[1], head_dim))
        v_dense = v.new_zeros((batch_size, max_seqlen_k, v.shape[1], head_dim))
        _copy_paged_kv_to_dense(k_dense, v_dense)
    elif k_is_dense:
        k_dense = k.view(batch_size, max_seqlen_k, k.shape[1], head_dim)
        v_dense = v.view(batch_size, max_seqlen_k, v.shape[1], head_dim)
    else:
        k_dense = k.new_zeros((batch_size, max_seqlen_k, k.shape[1], head_dim))
        v_dense = v.new_zeros((batch_size, max_seqlen_k, v.shape[1], head_dim))
        _copy_compact_to_dense(k_dense, k, cu_seqlens_k, max_seqlen_k)
        _copy_compact_to_dense(v_dense, v, cu_seqlens_k, max_seqlen_k)
    out_dense = torch.empty_like(q_dense)

    if has_rab:
        assert rab.dtype == torch.bfloat16, "RAB/DRAB forward bias must be BF16"
        assert rab.shape[0] == batch_size, "RAB batch dim must match Q/K batch size"
        if rab.shape[1] == 1 and q.shape[1] != 1:
            rab = rab.expand(batch_size, q.shape[1], rab.shape[-2], rab.shape[-1])
        assert rab.shape[1] == q.shape[1], "RAB head dim must be 1 or match Q heads"
        if is_target and (not q_is_dense or not k_is_dense or is_paged_kv):
            rab_q_len = q_dense_len if is_paged_delta_q else max_seqlen_q
            rab_dense = rab.new_zeros((batch_size, q.shape[1], rab_q_len, max_seqlen_k))
            for b in range(batch_size):
                q_begin = int(cu_seqlens_q[b].item())
                q_end = int(cu_seqlens_q[b + 1].item())
                k_begin = int(cu_seqlens_k[b].item())
                k_end = int(cu_seqlens_k[b + 1].item())
                q_len = q_end - q_begin
                k_len = k_end - k_begin
                target_len = _target_len(b)
                q_prefix = q_len - target_len
                k_prefix = k_len - target_len
                rab_b = rab[b]
                if is_paged_delta_q:
                    q_offset = k_len - q_len
                    if rab_b.shape[-2] >= k_len:
                        rab_dense[b, :, :k_len, :k_len].copy_(rab_b[:, :k_len, :k_len])
                    else:
                        rab_dense[b, :, q_offset : q_offset + q_len, :k_len].copy_(
                            rab_b[:, :q_len, :k_len]
                        )
                else:
                    q_target_begin = max_seqlen_q - target_len
                    k_target_begin = max_seqlen_k - target_len
                    rab_dense[b, :, :q_prefix, :k_prefix].copy_(
                        rab_b[:, :q_prefix, :k_prefix]
                    )
                    rab_dense[b, :, :q_prefix, k_target_begin:max_seqlen_k].copy_(
                        rab_b[:, :q_prefix, k_prefix:k_len]
                    )
                    rab_dense[b, :, q_target_begin:max_seqlen_q, :k_prefix].copy_(
                        rab_b[:, q_prefix:q_len, :k_prefix]
                    )
                    rab_dense[
                        b, :, q_target_begin:max_seqlen_q, k_target_begin:max_seqlen_k
                    ].copy_(rab_b[:, q_prefix:q_len, k_prefix:k_len])
            rab_arg = rab_dense.contiguous()
        elif rab.shape[-2] < q_dense_len or rab.shape[-1] < max_seqlen_k:
            rab_dense = rab.new_zeros((batch_size, q.shape[1], q_dense_len, max_seqlen_k))
            rab_dense[:, :, : rab.shape[-2], : rab.shape[-1]].copy_(rab)
            rab_arg = rab_dense.contiguous()
        else:
            rab_arg = rab[:, :, :q_dense_len, :max_seqlen_k].contiguous()
    else:
        # The SM120 CuTe DSL kernel keeps mRAB in its ABI.  The no-RAB branch
        # is compile-time eliminated, so this compact aligned dummy tensor is never
        # read by the kernel.
        rab_arg = hstu_varlen_fwd_120_bf16_cutedsl.dummy_rab_cache.get(q.device)
        if rab_arg is None:
            rab_arg = torch.empty((1, 1, 8, 8), dtype=torch.bfloat16, device=q.device)
            hstu_varlen_fwd_120_bf16_cutedsl.dummy_rab_cache[q.device] = rab_arg

    dummy_i32 = hstu_varlen_fwd_120_bf16_cutedsl.dummy_i32_cache.get(q.device)
    if dummy_i32 is None:
        dummy_i32 = torch.zeros((1,), dtype=torch.int32, device=q.device)
        hstu_varlen_fwd_120_bf16_cutedsl.dummy_i32_cache[q.device] = dummy_i32
    num_contexts_arg = num_contexts.contiguous() if is_context else dummy_i32
    num_targets_arg = num_targets.contiguous() if is_target else dummy_i32
    if is_arbitrary:
        assert func.dtype == torch.int32, "arbitrary func must be int32"
        assert func.ndim == 3, "arbitrary func must have shape [head_func, n_func, L_func]"
        assert func.shape[0] == 1, "SM120 CuTe DSL prototype expects head_func=1"
        func_num = int(func.shape[1])
        dense_func_rows = batch_size * q_dense_len
        if is_paged_delta_q or (not q_is_dense) or func.shape[2] < dense_func_rows:
            func_dense = torch.zeros(
                (func.shape[0], func_num, dense_func_rows + 256),
                dtype=func.dtype,
                device=func.device,
            )
            for b in range(batch_size):
                q_begin = int(cu_seqlens_q[b].item())
                q_end = int(cu_seqlens_q[b + 1].item())
                q_len = q_end - q_begin
                if is_paged_delta_q:
                    k_len = int(cu_seqlens_k[b + 1].item() - cu_seqlens_k[b].item())
                    dst_begin = b * q_dense_len + (k_len - q_len)
                else:
                    dst_begin = b * q_dense_len
                func_dense[:, :, dst_begin : dst_begin + q_end - q_begin].copy_(
                    func[:, :, q_begin:q_end]
                )
            func_arg = func_dense.contiguous()
        else:
            func_arg = func.contiguous()
    else:
        func_arg = dummy_i32.view(1, 1, 1)
        func_num = 0

    q_tensor, k_tensor, v_tensor, o_tensor, rab_tensor = [
        _make_compact_dynamic_tensor(t)
        for t in (q_dense, k_dense, v_dense, out_dense, rab_arg)
    ]
    num_contexts_tensor, num_targets_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=t.ndim - 1
        )
        for t in (num_contexts_arg, num_targets_arg)
    ]
    func_tensor = from_dlpack(func_arg.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=func_arg.ndim - 1
    )

    current_stream = cutlass_torch.default_stream()
    compile_key = (
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        q.shape[1],
        head_dim,
        kBlockM,
        kBlockN,
        num_threads,
        enable_fast_sigmoid,
        is_causal,
        is_local,
        is_context,
        is_target,
        is_arbitrary,
        func_num,
        target_group_size,
        window_size_left,
        window_size_right,
        alpha,
        scale_output,
        has_rab,
    )

    if compile_key not in hstu_varlen_fwd_120_bf16_cutedsl.compile_cache:
        hstu_fwd_sm120 = HSTUAttentionForwardSm120CuteDsl(
            cutlass.BFloat16,
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            q.shape[1],
            head_dim=head_dim,
            m_block_size=kBlockM,
            n_block_size=kBlockN,
            num_threads=num_threads,
            enable_fast_sigmoid=enable_fast_sigmoid,
            enable_block_rasterization=False,
            is_causal=is_causal,
            is_local=is_local,
            is_context=is_context,
            is_target=is_target,
            is_arbitrary=is_arbitrary,
            func_num=func_num,
            target_group_size=target_group_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            alpha=alpha,
            has_rab=has_rab,
            scale_output=scale_output,
        )
        with torch.cuda.nvtx.range("hstu_varlen_fwd_120_bf16_cutedsl_compile"):
            hstu_varlen_fwd_120_bf16_cutedsl.compile_cache[compile_key] = cute.compile(
                hstu_fwd_sm120,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                rab_tensor,
                num_contexts_tensor,
                num_targets_tensor,
                func_tensor,
                current_stream,
            )

    with torch.cuda.nvtx.range("hstu_varlen_fwd_120_bf16_cutedsl"):
        hstu_varlen_fwd_120_bf16_cutedsl.compile_cache[compile_key](
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            rab_tensor,
            num_contexts_tensor,
            num_targets_tensor,
            func_tensor,
            current_stream,
        )

    if q_is_dense and not is_paged_delta_q:
        return out_dense.view_as(q), None

    out = torch.empty_like(q)
    for b in range(batch_size):
        q_begin = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b + 1].item())
        q_len = q_end - q_begin
        target_len = _target_len(b)
        if is_paged_delta_q:
            k_len = int(cu_seqlens_k[b + 1].item() - cu_seqlens_k[b].item())
            q_offset = k_len - q_len
            out[q_begin:q_end].copy_(out_dense[b, q_offset : q_offset + q_len])
        elif target_len > 0:
            prefix_len = q_len - target_len
            target_begin = max_seqlen_q - target_len
            out[q_begin : q_begin + prefix_len].copy_(out_dense[b, :prefix_len])
            out[q_begin + prefix_len : q_end].copy_(
                out_dense[b, target_begin:max_seqlen_q]
            )
        else:
            out[q_begin:q_end].copy_(out_dense[b, :q_len])
    return out, None


hstu_varlen_fwd_120_bf16_cutedsl.compile_cache = {}
hstu_varlen_fwd_120_bf16_cutedsl.dummy_rab_cache = {}
hstu_varlen_fwd_120_bf16_cutedsl.dummy_i32_cache = {}
