#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024, NVIDIA Corporation & AFFILIATES.

# pyre-strict

import os
from typing import Any, Optional, Tuple
from .library import *  # noqa: F401, F403
import torch
import torch.nn as nn

def _hstu_debug_enabled() -> bool:
    return os.getenv("HSTU_DEBUG_SCALE_LOG", "0") == "1"

def _round_descale_to_e8m0(scale: torch.Tensor, min_scale: float = 1e-6) -> torch.Tensor:
    """
    Round positive descale to E8M0-dequantized values (power-of-two).
    This keeps python-side quantization consistent with kernel-side ue8m0 scale use.
    """
    s = torch.clamp(scale.to(torch.float32), min=min_scale)
    # E8M0 has no mantissa: keep only exponent. Use ceil(log2(.)) to match
    # cudaRoundPosInf behavior used in kernel float->e8m0 conversion.
    return torch.exp2(torch.ceil(torch.log2(s)))

def _debug_tensor_sample(name: str, t: Optional[torch.Tensor], max_elems: int = 8) -> None:
    if t is None:
        print(f"[HSTU_DEBUG] {name}: None")
        return
    flat = t.detach().reshape(-1)
    show_n = min(max_elems, flat.numel())
    sample = flat[:show_n]
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        sample = sample.float()
    else:
        sample = sample.to(torch.float32)
    sample_list = [float(x) for x in sample.cpu().tolist()]
    print(
        f"[HSTU_DEBUG] {name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"sample={sample_list}"
    )

def quantize_for_two_directions(x, seq_offsets, fp8_type=torch.float8_e4m3fn):
    B = seq_offsets.size(0) - 1
    fp8_max = 448.0 if fp8_type == torch.float8_e4m3fn else 57344.0
    # x: (total_seq, head, dim)
    if x.dim() != 3:
        raise ValueError("AssertError: x in quantize_for_two_directions should be three dimensions")

    with torch.no_grad():
        x_descale = torch.amax(x.abs(), dim=-1, keepdim=True).to(torch.float32) / fp8_max
        x_descale = _round_descale_to_e8m0(x_descale)
        x_quantized = (x / x_descale).to(fp8_type)
        x_descale = x_descale.squeeze(-1)
        x_descale = nn.functional.pad(x_descale, (0, 0, 0, 128)).to(torch.float32)
        x_descale = x_descale.transpose(1, 0).contiguous()

        cu_seqlens_xt_descale = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
        for i in range(B):
            actual_len = seq_offsets[i + 1] - seq_offsets[i]
            xt_descale_len = (actual_len + 127) // 128
            cu_seqlens_xt_descale[i + 1] = cu_seqlens_xt_descale[i] + xt_descale_len

        xt_descale = torch.zeros(cu_seqlens_xt_descale[-1], x.shape[1], x.shape[2], dtype=torch.float32, device='cuda')
        xt_quantized = x.to(fp8_type)
        for i in range(B):
            xt_descale_len = cu_seqlens_xt_descale[i + 1] - cu_seqlens_xt_descale[i]
            for j in range(xt_descale_len - 1):
                xt_descale[cu_seqlens_xt_descale[i] + j] = _round_descale_to_e8m0(
                    torch.amax(
                        x[seq_offsets[i] + j * 128 : seq_offsets[i] + (j + 1) * 128].abs(),
                        dim=0,
                        keepdim=True,
                    ) / fp8_max
                )
                xt_quantized[seq_offsets[i] + j * 128 : seq_offsets[i] + (j + 1) * 128] = (x[seq_offsets[i] + j * 128 : seq_offsets[i] + (j + 1) * 128] / xt_descale[cu_seqlens_xt_descale[i] + j]).to(fp8_type)

            xt_descale[cu_seqlens_xt_descale[i] + xt_descale_len - 1] = _round_descale_to_e8m0(
                torch.amax(
                    x[seq_offsets[i] + (xt_descale_len - 1) * 128 : seq_offsets[i+1]].abs(),
                    dim=0,
                    keepdim=True,
                ) / fp8_max
            )
            xt_quantized[seq_offsets[i] + (xt_descale_len - 1) * 128 : seq_offsets[i+1]] = (x[seq_offsets[i] + (xt_descale_len - 1) * 128 : seq_offsets[i+1]] / xt_descale[cu_seqlens_xt_descale[i] + xt_descale_len - 1]).to(fp8_type)

    return x_quantized, x_descale, xt_quantized, xt_descale, cu_seqlens_xt_descale

def quantize_for_block_scale_qk_along_d(x, seq_offsets, fp8_type=torch.float8_e4m3fn):
    """
    Quantize Q/K with 1x128 granularity along D (K-dimension for QK GEMM).
    For each token/head and each 128-wide D chunk, compute one descale.
    """
    if x.dim() != 3:
        raise ValueError("AssertError: x in quantize_for_block_scale_qk_along_d should be three dimensions")
    B = seq_offsets.size(0) - 1
    head = x.size(1)
    dim = x.size(2)
    if dim % 128 != 0:
        raise ValueError(f"AssertError: D must be divisible by 128 for q/k K-dim block scale, got D={dim}")
    d_chunks = dim // 128
    fp8_max = 448.0 if fp8_type == torch.float8_e4m3fn else 57344.0

    cu_seqlens_x_descale = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
    x_quantized = torch.empty_like(x, dtype=fp8_type, device='cuda')
    x_descale_list = []

    with torch.no_grad():
        for i in range(B):
            start = int(seq_offsets[i].item())
            end = int(seq_offsets[i + 1].item())
            actual_len = end - start

            # User-requested constraint for current debug path.
            if actual_len % 128 != 0:
                raise ValueError(
                    f"AssertError: quant_mode=2 requires N divisible by 128, got N={actual_len} in batch {i}"
                )

            cur = x[start:end].view(actual_len, head, d_chunks, 128)
            cur_scale = torch.amax(cur.abs(), dim=3, keepdim=False).to(torch.float32) / fp8_max
            cur_scale = _round_descale_to_e8m0(cur_scale)
            cur_quant = (cur / cur_scale.unsqueeze(-1)).to(fp8_type).view(actual_len, head, dim)
            x_quantized[start:end] = cur_quant

            # Layout for kernel side: [head, total_scale_rows], with one scale-row per (token, d_chunk)
            cur_scale_flat = cur_scale.permute(0, 2, 1).reshape(actual_len * d_chunks, head)  # [N*Dblk, H]
            x_descale_list.append(cur_scale_flat)
            cu_seqlens_x_descale[i + 1] = cu_seqlens_x_descale[i] + actual_len * d_chunks

    x_descale = torch.cat(x_descale_list, dim=0).transpose(1, 0).contiguous()  # [H, total]
    return x_quantized, x_descale, cu_seqlens_x_descale


def quantize_for_block_scale_v_along_n(x, seq_offsets, block_size=128, fp8_type=torch.float8_e4m3fn):
    # x: (total_seq, head, dim)
    # V quantization along N with 1x128 granularity.
    if x.dim() != 3:
        raise ValueError("AssertError: x in quantize_for_block_scale_v_along_n should be three dimensions")
    B = seq_offsets.size(0) - 1
    head = x.size(1)
    dim = x.size(2)
    fp8_max = 448.0 if fp8_type == torch.float8_e4m3fn else 57344.0

    cu_seqlens_x_descale = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
    x_quantized_list = []
    x_descale_list = []

    with torch.no_grad():
        for i in range(B):
            actual_len = seq_offsets[i + 1] - seq_offsets[i]
            if int(actual_len.item()) % 128 != 0:
                raise ValueError(
                    f"AssertError: quant_mode=2 requires N divisible by 128, got N={int(actual_len.item())} in batch {i}"
                )
            cur_bs_tensor = x[seq_offsets[i]:(seq_offsets[i] + actual_len)]
            actual_len_padding_block_num = (actual_len + block_size - 1) // block_size
            cu_seqlens_x_descale[i + 1] = cu_seqlens_x_descale[i] + actual_len_padding_block_num

            cur_padding_len = actual_len_padding_block_num * block_size - actual_len
            if cur_padding_len > 0:
                pad_tensor = torch.zeros(cur_padding_len, cur_bs_tensor.shape[1], cur_bs_tensor.shape[2], device=cur_bs_tensor.device, dtype=cur_bs_tensor.dtype)
                cur_bs_tensor = torch.cat([cur_bs_tensor, pad_tensor], dim=0)
            else:
                cur_bs_tensor = cur_bs_tensor
            cur_bs_tensor = cur_bs_tensor.view(actual_len_padding_block_num, block_size, head, dim)
            cur_bs_scale_tensor = torch.amax(cur_bs_tensor.abs(), dim=(1, 3), keepdim=True).to(torch.float32) / fp8_max
            cur_bs_scale_tensor = _round_descale_to_e8m0(cur_bs_scale_tensor)
            x_descale_list.append(cur_bs_scale_tensor)
            cur_bs_tensor_quantized = (cur_bs_tensor / cur_bs_scale_tensor).to(fp8_type).view(actual_len_padding_block_num * block_size, head, dim)[0:actual_len] #[actual_len_padding_block_num * cur_block_size, head, dim] - > [actual_len, head, dim]
            x_quantized_list.append(cur_bs_tensor_quantized)

        x_quantized = torch.cat(x_quantized_list, dim=0)
        x_descale = torch.cat(x_descale_list, dim=0) # [total_seq, head]

    assert x_quantized.shape == x.shape, "assert x_quantized shape must equal to x shape"
    return x_quantized, x_descale.squeeze(1).squeeze(-1).transpose(1, 0).contiguous(), cu_seqlens_x_descale #For x_descale, the original layout is ([sum(cur_bs_len/bm), head]: (head, 1)), and we transform into ([head, sum(cur_bs_len/bm): (sum(cur_bs_len/bm), 1)])


def pack_descale_to_e8m0x4_int32(descale: torch.Tensor) -> torch.Tensor:
    # Convert float descale tensor (already e8m0-rounded in this path) into int32 packed e8m0x4.
    s = torch.clamp(descale.to(torch.float32), min=1e-10)
    exp_unbiased = torch.ceil(torch.log2(s))
    exp_biased = torch.clamp(exp_unbiased + 127.0, 0.0, 255.0).to(torch.uint8)
    word = exp_biased.to(torch.int32)
    return (word | (word << 8) | (word << 16) | (word << 24)).contiguous()


# Backward compatibility for older call sites.
def quantize_for_block_scale(x, seq_offsets, block_size=128, fp8_type=torch.float8_e4m3fn):
    return quantize_for_block_scale_v_along_n(x, seq_offsets, block_size=block_size, fp8_type=fp8_type)

def get_bm_and_bn_block_size_fwd(rab, dim):
    """
    Design for fp8, Returns the block size for BM and BN. Need to be the same as the "get_tile_size_fwd" function.
    BM: Block size for the first dimension of the input tensor.
    BN: Block size for the second dimension of the input tensor.
    """
    if rab is not None:
        if dim == 64:
            return 128, 64   # kBlockM=128, kBlockN=64 (utils.h: {128, 64, 4} for FP8+rab+hdim64)
        elif dim == 128:
            return 128, 128  # kBlockM=128, kBlockN=128 (utils.h: {128, 128, 8} for FP8+rab+hdim128)
        else:
            return 128, 128  # kBlockM=64, kBlockN=128 (utils.h: {64, 128, 8} for FP8+rab+hdim>128)
    else:
        if dim == 64:
            return 128, 128
        elif dim == 128:
            return 128, 128
        else:
            return 128, 64

def get_bm_and_bn_block_size_bwd():
    """
    Design for fp8, Returns the block size for BM and BN. Need to be the same as the "get_tile_size_bwd" function.
    BM: Block size for the first dimension of the input tensor.
    BN: Block size for the second dimension of the input tensor.
    """
    return 64, 128

def quantize_for_head_batch_tensor(x, seq_offsets, quant_mode=3, fp8_type=torch.float8_e4m3fn):
    B = seq_offsets.size(0) - 1
    head = x.size(1)
    fp8_max = 448.0 if fp8_type == torch.float8_e4m3fn else 57344.0
    # x: (total_seq, head, dim)
    if x.dim() != 3:
        raise ValueError("AssertError: x in quantize_for_head_batch_tensor should be three dimensions")
    if quant_mode != 3 and quant_mode != 4 and quant_mode != 5:
        raise ValueError("AssertError: quant_mode in quantize_for_head_batch_tensor should be 3, 4 or 5")

    if quant_mode == 3:
        with torch.no_grad():
            x_descale = torch.zeros(B, head, dtype=torch.float32, device='cuda')
            x_quantized = torch.zeros_like(x, dtype=fp8_type, device='cuda')
            for i in range(B):
                x_descale[i, :] = _round_descale_to_e8m0(
                    torch.amax(
                        x[seq_offsets[i]:seq_offsets[i+1], :, :].abs(),
                        dim=(0, 2),
                        keepdim=True,
                    ).squeeze(0).squeeze(-1) / fp8_max
                )
                x_quantized[seq_offsets[i]:seq_offsets[i+1], :, :] = (x[seq_offsets[i]:seq_offsets[i+1], :, :] / x_descale[i, :].unsqueeze(0).unsqueeze(-1)).to(fp8_type)
        return x_quantized, x_descale
    elif quant_mode == 4:
        with torch.no_grad():
            x_descale = torch.zeros(B, dtype=torch.float32, device='cuda')
            x_quantized = torch.zeros_like(x, dtype=fp8_type, device='cuda')
            for i in range(B):
                x_descale[i] = _round_descale_to_e8m0(
                    torch.amax(x[seq_offsets[i]:seq_offsets[i+1], :, :].abs(), keepdim=True) / fp8_max
                )
                x_quantized[seq_offsets[i]:seq_offsets[i+1], :, :] = (x[seq_offsets[i]:seq_offsets[i+1], :, :] / x_descale[i]).to(fp8_type)
        return x_quantized, x_descale
    else:
        with torch.no_grad():
            x_descale = _round_descale_to_e8m0(
                torch.amax(x.abs(), keepdim=True).squeeze(0).squeeze(-1) / fp8_max
            )
            x_quantized = (x / x_descale).to(fp8_type)
        return x_quantized, x_descale


class HstuAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(  # pyre-ignore[14]
        ctx,  # pyre-ignore[2]
        q: torch.Tensor,  # need grad
        k: torch.Tensor,  # need grad
        v: torch.Tensor,  # need grad
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        seqused_q: Optional[torch.Tensor],
        seqused_k: Optional[torch.Tensor],
        max_seqlen_q: int,
        max_seqlen_k: int,
        scaling_seqlen: int,
        num_contexts: torch.Tensor,
        num_targets: torch.Tensor,
        target_group_size: int,
        window_size: Tuple[int, int] = (-1, -1),
        alpha: float = 1.0,
        rab: Optional[torch.Tensor] = None,  # need grad
        has_drab: bool = False,
        func: Optional[torch.Tensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        page_offsets: Optional[torch.Tensor] = None,
        page_ids: Optional[torch.Tensor] = None,
        last_page_lens: Optional[torch.Tensor] = None,
        quant_mode: Optional[int] = -1,
    ) -> torch.Tensor:
        assert q.dim() == 3, "q shape should be (L, num_heads, head_dim)"
        assert k.dim() == 3, "k shape should be (L, num_heads, head_dim)"
        assert v.dim() == 3, "v shape should be (L, num_heads, hidden_dim)"

        major_version = torch.cuda.get_device_capability()[0]
        assert major_version == 8 or major_version == 9 or major_version == 10 or major_version >= 12, \
            f"Only support sm80, sm90, sm100, and sm120+, got sm{major_version}0"
        if major_version >= 12:
            # SM120+ (Blackwell): dispatch to hstu_varlen_fwd_120
            q_descale = None
            k_descale = None
            v_descale = None
            sf_q_packed = None
            sf_k_packed = None
            sf_v_packed = None
            cu_seqlens_q_block_descale = None
            cu_seqlens_kv_block_descale = None
            cu_seqlens_v_block_descale = None
            if quant_mode == 0:
                # Per-tensor FP8 quantization
                fp8_max = 448.0
                q_scale = _round_descale_to_e8m0(q.float().abs().max() / fp8_max)
                k_scale = _round_descale_to_e8m0(k.float().abs().max() / fp8_max)
                v_scale = _round_descale_to_e8m0(v.float().abs().max() / fp8_max)
                q = (q / q_scale).to(torch.float8_e4m3fn)
                k = (k / k_scale).to(torch.float8_e4m3fn)
                v = (v / v_scale).to(torch.float8_e4m3fn)
                q_descale = q_scale.reshape(1).to(torch.float32)
                k_descale = k_scale.reshape(1).to(torch.float32)
                v_descale = v_scale.reshape(1).to(torch.float32)
            elif quant_mode == 2:
                # Blockwise FP8 quantization
                dim = q.shape[-1]
                bm, bn = get_bm_and_bn_block_size_fwd(rab, dim)
                q_raw, k_raw, v_raw = q, k, v
                q, q_descale, cu_seqlens_q_block_descale = quantize_for_block_scale_qk_along_d(
                    q, cu_seqlens_q, fp8_type=torch.float8_e4m3fn)
                k, k_descale, cu_seqlens_kv_block_descale = quantize_for_block_scale_qk_along_d(
                    k, cu_seqlens_k, fp8_type=torch.float8_e4m3fn)
                v, v_descale, cu_seqlens_v_block_descale = quantize_for_block_scale_v_along_n(
                    v, cu_seqlens_k, block_size=bn, fp8_type=torch.float8_e4m3fn)
                # Col-major V: only for WS TMA path (!has_rab).
                # Has_rab=True falls back to cp.async which requires row-major V.
                if rab is None:
                    v = v.permute(2, 1, 0).contiguous().permute(2, 1, 0)
                sf_q_packed = pack_descale_to_e8m0x4_int32(q_descale)
                sf_k_packed = pack_descale_to_e8m0x4_int32(k_descale)
                # v_descale has shape [H, total_blocks]; expand to [H, total_tokens] so
                # sf_v_packed has the same layout as sf_k_packed and TMA SFV can use
                # identical kBlockN-element tiles indexed by nb_abs (same as SFB).
                sf_v_packed = pack_descale_to_e8m0x4_int32(v_descale).repeat_interleave(bn, dim=1)
                if _hstu_debug_enabled():
                    print(
                        f"[HSTU_DEBUG] quant_mode=2 bm={bm} bn={bn} "
                        f"q_head_stride={q_descale.stride(0) if q_descale is not None else 'None'} "
                        f"kv_head_stride={k_descale.stride(0) if k_descale is not None else 'None'}"
                    )
                    _debug_tensor_sample("q_raw", q_raw)
                    _debug_tensor_sample("k_raw", k_raw)
                    _debug_tensor_sample("v_raw", v_raw)
                    _debug_tensor_sample("q_fp8", q)
                    _debug_tensor_sample("k_fp8", k)
                    _debug_tensor_sample("v_fp8", v)
                    _debug_tensor_sample("q_descale", q_descale)
                    _debug_tensor_sample("k_descale", k_descale)
                    _debug_tensor_sample("v_descale", v_descale)
                    _debug_tensor_sample("sf_q_packed", sf_q_packed)
                    _debug_tensor_sample("sf_k_packed", sf_k_packed)
                    _debug_tensor_sample("sf_v_packed", sf_v_packed)
                    _debug_tensor_sample("cu_seqlens_q_block_descale", cu_seqlens_q_block_descale)
                    _debug_tensor_sample("cu_seqlens_kv_block_descale", cu_seqlens_kv_block_descale)
            # SM120 kernel always reads RAB as BF16; convert float16 RAB if needed.
            rab_kernel = rab.to(torch.bfloat16) if rab is not None and rab.dtype == torch.float16 else rab
            out, rab_padded = torch.ops.fbgemm.hstu_varlen_fwd_120(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                scaling_seqlen,
                num_contexts,
                num_targets,
                target_group_size,
                window_size[0],
                window_size[1],
                alpha,
                rab_kernel,
                func,
                quant_mode if quant_mode is not None else -1,
                q_descale,
                k_descale,
                v_descale,
                sf_q_packed,
                sf_k_packed,
                sf_v_packed,
                cu_seqlens_q_block_descale,
                cu_seqlens_kv_block_descale,
                cu_seqlens_v_block_descale,
            )
        elif major_version == 8:
            out, rab_padded = torch.ops.fbgemm.hstu_varlen_fwd_80(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                scaling_seqlen,
                num_contexts,
                num_targets,
                target_group_size,
                window_size[0],
                window_size[1],
                alpha,
                rab,
                func,
                kv_cache,
                page_offsets,
                page_ids,
                last_page_lens,
            )
        elif major_version == 9:
            vt = None
            q_descale = None
            k_descale = None
            v_descale = None
            vt_descale = None
            cu_seqlens_vt_descale = None
            cu_seqlens_q_block_descale = None
            cu_seqlens_kv_block_descale = None
            ctx.q_fp16 = q
            ctx.k_fp16 = k
            ctx.v_fp16 = v
            output_dtype = 0 if q.dtype == torch.bfloat16 else 1
            if quant_mode == 0:
                q = q.to(torch.float8_e4m3fn)
                k = k.to(torch.float8_e4m3fn)
                v = v.to(torch.float8_e4m3fn)
                q_descale = torch.tensor([1.0], dtype=torch.float32, device='cuda')
                k_descale = torch.tensor([1.0], dtype=torch.float32, device='cuda')
                v_descale = torch.tensor([1.0], dtype=torch.float32, device='cuda')
            elif quant_mode == 1:
                q, q_descale, _, _, _ = quantize_for_two_directions(q, cu_seqlens_q, fp8_type=torch.float8_e4m3fn)
                k, k_descale, _, _, _ = quantize_for_two_directions(k, cu_seqlens_k, fp8_type=torch.float8_e4m3fn)
                v, v_descale, vt, vt_descale, cu_seqlens_vt_descale = quantize_for_two_directions(v, cu_seqlens_k, fp8_type=torch.float8_e4m3fn)
                vt = vt.transpose(0, 2).contiguous().transpose(0, 2).detach()
            elif quant_mode == 2: #block_scale
                dim = q.shape[-1]
                bm, bn = get_bm_and_bn_block_size_fwd(rab, dim)
                q, q_descale, cu_seqlens_q_block_descale = quantize_for_block_scale_qk_along_d(
                    q, cu_seqlens_q, fp8_type=torch.float8_e4m3fn)
                k, k_descale, cu_seqlens_kv_block_descale = quantize_for_block_scale_qk_along_d(
                    k, cu_seqlens_k, fp8_type=torch.float8_e4m3fn)
                v, v_descale, _ = quantize_for_block_scale_v_along_n(
                    v, cu_seqlens_k, block_size=bn, fp8_type=torch.float8_e4m3fn)
            elif quant_mode == 3 or quant_mode == 4 or quant_mode == 5:
                q, q_descale = quantize_for_head_batch_tensor(q, cu_seqlens_q, quant_mode=quant_mode, fp8_type=torch.float8_e4m3fn)
                k, k_descale = quantize_for_head_batch_tensor(k, cu_seqlens_k, quant_mode=quant_mode, fp8_type=torch.float8_e4m3fn)
                v, v_descale = quantize_for_head_batch_tensor(v, cu_seqlens_k, quant_mode=quant_mode, fp8_type=torch.float8_e4m3fn)

            out, rab_padded = torch.ops.fbgemm.hstu_varlen_fwd_90(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                scaling_seqlen,
                num_contexts,
                num_targets,
                target_group_size,
                window_size[0],
                window_size[1],
                alpha,
                rab,
                func,
                quant_mode,
                output_dtype,
                vt,
                cu_seqlens_vt_descale,
                q_descale,
                k_descale,
                v_descale,
                vt_descale,
                cu_seqlens_q_block_descale,
                cu_seqlens_kv_block_descale,
            )
        else:
            assert seqused_q is None and seqused_k is None, \
                "HSTU-Blackwell does not support seqused_q and seqused_k"
            from fbgemm_gpu.experimental.hstu.hstu_blackwell import hstu_ops_gpu as _sm100
            out, rab_padded = _sm100.hstu_varlen_fwd_100(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                num_contexts,
                num_targets,
                target_group_size,
                window_size[0],
                window_size[1],
                alpha,
                rab,
                func,
                paged_kv=kv_cache,
                page_ids=page_ids,
                page_indptrs=page_offsets,
            )

        ctx.save_for_backward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            num_contexts,
            num_targets,
            rab_padded,
        )
        ctx.major_version = major_version
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.scaling_seqlen = scaling_seqlen
        ctx.target_group_size = target_group_size
        ctx.alpha = alpha
        ctx.window_size_left = window_size[0]
        ctx.window_size_right = window_size[1]
        ctx.has_drab = has_drab
        ctx.func = func
        ctx.quant_mode = quant_mode
        return out

    @staticmethod
    def backward(  # pyre-ignore[14]
        ctx,  # pyre-ignore[2]
        dout: torch.Tensor,
        *args: Any,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            num_contexts,
            num_targets,
            rab_padded,
        ) = ctx.saved_tensors

        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        scaling_seqlen = ctx.scaling_seqlen
        target_group_size = ctx.target_group_size
        window_size_left = ctx.window_size_left
        window_size_right = ctx.window_size_right
        alpha = ctx.alpha
        has_drab = ctx.has_drab
        func = ctx.func
        quant_mode = ctx.quant_mode

        if ctx.major_version == 8:
            dq, dk, dv, dRab = torch.ops.fbgemm.hstu_varlen_bwd_80(
                dout,
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                scaling_seqlen,
                None,
                None,
                None,
                num_contexts,
                num_targets,
                target_group_size,
                window_size_left,
                window_size_right,
                alpha,
                rab_padded,
                has_drab,
                func,
                False,  # deterministic
            )
        elif ctx.major_version == 9:
            dout_t = None
            qt = None
            kt = None
            q_descale = None
            qt_descale = None
            k_descale = None
            kt_descale = None
            v_descale = None
            do_descale = None
            dot_descale = None
            cu_seqlens_qt_descale = None
            cu_seqlens_kt_descale = None
            cu_seqlens_q_block_descale = None
            cu_seqlens_kv_block_descale = None
            quant_mode = ctx.quant_mode
            bwd_fp8_type = torch.float8_e4m3fn
            output_dtype = 0 if dout.dtype == torch.bfloat16 else 1
            if quant_mode == 0:
                q = ctx.q_fp16.to(bwd_fp8_type)
                k = ctx.k_fp16.to(bwd_fp8_type)
                v = ctx.v_fp16.to(bwd_fp8_type)
                dout = dout.to(bwd_fp8_type)
                q_descale = torch.tensor([1.0], dtype=torch.float32, device='cuda')
                k_descale = torch.tensor([1.0], dtype=torch.float32, device='cuda')
                v_descale = torch.tensor([1.0], dtype=torch.float32, device='cuda')
                do_descale = torch.tensor([1.0], dtype=torch.float32, device='cuda')
            elif quant_mode == 1:
                q, q_descale, qt, qt_descale, cu_seqlens_qt_descale = quantize_for_two_directions(ctx.q_fp16, cu_seqlens_q, fp8_type=bwd_fp8_type)
                qt = qt.transpose(0, 2).contiguous().transpose(0, 2).detach()
                k, k_descale, kt, kt_descale, cu_seqlens_kt_descale = quantize_for_two_directions(ctx.k_fp16, cu_seqlens_k, fp8_type=bwd_fp8_type)
                kt = kt.transpose(0, 2).contiguous().transpose(0, 2).detach()
                v, v_descale, _, _, _ = quantize_for_two_directions(ctx.v_fp16, cu_seqlens_k, fp8_type=bwd_fp8_type)
                dout, do_descale, dout_t, dot_descale, _ = quantize_for_two_directions(dout, cu_seqlens_q, fp8_type=bwd_fp8_type)
                dout_t = dout_t.transpose(0, 2).contiguous().transpose(0, 2).detach()
            elif quant_mode == 2:
                dim = q.shape[-1]
                bm, bn = get_bm_and_bn_block_size_bwd()
                q, q_descale, cu_seqlens_q_block_descale = quantize_for_block_scale_qk_along_d(
                    ctx.q_fp16, cu_seqlens_q, fp8_type=bwd_fp8_type)
                k, k_descale, cu_seqlens_kv_block_descale = quantize_for_block_scale_qk_along_d(
                    ctx.k_fp16, cu_seqlens_k, fp8_type=bwd_fp8_type)
                v, v_descale, _ = quantize_for_block_scale_v_along_n(
                    ctx.v_fp16, cu_seqlens_k, block_size=bn, fp8_type=bwd_fp8_type)
                dout, do_descale, _ = quantize_for_block_scale_qk_along_d(
                    dout, cu_seqlens_q, fp8_type=bwd_fp8_type)
            elif quant_mode == 3 or quant_mode == 4 or quant_mode == 5:
                q, q_descale = quantize_for_head_batch_tensor(ctx.q_fp16, cu_seqlens_q, quant_mode=ctx.quant_mode, fp8_type=bwd_fp8_type)
                k, k_descale = quantize_for_head_batch_tensor(ctx.k_fp16, cu_seqlens_k, quant_mode=ctx.quant_mode, fp8_type=bwd_fp8_type)
                v, v_descale = quantize_for_head_batch_tensor(ctx.v_fp16, cu_seqlens_k, quant_mode=ctx.quant_mode, fp8_type=bwd_fp8_type)
                dout, do_descale = quantize_for_head_batch_tensor(dout, cu_seqlens_q, quant_mode=ctx.quant_mode, fp8_type=bwd_fp8_type)

            dq, dk, dv, dRab = torch.ops.fbgemm.hstu_varlen_bwd_90(
                dout,
                dout_t,
                q,
                qt,
                k,
                kt,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                scaling_seqlen,
                None,
                None,
                None,
                num_contexts,
                num_targets,
                target_group_size,
                window_size_left,
                window_size_right,
                alpha,
                quant_mode,
                rab_padded,
                has_drab,
                func,
                q_descale,
                qt_descale,
                k_descale,
                kt_descale,
                v_descale,
                do_descale,
                dot_descale,
                cu_seqlens_qt_descale,
                cu_seqlens_kt_descale,
                cu_seqlens_q_block_descale,
                cu_seqlens_kv_block_descale,
                output_dtype,
                False,  # deterministic
            )
        else:
            assert seqused_q is None and seqused_k is None, \
                "HSTU-Blackwell does not support seqused_q and seqused_k"
            from fbgemm_gpu.experimental.hstu.hstu_blackwell import hstu_ops_gpu as _sm100
            dq, dk, dv, dRab = _sm100.hstu_varlen_bwd_100(
                dout,
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                None,
                None,
                None,
                num_contexts,
                num_targets,
                target_group_size,
                window_size_left,
                window_size_right,
                alpha,
                rab_padded,
                has_drab,
                func,
                False,  # deterministic
            )

        # q & k grad shape
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dRab if ctx.has_drab else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# pyre-ignore[3]
def hstu_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    scaling_seqlen: int,
    num_contexts: torch.Tensor,
    num_targets: torch.Tensor,
    target_group_size: int = 1,
    window_size: Tuple[int, int] = (-1, -1),
    alpha: float = 1.0,
    rab: Optional[torch.Tensor] = None,
    has_drab: bool = False,
    kv_cache: Optional[torch.Tensor] = None,
    page_offsets: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    last_page_lens: Optional[torch.Tensor] = None,
    func: Optional[torch.Tensor] = None,
    quant_mode: Optional[int] = -1,
):
    """
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths of the sequences in the batch, used to index into kv.
        seqused_q: (batch_size,). The number of valid tokens in each query sequence. If None, all tokens are valid.
        seqused_k: (batch_size,). The number of valid tokens in each key sequence. If None, all tokens are valid.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        scaling_seqlen: int. Scaling factor for the output. Default is -1, which means same as max_seqlen_q.
        num_contexts: (batch_size,). Number of context tokens in each batch.
        num_targets: (batch_size,). Number of target tokens in each batch.
        target_group_size: int. Number of target tokens in each group.
        window_size: (left, right). If not (-1, -1), implements sliding window local attention. If (-1, 0), implements causal attention.
        alpha: float. Scaling factor between add rab and silu.
        rab: (batch_size, max_seqlen_k, max_seqlen_k). Random access bias for the key.
        has_drab: bool. Whether to apply random access bias for the key.
        kv_cache: (page_num, 2, page_size, nheads, headdim). Key and value paged cache.
        page_offsets: (batch_size + 1,). The cumulative sequence lengths of the page_ptr in the batch, used to index into kv_cache.
        page_ids: (page_offsets[-1],). The ids of the pages in the batch.
        last_page_lens: (batch_size,). The lengths of the last pages in the batch.
        func: (nheads, total_q + 256). Function to describe the mask shape in arbitrary mask.
        quant_mode: int. Quantization mode.
    Return:
        out: (total, nheads, headdim).
    """
    if has_drab and (rab is None):
        raise ValueError(
            "AssertError: rab is None, but has_drab is True, is not allowed in backward"
        )
    if num_contexts is not None and window_size != (-1, 0):
        raise ValueError(
            "AssertError: context is True and causal is not True, this is undefined behavior"
        )
    if num_targets is not None and window_size != (-1, 0):
        raise ValueError(
            "AssertError: target is True and causal is not True, this is undefined behavior"
        )
    if num_targets is None and target_group_size < 1:
        raise ValueError(
            "AssertError: target_group_size should be greater than 0 when target is True"
        )
    if max_seqlen_q > max_seqlen_k:
        raise ValueError(
            "AssertError: seq_len_q >= seq_len_k, this is undefined behavior"
        )

    return HstuAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        scaling_seqlen,
        num_contexts,
        num_targets,
        target_group_size,
        window_size,
        alpha,
        rab,
        has_drab,
        func,
        kv_cache,
        page_offsets,
        page_ids,
        last_page_lens,
        quant_mode,
    )


class HstuAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        seqused_q: Optional[torch.Tensor],
        seqused_k: Optional[torch.Tensor],
        max_seqlen_q: int,
        max_seqlen_k: int,
        scaling_seqlen: int = -1,
        num_contexts: Optional[torch.Tensor] = None,
        num_targets: Optional[torch.Tensor] = None,
        target_group_size: Optional[int] = 1,
        window_size: Tuple[int, int] = (-1, -1),
        alpha: float = 1.0,
        rab: Optional[torch.Tensor] = None,
        has_drab: Optional[bool] = False,
        func: Optional[torch.Tensor] = None,
    ):
        q = qkv[:, 0, :, :].detach()
        k = qkv[:, 1, :, :].detach()
        v = qkv[:, 2, :, :].detach()
        major_version = torch.cuda.get_device_capability()[0]
        assert major_version == 8 or major_version == 9 or major_version == 10 or major_version >= 12, "Only support sm8x, sm90, sm100, and sm120+"
        if major_version >= 12:
            out, rab_padded = torch.ops.fbgemm.hstu_varlen_fwd_120(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                seqused_q, seqused_k,
                max_seqlen_q, max_seqlen_k,
                scaling_seqlen,
                num_contexts, num_targets, target_group_size,
                window_size[0], window_size[1],
                alpha, rab, func,
                -1,   # quant_mode = BF16
                None, None, None,  # descale_q/k/v
                None, None, None,  # sf_q/k/v_packed
                None, None, None,  # cu_seqlens_q/kv/v_block_descale
            )
        elif major_version == 8:
            out, rab_padded = torch.ops.fbgemm.hstu_varlen_fwd_80(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                scaling_seqlen,
                num_contexts,
                num_targets,
                target_group_size,
                window_size[0],
                window_size[1],
                alpha,
                rab,
                func
            )
        elif major_version == 9:
            out, rab_padded = torch.ops.fbgemm.hstu_varlen_fwd_90(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                scaling_seqlen,
                num_contexts,
                num_targets,
                target_group_size,
                window_size[0],
                window_size[1],
                alpha,
                rab,
                func,
                -1, # quant_mode
                0 if q.dtype == torch.bfloat16 else 1,
            )
        else:
            assert seqused_q is None and seqused_k is None, \
                "HSTU-Blackwell does not support seqused_q and seqused_k"
            from fbgemm_gpu.experimental.hstu.hstu_blackwell import hstu_ops_gpu as _sm100
            out, rab_padded = _sm100.hstu_varlen_fwd_100(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                num_contexts,
                num_targets,
                target_group_size,
                window_size[0],
                window_size[1],
                alpha,
                rab,
                func,
            )

        ctx.save_for_backward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            num_contexts,
            num_targets,
            rab_padded,
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.scaling_seqlen = scaling_seqlen
        ctx.target_group_size = target_group_size
        ctx.alpha = alpha
        ctx.window_size_left = window_size[0]
        ctx.window_size_right = window_size[1]
        ctx.has_drab = has_drab
        ctx.func = func
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            num_contexts,
            num_targets,
            rab_padded,
        ) = ctx.saved_tensors

        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        scaling_seqlen = ctx.scaling_seqlen
        target_group_size = ctx.target_group_size
        window_size_left = ctx.window_size_left
        window_size_right = ctx.window_size_right
        alpha = ctx.alpha
        has_drab = ctx.has_drab
        func = ctx.func
        qkv_shape = (q.shape[0], 3, q.shape[1], q.shape[2])
        dqkv = torch.empty(qkv_shape, device=q.device, dtype=q.dtype)
        major_version = torch.cuda.get_device_capability()[0]
        assert major_version == 8 or major_version == 9 or major_version == 10, "Only support sm8x and sm90 and sm100"
        if major_version == 8:
            _, _, _, dRab = torch.ops.fbgemm.hstu_varlen_bwd_80(
                dout,
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                scaling_seqlen,
                dqkv[:,0,:,:], # dq
                dqkv[:,1,:,:], # dk
                dqkv[:,2,:,:], # dv
                num_contexts,
                num_targets,
                target_group_size,
                window_size_left,
                window_size_right,
                alpha,
                rab_padded,
                has_drab,
                func,
                False,  # deterministic
            )
        elif major_version == 9:
            _, _, _, dRab = torch.ops.fbgemm.hstu_varlen_bwd_90(
                dout,
                None,
                q,
                None,
                k,
                None,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                scaling_seqlen,
                dqkv[:,0,:,:], # dq
                dqkv[:,1,:,:], # dk
                dqkv[:,2,:,:], # dv
                num_contexts,
                num_targets,
                target_group_size,
                window_size_left,
                window_size_right,
                alpha,
                -1, # quant_mode
                rab_padded,
                has_drab,
                func,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0 if q.dtype == torch.bfloat16 else 1,
                False,  # deterministic
            )
        else:
            assert seqused_q is None and seqused_k is None, \
                "HSTU-Blackwell does not support seqused_q and seqused_k"
            from fbgemm_gpu.experimental.hstu.hstu_blackwell import hstu_ops_gpu as _sm100
            _, _, _, dRab = _sm100.hstu_varlen_bwd_100(
                dout,
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dqkv[:,0,:,:], # dq
                dqkv[:,1,:,:], # dk
                dqkv[:,2,:,:], # dv
                num_contexts,
                num_targets,
                target_group_size,
                window_size_left,
                window_size_right,
                alpha,
                rab_padded,
                has_drab,
                func,
                False,  # deterministic
            )
        if has_drab:
            rab_head = rab_padded.size(1)
            dRab = dRab.view(-1, rab_head, max_seqlen_k, max_seqlen_k)

        # q & k grad shape
        return (
            dqkv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dRab if ctx.has_drab else None,
            None,
            None
        )

def hstu_attn_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    scaling_seqlen: int = -1,
    num_contexts: Optional[torch.Tensor] = None,
    num_targets: Optional[torch.Tensor] = None,
    target_group_size: Optional[int] = 1,
    window_size: Tuple[int, int] = (-1, -1),
    alpha: float = 1.0,
    rab: Optional[torch.Tensor] = None,
    has_drab: Optional[bool] = False,
    func: Optional[torch.Tensor] = None,
):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths of the sequences in the batch, used to index into kv.
        seqused_q: (batch_size,). The number of valid tokens in each query sequence. If None, all tokens are valid.
        seqused_k: (batch_size,). The number of valid tokens in each key sequence. If None, all tokens are valid.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        scaling_seqlen: int. Scaling factor for the output. Default is -1, which means same as max_seqlen_q.
        num_contexts: (batch_size,). Number of context tokens in each batch.
        num_targets: (batch_size,). Number of target tokens in each batch.
        target_group_size: int. Number of target tokens in each group.
        window_size: (left, right). If not (-1, -1), implements sliding window local attention. If (-1, 0), implements causal attention.
        alpha: float. Scaling factor between add rab and silu.
        rab: (batch_size, max_seqlen_k, max_seqlen_k). Random access bias for the key.
        has_drab: bool. Whether to apply random access bias for the key.
        func: (nheads, total_q + 256). Function to describe the mask shape in arbitrary mask.
    Return:
        out: (total, nheads, headdim).
    """
    if has_drab and (rab is None):
        raise ValueError("AssertError: rab is None, but has_drab is True, is not allowed in backward")
    if num_contexts != None and window_size != (-1, 0):
        raise ValueError("AssertError: context is True and causal is not True, this is undefined behavior")
    if num_targets != None and window_size != (-1, 0):
        raise ValueError("AssertError: target is True and causal is not True, this is undefined behavior")
    if num_targets is None and target_group_size < 1:
        raise ValueError("AssertError: target_group_size should be greater than 0 when target is True")
    if max_seqlen_q > max_seqlen_k:
        raise ValueError("AssertError: seq_len_q >= seq_len_k, this is undefined behavior")

    return HstuAttnQKVPackedFunc.apply(
        qkv,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        scaling_seqlen,
        num_contexts,
        num_targets,
        target_group_size,
        window_size,
        alpha,
        rab,
        has_drab,
        func,
    )


# api for hstu attention when rab and delta_q are not used
@torch.fx.wrap
def cuda_hstu_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    scaling_seqlen: int,
    num_targets: torch.Tensor,
    window_size: Tuple[int, int] = (-1, -1),
    alpha: float = 1.0,
    is_train: bool = True,
) -> torch.Tensor:
    if is_train:
        out = hstu_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            scaling_seqlen,
            None,  # num_contexts, # pyre-ignore[6]
            num_targets,
            1,  # target_group_size
            window_size,
            alpha,
        )

    else:
        major_version = torch.cuda.get_device_capability()[0]
        assert major_version == 8 or major_version == 9, "Only support sm80 and sm90"
        if major_version == 8:
            out, _ = torch.ops.fbgemm.hstu_varlen_fwd_80(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                scaling_seqlen,
                max_seqlen_q,
                max_seqlen_k,
                None,  # num_contexts,
                num_targets,
                1,  # target_group_size
                window_size[0],
                window_size[1],
                alpha,
            )
        else:
            out, _ = torch.ops.fbgemm.hstu_varlen_fwd_90(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                scaling_seqlen,
                None, # num_contexts,
                num_targets,
                1,
                window_size[0],
                window_size[1],
                alpha,
                None, # rab
                None, # func
                -1, # quant_mode
                0 if q.dtype == torch.bfloat16 else 1,
            )
        return out
    return out
