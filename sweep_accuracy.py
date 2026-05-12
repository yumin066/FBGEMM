#!/usr/bin/env python3
"""
Sweep H and SEQ (powers of 2) to find which config breaks FP8 accuracy.
No kernel debug prints needed — just measure cos_sim for each (H, SEQ) combo.
"""
import os
import random
import torch, sys
sys.path.insert(0, "/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu")
sys.path.insert(0, "/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu/fbgemm_gpu/experimental/hstu/test")
import hstu  # noqa
from hstu.cuda_hstu_attention import (
    get_bm_and_bn_block_size_fwd,
    quantize_for_block_scale_qk_along_d,
    quantize_for_block_scale_v_along_n,
    quantize_paged_kv_cache_for_block_scale,
    pack_descale_to_e8m0x4_int32,
)
from hstu_test import generate_paged_kv_input, _hstu_paged_kv_attention

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # Make library-side behavior as deterministic as possible.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
DEVICE = "cuda"
BS = 1
ALPHA = 1.0
SWEEP_DIMS = [32, 64, 128, 256]
SWEEP_HEADS = [1, 4]
SWEEP_SEQS = [128, 256, 512]

# Baseline defaults to avoid branch interference in debugging runs.
# Users can still override from shell when explicitly needed.
os.environ.setdefault("HSTU_EXP_A_SYNC_K", "0")
os.environ.setdefault("HSTU_EXP_B_PBUF_IN_SK", "0")

EXP_A = os.getenv("HSTU_EXP_A_SYNC_K", "0")
EXP_B = os.getenv("HSTU_EXP_B_PBUF_IN_SK", "0")
RUN_PAGED_KV_SWEEP = os.getenv("HSTU_SKIP_PAGED_KV_SWEEP", "0") != "1"
print(f"[sweep config] HSTU_EXP_A_SYNC_K={EXP_A} HSTU_EXP_B_PBUF_IN_SK={EXP_B}")

def run_attn(q, k, v, cu_seqlens, max_seqlen, num_targets, scaling_seqlen, quant_mode,
             q_descale=None, k_descale=None, v_descale=None,
             sf_q=None, sf_k=None, sf_v=None, cu_q_blk=None, cu_kv_blk=None, cu_v_blk=None):
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        q, k, v, cu_seqlens, cu_seqlens, None, None,
        max_seqlen, max_seqlen, scaling_seqlen, None, num_targets, 1,
        -1, -1, ALPHA, None, None, quant_mode, q_descale, k_descale, v_descale,
        sf_q, sf_k, sf_v,
        cu_q_blk, cu_kv_blk, cu_v_blk)
    return out


def fullpath_ground_truth(q_like, k_like, v_like, alpha, scaling_seqlen):
    """
    Full-path reference for current sweep setup:
      S = silu(alpha * (Q @ K^T))    (no causal/local/arbitrary mask here)
      O = (S @ V) / scaling_seqlen
    Shapes:
      Q,K,V: [seq, heads, dim]
      O:     [seq, heads, dim]
    """
    qf = q_like.to(torch.bfloat16)
    kf = k_like.to(torch.bfloat16)
    vf = v_like.to(torch.bfloat16)
    s = torch.einsum("mhd,nhd->mhn", qf, kf).to(torch.bfloat16)
    s = torch.nn.functional.silu(s * torch.tensor(alpha, device=s.device, dtype=torch.bfloat16)).to(torch.bfloat16)
    o = (torch.einsum("mhn,nhd->mhd", s, vf) /
         torch.tensor(float(scaling_seqlen), device=s.device, dtype=torch.bfloat16)).to(torch.bfloat16)
    return o


def e8m0_dequant(fp8_tensor: torch.Tensor, descale: torch.Tensor,
                 total_tokens: int) -> torch.Tensor:
    """
    Dequantize FP8 tensor using e8m0-rounded block scales.

    The kernel stores scales as e8m0 (= ceil(log2(scale)) rounded up to nearest
    power-of-2).  This function reproduces the same rounding so that the Python
    ground-truth uses exactly the same effective scale as the hardware.

    The block_size (tokens per scale) is inferred from descale.shape[1]:
      block_size = ceil(total_tokens / n_blocks)
    This handles both per-token (Q/K: n_blocks=SEQ, block_size=1) and
    per-tile (V: n_blocks=SEQ//128, block_size=128) layouts automatically.

    Args:
        fp8_tensor : [total_tokens, H, D]  (any dtype, cast to float internally)
        descale    : [H, n_blocks]  raw (un-rounded) scale factors produced by
                     quantize_for_block_scale_*
        total_tokens: number of tokens in the sequence (= fp8_tensor.shape[0])
    Returns:
        BF16 tensor [total_tokens, H, D] dequantized with e8m0-rounded scales
    """
    if descale.dim() == 3:
        H, n_blocks, d_chunks = descale.shape
        block_size = (total_tokens + n_blocks - 1) // n_blocks
        exp = torch.ceil(torch.log2(descale.float().clamp(min=1e-38)))
        scale_eff = torch.pow(2.0, exp)  # [H, n_blocks, D/128]
        out = fp8_tensor.float()
        out_view = out.view(total_tokens, H, d_chunks, 128)
        for blk in range(n_blocks):
            start = blk * block_size
            end = min(start + block_size, total_tokens)
            if start >= total_tokens:
                break
            out_view[start:end] = out_view[start:end] * scale_eff[:, blk, :].view(1, H, d_chunks, 1)
        return out.to(torch.bfloat16)

    H, n_blocks = descale.shape
    block_size = (total_tokens + n_blocks - 1) // n_blocks  # auto-infer
    # Round to nearest power-of-2 (e8m0 ceiling: scale_eff = 2^ceil(log2(scale)))
    exp = torch.ceil(torch.log2(descale.float().clamp(min=1e-38)))
    scale_eff = torch.pow(2.0, exp)  # [H, n_blocks]

    out = fp8_tensor.float()  # [total_tokens, H, D]
    for blk in range(n_blocks):
        start = blk * block_size
        end = min(start + block_size, total_tokens)
        if start >= total_tokens:
            break
        # scale_eff[:, blk] shape [H] → broadcast over [end-start, H, D]
        out[start:end] = out[start:end] * scale_eff[:, blk].view(1, H, 1)
    return out.to(torch.bfloat16)


def dequantize_paged_kv_cache(kv_cache_raw: torch.Tensor,
                              kv_cache_fp8: torch.Tensor,
                              block_size: int) -> torch.Tensor:
    pages, _, page_size, heads, dim = kv_cache_raw.shape
    assert page_size == block_size
    fp8_max = 448.0

    k_raw = kv_cache_raw[:, 0].contiguous()
    v_raw = kv_cache_raw[:, 1].contiguous()
    k_fp8 = kv_cache_fp8[:, 0].float()
    v_fp8 = kv_cache_fp8[:, 1].float()

    d_chunks = (dim + 127) // 128
    k_raw_flat = k_raw.view(pages * page_size, heads, dim)
    k_fp8_flat = k_fp8.view(pages * page_size, heads, dim)
    k_deq_flat = torch.empty_like(k_raw_flat, dtype=torch.float32)
    for d_chunk in range(d_chunks):
        d0 = d_chunk * 128
        d1 = min(dim, d0 + 128)
        k_scale = torch.amax(k_raw_flat[:, :, d0:d1].abs(), dim=2).to(torch.float32) / fp8_max
        k_scale = torch.pow(2.0, torch.ceil(torch.log2(k_scale.clamp(min=1e-38))))
        k_deq_flat[:, :, d0:d1] = k_fp8_flat[:, :, d0:d1] * k_scale.unsqueeze(-1)
    k_deq = k_deq_flat.view(pages, page_size, heads, dim)

    v_scale = torch.amax(v_raw.abs(), dim=(1, 3)).to(torch.float32) / fp8_max
    v_scale = torch.pow(2.0, torch.ceil(torch.log2(v_scale.clamp(min=1e-38))))
    v_deq = v_fp8 * v_scale[:, None, :, None]

    kv_cache_deq = torch.empty_like(kv_cache_raw, dtype=torch.bfloat16)
    kv_cache_deq[:, 0] = k_deq.to(torch.bfloat16)
    kv_cache_deq[:, 1] = v_deq.to(torch.bfloat16)
    return kv_cache_deq


def quantize_paged_inputs_for_block_scale(q, k, v, kv_cache, cu, block_size):
    q_fp8, q_descale, cu_q_blk = quantize_for_block_scale_qk_along_d(
        q, cu, fp8_type=torch.float8_e4m3fn)
    k_fp8, k_descale, cu_k_blk = quantize_for_block_scale_qk_along_d(
        k, cu, fp8_type=torch.float8_e4m3fn)
    v_fp8, v_descale, cu_v_blk = quantize_for_block_scale_v_along_n(
        v, cu, block_size=block_size, fp8_type=torch.float8_e4m3fn)

    sf_q = pack_descale_to_e8m0x4_int32(q_descale)
    sf_k = pack_descale_to_e8m0x4_int32(k_descale)
    sf_v = pack_descale_to_e8m0x4_int32(v_descale).repeat_interleave(block_size, dim=1)

    kv_cache_fp8, sf_k_cache, sf_v_cache = quantize_paged_kv_cache_for_block_scale(
        kv_cache, block_size=block_size, fp8_type=torch.float8_e4m3fn)
    sf_k = torch.cat([sf_k_cache, sf_k], dim=1).contiguous()
    sf_v = torch.cat([sf_v_cache, sf_v], dim=1).contiguous()

    q_deq = e8m0_dequant(q_fp8, q_descale, q.shape[0])
    k_deq = e8m0_dequant(k_fp8, k_descale, k.shape[0])
    v_deq = e8m0_dequant(v_fp8, v_descale, v.shape[0])
    kv_cache_deq = dequantize_paged_kv_cache(kv_cache, kv_cache_fp8, block_size)

    return {
        "q_fp8": q_fp8, "k_fp8": k_fp8, "v_fp8": v_fp8,
        "kv_cache_fp8": kv_cache_fp8,
        "q_descale": q_descale, "k_descale": k_descale, "v_descale": v_descale,
        "sf_q": sf_q, "sf_k": sf_k, "sf_v": sf_v,
        "cu_q_blk": cu_q_blk, "cu_k_blk": cu_k_blk, "cu_v_blk": cu_v_blk,
        "q_deq": q_deq, "k_deq": k_deq, "v_deq": v_deq,
        "kv_cache_deq": kv_cache_deq,
    }


def quantize_nonpaged_inputs_for_block_scale(q, k, v, cu, block_size):
    q_fp8, q_descale, cu_q_blk = quantize_for_block_scale_qk_along_d(
        q, cu, fp8_type=torch.float8_e4m3fn)
    k_fp8, k_descale, cu_k_blk = quantize_for_block_scale_qk_along_d(
        k, cu, fp8_type=torch.float8_e4m3fn)
    v_fp8, v_descale, cu_v_blk = quantize_for_block_scale_v_along_n(
        v, cu, block_size=block_size, fp8_type=torch.float8_e4m3fn)

    return {
        "q_fp8": q_fp8, "k_fp8": k_fp8, "v_fp8": v_fp8,
        "q_descale": q_descale, "k_descale": k_descale, "v_descale": v_descale,
        "sf_q": pack_descale_to_e8m0x4_int32(q_descale),
        "sf_k": pack_descale_to_e8m0x4_int32(k_descale),
        "sf_v": pack_descale_to_e8m0x4_int32(v_descale).repeat_interleave(block_size, dim=1),
        "cu_q_blk": cu_q_blk, "cu_k_blk": cu_k_blk, "cu_v_blk": cu_v_blk,
    }


def run_nonpaged_attn_quantized(tensors, cu, seqlen, num_targets, window_size):
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        tensors["q_fp8"], tensors["k_fp8"], tensors["v_fp8"],
        cu, cu, None, None,
        seqlen, seqlen, -1,
        None, num_targets, 1,
        window_size[0], window_size[1],
        ALPHA,
        None, None,
        2,
        tensors["q_descale"], tensors["k_descale"], tensors["v_descale"],
        tensors["sf_q"], tensors["sf_k"], tensors["sf_v"],
        tensors["cu_q_blk"], tensors["cu_k_blk"], tensors["cu_v_blk"],
    )
    return out


def run_paged_attn_quantized(
    tensors,
    cu_q,
    cu_k,
    max_seqlen_q,
    max_seqlen_k,
    num_targets,
    window_size,
    page_offsets,
    page_ids,
    last_page_lens,
):
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        tensors["q_fp8"], tensors["k_fp8"], tensors["v_fp8"],
        cu_q, cu_k, None, None,
        max_seqlen_q, max_seqlen_k, -1,
        None, num_targets, 1,
        window_size[0], window_size[1],
        ALPHA,
        None, None,
        2,
        tensors["q_descale"], tensors["k_descale"], tensors["v_descale"],
        tensors["sf_q"], tensors["sf_k"], tensors["sf_v"],
        tensors["cu_q_blk"], tensors["cu_k_blk"], tensors["cu_v_blk"],
        tensors["kv_cache_fp8"], page_offsets, page_ids, last_page_lens,
    )
    return out


def make_paged_cache_from_contiguous_kv(k, v, batch_size, seqlen, page_size, page_ids):
    total_pages = int(page_ids.numel())
    heads = k.shape[1]
    dim = k.shape[2]
    kv_cache = torch.empty(
        (total_pages, 2, page_size, heads, dim),
        dtype=k.dtype,
        device=k.device,
    )
    pages_per_batch = seqlen // page_size
    for b in range(batch_size):
        for p in range(pages_per_batch):
            logical_page = b * pages_per_batch + p
            page_id = int(page_ids[logical_page].item())
            row0 = b * seqlen + p * page_size
            row1 = row0 + page_size
            kv_cache[page_id, 0] = k[row0:row1]
            kv_cache[page_id, 1] = v[row0:row1]
    return kv_cache


def compute_paged_nonpaged_same_input_metrics(batch_size, heads, seqlen, mode, dim):
    D = dim
    PAGE_SIZE = 64
    assert seqlen % PAGE_SIZE == 0
    seed = SEED + 9000 + batch_size * 100 + heads * 10 + seqlen + (1 if mode == "causal" else 0)
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)

    total_q = batch_size * seqlen
    q = torch.randn((total_q, heads, D), generator=g, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn((total_q, heads, D), generator=g, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn((total_q, heads, D), generator=g, device=DEVICE, dtype=torch.bfloat16)
    q = q.to(torch.float8_e4m3fn).to(torch.bfloat16)
    k = k.to(torch.float8_e4m3fn).to(torch.bfloat16)
    v = v.to(torch.float8_e4m3fn).to(torch.bfloat16)

    cu = torch.arange(0, total_q + 1, seqlen, dtype=torch.int32, device=DEVICE)
    pages_per_batch = seqlen // PAGE_SIZE
    total_pages = batch_size * pages_per_batch
    page_offsets = torch.arange(
        0, total_pages + 1, pages_per_batch, dtype=torch.int32, device=DEVICE)
    page_ids = torch.randperm(total_pages, generator=g, dtype=torch.int32, device=DEVICE)
    last_page_lens = torch.full((batch_size,), PAGE_SIZE, dtype=torch.int32, device=DEVICE)
    kv_cache = make_paged_cache_from_contiguous_kv(
        k, v, batch_size, seqlen, PAGE_SIZE, page_ids)

    window_size = (-1, 0) if mode == "causal" else (-1, -1)
    num_targets = (
        torch.zeros(batch_size, dtype=torch.int32, device=DEVICE)
        if mode == "causal"
        else None
    )
    nonpaged = quantize_nonpaged_inputs_for_block_scale(q, k, v, cu, PAGE_SIZE)
    paged = quantize_paged_inputs_for_block_scale(q, k, v, kv_cache, cu, PAGE_SIZE)

    out_nonpaged = run_nonpaged_attn_quantized(
        nonpaged, cu, seqlen, num_targets, window_size)
    out_paged = run_paged_attn_quantized(
        paged, cu, cu, seqlen, seqlen, num_targets, window_size,
        page_offsets, page_ids, last_page_lens)
    torch.cuda.synchronize()

    cos, max_err, mean_err = metric_report(out_paged.flatten(), out_nonpaged.flatten())
    _, _, _, rel_l2 = norm_report(out_paged.flatten(), out_nonpaged.flatten())
    return cos, max_err, mean_err, rel_l2


def sum_report(a: torch.Tensor, b: torch.Tensor):
    sa = float(a.float().sum().item())
    sb = float(b.float().sum().item())
    return sa, sb, abs(sa - sb)


def metric_report(a: torch.Tensor, b: torch.Tensor):
    diff = (a.float() - b.float()).abs()
    max_err = float(diff.max().item())
    mean_err = float(diff.mean().item())
    cos = float(torch.nn.functional.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item())
    return cos, max_err, mean_err

def norm_report(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12):
    af = a.float()
    bf = b.float()
    diff = af - bf
    na = float(torch.linalg.vector_norm(af).item())
    nb = float(torch.linalg.vector_norm(bf).item())
    nd = float(torch.linalg.vector_norm(diff).item())
    rel = nd / (na + eps)
    return na, nb, nd, rel


def fmt_metric(value, width: int, precision: int = 4) -> str:
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value:>{width}.{precision}f}"


def stage_error_probe(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      alpha: float, scaling_seqlen: int):
    """
    Stage-wise diagnostic (math-level emulation):
      - GEMM1 stage compares S_bf16 vs S_fp8path
      - GEMM2 stage compares O_bf16path vs O_fp8path
    This helps decide whether major divergence appears before or after GEMM2.
    """
    q_b = q.to(torch.bfloat16)
    k_b = k.to(torch.bfloat16)
    v_b = v.to(torch.bfloat16)
    q_f8 = q.to(torch.float8_e4m3fn)
    k_f8 = k.to(torch.float8_e4m3fn)
    v_f8 = v.to(torch.float8_e4m3fn)

    # GEMM1 + activation
    s_b = torch.einsum("mhd,nhd->mhn", q_b, k_b).to(torch.bfloat16)
    s_b = torch.nn.functional.silu(
        s_b * torch.tensor(alpha, device=s_b.device, dtype=torch.bfloat16)
    ).to(torch.bfloat16)
    s_f = torch.einsum("mhd,nhd->mhn", q_f8.float(), k_f8.float())
    s_f = torch.nn.functional.silu(s_f * float(alpha))

    # GEMM2 path operands (matching kernel intent: BF16 path uses bf16-P, FP8 path uses fp8-P)
    p_b = s_b
    p_f8 = s_f.to(torch.float8_e4m3fn)
    o_b = torch.einsum("mhn,nhd->mhd", p_b.float(), v_b.float()) / float(scaling_seqlen)
    o_f = torch.einsum("mhn,nhd->mhd", p_f8.float(), v_f8.float()) / float(scaling_seqlen)

    s_cos, s_max, s_mean = metric_report(s_b.flatten(), s_f.flatten())
    _, _, _, s_rel_l2 = norm_report(s_b.flatten(), s_f.flatten())
    o_cos, o_max, o_mean = metric_report(o_b.flatten(), o_f.flatten())
    _, _, _, o_rel_l2 = norm_report(o_b.flatten(), o_f.flatten())
    return (s_cos, s_max, s_mean, s_rel_l2), (o_cos, o_max, o_mean, o_rel_l2)


def compute_paged_kv_metrics(batch_size, heads, new_history_len, prev_history_len,
                             target_len, dim=128, dtype=torch.float16):
    D = dim
    PAGE_SIZE = 64
    torch.manual_seed(SEED + batch_size * 100 + heads * 10 + new_history_len + prev_history_len + target_len + dim)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + batch_size * 100 + heads * 10 + new_history_len + prev_history_len + target_len + dim)

    (
        _,
        cu_seqlens_q,
        cu_seqlens_k,
        num_targets,
        page_offsets,
        page_ids,
        last_page_lens,
        q,
        k,
        v,
        kv_cache,
        mask,
    ) = generate_paged_kv_input(
        batch_size=batch_size,
        heads=heads,
        max_seq_len_q=new_history_len,
        max_seq_len_k=prev_history_len,
        max_target_len=target_len,
        attn_dim=D,
        hidden_dim=D,
        page_size=PAGE_SIZE,
        dtype=dtype,
        full_batch=True,
    )

    max_seqlen_q = new_history_len + target_len
    max_seqlen_k = new_history_len + prev_history_len + target_len

    tensors = quantize_paged_inputs_for_block_scale(
        q, k, v, kv_cache, cu_seqlens_q, PAGE_SIZE)
    ref = _hstu_paged_kv_attention(
        num_heads=heads,
        attention_dim=D,
        linear_dim=D,
        seqlen_q=max_seqlen_q,
        seqlen_k=max_seqlen_k,
        q=tensors["q_deq"],
        k=tensors["k_deq"],
        v=tensors["v_deq"],
        q_offsets=cu_seqlens_q,
        k_offsets=cu_seqlens_k,
        num_targets=num_targets,
        invalid_attn_mask=mask,
        alpha=ALPHA,
        upcast=True,
        kv_cache=tensors["kv_cache_deq"],
        page_offsets=page_offsets,
        page_ids=page_ids,
        last_page_lens=last_page_lens,
    )

    out = run_paged_attn_quantized(
        tensors,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_targets,
        (-1, 0),
        page_offsets,
        page_ids,
        last_page_lens,
    )
    torch.cuda.synchronize()

    cos, max_err, mean_err = metric_report(out.flatten(), ref.flatten())
    _, _, _, rel_l2 = norm_report(out.flatten(), ref.flatten())
    if cos < 0.995:
        raise AssertionError(f"paged KV fp8_gt_cos={cos:.6f} < 0.995")
    return cos, max_err, mean_err, rel_l2, last_page_lens.detach().cpu().tolist()


def make_full_paged_kv_input(batch_size, heads, seqlen, dim=128, dtype=torch.float16):
    D = dim
    PAGE_SIZE = 64
    pages_per_batch = (seqlen + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch_size * pages_per_batch
    total_q = batch_size * seqlen

    q = torch.empty((total_q, heads, D), dtype=dtype, device=DEVICE).uniform_(-1, 1)
    # Full paged-KV reads K/V from kv_cache only.  K/V tensors are schema
    # placeholders used by the Python quantization wrapper for metadata.
    k = torch.empty_like(q).uniform_(-1, 1)
    v = torch.empty_like(q).uniform_(-1, 1)
    kv_cache = torch.empty(
        (total_pages, 2, PAGE_SIZE, heads, D),
        dtype=dtype,
        device=DEVICE,
    ).uniform_(-1, 1)

    cu = torch.arange(
        0, total_q + 1, seqlen, dtype=torch.int32, device=DEVICE
    )
    page_offsets = torch.arange(
        0, total_pages + 1, pages_per_batch, dtype=torch.int32, device=DEVICE
    )
    page_ids = torch.randperm(total_pages, dtype=torch.int32, device=DEVICE)
    last_len = seqlen - (pages_per_batch - 1) * PAGE_SIZE
    last_page_lens = torch.full(
        (batch_size,), last_len, dtype=torch.int32, device=DEVICE
    )
    return q, k, v, kv_cache, cu, page_offsets, page_ids, last_page_lens


def reconstruct_full_paged_kv(kv_cache, page_offsets, page_ids, last_page_lens, batch_idx):
    k_chunks = []
    v_chunks = []
    start = int(page_offsets[batch_idx].item())
    end = int(page_offsets[batch_idx + 1].item())
    for page_pos in range(start, end):
        page_id = int(page_ids[page_pos].item())
        valid = kv_cache.shape[2]
        if page_pos == end - 1:
            valid = int(last_page_lens[batch_idx].item())
        k_chunks.append(kv_cache[page_id, 0, :valid])
        v_chunks.append(kv_cache[page_id, 1, :valid])
    return torch.cat(k_chunks, dim=0), torch.cat(v_chunks, dim=0)


def full_paged_ground_truth(q, kv_cache, cu, page_offsets, page_ids,
                            last_page_lens, alpha, scaling_seqlen):
    out = torch.empty_like(q)
    batch_size = cu.numel() - 1
    for b in range(batch_size):
        q0 = int(cu[b].item())
        q1 = int(cu[b + 1].item())
        q_b = q[q0:q1].float()
        k_b, v_b = reconstruct_full_paged_kv(
            kv_cache, page_offsets, page_ids, last_page_lens, b)
        s = torch.einsum("mhd,nhd->mhn", q_b, k_b.float())
        s = torch.nn.functional.silu(s * float(alpha))
        o = torch.einsum("mhn,nhd->mhd", s, v_b.float()) / float(scaling_seqlen)
        out[q0:q1] = o.to(out.dtype)
    return out


def compute_full_paged_kv_metrics(batch_size, heads, seqlen, dim=128, dtype=torch.float16):
    torch.manual_seed(SEED + 7000 + batch_size * 100 + heads * 10 + seqlen + dim)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + 7000 + batch_size * 100 + heads * 10 + seqlen + dim)

    q, k, v, kv_cache, cu, page_offsets, page_ids, last_page_lens = (
        make_full_paged_kv_input(batch_size, heads, seqlen, dim=dim, dtype=dtype)
    )
    tensors = quantize_paged_inputs_for_block_scale(q, k, v, kv_cache, cu, 64)
    ref = full_paged_ground_truth(
        tensors["q_deq"], tensors["kv_cache_deq"],
        cu, page_offsets, page_ids, last_page_lens, ALPHA, seqlen)
    out = run_paged_attn_quantized(
        tensors,
        cu,
        cu,
        seqlen,
        seqlen,
        None,
        (-1, -1),
        page_offsets,
        page_ids,
        last_page_lens,
    )
    torch.cuda.synchronize()

    cos, max_err, mean_err = metric_report(out.flatten(), ref.flatten())
    _, _, _, rel_l2 = norm_report(out.flatten(), ref.flatten())
    if cos < 0.995:
        raise AssertionError(f"full paged KV fp8_gt_cos={cos:.6f} < 0.995")
    return cos, max_err, mean_err, rel_l2, last_page_lens.detach().cpu().tolist()

print(
    f"{'D':>4} {'H':>4} {'SEQ':>6} | "
    f"{'bf16_fp8_cos':>12} {'max_err':>10} {'mean_err':>10} | "
    f"{'||bf16||':>10} {'||fp8||':>10} {'rel_l2':>10} | "
    f"{'bf16_gt_cos':>11} {'bf16_gt_max':>12} {'bf16_gt_mean':>12} | "
    f"{'fp8_gt_cos':>10} {'fp8_gt_max':>11} {'fp8_gt_mean':>11} | "
    f"{'paged_c_cos':>12} {'paged_c_max':>12} {'paged_c_mean':>13} | "
    f"{'paged_f_cos':>12} {'paged_f_max':>12} {'paged_f_mean':>13}"
)
print("-" * 256)
if RUN_PAGED_KV_SWEEP:
    print(
        "[paged kv] paged_f mirrors every non-paged full sweep row; "
        "appended columns compare against dequantized FP8/e8m0 references"
    )

USE_ONES = os.getenv("HSTU_USE_ONES", "0") == "1"
if USE_ONES:
    print("[input mode] Q=K=V=ones (all 1.0)")

for D in SWEEP_DIMS:
    _, fp8_bn = get_bm_and_bn_block_size_fwd(None, D)
    for H in SWEEP_HEADS:
        for SEQ in SWEEP_SEQS:
            g = torch.Generator(device=DEVICE)
            g.manual_seed(SEED)
            if USE_ONES:
                q = torch.ones((BS * SEQ, H, D), device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn).to(torch.bfloat16)
                k = torch.ones((BS * SEQ, H, D), device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn).to(torch.bfloat16)
                v = torch.ones((BS * SEQ, H, D), device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn).to(torch.bfloat16)
            else:
                q = torch.randn((BS * SEQ, H, D), generator=g, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn).to(torch.bfloat16)
                k = torch.randn((BS * SEQ, H, D), generator=g, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn).to(torch.bfloat16)
                v = torch.randn((BS * SEQ, H, D), generator=g, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn).to(torch.bfloat16)
            cu = torch.tensor([0, SEQ], dtype=torch.int32, device=DEVICE)
            num_tgts = torch.zeros(BS, dtype=torch.int32, device=DEVICE)

            try:
                out_bf16 = run_attn(q, k, v, cu, SEQ, num_tgts, SEQ, -1)
                torch.cuda.synchronize()
            except Exception as e:
                out_bf16 = None
                bf16_error = str(e)
            else:
                bf16_error = None

            try:
                # blockwise-scale fp8 (quant_mode=2), real per-block scales.
                # q/k are quantized along D (1 scale per token for D=128).
                # v is quantized along N (1 scale per 128-row block).
                q_raw = q  # already FP8-rounded BF16
                k_raw = k
                v_raw = v
                q_fp8, q_descale, cu_q_blk = quantize_for_block_scale_qk_along_d(
                    q_raw, cu, fp8_type=torch.float8_e4m3fn)
                k_fp8, k_descale, cu_kv_blk = quantize_for_block_scale_qk_along_d(
                    k_raw, cu, fp8_type=torch.float8_e4m3fn)
                v_fp8, v_descale, cu_v_blk = quantize_for_block_scale_v_along_n(
                    v_raw, cu, block_size=fp8_bn, fp8_type=torch.float8_e4m3fn)
                sf_q = pack_descale_to_e8m0x4_int32(q_descale)
                sf_k = pack_descale_to_e8m0x4_int32(k_descale)
                # TMA SFV requires sf_v expanded from [H, total_blocks] to [H, total_tokens]
                # so that each block's scale repeats kBlockN times (one copy per token).
                sf_v = pack_descale_to_e8m0x4_int32(v_descale).repeat_interleave(fp8_bn, dim=1)
                # q_descale/k_descale: [H, SEQ]  (per-token, D-direction quant)
                # v_descale:           [H, SEQ//128]  (per-tile, N-direction quant)
                # cu_kv_blk: K's per-token cu_seqlens (offset in tokens)
                # cu_v_blk:  V's per-block cu_seqlens (offset in 128-token blocks)
                out_fp8 = run_attn(q_fp8, k_fp8, v_fp8, cu, SEQ, num_tgts, SEQ, 2,
                                   q_descale, k_descale, v_descale,
                                   sf_q, sf_k, sf_v, cu_q_blk, cu_kv_blk, cu_v_blk)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"{D:>4} {H:>4} {SEQ:>6} | FP8 ERROR: {e}")
                import traceback; traceback.print_exc()
                continue

            o_f = out_fp8.float().flatten()
            if out_bf16 is not None:
                o_b = out_bf16.float().flatten()
                cos, me, mn = metric_report(o_b, o_f)
                n_b, n_f, _, rel_l2 = norm_report(o_b, o_f)
            else:
                cos = me = mn = n_b = rel_l2 = None
                n_f = float(torch.linalg.vector_norm(o_f.float()).item())

            # Dequantized ground truth for FP8: reproduces exactly the e8m0-rounded
            # scales that the kernel uses, so cross-head scale non-uniformity cancels out.
            q_deq = e8m0_dequant(q_fp8, q_descale, BS * SEQ)
            k_deq = e8m0_dequant(k_fp8, k_descale, BS * SEQ)
            v_deq = e8m0_dequant(v_fp8, v_descale, BS * SEQ)
            gt_deq = fullpath_ground_truth(q_deq, k_deq, v_deq, ALPHA, SEQ)
            gt_deq_flat = gt_deq.float().flatten()
            cos_f_gt_pre, _, _ = metric_report(o_f, gt_deq_flat)
            flag = " <<<" if cos_f_gt_pre < 0.95 else ""

            gt = fullpath_ground_truth(q, k, v, ALPHA, SEQ).float().flatten()
            if out_bf16 is not None:
                cos_b_gt, me_b_gt, mn_b_gt = metric_report(o_b, gt)
            else:
                cos_b_gt = me_b_gt = mn_b_gt = None
            # FP8 kernel vs dequantized ground truth (correct comparison)
            cos_f_gt, me_f_gt, mn_f_gt = metric_report(o_f, gt_deq_flat)
            if RUN_PAGED_KV_SWEEP:
                paged_c_cos, paged_c_max, paged_c_mean, _, _ = compute_paged_kv_metrics(
                    batch_size=BS,
                    heads=H,
                    new_history_len=SEQ,
                    prev_history_len=0,
                    target_len=0,
                    dim=D,
                )
                paged_f_cos, paged_f_max, paged_f_mean, _, _ = compute_full_paged_kv_metrics(
                    batch_size=BS,
                    heads=H,
                    seqlen=SEQ,
                    dim=D,
                )
                paged_cols = (
                    f" | {paged_c_cos:>12.4f} {paged_c_max:>12.6f} {paged_c_mean:>13.6f}"
                    f" | {paged_f_cos:>12.4f} {paged_f_max:>12.6f} {paged_f_mean:>13.6f}"
                )
            else:
                paged_cols = (
                    f" | {'N/A':>12} {'N/A':>12} {'N/A':>13}"
                    f" | {'N/A':>12} {'N/A':>12} {'N/A':>13}"
                )
            print(
                f"{D:>4} {H:>4} {SEQ:>6} | "
                f"{fmt_metric(cos, 12, 4)} {fmt_metric(me, 10, 6)} {fmt_metric(mn, 10, 6)} | "
                f"{fmt_metric(n_b, 10, 4)} {fmt_metric(n_f, 10, 4)} {fmt_metric(rel_l2, 10, 6)} | "
                f"{fmt_metric(cos_b_gt, 11, 4)} {fmt_metric(me_b_gt, 12, 6)} {fmt_metric(mn_b_gt, 12, 6)} | "
                f"{cos_f_gt:>10.4f} {me_f_gt:>11.6f} {mn_f_gt:>11.6f}{paged_cols}{flag}"
            )
            if bf16_error is not None:
                print(
                    f"{'':>18}   [bf16 unsupported for this row: {bf16_error}]"
                )
        sys.stdout.flush()


def run_same_input_paged_vs_nonpaged():
    print("\n[paged vs non-paged same input] same raw Q/K/V, same FP8 block scales")
    print(
        f"{'D':>4} {'H':>4} {'SEQ':>6} {'MODE':>8} | "
        f"{'cos':>10} {'max_err':>12} {'mean_err':>12} {'rel_l2':>10}"
    )
    print("-" * 78)
    for D in SWEEP_DIMS:
        for H in SWEEP_HEADS:
            for SEQ in SWEEP_SEQS:
                for mode in ("full", "causal"):
                    cos, max_err, mean_err, rel_l2 = compute_paged_nonpaged_same_input_metrics(
                        batch_size=BS,
                        heads=H,
                        seqlen=SEQ,
                        mode=mode,
                        dim=D,
                    )
                    if cos < 0.999:
                        raise AssertionError(
                            f"same-input paged vs non-paged D={D} {mode} cos={cos:.6f} < 0.999"
                        )
                    print(
                        f"{D:>4} {H:>4} {SEQ:>6} {mode:>8} | "
                        f"{cos:>10.6f} {max_err:>12.6f} {mean_err:>12.6f} {rel_l2:>10.6f}"
                    )


if RUN_PAGED_KV_SWEEP:
    run_same_input_paged_vs_nonpaged()


def run_paged_kv_case(label, batch_size, heads, new_history_len, prev_history_len,
                      target_len, dim=128, dtype=torch.float16):
    cos, max_err, mean_err, rel_l2, last_page_lens = compute_paged_kv_metrics(
        batch_size=batch_size,
        heads=heads,
        new_history_len=new_history_len,
        prev_history_len=prev_history_len,
        target_len=target_len,
        dim=dim,
        dtype=dtype,
    )
    print(
        f"{label:>22} | D={dim:<3} B={batch_size:<2} H={heads:<2} "
        f"new={new_history_len:<3} prev={prev_history_len:<3} tgt={target_len:<3} | "
        f"cos={cos:.6f} max={max_err:.6f} mean={mean_err:.6f} rel_l2={rel_l2:.6f} "
        f"last_page={last_page_lens}"
    )


def run_full_paged_kv_case(label, batch_size, heads, seqlen, dim=128, dtype=torch.float16):
    cos, max_err, mean_err, rel_l2, last_page_lens = compute_full_paged_kv_metrics(
        batch_size=batch_size,
        heads=heads,
        seqlen=seqlen,
        dim=dim,
        dtype=dtype,
    )
    print(
        f"{label:>22} | D={dim:<3} B={batch_size:<2} H={heads:<2} "
        f"seq={seqlen:<3} full-paged | "
        f"cos={cos:.6f} max={max_err:.6f} mean={mean_err:.6f} rel_l2={rel_l2:.6f} "
        f"last_page={last_page_lens}"
    )


if RUN_PAGED_KV_SWEEP and os.getenv("HSTU_PAGED_KV_EDGE_CASES", "0") == "1":
    print("\n[paged kv edge cases] SM120 FP8 quant_mode=2")
    print(
        f"{'case':>22} | {'shape':>24} | "
        f"{'metrics':>55}"
    )
    print("-" * 112)
    run_paged_kv_case("edge_partial_page", batch_size=1, heads=2,
                      new_history_len=64, prev_history_len=32, target_len=64)
    run_paged_kv_case("edge_target_zero", batch_size=1, heads=1,
                      new_history_len=128, prev_history_len=64, target_len=0)
    run_full_paged_kv_case("edge_full_paged", batch_size=1, heads=2, seqlen=256)


if os.getenv("HSTU_STAGE_PROBE", "0") == "1":
    print("\n[stage probe] fixed q/k random, compare v=1 vs v=randn")
    print(
        f"{'D':>4} {'SEQ':>6} {'V_MODE':>8} | "
        f"{'S_cos':>8} {'S_max':>10} {'S_mean':>10} {'S_rel_l2':>10} | "
        f"{'O_cos':>8} {'O_max':>10} {'O_mean':>10} {'O_rel_l2':>10}"
    )
    print("-" * 112)
    for D in [64, 128]:
        for SEQ in [64, 128, 256, 512]:
            gq = torch.Generator(device=DEVICE)
            gk = torch.Generator(device=DEVICE)
            gq.manual_seed(SEED + D * 1000 + SEQ)
            gk.manual_seed(SEED + D * 1000 + SEQ + 1)
            q_fix = torch.randn((BS * SEQ, 1, D), generator=gq, device=DEVICE, dtype=torch.bfloat16)
            k_fix = torch.randn((BS * SEQ, 1, D), generator=gk, device=DEVICE, dtype=torch.bfloat16)
            q_fix = q_fix.to(torch.float8_e4m3fn).to(torch.bfloat16)
            k_fix = k_fix.to(torch.float8_e4m3fn).to(torch.bfloat16)

            for v_mode in ["ones", "randn"]:
                if v_mode == "ones":
                    v_cur = torch.ones((BS * SEQ, 1, D), device=DEVICE, dtype=torch.bfloat16)
                else:
                    gv = torch.Generator(device=DEVICE)
                    gv.manual_seed(SEED + D * 1000 + SEQ + 2)
                    v_cur = torch.randn((BS * SEQ, 1, D), generator=gv, device=DEVICE, dtype=torch.bfloat16)
                v_cur = v_cur.to(torch.float8_e4m3fn).to(torch.bfloat16)

                s_m, o_m = stage_error_probe(q_fix, k_fix, v_cur, ALPHA, SEQ)
                print(
                    f"{D:>4} {SEQ:>6} {v_mode:>8} | "
                    f"{s_m[0]:>8.4f} {s_m[1]:>10.6f} {s_m[2]:>10.6f} {s_m[3]:>10.6f} | "
                    f"{o_m[0]:>8.4f} {o_m[1]:>10.6f} {o_m[2]:>10.6f} {o_m[3]:>10.6f}"
                )
