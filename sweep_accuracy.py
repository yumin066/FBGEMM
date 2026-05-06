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
    hstu_attn_varlen_func,
    quantize_for_block_scale_qk_along_d,
    quantize_for_block_scale_v_along_n,
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
SWEEP_DIMS = [128]  # D=64 disabled in this build (HSTU_DISABLE_HDIM64=TRUE)
SWEEP_HEADS = [1, 4]
SWEEP_SEQS = [128, 256, 512]

# Baseline defaults to avoid branch interference in debugging runs.
# Users can still override from shell when explicitly needed.
os.environ.setdefault("HSTU_EXP_A_SYNC_K", "0")
os.environ.setdefault("HSTU_EXP_B_PBUF_IN_SK", "0")
os.environ.setdefault("HSTU_DEBUG_GEMM1_ONLY", "0")

EXP_A = os.getenv("HSTU_EXP_A_SYNC_K", "0")
EXP_B = os.getenv("HSTU_EXP_B_PBUF_IN_SK", "0")
GEMM1_ONLY = os.getenv("HSTU_DEBUG_GEMM1_ONLY", "0") == "1"
RUN_PAGED_KV_SWEEP = os.getenv("HSTU_SKIP_PAGED_KV_SWEEP", "0") != "1"
print(f"[sweep config] HSTU_EXP_A_SYNC_K={EXP_A} HSTU_EXP_B_PBUF_IN_SK={EXP_B} HSTU_DEBUG_GEMM1_ONLY={int(GEMM1_ONLY)}")

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
                             target_len, dtype=torch.float16):
    D = 128
    PAGE_SIZE = 64
    torch.manual_seed(SEED + batch_size * 100 + heads * 10 + new_history_len + prev_history_len + target_len)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + batch_size * 100 + heads * 10 + new_history_len + prev_history_len + target_len)

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

    ref = _hstu_paged_kv_attention(
        num_heads=heads,
        attention_dim=D,
        linear_dim=D,
        seqlen_q=max_seqlen_q,
        seqlen_k=max_seqlen_k,
        q=q,
        k=k,
        v=v,
        q_offsets=cu_seqlens_q,
        k_offsets=cu_seqlens_k,
        num_targets=num_targets,
        invalid_attn_mask=mask,
        alpha=ALPHA,
        upcast=True,
        kv_cache=kv_cache,
        page_offsets=page_offsets,
        page_ids=page_ids,
        last_page_lens=last_page_lens,
    )

    out = hstu_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        scaling_seqlen=-1,
        num_contexts=None,
        num_targets=num_targets,
        target_group_size=1,
        window_size=(-1, 0),
        alpha=ALPHA,
        rab=None,
        has_drab=False,
        kv_cache=kv_cache,
        page_offsets=page_offsets,
        page_ids=page_ids,
        last_page_lens=last_page_lens,
        func=None,
        quant_mode=2,
    )
    torch.cuda.synchronize()

    cos, max_err, mean_err = metric_report(out.flatten(), ref.flatten())
    _, _, _, rel_l2 = norm_report(out.flatten(), ref.flatten())
    if cos < 0.995:
        raise AssertionError(f"paged KV fp8_gt_cos={cos:.6f} < 0.995")
    return cos, max_err, mean_err, rel_l2, last_page_lens.detach().cpu().tolist()

if GEMM1_ONLY:
    print("[mode] GEMM1 bypass: output = sum(Q×K^T tiles) / scaling_seqlen, GEMM2 skipped")
    print(
        f"{'D':>4} {'H':>4} {'SEQ':>6} | "
        f"{'bf16_fp8_cos':>12} {'max_err':>10} {'mean_err':>10} | "
        f"{'||bf16||':>10} {'||fp8||':>10} {'rel_l2':>10} | "
        f"{'bf16_gt_cos':>11} {'bf16_gt_max':>12} {'fp8_gt_cos':>11} {'fp8_gt_max':>11}"
    )
    print("-" * 130)
else:
    print(
        f"{'D':>4} {'H':>4} {'SEQ':>6} | "
        f"{'bf16_fp8_cos':>12} {'max_err':>10} {'mean_err':>10} | "
        f"{'||bf16||':>10} {'||fp8||':>10} {'rel_l2':>10} | "
        f"{'bf16_gt_cos':>11} {'bf16_gt_max':>12} {'bf16_gt_mean':>12} | "
        f"{'fp8_gt_cos':>10} {'fp8_gt_max':>11} {'fp8_gt_mean':>11} | "
        f"{'paged_gt_cos':>12} {'paged_gt_max':>12} {'paged_gt_mean':>13}"
    )
    print("-" * 213)
    if RUN_PAGED_KV_SWEEP:
        print("[paged kv] appended columns use the current causal/target paged-KV path")

USE_ONES = os.getenv("HSTU_USE_ONES", "0") == "1"
if USE_ONES:
    print("[input mode] Q=K=V=ones (all 1.0)")

for D in SWEEP_DIMS:
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
                print(f"{D:>4} {H:>4} {SEQ:>6} | BF16 ERROR: {e}")
                continue

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
                    v_raw, cu, block_size=128, fp8_type=torch.float8_e4m3fn)
                sf_q = pack_descale_to_e8m0x4_int32(q_descale)
                sf_k = pack_descale_to_e8m0x4_int32(k_descale)
                # TMA SFV requires sf_v expanded from [H, total_blocks] to [H, total_tokens]
                # so that each block's scale repeats kBlockN times (one copy per token).
                sf_v = pack_descale_to_e8m0x4_int32(v_descale).repeat_interleave(128, dim=1)
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

            o_b = out_bf16.float().flatten()
            o_f = out_fp8.float().flatten()
            cos, me, mn = metric_report(o_b, o_f)
            n_b, n_f, _, rel_l2 = norm_report(o_b, o_f)

            # Dequantized ground truth for FP8: reproduces exactly the e8m0-rounded
            # scales that the kernel uses, so cross-head scale non-uniformity cancels out.
            q_deq = e8m0_dequant(q_fp8, q_descale, BS * SEQ)
            k_deq = e8m0_dequant(k_fp8, k_descale, BS * SEQ)
            v_deq = e8m0_dequant(v_fp8, v_descale, BS * SEQ)
            gt_deq = fullpath_ground_truth(q_deq, k_deq, v_deq, ALPHA, SEQ)
            gt_deq_flat = gt_deq.float().flatten()
            cos_f_gt_pre, _, _ = metric_report(o_f, gt_deq_flat)
            flag = " <<<" if cos_f_gt_pre < 0.95 else ""

            if GEMM1_ONLY:
                # Python reference: accumulate Q×K^T tiles the same way the kernel does.
                # Each tile [kBlockM × kBlockN] is summed into a [kBlockM × kHeadDim]
                # buffer starting at column 0.  BF16 and FP8 can use different kBlockN
                # on SM120, so build their debug references separately.
                def gemm1_debug_reference(q_src: torch.Tensor, k_src: torch.Tensor, k_bn: int):
                    k_bm = 128
                    q_f = q_src.float()
                    k_f = k_src.float()
                    n_m = (SEQ + k_bm - 1) // k_bm
                    n_n = (SEQ + k_bn - 1) // k_bn
                    gt = torch.zeros(SEQ, H, D, dtype=torch.float32, device=DEVICE)
                    for im in range(n_m):
                        q_tile = q_f[im*k_bm:(im+1)*k_bm]
                        for in_ in range(n_n):
                            k_tile = k_f[in_*k_bn:(in_+1)*k_bn]
                            s_tile = torch.einsum("mhd,nhd->mhn", q_tile, k_tile)
                            actual_n = s_tile.shape[2]
                            gt[im*k_bm:(im+1)*k_bm, :, :actual_n] += s_tile
                    return (gt / SEQ).flatten()

                gt_b = gemm1_debug_reference(q, k, 128)
                gt_f = gemm1_debug_reference(
                    q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn), 128)
                cos_b_gt, me_b_gt, _ = metric_report(o_b, gt_b)
                cos_f_gt, me_f_gt, _ = metric_report(o_f, gt_f)
                print(
                    f"{D:>4} {H:>4} {SEQ:>6} | "
                    f"{cos:>12.4f} {me:>10.6f} {mn:>10.6f} | "
                    f"{n_b:>10.4f} {n_f:>10.4f} {rel_l2:>10.6f} | "
                    f"{cos_b_gt:>11.4f} {me_b_gt:>12.6f} {cos_f_gt:>11.4f} {me_f_gt:>11.6f}{flag}"
                )
            else:
                gt = fullpath_ground_truth(q, k, v, ALPHA, SEQ).float().flatten()
                cos_b_gt, me_b_gt, mn_b_gt = metric_report(o_b, gt)
                # FP8 kernel vs dequantized ground truth (correct comparison)
                cos_f_gt, me_f_gt, mn_f_gt = metric_report(o_f, gt_deq_flat)
                if RUN_PAGED_KV_SWEEP:
                    paged_cos, paged_max, paged_mean, _, _ = compute_paged_kv_metrics(
                        batch_size=BS,
                        heads=H,
                        new_history_len=SEQ,
                        prev_history_len=0,
                        target_len=0,
                    )
                    paged_cols = f" | {paged_cos:>12.4f} {paged_max:>12.6f} {paged_mean:>13.6f}"
                else:
                    paged_cols = f" | {'N/A':>12} {'N/A':>12} {'N/A':>13}"
                print(
                    f"{D:>4} {H:>4} {SEQ:>6} | "
                    f"{cos:>12.4f} {me:>10.6f} {mn:>10.6f} | "
                    f"{n_b:>10.4f} {n_f:>10.4f} {rel_l2:>10.6f} | "
                    f"{cos_b_gt:>11.4f} {me_b_gt:>12.6f} {mn_b_gt:>12.6f} | "
                    f"{cos_f_gt:>10.4f} {me_f_gt:>11.6f} {mn_f_gt:>11.6f}{paged_cols}{flag}"
                )
        sys.stdout.flush()


def run_paged_kv_case(label, batch_size, heads, new_history_len, prev_history_len,
                      target_len, dtype=torch.float16):
    cos, max_err, mean_err, rel_l2, last_page_lens = compute_paged_kv_metrics(
        batch_size=batch_size,
        heads=heads,
        new_history_len=new_history_len,
        prev_history_len=prev_history_len,
        target_len=target_len,
        dtype=dtype,
    )
    print(
        f"{label:>22} | B={batch_size:<2} H={heads:<2} "
        f"new={new_history_len:<3} prev={prev_history_len:<3} tgt={target_len:<3} | "
        f"cos={cos:.6f} max={max_err:.6f} mean={mean_err:.6f} rel_l2={rel_l2:.6f} "
        f"last_page={last_page_lens}"
    )


if (
    not GEMM1_ONLY
    and RUN_PAGED_KV_SWEEP
    and os.getenv("HSTU_PAGED_KV_EDGE_CASES", "0") == "1"
):
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
