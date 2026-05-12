"""Directed correctness checks for the cuTile D256 BF16 prototype."""

import math

import torch

from .wrapper import TileToolchainError, hstu_bf16_d256_cudatile, require_available


def _make_inputs(batch: int, seqlen: int, heads: int):
    q = torch.randn((batch * seqlen, heads, 256), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu = torch.arange(0, (batch + 1) * seqlen, seqlen, device="cuda", dtype=torch.int32)
    return q, k, v, cu


def _reference(q, k, v, cu, seqlen: int, *, alpha: float, causal: bool):
    offsets = [int(x) for x in cu.detach().cpu().tolist()]
    heads = q.shape[1]
    pieces = []
    for b in range(len(offsets) - 1):
        start = offsets[b]
        end = offsets[b + 1]
        q_b = q[start:end]
        k_b = k[start:end]
        v_b = v[start:end]
        out_b = torch.empty_like(q_b)
        q_len = q_b.shape[0]
        k_len = k_b.shape[0]
        qk_offset = k_len - q_len
        q_idx = torch.arange(q_len, device=q.device)[:, None]
        k_idx = torch.arange(k_len, device=q.device)[None, :]
        valid = torch.ones((q_len, k_len), device=q.device, dtype=torch.bool)
        if causal:
            valid = k_idx <= qk_offset + q_idx
        for h in range(heads):
            scores = q_b[:, h].float() @ k_b[:, h].float().T
            scores = scores.masked_fill(~valid, -torch.inf)
            s = scores * alpha
            p = torch.nn.functional.silu(s).masked_fill(~valid, 0.0)
            out = p.to(torch.bfloat16) @ v_b[:, h]
            out_b[:, h] = (out / seqlen).to(torch.bfloat16)
        pieces.append(out_b)
    return torch.cat(pieces, dim=0).contiguous()


def run_case(*, causal: bool, batch: int = 1, seqlen: int = 128, heads: int = 1):
    q, k, v, cu = _make_inputs(batch, seqlen, heads)
    out = hstu_bf16_d256_cudatile(
        q,
        k,
        v,
        cu,
        cu,
        seqlen,
        seqlen,
        alpha=1.0,
        causal=causal,
        scaling_seqlen=seqlen,
    )
    ref = _reference(q, k, v, cu, seqlen, alpha=1.0, causal=causal)
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    max_err = (out.float() - ref.float()).abs().max().item()
    print(
        f"causal={causal} batch={batch} seqlen={seqlen} heads={heads} "
        f"cos={cos:.6f} max_err={max_err:.6f}"
    )
    if not math.isfinite(cos) or cos < 0.995:
        raise AssertionError(f"cosine similarity too low: {cos}")
    return cos, max_err


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    try:
        require_available()
        for batch in (1, 2):
            for heads in (1, 4):
                run_case(causal=False, batch=batch, seqlen=128, heads=heads)
                run_case(causal=True, batch=batch, seqlen=128, heads=heads)
    except TileToolchainError as exc:
        print(f"SKIP: {exc}")
        return


if __name__ == "__main__":
    main()

