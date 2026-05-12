"""Small benchmark harness for the cuTile D256 BF16 prototype."""

import time

import torch

from .test_d256 import _make_inputs
from .wrapper import hstu_bf16_d256_cudatile, require_available


def bench_case(*, causal: bool, batch: int, seqlen: int, heads: int, iters: int = 100):
    q, k, v, cu = _make_inputs(batch, seqlen, heads)
    require_available()
    for _ in range(10):
        hstu_bf16_d256_cudatile(q, k, v, cu, cu, seqlen, seqlen, causal=causal)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        hstu_bf16_d256_cudatile(q, k, v, cu, cu, seqlen, seqlen, causal=causal)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000.0 / iters
    print(f"causal={causal} batch={batch} seqlen={seqlen} heads={heads} ms={ms:.4f}")


def main():
    for causal in (False, True):
        bench_case(causal=causal, batch=1, seqlen=128, heads=1)
        bench_case(causal=causal, batch=2, seqlen=256, heads=4)


if __name__ == "__main__":
    main()

