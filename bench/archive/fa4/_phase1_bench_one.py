"""Helper: benchmark a single backward config and print CSV results.

Invoked by phase1_bwd_sweep.py with FA4_SM120_BWD_* env vars set.
Outputs one line: "self_fwd,self_bwd,cross_fwd,cross_bwd" (ms) or "ERROR:msg".
"""

import sys
import time
import traceback
import torch

BATCH = 2
N_HEADS = 32
HEAD_DIM = 64
SELF_SEQLEN = 4096
CROSS_KV_SEQLEN = 256
DTYPE = torch.bfloat16

WARMUP = 5
ITERS = 30


def _sync():
    torch.cuda.synchronize()


def _make_qkv(q_len, kv_len):
    q = torch.randn(
        BATCH, q_len, N_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda", requires_grad=True
    )
    k = torch.randn(
        BATCH, kv_len, N_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda", requires_grad=True
    )
    v = torch.randn(
        BATCH, kv_len, N_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda", requires_grad=True
    )
    return q, k, v


def bench_bwd(q_len, kv_len, flash_attn_func):
    """Returns (fwd_ms, bwd_ms) or (None, None) on failure."""
    q, k, v = _make_qkv(q_len, kv_len)

    # Warmup (includes JIT compilation)
    try:
        out, _lse = flash_attn_func(q, k, v)
        grad = torch.randn_like(out)
        out.backward(grad)
    except Exception as e:
        # Print full traceback to stderr for debugging
        traceback.print_exc(file=sys.stderr)
        msg = f"{type(e).__name__}: {e}"
        return None, None, msg[:200]

    for _ in range(WARMUP - 1):
        q.grad = k.grad = v.grad = None
        out, _lse = flash_attn_func(q, k, v)
        grad = torch.randn_like(out)
        out.backward(grad)
    _sync()

    # Timed iterations
    fwd_times = []
    bwd_times = []
    for _ in range(ITERS):
        q.grad = k.grad = v.grad = None
        _sync()
        t0 = time.perf_counter()
        out, _lse = flash_attn_func(q, k, v)
        _sync()
        t1 = time.perf_counter()
        grad = torch.randn_like(out)
        out.backward(grad)
        _sync()
        t2 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)

    # Drop first few for stability
    drop = min(3, ITERS // 4)
    fwd_times = fwd_times[drop:]
    bwd_times = bwd_times[drop:]
    return sum(fwd_times) / len(fwd_times), sum(bwd_times) / len(bwd_times), None


def main():
    try:
        from flash_attn.cute import flash_attn_func
    except Exception as e:
        print(f"ERROR:import failed: {e}")
        return

    # Self-attention
    self_fwd, self_bwd, err = bench_bwd(SELF_SEQLEN, SELF_SEQLEN, flash_attn_func)
    if err:
        print(f"ERROR:self-attn: {err}")
        return

    # Cross-attention
    cross_fwd, cross_bwd, err = bench_bwd(SELF_SEQLEN, CROSS_KV_SEQLEN, flash_attn_func)
    if err:
        print(f"ERROR:cross-attn: {err}")
        return

    print(f"{self_fwd:.4f},{self_bwd:.4f},{cross_fwd:.4f},{cross_bwd:.4f}")


if __name__ == "__main__":
    main()
