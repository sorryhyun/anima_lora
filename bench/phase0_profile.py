"""Phase 0: Profile current SM120 FA4 path.

Goals:
  1. Separate forward vs backward timing at Anima's exact shapes
  2. Self-attention vs cross-attention breakdown
  3. num_stages sweep (set FA4_SM120_FWD_STAGES=N before running)
  4. Identify where wall-clock time actually goes

Usage:
  # Baseline (num_stages=1, the current default)
  python bench_phase0.py

  # Test with 2-stage pipelining
  FA4_SM120_FWD_STAGES=2 python bench_phase0.py

  # Test with 3-stage pipelining
  FA4_SM120_FWD_STAGES=3 python bench_phase0.py
"""

import os
import time
import torch

# ---------------------------------------------------------------------------
# Anima model shapes
# ---------------------------------------------------------------------------
BATCH = 2
N_HEADS = 32
HEAD_DIM = 64
SELF_SEQLEN = 4096       # image tokens (constant-token bucketing)
CROSS_KV_SEQLEN = 256    # text encoder output (padded to max_length)
DTYPE = torch.bfloat16

WARMUP = 5   # includes JIT compilation
ITERS = 30


def _sync():
    torch.cuda.synchronize()


def _make_qkv(batch, q_len, kv_len, heads, dim, dtype, requires_grad=False):
    q = torch.randn(batch, q_len, heads, dim, dtype=dtype, device="cuda", requires_grad=requires_grad)
    k = torch.randn(batch, kv_len, heads, dim, dtype=dtype, device="cuda", requires_grad=requires_grad)
    v = torch.randn(batch, kv_len, heads, dim, dtype=dtype, device="cuda", requires_grad=requires_grad)
    return q, k, v


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def bench_forward(q, k, v, flash_attn_func, warmup=WARMUP, iters=ITERS):
    """Time forward only (no grad)."""
    with torch.no_grad():
        for _ in range(warmup):
            flash_attn_func(q, k, v)
        _sync()

        start = time.perf_counter()
        for _ in range(iters):
            flash_attn_func(q, k, v)
        _sync()
    return (time.perf_counter() - start) / iters * 1000


def bench_backward(q, k, v, flash_attn_func, warmup=WARMUP, iters=ITERS):
    """Time backward only (forward is included but timed separately).

    Returns (fwd_ms, bwd_ms) or None if backward compilation fails.
    """
    # Warmup: full fwd+bwd to JIT-compile backward kernel
    try:
        out, _lse = flash_attn_func(q, k, v)
        grad = torch.randn_like(out)
        out.backward(grad)
    except Exception as e:
        return None, str(e).split('\n')[0]

    for _ in range(warmup - 1):
        out, _lse = flash_attn_func(q, k, v)
        grad = torch.randn_like(out)
        out.backward(grad)
    _sync()

    # Time forward and backward separately
    fwd_times = []
    bwd_times = []
    for _ in range(iters):
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
    drop = min(3, iters // 4)
    fwd_times = fwd_times[drop:]
    bwd_times = bwd_times[drop:]
    return sum(fwd_times) / len(fwd_times), sum(bwd_times) / len(bwd_times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    num_stages = os.environ.get("FA4_SM120_FWD_STAGES", "1")
    print("=" * 70)
    print(f"Phase 0: SM120 FA4 profiling  (num_stages={num_stages})")
    print(f"Shapes: B={BATCH}, H={N_HEADS}, D={HEAD_DIM}, "
          f"self_seq={SELF_SEQLEN}, cross_kv={CROSS_KV_SEQLEN}")
    print("=" * 70)

    from flash_attn.cute import flash_attn_func as fa4_func_raw

    # We need the raw func (returns out, lse) for backward timing
    # and the no-lse version for forward-only timing
    def fa4_fwd_only(q, k, v):
        return fa4_func_raw(q, k, v)

    # --- Self-attention ---
    print(f"\n--- Self-attention ({SELF_SEQLEN} x {SELF_SEQLEN}) ---")

    q, k, v = _make_qkv(BATCH, SELF_SEQLEN, SELF_SEQLEN, N_HEADS, HEAD_DIM, DTYPE)
    fwd_ms = bench_forward(q, k, v, fa4_fwd_only)
    print(f"  Forward only:  {fwd_ms:.2f} ms")
    del q, k, v

    q, k, v = _make_qkv(BATCH, SELF_SEQLEN, SELF_SEQLEN, N_HEADS, HEAD_DIM, DTYPE, requires_grad=True)
    result = bench_backward(q, k, v, fa4_func_raw)
    bwd_ok = result[0] is not None
    if bwd_ok:
        fwd_grad_ms, bwd_ms = result
        total_ms = fwd_grad_ms + bwd_ms
        print(f"  Forward (grad): {fwd_grad_ms:.2f} ms")
        print(f"  Backward:       {bwd_ms:.2f} ms")
        print(f"  Total fwd+bwd:  {total_ms:.2f} ms")
        print(f"  Bwd/Fwd ratio:  {bwd_ms / fwd_grad_ms:.2f}x")
    else:
        print(f"  Backward FAILED: {result[1]}")
        print(f"  (Forward-only numbers still valid)")
    del q, k, v

    # --- Cross-attention ---
    print(f"\n--- Cross-attention ({SELF_SEQLEN} x {CROSS_KV_SEQLEN}) ---")

    q, k, v = _make_qkv(BATCH, SELF_SEQLEN, CROSS_KV_SEQLEN, N_HEADS, HEAD_DIM, DTYPE)
    fwd_ms_cross = bench_forward(q, k, v, fa4_fwd_only)
    print(f"  Forward only:  {fwd_ms_cross:.2f} ms")
    del q, k, v

    if bwd_ok:
        q, k, v = _make_qkv(BATCH, SELF_SEQLEN, CROSS_KV_SEQLEN, N_HEADS, HEAD_DIM, DTYPE, requires_grad=True)
        result_cross = bench_backward(q, k, v, fa4_func_raw)
        if result_cross[0] is not None:
            fwd_grad_ms_cross, bwd_ms_cross = result_cross
            total_ms_cross = fwd_grad_ms_cross + bwd_ms_cross
            print(f"  Forward (grad): {fwd_grad_ms_cross:.2f} ms")
            print(f"  Backward:       {bwd_ms_cross:.2f} ms")
            print(f"  Total fwd+bwd:  {total_ms_cross:.2f} ms")
            print(f"  Bwd/Fwd ratio:  {bwd_ms_cross / fwd_grad_ms_cross:.2f}x")
        else:
            bwd_ok = False
        del q, k, v

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    N_SELF = 24   # Anima DiT layers
    N_CROSS = 24

    if bwd_ok:
        self_total = total_ms * N_SELF
        cross_total = total_ms_cross * N_CROSS
        attn_total = self_total + cross_total

        print(f"\n  Per-layer self-attn  (fwd+bwd): {total_ms:.2f} ms")
        print(f"  Per-layer cross-attn (fwd+bwd): {total_ms_cross:.2f} ms")
        print(f"\n  Estimated attention time per training step:")
        print(f"    Self-attn  ({N_SELF} layers): {self_total:.1f} ms")
        print(f"    Cross-attn ({N_CROSS} layers): {cross_total:.1f} ms")
        print(f"    Total attention:        {attn_total:.1f} ms")

        self_fwd_total = fwd_grad_ms * N_SELF
        cross_fwd_total = fwd_grad_ms_cross * N_CROSS
        fwd_total = self_fwd_total + cross_fwd_total
        print(f"\n  Forward-only attention per step: {fwd_total:.1f} ms")
        print(f"  Forward fraction of attention:  {fwd_total / attn_total * 100:.1f}%")
        print(f"  -> A 2x forward speedup saves: {fwd_total / 2:.1f} ms/step")
        print(f"  -> A 3x forward speedup saves: {fwd_total * 2 / 3:.1f} ms/step")
    else:
        # Forward-only summary
        self_fwd = fwd_ms * N_SELF
        cross_fwd = fwd_ms_cross * N_CROSS
        fwd_total = self_fwd + cross_fwd
        print(f"\n  NOTE: Backward compilation failed — forward-only numbers below")
        print(f"\n  Per-layer self-attn  (fwd): {fwd_ms:.2f} ms")
        print(f"  Per-layer cross-attn (fwd): {fwd_ms_cross:.2f} ms")
        print(f"\n  Estimated forward attention per training step:")
        print(f"    Self-attn  ({N_SELF} layers): {self_fwd:.1f} ms")
        print(f"    Cross-attn ({N_CROSS} layers): {cross_fwd:.1f} ms")
        print(f"    Total forward attention: {fwd_total:.1f} ms")
        print(f"\n  -> A 2x forward speedup saves: {fwd_total / 2:.1f} ms/step")
        print(f"  -> A 3x forward speedup saves: {fwd_total * 2 / 3:.1f} ms/step")


if __name__ == "__main__":
    main()
