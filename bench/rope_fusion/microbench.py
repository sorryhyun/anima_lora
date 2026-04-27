"""RoPE fusion microbenchmark.

Compares Anima's hand-rolled `apply_rotary_pos_emb_qk` against
`flash_attn.layers.rotary.apply_rotary_emb` at the realistic per-block
shape used in the DiT (B=1, S=4096 constant-token, H=16, D=128, bf16).

Reports:
  - Numerical equivalence (max abs err vs current path)
  - Median per-call us under torch.cuda.Event
  - Per-step extrapolation: 28 blocks * 2 (q+k) calls

Run: python bench/rope_fusion/microbench.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from library.anima.models import apply_rotary_pos_emb_qk  # noqa: E402

try:
    from flash_attn.layers.rotary import apply_rotary_emb as fa_apply_rotary
except ImportError:
    print("flash_attn not installed; aborting.")
    sys.exit(1)


# Anima defaults: model_channels=2048, num_heads=16 -> head_dim=128.
# Constant-token bucket = 4096. bf16 mixed precision.
B, S, H, D = 1, 4096, 16, 128
DTYPE = torch.bfloat16
DEVICE = "cuda"
N_ITERS = 200
N_WARMUP = 50


def make_inputs():
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(B, S, H, D, dtype=DTYPE, device=DEVICE)
    # Anima cos/sin layout from VideoRopePosition3DEmb.generate_embeddings:
    #   cat([emb_t, emb_h, emb_w] * 2, dim=-1) -> [T*H*W, 1, 1, D]
    # The "* 2" duplicates the freqs along the last dim, so cos[..., :D/2] == cos[..., D/2:].
    half_freqs = torch.randn(S, D // 2, dtype=torch.float32, device=DEVICE)
    full = torch.cat([half_freqs, half_freqs], dim=-1)  # [S, D]
    cos_anima = full.cos().unsqueeze(1).unsqueeze(1)  # [S, 1, 1, D]
    sin_anima = full.sin().unsqueeze(1).unsqueeze(1)  # [S, 1, 1, D]
    # flash_attn wants (seqlen_rotary, rotary_dim/2) — un-duplicated.
    cos_fa = full[:, : D // 2].cos()  # [S, D/2]
    sin_fa = full[:, : D // 2].sin()  # [S, D/2]
    return q, k, (cos_anima, sin_anima), (cos_fa, sin_fa)


def _bench(fn, n_iters=N_ITERS, n_warmup=N_WARMUP):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    for i in range(n_iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    times_ms.sort()
    return {
        "median_us": times_ms[n_iters // 2] * 1000.0,
        "p10_us": times_ms[n_iters // 10] * 1000.0,
        "p90_us": times_ms[(n_iters * 9) // 10] * 1000.0,
    }


def main() -> None:
    q, k, (cos_a, sin_a), (cos_fa, sin_fa) = make_inputs()

    # --- numerical equivalence ---
    q_ref, k_ref = apply_rotary_pos_emb_qk(
        q.clone(), k.clone(), (cos_a, sin_a), tensor_format="bshd"
    )
    q_fa = fa_apply_rotary(q.clone(), cos_fa, sin_fa, interleaved=False, inplace=False)
    k_fa = fa_apply_rotary(k.clone(), cos_fa, sin_fa, interleaved=False, inplace=False)
    err_q = (q_ref.float() - q_fa.float()).abs().max().item()
    err_k = (k_ref.float() - k_fa.float()).abs().max().item()
    ref_scale = q_ref.float().abs().max().item()
    print(f"Numerical equivalence (bf16):")
    print(f"  max|q_anima - q_fa| = {err_q:.3e}  (ref scale {ref_scale:.3e})")
    print(f"  max|k_anima - k_fa| = {err_k:.3e}")
    print()

    # --- benchmarks ---
    def run_anima():
        apply_rotary_pos_emb_qk(q, k, (cos_a, sin_a), tensor_format="bshd")

    def run_fa_oop():
        fa_apply_rotary(q, cos_fa, sin_fa, interleaved=False, inplace=False)
        fa_apply_rotary(k, cos_fa, sin_fa, interleaved=False, inplace=False)

    def run_fa_inplace():
        # Reset to fresh tensors each call, since inplace mutates input.
        q_buf = q.clone()
        k_buf = k.clone()
        fa_apply_rotary(q_buf, cos_fa, sin_fa, interleaved=False, inplace=True)
        fa_apply_rotary(k_buf, cos_fa, sin_fa, interleaved=False, inplace=True)

    # Compile the current path (matches production: per-block _forward is compiled).
    compiled_anima = torch.compile(
        apply_rotary_pos_emb_qk, backend="inductor", dynamic=False
    )

    def run_anima_compiled():
        compiled_anima(q, k, (cos_a, sin_a), tensor_format="bshd")

    rows = [
        ("anima eager (apply_rotary_pos_emb_qk)", run_anima),
        ("anima torch.compile (inductor)", run_anima_compiled),
        ("flash_attn out-of-place (q+k)", run_fa_oop),
        ("flash_attn in-place (q+k, +clone overhead)", run_fa_inplace),
    ]

    print(f"Shape: B={B} S={S} H={H} D={D} dtype={DTYPE}")
    print(f"Iters: {N_ITERS} (warmup {N_WARMUP})")
    print(f"{'impl':<48} {'median us':>10} {'p10':>8} {'p90':>8}")
    print("-" * 76)
    results = {}
    for name, fn in rows:
        r = _bench(fn)
        results[name] = r
        print(f"{name:<48} {r['median_us']:>10.2f} {r['p10_us']:>8.2f} {r['p90_us']:>8.2f}")
    print()

    # --- per-step extrapolation ---
    # Each DiT self-attn block calls apply_rotary_pos_emb_qk once (q+k together).
    # The current op already does both arms inside one call; flash_attn requires
    # two calls (q, then k). So compare: 1x anima vs 2x flash_attn per block.
    blocks = 28
    per_step_anima = results["anima eager (apply_rotary_pos_emb_qk)"]["median_us"] * blocks
    per_step_anima_c = results["anima torch.compile (inductor)"]["median_us"] * blocks
    per_step_fa = results["flash_attn out-of-place (q+k)"]["median_us"] * blocks
    print(f"Per-step extrapolation ({blocks} self-attn blocks, q+k both rotated):")
    print(f"  anima eager:    {per_step_anima:>8.1f} us")
    print(f"  anima compiled: {per_step_anima_c:>8.1f} us")
    print(f"  flash_attn:     {per_step_fa:>8.1f} us")
    if per_step_anima_c > 0:
        delta = per_step_anima_c - per_step_fa
        pct = 100.0 * delta / per_step_anima_c
        print(f"  delta vs compiled: {delta:>+8.1f} us  ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
