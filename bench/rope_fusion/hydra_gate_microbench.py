"""HydraLoRA gate-weighted up-projection microbenchmark.

Per `configs/methods/lora.toml`:
  - num_experts = 12
  - network_dim = 32 (lora_dim)
  - hydra_router_layers = mlp.layer[12]
    -> Layer1 LinearShape(in=2048, out=8192)  ×28 blocks
    -> Layer2 LinearShape(in=8192, out=2048)  ×28 blocks
    -> 56 Hydra modules per step total.

We measure ONLY the gate-weighted up step:
    combined = einsum("be,eod->bod", gate, up_weight)   # (B,O,R) materialized
    out      = bmm(lx_3d, combined.transpose(1,2))      # (B,S,O)

Variants:
  A. current eager (fp32 cast + einsum + bmm)
  B. current under torch.compile (production path)
  C. fused alt: bf16 storage, fp32 accum via a single matmul reshape
       view (B*E, R) @ (E, O*R)? no — that doesn't work directly.
     Actually try: collapse to a per-step weight via gate.float() @ up.flatten,
       then use F.linear. Fewer kernels.
  D. baseline: single F.linear (no gate, no expert dim) — the "lower bound".

Reports per-step (×28) for each layer.
Run: python bench/rope_fusion/hydra_gate_microbench.py
"""

from __future__ import annotations

import torch

torch.manual_seed(0)

DEVICE = "cuda"
BF16 = torch.bfloat16
FP32 = torch.float32

B = 1
S = 4096
R = 32
E = 12

LAYERS = [
    ("layer1 (mlp 2048->8192)", 2048, 8192),
    ("layer2 (mlp 8192->2048)", 8192, 2048),
]
N_BLOCKS = 28
N_ITERS = 200
N_WARMUP = 50


def make_inputs(in_dim: int, out_dim: int):
    # lx is fp32 (bottleneck policy: lora_down forces fp32 inside).
    lx = torch.randn(B, S, R, dtype=FP32, device=DEVICE)
    # up_weight stored bf16 (adapter storage policy at training).
    up_weight = torch.zeros(E, out_dim, R, dtype=BF16, device=DEVICE)
    torch.nn.init.normal_(up_weight, std=0.01)
    # gate is bf16 (autocast dtype) coming out of softmax(router(...)).
    gate = torch.softmax(torch.randn(B, E, dtype=BF16, device=DEVICE), dim=-1)
    return lx, up_weight, gate


# --- variants ---


def hydra_current(lx, up_weight, gate):
    """Exact code from networks/lora_modules/hydra.py:309-318 (no expert mask
    branch — that's an autograd-time elementwise that compile fuses away)."""
    combined = torch.einsum("be,eod->bod", gate.float(), up_weight.float())
    lx_3d = lx.reshape(B, -1, lx.shape[-1])
    out = torch.bmm(lx_3d, combined.transpose(1, 2))
    return out


def hydra_alt_flat_linear(lx, up_weight, gate):
    """Alternative: build the per-sample combined weight via a single matmul,
    then F.linear. For B=1 this collapses to a normal linear — no bmm.

    combined_flat = gate.float() @ up_weight.float().flatten(1, 2)   # (B, O*R)
    combined      = combined_flat.view(B, O, R)
    out           = F.linear(lx, combined.squeeze(0))                # B=1 case
    """
    O = up_weight.shape[1]
    up_flat = up_weight.float().reshape(E, O * R)
    combined_flat = gate.float() @ up_flat  # (B, O*R)
    combined = combined_flat.view(B, O, R)
    if B == 1:
        out = torch.nn.functional.linear(lx, combined.squeeze(0))
    else:
        out = torch.bmm(lx, combined.transpose(1, 2))
    return out


def hydra_baseline_lora(lx, up_weight_single):
    """Lower-bound: ungated LoRA up — a single F.linear with one (O, R) weight."""
    return torch.nn.functional.linear(lx, up_weight_single)


# --- bench harness ---


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
    return times_ms[n_iters // 2] * 1000.0


def main() -> None:
    print(f"Hydra gate-up microbench  (B={B} S={S} R={R} E={E})")
    print(f"{'shape':<28} {'variant':<32} {'med us':>9} {'×28 blk':>10}")
    print("-" * 84)
    for tag, in_dim, out_dim in LAYERS:
        lx, up_weight, gate = make_inputs(in_dim, out_dim)
        # baseline-LoRA weight (fp32 to match the accum policy).
        up_baseline = torch.zeros(out_dim, R, dtype=FP32, device=DEVICE)
        torch.nn.init.normal_(up_baseline, std=0.01)

        # Numerical sanity (current vs alt should agree closely)
        out_cur = hydra_current(lx, up_weight, gate)
        out_alt = hydra_alt_flat_linear(lx, up_weight, gate)
        rel_err = (
            (out_cur - out_alt).abs().max() / out_cur.abs().max().clamp_min(1e-9)
        ).item()
        print(f"{tag:<28} {'(equiv check rel_err)':<32} {rel_err:>9.2e}")

        cur = _bench(lambda: hydra_current(lx, up_weight, gate))
        compiled_cur = torch.compile(
            hydra_current, backend="inductor", dynamic=False
        )
        cur_compiled = _bench(lambda: compiled_cur(lx, up_weight, gate))

        compiled_alt = torch.compile(
            hydra_alt_flat_linear, backend="inductor", dynamic=False
        )
        alt_compiled = _bench(lambda: compiled_alt(lx, up_weight, gate))

        baseline = _bench(lambda: hydra_baseline_lora(lx, up_baseline))
        compiled_base = torch.compile(
            hydra_baseline_lora, backend="inductor", dynamic=False
        )
        base_compiled = _bench(lambda: compiled_base(lx, up_baseline))

        for label, t in [
            ("current eager", cur),
            ("current compiled", cur_compiled),
            ("alt-flat-linear compiled", alt_compiled),
            ("baseline LoRA eager (lower bound)", baseline),
            ("baseline LoRA compiled (lower bound)", base_compiled),
        ]:
            print(f"{tag:<28} {label:<32} {t:>9.1f} {t * N_BLOCKS / 1000:>9.2f}ms")
        print()

    # NOTE: per-step total = (layer1 cost + layer2 cost) × 28 blocks for
    # whichever variant. Plus router/softmax/pooling — not measured here
    # (those are tiny vs the matmul mass).


if __name__ == "__main__":
    main()
