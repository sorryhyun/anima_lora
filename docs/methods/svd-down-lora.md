# SVD-Down LoRA initialization

Seed plain LoRA's `lora_down` (input basis) from the top-r right singular
vectors of the pretrained weight `W₀`, instead of a random Kaiming basis. It is
**ordinary LoRA after init** — same module, same saved weights, same merge and
inference path — only the down projection's starting directions change.

## Quick start

In `configs/methods/lora.toml`, set on the plain-LoRA path:

```toml
down_init = "weight_svd"   # default: "kaiming"
```

Then `make lora`. No new flags, no new checkpoint format.

## What it does

For an adapted Linear with `W₀ = UΣVᵀ`, initialize

```
A₀ = V_r^T / √3      (lora_down)
B₀ = 0               (lora_up)
```

so `ΔW = sB₀A₀ = 0` at init — the pretrained `W₀` is untouched and the first
forward is unchanged. The `1/√3` matches the expected row-norm of the Kaiming
default (a row of `V_rᵀ` has norm 1; a Kaiming row has `E[‖·‖²] ≈ 1/3`), so this
is a **better direction, not a larger step**.

Why bother: it reads the input directions the pretrained Linear is most
responsive to, while keeping plain LoRA's full first-step tangent — the whole
`d_out × r` up-projection `B` gets gradient on step 1 (only `lora_down` is
dormant, exactly like Kaiming LoRA). This is the half of OrthoInit worth keeping
without its cold-start bottleneck (OrthoInit's diagonal-only first step gives
gradient to just the `r` singular-value scalars). Both `A` and `B` stay
trainable, so the adapter can rotate away from the SVD basis immediately.

## Scope

- **Plain LoRA only**, **Linear layers only** (v0). Conv2d keeps Kaiming. The
  config resolver rejects `down_init="weight_svd"` combined with ortho / Hydra /
  Chimera / MoE paths — those carry their own basis parameterization.
- Composes with T-LoRA (the `_timestep_mask` acts on the bottleneck after
  `lora_down`) and with channel-scaling (absorption runs after init, as for
  Kaiming).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `down_init` | `"kaiming"` | `"kaiming"` (default `kaiming_uniform_(a=√5)`) or `"weight_svd"` (SVD-Down) |

## Implementation

| File | Role |
|------|------|
| `networks/lora_modules/lora.py` | `_init_down_weight_svd()` — randomized SVD of `W₀`, copies `V_rᵀ/√3` into `lora_down` |
| `networks/lora_anima/config.py` | `down_init` cfg field + validation (Linear/plain-LoRA-only guard) |

The randomized SVD (`torch.svd_lowrank`, `q = min(rank+6, …)`, `niter=2`) is the
same construction already used in `networks/lora_modules/ortho.py` — no new
numerical machinery. Startup cost is paid once per adapted Linear at init.

## Status

Phase 0 parameterization probe **passed** all gates
(`bench/turbo/probe_ortho_init_step.py`,
`bench/turbo/results/20260621-2149-svd-down-phase0-clean/`): zero-output at init,
gradient in `lora_up` only on step 1, step-1 `‖ΔW‖_F` within 0.5×–2× plain LoRA
(measured ≈1.00×), and improved update alignment in the W₀-aligned regime
(`cos_ideal` 0.42 vs plain LoRA 0.14) with no harm in the isotropic regime.

Original proposal & full theory: `_archive/proposals/svd_down_lora_init.md`.

## Where this came from

The line started with **StelLA** (NeurIPS 2025), not with the internal probe —
the archived proposal credits `bench/turbo/probe_ortho_init_step.py`, but that
probe was the trigger, not the source. StelLA's three-factor `USVᵀ` (U, V on the
Stiefel manifold, S carrying amplitude) **is** the repo's OrthoInit
parameterization `ΔW = s·P·diag(λ)·Q`, so the cold-start critique SVD-Down is
built on is a critique of StelLA's factorization. Its Table 5 initialization
ablation — the SVD seed *washes out* (SVD-major ≈ SVD-minor ≈ random) once the
subspace is trainable — is the question SVD-Down answers for free LoRA: keep the
principal input basis, drop the manifold constraint and the paired-dyad cold
start. Reading StelLA forked into two proposals in one commit (`2674b59e`,
2026-06-22): this one for plain LoRA, and `docs/proposal/stella_chimera.md` for
the chimera case.

## References

- Li et al., [StelLA: Subspace Learning in Low-rank Adaptation using Stiefel
  Manifold](https://arxiv.org/abs/2510.01938), NeurIPS 2025 (Spotlight) —
  origin of this line; see above. Code:
  <https://github.com/SonyResearch/stella>.
- Hu et al., [LoRA](https://arxiv.org/abs/2106.09685), 2021.
- Meng et al., [PiSSA](https://arxiv.org/abs/2404.02948), NeurIPS 2024 — also
  starts from principal components but residualizes the base weight (SVD-Down
  keeps `W₀` unchanged + `ΔW=0`).
- Paischer et al., [EVA](https://arxiv.org/abs/2410.07170), NeurIPS 2025 /
  Wang et al., [LoRA-GA](https://arxiv.org/abs/2407.05000), 2024 — data/gradient-
  informed bases; the principled next arm if weight-SVD fails on large domain
  shifts.
