# SVD-Down LoRA initialization

Seed plain LoRA's `lora_down` (input basis) from the top-r right singular
vectors of the pretrained weight `W₀`, instead of a random Kaiming basis. It is
ordinary LoRA after init — same module, same saved weights, same merge and
inference path — only the down projection's starting directions change.

## Quick start

`configs/methods/lora.toml` already sets it for `make lora` (the code default is
`"kaiming"`):

```toml
down_init = "weight_svd"
```

The checkpoint format is unchanged.

## What it does

For an adapted Linear with `W₀ = UΣVᵀ`, initialize

```
A₀ = V_r^T / √3      (lora_down)
B₀ = 0               (lora_up)
```

so `ΔW = sB₀A₀ = 0` at init — the pretrained `W₀` is untouched and the first
forward is unchanged. The `1/√3` matches the expected row-norm of the Kaiming
default (a row of `V_rᵀ` has norm 1; a Kaiming row has `E[‖·‖²] ≈ 1/3`), so this
is a better direction, not a larger step.

It reads the input directions the pretrained Linear is most
responsive to, while keeping plain LoRA's full first-step tangent — the whole
`d_out × r` up-projection `B` gets gradient on step 1 (only `lora_down` is
dormant, exactly like Kaiming LoRA), unlike a diagonal-only parameterization
where the first step's gradient would reach just the `r` singular-value
scalars. Both `A` and `B` stay trainable, so the adapter can rotate away from
the SVD basis immediately.

## Scope

- **Plain LoRA only**, Linear layers only (v0). Conv2d keeps Kaiming. The
  config resolver rejects a non-Kaiming `down_init` combined with the Hydra
  MoE path — it carries its own basis parameterization.
- Composes with T-LoRA (the `_timestep_mask` acts on the bottleneck after
  `lora_down`) and with channel-scaling (absorption runs after init, as for
  Kaiming).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `down_init` | `"kaiming"` | `"kaiming"` (default `kaiming_uniform_(a=√5)`), `"weight_svd"` (SVD-Down, this doc), or the gradient-seeded `"grad_svd"` / `"basis_file"` (see below) |
| `grad_basis_file` | — | Network arg: path to a gradient-basis artifact. Required by `down_init="basis_file"`; written automatically by `"grad_svd"`. |
| `svd_slice` | `0` | `weight_svd` only: seed from right singular vectors `[k·r, (k+1)·r)` instead of the top-r. Slices of one orthonormal basis are mutually orthogonal, so adapters trained with different `k` never share an input subspace at merge — a per-artist address. The window must fit every targeted layer (`(k+1)·r ≤ min(W.shape)`; the base DiT's 256-row adaln `.1` Linears cap r=32 at `k ≤ 7`) or the init refuses. Stamped as `ss_svd_slice`. Since 2026-09-12 the basis is exact (the earlier `q=r+6` randomized sketch captured only 0.80–0.93 of the true top-r on real DiT layers and re-drew per call, so slices from it were not orthogonal); slice 0 therefore now is the actual top-r. Motivation and the (so far flat) merge reads: `bench/merge_basis/README.md`. |

## Implementation

| File | Role |
|------|------|
| `networks/lora_modules/lora.py` | `_top_right_singular_vectors()` — the exact basis; `_init_down_weight_svd()` copies `V_rᵀ/√3` into `lora_down` |
| `networks/lora_anima/config.py` | `down_init` cfg field + validation (Linear/plain-LoRA-only guard) |
| `networks/grad_basis.py` | The gradient-seeded siblings: sketch, basis artifact I/O, and the shared `V_rᵀ/√3` copy |

The basis is computed per adapted Linear at init, from an **eigendecomposition of
the smaller Gram matrix** rather than `torch.linalg.svd` — the same subspace, but
**7.5× cheaper** end to end on the base DiT (54.2 s → 7.2 s for all 448
`blocks.*` Linears at r=32, 5070 Ti) and with tighter-orthonormal columns than
cuSOLVER's Jacobi SVD (1e-6 vs 1e-3 max off-diagonal), which is the property
`svd_slice` leans on. Measured capture against the exact V is 1.000 on every
weight group.

Settled by the same measurement:

- **Batching the per-layer calls by shape buys nothing.** cuSOLVER has no batched
  kernel at these sizes and loops internally: 34.4 s → 35.0 s for the 168
  `(2048, 2048)` layers.
- **The Gram route needs a conditioning guard**, because squaring the spectrum
  sinks the bottom of the requested window under the fp32 eigh error floor — the
  1569:1 `(6144, 256)` Linears returned an orthonormal but *wrong* slice-7
  window (0.796 capture), and for `in > out` (where V only follows through
  `W^T U / σ`) the 3325:1 adaln `.1` Linears lost orthogonality outright (0.49).
  Both branches read `λ_cols/λ_0` off the eigenvalues they just computed and hand
  the layer to `linalg.svd` below `_GRAM_EIG_FLOOR` — those are the narrow,
  cheap layers, so the fallback costs ~1 s of the 7.2 s.

A disk-cached basis was considered instead and is not worth it: the basis is a
pure function of `W₀`, so it caches cleanly, but one artifact is 56 MB–903 MB
depending on stored columns/dtype, it is baked to the *specific* checkpoint (base
vs aesthetic vs any merged/souped DiT — not just to depth, so `load_basis`'s
block-count guard would not catch a mismatch, and a wrong basis is a silently
worse init rather than an error), and it would now be saving ~7 s per run.

## Status

Phase 0 parameterization probe passed all gates
(`bench/turbo/results/20260621-2149-svd-down-phase0-clean/`; the probe script
has since been removed from the live tree): zero-output at init,
gradient in `lora_up` only on step 1, step-1 `‖ΔW‖_F` within 0.5×–2× plain LoRA
(measured ≈1.00×), and improved update alignment in the W₀-aligned regime
(`cos_ideal` 0.42 vs plain LoRA 0.14) with no harm in the isotropic regime.

Original proposal & full theory: `_archive/proposals/svd_down_lora_init.md`.

## Gradient-seeded siblings (`grad_svd` / `basis_file`)

Same seed shape, different basis: instead of `W₀`'s top-r right singular
vectors, take the top-r row space of the **task gradient** (LoRA-GA / LoRA-One
lineage — with `B = 0` the first optimizer step is the rank-r truncated full-FT
step). Everything else is identical, `V_rᵀ/√3` included, so the modes differ
only in which directions `A` starts in.

- `"grad_svd"` — `train.py` sketches the run's own cached dataset against the
  frozen DiT before the network is built (~1.3 s/image), writes
  `<output_name>.grad_basis.safetensors` beside the checkpoint, and loads it
  back through the `basis_file` path. Refused with `blocks_to_swap > 0`.
- `"basis_file"` — read a basis built once over many artists
  (`bench/grad_init/build_universal_basis.py`). No per-run backward.

Measured on this DiT (`bench/grad_init/README.md`): `weight_svd` passes **0.21**
of an artist's first-step gradient energy, the artist's own gradient basis
**0.74**, a 20-artist universal basis **0.633** on held-out artists. A basis is
**depth-baked** (module names carry the block index) and `load_basis` refuses a
depth mismatch. Whether any of this survives training is `docs/proposal/grad_basis_init.md` §E1.

## Origin

The line came from StelLA (NeurIPS 2025): its three-factor `USVᵀ` was the repo's
OrthoInit parameterization `ΔW = s·P·diag(λ)·Q` (since removed along with the rest
of the OrthoLoRA/OrthoHydra family), and its Table 5 ablation (the SVD
seed washes out once the subspace is trainable) is the question SVD-Down answers for
free LoRA — keep the principal input basis, drop the manifold constraint and the
paired-dyad cold start. The retired ChimeraHydra counterpart proposal is archived at
`_archive/docs/proposal/stella_chimera.md`.

## References

- Li et al., [StelLA: Subspace Learning in Low-rank Adaptation using Stiefel
  Manifold](https://arxiv.org/abs/2510.01938), NeurIPS 2025 (Spotlight) —
  origin of this line. Code:
  <https://github.com/SonyResearch/stella>.
- Hu et al., [LoRA](https://arxiv.org/abs/2106.09685), 2021.
- Meng et al., [PiSSA](https://arxiv.org/abs/2404.02948), NeurIPS 2024 — also
  starts from principal components but residualizes the base weight (SVD-Down
  keeps `W₀` unchanged + `ΔW=0`).
- Paischer et al., [EVA](https://arxiv.org/abs/2410.07170), NeurIPS 2025 /
  Wang et al., [LoRA-GA](https://arxiv.org/abs/2407.05000), 2024 — data/gradient-
  informed bases; the principled next arm if weight-SVD fails on large domain
  shifts.
