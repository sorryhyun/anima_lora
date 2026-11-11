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

At init the output is zero, step 1 puts gradient in `lora_up` only, and the
step-1 `‖ΔW‖_F` is ≈1.00× plain LoRA's; update alignment in the W₀-aligned
regime is `cos_ideal` 0.42 vs plain LoRA's 0.14, with no change in the isotropic
regime (`bench/turbo/results/20260621-2149-svd-down-phase0-clean/`).

## Scope

- **Plain LoRA only**, Linear layers only. Conv2d keeps Kaiming. The
  config resolver rejects a non-Kaiming `down_init` combined with the Hydra
  MoE path — it carries its own basis parameterization.
- Composes with T-LoRA (the `_timestep_mask` acts on the bottleneck after
  `lora_down`) and with channel-scaling (absorption runs after init, as for
  Kaiming).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `down_init` | `"kaiming"` | `"kaiming"` (default `kaiming_uniform_(a=√5)`), `"weight_svd"` (SVD-Down, this doc), `"weight_svd_nodc"` (below), or the gradient-seeded `"grad_svd"` / `"basis_file"` (see below) |
| `grad_basis_file` | — | Network arg: path to a gradient-basis artifact. Required by `down_init="basis_file"`; written automatically by `"grad_svd"`. |
| `svd_slice` | `0` | `weight_svd` only: seed from right singular vectors `[k·r, (k+1)·r)` instead of the top-r. Slices of one orthonormal basis are mutually orthogonal, so adapters trained with different `k` never share an input subspace at merge — a per-artist address. The window must fit every targeted layer (`(k+1)·r ≤ min(W.shape)`; the base DiT's 256-row adaln `.1` Linears cap r=32 at `k ≤ 7`) or the init refuses. Stamped as `ss_svd_slice`. Merge reads: `_archive/bench/merge_basis/README.md`. |
| `svd_slice_count` | `0` | `> 0` interleaves the slices: slice `k` takes vectors `{k, k+N, k+2N, …}` instead of `[k·r, (k+1)·r)`, so every slice owns one of the top-N directions. Contiguous slices hand slice 0 the high-energy top of the spectrum (on the base DiT, self_attn q/k and mlp.layer1 inputs put 1.6–2.4× more energy in slice 0 than in any other slice). Needs `N·r ≤ min(W.shape)`; requires `svd_slice < N`. Stamped as `ss_svd_slice_count`. |

**`weight_svd_nodc`** takes the basis of `W(I − ûûᵀ)` instead, `û = x̄/‖x̄‖` the calibrated signed input mean (`networks/calibration/channel_mean.safetensors`, `analyze_lora_input_channels.py --dump_channel_mean`), so every row starts ⊥ x̄ and `lora_down·x̄ = 0`. On the base DiT the loud input channels are the DC (x̄/mean|x| ≈ ±1 on them), and `weight_svd` puts 66 % of the bottleneck energy `h = A₀x` on it (kaiming 31 %), which makes 78 % of the first `lora_up` gradient the token-uniform term `(Σg)h̄ᵀ`. After a 256-step run `‖down·û‖/‖down‖` is 0.013 against 0.079 for `weight_svd`. adaln keeps `weight_svd` (its σ-only input is almost all mean); `svd_slice` applies as for `weight_svd`. The same init exists on MOSS (`docs/moss/findings/t2i_lora_recipe.md` §3). Blind A/B: s43, `docs/anima/findings/lora_recipe.md`.

**adaln is never sliced.** `adaln_up_*` modules take the top-r whatever `svd_slice` / `svd_slice_count` say. Their input is a σ-only vector that is nearly constant over σ ∈ [0, 1] (cos ≥ 0.99 to its mean): 73% of its energy sits on W0's first right singular vector and 19% on the second. Under contiguous slices slice 0 owns 99.2% of that energy, so only that shard's adaln learns anything (in an 8-shard soup its delta was ~9× each other shard's). No orthogonal layout can split a ~2-D input, so the shards share one init and `merge_loras.py` averages their adaln deltas (`--adaln average`, the default) instead of summing them.

## Implementation

| File | Role |
|------|------|
| `networks/lora_modules/lora.py` | `_top_right_singular_vectors()` — the exact basis; `_init_down_weight_svd()` copies `V_rᵀ/√3` into `lora_down` |
| `networks/lora_anima/config.py` | `down_init` cfg field + validation (Linear/plain-LoRA-only guard) |
| `networks/grad_basis.py` | The gradient-seeded siblings: sketch, basis artifact I/O, and the shared `V_rᵀ/√3` copy |

The basis is computed per adapted Linear at init, from an **eigendecomposition of
the smaller Gram matrix** rather than `torch.linalg.svd` — the same subspace,
7.5× cheaper on the base DiT (54.2 s → 7.2 s for all 448 `blocks.*` Linears at
r=32, 5070 Ti) and with tighter-orthonormal columns than cuSOLVER's Jacobi SVD
(1e-6 vs 1e-3 max off-diagonal), which `svd_slice` relies on. Capture against
the exact V is 1.000 on every weight group.

Squaring the spectrum sinks the bottom of the requested window under the fp32
eigh error floor on ill-conditioned layers (the 1569:1 `(6144, 256)` Linears give
a wrong slice-7 window; for `in > out` the 3325:1 adaln `.1` Linears lose
orthogonality). Both branches read `λ_cols/λ_0` off the computed eigenvalues and
hand the layer to `linalg.svd` below `_GRAM_EIG_FLOOR` — those are the narrow,
cheap layers, ~1 s of the 7.2 s.

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
depth mismatch. After training, every `down_init` arm renders the same
(`bench/grad_init/README.md` §E1).

## References

- Li et al., [StelLA: Subspace Learning in Low-rank Adaptation using Stiefel
  Manifold](https://arxiv.org/abs/2510.01938), NeurIPS 2025 (Spotlight). Code:
  <https://github.com/SonyResearch/stella>.
- Hu et al., [LoRA](https://arxiv.org/abs/2106.09685), 2021.
- Meng et al., [PiSSA](https://arxiv.org/abs/2404.02948), NeurIPS 2024 — also
  starts from principal components but residualizes the base weight (SVD-Down
  keeps `W₀` unchanged + `ΔW=0`).
- Paischer et al., [EVA](https://arxiv.org/abs/2410.07170), NeurIPS 2025 /
  Wang et al., [LoRA-GA](https://arxiv.org/abs/2407.05000), 2024 — data/gradient-
  informed bases.
- Proposal and theory: `_archive/proposals/svd_down_lora_init.md`; gradient
  modes: `_archive/proposals/grad_basis_init.md`.
