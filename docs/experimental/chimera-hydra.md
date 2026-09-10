# ChimeraHydra — dual-A additive routing for timestep-aware MoE

> **Shipped configuration (2026-05, code consolidation).** The implementation
> now ships a single chimera variant: the content pool is routed by the
> network-level ContentRouter on pooled `crossattn_emb` (NOT the per-Linear
> `lx_c` softmax this doc's "Content half" section originally described), and
> both pools are always centered-gate. The per-Linear content router
> (`content_router_source="input"`), its `content_router_init_std` knob, and the
> non-centered path were removed; `content_router_source` / `chimera_centered_gate`
> are no longer config knobs. Checkpoints stamped with the retired per-Linear
> content router no longer load. The prose below describes the original design
> space — read the "Content half" / per-Linear router parts as historical
> motivation, not the current code.

## Why "chimera"

Two HydraLoRAs glued at the residual. Each half is a complete asymmetric LoRA in the sense of HydraLoRA (Tian et al., NeurIPS'24, [arXiv:2404.19245](https://arxiv.org/abs/2404.19245)) — one shared down-projection A, K B-heads, one router. The two halves differ only in what their router reads:

- Content half (HydraLoRA shape): `A_c x → K_c B_c[k] (A_c x)`, routed by a per-Linear router on pooled `lx_c` (the content-side rank-r latent). Same content router HydraLoRA's paper specifies.
- Frequency half (HydraLoRA shape): `A_f x → K_f B_f[j] (A_f x)`, routed by a single network-level router fed `concat(FEI(z_t), σ-features)`. The router shape and "frequency-energy indicator" feature lineage trace to FeRA (arXiv:[2511.17979](https://arxiv.org/abs/2511.17979)).

Pool outputs are added. No multiplicative gate, no σ-band overlap mask, no curriculum:
```
Δy = Σ_c π_c[c] · B_c[c] (A_c x · λ_c · mask_t(σ))     ◄ content half
   + Σ_f π_f[f] · B_f[f] (A_f x · λ_f)                   ◄ freq half
```

The asymmetric T-LoRA mask on the content half (full rank on the freq half) follows the TimeStep Master asymmetric-mixture pattern (Wang et al., arXiv:[2503.07416](https://arxiv.org/abs/2503.07416)) — a "core" expert pool stays always-on at full capacity (here: freq, which wants full rank for high-σ coarse refinement) while a "context" expert pool is rank-modulated by t (here: content, which is the layout/identity memorization risk surface).

Three papers, one cell:

| Paper | What ChimeraHydra borrows |
|---|---|
| HydraLoRA ([2404.19245](https://arxiv.org/abs/2404.19245)) — Tian et al., NeurIPS 2024 | The asymmetric `1A → many Bs + router` shape. ChimeraHydra runs two of these stacks per Linear with disjoint A's. |
| TimeStep Master ([2503.07416](https://arxiv.org/abs/2503.07416)) — Wang et al., 2025 | Asymmetric per-pool timestep treatment. The freq half is the always-on full-rank "core"; the content half is the rank-masked "context" branch. T-LoRA's rank schedule is the masking primitive. |
| FeRA ([2511.17979](https://arxiv.org/abs/2511.17979)) — 2025 | The frequency-energy indicator (FEI) routing input + the network-level soft frequency router that fuses freq-specific adapter experts. ChimeraHydra reuses the same router shape (Linear → SiLU → Linear → softmax/τ) and FEI-on-`z_t` input as Anima's existing FeRA implementation. |

> Supersedes the earlier single-A chimera (one shared A per Linear, SVD column space sliced into content + freq sub-spaces). The prior framing was structurally closer to "hydra with two routers" than "two hydras"; the dual-A move makes the chimera metaphor honest and upgrades pool orthogonality from output-side-only to both sides (see "Free orthogonality" below). Earlier `_chimera.safetensors` files stop loading by design — same precedent as the pre-three-axis adapter migrations ([`networks/CLAUDE.md`](../../networks/CLAUDE.md) §"Three-axis routing surface").
>
> Also supersedes the staged 2D design (multiplicative gate `g_c ⊙ g_t` + Phase 1/2/3 curriculum) preserved in git history. Staging existed to break gradient confounding in the multiplicative gate; additive composition removes the confounding directly.

## Quick start

```bash
make exp-chimera                                    # default: K_c=4, K_f=2, r=32
python tasks.py exp-chimera                         # cross-platform
make lora-gui GUI_PRESETS=chimera_hydra             # GUI-friendly variant config
make test                                           # inference against latest *_chimera.safetensors
```

`make exp-chimera` drives `configs/methods/chimera.toml`; `make lora-gui GUI_PRESETS=chimera_hydra` drives `configs/gui-methods/chimera_hydra.toml`.

## Per-Linear shape

```
Trainable per Linear:
  A_c = Cayley(S_q_c) · Q_basis_c          (r, in)            content down
  A_f = Cayley(S_q_f) · Q_basis_f          (r, in)            freq    down
  B_c[k] = P_bases_c[k] · Cayley(S_p_c[k]) (out, r)  k=0..K_c-1
  B_f[j] = P_bases_f[j] · Cayley(S_p_f[j]) (out, r)  j=0..K_f-1
  λ_c, λ_f                                 (1, r) each
  router (Linear r → K_c)                                     content router

Frozen buffers (SVD basis):
  Q_basis_c, Q_basis_f                     (r, in) each
  P_bases_c, P_bases_f                     (K_c, out, r), (K_f, out, r)

Network-level (single, shared across all chimera Linears):
  FreqRouter:  Linear(F_in → H) → SiLU → Linear(H → K_f) → softmax/τ
               F_in = fei_feature_dim + sigma_feature_dim
```

```
                            z_t  (B, C, H_lat, W_lat)
                                    │
                FrequencyEnergyIndicator (DoG pyramid → simplex)        (library/runtime/fei.py)
                                    │
                                FEI(z_t)
                                    │            σ ─► sinusoidal-features (t_embedder form)
                                    └─────────┬─────────┘
                                              │  concat
                                              ▼
                                FreqRouter (one per network)
                                              │
                                         π_f (B, K_f)
                                              │
        ╔═════════════════════════════════════╧═════════════════════════════════════╗
        ║            ChimeraHydra Linear m0         ║   ChimeraHydra Linear m1   …  ║
        ║  ┌──────────────────────────┐             ║   (same shape)                ║
        ║  │  content half             │             ║                               ║
        ║  │   x ──► A_c ──► lx_c     │             ║                               ║
        ║  │   pool(lx_c) ──► router_c ──► π_c (B,K_c) ║                            ║
        ║  │   Σ_c π_c[c] · B_c[c] (lx_c · λ_c · mask_t(σ))                         ║
        ║  ├──────────────────────────┤             ║                               ║
        ║  │  freq half                │             ║                               ║
        ║  │   x ──► A_f ──► lx_f     │             ║                               ║
        ║  │   π_f from FreqRouter (broadcast)      ║                               ║
        ║  │   Σ_f π_f[f] · B_f[f] (lx_f · λ_f)     ║                               ║
        ║  └──────────────────────────┘             ║                               ║
        ║  out = base(x) + Δy_c + Δy_f                                              ║
        ╚═══════════════════════════════════════════════════════════════════════════╝
```

## Free orthogonality on both sides

The base weight's SVD partitions cleanly between the two halves at init:

- Right-singular split. Top `2r` right-singular vectors of `W`: first `r` → `Q_basis_c`, next `r` → `Q_basis_f`. So `Q_basis_c.row_space ⊥ Q_basis_f.row_space` by SVD orthonormality. The two A's project `x` into disjoint subspaces.
- Left-singular split. Top `(K_c+K_f)·r` left-singular vectors: first `K_c·r` reshape into `P_bases_c (K_c, out, r)`, next `K_f·r` into `P_bases_f (K_f, out, r)`. So every `B_c[k].col_space ⊥ B_f[j].col_space` for all `k, j` — and within each pool, `B_c[k] ⊥ B_c[k']` for `k≠k'` (OrthoHydra's existing pool-internal orthogonality argument).

Pools cannot fight over the same singular directions on either side. Cayley rotates each within its assigned subspace — orthogonality is preserved through training, *not* learned. Validated on a smoke build: `|max(Q_c · Q_f^T)| ≈ 6e-4`, `|max(B_c.cols · B_f.cols)| ≈ 8e-4` (noise floor of `svd_lowrank(niter=2)`).

This is strictly stronger than the previous 1-A version, which gave only output-side orthogonality (B-pool subspaces) while sharing one A.

Narrow-layer fallback. When `min(out, in) < max(K_c+K_f, 2)·r` (rare on Anima's mlp.layer1/2 targets — `in≈3072`, `(K_c+K_f)·r = 192` at default), both pools fall back to replicating the top-r singular slice. Pool-orthogonality is lost; pools start identical and rely on Cayley divergence.

### OrthoInit option (`use_ortho_init=true`)

Cayley keeps `colspace(ΔW) ⊆ top-2r(W₀)` for the whole run — the "chimera-ortho feels weak" cap (see [[project_orthoinit_variant]]). Setting `use_ortho_init=true` alongside the chimera flag flips each pool's `Q_basis_{c,f}` / `P_bases_{c,f}` from frozen buffers + Cayley skew to trainable fp32 Parameters (no `S_*`, no `_eye_r`, fp32 master kept for Adam). ΔW can then leave the principal 2r-subspace — full LoRA expressivity per pool — while keeping the W₀-aligned warm start. Pool orthogonality holds at init (same SVD partition above) and is free to drift thereafter; this is the per-pool trainable-basis route, not a joint Stiefel on `[P_c|P_f]`.

`ΔW=0` at init is unaffected — it comes from the centered uniform gate (`π − 1/K`), not the basis, so `λ_init > 0` is fine. Save distills with `R = I` to the identical `*_chimera.safetensors` free-form layout (the save discriminator keys on `.Q_basis_c`, covering both parameterizations), so inference / loading / merge / ComfyUI need no OrthoInit awareness. Implemented as a flag-branch inside `ChimeraHydraLoRAModule` (forward, `__init__`, `distill_save_state_dict`); threaded `resolver → chimera_hydra spec → cfg.use_ortho_init → network.py`. Tests: `tests/test_lora_dtype_policy.py::test_chimera_ortho_init_*`.

### Per-expert capability levers (frozen-Cayley path)

OrthoInit fixes "feels weak" by freeing the bases — but free bases let experts drift into the same subspace under the shared denoise loss ("averaged network", observed collapsing ~4k steps). These two knobs deepen each expert while keeping the frozen-disjoint cage, so cross-expert orthogonality stays structurally invariant and experts cannot collapse together. Both require `use_ortho_init=false` and distill into the standard `(K, out, r)` up-stack — inference / on-disk layout unchanged.

- `chimera_expert_basis_mult` (m ≥ 1) — each expert gets an over-complete `(out, m·r)` frozen pool carved from a disjoint U-slice plus an `m·r` Cayley rotation; the forward selects the first `r` columns of the rotated pool, a Stiefel(m·r, r) point. Unlike the canonical case (where the `r×r` rotation is a no-op on the span), the rotation now genuinely moves the expert's r-dim colspace within its private m·r pool — the subspace itself trains, but only inside the disjoint slice. Per-layer auto-downgrade to `M=r` when `(K_c+K_f)·m·r` overflows a Linear's width (logged). `m=1` reproduces the canonical r-slice bit-for-bit. The A's solve at size `r`, the B-pools at size `M`, in two batched Cayley calls (one when `M=r`).
- `chimera_expert_diag` — a per-expert `(K, r)` trainable diagonal σ (init 1) folded into the up-projection: the learnable singular spectrum the orthogonal-only frozen path lacks (its intrinsic singular values are pinned to 1). `ΔW=0` at init still holds via the centered gate regardless of σ.

`bench/chimera/run_bench.py expert_capacity` fits an off-r-slice target: subspace freedom (`m=2`) is the dominant lever (~×700 reach vs the r-slice baseline), the diagonal a minor ~×1.15, with cross-expert column cosine ≈ 0 throughout (no collapse). Invariants: `tests/test_chimera_expert_capacity.py` (ΔW=0 at init, distill round-trip faithful, cross-expert orthogonality preserved, default path unchanged).

## T-LoRA per-half composition

`use_timestep_mask = true` applies the rank mask `mask_t(σ)` to the content half only. The freq half keeps full rank at every t.

```
content : lx_c = (A_c x) · λ_c · mask_t(σ)     ◄ rank-modulated by t
freq    : lx_f = (A_f x) · λ_f                  ◄ always full rank
```

This is the TimeStep Master "core/context" asymmetric-mixture split applied per-pool: the freq pool's job is coarse-stage / high-σ refinement, which T-LoRA's argument says wants full rank; the content pool is the layout/identity memorization risk surface, which T-LoRA's mask is designed to prevent. The asymmetry falls out cleanly from the dual-A structure — no shared latent to coordinate.

## Where it sits in the three-axis matrix

ChimeraHydra is a fourth dispatch cell on top of the shared-A row, opt-in via `use_chimera_hydra=true`:

| Variant | `use_moe_style` | `route_per_layer` | `router_source` | Extra |
|---|---|---|---|---|
| Plain LoRA / OrthoLoRA / T-LoRA | `False` | — | `"none"` | — |
| HydraLoRA (paper) | `"shared_A"` | `True` | `"input"` | — |
| σ-router on Hydra | `"shared_A"` | `True` | `"sigma"` | — |
| FEI-on-Hydra (`lora.toml` default) | `"shared_A"` | `True` | `"fei"` | — |
| FeRA (independent-A) | `"independent_A"` | `False` | `"fei"` | — |
| ChimeraHydra (this doc) | `"shared_A"` | `True` | `"input"` | `use_chimera_hydra=true` |

`LoRANetworkCfg.from_kwargs` pins the three-axis fields to `("shared_A", True, "input")` whenever `use_chimera_hydra=true` — chimera is the only routing knob you set, and its dual-pool/dual-A nature is treated as an extension of the shared-A row.

## Implementation map

| File | Role |
|------|------|
| `networks/lora_modules/chimera.py` | `ChimeraHydraLoRAModule` (training): owns `(Q_basis_c, S_q_c)` + `(Q_basis_f, S_q_f)` + `(P_bases_c, S_p_c)` + `(P_bases_f, S_p_f)` + `lambda_c` + `lambda_f` + content `router` (Linear `r → K_c`) + `_freq_routing_weights` buffer. One batched `(2 + K_c + K_f, r, r)` Cayley solve per forward. `ChimeraHydraInferenceModule`: free-form (`lora_down_c`, `lora_up_c_weight`, `lora_down_f`, `lora_up_f_weight`, `router`) twin built at load from a distilled chimera checkpoint. |
| `networks/lora_anima/network.py` | `FreqRouter` (network-level), `_wire_shared_freq_routing_buffers` aliases every chimera module's freq buffer to one tensor; `set_fei` fires the router and broadcasts `π_f` via direct slot assignment (preserves grad_fn — `∂L_denoise/∂π_f → FreqRouter params`). `_get_chimera_balance_loss` splits each module's gate at `K_c` into independent Switch losses weighted by `_balance_w_content` / `_balance_w_freq`. Module construction passes `num_experts_content` + `num_experts_freq` into both chimera classes. |
| `networks/lora_anima/config.py` | `LoRANetworkCfg.use_chimera_hydra` / `num_experts_content` / `num_experts_freq` / `balance_w_content` / `balance_w_freq` / `freq_router_init_std` / `content_router_lr_scale` / `freq_router_lr_scale`. `from_kwargs` pins the three-axis fields when chimera is on; `from_weights` reconsumes the chimera-specific metadata stamps. |
| `networks/lora_anima/factory.py` | `create_network` builds the chimera spec via `resolve_network_spec` (`ChimeraHydraLoRAModule` for training); `create_network_from_weights` detects `ss_use_chimera_hydra="true"` + sniffs `.lora_up_c_weight` / `.lora_up_f_weight` keys to confirm dual-A format, then keeps the `chimera_hydra` spec but overrides `module_class = ChimeraHydraInferenceModule`. Surfaces the chimera-specific σ/FEI dims into the cfg slots the FreqRouter reads. |
| `networks/lora_anima/loading.py` | `_stack_chimera_lora_ups` folds per-expert `.lora_ups_c.{i}.weight` / `.lora_ups_f.{j}.weight` into stacked Parameters; `_refuse_split_chimera_keys` undoes the per-pool q/k/v split. |
| `networks/lora_save.py` | `_convert_chimera_dual_a_to_hydra` distills both pools' Cayley layout to free-form (`lora_down_{c,f}.weight` + `lora_up_{c,f}_weight`); `_build_chimera_moe_state_dict` expands to per-expert `lora_ups_{c,f}.{i}.weight` + per-pool q/k/v defuse + writes `*_chimera.safetensors`. Top-level `freq_router.*` passes through both steps. |
| `networks/__init__.py` | `NETWORK_REGISTRY["chimera_hydra"]` with `save_variant="chimera_hydra_moe"`. `_post_init_hydra` stamps `_use_chimera_hydra` + per-pool balance weights on the network. |
| `library/inference/models.py` | `_is_chimera_moe(path)` peeks `ss_use_chimera_hydra` from safetensors metadata. Chimera files take the existing Hydra-mode dynamic-hook branch but skip the `lora_unet_*` filter (so top-level `freq_router.*` survives) and pass `file=path` to `create_network_from_weights`. |
| `library/training/forward/router_conditioning.py` | Routes `set_sigma` → `set_fei` once per step. Chimera force-enables `use_fei_router` so the FEI/σ pipeline fires every step regardless of `cfg.router_source`. |
| `configs/methods/chimera.toml` | Method config driving `make exp-chimera`. |
| `configs/gui-methods/chimera_hydra.toml` | Self-contained variant config driving `make lora-gui GUI_PRESETS=chimera_hydra`. |
| `scripts/experimental_tasks/training.py::cmd_chimera` | `exp-chimera` shim. |

## Parameter count

Per adapted Linear at default `network_dim=32, K_c=4, K_f=2`:

```
S_q_c, S_q_f       (r, r) each            = 2 · 1024              ≈ 2.0k
S_p_c, S_p_f       (K_*, r, r) each       = 4·1024 + 2·1024       ≈ 6.1k
λ_c, λ_f           (1, r) each            = 2 · 32                = 64
content router     (K_c, r) + (K_c,)      = 4·32 + 4              = 132
Q_basis_c, Q_basis_f, P_bases_c, P_bases_f  — frozen buffers, not counted
```

≈ 8.3k trainable params per chimera Linear. ~+1k vs the previous 1-A version (extra `S_q_f` + extra `λ_f`); the 2× increase in down-projection capacity is in the frozen `Q_basis_*` buffers, which don't enter the optimizer.

Network-level FreqRouter:

```
Linear(F_in → H) + Linear(H → K_f) = (2 + 0) · 64 + 64 · 2 + biases  ≈ 0.4k
```

Negligible; one router serves the whole network.

Total at default `chimera.toml` regex (`*mlp.layer[12]`) on Anima's 28 blocks × 2 MLP layers = ~56 chimera modules × 8.3k ≈ 0.47M trainable params + 0.4k for the freq router. Roughly on par with the existing FEI-on-Hydra default; ~3× smaller than FeRA-style independent-A on the same regex.

## Knobs (`configs/methods/chimera.toml`)

| Param | Default | Notes |
|---|---|---|
| `use_chimera_hydra` | `true` | Activation flag. Triggers all three-axis pins + builds the FreqRouter. |
| `freq_router_mode` | `"fei"` | Freq-pool routing. `"fei"` (default) hardwires `π_f = normalize(FEI**(1/τ))` — no learned router, no σ-features, no freq balance loss; requires `K_f == fei_feature_dim`. `"learned"` builds the FreqRouter MLP over `concat(FEI, σ-features)`. See [§Freq routing mode](#freq-routing-mode-learned-vs-hardwired-fei). |
| `freq_router_tau` | `1.0` | Temperature on the hardwired FEI gate (`"fei"` mode only). `1.0` = raw FEI passthrough; `<1` sharpens the low/high crossover, `>1` flattens it. Inert under `"learned"`. |
| `num_experts_content` (`K_c`) | 4 | Content pool size. |
| `num_experts_freq` (`K_f`) | 2 | Freq pool size. Both pools share `network_dim` for now (per-pool rank knob is intentionally unexposed — add if a bench shows asymmetry helps). |
| `network_dim` (rank) | 32 | Per-pool rank `r` (same for both halves). SVD partition needs `min(out, in) ≥ max(K_c+K_f, 2)·r` for free orthogonality; at default that's 192, well below typical Anima MLP `in≈3072`. |
| `network_alpha` | 32 | LoRA scale `α / r = 1`. |
| `balance_loss_weight` | `1.0` | Outer balance multiplier (warmup-gated). At 1.0 the per-pool weights below are the only effective scalars. |
| `balance_w_content` | `2e-7` | `w_c` in `L_balance = w_c · switch(π_c) + w_f · switch(π_f)`. Within the OrthoHydra production safe range (`[[project_hydra_balance_weight_ceiling]]`). |
| `balance_w_freq` | `0` | Default 0 — the freq router's K_f=2 default has very little symmetry to break (binary gate); raise to ~`1e-6` if K_f≥3 and the freq pool stays uniform. |
| `fei_feature_dim` | `2` | FEI simplex bands (`e_low, e_high`). Same as FEI-on-Hydra default. |
| `fei_sigma_low_div` | `4.0` | `σ_low = min(H_lat, W_lat) / div` for the DoG kernel. 2026-05-13 dataset sweep picked 4 over 8 (`[[project_fera_probe_2band_decision]]`). |
| `sigma_feature_dim` | `0` | Width of sinusoidal-σ slice fed to the FreqRouter. Off by default — FreqRouter sees FEI(2) only, since FEI is itself a function of `z_t` and gives the router σ-correlated signal indirectly. Re-enable (`8` / `16` / `32`) if freq-pool entropy stays pinned at uniform after warmup. |
| `freq_router_init_std` | `0.1` | `N(0, std)` on the FreqRouter's output Linear. Non-zero is load-bearing — zero-init would make the freq pool a fixed point of the additive composition. See `chimera.py::FreqRouter` docstring. |
| `router_hidden_dim` | `64` | FreqRouter MLP hidden width. Shared with `GlobalRouter` (FeRA). |
| `router_tau` | `0.7` | Softmax temperature on FreqRouter output. Lower → sharper freq specialization. |
| `network_content_router_lr_scale` | `10` | Multiplier on `unet_lr × router_lr_scale` for the per-Linear content router. The std=0.01 init can take many steps to leave symmetry — bumping to 5–10× is a faster lever than raising `balance_w_content`. |
| `network_freq_router_lr_scale` | `1.0` | Multiplier for the FreqRouter. Independent of `content_router_lr_scale`. |
| `use_ortho` | `true` | Cayley-rotated SVD basis. Implicit in the chimera class; this flag governs the unrouted-fallback Linears (router_targets-excluded → OrthoLoRA at training; saved as plain LoRA after distill). |
| `use_timestep_mask` | `true` | T-LoRA. Applied to the content half only inside the chimera forward. |
| `min_rank` | `8` | T-LoRA floor — content half retains at least this many ranks at every t. |
| `router_targets` | `.*(mlp\\.layer[12])$` | Regex matching which Linears become chimera leaves. Non-matching layers fall back to OrthoLoRA at training, plain LoRA at inference (after the OrthoLoRA → LoRA save-time distill). |

## Freq routing mode: learned vs hardwired FEI

`freq_router_mode` controls how the freq pool's gate `π_f` is produced. The default is now `"fei"` (hardwired), with `"learned"` (the original FreqRouter MLP) kept as a fallback.

Why hardwired is the default. FEI (`library/runtime/fei.py::compute_fei_2band`) already returns a normalized 2-simplex `(e_low, e_high)`, and the freq pool is `K_f = 2`. So FEI is literally a routing distribution over `{low-expert, high-expert}` — the learned MLP was mapping a 2-simplex back onto a 2-simplex.

The archived FEI trace (`_archive/bench/fera/results/20260515-1112-anima-base-v1.0-low_div4/fei_traces.png`, the `fei_sigma_low_div=4.0` run that matches production) settles the router-vs-hardwired question:

- At high σ (early steps) FEI is deterministic — `e_low ≈ 0.0002, std ≈ 3e-5` across 4 prompts × 2 seeds. All-high; nothing to learn.
- At low/mid σ FEI carries real per-prompt signal: at σ=0.5, prompt 0 sits at `e_low ≈ 0.08` while prompt 2 is at `e_low ≈ 0.36` — a 4.5× spread at the same timestep (`std ≈ 0.16` at σ=0.43). FEI is not σ in disguise in the back half of denoising.

Both facts favor hardwiring. The per-prompt signal lives in FEI itself, and FEI is recomputed per-sample from `z_t` each step, so `π_f = FEI` keeps 100% of that signal. The learned router's only added value was a reshape over inputs (`FEI + σ`) no richer than FEI — and σ is the non-discriminating half (identical across prompts at a given step). Hardwiring also makes `expert_0 ≡ low band, expert_1 ≡ high band` true by construction (no router collapse possible) and rebuts the "freq pool just duplicates T-LoRA's timestep schedule" concern — the per-prompt spread is exactly what a pure-`t` mask cannot see.

Tradeoffs of `"fei"`. (1) It locks `K_f` to the FEI band count (`num_experts_freq == fei_feature_dim`, enforced at config time) — a feature, since it forces the freq pool to be a genuine band-split. (2) It drops the freq balance loss (`balance_w_freq` is forced to 0 — a fixed-function gate has no router params to balance) and the σ-feature fusion. Expert utilization is then frequency-faithful rather than load-balanced — at high σ the gate is ~99% high-expert by design, and across a batch sampling all σ both experts still train.

Implementation. `freq_router_mode="fei"` leaves `network.freq_router = None`; `LoRANetwork.set_fei` broadcasts `_fei_temperature(FEI, τ)` straight to each module's `_freq_routing_weights` (detached — no grad_fn, like the T-LoRA mask). Stamped to `ss_chimera_freq_router_mode` / `ss_chimera_freq_router_tau`; an absent stamp loads as `"learned"`, so pre-2026-05-27 checkpoints are unaffected. This makes the `C-fei` falsification (§Risks) moot for the router itself — there is no router to absorb or miss the signal.

## Save format

`output/ckpt/<output_name>_chimera.safetensors` keys (`output_name = "anima_chimera"` by default):

```
# Network-level FreqRouter — fp32
freq_router.net.0.weight                              (router_hidden_dim, fei_feature_dim + sigma_feature_dim)
freq_router.net.0.bias                                (router_hidden_dim,)
freq_router.net.2.weight                              (num_experts_freq, router_hidden_dim)
freq_router.net.2.bias                                (num_experts_freq,)

# Per-adapted-Linear distilled chimera dual-A keys (q/k/v defused on attention prefixes)
lora_unet_<dotted_path>.lora_down_c.weight            (r, in)               content half
lora_unet_<dotted_path>.lora_ups_c.{0..K_c-1}.weight  (out, r)              content B's
lora_unet_<dotted_path>.lora_down_f.weight            (r, in)               freq half
lora_unet_<dotted_path>.lora_ups_f.{0..K_f-1}.weight  (out, r)              freq B's
lora_unet_<dotted_path>.router.weight                 (K_c, r)              content router
lora_unet_<dotted_path>.router.bias                   (K_c,)
lora_unet_<dotted_path>.alpha                         ()
```

The two halves live under one prefix (one chimera module per Linear); the `_c` / `_f` sub-key suffixes distinguish content vs freq. The freq half has no per-Linear router (uses the network-level FreqRouter at top-level `freq_router.*`).

Distilled at save, dual-A at load. Save runs `_convert_chimera_dual_a_to_hydra` to fold each pool's Cayley `(S_q_*, S_p_*, P_bases_*, Q_basis_*, lambda_*)` into per-pool free-form (`lora_down_*` + stacked `lora_up_*_weight`), then `_build_chimera_moe_state_dict` expands the stacks to per-expert `.lora_ups_*.{i}.weight` and per-pool defuses fused `qkv_proj` / `kv_proj` prefixes (cloning `router.*` / `alpha` / `inv_scale` into each q/k/v split).

Load (`library/inference/models.py::_is_chimera_moe`) sniffs `ss_use_chimera_hydra="true"` from metadata, then `factory.create_network_from_weights` confirms by checking for `.lora_up_c_weight` / `.lora_up_f_weight` keys (post-stack form), keeps the chimera_hydra spec, and overrides `module_class = ChimeraHydraInferenceModule`. The Cayley training class is therefore training-only — checkpoint resume silently drops the orthogonal parameterization and continues on the free-form inference class (matches the OrthoHydra → Hydra precedent).

Metadata stamps:

```
ss_network_spec                = "chimera_hydra"
ss_use_chimera_hydra           = "true"
ss_num_experts_content         = "4"
ss_num_experts_freq            = "2"
ss_chimera_fei_feature_dim     = "2"
ss_chimera_sigma_feature_dim   = "0"     # currently off; "16" in the GUI variant
ss_chimera_fei_sigma_low_div   = "4.0"
ss_use_moe_style               = "shared_A"
ss_route_per_layer             = "true"
ss_router_source               = "input"
```

## Compatibility

| Component | Compat | Notes |
|---|---|---|
| Training loop | ✅ | `apply_router_conditioning` fires `set_sigma` → `set_fei` every step; chimera force-enables `use_fei_router` so the FEI/σ pipeline runs unconditionally. The FreqRouter executes with grad so `∂L_denoise/∂π_f` reaches its parameters via the slot-assigned `_freq_routing_weights` buffer (same contract as FeRA's GlobalRouter). |
| Standard inference | ✅ | `library/inference/models.py::_is_chimera_moe` routes the file through the Hydra-mode dynamic-hook path; the factory rebuilds `ChimeraHydraInferenceModule` + `FreqRouter`; `library/inference/adapters.py::compute_and_set_hydra_fei` fires `set_fei` per Euler step. |
| Spectrum inference | ⚠ | Per-step `set_fei` is wired, but on a Spectrum cached step the gate is updated while the cached features may have been forecast from a different gate distribution. Bench against `--spectrum` before relying on it. Same caveat as FeRA. |
| `torch.compile` | ✅ | Two down-projects + one batched Cayley solve + two bmm. Shape-static under constant-token bucketing. The forward issues both pool bmms regardless of mask state so dynamo doesn't recompile at T-LoRA flip points. |
| `blocks_to_swap` | ✅ | Each chimera module replaces its base Linear in-place. The network-level FreqRouter stays on the main device. |
| `gradient_checkpointing` | ✅ | The adapter is a thin Linear-replacement; checkpointing at block granularity wraps it correctly. |
| Modulation guidance | ✅ orthogonal | AdaLN path is untouched. |
| T-LoRA | ✅ | Built-in (per-half asymmetric masking — content rank-modulated, freq full-rank). |
| OrthoLoRA | ✅ | `use_ortho=true` is the chimera default. |
| ComfyUI | ❌ | The 2-A on-disk layout (`lora_ups_c.{i}` + `lora_ups_f.{j}` + dual `lora_down_{c,f}` per Linear) is NOT what the `comfyui-hydralora` node currently understands (it expects the legacy 1-A Hydra-MoE shape). The legacy chimera node-loader tests (removed with the extracted node) exercised the legacy synthetic layout, not the new emitter. ComfyUI loader needs ~150 lines of new code to read the 2-A keys + broadcast `π_f` per step. |
| Static merge into DiT | ❌ | `scripts/toolkits/merge_to_dit.py` refuses MoE methods by default (router is sample-dependent). `--allow-partial` would drop the chimera portion entirely. |
| FeRA / hydra-moe loaded simultaneously | ❌ | One router scheme per checkpoint; `models.py` refuses two MoE files in one `--lora_weight` list. |

## Cold-start risk and diagnostics

Two routers init random ⇒ risk one pool dominates while the other settles at uniform (local minimum). Three structural mitigations are built in:

1. Per-pool balance loss (`w_c`, `w_f` independent). A single combined balance term would let the optimizer satisfy the constraint by flattening one pool to uniform while concentrating the other.
2. Non-zero FreqRouter init (`freq_router_init_std=0.1`). Output is near-uniform but *not at* uniform at step 0; the freq router immediately differentiates as FEI/σ vary across the batch.
3. SVD partition is asymmetric-by-construction. Each pool starts in a different singular subspace of the base weight on both A and B sides — pools cannot collapse onto identical projections even at step 0.

Live diagnostic: watch per-pool `Σ‖π[k] − 1/K‖²` in the first 1k steps. If the freq pool stays < 1e-3 (flat-uniform) while content diverges, raise `balance_w_freq` or sweep `freq_router_init_std`.

Persistent freq-pool flatness after warmup ⇒ the freq router has no signal the content router didn't already capture via `lx_c`-σ correlation. That's the redundancy failure mode (proposal §Risks #1 — bench plan calls out a `C-fei` falsification cell that feeds FEI into the content router; if `C-fei ≈ ChimeraHydra`, the freq pool is redundant).

## What to measure

ChimeraHydra's bet: dual-A + structurally-enforced router-input separation makes the freq pool learn `σ`-aware refinement the content router can't, *without* phased training. The whole point hinges on whether (a) the freq pool actually trains and (b) it picks up signal the single-router OrthoHydra was leaving on the table.

1. Per-pool gate entropy + divergence. Median across chimera Linears. Both pools should diverge from uniform after warmup; freq pool diverging on σ-buckets is the load-bearing signal.
2. Freq-gate variance across σ buckets at inference. Bin σ ∈ [0,1] into 3–5 buckets; for each bucket, log mean π_f. Variance across buckets > 0.01 → freq router is using σ. Below floor ⇒ freq pool is dead weight.
3. Per-half contribution norms. `‖Δy_c‖` vs. `‖Δy_f‖` per Linear, averaged over the dataset. Healthy ratio ~0.3–3.0; out-of-range = one half dominating.
4. Per-expert usage histograms per pool. Argmax frequency across the K_c content experts and the K_f freq experts. Flat-ish distributions in each pool are the success case; one column near zero = collapse.
5. A/B vs single-A chimera at matched E. Same dataset, matched epochs/lr, same K_c+K_f. Whichever wins on CMMD + sample quality tells us whether the dual-A structural separation buys anything over the previous single-A version.
6. A/B vs FEI-on-Hydra at matched E. Same dataset, `num_experts = K_c + K_f`. If FEI-on-Hydra matches chimera, the freq pool is redundant and dual-A should be archived.
7. C-fei falsification. Feed FEI into the content router (one cfg toggle on a separate run): if results match chimera, the freq pool is redundant and the dual-A design should be archived.
8. Sample quality vs `make lora`. CMMD ([[project_cmmd_val_signal]]) is the primary signal; FM val-MSE is uninformative on Anima ([[project_fm_val_loss_uninformative]]).

## Hyperparameters worth sweeping

| Knob | Default | Range to try | Why |
|---|---|---|---|
| `num_experts_content` (`K_c`) | 4 | 2, 3, 4, 5 | Content capacity; chimera defaults skew toward content because text-conditional routing has more dimensions to partition. |
| `num_experts_freq` (`K_f`) | 2 | 2, 3, 4 | FEI-on-Anima is bimodal ([[project_fera_probe_2band_decision]]); K_f=3 has historically collapsed back to K_f=2. |
| `balance_w_content` | `2e-7` | `1e-7` … `1e-5` | Same Pareto region as ortho-hydra ([[project_hydra_balance_weight_ceiling]]). |
| `balance_w_freq` | `0` | `0`, `1e-7`, `1e-6` | Default 0 — at K_f=2 the binary gate has little symmetry to break. Raise if K_f≥3 and freq stays uniform. |
| `freq_router_init_std` | `0.1` | `0.05`, `0.1`, `0.3` | Higher → freq pool starts further from uniform but signal-to-noise drops. Never zero (fixed point). |
| `router_tau` (FreqRouter) | `0.7` | `0.3`, `0.7`, `1.0`, `2.0` | Lower τ → sharper freq specialization, more sensitive to FEI noise. |
| `sigma_feature_dim` | `0` | `0`, `8`, `16`, `32` | Currently off — FreqRouter sees FEI only. Re-enable (and sweep) if freq-pool entropy stays pinned at uniform after warmup, suggesting FEI(2) isn't a wide enough input. |
| `fei_sigma_low_div` | `4.0` | `2`, `4`, `8` | Same Pareto region as FEI-on-Hydra; 4 picked by 2026-05-13 dataset sweep. |
| `network_dim` | 32 | 16, 32, 64 | Per-pool rank `r`. Slice width per expert = `min(out, in) / (K_c + K_f)` for the SVD partition; at `r=16, K_c+K_f=6` slices get narrow — verify expressivity vs single-A at matched total rank. |
| `multiplier` (inference) | 1.0 | 0.0, 0.5, 1.0, 1.5 | `0.0` short-circuits to frozen base for clean ablation. |

## Files

- [`networks/lora_modules/chimera.py`](../../networks/lora_modules/chimera.py) — `ChimeraHydraLoRAModule` (training) + `ChimeraHydraInferenceModule` (load-time free-form twin).
- [`networks/lora_anima/network.py`](../../networks/lora_anima/network.py) — `FreqRouter`, `_wire_shared_freq_routing_buffers`, `set_freq_routing_weights`, `_get_chimera_balance_loss`, FreqRouter param group.
- [`networks/lora_anima/loading.py`](../../networks/lora_anima/loading.py) — `_stack_chimera_lora_ups`, `_refuse_split_chimera_keys`.
- [`networks/lora_anima/config.py`](../../networks/lora_anima/config.py) — chimera cfg fields + three-axis pin.
- [`networks/lora_anima/factory.py`](../../networks/lora_anima/factory.py) — chimera dual-A detect at load + class swap.
- [`networks/lora_save.py`](../../networks/lora_save.py) — `_convert_chimera_dual_a_to_hydra`, `_build_chimera_moe_state_dict`.
- [`networks/__init__.py`](../../networks/__init__.py) — `NETWORK_REGISTRY["chimera_hydra"]`, `resolve_network_spec` dispatch, `_post_init_hydra` per-pool stamping.
- [`configs/methods/chimera.toml`](../../configs/methods/chimera.toml) — canonical method config (`make exp-chimera`).
- [`configs/gui-methods/chimera_hydra.toml`](../../configs/gui-methods/chimera_hydra.toml) — GUI-friendly variant config.
- [`scripts/experimental_tasks/training.py`](../../scripts/experimental_tasks/training.py) — `cmd_chimera` shim.
- [`_archive/proposals/chimera_hydra.md`](../proposal/chimera_hydra.md) — design rationale, bench plan, decision tree, risks.

## Status

Experimental. Dual-A code lands and round-trip is verified (max abs diff ~9e-3 from bf16 vs fp32 precision in train/inference paths; structural orthogonality confirmed at 6e-4 / 8e-4 on both sides). No bench results yet. Existing 1-A chimera checkpoints do not load — retrain to produce the dual-A format.

ComfyUI mirror needs ~150 lines of new node-side code to handle the 2-A on-disk layout + broadcast `π_f` per step. The synthetic-data node-loader tests (since removed) covered the legacy 1-A loader path and need a parallel 2-A test set when the node is updated.

The proposal's bench plan (cells A / B / C / C+T / C-split / C-fei) plus a new dual-A vs single-A A/B is the prerequisite before promoting chimera to a default LoRA-family variant.

See [`_archive/proposals/chimera_hydra.md`](../proposal/chimera_hydra.md) §"Decision tree" for the ship/archive criteria after bench results land.
