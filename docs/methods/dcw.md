# DCW — Post-Step SNR-t Bias Correction

Training-free, sampler-level correction that closes the SNR-t bias of flow-matching DiTs by mixing each Euler step's `prev_sample` toward (or away from) the model's `x0_pred`.

Paper: [Elucidating the SNR-t Bias of Diffusion Probabilistic Models](https://arxiv.org/abs/2604.16044) (Yu et al., CVPR 2026)

**Read first:** `archive/dcw/findings.md`. The paper's bias direction does not reproduce on Anima — Anima's λ is **negative**, opposite the paper. Everything below assumes you've internalized that.

## Two modes

| Mode | What `λ` is | When to use |
|---|---|---|
| **scalar** (v0/v1) — `--dcw` | a single global constant tuned offline (default `−0.015`) | minimal/safe default; one-line ablation; fallback when v4 isn't calibrated |
| **v4 learnable** — `--dcw_v4 <artifact>` | a function of `(aspect, prompt, observed prefix gap)` produced at runtime by a small MLP | per-prompt amplitude + per-trajectory steering; trained per checkpoint via `make dcw` |

The math at the apply site is identical:

```
denoised   = latents − σ_i · v                       # x0_pred (FLUX velocity convention)
prev       = Euler/ER-SDE step                        # prev_sample
diff       = prev − denoised
diff_LL    = haar_idwt(LL(diff), 0, 0, 0)             # LL-only band mask
prev      += λ_i · diff_LL                            # DCW correction
```

What differs is how `λ_i` is produced. In scalar mode, `λ_i = λ · sched(σ_i)`. In v4, `λ_i = base + bucket_corr + α_eff · μ_g[i] / Σ_tail(μ_g·S_pop)` with the controller observing the first `k=7` step's LL gap before firing.

See `docs/proposal/dcw-learnable-calibrator-v4.md` for the full v4 derivation, gates, and fallback ladder.

## Quick start

```bash
make test-dcw                           # latest LoRA + scalar λ=−0.015 (defaults baked in)
make test-dcw-v4                        # latest LoRA + v4 controller (auto-resolves latest fusion_head)
make test-spectrum-dcw                  # Spectrum + scalar DCW composed
```

Scalar mode also works on any `inference.py` invocation:

```bash
python inference.py --dcw                                      \
    --dcw_lambda -0.015 --dcw_schedule one_minus_sigma         \
    --dcw_band_mask LL                                         \
    ...  # other inference args
```

v4 mode (auto-resolves the most-recent `fusion_head.safetensors` under `post_image_dataset/dcw/` first, then `bench/dcw/results/`):

```bash
python inference.py --dcw_v4 auto --dcw_v4_disable_shrinkage   \
    ...
```

The tqdm progress bar shows per-step `λ` (and `α` once the head fires) so the controller's trajectory is interpretable in real time.

## v4 architecture in one paragraph

Three input channels feed one shared MLP. (1) **Aspect prior** — per-bucket profile `(μ_g[i], S_pop[i], λ_scalar)` indexed by `(H, W)`. (2) **Prompt embedding** — mean-pooled `crossattn_emb` (`c_pool`, 1024-dim) plus auxiliary scalars `[caption_length, cos(c_pool, μ_centroid), token_l2_std]`. (3) **Observed prefix** — LL-band Haar norms of `noise_pred` over the first `k=7` denoising steps (free at inference; the post-CFG velocity is already computed). The MLP outputs `(α̂, log σ̂²)` at step `k`, then `α_eff` is distributed across the remaining `(N − k)` steps proportionally to `μ_g[i]`. Per-step `λ_i` is clamped to `±3·|λ_scalar[aspect]|` as an overshoot guard.

## Calibration: `make dcw`

```bash
make dcw                                # full calibration: ~3-5h on a 5060 Ti
make dcw --n_images 32 --n_seeds 2      # smaller pool, ~1h, lower σ̂² fidelity
make dcw-train                          # train-only on existing pool (~30s)
```

`make dcw` runs `scripts/dcw_measure_bias.py --dump_per_sample_gaps` against three aspect buckets (1024², 832×1248, 1248×832) at the production env (CFG=4, mod_w=3.0), then chains `scripts/dcw_train_fusion_head.py` on the pooled output. Defaults: 80 prompts × 3 seeds × 3 buckets. All outputs land in `post_image_dataset/dcw/<timestamp>-<label>/`; the trainer also reads from `bench/dcw/results/` so the legacy A2 calibration runs (S_pop, λ_scalar per bucket) and prototype trajectories continue to count.

End artifact: `<run>/fusion_head.safetensors` — single file, ~285k params + per-aspect bucket profile + standardization stats + metadata. `make test-dcw-v4` auto-resolves the newest by mtime across both roots.

## CLI

| Flag | Mode | Default | Notes |
|------|------|---------|-------|
| `--dcw` | scalar | off | Enable post-step correction with a constant `λ`. |
| `--dcw_lambda` | scalar | `-0.015` | Negative on Anima — see findings. Tuned for `--dcw_band_mask LL`; use `-0.010` if you switch to `all`. |
| `--dcw_schedule` | scalar | `one_minus_sigma` | One of `one_minus_sigma`, `sigma_i`, `const`, `none`. |
| `--dcw_band_mask` | scalar | `LL` | Haar subband mask: `LL`, `HH`, `LH+HL+HH`, `all`. LL-only is strictly better than `all` on Anima — see §LL-only correction. |
| `--dcw_v4` | v4 | unset | Path to `fusion_head.safetensors` (or directory containing one). When set, overrides scalar `--dcw_lambda` with per-step controller output. |
| `--dcw_v4_warmup_k` | v4 | (from artifact) | Override the warmup-k baked into the artifact metadata. |
| `--dcw_v4_disable_shrinkage` | v4 | off | Skip σ̂²-based shrinkage on `α̂`. **Recommended** while the prototype's σ̂² channel doesn't pass Gate B. |
| `--dcw_v4_disable_backstop` | v4 | off | Skip the caption-length backstop. Currently a no-op (`tau_short` not yet shipped in the artifact). |

The final step (`σ_{i+1} == 0`) is always skipped in both modes — at that step `prev == x0_pred` exactly, so DCW would be a numerical no-op.

## When to use which

Scalar DCW helps when the **target is detail-dense** (busy compositions, intricate textures, complex backgrounds) — the late-step bias correction tightens edges and recovers fine structure. It is **not helpful — and can hurt — when the target is intentionally simple** (e.g. the flat, minimal style of channel/caststation-class artists). On those, the correction over-sharpens what should be deliberately smooth, and the scalar baseline is preferable.

The v4 controller was designed to handle this prompt-dependent gap automatically — its `c_pool` channel + observed-prefix channel together can downweight `α_eff` on flat-style prompts (e.g. via the implicit α̂ prediction shrinking when `g_obs[0:k]` looks small). The caption-length backstop (`tau_short`) is a separate planned safety net for short prompts; not yet calibrated. Until Gate C (perceptual side-by-side, see proposal §"Quality gates") passes, scalar remains the safer default for production runs of unknown style.

## Composition

DCW lives at the sampler boundary, not inside any module — composes with everything below.

| Composes with | How |
|---|---|
| `--sampler er_sde` | Applied post-`er_sde.step`. |
| `--tiled_diffusion` | Applied to post-merge latents, not per-tile. **v4 controller currently no-ops in tiled mode** (single-tile assumption in `c_pool`/`g_obs`); scalar still works. |
| `--spectrum` | Applied at the same sampler-step site on both actual-forward and cached-step branches. On cached steps, `x0_pred = latents − σ_i · noise_pred` carries Spectrum's prediction error; correction is bias-agnostic so this is fine. v4 hasn't been ablated on cached steps. |
| `--lora_weight` / Hydra / OrthoLoRA / T-LoRA / ReFT / postfix | Orthogonal — no module patching, no extra weights. v4 calibrates against the base DiT by default; per-LoRA calibration is `make dcw --lora_weight <path>` (writes a separate artifact). |

Untested at v0:
- CFG ≠ 4 with v4 (the prototype was calibrated at CFG=4; falls back to bucket prior or scalar via the proposal's fallback ladder).
- APEX (APEX trains around the bias).
- Stacked LoRA / OrthoLoRA / T-LoRA / ReFT (one row per family).

## v4 status (prototype)

Trained on existing `bench/dcw/results/` data — 176 rows, 40 unique stems, 8-fold prompt-stratified CV. Headline gates from `bench/dcw/results/20260504-1831-v4-fusion-head-prototype/`:

| Metric | Threshold | Prototype |
|---|---|---|
| r(α̂_p, mean_s r) per-prompt | ≥ 0.6 | **+0.89** ✓ |
| r(α̂_p,s, r_p,s) seed-conditional | ≥ 0.7 | **+0.88** ✓ |
| r(σ̂_p, std_s r) | ≥ 0.4 | −0.01 ✗ |
| NLL improvement vs N(0, σ²_pop) | ≥ 15% | +5.7% ✗ |

α̂ channels pass strongly; σ̂² channel doesn't (under-supervised at one-seed-per-prompt-mostly data). **Ships with `--dcw_v4_disable_shrinkage` by default** until `make dcw`'s 3-seed pool reruns the gate. If σ̂² still fails after, the controller stays shrinkage-off in production — α̂ alone with the clamp guard is gate-passing.

**Tail-formula correction.** The proposal pseudocode's `head_corr = α_eff · μ_g[i] / tail_norm` mixes gap-units (α_eff is the integrated tail gap residual, not λ) with λ-units. The controller actually uses the LSQ-projected form `Δλ_i = −α_eff · μ_g[i] / Σ_{tail}(μ_g · S_pop)`, which matches the proposal's intent of distributing correction proportional to μ_g while preserving units.

**Output clamp.** Per-step `λ_i` is bounded at `±3 · |λ_scalar[aspect]|` as a safety guard. On the prototype's noisy α̂ this is currently binding on every tail step (visible in the tqdm trace as a flat tail). Once shrinkage is calibrated this clamp should rarely bind.

## Anima form details (kept for reference)

### Why λ < 0

Yu et al.'s Key Finding 2 (`||v_θ(x̂_t)|| > ||v_θ(x_t_fwd)||`) does **not** reproduce on Anima — the inequality is reversed at every late step, integrated signed gap −405.6 on the 24-step baseline. Paper-form positive λ widens `|gap|` on Anima; closing the gap requires negative λ. Speculative mechanism (manifold-mismatch readout) is in `archive/dcw/README.md §"Observed on Anima"`.

### Why `(1 − σ)` schedule (scalar mode)

The bias is concentrated at low σ on Anima — `gap` is small around σ=0.5 and grows to ≈−64 by σ=0.04. The paper's `σ_i` decay would put correction in the wrong place; `const` overcorrects mid-trajectory and sign-flips the gap by step 15 (visible as over-smoothing). `(1 − σ)` weights late steps heaviest, matches the bias envelope, and dominated the 8-prompt visual panel. v4 inherits the `(1−σ)·λ_scalar` envelope as `base_lambda` and adds bucket + head corrections on top.

### Scalar λ calibration (when to retune)

The default `λ=-0.015` was derived from two independent estimates that agreed (perceptual winner of a wide sweep + closed-form fit on a narrow sweep). To re-tune for a different checkpoint / CFG-on / on a LoRA stack:

1. `python scripts/dcw_measure_bias.py --dcw_sweep --dcw_scalers 0 -0.010 -0.020` (or any 3+ anchors).
2. Read `λ*` from the printed line — `(1−σ)`-weighted least-squares optimum:
   ```
   s_i  = ∂gap/∂λ                            (finite-diff from any 2 anchors)
   w_i  = (1 − σ_i)
   λ*   = − Σ w_i · g_i · s_i  /  Σ w_i · s_i²       (over i ≥ N/2)
   ```
3. Confirm with a tighter sweep `{λ*−ε, λ*, λ*+ε}`.

For v4 calibration, use `make dcw` instead — it produces the per-aspect bucket profile + fusion head in one go.

## LL-only correction (2026-05-03 finding)

`bench/dcw/results/20260503-2102-band-mask-eyeball/` ran a per-Haar-subband sweep on the same 4-image / 2-seed bench. Headline:

| Config | late-half integrated \|gap\| | Δ vs baseline | per-band signed gap (LL / LH / HL / HH) |
|---|---|---|---|
| baseline | 330.1 | — | −317 / −165 / −165 / −127 |
| **`λ=-0.01_one_minus_sigma_LL`** | **235.7** | **−28.6%** | **−225 / −120 / −122 / −92** *(all bands improved)* |
| `λ=-0.01_one_minus_sigma_all` | 340.6 | **+3.2%** | −180 / −240 / −242 / −222 *(LL improved, detail bands worsened)* |
| `λ=-0.01_one_minus_sigma_HH` | 363.6 | +10.2% | −300 / −146 / −147 / **−287** *(HH sign-flipped)* |

Restricting the correction to LL is **strictly better** by every metric we checked: lower late-half |gap|, no sign flips, all four per-band gaps improved vs baseline, and visually equivalent or slightly better on the 4-image panel. The mechanism: LL is an upstream causal lever — applying LL-only correction at step `i` propagates through the DiT's nonlinear forward and tightens all four band gaps at step `i+1` and after. Detail bands are downstream symptoms, not independent failures.

**Both modes ship LL-only.** Scalar default `--dcw_band_mask LL`; v4 controller hardcodes `LL` (the broadband ablation hasn't been re-run on v4 and isn't a near-term priority).

### LL-only λ magnitude (scalar)

| λ | late-half \|gap\| | Δ vs baseline | max \|gap\| |
|---|---|---|---|
| baseline | 330.1 | — | 64.0 |
| −0.005 | 281.5 | −14.7% | 53.0 |
| −0.010 | 235.7 | −28.6% | 42.1 |
| **−0.015** | **192.6** | **−41.7%** | **31.8** |

Closes 83% of the LL gap at the worst step (σ=0.04) and leaves headroom for per-LoRA calibration to push either direction. The closed-form solver predicts λ* ≈ −0.033 but that extrapolation crosses the nonlinear regime where `LL_const`-style overshoot kicks in (|λ · w(σ)| > ~0.01 late-step).

## Limitations / open questions

- **σ̂² channel under-trained** (Gate B fails on prototype). 3-seed `make dcw` rerun is the next experiment; if it still fails, ship with shrinkage permanently disabled and rely on the clamp guard.
- **Tiled inference** — v4 controller no-ops; scalar still works. The tile-merge boundary makes single-tile `c_pool` / `g_obs` ill-defined.
- **CFG drift** — v4 calibrated at CFG=4 only. Other CFGs fall back to scalar (proposal §"Risks" #7).
- **Cached-Spectrum `x0_pred`** is biased by Chebyshev forecaster error. Empirically should still help (correction is bias-agnostic) but worth one explicit ablation row.
- **Sign-flip vs the paper** unresolved — three speculative mechanisms in `archive/dcw/README.md`; cleanest test (smaller / pixel-space DiT) is out of scope.

## Related code

| File | Role |
|---|---|
| `networks/dcw.py` | `apply_dcw` (the apply site, shared by both modes) + `FusionHead` (shared by trainer + inference) + `haar_LL_norm` |
| `library/inference/dcw_v4.py` | `OnlineFusionDCWController` — loads artifact, observes warmup, fires head at step `k`, emits per-step `λ_i` |
| `library/inference/generation.py` | controller setup pre-loop + per-step apply at the DCW call site (non-tiled path) |
| `scripts/dcw_measure_bias.py` | offline trajectory dump + S_pop sweep — produces `gaps_per_sample.npz` consumed by the trainer |
| `scripts/dcw_train_fusion_head.py` | offline head training — produces `fusion_head.safetensors` |
| `scripts/tasks/dcw.py` | `make dcw` / `make dcw-train` task wrappers |
| `docs/proposal/dcw-learnable-calibrator-v4.md` | v4 derivation, gates, fallback ladder, evidence appendix |
