# Variance-Reduced FM Loss (AsymFlow §5.2)

Training-loss-level integration of the control-variate correction from
Chen et al., Asymmetric Flow Matching for Pixel-Space Generation
(arXiv:2605.12964 §5.2). The flow-matching MSE estimator is variance-reduced
by pairing each step with a no-grad forward of the base DiT on the
FEI-low-passed latent. For LoRA-family runs the base DiT is frozen and the
adapter is additive, so we get the "frozen reference" by reusing the
trainable DiT with `network.set_multiplier(0)` for the no-grad pass — no
second model copy in VRAM. ~99.8% of the per-sample loss variance was found
recoverable on Anima at the global-λ optimum in the headroom bench
(since removed; verdict HEADROOM).

This is a loss-level change, not a new adapter — the trained checkpoint
inferences identically to a standard FM-trained one. Composes with every
adapter family in `networks/lora_anima/`.

> Framing. "Variance reduction" follows the paper, but the gradient of
> `||y + λz||²` is biased toward the high-frequency residual `x_0 − x_0^L`:
> `z` is detached, but co-varies with `∂u_pred/∂θ` through the shared input
> `x_t`, so `E[z · ∂u_pred/∂θ] ≠ 0` and the standard FM stationary point is
> shifted. That bias is the intended training signal — spend adapter
> capacity on what the base can't already explain. Read this as
> base-residual control-variate FM, not unbiased estimation of standard
> FM. The loss-level reduction (ρ² ≈ 0.9999, see headroom bench) is real;
> whether it translates into a corresponding gradient-variance reduction
> for the optimizer is still an open empirical question — see
> [Open questions](#open-questions).

## TL;DR

```
# Standard FM (per element, in velocity-target form)
y        = u_pred − (ε − x_0)                       # = (x_0 − x̂_0) / σ_t
L_FM     = ||y||²

# VR loss (this addition)
x_0^L    = gaussian_blur_2d(x_0, σ_low)             # FEI-aligned low-pass
x_t^L    = (1 − σ_t) · x_0^L + σ_t · ε              # paired noisy input, SAME ε
u_pred^L = anima(x_t^L, t, te)  with mult=0         # no-grad bypass forward
z        = u_pred^L − (ε − x_0^L)                   # control variate residual
L_VR     = ||y + λ · z||²                           # gradient flows through y only
```

with `λ = −Cov(y, z) / Var(z)` estimated online (per-batch covariance + EMA
across batches, β default 0.01). One extra no-grad forward per step
(~+40% step cost). The "frozen reference" is the trainable DiT itself with
`network.set_multiplier(0)` during the no-grad pass — equivalent to a frozen
base DiT for LoRA-family runs (the base is frozen, adapters are additive),
with zero extra VRAM for a second model copy.

## Quick start

```bash
# Enable on any LoRA-family run:
make lora --vr_loss_weight 1.0
make lora-gui GUI_PRESETS=tlora --vr_loss_weight 1.0
python tasks.py lora --vr_loss_weight 1.0

# Or flip the keys in configs/methods/lora.toml (commented out by default):
#   vr_loss_weight = 1.0
#   vr_fei_sigma_low_div = 4.0
```

VR is gated off by default (`vr_loss_weight = 0.0`). When `> 0`, the trainer
runs one extra no-grad forward per step through the trainable DiT with
`network.set_multiplier(0)` (zeros the LoRA contributions). No
extra model is loaded into VRAM. The only cost is the extra forward
(~+40% step time); low-VRAM presets can run this — they just pay the
compute.

## What it actually does

```
                              latents x_0  (B, C, H_lat, W_lat)
                                    │
                       ┌────────────┴───────────┐
                       │                        │
                       │              gaussian_blur_2d(σ_low)
                       │                        │   library/runtime/fei.py
                       │                        ▼
                       │                       x_0^L
                       │                        │
                       │   ε ───────┐           │   ε (same draw)
                       ▼            ▼           ▼            ▼
              (1−σ_t)·x_0 + σ_t·ε       (1−σ_t)·x_0^L + σ_t·ε
                       │                        │
                       │                        ▼
                       │            ┌─ same DiT, mult=0 (no_grad, bf16)
                       │            │  (no separate model load)
                       ▼            ▼
            trainable DiT      u_pred^L
                  │                  │
              u_pred                 │
                  │                  │
                  └──── y = u_pred − (ε − x_0) ──┐
                                                 │
                  z = u_pred^L − (ε − x_0^L) ────┤
                                                 ▼
                              loss = ||y + λ · z||²    (per element)
                                          │
                              λ_batch = −Cov(y_det, z) / Var(z)
                              λ_ema   ← (1−β)·λ_ema + β·λ_batch
```

### Why a low-pass `x_0^L` (paper mapping)

AsymFlow §5.2's variance reduction hinges on a paired construction: a
target `x_0^L` the frozen reference predicts accurately, and an orthogonal
residual `x_0 − x_0^L` that the finetune actually has to learn. In the
paper, `x_0^L = Az_0` is a patch-wise Procrustes lift from the pretrained
latent into a low-rank pixel subspace `Im(P)` (Appendix A.1). Because the
frozen latent model nails this component by construction, its prediction
deviation `d^L = x_0^L − x̂_0^L` carries the shared noise structure that
also pollutes the full-rank training residual `d = x_0 − x̂_0`. With a
patch-wise `λ* = −⟨d^L, d⟩ / ‖d^L‖²` (paper Eq. 18 / Appendix A.3), VR
cancels exactly that shared variance, leaving only the new low-level
mismatch the finetune is supposed to close.

Anima lives entirely in latent space, so there's no latent-to-pixel lift
and no Procrustes subspace to construct. The natural Anima analog is the
FEI Gaussian low-pass: it splits the latent into a structural band the
frozen base predicts confidently and a detail band that's left for the
adapter to learn. That's the same role `Pε` / `(I − P)ε` plays in the
paper, just expressed in spatial frequency instead of patch-PCA coordinates.

Mapping the paper onto our setup:

| AsymFlow §5.2 | This integration |
|---|---|
| Frozen latent base (separate pretrained model) | `network.set_multiplier(0)` on the trainable DiT — LoRA is additive, so the bypass forward *is* the base DiT |
| Low-rank target `x_0^L = Az_0` (Procrustes lift) | `x_0^L = gaussian_blur_2d(x_0, σ_low)` (FEI low-pass) |
| Orthogonal complement `(I − P)x_0` | The high-frequency band `x_0 − x_0^L` |
| Patch-wise `λ*` (Appendix A.3, clamped to `[0, 1]`) | Global EMA `λ`; per-element / per-band falsified below |

```
σ_low = min(H_lat, W_lat) / vr_fei_sigma_low_div    # default 4.0
x_0^L = gaussian_blur_2d(x_0, σ_low)
```

What makes FEI specifically right here:

1. Aligned with what the base already knows. Anima pretraining locks in
   the low-frequency / structural content; LoRA fine-tunes the detail. The
   FEI low-pass picks out exactly the band where the frozen-reference
   prediction is accurate — the precondition for high ρ² in the control
   variate.
2. Same band split the adapter already routes on. HydraLoRA / FeRA routing
   on `router_source = "fei"` is conditioned on the same
   `library/runtime/fei.py::gaussian_blur_2d` split, so the control variate
   inherits the existing inductive bias instead of inventing a new one.
3. In-tree, kernel-cached. `gaussian_blur_2d` is fp32-safe and the kernel
   is cached — no new module, no extra alloc.
4. Free diagnostic axis. Per-FEI-band ρ² in the headroom bench
   (since removed) confirms the
   paper's mechanism transposes: the high-band ρ² (mid-t median 0.998,
   λ_global −0.996 ± 0.002) is what carries the headroom, exactly as
   `Im(I − P)` residual carries the win against `Im(P)` deviation in
   AsymFlow.

Default `vr_fei_sigma_low_div = 4.0` matches the live training default in
the FEI-routed Hydra variant in `configs/methods/lora.toml`.

### λ estimation

`λ = −Cov(y, z) / Var(z)` minimizes `Var(y + λz)`. We use the global form
(scalar λ over all latent positions and batch elements), not the per-element
or per-band variants:

```python
y_d = (u_pred − (ε − x_0)).detach()                 # detached residual
cov = (y_d * z).sum()
var = (z * z).sum().clamp_min(1e-12)
λ_batch = float(−(cov / var))
λ_ema   = (1 − β) · λ_ema + β · λ_batch             # β = vr_lambda_beta, default 0.01
```

`λ_ema` is initialized to `λ_batch` on step 0 (no warm-up window). Bench
confirmed `λ_global = −0.996 ± 0.002` across all 36 (sample, t) pairs at
N=32, so the online estimator converges fast and a small β is well
conditioned.

What `λ ≈ −1` means. Substituting λ = −1 into the loss:

```
||y − z||²  =  ||(u_pred − u_pred^L) − ((ε − x_0) − (ε − x_0^L))||²
            =  ||Δu_adapter − (x_0^L − x_0)||²
            =  ||Δu_adapter + x_0^H||²      where x_0^H ≡ x_0 − x_0^L
```

So the EMA-converged loss is asking the adapter's delta-prediction
`Δu_adapter = u_pred − u_pred^L` to cancel the high-frequency velocity
residual `x_0^H` — exactly the band the FEI low-pass discards. This is
the operational statement of "spend adapter capacity on what the base
can't already explain". Because the online EMA lands so close to the
fixed `λ = −1` limit, a fixed-λ control bench (see Open questions) is
the cleanest remaining ablation: if it matches learned-EMA within noise,
the cov/var bookkeeping can go and the loss becomes a one-liner.

Per-element λ (v2) and per-FEI-band `λ_k` (v3) were considered as refinements
and bench-falsified on Anima by the perband-headroom run
(since removed; n=24 mid-t pairs, T-LoRA-merged vs base):

- v2 (per-element λ): `reduction_per_elem − reduction_global` mean +7.9e-6
  (= +0.00079% absolute).
- v3 (per-band λ): `perband__reduction_combined − reduction_global` mean
  −3.4e-6 (= −0.00034%; sign is an estimator artifact — the within-band
  optima of `(λ_low, λ_high)` aren't the joint optimum because the FEI bands
  aren't statistically orthogonal across noise samples. The joint-optimum
  upper bound is the per-element number, +7.9e-6).

The scalar global λ is already at the asymptote (`reduction_global ≈ 0.9999`
mid-t mean), so there is no remaining variance for a richer λ to cancel.
v2 / v3 will not ship.

### Adapter-bypass reference

The "frozen reference" is the trainable DiT itself, run no-grad with the
LoRA network's multiplier temporarily set to 0:

```python
_orig_mult = float(getattr(network, "multiplier", 1.0))
network.set_multiplier(0.0)         # zeros LoRA (network.py:860-865)
try:
    with torch.no_grad():
        ref_pred = anima(x_t_L_5d, timesteps, crossattn_emb, padding_mask=padding_mask, **kw)
finally:
    network.set_multiplier(_orig_mult)
```

For LoRA-family runs this is bit-equivalent to a frozen copy of the base
DiT: the base weights are frozen for the whole training run, and adapters
are additive residuals on top — turning the multiplier to zero collapses
the model to its base. No `--vr_frozen_ref_dit` flag, no second model copy
in VRAM, no constant-token-bucket state mirroring to keep in sync.

`set_multiplier(0)` covers `LoRA` / `OrthoLoRA` / `HydraLoRA` /
`StackedExperts`.
Postfix's `network.append_postfix` modifies `crossattn_emb` *before* the
DiT call, not the DiT itself, so the bypass forward receives the same
postfix-appended tokens as the gradient forward — postfix is therefore
not nulled (it can't be: postfix isn't an additive residual on weights).

> **Semi-gradient caveat (postfix + VR).** When postfix is trainable, `z`
> depends on the postfix parameters through `crossattn_emb`, but its
> gradient w.r.t. those parameters is dropped (`z` is detached). This is
> a semi-gradient method on the postfix path: the cov/var λ estimate
> is well-defined, but the postfix gradient no longer minimizes the loss
> you wrote down. Plain LoRA-family runs are unaffected. Treat
> postfix+VR as experimental; gate it explicitly per-run rather than
> mixing the two by default.

Hooks on the unet (functional MSE captures) fire on this forward too,
but they are consumed before the VR block runs — see the order in
`train.py::get_noise_pred_and_target`.

## Implementation map

| Layer | File | Role |
|---|---|---|
| CLI args | `library/anima/training.py` | `--vr_loss_weight`, `--vr_fei_sigma_low_div`, `--vr_sigma_min`, `--vr_lambda_beta` |
| Config gate | `configs/methods/lora.toml` | Commented `vr_loss_weight = 1.0` block; uncomment to enable |
| Forward + stash | `train.py::get_noise_pred_and_target` | Builds `x_0^L`, `x_t^L`, calls `network.set_multiplier(0)` + no-grad `anima(...)` + restore, stashes `ctx.aux['vr'] = {'z': ..., 'state': ...}` |
| Loss handler | `library/training/losses.py::_flow_matching_vr_loss` | Computes `(y + λ·z)²`, updates `state['lambda_ema']` in place |
| Composer gate | `library/training/losses.py::build_loss_composer` | Replaces `flow_match` → `flow_matching_vr` when `vr_loss_weight > 0` |
| Headroom bench | (removed) | The Stage 0 ρ² probe + integration plan this doc summarizes |

The trainer↔loss-handler contract is the `ctx.aux['vr']` dict:

```python
self._extras_for_step["vr"] = {
    "z": z_residual.detach(),      # (B, C, H_lat, W_lat), no_grad
    "state": self._vr_state,       # {"lambda_ema": float | None, "lambda_batch": float}
}
```

Both `_extras_for_step` and the `state` dict are reset / persistent per the
trainer's existing conventions: `_extras_for_step` clears every step,
`_vr_state` persists across steps (mutated in place by the loss handler so
the EMA survives).

## Compute cost

Per training step:

| | grad fwd | no-grad fwd | bwd | net |
|---|---|---|---|---|
| Standard FM | 1 | 0 | 1 | 1× |
| VR loss     | 1 | 1 | 1 | ~1.4× |

The no-grad forward runs in `torch.no_grad()` inside the same
`accelerator.autocast()` scope as the main forward — no checkpointing,
no graph save, no second backward. On a 5060 Ti at the typical Anima
bucket (`128×128` latent), the extra forward is ~40% the cost of the
gradient-tracked forward, so net step cost is ~1.4×.

If block swap is on, the bypass forward also pays the swap cost. That's
compute, not memory, but it slightly inflates the 1.4× figure on heavily
swapped presets. Forward hooks (functional MSE capture) also fire on
the bypass forward — they're benign because the captured state is
consumed before the VR block runs.

Net training win requires VR to give >1.4× effective convergence. The
paper reports +0.96 HPSv3 from VR alone on AsymFLUX.2 klein (Table 3).
On Anima, a short A/B at r=16 / 2.56k steps took 60min with VR vs. 50min
standard FM, and VR samples were visibly the quality win at the +40%
step-cost overhead (eyeball A/B; quantitative HPSv3 / VQA pass still
optional).

## Memory

No extra VRAM beyond what standard FM uses. The control-variate forward
reuses the trainable DiT (with adapter multiplier=0), so there is no
second 2B model held in memory. Low-VRAM presets (`low_vram`, `fast_16gb`)
can run VR — they just pay the ~+40% compute.

## Config knobs

| Flag | Default | Meaning |
|---|---|---|
| `vr_loss_weight` | `0.0` | Gate and overall scale on the VR loss term. `0.0` disables (standard FM); `1.0` matches the paper recipe; smaller values let other losses contribute relatively more. |
| `vr_fei_sigma_low_div` | `4.0` | Divisor for `σ_low = min(H_lat, W_lat) / div`. Matches the live FEI default. |
| `vr_sigma_min` | `1e-3` | Defensive floor on `σ_t` in the `1/σ_t` factor (AsymFlow §6.1). Not consumed by the shipped loss handler — we work in velocity-target form so `σ_t` cancels algebraically. The parser flag is kept as a no-op for compatibility; the per-element-λ extension it was reserved for was falsified on the headroom rig (scalar λ already at the asymptote). |
| `vr_lambda_beta` | `0.01` | EMA rate on `λ`. `λ_ema ← (1−β)·λ_ema + β·λ_batch`. |

## Open questions

- Fixed `λ = −1` vs learned EMA — `λ_ema` settles at −0.996, so the
  loss is operationally `||y − z||²`. A short bench fixing `λ = −1` (drop
  cov/var bookkeeping, single hyperparameter-free loss) against the
  learned-EMA path should match within noise; if it does, ship the
  fixed-λ variant. This is the cheapest remaining ablation.
- Gradient-level diagnostics — the headroom bench measures loss-level
  ρ² (0.9999), but the optimizer cares about `Var[g]` and
  `cos(g_vr, g_full-batch)`. Add a small probe that logs:
  `Var[g_standard]`, `Var[g_vr]`, `cos(g_vr, g_standard)`,
  `cos(g_vr, g_largebatch_reference)` — this is the missing link between
  "99.99% loss variance recovered" and "the optimizer actually does
  better". Cheap; can run alongside the fixed-λ bench.
- Wall-clock-matched A/B — current A/B is matched-step (60min VR vs
  50min standard). The honest comparison gives standard FM the 1.4× step
  budget VR pays for; only then is the quality delta attributable to VR
  vs to more compute. Run before stamping v1 as "shipped quality win".
- Mid-training ρ² stability — the bench used a *merged* T-LoRA
  against base, where `u_pred ≈ u_pred^L` is true by construction. Re-probe
  ρ² at, say, step 1k / 2k of a live run to confirm the correlation
  doesn't collapse once the adapter has drifted from base. If it does,
  the EMA λ will track but the variance-reduction headroom shrinks.
- DDP / accumulation correctness — `cov` and `var` for λ are computed
  per-rank in `_flow_matching_vr_loss`. If anyone runs Anima multi-GPU,
  this needs an `accelerator.reduce(...)` across ranks before the EMA
  update; otherwise λ silently desynchronizes across workers. Single-GPU
  runs (the typical Anima training preset) are unaffected.
- Reference granularity — current code always reads "pure base"
  (multiplier=0 on every step). A variant could use the current trainable
  adapter at some scale (e.g. multiplier=0.5 or the resumed multiplier) as
  the control variate — that's a one-line change to the `set_multiplier`
  call, but needs thinking about whether the residual `z` stays decorrelated
  from the gradient, since the bypass forward now shares a non-trivial
  function with the gradient forward. The pure-base choice keeps `z`
  cleanly independent of the trainable LoRA's current state. Still open;
  no measured signal that it matters.
- CFG-dropout interaction — the loss uses the same (possibly dropped)
  crossattn_emb for both forwards in a step, so cancellation is preserved.
- LPIPS perceptual correction — the paper pairs VR with an LPIPS term
  to absorb the bias from `E[x_0^L | x_t] ≈ E[x_0^L | x_t^L]`. We skip it
  and sample quality holds at r=16 / 2.56k steps; if a longer-step quality
  regression ever surfaces, LPIPS is the first thing to try (not richer λ —
  per-element / per-band variants were falsified on the headroom rig).

## References

- AsymFlow paper (arXiv:2605.12964), Chen et al., 2026. §5.2 is the
  variance-reduction section; §6.1 covers the `σ_min` clamp.
- The VR-loss headroom bench (no longer in-tree) — the diagnostic that gated v1 + the full integration plan.
- `docs/methods/hydra-lora.md` — FEI routing background.
- `[[project_fera_probe_2band_decision]]` — why we use 2 bands, not 3.
- `[[project_fm_val_loss_uninformative]]` — why Stage 1 needs HPSv3/VQA,
  not just val FM curves.
