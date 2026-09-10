# SPD — Spectral Progressive Diffusion (training-free multi-resolution inference)

Port of Xiao et al., Spectral Progressive Diffusion (arXiv:2605.18736, [project page](https://howardxiao.ca/speed/)). Grow the spatial resolution along the denoising trajectory: run the early, noise-dominated steps at low resolution, then inject high-frequency detail via spectral noise expansion only once finer frequencies emerge from noise. Because the latent power spectrum decays as a power law (`P_ω ∝ |ω|^{-β}`, β ≈ 2.26 on Anima — `_archive/spd/bench/`), high frequencies carry far less signal and are cheap to defer.

SPD ships as a pure training-free inference stack: the bare DiT, or any existing LoRA checkpoint, runs the multi-resolution trajectory through the standard inference path with no training. (A trajectory-adapter *fine-tune* — a plain LoRA trained on the SPD velocity targets, the old "Case B" — was explored and then archived 2026-07-05 to `_archive/spd/`; see [Archived fine-tune](#archived-fine-tune) below.)

- Implementation: `networks/spd.py` (sampler-level runner, self-registers at import like `networks/spectrum.py`).
- Dispatch: `--spd` in `inference.py`, routed by `library/inference/generation.py::generate_body`.
- Bench / preconditions (archived): `_archive/spd/bench/` (+ `_archive/spd/bench/plan.md` for the gated phase history).

## Quick start

```bash
make test SPD=1                              # latest LoRA + SPD single-late knee (0.5→1.0 @ σ0.5)
make test SPD=1 NOLORA=1                     # bare DiT + SPD
```

Or add `--spd` to any `inference.py` invocation:

```bash
python inference.py --spd \
    --spd_stages 0.5 1.0 \
    --spd_transition_sigmas 0.5 \
    ...  # other inference args
```

`SPD=1` composes into every `test-*` target the way `SPECTRUM=1` does, but the two are mutually exclusive (both replace the denoise loop) — `SPECTRUM=1 SPD=1` raises.

### Schedule flags

| Flag | Meaning | Default |
|---|---|---|
| `--spd_stages` | Ascending resolution scales, e.g. `0.5 1.0` or `0.5 0.75 1.0`. A trailing `1.0` is appended if missing. | `0.5 1.0` |
| `--spd_transition_sigmas` | σ thresholds (in `[0,1]`) at which to spectral-expand to the next stage; `len = len(stages) − 1`. | `0.7` per handoff |

Note the two defaults differ by entry point: the bare `inference.py --spd` default (`_resolve_spd_schedule`) is the conservative σ0.7 knee, while the `make test SPD=1` wrapper (`_spd_flags` in `scripts/tasks/inference.py`) ships the bench-recommended single-late σ0.5 knee. Pass explicit flags to pin one.

## How it works

Standard inference pins one static token grid (`generation.py` reads `h_latent/w_latent` and builds `padding_mask` once — see the constant-token-bucketing invariant). SPD instead starts at a down-sampled latent grid and rebuilds the grid + `padding_mask` at each stage transition.

```
stage 0 (e.g. 0.5×)          transition at σ ≤ σ_t           stage 1 (1.0×)
┌─────────────────┐          ┌──────────────────┐           ┌────────────────────┐
DCT low-pass init  →  Euler steps at low res  →  spectral expand  →  Euler steps at full res
(dct_lowpass_init)     (noise-dominated)         (spectral_expand)    (HF detail revealed)
```

1. Low-pass init (`dct_lowpass_init`, paper `T_Φ`) — 2D type-II DCT of the full-res init latent, keep the low-frequency block, iDCT down to the stage grid (snapped to `patch_spatial`).
2. Low-res denoise — plain velocity-form Euler on the small grid. The DiT generalizes to the lower token count (validated in Phase 2; the per-Linear LoRA delta is token-count-agnostic, so adapters ride along).
3. Spectral expansion (`spectral_expand`, Eq. i–iii + 5–6) at the first step where `σ ≤ transition_sigma`: embed the current DCT block into the larger grid, fill the newly representable HF slots with σ-scaled noise, iDCT, scale by `κ = r/(1+(r−1)σ)`, and align the timestep to `σ̃ = rσ/(1+(r−1)σ)` (where `r = scale_hi/scale_lo`).
4. Re-space the remaining σ schedule to land on `σ̃` (Sec 4.3) and continue Euler at the higher resolution. Multiple stages chain the same expansion.

If the trajectory never reaches full res (under-specified schedule), a bicubic rescue upsamples the final latent so the VAE can decode.

## v0 scope & limitations

Spelled out in the `networks/spd.py` module docstring; the load-bearing ones:

- Euler only. Spectral expansion re-spaces the remaining σ schedule mid-loop; `ERSDESampler`/`LCMSampler` precompute their coefficients from the *full* schedule at construction and cannot follow the reshape. A requested stochastic sampler falls back to Euler with a one-time warning.
- Mutually exclusive with `--spectrum` — both replace the denoise loop.
- Does not compose with SMC-CFG. It acts at the sampler boundary on the (re-spaced) σ and is unvalidated against the mid-loop reshape; passing it with `--spd` warns and ignores.
- Composes with LoRA / Hydra / soft-tokens / P-GRAFT — the per-step adapter setters (`set_hydra_*`, `compute_and_set_hydra_fei`, soft-token splice, P-GRAFT cutoff) are mirrored from the standard loop.
- `torch.compile` cost: each resolution stage is a distinct token count = a distinct compiled graph (S stages → S graphs), breaking the single-static-shape guarantee. Per-block compile mode amortizes this.

## Schedules — what the bench found

From the Phase-2 schedule sweep (`_archive/spd/bench/README.md`, 832×1216, 2 seeds, with wall-clock):

| schedule (stages @ σ) | speedup | notes |
|---|---|---|
| `0.5/0.75/1.0 @ 0.8/0.6` (community `GJ5Rt3Xz`) | ×1.43 | transitions too early — wastes the low-res budget |
| `0.5→1.0 @ 0.7` (single-early) | ×1.32 | one handoff, high σ; slowest |
| **`0.5→1.0 @ 0.5` (single-late)** | ×1.65 | recommended knee — one handoff, simplest, cleanest |
| `0.5/0.75/1.0 @ 0.7/0.5` (2-stage-late) | ×1.61 | drop-in keeping the community shape |
| `0.4/0.7/1.0 @ 0.7/0.5` (aggressive) | ×1.73 | fastest; more divergence/variance |

A single, late handoff matches a 2-stage ramp in quality while being simpler and faster (fewer HF injections). SPD-on-Anima also runs consistently sharper / higher-contrast than baseline — a behavioral signature, not a transparent speedup.

Caveat: even tuned (×1.73 max), standalone SPD is slower than the Spectrum node we already ship (~×3.75). The real open question is SPD ∘ Spectrum (orthogonal: token-reduction vs block-skipping) and SPD vs Turbo at matched quality — both untested. The speed/quality bench (not yet written) is the remaining Phase-3 TODO.

## Why it works on Anima — precondition history

SPD was integrated behind a gated plan (`_archive/spd/bench/plan.md`) — each phase can kill the idea before paying for the next:

- Phase 0 — spectral premise: PASS. `P_ω ∝ |ω|^{-β}` with β = 2.26, R² = 0.9994 (200 imgs), 30/30 artists in `[2,3]`. The exponent is a property of the VAE latent space, not of any style. (`measure_latent_spectrum.py`, `per_artist_spectrum.py`.)
- Phase 1 — autoregression dynamics: DONE, WEAK. `σ_resolve` is cleanly monotone in frequency (low bands lock by σ≈0.75, top band only by σ≈0.29 — the Fig-2b picture), so early-low-res is justified *in principle*. But the principled δ=0.01 schedule is conservative: it sanctions only ×1.15–1.22, far below the ×1.65 the hand-tuned σ≈0.5 knee gets. That gap is the headroom the (now-archived) fine-tune chased. (`measure_autoregression.py`.)
- Phase 2 — resolution generalization: PASS. The bare DiT denoises low-res latents and accepts the spectral-expansion handoff with no instability (std ×0.95, no NaN, no smear/double-image), across both single-stage and community schedules. This was the real go/no-go — if the model mushed at low res, training-free SPD would be dead. (`probe_lowres_denoise.py`.)
- Phase 3 — integration: shipped (v0). Runner + CLI in `networks/spd.py`; speed/quality bench + SPD∘Spectrum study still open.

## Archived fine-tune

A plain-LoRA trajectory-adapter fine-tune ("Case B") — trained on the stage-specific straight-line velocity targets (§4.3, Eq. 11–14) so the model sees the multi-resolution trajectory instead of generalizing to it zero-shot — was implemented, trained (8k steps, single-late knee), and found only marginally cleaner than training-free SPD at the same schedule without a measured speed/quality win. It was archived 2026-07-05 to `_archive/spd/`:

- Trainer + config: `_archive/spd/scripts/distill_spd.py`, `_archive/spd/configs/spd.toml`.
- Target-construction + SNR-gated-loss helpers (formerly in `networks/spd.py`): `_archive/spd/networks/spd_train_targets.py`.
- Design + bench history: `_archive/proposals/spd_finetune_lora.md`, `_archive/spd/bench/plan.md`.

The live tree keeps only the training-free inference runner. If SPD ever earns a trained adapter, re-integrate from the archive rather than re-deriving the velocity-target math.
