# REPA-DoG — band-pass the alignment target before the Gram match

Status: **ARCHIVED — SUCCESSFUL (2026-06-17).** The band-pass alignment target
won (Phase 0 passed; band-pass beats the shipped `spatial_norm` on all three
axes; `[[project_repa_dog_phase0_passed]]`). Moved out of active proposals.
Original status preserved below.

Status: **PHASE 1 WIRED 2026-06-13.** Phase 0 (data-only σ-sweep probe,
`bench/repa/probe_dog_target.py`, 3000 PE-Spatial sidecars) PASSED, so the
training arm is now implemented: `dog_standardize` in `library/training/repa.py`
+ the `repa_target_dog` / `repa_dog_sigma1_div` / `repa_dog_sigma2_div` /
`repa_dog_norm_std` knobs (`configs/methods/lora.toml`, allowlisted in
`networks/__init__.py`). **Default off** (`repa_target_dog = false`) — when on it
*replaces* the `spatial_norm` block in the relational target preprocess (DoG at
σ₁→0 *is* DC removal, same family). The A/B run + readout (below) is still open;
hold `repa_dog_norm_std = 0` (empirical std, = shipped `spatial_norm`) so the only
change vs baseline is the band-pass — the paper's std confound. Builds on the
shipped relational arm + `repa_spatial_norm` (`docs/experimental/repa.md`, REPA v2
Phase 0 closed 2026-06-12, `[[project_repa_v2_relational_won]]`). Supersedes —
and removes — the global-anchor line (`repa_global_weight`, **refuted**,
`_archive/proposals/repa_global_anchor.md`); that arm's code, config, calib, and
`bench/repa/build_calib.py` were deleted with this change. See Why now.

**Phase 0 result** (`bench/repa/results/20260613-*-dog-axis-full/`, stable over
two 3000-sample runs). Best +1a (broad low-band strip, `σ_lp = min(gh,gw)/16–32`)
beats the shipped DC-only `spatial_norm` baseline on **all 3 content axes** —
character +0.034 (0.717→0.751 AUC), copyright +0.055, artist +0.046 — and the
DoG direction is the live one (✓ the paper's RMSC↔discriminability link replicates
on PE-Spatial, the encoder-mismatch caveat). **Surprise:** point 0 (DC-only
removal, what we ship) is a **local dip** — it scores *below* even the keep-DC
(−1) endpoint on every axis, so the win isn't only "DoG > spatial_norm" but
"spatial_norm is leaving contrast on the table; broad low-band strip recovers it."
**+1b ≈ +1a** at the winning σ_lp (mild harm at small σ_lp) → keep **σ₂ off**, as
the proposal predicted. Open: the A/B's CMMD + style gate, and the std confound
(`repa_dog_norm_std`), still untested — discriminability ≠ generation quality.

Source: *Spectrum Matching: a Unified Perspective for Superior Diffusability in
Latent Diffusion* (arXiv:2603.14645v1, repo root). The DoG-on-REPA idea is §3.5
+ §4.2 of that paper; everything else in it (ESM/DSM VAE regularizers) is
out of scope here.

## Why now

Two independent threads point at the same lever — the **low-frequency / global
band of the alignment target is a distractor**, and removing *more* of it is the
live direction:

1. **Our own refuted experiment.** The global-anchor arm
   (`repa_global_weight`) tried to *re-inject* the per-image global component
   that `spatial_norm` strips. A/B at `0.01` and `0.001` both **degraded** CMMD
   (`_archive/proposals/repa_global_anchor.md`). Read on the frequency axis,
   that arm is the **−1 endpoint**: adding the global/DC band back hurts. The
   shipped `spatial_norm` (strip DC) is the **0 point**. DoG is the **+1
   endpoint**: strip a *broader* low band than DC alone. We've now measured one
   end of the axis and it confirms the sign.

2. **The paper's reframing.** Prop 3.2 proves iREPA's RMS Spatial Contrast
   (RMSC) equals the **directional spectral energy** of the target (AC energy of
   the L2-normalized token field, DC excluded). iREPA's spatial normalization
   `(Z − α·mean)/std` is exactly **DC removal**; the paper generalizes it to a
   **Difference-of-Gaussians band-pass** `DoG(Z) = G_σ₁*Z − G_σ₂*Z`, then
   `/std`, suppressing a broader low band *and* rolling off very-high freq. On
   ImageNet-256 / SiT-B/2 / DINOv2 they report REPA-DoG > iREPA > REPA in gFID.

So the increment over what we ship is precise: `spatial_norm` already removes the
DC/global band (the thing the global-anchor arm proved is a distractor). DoG asks
whether suppressing the **mid-low band just above DC** — and optionally the
high-freq tail — buys more.

### The local connection — we already own the operator

This is not a from-scratch build. `library/runtime/fei.py::gaussian_blur_2d` is
the separable, cached, fp32, reflect-padded, bucket-invariant Gaussian a DoG
needs: `DoG(Z) = gaussian_blur_2d(Z, σ₁) − gaussian_blur_2d(Z, σ₂)`. And FEI's
2-band high-pass residual `z − LP(z, σ)` **is the σ₁→0 corner of DoG** — same
operator family; the paper's band-pass just gives the inner kernel a finite σ₁
to also kill the very-high frequencies. The divisor-sweep methodology in
`bench/fera_artist/probe_fei_artist.py` (`_argmax_divisor` by `std(e_low)`, with
artist-balanced sampling so the top-3 artists' ~22% image share doesn't swing
the score) is the exact shape of the Phase-0 σ-selection below.

Two things do **not** transfer: FEI runs on the **latent `z_t`** as a HydraLoRA
routing key (tuned `div=4` for the ~4096-patch latent grid); REPA-DoG runs on
the **PE-Spatial target features** as loss preprocessing on a much coarser grid,
so σ needs its own calibration. And FEI measures **magnitude band-energy**
(‖LP‖² vs ‖HP‖²) whereas the paper's RMSC measures **direction-field variance** —
both live on a DoG but read different things off it. fera_artist gives us the
operator + the sweep recipe, not the metric.

## Phase 0 — data-only σ-axis probe (the gate)

Fork `bench/fera_artist/probe_fei_artist.py` into `bench/repa/probe_dog_target.py`.
No DiT, no text encoder — load cached PE-Spatial sidecars, reshape each to its
`(gh, gw)` grid, and sweep the four points of the low-freq-strip axis:

| point | target transform | role |
|---|---|---|
| −1 | `+ global` (re-add patch-mean) | refuted endpoint (sanity: should be worst) |
| 0 | `spatial_norm` (DC removal) | shipped baseline |
| +1a | DoG low side, σ₂→∞ (`LP(σ₁)` subtracted, no high rolloff) | low-band increment |
| +1b | full DoG band-pass (`LP(σ₁) − LP(σ₂)`) | + high-freq rolloff (σ₂ risk) |

**Readout = target discriminability**, reusing the global-anchor probe's metric:
AUC of `P(in-group cosine > out-group cosine)` over character/copyright/artist
pairs from `caption_index.json` (`bench/repa/discriminability.py`,
`[[project_pe_cls_collapse_patchmean]]`), computed on the **Gram affinity** the
relational arm actually matches (per-token L2-norm → pairwise cosine), not on raw
features. Sweep σ₁ as a bucket-invariant divisor `min(gh,gw)/div₁` (mirror FEI).

**Gate.** If AUC rises monotonically −1 → 0 → +1a, the low-side DoG is worth
training and we leave σ₂ off. If it flattens after the shipped `spatial_norm`
(point 0), DoG is **irrelevant, not harmful** — close the line cheaply, the DC
removal we ship already owns the usable contrast. If +1b < +1a, the high-freq
rolloff hurts → pin σ₂→∞ permanently.

## Design (only if Phase 0 passes)

Target-side preprocessing in `library/training/repa.py`, composing with the
existing relational path — **byte-identical no-op when off**:

1. **New transform** `dog_standardize(pe, gh, gw, sigma1_div, sigma2_div, norm_std)`
   *(implemented)*: reshape `(B, N, d) → (B, d, gh, gw)` row-major, apply `H(Z)`,
   `/ (std + ε)`, flatten back. `σ₁ = min(gh,gw)/sigma1_div` is the **outer** kernel
   (broad low band removed); `σ₂ = min(gh,gw)/sigma2_div` the **inner** tighter one.
   `H(Z) = Z − LP(Z, σ₁)` when `sigma2_div ≤ 0` (high-pass / +1a corner), else
   `LP(Z, σ₂) − LP(Z, σ₁)` (band-pass / +1b). Reuses
   `library/runtime/fei.py::gaussian_blur_2d` (kernel clamped to the grid) — no new
   conv code. `norm_std = 0` ⇒ empirical per-channel std (= `spatial_norm`); `> 0` ⇒
   fixed const (the paper's regime, optional ablation).
2. **Slot-in** in `relational_align_loss` / the adapter's `extra_forwards`:
   when `repa_target_dog` is on, apply `dog_standardize` to `pe` (after CLS-drop,
   on the resolved grid) **instead of** the `spatial_norm` mean/std block in
   `relational_gram_loss`. DoG with σ₁→0, σ₂→∞ reduces to DC-removal, so
   `spatial_norm` stays the degenerate special case for the axis sweep.
3. **DiT side untouched.** This is purely target preprocessing; the student
   path is bit-identical, so the change is provably inert when the knob is off
   and cannot regress non-REPA runs.

Config knobs (`configs/methods/lora.toml` + **allowlist in `networks/__init__.py`
`*_KWARG_FLAGS`** — else inert + config-test fail, `[[project_network_kwarg_toml_allowlist]]`):

- `repa_target_dog` (bool, default **false** ⇒ no-op)
- `repa_dog_sigma1_div` (bucket-invariant `σ₁ = min(gh,gw)/div₁`, from Phase 0)
- `repa_dog_sigma2_div` (`0` ⇒ low-pass-subtract only / σ₂→∞; else the band-pass)
- `repa_dog_norm_std` (default match `spatial_norm` — see confound below)

## A/B plan & pre-registered readout

Single arm vs current relational-only + `spatial_norm`, same data/preset/steps:

- **Arm**: `--repa_target_dog true` with the Phase-0 σ₁ winner, σ₂ off unless
  +1b won the probe.
- **Confound control (mandatory).** The paper's REPA-DoG runs
  `normalization_std = 1.0` vs iREPA `0.6` (their Table 6), **unablated** — so
  part of their reported win may be the std change, not the band-pass. Hold
  `repa_dog_norm_std` equal to the shipped `spatial_norm` std, or the A/B can't
  attribute the delta to DoG.
- **Primary metric**: CMMD val signal — lower
  wins. FM val loss is uninformative.
- **Hard gate (style)**: qualitative anime-style pass on the fixed sample
  prompts. Aggressive low-freq stripping touches the global/style axis — the v1
  burn hazard; any visible style drift =
  FAIL regardless of CMMD.
- **Readout**: CMMD improves and style holds → keep, tune σ₁. Flat / style
  drifts → close; `spatial_norm` already owns the usable structure.

## Risks / honest limits

- **Paper evidence is thin where we're borrowing.** REPA-DoG's gFID margins are
  small (4.98 vs iREPA 5.07 vs REPA 5.68 @400k, no seeds/error bars), single
  backbone (SiT-B/2), single dataset (ImageNet-256), and the σ₁/σ₂ are **never
  ablated** in the paper. The std confound (above) is unaddressed. Treat the
  paper as motivation, not validation.
- **Encoder mismatch.** Their evidence is DINOv2/v3; we align to **PE-Spatial**
  (CLIP-lineage), which already behaves differently — PE CLS collapsed. The RMSC↔quality correlation that
  justifies DoG may not replicate on PE; that's exactly what Phase 0 tests.
- **Could be a no-op.** `spatial_norm` already removes DC. If the mid-low band
  carries no extra distractor energy, DoG adds nothing — likely outcome is
  "irrelevant," not "harmful." That's an acceptable cheap close.
- **σ₂ is the only harmful corner.** The high-freq rolloff is the one part the
  global-anchor result said nothing about; if fine PE tokens carry useful Gram
  affinity, σ₂ degrades where DC removal never did. Default σ₂ off until proven.
- **Coarse grid.** On a small PE grid a band-pass can wipe almost everything;
  validate the grid survives the σ₂ subtraction in Phase 0.

## Explicitly NOT doing

- **Global-anchor / re-injecting the DC band** — refuted A/B
  (`_archive/proposals/repa_global_anchor.md`). Do not re-propose; DoG is the
  opposite sign and the live direction.
- **The paper's ESM/DSM VAE regularizers** — those retrain the VAE from scratch
  (CelebA/ImageNet, 500k–600k steps). Out of scope for a frozen-base LoRA repo.
- **Switching the target encoder to DINOv2/v3** to match the paper — PE-Spatial
  is the shipped tower; the question is whether DoG helps *our* target, not
  whether we can reproduce their setup.
