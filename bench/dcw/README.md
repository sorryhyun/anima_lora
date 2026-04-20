# DCW bias bench

Direct, model-agnostic measurement of the SNR-t bias on Anima's flow-matching
DiT. Also a hyperparameter-selector for DCW: tune the scaler against the
*actual* bias the method is designed to fix, not against a downstream FID
proxy.

Paper: Yu et al., *Elucidating the SNR-t Bias of Diffusion Probabilistic
Models* (CVPR 2026, [arXiv:2604.16044](https://arxiv.org/abs/2604.16044)).
Upstream code lives in `../../DCW/`.

Design context in this directory:
- [`proposal.md`](proposal.md) — reframing Anima's existing variants through the SNR-t-bias lens; why DCW is the right next lever.
- [`plan.md`](plan.md) — DCW integration plan (inference path, config surface, sweep strategy).
- This README — what the bench measures, how to run it, and what the 2026-04-20 baseline told us.

## What this does

At every denoising step `i`, with a fixed pretrained DiT, we evaluate the
velocity on two samples of the same nominal timestep `t_i`:

- **Forward branch:** `x_t_fwd = (1 − σ_i) · x_0 + σ_i · ε`, where `x_0` is
  the VAE-encoded latent of a real training image and `ε ~ N(0, I)`.
- **Reverse branch:** `x̂_t_i` — the latent at step `i` of a reverse
  sampling trajectory started from pure noise (no conditioning changes).

The paper's Fig. 1c shows `||v_θ(x̂_t, t)|| > ||v_θ(x_t_fwd, t)||` across
all `i` for trained DDPM/ADM/EDM models on CIFAR-10, CelebA, LSUN,
ImageNet. This is their Key Finding 2: reverse samples systematically have
lower SNR, and the network's prediction norm is SNR-monotone (Key
Finding 1), so the reverse-branch norm is systematically larger.

**We want to check this on Anima before committing to DCW integration.**
If the gap is tiny or missing, DCW's premise doesn't hold on flow-matching
DiTs at our scale, and a different intervention is needed.

## What this lets us do

1. **Diagnostic** (default, ~3–5 min): compute the baseline gap on 4
   images × 2 seeds. Look at `gap_curves.png` and the printed
   `integrated signed gap`. If it's clearly positive (paper predicts
   "yes"), DCW has a mechanism to bite.
2. **Tune DCW** (`--dcw_sweep`, ~20–40 min depending on grid): the same
   measurement, but across a grid of DCW `(scaler, schedule)` configs.
   The config that produces the smallest integrated `|gap|` is the one
   closing the bias best.

Schedule forms supported (pixel-mode only — wavelet sweeps are downstream):

| schedule | `scaler(i)` | intuition |
|---|---|---|
| `const` | `λ` | FLUX upstream default |
| `sigma_i` | `λ · σ_i` | paper Eq. 20 — stronger early, decays to 0 at σ=0 |
| `one_minus_sigma` | `λ · (1 − σ_i)` | inverse — stronger late |

## Data

Consumes cached latents + text embeddings from `post_image_dataset/`:

- `<stem>_<H>x<W>_anima.npz` — `latents_<H>x<W>` key, shape `(16, H, W)` float32
- `<stem>_anima_te.safetensors` — `crossattn_emb_v0..v7` keys, shape
  `(512, 1024)` bfloat16 (post-LLMAdapter, pad-masked, ready for DiT)

Both are produced by `make preprocess` and already sit in
`post_image_dataset/`. **No VAE or text encoder is loaded at runtime** —
the bench is DiT-only. Variable-resolution latents are fine: each
reverse trajectory runs on one sample's shape at a time.

## Usage

From `anima_lora/`:

```bash
# Diagnostic (is the bias even there?)
uv run python bench/dcw/measure_bias.py \
    --dit models/diffusion_models/anima-preview3-base.safetensors \
    --n_images 4 --n_seeds 2

# Tune DCW: sweep scaler × schedule
# NOTE: default grid is both-signed and biased negative on purpose — see the
# "Observed on Anima" section. Positive λ rows double as a direction check.
uv run python bench/dcw/measure_bias.py \
    --dit models/diffusion_models/anima-preview3-base.safetensors \
    --dcw_sweep \
    --dcw_scalers -0.3 -0.1 -0.03 -0.01 0.0 0.01 0.1 \
    --dcw_schedules one_minus_sigma const

# Faster preview
uv run python bench/dcw/measure_bias.py \
    --dit models/... \
    --n_images 2 --n_seeds 1 --infer_steps 12
```

Pick any trained checkpoint by pointing `--dit` at an already-baked
`safetensors` (see `scripts/merge_to_dit.py`). This bench doesn't load
LoRA adapters directly — feed it the model state you actually want to
measure.

Pass `--text_variant N` (0–7) to use a different cached caption variant;
`v0` is the canonical caption, `v1..v7` are augmented variants produced
during preprocessing. Default is `v0`.

## Output

Writes to `bench/dcw/results/<timestamp>/`:

| file | contents |
|---|---|
| `config.json` | CLI args verbatim |
| `summary.json` | configs ranked by integrated absolute gap |
| `per_step.csv` | wide format, one row per step × (v_fwd, v_rev, gap) per config |
| `gap_curves.png` | left: baseline forward vs reverse ‖v‖ (Fig. 1c repro); right: `gap(i)` overlay across configs |

And prints a headline ranking to stdout:

```
=== SNR-t bias measurement ===
baseline integrated signed gap: +X.XXX  (>0 means reverse > forward, as the paper predicts)

configs ranked by integrated |gap|  (smaller = closer alignment):
   1. λ=0.03_sigma_i                  |gap|= X.XXX  signed=+X.XXX
   ...
```

## Interpretation

Guidance for reading a run:

- **baseline signed gap > 0** → paper's finding reproduces; DCW (positive λ)
  has a mechanism to bite. Magnitude tells you how biased the default
  trajectory is.
- **baseline signed gap ≈ 0** → no reproducible bias on our model; DCW
  integration would be decorative. Investigate why (sampler difference?
  flow-matching vs DDPM? scale? CFG interaction?) before proceeding.
- **baseline signed gap < 0** → reverse velocity is *smaller* than forward
  at the same `t`; sign-flipped from the paper. Not necessarily a bug — on
  Anima at inference-matched settings this is what we observe (see
  "Observed on Anima" below). DCW's paper-form positive λ would *widen*
  |gap|; closing it requires negative λ.
- Under `--dcw_sweep`, expect the winning config's `|gap|` to be a
  sizeable fraction (30–80%) smaller than baseline. If the best config
  barely moves the needle, DCW is mechanistically right but
  under-powerful on our model — widen the `--dcw_scalers` range.
- If `signed_gap` changes sign as `λ` grows, you've crossed the minimum;
  narrow the grid around the sign-flip point.

## Observed on Anima — 2026-04-20 baseline

Run: `--infer_steps 28 --flow_shift 1.0 --n_images 6 --n_seeds 3`
(inference-matched to `make test`), plain Euler, no CFG, no DCW.

| quantity | value |
|---|---|
| integrated signed gap | **−409.9** (paper predicts +) |
| integrated \|gap\| | 417.7 |
| positive-gap regime | only step 1 (σ≈0.96), peak +2.8 |
| sign flip | step ~2 (σ ≈ 0.93) |
| late-step minimum | −61.6 at σ ≈ 0.036 |

The gap is **near-monotone negative and grows roughly linearly in step
index**, accelerating in the last ~5 steps. An earlier run at
`flow_shift=3.0, 20 steps` showed a biphasic shape (positive early,
negative late); that was a shift-schedule artifact. Under production
inference settings the bias is unambiguously negative.

This is the opposite sign from Yu et al.'s Key Finding 2. Plausible
reasons (speculative; all would benefit from ablation):

1. Flow-matching velocity has a different SNR-vs-norm relationship than
   the ε-predicting DPMs the paper measured.
2. Reverse trajectories at low σ live on (or near) the learned data
   manifold and produce small residual velocities; forward-noised real
   latents at the same σ sit slightly off that manifold and elicit
   larger "pull-back" velocities from the network.
3. Accumulated drift: by step `i`, reverse and forward branches are at
   different manifold locations, and the velocity field's norm differs
   systematically there.

Implication for DCW: the paper's `+λ · σ_i` schedule would amplify the
gap on Anima. To test DCW's mechanism here we need **negative λ** and a
schedule that concentrates correction at low σ (where |gap| is largest).
The default `--dcw_scalers` grid in `measure_bias.py` has been updated
to reflect this.

## Next actions

Work items, ordered by dependency. Each is a self-contained step.

1. **Run the negative-λ sweep** with the updated defaults:
   ```bash
   uv run python bench/dcw/measure_bias.py \
       --dit models/diffusion_models/anima-preview3-base.safetensors \
       --dcw_sweep --infer_steps 28 --flow_shift 1.0 \
       --n_images 6 --n_seeds 3
   ```
   - **Mechanism test:** does negative λ monotonically reduce |gap|, and
     positive λ widen it? If yes → the `prev − x0_pred` residual is a
     valid, appropriately-signed lever on Anima.
   - **Best-config shortlist:** the (λ, schedule) pair with smallest
     |gap|, plus its neighbours, feeds step 3.
   - **Direction-sanity:** positive λ rows act as a built-in sign check.
     Any result where positive λ improves |gap| means the measurement or
     my interpretation is wrong — stop and debug.
2. **Sweep sanity-check at a different flow_shift** (`--flow_shift 3.0`
   or `5.0`) to confirm the negative-gap finding isn't shift-schedule
   specific. One run, baseline only (no sweep). If the sign flips back
   to positive at a different shift, the "biphasic at shift=3" pattern
   from earlier is more informative than dismissed — revisit.
3. **Image-quality validation of the winner**. Closing |gap| is a
   mechanistic victory, not a perceptual one. Take the top `(λ,
   schedule)` from step 1, generate a panel of ~16 images with fixed
   seeds at baseline vs. best-DCW vs. a direction-flipped config, and
   inspect. Optional quantitative: a quick CLIP-image consistency or
   FID-lite on a held-out prompt set.
4. **Decision point.** After steps 1–3:
   - If |gap| closes *and* samples improve → proceed with `plan.md`
     integration, but invert the default-scaler sign and change the
     default schedule to `one_minus_sigma`. Update the plan's §3 sweep
     grid to reflect empirical findings.
   - If |gap| closes *but* samples are unchanged → DCW is mechanistically
     applicable but perceptually inert on Anima. Document and shelf;
     follow up with the speculative proposals in `proposal.md §4`
     (principled T-LoRA schedule, DCW-folded APEX regularizer, etc.).
   - If |gap| does not close with either sign → the premise fails on
     flow-matching DiTs. Write up the negative result and pivot.
5. **(Optional) Norm-matching alternative.** If DCW proper doesn't help,
   test a direct `v_rev *= ||v_fwd|| / ||v_rev||` correction at each
   step. The measured gap curve is exactly the multiplier. This is a
   stronger intervention than DCW — no (λ, schedule) to tune — and
   bypasses the question of whether `prev − x0_pred` is the right
   residual direction on Anima.

Throughout: extend `--n_images` and `--n_seeds` on any run whose result
you want to cite as a finding rather than a direction-check. The current
`6 × 3` is enough to see the shape; `16 × 5` is about the floor for
narrower claims.

## Caveats

- **CFG is not applied.** The measurement runs conditional-only. This
  simplifies the forward/reverse comparison (same conditioning on both
  branches) but means measured gaps won't match paper magnitudes
  precisely. The *shape* of the gap curve is what we want.
- **Each reverse trajectory uses a single seed's initial noise** and a
  single forward noise per step. The forward-noise RNG is re-seeded
  identically across DCW configs so the forward branch is bit-identical;
  any gap-curve delta across configs is attributable to DCW changing the
  reverse trajectory alone.
- **We measure pixel-mode DCW only.** Wavelet modes (`low`, `high`,
  `dual`) add DWT/iDWT per step and introduce two more hyperparameters.
  Once pixel mode is shown to help, add a wavelet pass as a follow-up.
- **Sample size is small on purpose.** 4 samples × 2 seeds × 20 steps is
  enough to see directional effects at a glance; it won't give
  publication-quality error bars. Bump `--n_images` and `--n_seeds` for
  tighter estimates.
- **Cached VAE latents are treated as `x_0`.** The paper uses real
  images directly; here the analogue is real-image *latents* (the space
  the DiT operates in). This is the correct choice for an
  Anima-intrinsic measurement.
