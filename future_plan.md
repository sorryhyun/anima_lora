# Anima — Future Plan

Living roadmap for work after APEX Phase 1 lands. See `docs/methods/apex.md` for the
shipped APEX implementation and `proposal.md`-era history is now folded into
that doc. This file tracks what comes next, ordered by payoff × readiness.

## Status snapshot (2026-04-15)

- **APEX Phase 1 — implemented, trainable.** `make apex` runs end-to-end on
  the default preset. Warm-start from `anima-tlora-0415-12.safetensors`,
  scalar `ConditionShift(a=-1, b=0.5)`, ratio-based warmup/rampup
  (`0.20 / 0.10`), `L_mix` primary form, `apex_omega` weighting, 3 DiT
  forwards per step, block swap forced off.
- **Outstanding Phase 1 validation** — a full training run through rampup,
  NFE=1/2/4 visual sweep against the warm-start baseline, and a FID check on
  a held-out bucket. These are small tasks but they gate Phase 2.

## Immediate next (days, not weeks)

### 1. Phase 1 sign-off run

Run the default `apex.toml` for its full budget (currently 2 epochs, 11.6k
steps). Collect:

- Loss curves for `L_sup`, `L_mix`, `L_fake` separately — the aggregate loss
  alone is not diagnostic once L_mix and L_fake activate. Split the logging
  before the run starts.
- Drift of `ConditionShift.(a, b)` across training. Phase 0 on a 2D toy
  settled at `(-1.08, 0.62)`; if Anima drifts outside Table 7's stable region
  `a ∈ [-1.0, -0.5]`, `b ∈ [0.1, 1.0]`, something is wrong.
- NFE=1/2/4 samples against the warm-start baseline at fixed seed +
  prompt set, every `save_every_n_epochs` checkpoint.

Decision gate: NFE=1 coherent (not noise, not mode-collapsed gray) and
NFE=4 visually matching or exceeding the baseline at NFE=30. If not, the
most likely suspect is `apex_shift_lr_scale` being too high or the warmup
ratio being too short for this dataset — tune one at a time.

### 2. Per-term loss logging

`_process_batch_inner` currently aggregates `L_sup + L_mix + L_fake` into a
single `loss` before the accelerator accumulates. Add a `self._apex_loss_parts`
dict stashed alongside `_apex_aux`, and surface the three terms in the
logged metrics (same pattern as the functional loss). Without this, Phase 1
debug is fly-by-feel.

### 3. ComfyUI round-trip

Confirm that the current `save_weights` + `load_weights` path correctly
persists `apex.condition_shift.a`, `apex.condition_shift.b`. Quick script:
load an APEX checkpoint into a fresh network, read `network.apex_condition_shift.a`,
assert it matches the saved value. This is 30 minutes of work and unblocks
any custom-node integration later.

## Phase 2 — GRAFT integration (1–2 days after Phase 1 signs off)

The GRAFT candidate-generation phase is today's bottleneck: a 30-step
inference per candidate, per prompt, per iteration. APEX at NFE=1–4 collapses
this to 3–10% of its current cost, making interactive curation viable.

### Work items

- `scripts/graft_step.py`: accept `--apex_lora <path>` and, when set, use it
  only for the candidate generation sub-phase (training still uses the
  non-APEX LoRA). The two LoRAs are different artifacts; APEX is a
  distillation of the baseline, not a replacement.
- Sampler path: candidate generation should default to NFE=2 (conservative)
  and expose `--apex_nfe` for tuning. NFE=1 is headline but tends to lose
  diversity; NFE=2–4 hits the quality/speed sweet spot in the paper's
  Table 2.
- P-GRAFT interaction: the current P-GRAFT setting disables LoRA for the
  last 25% of denoising for diversity. At NFE=1–4 this becomes 0–1 steps of
  LoRA-off, which may or may not be meaningful. Worth checking whether
  P-GRAFT still helps at low NFE, or whether the APEX LoRA's narrower mode
  coverage makes P-GRAFT redundant.
- Wall-clock target: iteration time (training end → candidates ready) drops
  by ≥10×. Paper claims 15× end-to-end for the equivalent loop on 20B; at
  our scale 10× is a safe commitment.

### Risks

- **Mode coverage.** Distilled students have narrower sampling distributions.
  For GRAFT this may actually help (more stylistically consistent candidates
  surviving curation) — but if the curation loop starts converging on a
  single style, that's a sign we've lost too much coverage. Mitigation:
  keep the baseline FM LoRA in the loop for the first few iterations, swap
  in the APEX LoRA once the survivor set is stable.

## Phase 3 — Fast inversion (1 day)

`archive/inversion/invert_embedding.py` today does multi-timestep averaging per step:
for each optimization step, it runs the DiT at several `t` values and
averages the loss. With an APEX LoRA the network is consistency-trained,
so a single endpoint-prediction query at large `t` (effectively NFE=1)
should suffice.

### Work items

- `--apex_mode` flag on `invert_embedding.py` that loads an APEX LoRA and
  replaces the multi-`t` loss with a single endpoint query at `t ∈ [0.75,
  0.95]` (sampled per step for stochasticity).
- Benchmark per-image inversion wall-clock on `post_image_dataset/` corpus
  against the current path. Target: ≥5× faster.
- Quality check via `archive/inversion/interpret_inversion.py`. Accept degradation
  on OOD targets — document which prompts break, don't block on them.

### Risks

- **Single `t` is noisy.** The current multi-`t` averaging exists because
  any single `t` gives a high-variance gradient. A consistency-trained
  student should have lower variance by construction, but "should" is not
  "does" — measure and fall back to a mini-average (e.g. 3 `t` values) if
  single-`t` loss diverges.

## Phase 4 — T-LoRA / OrthoLoRA-compatible APEX (week-scale)

Phase 1 is plain LoRA only. T-LoRA (timestep-dependent rank masking) and
OrthoLoRA (SVD-parameterized) are the methods we actually prefer for
training, so APEX needs to compose with them.

### Why it's deferred

T-LoRA varies effective rank with `t` via a power-law schedule, driven by
`network.set_timestep_mask(timesteps, max_timestep=1.0)` called in
`get_noise_pred_and_target`. APEX's 3 forwards use different timesteps:

1. Real forward at `t`.
2. Fake-sg forward at `t` (same).
3. Fake-on-fake forward at `t_fake` (fresh).

The timestep mask is mutable state on the network, so forward #3 requires
re-setting the mask mid-step, and we need to verify nothing in the autograd
graph of forwards #1/#2 depends on the mask state changing. Plain LoRA
dodges this because its mask is a no-op.

OrthoLoRA is stricter: it uses SVD-based parameterization (`P·diag(λ)·Q`)
with an orthogonality regularization term that currently assumes a single
base forward. `save_weights` already converts OrthoLoRA → plain LoRA via
thin SVD at save time, which is why we can *warm-start* APEX from a T-LoRA
checkpoint even though APEX itself trains plain LoRA. Training OrthoLoRA
*with* APEX is a different question and needs the regularization term to
be re-derived for the 3-forward case.

### Work items

- T-LoRA: audit `set_timestep_mask` call sites, add a scoped "set and
  restore" pattern so each APEX forward gets the right mask and state is
  restored after. Unit test that the real-forward backward produces
  identical gradients with/without the fake forwards interleaving.
- OrthoLoRA: decide whether the ortho regularization runs on the combined
  `F_real + F_fake_on_fake_xt` path or just on real. Probably just real —
  the fake branch is a target, not the "main" model — but this needs a
  paragraph of thought and a short toy experiment.
- Phase 0-style 2D bench for both variants before touching the Anima
  training loop. `archive/bench/apex_phase0.py` is the template.

## Phase 5 — ComfyUI loader (2 days)

`custom_nodes/` already has a HydraLoRA loader pattern. APEX needs the same
treatment:

- Node that loads `*.safetensors` and splits LoRA keys from
  `apex.condition_shift.*` keys.
- At runtime, applies `c_fake = a·c + b` to the text conditioning **before**
  the sampler's DiT forward. The sampler itself is unchanged — only the
  conditioning gets rewritten.
- Backward-compat: loading an APEX checkpoint into a vanilla LoRA loader
  node should still work (plain LoRA delta), but generates visibly worse
  output because the network was trained against a shifted condition. Warn
  on load if `apex.condition_shift.*` keys are present but the node is the
  plain-LoRA variant.

## Infrastructure / debt

Parallel to the APEX phases, there is a small pile of cleanup that will make
future work easier and shouldn't be deferred indefinitely.

### Remove the cold-start guard string mismatch

`train.py:1625` raises with "APEX training requires either --apex_warmup_ratio > 0
(or --apex_warmup_steps > 0) or --network_weights <path>". The current wording
is fine but should be revisited once `apex_warmup_steps` is deprecated in favor
of ratios (see below).

### Deprecate `apex_warmup_steps` / `apex_rampup_steps`

The absolute-step variants were added as explicit overrides during the steps
→ ratio migration. They're useful now as an escape hatch but will rot if
left forever. Once two APEX training runs have completed successfully on
ratio-only config, drop the absolute args and simplify the resolution logic
in `_process_batch_inner`.

### Docs

- `docs/guidelines/training.md` references the historical method list. Add APEX.
- `CLAUDE.md` mentions all LoRA variants in the "Key entry points" table.
  Add a row for APEX / `ConditionShift`.

### Tests

- `tests/test_apex_loss.py`: port Phase 0's gradient equivalence test
  (`L_mix` vs. `G_APEX`) to a real-network-shaped toy. Blocks future
  refactors from silently breaking Theorem 1.
- `tests/test_condition_shift.py`: verify scalar / diag / full forward
  shapes, dtype safety under bf16 autocast, and param counts.
- `tests/test_apex_schedule.py`: boundary cases of
  `apex_schedule_weights` — 0 warmup, 0 rampup, step before/at/after
  warmup boundary, rampup saturation.

## Explicitly not doing

- **Not** re-enabling block swap for APEX. Multi-forward + backward-hook
  block swap is a rabbit hole; the user-facing fix is to reduce batch /
  resolution instead.
- **Not** implementing the released upstream `apex.py` time-sign variant.
  It requires signed-timestep handling in our `t_embedder` that we do not
  want to maintain.
- **Not** maintaining an EMA teacher. The paper's stop-gradient is
  sufficient (§B.1) and Phase 0 confirmed this on the toy. EMA can be
  added later as a stabilization knob if long training runs turn out to
  need it.
- **Not** replacing existing LoRA / T-LoRA / HydraLoRA methods. APEX is an
  additional training objective for the specific one-step-distillation use
  case; baseline LoRAs remain the quality ceiling for many-step inference.
