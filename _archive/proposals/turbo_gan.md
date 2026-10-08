# Plan — porting FastGen learnings into Turbo (+ a network abstraction seam)

Status: **ARCHIVED — SUCCESSFUL (2026-06-17).** The FastGen-style teacher-feature
GAN head landed in the distill loop (`scripts/distill_turbo/`,
`networks/methods/turbo_dmd.py` — discriminator + GAN losses) and proved out.
Moved out of active proposals. Original status preserved below.

Status: **proposal / not started**. Source of the ideas: NVlabs **FastGen** (`FastGen/`,
Apache-2.0), specifically `fastgen/methods/distribution_matching/{dmd2,f_distill}.py`,
`fastgen/networks/discriminators.py`, and `fastgen/networks/network.py`'s uniform net
interface. This doc scopes three changes, ordered by value/risk, with decision gates so
each can be killed independently.

Our Turbo (`scripts/distill_turbo/`, `networks/methods/turbo_dmd.py`) is a hand-rolled
**DP-DMD**: teacher CFG real-score vs. fake-score VSD surrogate, plus a diversity-anchored
first step. It is, structurally, **DMD2 with the GAN amputated**. Ideas 1 and 2 add the
missing adversarial machinery; idea 3 is the refactor that makes 1 (and future
distillation methods) cheap to wire.

---

## Idea 1 — DMD2 teacher-feature GAN head (the main one)

### What FastGen does
`DMD2Model` (`dmd2.py`) trains **VSD + GAN**. The discriminator is *not* a separate
network: it taps the **frozen teacher DiT's intermediate block activations**
(`feature_indices`, default = middle block) and runs a **~0.5M-param 2-layer conv head**
on top (`Discriminator_ImageDiT`: `[B, inner_dim, H, W] → conv → GroupNorm → LeakyReLU →
conv1x1 → AdaptiveAvgPool → logit`). Losses are plain softplus hinge (`common_loss.py`):

- generator: `softplus(-fake_logits).mean()`
- discriminator: `softplus(fake_logits).mean() + softplus(-real_logits).mean()`
- optional approximate-R1 on the disc head (APT-style).

QwenImage config (closest net to Anima — 16-ch latent, flow-matching DiT) uses
`gan_loss_weight_gen=0.03`, `gan_use_same_t_noise=True`, `fake_score_pred_type="x0"`,
CFG=4.0, `student_sample_steps=4`. Those are real starting hyperparameters, not guesses.

### Why it fits Turbo cheaply
- The teacher (frozen base DiT) is **already resident** in the distill loop — the GAN
  reuses it as a feature extractor; only the tiny conv head is new trainable weight.
- Our loop **already alternates** student vs. fake/critic updates
  (`fake_steps_per_student_step`, `distill.py` §5). The discriminator trains on the **same
  cadence as the fake** — co-located in the fake-update block.
- The generator GAN term slots into the existing student-loss assembly
  (`distill.py:716`, `loss_student = loss_dmd`) as one additive term.

### Integration points (real seams)
- **Feature tap.** Anima DiT exposes `self.blocks` (ModuleList) + `self.final_layer`
  (`library/anima/models.py:1257,1271`). Spectrum already registers a
  `register_forward_pre_hook` on `final_layer` to capture block output
  (`networks/spectrum.py:296-303`). Reuse the identical pattern: a forward hook on
  `anima.blocks[idx]` captures that block's token output `(B, L, D)`. One hook,
  removed in a `finally`.
- **Token → conv layout.** Blocks emit flattened tokens `(B, L, D)`; the conv head wants
  `(B, D, H_p, W_p)`. Un-flatten using the sample's patch grid (derivable from the latent
  shape / bucket). **This is the main subtlety** under native-shape bucketing
  (`library/datasets/buckets.py`): each forward runs at its real token count, and under
  `compile_blocks` the forward is fake-5D-flattened, so the (H_p, W_p) reshape must come
  from the *latent*, not a static constant. A pooled-token variant (mean over L → a 1-D
  MLP head) sidesteps the reshape entirely and is the safer v0 — start there, only go
  spatial if v0 underperforms.
- **New trainable module + optimizer.** Mirror the fake: a small `Discriminator` (port
  `Discriminator_ImageDiT`, or the pooled-MLP v0) with its own optimizer, stepped inside
  the fake-update block. `TurboDMDNetwork` gains a `disc` handle + `disc_params()`;
  `distill.py` gains a third optimizer.
- **Config.** Add `gan_loss_weight_gen` (default **0.0** → exact current behavior),
  `gan_feature_block_idx`, `gan_disc_lr`, `gan_r1_weight` to `distill_turbo/config.py` +
  `configs/methods/turbo.toml`. Off-by-default keeps the shipped recipe byte-identical.

### torch.compile / block-swap caveats (flagged from memory)
- `compile_blocks()` traces the block forward; a forward hook added **after** compile may
  land on a stale graph. Register the disc hook in the same place block hooks are managed,
  and re-verify under `--torch_compile` before trusting numbers.
- Block-swap desyncs on extra DiT forwards.
  The GAN adds **no new teacher forward** if we tap the disc features from the
  teacher forward we already run for the real score — capture features *during* the
  existing `_teacher_cfg_velocity` call rather than a second pass. Audit this explicitly;
  it's the difference between "free" and "+1 forward/step".

### Cost / tradeoff (be honest)
Turbo loses some of its current "two-optimizer, plain-LoRA-out" simplicity (third
optimizer + a non-shipped disc module — already precedented by the non-shipped fake).
The **student output stays a plain LoRA** (disc is training scaffolding, discarded at
save, exactly like `fake`). Net new cost: one tiny module, one optimizer, one hook.

### Decision gate 1
A/B in `bench/turbo/` (create if absent — Tier 1.5 needs a bench script + invariant test):
`gan_loss_weight_gen=0` vs `0.03`, fixed seed/data/steps, 2-step `--cfg 1.0` inference.
Ship only if GAN measurably improves sample quality (CMMD / human A-B) **without**
diversity collapse (reuse `diversity.py` validation). Kill if no signal or if the compile/
block-swap interaction proves fragile.

---

## Idea 2 — f-distill reweighting (stacks on Idea 1, ~free)

### What FastGen does
`FdistillModel` (`f_distill.py`, ~180 lines, subclass of `DMD2Model`) reweights the VSD
loss by an f-divergence weight `h = f'(r)`, where the density ratio
`r = exp(disc_logits)` comes **for free** from the GAN head added in Idea 1. Weighting
functions for `{kl, rkl, js, sf, neyman, sh, jf}` are a one-liner table (`f_distill.py:20`).
Numerical care worth copying: clamp logits to ±10, clamp `r` to `[ratio_lower, upper]`,
and an **EMA per-timestep histogram normalization** of `r` (bins over `t`, lines 72-98)
so the weight isn't dominated by the t-distribution of the batch. Reported wins:
CIFAR FID 1.99→1.85, ImageNet 1.12→1.11 — small but consistent, and it targets
mode-collapse, which is exactly what DP-DMD's diversity term also fights.

### Why it's cheap here
Our student loss is already a **detached-signal × x_pred** DMD surrogate
(`grad_signal`, `distill.py:678,713`). f-distill is just multiplying that signal by a
per-sample scalar `h(t, r)`. No new forward, no new module — only the disc logits
(present once Idea 1 lands) and the `h` computation.

### Integration points
- `grad_signal = grad_signal * h.view(B,1,1,1)` right before the surrogate assembly.
- Port `_get_f_div_weighting_h` + the `bins` EMA buffer (register on the student net so it
  serializes — but it's training-only scaffolding, so it does **not** need to ship in the
  saved LoRA; verify it's stripped by `save_student`'s `.lora_/.alpha` key filter,
  `turbo_dmd.py:326`).
- Config: `f_div` (default `"rkl"` ≡ uniform h ≡ plain DMD2, so off-by-default), plus
  `ratio_lower/upper`, `ratio_ema_rate`, `bin_num`.

### Decision gate 2
Strictly a **phase-2 follow-on**: requires the disc from Idea 1. A/B `f_div="rkl"` (no-op)
vs `"js"`/`"kl"` on the same `bench/turbo/` harness. Low priority — only chase if Idea 1
ships and we want to push the diversity/quality frontier further. Interaction risk: the
DP-DMD diversity anchor and f-distill both attack mode-collapse; they may not be additive.
Bench both-on vs each-alone.

---

## Idea 3 — Network abstraction seam (the enabling refactor)

### What FastGen has
Every network exposes a **uniform** call surface (`fastgen/networks/network.py`):
`net(x, t, condition, fwd_pred_type="x0"|"v"|..., feature_indices=..., return_features_early=...)`
over a `noise_scheduler` abstraction (`forward_process` / `sample_t` / `latents` /
`max_t`). DMD2/f-distill/CM/MeanFlow are all written **once** against that interface and
run unchanged across EDM→SDXL→Flux→QwenImage→Wan. The two hooks that matter for us are
`feature_indices` + `return_features_early` — that's the *exact seam* Idea 1 needs.

### What we have
Turbo open-codes the equivalent: `_forward(view, x, t, cond, no_grad)` wraps the Anima DiT
and squeezes dim-2 (`distill.py:340`), `renoise` / `get_timesteps_sigmas` stand in for the
scheduler. It works, but the feature tap for Idea 1 currently has **nowhere clean to live**
— hence the forward-hook workaround above.

### Proposal (scoped — NOT a framework rewrite)
Do **not** adopt FastGen's whole abstraction. Add the minimal seam that pays for Idea 1
and any future distillation method:

1. **A feature-tap option on the Anima DiT forward.** Add an opt-in
   `return_block_features: set[int] | None` (default `None` = current behavior, zero cost)
   to `models.py::Anima.forward`, returning `(velocity, {idx: feat})` when requested. This
   replaces the fragile external forward-hook with a first-class, compile-visible path —
   directly removing the biggest risk in Idea 1's gate. Keeps the dim-2 / native-flatten
   handling *inside* the model where the bucket grid is known (resolves the token→conv
   reshape cleanly).
2. **Leave `_forward` / `renoise` / scheduler as-is.** They're fine. Don't chase a
   `noise_scheduler` object — we already have `get_timesteps_sigmas` + `renoise` and a full
   abstraction is a large refactor with no immediate payoff.

### Why bundle it here
The feature tap is the shared dependency of Idea 1. Building it as a real model API (3.1)
instead of a hook is strictly better under torch.compile and block-swap, and it's the one
piece of FastGen's abstraction that earns its keep right now. The rest of FastGen's
network-agnostic surface is noted as **future reference**, not in-scope.

### Decision gate 3
3.1 (the feature-tap API) is a **prerequisite** for a robust Idea 1 — build it first if
Idea 1 is greenlit, and prove it's a bit-exact no-op when `return_block_features=None`
(invariant test: compiled + eager forward unchanged vs. `main`). 3.2 (full scheduler
abstraction) is explicitly **out of scope** / not-now.

**Status: 3.1 LANDED.** `forward_mini_train_dit` now takes
`return_block_features: set[int] | None` + `return_features_early: bool`
(`_run_blocks` gained `capture_blocks` / `feature_sink` / `stop_after_block`); both
default off → bit-exact no-op (`tests/test_feature_tap.py`). Idea 1's GAN forwards
now use it instead of the external block hook: the grad-bearing **generator** forward
early-exits at the deepest tap, so only `blocks[0..k]` run and retain activations —
this is what fixed the GAN OOM (the full-stack grad forward was the culprit).
Unsupported with block swap (raises) — the Turbo teacher is resident, so that never
fires in practice.

---

## Sequencing

```
3.1 feature-tap API  ──►  Idea 1 GAN head  ──►  Gate 1 (bench/turbo A/B)
   (bit-exact no-op test)     (off by default)        │
                                                       ├─ ship (off-by-default flag) ─► Idea 2 f-distill ─► Gate 2
                                                       └─ kill ─► revert, keep 3.1 (harmless no-op)
```

- **3.1 first** (small, testable, de-risks the rest). ~1 focused change + invariant test.
- **Idea 1** behind `gan_loss_weight_gen=0.0`. Largest single change; gated on bench.
- **Idea 2** only if Idea 1 ships and we want more. Tiny diff on top.
- **3.2** never, unless a second distillation method (sCM/MeanFlow) actually lands and
  wants it.

## Risks / open questions
- **Free-vs-+1-forward**: confirm disc features can be captured from the *existing* teacher
  forward (real-score pass). If not, the +1 forward/step changes the cost math and the
  block-swap audit.
- **Diversity ↔ GAN ↔ f-distill** may be partially redundant (all fight mode-collapse).
  Bench combinations; don't assume additivity.
- **Compile fragility**: the whole reason 3.1 exists. If 3.1's no-op test passes, Idea 1's
  compile risk mostly evaporates.
- **Out of scope**: video discriminators (`Discriminator_VideoDiT`), CM/MeanFlow/LADD/
  self-forcing, FSDP2 meta-init — all noted in FastGen as reference, none in this plan.

## Contributing tier
GAN head + f-distill are numerics-changing → **Tier 1.5** (bench script + invariant test
required; `bench/turbo/` + `bench/_common.py` envelope). Feature-tap API is a no-op-by-
default model change → invariant test only.

## References
- FastGen: `FastGen/fastgen/methods/distribution_matching/dmd2.py`, `f_distill.py`;
  `FastGen/fastgen/networks/discriminators.py`; `FastGen/fastgen/methods/common_loss.py`;
  `FastGen/fastgen/configs/experiments/QwenImage/config_dmd2.py`.
- Ours: `networks/methods/turbo_dmd.py`, `scripts/distill_turbo/distill.py`,
  `library/anima/models.py` (DiT blocks), `networks/spectrum.py` (block-hook precedent).
- Papers: DMD2 (Yin et al. 2024, arXiv:2405.14867), f-distill (Xu et al. 2025,
  arXiv:2502.15681), DP-DMD (Wu et al., arXiv:2602.03139 — our current Turbo).
- Docs: `docs/experimental/dpdmd.md` (ops), `docs/structure/dpdmd.md` (structure).
</content>
</invoke>
