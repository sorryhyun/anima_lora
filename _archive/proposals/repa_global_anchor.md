# REPA global-anchor arm — re-inject the discarded global component

Status: **ARCHIVED — REFUTED by A/B (2026-06-13).** The Phase-0 *target* probe
passed (patch-mean + z-score is the most discriminative global target), and the
trainable arm was wired default-off, but the actual A/B closed the line:
`--repa_global_weight` at both **0.01 and 0.001 degraded** the CMMD val signal,
with no offsetting style gain. Re-injecting the global/DC component the
relational arm deliberately strips makes things worse — the stripped band is a
distractor, not lost signal.

This is not a dead end so much as a **signpost**: read on the frequency axis,
this arm is the −1 endpoint (add DC back → worse), the shipped `spatial_norm` is
the 0 point (strip DC), and the live successor pushes to the +1 endpoint (strip
a *broader* low band). That successor is **REPA-DoG**
(`docs/proposal/repa_dog_target.md`), which inherits this result as its
strongest local datapoint. The wired knobs (`repa_global_weight`,
`REPAGlobalHead`, `global_anchor_loss`, the patch-mean calib) remain in tree,
default-off and inert; leave them for now or strip in a later cleanup.

---

_Original proposal below (retained for the Phase-0 probe findings, which stand)._

Status (original): **WIRED (default-off) — A/B not yet run.** Phase 0 PASSED (target
validated) by a data-only probe that settled the open design question (which
global target, which normalization) *before* any compute. The trainable arm is
now implemented per the Design below: `repa_global_weight` (default 0.0 ⇒
byte-identical no-op) in `configs/methods/lora.toml`, the patch-mean z-score
calib shipped at `networks/calibration/pe_patchmean_stats.safetensors`
(regen: `uv run python bench/repa/build_calib.py`), `REPAGlobalHead` +
`global_anchor_loss` in `library/training/repa.py`, the `repa_global` loss
handler in `library/training/losses.py`, and the kwarg allowlist in
`networks/__init__.py`. Run the A/B with `--repa_global_weight 0.03` (below).

Probe: `bench/repa/discriminability.py` (run
`bench/repa/results/20260613-1249-discriminability/`, N=3058, no model
forward). Depends on REPA v2
Phase 0 (closed 2026-06-12, relational/Gram arm won — `docs/experimental/repa.md`).

## Why now

Reading the REG paper (NeurIPS-2025, repo root) raised one transferable
question. REG's own ablations (Tab. 5/6) say the gain comes from the
**high-level global/class signal**, not the patch alignment. Our validated
relational arm does the opposite: it aligns the PE-Spatial *patch* grid and
`spatial_norm` **deliberately standardizes the per-token global component away**
to keep pairwise cosines informative. So we are provably throwing out exactly
the signal REG credits. The full REG mechanism (a denoised class-token *input*)
is **not portable to LoRA-on-frozen-base** — the base DiT has no pretrained
handling of an extra token slot and a low-rank adapter can't grow one — so the
only on-ramp is to add the global signal as a **training-only auxiliary
alignment target**, not an input.

That left two unknowns the user flagged from prior observation: the PE **CLS
cosine saturates at ~0.99 on near-twin pairs**, so a raw-CLS target could be a
near-constant (dead) objective. Both unknowns are now answered by the probe.

## What the probe settled

Data-only discriminability = AUC of `P(in-group cosine > out-group cosine)`,
pairs labeled by character/copyright/artist from `caption_index.json`, scored
under four target normalizations, for CLS vs pooled patch-mean (3058 images):

| target | raw | center | zscore | whiten |
|---|---|---|---|---|
| **CLS** (character) | 0.594 | 0.630 | 0.643 | 0.711 |
| **patch-mean** (character) | 0.730 | 0.748 | **0.764** | 0.794 |

- **Collapse confirmed.** Raw CLS nearest-neighbour cosine **0.984** (p95
  0.996), **33% of variance in one PC**. That single shared direction is the
  ~0.99 floor — and why raw CLS is a near-dead target (AUC 0.59).
- **Normalization is mandatory and works** (raw→center→zscore→whiten monotone).
- **The CLS is the wrong source.** Pooled **patch-mean beats CLS at every
  scheme**; patch-mean *raw* (0.73) already edges *whitened* CLS (0.71). The CLS
  token is the most collapsed thing in the file.
- **Pragmatic winner: patch-mean + per-dim z-score** — 0.764 / 0.712 / 0.807
  (char / copy / artist). A fixed precomputed affine, no per-batch covariance,
  ~as strong as whitening.

## Design

Add an optional **global-anchor term** to `library/training/repa.py`,
complementary to (not replacing) the relational arm:

1. **Target** = mean over PE-Spatial patch tokens (drop CLS, `pe[:, 1:].mean(1)`
   — the patch tokens already loaded; no new preprocessing), passed through a
   **fixed per-dim affine** `(x − μ)/σ` from a precomputed calib, then L2-norm.
   μ/σ are the dataset patch-mean stats — compute once, ship as a small
   `networks/calibration/pe_patchmean_stats.safetensors` (mirrors the
   channel-scaling / cond-stream calib pattern, `[[project_channel_scaling_cond_stream_needs_own_calib]]`).
2. **DiT side** = the same block-8 capture the relational arm already pools,
   reduced to one vector (mean over the pooled grid), projected by a tiny
   train-only head to `d_enc=768`, z-scored by the **same** calib, L2-norm.
3. **Loss** = `1 − cos(dit_global, pe_global_norm)`, weighted by a new
   `repa_global_weight` (default **0.0** ⇒ byte-identical no-op), added
   alongside the existing relational term in the `repa` loss handler
   (`library/training/losses.py::_repa_loss`).

This is genuinely complementary: `spatial_norm` removes the per-token global/DC
component from the relational arm, so the global-anchor term re-injects exactly
that — using the most discriminative available target rather than the dead CLS.

Config (`configs/methods/lora.toml` + allowlist in `networks/__init__.py`
`*_KWARG_FLAGS`, else inert + config-test fail):
`repa_global_weight` (0.0 off), `repa_global_norm = "zscore"`,
`repa_global_calib = "networks/calibration/pe_patchmean_stats.safetensors"`.

## A/B plan & pre-registered readout

Single arm vs current relational-only, same data/preset/steps:

- **Arm**: `--repa_global_weight 0.03` (≈ half the relational weight; the
  relational arm is ~4% of total loss at 0.05, so start smaller and tune up).
- **Primary metric**: CMMD val signal — lower
  wins. (FM val loss is uninformative here, `[[project_fm_val_loss_uninformative]]`.)
- **Hard gate (style)**: qualitative anime-style pass on the fixed sample
  prompts. The global/style axis is the v1 burn hazard; any visible style drift = FAIL
  regardless of CMMD.
- **Readout**: CMMD improves and style holds → keep, tune weight. CMMD flat /
  style drifts → close; the spatial arm already owns the usable structure.

## Risks / honest limits

- **Effect is expected modest.** AUCs are 0.71–0.81, not separation. This is a
  small auxiliary nudge, not a headline lever.
- **Label proxy understates and overstates.** Different characters by the same
  artist share global style → out-group not purely negative (true
  discriminability likely a bit higher); but it also means the target encodes
  *style*, which is the exact thing the style gate must police.
- **Encoder caveat.** PE is CLIP-lineage; REG's Tab. 3 found CLIP-L class tokens
  much weaker than DINOv2 for this trick. We sidestep the class token entirely
  (patch-mean), but the global PE signal may still be a weaker anchor than a
  self-supervised encoder would give.

## Explicitly NOT doing

- **REG input entanglement** (denoised class-token slot) — incompatible with
  LoRA-on-frozen-base; do not re-propose. See `[[project_repa_v2_relational_won]]`.
- **Raw-CLS alignment** — bench-dead (AUC 0.59). Do not re-propose.
- **Whitening as the live target norm** — marginal over zscore, needs dataset
  covariance at train time; zscore affine is the chosen form.
