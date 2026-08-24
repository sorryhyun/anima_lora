# Mod guidance — finished (2026-08-24)

Text-conditioned AdaLN steering via a distilled `pooled_text_proj` MLP
(Starodubcev et al., ICLR 2026). **Status: finished — shipped, and every
question the line asked has a terminal answer.** The distillation trainer
(this directory) was moved here from its old `distill_mod` home under
`scripts/` when the line
finished; the `distill-prep` / `distill-mod` targets were removed at
the same time (one-shot surface — the head is re-distilled only when the base
DiT changes). Ops: [`README.md`](README.md). The feature itself is unaffected
and its doc stays canonical: [`docs/inference/mod-guidance.md`](../../../docs/inference/mod-guidance.md).

## Shipped artifacts

- **`pooled_text_proj_0413.safetensors`** (~12 MB) — the distilled head, a
  GitHub release asset on this repo, auto-downloaded on first use by the
  ComfyUI node.
- **CLI**: `inference.py --pooled_text_proj <path> --mod_w …`, or `make test MOD=1`
  (auto-discovers the newest head in `output/ckpt/`). Composes with `SPECTRUM=1`.
- **ComfyUI**: folded into
  [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler)
  — `mod_w_profile` dropdown on the unified sampler, an Advanced variant with
  the raw sliders, and a standalone `MODEL → MODEL` patcher node.
- **Two validated profiles**: `step_i8_skip27` (default) and `step_i14` (safe,
  for LoRAs that show anatomy drift).

## Why it's finished

The bench (`bench/mod_guidance/`) was archived 2026-07-12 →
`_archive/bench/mod_guidance/` because every axis it probed came back
terminal. Verdicts, with evidence in
[`docs/findings/mod_guidance_quality_tag_axis.md`](../../../docs/findings/mod_guidance_quality_tag_axis.md):

- **What the head is**: a global tone / contrast / finishing operator — a
  polish knob conditional on a good base, **not** a content editor and not a
  "quality rescue". The original "quality axis" geometry framing is demoted
  (it was a content-magnitude axis; named-entity tags drive it 3–4× harder than
  `score_9`).
- **Schedule axes — both falsified.** σ-gating is dead (the whole effect is the
  σ≥0.45 structure-forming steps; the tail can't be dose-bought) and the layer
  axis is dose, not placement (between-block SSIM std below the noise floor;
  partial arms interpolate `off`→`full`). The hand-set `8–26` full-dose ships
  validated — no taper, no learnable per-block `w` allocator.
- **Can't be made more quality-selective by retraining.** The distilled proj is
  a tag-agnostic ~3× amplifier; the selectivity lives upstream in the base
  encoder (`rel_dpool` 0.059 quality vs 0.031 content). Conditioning the
  distill teacher on quality tags cannot move that ceiling.
- **Can't carry a content direction — architectural, not a fit gap.** The
  head's text-*derivative* is orthogonal to the teacher's (cos ≈ 0 at every σ,
  within ~1 SE) because the teacher's text response is ~99% AC while AdaLN can
  only write DC. A geometry-aware (GAD) distillation term and a σ-FiLM head
  were both wired and run to lift it; neither moved `cos` off zero. GAD ships
  at `gad_weight=0` (dead), σ-FiLM inert even when opted in.

## Open remainder

- **Anima-2.9B (40 blocks) is untested.** The head is depth-agnostic in shape
  (one MLP into the timestep embedding, `model_channels` 2048 on both), but
  the shipped weights were distilled against the 28-block teacher, and the
  block schedule's `--mod_end_layer` default tracks `num_blocks-1`. A 2.9B
  re-distill is the one thing this tree would still be run for.

## If it reopens (base-DiT change)

Re-distill with the loop in this directory (see [`README.md`](README.md)), then
resurrect `text_jacobian.py` + `channel_attribution.py` from
`_archive/bench/mod_guidance/` as the acceptance probes — and probe the head on
**its own training distribution** (a first run on real latents inflated the
error floor because the head was synth-trained).
