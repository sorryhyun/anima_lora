# CTCal's clean-teacher premise is inverted on Anima — don't build it

Evaluated CTCal (Guo, Ma, Zhang & Huang, "Rethinking Text-to-Image Diffusion
Models via Cross-Timestep Self-Calibration", CVPR 2026) for Anima. CTCal is a
LoRA fine-tuning method that adds an auxiliary loss `D(A_stu, A_tea)` aligning a
denoiser's cross-attention maps at a noisy timestep `t_stu` to the same
model's maps at a cleaner timestep `t_tea` (stop-grad on the teacher). Its
entire premise: cross-attention is sharp and well-localized at low noise and
degrades at high noise, so the clean step is a reliable teacher that
calibrates the noisy step.

On Anima the premise is inverted. Cross-attention is *blurriest* at the clean
end and *sharpest* at mid-to-high σ (≈0.75–0.85). The clean step CTCal needs as
its teacher is the worst-localized step; the noisy step CTCal wants to fix is
already Anima's sharpest. There is no clean→noisy quality gradient to distill, so
the method has nothing to exploit here. **Recommendation: do not build CTCal on
Anima.**

![Concentration and distinctness of content-token cross-attention maps vs σ, top
5 cross-attention blocks, scored on visible/clothing tokens (n=64). Both metrics
rise with σ from a minimum at the clean end (green band = CTCal's intended
teacher region) to a peak at σ≈0.85, then dip slightly at σ=0.95. The premise
requires the opposite slope.](assets/ctcal_inverted_premise.png)

## How it was tested

`_archive/bench/ctcal/probe_teacher_signal.py` (archived) — a no-grad
property probe over cached
`(latent, caption)` pairs on the base DiT, in the spirit of the REPA Phase-0
probes. For each image it noises the real cached latent to a σ grid, runs one
eager forward, and recovers each cross-attention block's `softmax(QK^T)` over the
text axis via a forward-pre-hook that recomputes the QK-normed q/k with the
module's own `compute_qkv` (flash never materializes the matrix, so we recompute
it for the content columns). Per content-token spatial map `M_c(i) = A[i,c]`
(Prompt-to-Prompt convention). Three rulers per block × σ:

- concentration = 1 − normalized spatial entropy of `M_c` (1 = a delta).
- distinctness = 1 − mean off-diagonal cosine between content-token maps.
- teacher_gap = 1 − cosine to the `σ_tea` map.

Anima is a flow-matching DiT with PixArt/Hunyuan-style separate cross-attention
blocks, so the mechanism is architecturally applicable — extraction was never the
blocker. The probe targets the empirical premise, not the plumbing.

## Why the conclusion is robust (three passes)

1. All-token aggregate (n=128). concentration rises from the clean end and
   peaks at σ≈0.65; teacher block 15 had conc 0.047 @ σ=0.45 vs 0.045 @ σ=0.95
   (flat), distinct 0.428 vs 0.457 (noisy end more distinct). Initial FAIL —
   but this scored over all content tokens, including `1girl`, artist /
   copyright tags and commas that have no spatial home and stay diffuse at every
   σ, diluting the average. The method (Stanza POS selection) does not score
   those, so this was not apples-to-apples.

2. Visible-token aggregate (n=64). Re-scored on the visible/noun tokens
   CTCal actually selects (`--score_filter visible`: clothing + visible objects
   /body, reconstructed from the 1:1-aligned cached `t5_input_ids`). Selection
   lifted the absolute signal (teacher block 13: conc 0.027 → 0.068 @ σ=0.45)
   — confirming token choice matters — but the direction did not change.
   concentration still rises to a peak at σ=0.75–0.85; both gate criteria fail in
   the *wrong direction*: conc 0.068 @ σ_tea < 0.071 @ σ_hi, distinct 0.562 <
   0.585. The plateau (σ 0.55–0.95 ≈ 0.071–0.076) is too broad and too close to
   the noisy end for any teacher σ to separate — `--sigma_tea 0.75` gives a
   1.07× concentration ratio (gate wants ≥1.15) and tied distinctness.

3. Best-case single image. `dan_10443908` (tags `pink sports bra`,
   `yoga pants` — a clean single figure that localizes beautifully in the
   renders) follows the same inverted curve, more cleanly: block 15 concentration
   0.018 @ σ=0.05 → 0.077 @ σ=0.85, distinctness 0.228 → 0.576. The striking
   render pattern is real, but it lives in the σ=0.65–0.85 columns; the
   leftmost (σ=0.05, CTCal's teacher) column is the blurriest of the row.

## Why Anima differs from the paper's SD2.1 / SD3 result

The paper itself flags that flow models under-train the clean end (logit-normal
sampler) and warns `t_tea=0` can hurt for SD3 — so we made the teacher σ a knob
and let the data place the peak. The likely mechanism: Anima is flow-matching
trained on booru tag-soup captions (512-token T5 sequences, dozens of tags
competing for attention), and [it resolves layout by mid σ](
sigma_signal_where_anima_resolves.md) (x0 reconstructs by σ≈0.45). Natural-language
COCO-style prompts on SD concentrate attention at the clean end; dense competing
tags + flow-matching push Anima's localizable structure to mid-high σ. The
`teacher_gap` is large (~0.59) but a red herring — maps differ across σ without
any quality advantage to the teacher, so distilling toward it transfers nothing.

## Reusable takeaways

- Anima cross-attention DOES localize — well, at mid-high σ (≈0.65–0.85), on
  noun/visible tokens. Useful for anything that wants to use attention maps
  (regional conditioning, layout control); just not for a clean→noisy
  distillation. The probe is the tool to confirm where, per block.
- Token selection is load-bearing for any attention-map metric here. Scoring
  over all content tokens buries the signal under meta tags; restrict to
  visible/noun tokens (the probe's `--score_filter`) or the conclusion can invert
  for the wrong reason. This is the same class of trap as
  [the spectral-fraction metric inversion](spectral_fraction_metric_inverts.md).
- The repo already ships the validated version of "inject spatial/representational
  structure during fine-tuning" — REPA — which beat its ablations. CTCal would
  be a weaker, more expensive cousin (a second forward per step that collides with
  the block-swap offloader) of capability already present.

## Status

Closed 2026-06-17. Bench archived to `_archive/bench/ctcal/` with its frozen
result runs. Do not re-propose CTCal for Anima without first re-running the probe
and showing the concentration/distinctness slope has flipped (clean end sharper).
