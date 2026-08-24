# Mod guidance: the pooled-text head is a global-tone / finishing lever — not a content or "quality" lever

<!-- check-docs: ignore-flags (folds in the demoted quality-axis geometry record; the geometry flags below are historical, not live) -->

Two independent probes converge on one verdict: the distilled `pooled_text_proj` →
AdaLN modulation head is a **global tone / contrast / finishing operator**, not a
content editor and not a "quality" lever.

1. **Image-space channel attribution** (`_archive/bench/mod_guidance/channel_attribution.py`)
   demotes the original "quality axis" framing to a *content-magnitude* axis and shows
   the pooled channel behaves like a grade/polish knob conditional on a good base.
2. **Text-Jacobian** (`_archive/bench/mod_guidance/text_jacobian.py`) explains *why*
   architecturally: the head matches the teacher pointwise but its text-*derivative* is
   orthogonal to the teacher's, and a DC/AC decomposition proves that ceiling is
   intrinsic — AdaLN can only write DC, the teacher's text response is ~99% AC.

Both the layer/σ *schedule* axes were also probed and killed; the shipped hand-set
`8–26` full-dose is validated. Details below, then the demoted geometry record kept as
a frozen mechanism map.

> **Bench ARCHIVED 2026-07-12** — every question this bench asked got a terminal
> answer, so `_archive/bench/mod_guidance/` moved to `_archive/bench/mod_guidance/` (scripts +
> all `results/`). The shipped *feature* is unaffected (`docs/inference/mod-guidance.md`,
> `project/finished/mod_guidance/`). If the pooled head is ever re-distilled (base-DiT change),
> resurrect `text_jacobian.py` + `channel_attribution.py` from the archive as the
> acceptance probes. A last one-shot geometry probe (`pool_decouple_probe.py`, run
> 2026-07-12, `_archive/bench/mod_guidance/results/20260712-1206-pool-decouple/`)
> confirmed: pooling choice is a non-issue (max-over-pad == max-over-real, cos 1.000;
> mean tracks max at ~0.9), max-pool saturation is mild (axis cos 0.94 at 12 tags), and
> the mod-neg wording rotates the tiny quality-tag steering delta (cos down to −0.38 at
> ‖δ‖≈0.02) — the "quality words barely drive the pooled axis" verdict restated, not a
> reopening.

---

## 1. What the pooled channel actually does (image-space channel attribution)

A tag edit enters the DiT through two separable inputs — the pooled/mod channel
(`max → pooled_text_proj → AdaLN`) and the cross-attn channel (full `crossattn_emb`
sequence), split via `pooled_text_override`. Full run (2026-05-31, 6 dense real
captions × 2 seeds, tags spliced at their *correct* `SLOT_ORDER` slot;
`results/20260531-0005-full/`):

| tag | pool_share (latent / PE) | cross_share | cos(cross,pool) |
|---|---|---|---|
| `score_9` | 0.50 / 0.77 | 0.97 | +0.27 |
| `masterpiece` | 0.50 / 0.71 | 1.04 | +0.27 |
| `holding a sword` (content) | 0.40 / 0.44 | 0.98 | +0.23 |

Reads (n=12/tag, pool_share σ≈0.2–0.3 — directional):

- **The mod channel is a real lever, not inert.** Quality tags route ~0.5 (latent) /
  ~0.7+ (PE) of their image movement through pooled, vs ~0.4 / 0.44 for the content
  tag — cross-attn is the larger single channel (~1.0), pooled a substantial secondary
  contributor, *more so for quality than content*. The original *intuition* (quality
  acts via pooled modulation) survives; the "axis"/geometry framing was the wrong part.
- **"Double-drive" is mild reinforcement, not over-saturation** (cos ≈ +0.26,
  additivity residual ≈ 0.63 — non-linear, not clean superposition).
- **DC-blowout is NOT reproduced** under the shipped schedule (blocks 8–26, tonal-DC
  blocks 0–7 protected): sweeping steering `w`→8 gives only +3% pixel-std drop / ~1%
  tone shift, and the effect *saturates* past w≈3. The doc's old pink/DC-blowout was an
  unprotected-schedule / duplication artifact.
- **Cross-attn is strongly order-sensitive** (reordering tags, pooled pinned, moves the
  image 64%/84% of a full seed change) — which is why tag placement is load-bearing.

**Qualitative read of the `swap` grids (eyeball, hypothesis, n=12/tag):** the pooled
channel behaves like a **global grade/polish operator** — consistent with AdaLN being a
per-feature-channel shift applied uniformly across all spatial positions (it can shift
tone/contrast/sharpness but cannot move content). When **BB is already good, BT
(pooled-only) reads slightly better** (a small finish on already-right content); when
**BB is poor, TB (cross-attn-only) beats BT** (only the content channel repairs
structure). This explains why `pool_share` is higher in PE (~0.7, tuned to global tone)
than latent (~0.5). **Practical framing: mod-guidance is a finishing knob conditional on
a good base, not a quality rescue.**

## 2. The quality-selectivity is upstream, not in the proj (base-vs-distill origin)

`--experiment origin`, n=12 dense real captions. The pooled→AdaLN path is **100%
distillation-induced** (base ships a zero-init `pooled_text_proj` with
`enable_pooled_text_modulation=False`; text reaches the base model *only* via
cross-attention). `origin` decomposes a tag's pooled response into upstream `rel_dpool`
(base encoder, no proj) vs `proj_gain` (distilled proj):

| | quality tags | content tags |
|---|---|---|
| `rel_dpool` (base encoder, no proj) | 0.059 | 0.031 |
| `proj_gain` (distilled proj) | 2.94 | 3.04 |

The base encoder already moves the pooled vector ~1.9× more for quality than content;
the distilled proj is a **tag-agnostic ~3× amplifier** (it learned no quality
preference — the distill data had no quality tags). **Consequence:** you cannot make
mod-guidance *more* quality-selective by conditioning the distill teacher on quality
tags — the ceiling is set by the base encoder. Productive levers exploit the existing
upstream signal (adaptive steering `w` on base quality), not proj retraining.

## 3. Why it can't carry a content direction (text-derivative is orthogonal)

> **RESOLVED 2026-06-05.** A geometry-aware (JVP/finite-difference GAD) distillation
> term + a σ-FiLM head were wired and run to try to lift the head's text-derivative
> onto the teacher's — they did **not** move `cos` off zero. A DC/AC decomposition
> explains why: the teacher's text response is ~99% AC, AdaLN can only write DC, so the
> reachable `cos` ceiling for *any* pooled-AdaLN head is 0.05–0.17 (**architectural**,
> not a fit gap) and the head already sits at it. GAD-for-mod-guidance ships at
> `gad_weight=0` (dead); σ-FiLM inert even when opted in. Full verdict:
> `_archive/gad/gad.md` → "GAD for mod-guidance" (archived 2026-06-12);
> see [[project_mod_guidance_sigma_film]].

`_archive/bench/mod_guidance/text_jacobian.py` (generation-free) perturbs the text from sample A
toward B on held-out `(latent, σ, noise)` and compares pathway output deltas: teacher
`ΔT = v(crossattn_B) − v(crossattn_A)` vs student `ΔS = v(pooled_B) − v(pooled_A)`. On
`pooled_text_proj-0602`, matched to its synth training distribution, n=96 pairs/σ
(`results/20260604-1848-0602-synth/`):

| σ | cos(ΔS,ΔT) | ratio ‖ΔS‖/‖ΔT‖ | rel_err | ΔSNR=‖ΔT‖/err_a |
|---|---|---|---|---|
| 0.10 |  0.002 | 0.826 | 0.025 | 0.80 |
| 0.40 |  0.003 | 0.262 | 0.024 | 1.32 |
| 0.70 | −0.002 | 0.103 | 0.047 | 1.43 |
| 0.90 | −0.005 | 0.053 | 0.108 | 1.41 |

`cos` SE ≈ 0.002, so every `cos` is within ~1 SE of zero — **orthogonal, not merely
degraded**, while the head reaches 2.5–11% pointwise error (the low val MSE it was
trained to). Two separable readings: **direction** (cos≈0) is confounded by the
architectural DC ceiling above; **magnitude** (ratio 0.83→0.05) is *not* — the head
transmits 83% of the text magnitude at σ=0.1 but only 5% at σ=0.9, ignoring text
exactly where the teacher leans on it most. Mechanism: pointwise MSE nails the
latent-dominated bulk; the ~2% text contribution sits below the error floor and is free
to point anywhere. **Always probe with the head's training distribution** — a first run
on real latents inflated `err_a` (0602 was synth-trained).

## 4. Schedule axis (σ + layer): both falsified — shipped `8–26` full-dose validated

A separate proposal asked whether the steering *schedule* (which blocks carry it,
whether to gate by σ) could beat the shipped hand-set `8–26`. Two phases on
`channel_attribution.py` killed **both** axes; recorded so it isn't re-proposed.

- **σ axis — DEAD (dose-controlled)** (`--experiment sigma_window`, n=12, w=3,
  `results/20260531-1155-phase0b/`): `uniform ≈ high045` exactly (SSIM-to-`off` 0.885
  both) — the whole effect is the σ≥0.45 structure-forming steps. Dose-matching the
  σ<0.45 tail 5× bought only ~2.3× effect (per-step **saturation** past w≈3). You
  cannot buy the grade in the 4 tail steps.
- **Layer axis — FALSIFIED, it's dose not placement** (`results/20260531-1259-phase0b/`
  + band sweep): between-block SSIM std 0.0079 sits *below* the n=12 noise floor; no
  single block is a drift block; `full08-26` moves ~1.5× more than the hardest single
  block → structure movement is **emergent from stacking 19 blocks, not localizable**.
  Band ordering is at chance. **Qualitative grid read (the verdict):** `full` wins in
  every case; partial arms look like an *interpolation* between `off` and `full` —
  weaker versions of the *same* correction, not different corrections. Interpolation ⇒
  pure scalar dose, not placement.
- **Methodological correction:** on this channel `full`'s low SSIM / high delta is
  *more correction*, not more damage — SSIM/`delta_norm` measure amount of correction
  here, only the grid read disambiguates. So the Phase-0 "cap/taper `w` to stop drift"
  thread is also unmotivated (tapering just reduces the correction).

**Consequence:** the hand-set `8–26` at full dose is validated — no layer lever, no σ
lever, no taper. A learnable `w` scheduler degenerates to a scalar already at its
saturated-optimal value; don't build the per-block headroom allocator.

## 5. The demoted geometry record (frozen mechanism map)

The original finding was a `pooled_text_proj`-**geometry** map (cosines between
projected pooled marginals) that labelled a "quality axis" and predicted a
double-drive/DC-blowout image degradation. **All of it lived in geometry — not one
image was sampled** — and it is superseded by §1–§3. What was wrong: (a) it is a
content-magnitude axis, not quality — an arbitrary artist tag drives it 3–4× harder
than `score_9` (`@sincos +0.31` vs `score_9 +0.07`); (b) the two-pole "score ladder is a
rotation" geometry and the recency-tags-oppose-`score_9` finding are **sparse-base
artifacts** — they collapse ~5–25× / lose stable sign on dense real prompts (max-pool
saturation); (c) the tail-append placement pitted correctly-placed artist tags against
an off-distribution `score_9`. **The one class-level takeaway that survives every base:
named-entity (artist/character) tags drive the pooled axis harder than any `score_X`
word**, and `absurdres` is the one quality tag that stays a positive driver everywhere.
Frozen tables + the base-sensitivity numbers are in git history (this file pre-2026-07);
the geometry scripts were never committed. Mechanism refs: `library/anima/models.py`
(base pooled inject, zero-init), `library/inference/corrections/mod_guidance.py`
(steering delta).

## Reproduce

Paths are post-archive — the scripts live under `_archive/bench/mod_guidance/` now.
The invocations below are written against that location; they import `bench/_common.py`,
so copy them back under `bench/` if the `sys.path` bootstrap doesn't resolve from the archive.

```bash
# Image-space channel attribution (swap / order / intensity / origin)
uv run python _archive/bench/mod_guidance/channel_attribution.py \
    --pooled_text_proj output/ckpt/pooled_text_proj-0530.safetensors \
    --experiment all --dataset_samples 6 \
    --tags "score_9,masterpiece,holding a sword" --seeds 0,1 --compile --label full

# σ / layer schedule axes
uv run python _archive/bench/mod_guidance/channel_attribution.py --pooled_text_proj output/ckpt/pooled_text_proj-0530.safetensors \
    --experiment sigma_window --dataset_samples 6 --seeds 0,1 --sigwin_dose both --compile --label phase0
uv run python _archive/bench/mod_guidance/channel_attribution.py --pooled_text_proj output/ckpt/pooled_text_proj-0530.safetensors \
    --experiment layer_window --dataset_samples 6 --seeds 0,1 --layerwin_mode single --compile --label phase0b

# Text-Jacobian (probe with the head's TRAINING distribution)
uv run python -m bench.mod_guidance.text_jacobian \
    --pooled_text_proj output/ckpt/pooled_text_proj-0602.safetensors \
    --synth_data_dir post_image_dataset/distill_mod_synth \
    --n_pairs 96 --sigmas 0.1 0.4 0.7 0.9 --h 1.0 --label 0602-synth
```
