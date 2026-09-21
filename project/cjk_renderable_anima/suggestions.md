# Suggestions for the Japanese rendering recipe

2026-09-20. Proposal based on the current source, [findings](findings.md),
[sentence record](sent_run.md), and dated reports. These suggestions supplement
`plan_step1.md` (archived under `_archive/cjk_renderable_anima/`); they are not measured improvements or launch
instructions. No new training or benchmark was run for this review.

Keep the mixed singles/real-words direction. Before scaling it, test whether
the apparent exposure cost partly comes from the anchor, and whether explicit
token-to-region supervision makes each multi-glyph example more useful. Both
proposals preserve the target artifact: a static ext-row table with frozen
DiT, llm_adapter, and text encoder.

## 1. Cheapest diagnostic: separate exposure from anchor strength

**Verified in code.** In the warm rows arm,
[`Trainables.regularized`](src/train/trainables.py) applies the anchor to
every warm row on every step:

\[
L_{\mathrm{anchor}}=\frac{\mu}{N}\sum_i\|f_i-f_i^0\|^2.
\]

Here N is the number of warm rows and f is `delta.raw`, in the run's row-norm
units. The data gradient reaches a row only when that row occurs in the
caption, through [`ExtDelta`](src/common/hooks.py). With p_i the probability
that an example contains row i, its expected objective gradient is

\[
p_i\,\mathbb E[\nabla_i L_{\mathrm{FM}}\mid i\text{ present}]
+\frac{2\mu}{N}(f_i-f_i^0).
\]

Thus a rare row faces a stronger anchor relative to its data signal. This
is an objective-level observation, not an exact model of Adam's trajectory;
repetition, glyph size, and context also change the conditional data gradient.

**Implication for the boost gate.** Oversampling changes both observations
per row and the balance between data and regularization. A boost win would
show that oversampling helps; it would not establish an irreducible cost of
1,000 multi-glyph draws. The threshold in `sent_run.md` item 8 remains specific
to its recipe. The early norm contraction and anchor-driven return toward the
seed in the [September 20 report](reports/step2_0920_box_weight_smoke_2026_09_20.md)
make this worth separating before budgeting another full table.

### Experiment

Reuse the existing eight-row boost comparison on ろ 事 カ そ ら め 知 も.
Let b_i be the measured boosted/original draw ratio for row i, including the
longer epoch's normalization.

| arm | sampling | anchor coefficient on the eight rows |
|---|---|---|
| A: existing Round 2 | original | μ = 0.3 |
| B: existing boost | boosted | μ = 0.3 |
| C: proposed control | original | μ / b_i |

C keeps the other rows' anchor coefficients unchanged. It uses the same raw
pack, seed table, data, 6k steps, batch, loss, learning-rate schedule, and
evaluation as A. Per-row coefficients are new code; there is no existing CLI
flag for this arm.

Compute incidence from the actual cached caption token IDs, as
[`Batcher._row_boost_reps`](src/train/stage.py) does. Keep separate counts for
all-caption incidence and multi-glyph text exposure: the former determines
whether a row receives any data gradient; the latter is the sentence ruler.

C approximately matches B's data-to-anchor ratio for the selected rows.
It does not reproduce B's context distribution, companion-row updates, or
Adam state, so this is a diagnostic rather than an equivalence claim.

- **C recovers B's held-string gain at lower exposure:** test a broader
  frequency-dependent anchor before paying the proposed exposure budget.
- **B improves and C does not:** extra observations, their contexts, or
  optimizer dynamics remain necessary; the exposure plan gains support.
- **Only trained strings improve:** investigate pool diversity before scaling.
- **Neither improves:** neither mechanism justifies a larger run.

Read gains against the same seed evaluation with `row_dose.py --rows`, and
read native singles for those selected rows as well as the standard
あ か す 日 set. More drift alone is not a pass. Repeat a promising result
with another training seed before adoption.

**Counter-evidence:** the global μ 0.3 → 0.1 run did not improve the
low-exposure bins. This limits confidence in C. Its narrower question is
whether targeted anchor changes reproduce the targeted boost without also
loosening frequent rows. A general frequency-dependent recipe would still
need to establish its overall scale and protect rare rows from drift.

## 2. Strongest new objective: supervise token occurrences in image space

**Verified in code.** The [compositor](src/common/render/scene.py) computes
glyph positions, including vertical placement and tilt, but returns a
bounding box for the complete text block. The training
[`weighted_fm_loss`](src/train/stage.py) supervises that box as a whole.
There is no explicit token-occurrence-to-region objective.

FM can learn those associations indirectly. The hypothesis is that direct
supervision would reduce the work needed to separate neighboring occurrences.
It fits the repetition/fusion failures and the finding that increasing the
whole-box weight moves the rows without reliably improving composition.
Neither observation proves an attention-routing failure.

### Diagnostic before training

Extend the attention ruler proposed in [freetext.md](freetext.md). Its
existing evidence is for text-bearing **entity** tokens; localization of
individual ext-token occurrences has not been established.

Compare readable English controls and successful/failed short Japanese
renders across σ 0.5–0.9. Inspect whether occurrence maps separate, disappear,
or overlap when glyphs repeat or fuse. Include native scene prompts, and
distinguish maps on noised training targets from maps during free sampling.
Choose layers/heads from this diagnostic rather than assuming FreeText's
entity-localizing layers also carry glyph identity.

Proceed only if occurrence maps have useful spatial correspondence and the
failure pattern supports the hypothesis. Correctly localized but misspelled
text would make this a lower-priority lever.

### Proposed training addition

Preserve per-occurrence regions from the compositor, with the same font
layout and tilt as the image. Align the rendered text span to its actual
Qwen pieces in the complete caption:

- Separate `は` and `い` tokens get separate regions.
- One token such as `こんにちは` gets the union of its five glyph regions.
- Repeated occurrences get separate targets, even when their row ID is shared.

The frozen adapter already carries position into contextual outputs; the
trainable rows would learn to use that path under an auxiliary loss:

\[
L=L_{\mathrm{FM}}+\lambda_{\mathrm{loc}}L_{\mathrm{localization}}
+L_{\mathrm{anchor}}.
\]

A candidate localization term compares each occurrence's spatially normalized
cross-attention map with its normalized target region. Use soft regions at
the actual attention-grid resolution. Check absolute attention mass too:
a correctly shaped map with negligible mass is not successful routing.
Retain max padding and its attention sinks. Capture differentiable maps at
selected layers during training; inference uses the ordinary pack without
regions or attention overrides.

The comparison is **the same mixed-data recipe with and without this loss**.
Hold examples, glyph-size distribution, exposure, σ band, FM weighting, and
anchor fixed. Use the real-word restriction and hold out strings. Report
wall time as well as steps: differentiable attention capture may add memory
and compute costs.

**Pass:** held-out words show better ordered content and fewer repeats or
extras, with a corresponding improvement on multi-glyph native prompts and
no material loss of native singles or scene fidelity. Better attention maps
alone do not count. Start at micro scale; repeat with another training seed
before scaling.

**Limits:** a region for a multi-glyph token cannot supervise its internal
spelling. Contextual outputs may distribute glyph information across several
tokens, making forced localization harmful. A frozen backbone may not expose
enough control through the rows. The diagnostic and ablation decide these.

Related work supports testing the mechanism: [DesignDiffusion](https://arxiv.org/abs/2503.01645)
uses character localization supervision, and
[UDiffText](https://arxiv.org/abs/2312.04884) supervises local attention with
character segmentation. Those systems also train diffusion-model components;
their results do not establish that this works with Anima's rows alone.

## 3. Strengthen the gate before M1

Add **held-out multi-glyph native rendering** to M0's adoption gate. Its
current held-word eval plus single-glyph native checks can miss the exact
failure in `sent_run.md` item 3: an eval-template gain that never reaches a
scene prompt.

Keep the prompt sets fixed across arms. Report ordinary scene prompts and
explicit bubble/sign prompts separately; adding a layout tag is a separate
condition. Use sub-exact lift for continuity, plus order, repetition/extra
glyphs, and visual reads. Exact match can remain a milestone despite its
current floor. Reader misses require sheets, and item-bootstrap intervals do
not capture variation between training seeds.

Recommended order: read the boost result; run the targeted anchor control;
inspect occurrence attention; if supported, compare M0's mixed arm against
the identical arm with localization supervision. Scale only after the
multi-glyph native gate moves. The existing exposure budget remains a useful
planning estimate, with its causal interpretation and transfer still open.
