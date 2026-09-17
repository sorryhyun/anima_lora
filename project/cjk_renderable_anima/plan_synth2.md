# plan_synth2 — the ΔFM line: paired-difference supervision for new rows

> **Proposal, 2026-09-17. Not launched.** Runs only after the sentence
> anchor arm finishes (`sent-a0p3-s05`: jobs `20260917-131126-da1e0a` train
> eval, `…-ff5332` native, `…-47801d` target — user's rule, one line on the
> GPU at a time). Nothing here changes `plan_synth.md`'s tracks A / B; if Δ0
> below passes, B1 (the 100 k base seed) is the run that collects the win.
> Origin: the user's three-candidate note of 2026-09-17 (paired-difference
> FM / cached local Jacobian / OCR-reward rows); this plan takes the first,
> the other two are in *Not this plan* with the reason.

The S line pays for identity in exposure: ≈ 1 000 draws per row
(`plan_synth.md`, exposure curve 1 330 / 670 / 490 → 100 / 75 / 36 %). Every
shortcut to the *address* is closed (encoder, composition, transplant, warm
start, Q — `findings.md` *Do not re-propose*). This line does not touch the
address. It changes the **supervision per draw** so that fewer draws carry
the same information about the glyph: the row is trained on the difference
between two renders of the same scene under the same noise, one with the new
glyph and one with a reference the base already draws.

## What it is

Notation as in the loop (`src/train/stage.py`, `fm_training_batch`):
`z_σ = (1−σ)·x + σ·ε`, target `v* = ε − x`, rows-only trainables.

Per composite item B (scene + new glyph, caption `c_B` with the ext row)
the data stage also holds its **sibling A**: the same scene, same erase,
same font / size / position / colour / tilt, a reference string in the
box, caption `c_A` = the same frame with only the quoted string swapped.
One noise `ε`, one `σ` for the pair. Loss

    L_Δ = ‖ (v_θ(z_σ^B, c_B) − sg v_θ(z_σ^A, c_A)) − (v_B* − v_A*) ‖²_w
        = ‖ r_B − r_A ‖²_w,                r_X = v_θ(z_σ^X, c_X) − v_X*

with `w` the existing box weight. Gradient w.r.t. the new row:
`(r_B − r_A) · ∂v_θ(B)/∂f_new` — **plain FM with the sibling's residual
subtracted as a control variate.** What that buys, region by region:

- **In the target** the noise cancels exactly: `v_B* − v_A* = x_A − x_B`,
  the clean pixel delta, zero outside the box for any reference.
- **Outside the box** the term is `‖v_θ(B) − v_θ(A)‖²`: match the
  known-text render of the same scene. That is a scene-preservation term
  against the wipe (`findings.md`: wipes are the delta norm) where today
  the loss is weight-1 plain MSE against a noisy target.
- **Inside the box** the shared part of the residual cancels: the
  posterior spread of the scene given `z_σ` (the x₀ side dominates at
  σ ≈ 0.8 where identity is decided and it depends on `z_σ`, not on the
  glyph), and — because A is a paste by the same renderer — the erase
  patch / ring-median fill / font rasterisation that both items carry
  (*Erase artefacts as a cue*, `plan_synth.md` open risks, is shared and
  drops out). What is left is the glyph delta plus the part of the model's
  B error the reference does not share.

The address is untouched: `f_new` is random-init and its Jacobian is taken
through `c_B` (the ぬ caption) exactly as today; `c_A` never contains the
row, branch A is `no_grad`, and nothing about the reference's embedding
reaches the parameters. Deployment format (`ExtDelta` table → vocab pack),
eval / native / target stages: unchanged.

What can leak is the **mean of `r_A`** — the target becomes `v_B* + r_A`,
so a systematic error of the base on the reference inside the box is
inherited (the か-with-Latin-strokes mechanism, `synth_micro_loop`). For a
calibrated model `E[r_A | z_σ] = 0`; the bias is bounded by how badly the
base renders the reference, which is why the reference is EN.

## The reference

**EN, pasted by our renderer, under the JA frame.** For each composite the
data stage draws a second image with the same `render_into_scene` fit
(font, `fs`, lines, colour, tilt angle, the erase) and a Latin string of
the same glyph count in place of the text; caption `scene_caption(sc,
ref_text)` — the JA frame the scene was drawn under (`japanese text` tag,
`Japanese text reads as "KA"`), only the quote differs. Rules:

- **Same frame in both branches.** A JA-frame B against an EN-frame A puts
  a frame-mismatch component into `r_A` that B does not share (rows are
  JA-frame-bound, `findings.md` rulers) — it adds noise instead of
  cancelling it. Frame is held; only the quoted string moves.
- **Never the romaji.** Reference letters are drawn from A–Z minus the
  target's romaji letters (す never gets S / U). The row cannot see the
  pairing either way; the rule keeps the design from reading as a mapping.
- **Matched extent.** One Latin letter per JA glyph, same `fs`, same
  column / line positions (`_draw_vertical_glyph` already draws per glyph,
  so a vertical Latin stack is free). The weighted box is the **union** of
  the two drawn boxes.
- **A small pool per scene.** `--pair_ref_pool 4`: four reference strings
  fixed per (scene, glyph count), so the reference captions stay ≈ 1 k
  unique (text cache is ≈ 1.3 MB per caption in RAM; 10 k distinct
  reference captions would be 13 GB on the 46 GB box).

**No warm start is required by the loss.** The one condition is that the
base can already draw the reference (so `r_A` is a scene residual, not
the floor's garbage); pasted Latin meets it with no table, so the EN arm
trains new rows **from random init**. Only a *kana* reference needs a
table — for the reference row, not the new one (Δ1); Δ2 / Δ3 warm-start
because the sentence recipe does, not because ΔFM does.

Why not the alternatives:

| reference | in-box cancellation | inherited bias | verdict |
|---|---|---|---|
| a known kana (す) | most (similar posterior) | most (base ≈ 34/36 on them, style quirks per glyph); **needs a trained table** — from scratch the base has no address for す and `r_A` is the floor's garbage | control arm only, on a warm start (Δ1) |
| EN letters, pasted (this plan) | middle | least among glyph refs (EN 24/24 flat); paste artefacts shared | **the arm** |
| the scene's own un-erased anchor ("hi", the base's render) | — | paste / erase artefacts are in B only → they no longer cancel; frame differs | no |
| blank box | least glyph-dependent | none in-box | caption differs by the whole clause → frame mismatch | no |

**Diagnostic for the leak, free in the loop:** `ref_bias` = in-box
‖mean_batch r_A‖ / mean_batch ‖r_A‖ (EMA over 100 steps). ≈ 0 means the
reference residual is spread, not systematic; a value that stays above
≈ 0.3 says the base draws the pasted Latin under the JA frame with a fixed
error, and the sheets are read for Latin-styled strokes on the trained
glyphs. If both show, the fallback is `--pair_sigma_min 0.7`: the paired
term on the identity band only, plain FM below it (stroke style is
committed at lower σ than identity).

## Code owed (small, all local to `src/`)

- **`common/render/scene.py`** — split `render_into_scene` into the fit
  (erase + `fit_text` + colour + tilt draw) and the draw, so a sibling text
  is drawn with the same fit. Returns both images and both boxes.
- **`data/synth.py` + `cli/data.py`** — `--pair_ref none|en` (default
  `none`; `kana` in Δ1). With `en`, every composite record gains
  `ref_file` (`img/scene_{i:05d}_ref.png`), `ref_text`, `ref_caption`,
  `ref_box`; `box` becomes the union. Reference strings from a per-scene
  pool of `--pair_ref_pool` Latin strings of the item's glyph count,
  romaji-excluded. Kinds: `single` first; `short` / `sentence` get vertical
  Latin stacks with the same cuts (Δ2).
- **`train/stage.py` + `cli/train.py`** — `--pair_loss 0|1`. `LatentStore`
  encodes the reference files into a parallel tensor (same shape family;
  ≈ 256 KB per item in fp32, bf16 on disk). `_encode_text` adds the
  reference captions. Per step: B's `fm_training_batch` as now; A's
  `z_σ^A = (1−σ)·x_A + σ·ε`, `v_A* = ε − x_A` with **the same `ε` and the
  same `ts`** (σ is drawn inside `fm_training_batch`; the A branch reuses
  its output, never a second draw); `pred_A = dit_forward(…, ref_captions)`
  under `torch.no_grad()`; `loss_fm = weighted_fm_loss(pred_B − pred_A,
  target_B − target_A, recs, box_weight)`. `tr.regularized` unchanged.
  Batches stay one (shape, source) — the sibling has the scene's shape by
  construction.
- **Log record** gains `fm_pair` (the trained loss), `fm_plain` (‖r_B‖²_w,
  free — `pred_B` / `target_B` exist; the number comparable to the control
  arm's `fm`), `pres` (outside-box ‖pred_B − pred_A‖²), `ref_bias`.
- **Gotchas to expect.** The `no_grad` forward is a second compiled graph
  per token family (grad mode is a guard) — one more compile at step 1,
  512² family only under `--scene_min_tokens 900`. Do **not** fold A into
  one 2×batch forward: batch 8 OOMs at 512² and A's activations are the
  waste. Wall ≈ +1 forward per step: 2.35 it/s → ≈ 1.7 it/s expected,
  measured on Δ0; that ratio is the price the gate has to beat.

Plain-FM controls run on the **same data dir** (`--pair_loss 0` ignores the
sibling files), so the two arms share items, scenes, fonts, seed and
batcher order — the comparison isolates the loss.

## Arms (one at a time, after the sentence arm)

**Δ0 — smoke, singles from scratch (≈ 1.8 h GPU total).** The micro-arm
regime (`synth_micro_loop`): 12 rows the current tables miss, trained from
random init, no warm start, so exposure per row is the only variable.
Inventory `chars:がぎぐげござガギグゲゴザ` (6 dakuten hiragana + 6 dakuten
katakana — the kana_ext family the 53k table is weakest on, `single_ext`
20/36 in the anchor sweep; eval group `single_extra`). Data: `--scenes sl1w --scene_drop sl1w:332,957
--scene_min_tokens 900 --shapes 512 --scene_frac 1.0 --scene_mix single=1.0
--flat_bubble 1.0 --scene_fill 0.7 --scene_min_glyph 28 --n_items 2000
--pair_ref en --pair_ref_pool 4`. Train: the S recipe (`--lr_rows 1e-3
--lr_decay cosine --free_residual 1e-3 --box_weight 4 --batch 4 --t_min 0.7
--t_max 0.9 --compile 1 --grad_ckpt 0`, singles band), `--seeds 2`,
`--c_flat 0`, no `--init_rows`. Four jobs:

| arm | `--pair_loss` | steps | draws / row | wall (est.) |
|---|---|---|---|---|
| `pair0_s1500` | 0 | 1 500 | ≈ 500 | ≈ 11 min |
| `pair0_s3000` | 0 | 3 000 | ≈ 1 000 | ≈ 21 min |
| `pairEN_s1500` | 1 | 1 500 | ≈ 500 | ≈ 15 min |
| `pairEN_s3000` | 1 | 3 000 | ≈ 1 000 | ≈ 30 min |

plus `native` (`--native_chars が,ガ,ご,ゴ --native_clauses en,swap`) on the
two 3 000-step arms. Two step counts because the deliverable is an
**exposure curve**, not one number: the curve's 490-draw point is 36 % for
plain FM and the question is where ΔFM sits at 500.

Read (`single_extra` = 12 chars × 2 seeds = 24 renders; per-glyph, on the
sheets, both readers — dakuten is where reader folds bite):

- **Pass:** `pairEN_s1500 ≥ pair0_s3000` on `single_extra` (half the draws
  for the same hits — what pays for the ≈ 1.4× wall), or `pairEN_s3000 ≥
  pair0_s3000 + 4/24`. Either → Δ2.
- **Preservation:** native `en cos` / `box IoU` and the composite sheet on
  the 3 000 pair; `pres` in the log. ΔFM is expected to wipe *less*
  (outside-box term); a ΔFM arm that wipes more than plain is a bug in
  the sibling (erase / box mismatch), read the pair images first.
- **Leak:** `ref_bias` ≤ 0.3 through the run; sheets show no Latin-styled
  strokes on the hits. Above that → the `--pair_sigma_min 0.7` variant
  before any other read.
- **Kill:** `pairEN ≤ pair0` at both step counts with `ref_bias` low and
  the pair images correct → the residuals do not share enough; close the
  line in one report (`reports/synth_pair_2026_09_XX.md`), one row in
  `findings.md` *What does not move it*.

Also read, not gated: `fm_plain` of the ΔFM arm vs `fm` of the control at
equal steps (does the variance reduction show as a lower plain residual
on the same items), and the row norm at the end (every arm drives it to
≈ 125–130; a ΔFM arm that lands lower with equal hits is the placement /
identity trade-off moving).

**Δ1 — kana-reference control (only if Δ0 passes; ≈ 45 min).** The same
12 rows warm-started onto the 53k + punct table (`--init_rows …
merge_punct/trained.pt --lr_warmup 500 --init_anchor 0.3`, the anchor
recipe), `--pair_ref kana` with references drawn from the 92 the table
renders (あ か す …, rotated per scene), against `--pair_ref en` on the
same warm start at 3 000 steps. Answers whether the extra in-box overlap of
a glyph-shaped reference is worth its bias; the EN arm stays the default
unless kana wins by ≥ 4/24 with the same `ref_bias`.

**Δ2 — the sentence recipe with pairs (≈ 3.5 h).** The `sent-a0p3-s05`
argv (`README.md` *How to run*) with `--pair_ref en --pair_loss 1`: every
kind paired, `short` / `sentence` siblings as vertical Latin stacks with
the item's cuts, 12 000 steps (half the baseline's 24 000), same warm start
and anchor. Read against the finished `sent-a0p3-s05` at 24 000 on the
same eval prompts: flat singles, `short` / `phrase` and their `_held`,
native both clauses, target stage. Gate: not below the baseline on
singles + native at half the steps. Guard: the sibling stacks must keep
one column when the item does (read the `_ref` images on the composite
sheet before launch — a Latin stack that wraps differently makes the
outside-box term fight the layout).

**Δ3 — B1 with pairs.** If Δ2 holds, Track B's 100 k base seed runs with
`--pair_loss 1` and the step count the Δ0 curve says buys ≈ 1 300 draws'
worth of identity per row. This is where the line pays: 11 h → whatever
the curve gives, on 307 rows, and every sentence run after warm-starts on
a base that holds its singles.

## Open risks

- **The residuals do not share enough.** The mechanism needs `r_A` and
  `r_B` to overlap beyond the box; if the base's scene posterior is not the
  dominant residual at σ 0.7–0.9 the cancellation is small and the arm
  measures the extra forward. Δ0's `fm_plain` vs `fm` at equal steps is
  the direct read.
- **The sibling is not a sibling.** Any difference outside the box (a
  second erase pass, a different tilt draw, colour rule reading a
  different anchor ink) puts scene content into `x_A − x_B` and the
  outside-box term trains the row on it. The fit must run once; the
  `_ref` images are diffed against their items outside the union box at
  data time (assert max |Δ| = 0 outside the box, in the data stage).
- **Reader on dakuten.** `single_extra` is read with the SFX reader's
  glyph fold; a か/が fold would hide the win. Per-glyph sheets and the
  `vl16` column are the cross-check; if the fold bites, the eval group
  gets an unfolded exact column (the punctuation-CER precedent).
- **RAM.** Reference latents (+ 2.6 GB at 10 k items fp32, half in bf16)
  and captions (pool-bounded) on top of the text cache; the 10 k-item
  build stays the limit for Δ2 / Δ3.
- **Compile.** Two graphs per family; if the dynamic-seq compile trips on
  the `no_grad` forward, `--compile 0` for Δ0 only (2× wall on a 30-min
  arm is acceptable, on Δ3 it is not — fix the compile before Δ2).
- **The bubble as a unit** (`plan_synth.md`): unchanged by this line — the
  reference sits in the same bubble, so nothing here separates glyph from
  canvas; that stays a placement / scene-stage question.

## Not this plan

- **Cached local Jacobian, inner updates on the row** (the note's item 2).
  The objective is an expectation over noise; a linearisation at one
  (`z_σ`, σ) is not it, and inner steps on a cached `J` overfit the draw.
  One backward already gives the exact `Jᵀr` for ≈ 2 forwards, while k
  JVPs cost k forwards for a rank-k approximation. The grad_basis probe
  (2026-09-12) is the measured version: 3× first-step capture, flat in
  the blind A/B.
- **OCR-reward row search** (item 3). Reward through a sampler plus a
  reader with a domain gap (the か/日 reversal that totals hid) is
  correction, not learning; reward hacking is the failure mode. Kept as a
  last-mile idea for rows with one consistent wrong read, after this line
  has a result, and only with per-glyph sheet reads.
- **Contrastive ΔFM** (push `v_θ(B)` away from `v_θ(A)`). Different sign,
  different aim (mode separation); this line matches the delta, it does
  not repel. Contrastive terms on text-free natives are already closed.
- **The romaji as the reference**, a shape-neighbour kana as the reference
  (the encoder / IDS verdict — addresses are not shape coordinates, and
  the reference is not an address here anyway), the un-erased scene anchor
  or a blank box as the reference (table above).
- **Warm-starting the new row at the reference's row.** That is the
  transplant / warm-start line (closed 2026-09-16: inherited trigger buys
  no steps). Random init + this loss; `--init_anchor` only for rows a
  source table already has.
- **A flat-first curriculum** (flat canvas → composite ΔFM → sentences;
  asked 2026-09-17). Twenty steps is 80 draws against ≈ 1 000 per row —
  flat is fast in per-step loss, not in identity; a flat-trained `f` does
  not carry to composites (transplant 0/64 vs composite-trained 23/64;
  rows are canvas-conditional, which is why the S line exists), so stage 2
  either erases it (no anchor) or pins it to the wrong direction (anchor);
  and ΔFM's cancellation is the *scene* residual, which flat canvases
  barely have — it makes composite training cheap directly, so composite
  from step 1 with flat at the 0.1 guard share stays the shape.
- **Running Δ0 beside the sentence arm.** One GPU line at a time; Δ0 is
  queued behind `20260917-131126-47801d`.
