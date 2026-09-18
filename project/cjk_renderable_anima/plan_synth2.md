# plan_synth2 — the ΔFM line: paired-difference supervision for new rows

> **Status 2026-09-18: the line is read out.** This file is now the ΔFM
> loss **as built** (mechanism, the reference, the flags) plus what each arm
> settled; the dated records are
> [`reports/synth_pair_2026_09_17.md`](reports/synth_pair_2026_09_17.md)
> (Δ0 / Δ0b), `reports/synth_pair_delta1_2026_09_18.md` (Δ1) and
> `reports/synth_s2a_2026_09_18.md` (the sentence A/B).
> **Verdict: ΔFM holds the scene and does not deliver the glyph.** It loses
> to plain FM on sentences (S2a) and stays a flag; whether it is the seed
> table's loss for **singles** is `plan_synth4.md` R4.4.
> Origin: the user's three-candidate note of 2026-09-17 (paired-difference
> FM / cached local Jacobian / OCR-reward rows); this plan takes the first,
> the other two are in *Not this plan* with the reason.

The S line pays for identity in exposure: ≈ 1 000 draws per row
(`plan_synth.md`, exposure curve 1 330 / 670 / 490 → 100 / 75 / 36 %). Every
shortcut to the *address* is closed (encoder, composition, transplant, warm
start, Q — `findings.md` *Do not re-propose*). This line does not touch the
address. It changes the **supervision per draw**: the row is trained on the
difference between two renders of the same scene under the same noise, one
with the new glyph and one with a reference the base already draws. It was
proposed as an exposure saver and is not one (Δ0); what it buys is a table
that renders inside a scene instead of wiping it.

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

**Measured (Δ0 / Δ0b, report):** ≈ 80 % of the residual cancels (`fm_pair`
0.013–0.02 vs `fm_plain` ≈ 0.10) and the outside-box term holds the scene.
The cancelled part is the scene residual — the sign-consistent part Adam
travels on (hence 2× the lr) and, under plain FM, what builds the row's
"bare canvas, one glyph" component: the wipe, which is also the only thing
that was removing the base's free-running JA pseudo-text.

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
table — for the reference row, not the new one (parked, *Not this plan*).

Why not the alternatives:

| reference | in-box cancellation | inherited bias | verdict |
|---|---|---|---|
| a known kana (す) | most (similar posterior) | most (base ≈ 34/36 on them, style quirks per glyph); **needs a trained table** — from scratch the base has no address for す and `r_A` is the floor's garbage | control arm only, on a warm start (parked) |
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

**Δ0 read:** `ref_bias` 0.26–0.28, flat, no Latin-styled strokes on the
hits; the systematic part is the co-text, not the reference. The σ-split
fallback is a no-op under `--t_min 0.7` and interpolates at 0.8 (Δ0b).

## As built

Commit `2ffacc78`: `src/data/pair.py`, `common/render/scene.py`
(`ref_text=`, one fit, two draws), `data/synth.py` (`--pair_ref none|en`,
`--pair_ref_pool`, `ref_file` / `ref_text` / `ref_caption` / `ref_box`,
union `box`, `sheet_scene_pair.png`), `train/stage.py` (`--pair_loss`,
`--pair_sigma_min`, `pair_branch` under `no_grad` with the batch's own `ε`
and `σ`, `pair_stats` → `fm_plain` / `pres` / `ref_bias`). 1.63 it/s against
2.35 plain (0.69×); the second compiled graph cost nothing. Plain-FM
controls run on the same data dir (`--pair_loss 0` ignores the siblings).
`--pair_ref_frame ja|en` (Δ0b arm 2; `data/pair.py::en_frame`) rewrites the
sibling captions under the scene's EN frame at train time — measured flat,
kept as a flag. `--pair_ref kana` is parked and not written; paired `short` / `sentence`
siblings **are** built (the Δ0.9 rework put every kind through one composite
loop, `render_into_scene` taking the item's own line lengths — `ref_lines`),
as is `--pair_flat` for flat items.

## What it bought, and where it died

Every arm below is on record; the launch detail is in the reports.

- **Δ0 / Δ0b** (2026-09-17, 12 dakuten rows, `reports/synth_pair_2026_09_17.md`):
  ΔFM **holds the scene** (en cos 0.93 vs 0.855 at equal row norm; images
  under 0.85 6 vs 20 of 64), **saves no exposure**, needs `--lr_rows 2e-3`
  for plain FM's travel (2e-3 × 1 500 ≡ 1e-3 × 3 000), and leaves the base's
  JA pseudo-text beside the glyph (joint hit ∧ en cos ≥ 0.85: 15–20 vs plain
  28 of 64). `ref_bias` 0.26–0.28 and flat, no Latin-styled strokes on the
  hits — the systematic part is the co-text, not the reference. Wipe ↔
  co-text is one axis, and neither `--pair_sigma_min` nor
  `--pair_ref_frame en` moves along it.
- **L0** (2026-09-17): for plain FM the exposure budget is **draws, not the
  lr integral** — doubling the lr doubled the travel (row norm 162 against
  `pair0_s3000`'s 149) and bought nothing (18 vs 21 of 24 singles). The lr
  lever is ΔFM's own. Rider: at 19/24 the control was not off the ceiling,
  and 12 rows cannot price it either way.
- **Flat share** (2026-09-17 night): a paired flat 0.1 arm is not *clearly*
  above flat 0 — singles 13 → 14, natives up a few on every count with the
  tail and en cos unmoved (McNemar p ≈ 0.17) — and the user's side-by-side
  read showed what the rulers missed: the flat items pull the render toward
  their own look (a large bold glyph, plain canvas or black box, the scene
  thinned; box IoU 0.47 → 0.39). **Δ1 and everything after run flat 0.**
- **Δ0.9 — dataset polish** (built 2026-09-17 night, no training). Specks are
  recorded and erased with the anchor's erase (`scenes/judge.py`,
  `--scene_rejudge` re-applies the rule on CPU); two rules from the user's
  read of the smoke build — `bubble_leak` (`--scene_max_offset 1.0`) and
  `--scene_open_lost 0.02`; per-kind scene routing in the data stage
  (`--single_scenes` + `--single_max_ar` for one-glyph texts,
  `--scene_one_bubble`, multi-glyph texts routed by fit); `fit_text`'s shrink
  step fixed (it had sat after a `return` since `c434a398`). Residual bar
  0.5 → 0.33 after a pair sheet showed anchor letters surviving. Pools after
  the rejudge and the `s1w` top-up: **s1 233, s1w 380, sl1w 213, ja_comic
  292**, one-glyph pool 586.
- **Δ1 — the full-inventory table with pairs** (`20260918-015823-5a6c3f`,
  `rows_synth_d1_d1_s53k`): 356 rows (kana + `kana_ext` + `kanji:200` + 13
  punct), `--scene_mix single=1.0`, 10 000 items, random init, `--pair_ref en
  --pair_loss 1 --lr_rows 2e-3`, `--c_flat 0`, flat 0, 53 000 steps ≈ 596
  draws/row. Read in `reports/synth_pair_delta1_2026_09_18.md`: the gate is
  missed on singles (12 vs 13) and on the native joint, and the whole loss is
  the `en` (JA-frame) clause — the glyph is there and the base's pseudo-text
  around it spoils the exact read. A singles-only build leaves the 18 small
  kana untrained by design.
- **Δ2 — the sentence recipe with pairs** moved to `plan_synth3.md` as S2 and
  **closed there 2026-09-18**: on sentence items plain FM beats ΔFM on every
  ruler that moves (`reports/synth_s2a_2026_09_18.md`). The premise — a fully
  addressed caption leaves no unaddressed text to invent — does not hold at
  this exposure. ΔFM stays a flag; the seed table's loss (a singles question)
  is `plan_synth4.md` R4.4.

## Risks that stayed live

- **The co-text is the base, not the loss** (Δ0b) — and it does **not**
  need unaddressed text: S2a's fully addressed captions still drew
  pseudo-JA around the glyph.
- **The micro regime flatters everything.** 12 rows from scratch reach
  83 % at 500 draws, and a 12-row arm comparison sits on a rerun chaos
  floor of cos ≈ 0.75 per row (`reports/row_blocks_alpha_2026_09_18.md`).
- **lr 2e-3 on a warm start** is untested: the anchor recipe was tuned at
  1e-3 and Δ1 was random init.
- **`--free_residual` is unscaled against the paired loss** (5–8× smaller
  than plain): at 1e-3 the ΔFM norm peaked at 126 and sagged to 115. At
  2e-3 the norm reaches 144, so it is not gating; if a later ΔFM arm under-travels
  again, 3e-4 is the first thing to move.
- **Reader on dakuten / on co-text.** `single` exact held up against the
  sheets on Δ0; native `both` under-counts ΔFM images where the glyph is
  visible beside pseudo-text. The sheets stay the second read.
- **RAM.** Reference latents (+ 2.6 GB at 10 k items fp32, half in bf16)
  and captions (pool-bounded) on top of the text cache; the 10 k-item
  build stays the limit.
- **The bubble as a unit** (`plan_synth.md`): unchanged by this line — the
  reference sits in the same bubble, so nothing here separates glyph from
  canvas.

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
- **A kana reference (was Δ1).** Parked: it was meant to buy more in-box
  cancellation, and cancellation is not the bottleneck (≈ 80 % on singles,
  86 % on sentences, no exposure saving). Needs a trained table for the
  reference row.
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
- **lr 3e-3.** Closed (`findings.md`: off-manifold at 2.4× row norm).
