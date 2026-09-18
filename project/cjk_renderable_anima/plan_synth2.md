# plan_synth2 — the ΔFM line: paired-difference supervision for new rows

> **Status 2026-09-17 night.** Δ0 and Δ0b are read and moved out of this
> plan — record [`reports/synth_pair_2026_09_17.md`](reports/synth_pair_2026_09_17.md),
> verdicts in [`findings.md`](findings.md) (*trigger vs canvas*, last
> bullet; *what does not move it*). What they left: **ΔFM holds the scene**
> (en cos 0.93 vs 0.855 at equal row norm, images under 0.85: 6 vs 20 of
> 64), **saves no exposure**, needs **`--lr_rows 2e-3`** for plain FM's
> travel (2e-3 × 1 500 ≡ 1e-3 × 3 000), and **costs the base's JA
> pseudo-text beside the glyph** on single-glyph natives (joint hit ∧ en
> cos ≥ 0.85: plain 28, ΔFM 15–20 of 64) — a cost no reweighting of the
> loss removes (wipe ↔ co-text is one axis).
> **User decision 2026-09-17:** the line switches to the paired loss. Order:
> **L0** (read: lr is not a plain-FM lever) → **Δ0.9** dataset polish → **Δ1** the full-inventory
> table with pairs (replaces the 53k seed run) → **Δ2** the sentence recipe
> with pairs. One GPU line at a time; the `sent-a0p3-s05` report is still
> owed before Δ2.
> Origin: the user's three-candidate note of 2026-09-17 (paired-difference
> FM / cached local Jacobian / OCR-reward rows); this plan takes the first,
> the other two are in *Not this plan* with the reason.
> **2026-09-18:** Δ2 moved to [`plan_synth3.md`](plan_synth3.md) as **S2**,
> with a ruler it did not have (exact match is 0/32 on every multi-glyph group
> of every arm) and the ΔFM-vs-plain A/B written as the line's kill decision.
> The "paired `short` / `sentence` siblings — code owed" note below is stale:
> the Δ0.9 rework put every kind through one composite loop.

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
table — for the reference row, not the new one (parked, *Not this plan*);
Δ2 warm-starts because the sentence recipe does, not because ΔFM does.

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
kept as a flag. `--pair_ref kana` (parked) and paired `short` / `sentence` kinds
(Δ2) are not written.

## Arms (one at a time)

All micro arms reuse `output/wake_probe/data_synth_pair_d0` (2 000 paired
dakuten singles, 12 rows) and the Δ0 argv (report, *What ran*); only the
named flags move. Eval `single` (24 renders) + `native` (`が,ガ,ご,ゴ`,
`en,swap`, 64 per clause) on every arm that is read for the scene — Δ0's
singles alone pointed the wrong way. Rulers: `single` exact, native
`both`, en cos, and the **joint** count `both ∧ en cos ≥ 0.85` from
`native_reads.json` (plain `pair0_s3000`: 28 en / 16 swap; tail under
0.85: 20 / 16).

**Δ0, Δ0b — done**, moved to `reports/synth_pair_2026_09_17.md` (arms,
argv, tables) and `findings.md`.

**L0 — is the exposure budget the lr integral? Read 2026-09-17 22:10:
no, for plain FM (micro regime).** `--pair_loss 0`, Δ0 data and argv,
750 steps:

| arm | `--lr_rows` | `single` | row norm | job |
|---|---|---|---|---|
| `pair0_s750` | 1e-3 | 19/24 | 93 | `20260917-214631-29a3a4` |
| `pair0_s750_lr2e-3` | 2e-3 | 18/24 | 162 | `…-da5512` |
| (`pair0_s1500` / `pair0_s3000`, 1e-3) | | 20 / 21 | 120 / 149 | Δ0 |

Doubling the lr doubled the travel (162 > `pair0_s3000`'s 149) and bought
nothing — the plan's *fail* branch: for plain FM the budget is draws, the
lr lever is ΔFM's own (its cancelled sign-consistent gradient). Two riders:
the control was **not** off the ceiling (19/24 at 250 draws per row, so the
pass condition was unreachable — 12 rows cannot price B1 either way), and
plain FM reads 19/24 at norm 93 where ΔFM reads 10/24 at norm 92 — "hits
track row norm" is a ΔFM fact; under plain FM the scene residual (the wipe)
buys identity per unit of travel. The s1500 guard arm was not run.

**Δ0.9 — dataset polish (no training; before any Δ1 data build).** The
scene pool was judged for the S line's sentence anchors and Δ0 pasted
single glyphs into it. Read of `scenes_sl1w` (276 kept of 1 000) and of
`data_synth_pair_d0` on 2026-09-17:

- **Stray text survives the judge.** The speck rule keeps any non-anchor
  detector box under ¼ of the anchor's area, un-erased. 39 of 276 kept
  scenes carry one (s1 13/269, s0 8/174); on the sheet ≈ a third are a
  *second bubble or a sign with the base's pseudo-text* (9, 30, 148, 157,
  192, 545, 662, 346, 462), the rest corner signatures / watermarks.
  `data_synth_pair_d0/img/scene_00000` (sl1w 192) is this case: the glyph
  floats on blank canvas above a bubble that still reads "Hav". Those
  items are training targets with pseudo-text beside the glyph — plain FM
  is asked to draw it, ΔFM cancels it (both siblings carry it); neither is
  wanted. Fix in `scenes/judge.py`: erase every speck with the anchor's
  erase (the JA-frame path already erases every box) and reject on its
  erase residual; `--scene_rejudge` re-applies the rule from stored reads,
  no GPU.
- **A lone glyph in a sentence-sized region.** sl1w anchors are sentences
  ("we should go home before dark"), so regions run to 214 × 42 px and a
  single at `fs` ≈ 40 sits alone in a strip, outside any bubble when the
  anchor was free text (sl1w 192, 330, 414, 545; the other 20
  `open_uniform` keeps are broken-outline bubbles and fine). For `single` /
  `short` kinds: closed-bubble scenes only, region aspect ≤ 2 (or the
  s0 / s1 short-anchor pools); sentence-sized regions stay for sentence
  kinds.
- **The negative prompt is not a reason to regenerate.** Scenes were drawn
  at cfg 4 with `worst quality, lowres, old, bad hands, bad anatomy, sepia,
  blurry, glitch, jpeg artifacts` — quality tags only, never in a caption,
  the same kind of negative the native / target stages render under.
  Regeneration is a **top-up** after the rejudge shrinks the pool (yield
  28 % per generated scene), not a redo.
- **Pool for Δ1 = the s1 recipe, grown; s0 dropped** (user, 2026-09-17).
  s1 (269 kept) has short anchors, four frames (bubble 121 / sign 58 /
  reads_as 52 / saying 38) and single-sized regions (aspect ≤ 2 on 254 of
  269; sl1w 198 of 276) — it is also the 53k run's frame set, so the gate
  compares like with like. s0 (174, bubble-only) is a subset of that mix
  and would push the bubble share 45 → 67 %; frames are the measured lever
  (frame-mix 2×2), so it stays out. s1's gap is shape: 448–512² only —
  generate the top-up with the s1 prompts / frames over sl1w's shape list
  (`576x448 … 384x640`, the mixed pool of record) under the fixed judge, to
  ≈ 600 kept (× `--pair_ref_pool 4` ≈ 2.4 k sibling captions ≈ 3 GB).
  sl1w / `ja_comic` stay the sentence pools for Δ2.
- **Multi-glyph one-piece rows (こんにちは = one Qwen piece) need long
  regions; route by fit, not by pool** (user, 2026-09-17). A one-piece word
  cannot be cut at a piece boundary, so it must fit one column / one line:
  5 glyphs × the 28 px floor = 140 px. Regions tall ≥ 1.3 AR / of those
  ≥ 150 px: s1 39 / 5, sl1w 61 / 20, **`ja_comic` 221 / 88**; wide regions:
  s1 86, **sl1w 130**, `ja_comic` 93. So sl1w is the *horizontal* word pool
  and `ja_comic` the vertical one; both join `--scenes` for word units only
  (the draw already picks scenes whose one-column capacity ≥ `len(text)`,
  `data/synth.py`), singles stay on the s1 pool under the aspect ≤ 2 rule.
  Check before the build: the per-kind fit report (the sentence line fell
  back to singles on 108 / 180 phrase draws) — a word row that does not
  meet its quota stays untrained (53k: word rows at norm 0.12); `ja_comic`
  has 1.64 anchor bubbles per image — **one-bubble scenes only** (user):
  of 359 kept, 198 have one anchor box (none of those carries another
  detector box), 182 with a closed bubble → 97 tall (39 at ≥ 140 px, the
  5-glyph column; sheet read 2026-09-17: all 39 show one bubble and no
  other text, four are multi-panel pages — 829, 1128, 1937, 2078 — the
  user's call) and 55 wide. Needs a data-stage filter on
  `len(boxes_anchor) == 1` (none today; `--scene_tall_ar` and
  `--scene_drop` exist).
- **Built 2026-09-17 night** (pools rejudged in place; originals kept as
  `scenes_<tag>/*_pre_speck.*`). Judge (`scenes/judge.py`): specks are
  recorded (`boxes_speck` / `speck_regions` / `speck_bubbles`) and erased by
  `render_into_scene` with the anchor's erase, `speck_erase` when that erase
  fails; the user's read of the smoke build (scene 00022 / 00024) added two
  rules — **`bubble_leak`** (`--scene_max_offset 1.0`: region centre more than
  one text-box half-size off the text = the flood leaked through an outline
  gap into a panel strip / figure; median 0.11, ja_comic 770 1.4, sl1w 962
  5.8) and **`--scene_open_lost 0.02`** (an open rectangle erase may paint
  over ≤ 2 % non-fill ink outside the text box; the ring seam test passed
  rectangles through the outline — s1 627 0.16, clean at ≤ 0.02, cut from
  0.03). Both thresholds set on review sheets of every kept anchor. Kept:
  **s1 269 → 237** (AR ≤ 2: 228), **sl1w 276 → 222**, **ja_comic 359 → 294**
  (one-bubble 178). `--scene_rejudge` runs one worker per core (≈ 5 s a pool).
  Data (`data/synth.py`, `--scene_mix` draw only): `--single_scenes` (pools a
  one-glyph text may use) + `--single_max_ar` (catches sl1w 192 / 330 / 414 /
  545 at AR 3.5–9), multi-glyph texts route by fit over every pool;
  `--scene_one_bubble <tags>`; a singles-only `--scene_mix single=1` no longer
  needs `--phrase_file`; composites record `scene_pool` and scene reuse is
  counted per pool (indices collide across runs). Also fixed `fit_text`'s
  shrink step, which sat after a `return` since `c434a398` (a font size that
  overflowed on the first try was retried unchanged and the line count
  refused). CPU smoke `data_smoke_d09` (`--scenes s1,sl1w,ja_comic
  --scene_one_bubble ja_comic --single_scenes s1 --single_max_ar 2`, kana +
  words:50, pairs): 600 / 600 drawn, 0 violations of the routing, 205 / 205
  multi-glyph items one column. Seen on its sheet, not fixed: 2–3-glyph
  words draw small in wide bubbles (column width caps `fs` at fill 0.7); a
  second s1 anchor bubble stays as an empty erased bubble.
  **Top-up queued** 23:41 (`20260917-234153-209e43`): tag `s1w`, s1 frames
  and short anchors over sl1w's shapes, `--scene_n 1600 --seed 3`
  (≈ 110 min, ≈ 370 kept expected at ≈ 23 %).
  **Top-up read 2026-09-18 01:30:** `s1w` 388 / 1 600 kept (24 %), even over
  the six shapes, sheet clean. **Residual bar 0.5 → 0.33** after the first
  Δ1 build's pair sheet showed 次 on a sign still reading "y" (s1w 1209,
  residual 0.36): every used scene at 0.35–0.48 (11, ≈ 150 items) kept anchor
  letters, 0.30–0.31 clean; job `20260918-013849-26448e` stopped before
  training, pools rejudged — s1 233, s1w 380, sl1w 213, ja_comic 292,
  one-glyph pool 586 (s1 225 + s1w 361).
- **Δ1 launched 2026-09-18 01:58** (`20260918-015823-5a6c3f`, data + train +
  eval, tag `synth_d1`, arm `d1_s53k`, ≈ 9 h). `--units kana kana_ext*1
  kanji:200*1 list:<13 punct>*1` (kanji 200 = the 53k run's set, user; っ /
  ッ are in `kana_ext`), `--scene_mix single=1.0 --n_items 10000`, scenes
  `s1,s1w,sl1w,ja_comic` routed as above, `--pair_ref en --pair_loss 1
  --lr_rows 2e-3`, random init, `--c_flat 0`, 53 000 steps, rest the 53k
  argv. First build read: **355 rows, not 373** — the singles pool excludes
  the 18 small kana by design, so a singles-only build leaves them untrained
  (as in the 53k run); items per row 12–43 (weighted draw, not a hard quota;
  median 28). Native (`が,ガ,ご,ゴ` × `en,swap`) runs as its own job after.
- **Read before Δ1:** `sheet_scene_pair.png` of the rebuilt data — no text
  in the image other than the pasted string, on every pair shown; pool size
  per shape after the rejudge (each of 250 rows needs ≈ 40 items inside the
  10 k build).

**Δ1 — the full-inventory table with pairs (replaces the 53k seed run;
budget to be set).** Kana + basic kanji, ≈ 250 rows, singles, **random
init, no warm start** (the loss needs none; transplant / pin: an inherited
trigger buys no steps), `--pair_ref en --pair_loss 1 --lr_rows 2e-3`,
`--c_flat 0`, **flat 0** (the arm below), uniform per-row quota (the 53k
run's 91 word rows got ≈ 8 items each and stayed at norm 0.12).

| budget | steps | draws / row (250 rows) | ΔFM wall at 1.63 it/s |
|---|---|---|---|
| 53k | 53 000 | 848 | ≈ 9.0 h |
| 35k | 35 000 | 560 | ≈ 6.0 h |

  ΔFM at 2e-3 ≈ plain FM at 1e-3 per draw (19 vs 20 of 24 at 1 500 steps),
  so the old curve prices it: 848 draws/row ≈ 80–85 %, 560 sits next to the
  490 → 36 % point. The 53k run itself was 490 draws/row over 433 rows.
- **Gate:** the 53k run's evals — singles / ext / kanji 13 · 18 · 18 of 36,
  katakana 1/12, native `en` 36/64 at en cos 0.882 — beaten on singles and
  on the **joint** native number with the tail under plain FM's.
- **Flat share under the pair loss — open, a micro arm before the build.**
  Every ΔFM number is **flat 0** (`data_synth_pair_d0` = 2 000 composites);
  flat 10 % is the plain-FM recipe (flat 0 closed 2026-09-15 on 6 rows:
  rows drift to Latin letters, the bubble becomes the canvas). As built, a
  flat batch under `--pair_loss 1` falls to **plain FM, unscaled**
  (`train/stage.py`: `if a.pair_loss and is_scene`) — a loss 5–8× the
  paired one, sharing Adam's `m` / `v` with the composite steps, pushing
  exactly the bare-canvas component ΔFM cancels. Arms on the Δ0 recipe
  (lr 2e-3 × 1 500, + native): flat 0 (have: 19/24, joint 15, tail 6) ·
  flat 0.1 plain (today's code) · **flat 0.1 paired** (flat sibling = the
  same flat canvas with the Latin reference — built 2026-09-17:
  `render_string(fit_text=)`, flat siblings in `data/synth.py` under
  `--pair_ref en`, train `--pair_flat 1|0`, log keys `*_flat`; data
  `data_synth_pair_d0f` = 2 000 composites + 222 flat, all paired). The
  user's read: plain flat is known behaviour — the arm that matters is
  paired flat 0.1 **clearly above** paired flat 0. Step count from the
  ΔFM flat-0 pair `pairEN_s750_lr2e-3` (queued: train `20260917-221722-34514c`,
  native `…-ebe2e0`) vs `pairEN_s1500_lr2e-3` (have) on native: if 750
  already reads, the flat arm runs at 750 against it, else at 1 500.
  **Read 22:45:** `single` 13/24 at 750 vs 19/24 at 1 500 (row norm 138 vs
  144 — travel is done by 750, identity is not: draws matter for ΔFM too);
  native en both / joint / tail / en cos 22 / 20 / 7 / 0.930 vs 18 / 15 /
  6 / 0.935, swap 9 / 9 / 0 vs 12 / 12 / 0 — the native already reads at
  750 and `single` is off the ceiling there. Plain `pair0_s750` native for
  reference: en 41 / 22 / 25 / 0.855 (ご 0/16), i.e. plain FM's native is
  saturated by 750. Sheet (が en): the glyph is visible on ≈ every image,
  beside JA pseudo-text lines exactly where the EN-reference render has EN
  pseudo-text lines — the co-text is what the base draws for these prompts
  under any `text reads as` clause, "hi" included.
  **Flat 0.1 paired, read 23:10** (`rows_synth_pair_d0f_pairENflat_s750_lr2e-3`,
  jobs `20260917-224728-945434` / `…-f47e7f`, no lr warmup on any arm):

  | 750 steps, lr 2e-3 | `single` | en both / joint / tail / en cos | swap both / joint / tail |
  |---|---|---|---|
  | paired, flat 0 | 13/24 | 22 / 20 / 7 / 0.930 | 9 / 9 / 0 |
  | paired, flat 0.1 paired | 14/24 | 27 / 23 / 7 / 0.927 | 12 / 12 / 0 |

  Not "clearly above": singles flat, native up by a few on every count with
  the tail and en cos unmoved; per-image discordant pairs 13 : 8 (en) and
  8 : 5 (swap), pooled 21 : 13 (McNemar p ≈ 0.17). Mechanically clean: the
  flat paired loss is 0.012 against 0.059 plain on the same items, `pres_flat`
  7e-4, norm trajectory identical to flat 0. Sheet (ゴ en, same prompts and
  seeds): flat 0 embeds the glyph in katakana pseudo-words (ゴマダグン,
  コゴン, ゴゴゴ, ガフハレ) on about half the images; with flat 0.1 it
  stands alone, large and clean, on 10 of 16 (6 of 16 before) — the
  unit-count reading of the flat items.
  **Corrected the same night (user, side-by-side read): flat 0.1 costs the
  scene, and the rulers under-read it.** Same prompt and seed, flat 0 above
  flat 0.1: the flat arm pulls the render toward the flat item's look — a
  large bold black glyph, a plain canvas or a black box behind it, the
  scene thinned (が / ガ / ご on p06, が on p01 / p02, the boxed が on the
  p02 swap). en cos barely moves (mean −0.003 en / −0.006 swap) because it
  is a whole-image cosine; the paired count does show it on swap (8 images
  lower by > 0.02 vs 2 higher; box IoU 0.47 → 0.39; en 10 vs 7). The native
  gain above and this loss are one thing — a big clean glyph is what the
  readers hit — i.e. the wipe ↔ readable axis again, entered through the
  data instead of the loss. **Δ1 runs flat 0.** Open with it: the plain-FM
  flat-0 failures (日 drawn as Latin "a", the bubble as the canvas) were
  not seen under ΔFM flat 0, but Δ0's 12 rows are all kana — read the
  first kanji rows of Δ1 on the sheets; if they drift, the lever is a small
  share of *small* paired flat glyphs (jitter sizes, 60 px), not 10 % at
  110–200 px.
- **Known going in:** singles-only ΔFM leaves the co-text on natives and
  strings render one piece (Run 3); Δ1 is a seed table for Δ2, not a
  shippable one. The ≈ 9 min `がガゴ` multi-glyph native on the Δ0 table is
  the cheap evidence for Δ2's premise and can run any time before Δ1.
- **Guard:** 2e-3 is measured at 12 rows only; at 250 rows a row is in
  ≈ 1.6 % of batches (dense AdamW, β₂ 0.99 — `v` decays between visits).
  Watch `delta_norm_mean` per draw against the Δ0 arms over the first few
  thousand steps; katakana got 1/12 at equal exposure in the 53k run, an
  over-quota is the first lever.
- **Chunking:** if the 10 k-item build forces it, train disjoint row
  blocks and concatenate by ext id (`--init_rows a.pt,b.pt`) — bit-exact
  for single-block captions; cross-block strings are untrained until a
  mixed pass with `--lr_warmup 500 --init_anchor 0.3`.

**Δ2 — the sentence recipe with pairs (≈ 3.5 h; after Δ1 and the
`sent-a0p3-s05` report).** The reading that keeps the line open: on a
single-glyph native the base free-runs a sentence's worth of JA text and
the table addresses one glyph of it, so everything else comes out as
pseudo-text; plain FM hides that by wiping the scene. When the caption
carries a whole sentence of trained rows there is no unaddressed text left
to invent — *if* that holds, ΔFM's cost disappears where the target lives
and its scene-holding stays. The `sent-a0p3-s05` argv (`README.md` *How to
run*) with `--pair_ref en --pair_loss 1`; every kind paired, `short` /
`sentence` siblings as vertical Latin stacks with the item's cuts (code
owed in `data/synth.py`). Warm rows from Δ1's table; steps and lr from L0 / Δ1.

- **Gate:** the target stage and native both clauses against the finished
  `sent-a0p3-s05` — joint number not below it, tail under it, and the
  sheets read for text *outside* the addressed string (the co-text count
  is the number this arm exists for). Flat singles, `short` / `phrase` and
  their `_held` not below the baseline.
- **Kill:** co-text beside a correctly rendered sentence at the single-glyph
  rate → close the line; ΔFM stays in the tree as a flag, one row in
  `findings.md` *What does not move it*.
- **Guard:** the sibling stacks keep one column when the item does (read
  the `_ref` images on the composite sheet before launch); warm rows keep
  lr 1e-3 unless L0's guard arm clears 2e-3 on a warm start.
- A cheaper look first, if wanted: the Δ0 tables under a native caption
  that quotes *several* trained glyphs (`がガゴ`) — does the co-text shrink
  as the addressed share of the text grows? One native stage, ≈ 9 min, no
  training.

## Open risks

- **The co-text is the base, not the loss — measured (Δ0b).** What is open
  is only whether it needs unaddressed text to appear; if a fully
  addressed sentence still draws pseudo-text beside it, ΔFM trades the
  wipe for a defect of the same size and the line closes at Δ2.
- **The micro regime flatters everything.** 12 rows from scratch reach
  83 % at 500 draws; L0 is read as a lever here and confirmed at a
  mid-size inventory before any multi-hour run.
- **lr 2e-3 on a warm start.** The anchor recipe was tuned at 1e-3; Δ2
  keeps 1e-3 on warm rows unless L0's guard arm says otherwise (Δ1 is
  random init, 2e-3).
- **`--free_residual` is unscaled against the paired loss** (5–8× smaller
  than plain): at 1e-3 the ΔFM norm peaked at 126 and sagged to 115. At
  2e-3 the norm reaches 144, so it is not gating; if a later ΔFM arm under-travels
  again, 3e-4 is the first thing to move.
- **Reader on dakuten / on co-text.** `single` exact held up against the
  sheets on Δ0; native `both` under-counts ΔFM images where the glyph is
  visible beside pseudo-text. The sheets stay the second read.
- **RAM.** Reference latents (+ 2.6 GB at 10 k items fp32, half in bf16)
  and captions (pool-bounded) on top of the text cache; the 10 k-item
  build stays the limit for Δ1 / Δ2.
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
  cancellation, and cancellation is not the bottleneck (≈ 80 % already, no
  exposure saving). Needs a trained table for the reference row. Revive
  only if Δ2 passes and in-box stroke quality is the complaint.
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
- **lr 3e-3.** Closed (`findings.md`: off-manifold at 2.4× row norm); L0
  tests 2e-3 only.
