# plan_synth — the S line, live plan (rows on self-generated scene composites)

> **Status 2026-09-15 (night):** the full-scale cap-only isolation was
> killed at 15 min (user: too large a run for the question) and the S
> line moved to a **micro loop** — 6 rows (あかす出人日), 2 000 steps,
> ≈ 25 min per arm incl. native — for the trigger / canvas / frame
> questions. Five arms + three re-renders settled today (table in
> *Micro loop*; chronology `reports/synth_micro_loop_2026_09_15.md`): the `c_flat` cap is not a lever
> (cap 0.75 = removed), **composite share is** (0.4 → 0.9: hit & kept
> 23 → 41, en cos 0.797 → 0.860), the rows are **bound to the JA clause
> frame** (in the EN ref's own caption か renders as Latin strokes 0/16
> while あ 11/16 lands as a word token would), and Q fixed on in
> training routes the trigger *across frames* for か (0 → 7) at an
> identity cost elsewhere (日 → 目) — glyph-dependent, not rankable on
> 6 rows. Rulers changed: **`en cos` / `box IoU` against a shared EN
> reference** (`English text reads as "hi"`, same prompt and seed)
> replace the floor-based kept margin; floor renders are off by default.
> **Flat 0 measured 20:50 and closed** — worse on every ruler, wipes
> unchanged; the flat items hold identity, the bubble is the composites'
> own canvas. **Frame-mix arm running (22:56, jobs `74db20` / `74f7f3`)**:
> 12 rows on the s0 + s1 scene pool (four prompt frames from the image,
> `sign` = the first non-bubble placement), new lettering fonts, inherited
> ink colour, ±7° tilt — data `synth_micro12_fm`, Q off; result owed below.
>
> What the S line is, how the instrument works, the S0 recipe / result and
> the measured budgets moved to [`synth.md`](synth.md); chronology is
> [`reports/`](reports/README.md). The P-line record (`plan.md`) stays the
> flat-only control; no P-line weights are used anywhere in the S line.
> Target artefact, kill criteria and the P2–P4 phase content in `plan.md`
> stand, re-based on the S table.

## Data mix (micro recipe of record; live build `data_synth_micro12_fm`, verdicts on `data_synth_micro6_c9`)

The S0b mix (40 flat / 40 composite / 20 flat phrases, one flat layout,
`c_flat` cap 1.5) is the flat-only prior measured as the problem: `f`
learns the flat canvas first and 40 % composites do not undo it. The
micro loop moved the share and the number moved with it — this table is
what stands. The shares are settled (c9, flat 0); the **frame / font /
ink / tilt axes** were added 2026-09-15 night and are what the running
arm measures.

| share | source | as built | why |
|---|---|---|---|
| **90 %** | scene composites: one unit in the scene's own text slot — a bubble, or a held sign (singles only at the micro scale; phrases return at full scale where the region holds them) | `--scene_frac 0.9`, `--scene_fill 0.7`, `erase_miss` gate; pool `--scenes s0,s1` = **443** scenes (s0 174 `reads_as` + s1 269 over four frames; the open-fill / seam rules of 2026-09-15 in) | **the measured lever**: 0.4 → 0.9 lifts hit & kept 23 → 41, en cos 0.797 → 0.860, with flat singles 12/12 and EN 24/24 held (on s0 alone) |
| render | glyph font / colour / tilt | `pick_font` over 16 faces (`assets/fonts/FONTS.md`: 源暎アンチック, 源柔 / 源真ゴシック, コーポレート・ロゴ, たぬき油性マジック, 破線G, こよみゆる, Noto Serif CJK; Noto Sans out), cmap-checked per string; ink = the anchor's own lettering colour (`anchor_ink`) when it contrasts; 30 % tilted ±7° | every constant of the render (one gothic face, black ink, dead-level) is one more thing a row can absorb — flat 0 showed the rows take whatever is constant. Untested as a lever; rides in the frame-mix arm |
| **10 %** | flat singles in the font bubble | `--flat_bubble 1.0` (one flat layout) | **identity and frame-independence exposure, measured**: flat 0 lost 日 to Latin "a", JA hits 60 → 46, swap 23 → 5, and wiped exactly as often — the flat items are not the wipe source. Keep; the share above 10 % is untested at 0.9 |
| 0 % | natural phrases on flat canvases | `--natural_frac 0` | a flat item is what the rows over-learn; if phrases come back at full scale they ride *inside composites*, not on flat canvases |
| 0 % | random-order strings | `--strings_frac 0` (S1's question; the lever stays) | |
| 0 % | real corpus crops | out (two in three labels wrong, `datacheck.md`) | |
| **caption frame** | s0 composites are all the JA `reads as` clause on a bubble | **built 2026-09-15 (`--scene_frames`)**: the scene prompt draws a frame — `reads_as` / `bubble_reads` (`There is a speech bubble that reads "…"`) / `saying` (`She is saying "…"`) / `sign` (`He is holding a sign that reads "…"`) — and the composite caption swaps the JA text into the *same* frame. In the live build: reads_as 1 472 / bubble_reads 781 / sign 379 / saying 248 composites | rows are frame-bound (か 0/16 as Latin strokes under `swap`); the frame now comes from the image, no caption-only `--frame_mix` needed. s1: 269/1000 kept (27 %), `bubble_reads` 36 %, **`sign` 34 % and the first non-bubble placement (99 px boards)**, `saying` 21 % (still a bubble), `reads_as` 17 %. **`sfx` set aside** (user, 23:00): 102 kept but the base draws the word on a title bar / banner, not as SFX — pool `s1sfx` exists, not in the mix |

Live build 2026-09-15 22:53 (`--only_chars あかすのみは出人日大目月`,
`--words 0`, `--n_items 3200`): 3 200 items = 320 font + 2 880 scene over
443 scenes, all singles, ≈ 27 flat + 240 composite items per row (the
c9 ratio, twice the rows). The 6-row `data_synth_micro6_c9` (1 600 items,
s0 only, Noto Sans, black ink, no tilt) is the control the arm reads
against; `data_synth_micro6` (0.4 share) the older one.

## Recipe of record (micro, rows arm, from scratch; full-scale run owed)

    Δ_r = f_r        (row-norm units; no shared vector)

- **No `c_flat`** (`--c_flat 0`): cap 0.75 ≡ no `c_flat` on every native
  number and on the sheets (23 vs 24 hit & kept); the switch is dropped,
  not tuned. No encoder, no warm start (`f_r` from zero), **no Q at train
  time** (glyph-dependent — decision tree). Rows-only is the S line.
- σ band 0.7–0.9; rows lr 1e-3, cosine; `μ‖f‖²` pull 1e-3
  (`--free_residual`); rectified flow on the band, `--box_weight 4` inside
  the swapped box.
- Batch 4, compile, no grad-ckpt, pool `448,512:2,448x512,512x448`.
- **Micro scale (what every verdict above is on):** 6 rows
  (`--only_chars あかす出人日` — 3 kana + 3 corpus kanji, one Qwen piece
  each), 2 000 steps ≈ 13 min at 2.5 it/s, + native on `en` and `swap`
  ≈ 25 min per arm. Arm `rows_synth_micro6_c9_m6c9_s2k_qoff` (job
  `54e3e5`). The frame-mix arm widens this to 12 rows (6 kana あかすのみは
  + 6 kanji 出人日大目月), same 2 000 steps (so ≈ 670 draws per row, half
  of c9's), arm `rows_synth_micro12_fm_m12fm_s2k_qoff` (jobs `74db20` /
  `74f7f3`); native on あ か す 日 as before.
- **Full scale (owed, the gate run):** inventory `--kana_ext --kanji 100
  --words 120`, held-out words 8, small kana only inside words; 24 000
  steps ≈ 2.6 h. 246-row interference at composite 0.9 is untested (W2's
  24-kana collapse at 512²); the micro verdicts are on mechanism, not on
  capacity.
- Eval: `stage native` renders trained conds only (`--native_floor 0`),
  both clauses (`--native_clauses en,swap`), scored against the shared
  `English text reads as "hi"` refs (`--stage enref` once, arm-independent).
- Controls: 0.4-share arms (`…micro6_m6_s2k_{nocflat,cap075}`), S0b
  (`rows_synth_s0b_s24k_S0b`, full scale, cap 1.5) and P0b
  (`encoder_wdsek_w120_s24k_p0b`, flat-only).

Exact train argv: `output/daemon/jobs/20260915-180203-54e3e5/job.json`
(`--stage train eval --arm rows --data_tag synth_micro6_c9 --train_steps
2000 --batch 4 --t_min 0.7 --t_max 0.9 --compile 1 --grad_ckpt 0
--aggressive_recompute 0 --lr_rows 1e-3 --lr_decay cosine --free_residual
1e-3 --box_weight 4 --seeds 2 --no_floor --c_flat 0`). The S0b argv stays
in `reports/synth_s0_s0b_2026_09_15.md` "S0b build + launch" as the
full-scale template (swap its data flags for the table above).

## Gates (S0 gates stand for the full-scale run; rulers re-based 2026-09-15)

Scene ruler is now **`en cos`** (PE-Spatial cos to the EN reference of
the same prompt/seed; floor ≈ 0.93–0.97 is the ceiling), placement ruler
**`box IoU`** (glyph box vs the "hi" box; floor 0.36–0.51, every trained
cond so far 0.05–0.25 — read with the sheets, it is harsh on small
boxes). `stage native` renders trained conds only (`--native_floor 1`
restores the old kept margin). Native runs on **both** clauses — `en`
(the trained JA frame) and `swap` (the EN ref's caption, word swapped) —
and a row counts as a word token only when it hits under `swap`.

- **native (EN clause, 8 held-out prompts × 4 kana × 2 seeds): hit & kept ≥ 24/64**, from
  P0b's measured baseline of **2/64** (its 32/64 hits are canvas wipes on
  the margin ruler). Scene-kept alone ≥ 56/64 (the delta must stop
  overriding the scene).
- singles ≥ 30/36 (P0b 36; reader noise), `single_ext` non-small ≥ 18/28,
  `single_kanji` ≥ 22/36, word ≥ 8/32, EN 24/24 — nothing P0b holds is
  lost.
- `phrase_held` > 0/16; `flip` order statistic ≥ 24/48 if strings are in.

Also read, not gated: row norm at the end of training (every arm drives
it to ≈ 125–130; the placement / identity trade-off is the delta norm),
per-char hits under `swap` (frame independence is per glyph, not per
arm — か vs あ), and the `native_swap` sheets before trusting a box IoU on
a small box.

## Prediction on record (frame-mix arm, ≈ 12 rows)

With captions drawn from the frame mix and Q off, `swap` hits rise on the
rows that are Latin-distorted today (か-type) while JA-clause hits hold
≥ 50/64 and en cos stays ≥ 0.86; rows already frame-independent (あ-type)
move little. If か-type rows stay at 0/16 under `swap` even when a third
of their exposure was the EN frame, the script decision is the DiT's
reading of the frame, not the row's (third branch below). Q on top of the
mix is expected to add nothing the mix did not, at the same identity cost
(目-type) — the arm is there to close Q, not to rescue it.

## Micro loop (2026-09-15 evening) — what is settled at 6 rows

Data: `synth_micro6` (60 / 40 flat / composite, 1 600 items) and
`synth_micro6_c9` (10 / 90). Native あかす日 × 8 prompts × 2 seeds; hits =
both readers.

| arm | JA clause hit / en cos | swap clause hit / en cos | note |
|---|---|---|---|
| 0.4, cap 0.75 | 56 / 0.817 | – | = no `c_flat` on every number |
| 0.4, no `c_flat` | 57 / 0.797 | 10 / 0.830 | |
| 0.4, Q fixed | 52 / 0.829 | – | |
| **0.9, Q off** | **60 / 0.860** | 23 / 0.903 | か 0/16 under swap (Latin strokes), あ 11/16 |
| 0.9, Q on | 48 / 0.838 | 23 / 0.885 | か 7/16 under swap, 日 → 目 (JA 3/16) |

Settled: cap ≠ lever; composite share = lever; rows are frame-bound;
Q = frame-independence for some glyphs at an identity pull for others.
Micro verdicts are on mechanism; 246-row interference is untested
(W2's 24-kana collapse) — the winner needs one full-scale run.

## Decision tree (next: frame-mix × Q, ≈ 12 rows)

Data: composite 0.9 on 12 rows (6 kana + 6 kanji) over the **s0 + s1
scene pool** (`data_synth_micro12_fm`), so the composites carry four
frames in image *and* caption (`reads_as` / `bubble_reads` / `saying` /
`sign`; built 2026-09-15, see the s1 and s1sfx sections of
`reports/synth_micro_loop_2026_09_15.md`) and identity stops leaning on
one clause; fonts, ink colour and tilt ride along (not separable in this
arm — if it moves, the ablation is s0-only data with the new renders).
Q off first (running); Q on only if the first branch below fails. Native
on `en` + `swap`. Read against c9 (JA 60 / 0.860, swap 23 / 0.903, seed-0
wipes 11/32, flat singles 12/12) with the caveat that 12 rows at 2 000
steps see half the draws per row.

- **Frame mix alone lifts swap hits (≥ 40/64) with JA hits held (≥ 50)
  and no 目-type identity loss** → Q stays closed; frame mix goes into
  the full-scale S recipe with composite 0.9.
- **Frame mix lifts swap hits only with Q on** → Q is part of the recipe;
  the identity pull is then the open cost (measure on the 6 kanji;
  `--free_residual` and the Q scale 1.0 → 0.5 are the two levers).
- **Neither moves swap hits** → the script decision lives in the DiT's
  reading of the frame, not in the row; that is W3-shaped (DiT-side)
  work, and the row path ships JA-clause-only.
- **Flat 0 is measured and closed** (2026-09-15 20:50,
  `rows_synth_micro6_c10_m6c10_s2k_flat0`): seed-0 wipes unchanged
  (11/32), JA hits 60 → 46, swap hits 23 → 5, 日 drifts to Latin "a";
  with no flat items the rows learn the *bubble* as their canvas (white
  disc on black). Flat 10 % stays. The wipe is the delta norm, not the
  mix; the next data lever after frame mix is **where composite text
  sits** — bubble / subtitle bar / directly on the scene, as the EN refs
  place "hi" — the composite form of the position-jitter idea (scene
  stage; needs non-bubble text placement in `stage/scenes.py`).

## Open risks

- **Small glyphs inside composites.** A kana in a 64–100 px bubble at 512
  is 4–6 latent tokens a side; the box-weighted loss and the 32 px floor
  are the mitigation. `--scene_fill 0.7` made this slightly worse (median
  51 px) in exchange for a realistic layout; if singles or dakuten fail,
  raise the bubble bar, not the loss.
- **Erase artefacts as a cue.** Ring-median fill inside a shaded bubble
  can leave a patch the row latches onto. `erase_miss` catches the
  wrong-blob case, not the patch; the sheets are the check.
- **The residual wipe is the delta, not the mix.** Seed 0 wipes 4/8
  prompts at flat 10 % and at flat 0 alike (was 7/8 at 60 %); every arm
  drives the row norm to ≈ 125, and that is what overrides the scene.
  The wipes are seed-shaped, so per-char n = 16 hides differences under
  ≈ 8. No data-mix arm is expected to move this further.
- **The bubble is a canvas.** Every composite puts the glyph inside a
  round white bubble; with flat items scarce the rows learn the bubble as
  their unit (flat 0: white disc on black). Placement diversity in the
  scene stage (subtitle bar / on-scene text) is the lever, not more
  bubbles.
- **Capacity at 0.9.** Every 0.9 verdict is on 6 rows drawn ≈ 1 200
  composite steps each (240 distinct items per row); the full inventory
  at 16 k items / 24 k steps gets ≈ 260 composite draws per row from ≈ 45
  distinct items, on ≈ 51 px glyphs — 5× less per row, and the 246-row
  interference of W2 on top. If full-scale singles fall, the levers are
  distinct composites per row (`--scene_n 2000`, `--scene_min_box` 56 →
  72) and steps, not the share.
- **Scene yield.** 17 % kept at 56 px, 174 after the erase gate; each
  scene serves ≈ 37 swaps. Raising the bar needs `--scene_n 2000` (the
  prompt list is a stable prefix, only the missing 1 000 render).

## Not this plan

- Regulariser strength / `out_scale` sweeps (scale probe: direction, not
  magnitude).
- Zeroing or shrinking `c` at inference (table-parts probe: every part
  alone is 0/16). Warm-starting from P0b is now *measured*, not just
  argued (2026-09-15, `--init_rows`): identity survives, the trigger never
  grows — not a shortcut. The flag stays for seeding from S0's `f` later.
- Inference-time guidance away from a `c`-only branch: `f + g` without `c`
  was 0/16, so that direction removes the glyph before the canvas. A
  64-render curiosity at most.
- The glyph encoder in any form (see *Recipe*); the S line is rows-only.
- Contrastive terms on text-free native images.
- Pasting onto the dataset's real images (off-manifold paste, caption
  style mismatch, nsfw/artist tags) — the self-generated scene replaces it.
- `c_flat` in any form: the micro loop measured cap 0.75 ≡ removed
  (2026-09-15); the S recipe drops the switch.
- Q (the quoted-EN adapter-output direction) as an inference-time
  replacement for a trained `c` — measured, halves exact hits. Q fixed on
  in training is *measured, not closed*: glyph-dependent (decision tree).
- The floor-based kept margin as a gate — replaced by `en cos` / `box
  IoU`; a bare ground with a bubble scored as kept, and the base itself
  wipes `portrait, simple background` for EN.
- Restarting a running arm for a monitor value (leak, `‖c_flat‖`): the
  gates read at the end, and a mid-run change loses attribution.
