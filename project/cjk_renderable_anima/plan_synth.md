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
> **Next arm**: frame-mix data × Q on/off on ≈ 12 rows (below).
>
> What the S line is, how the instrument works, the S0 recipe / result and
> the measured budgets moved to [`synth.md`](synth.md); chronology is
> [`reports/`](reports/README.md). The P-line record (`plan.md`) stays the
> flat-only control; no P-line weights are used anywhere in the S line.
> Target artefact, kill criteria and the P2–P4 phase content in `plan.md`
> stand, re-based on the S table.

## Data mix (S0b, `data_synth_s0b`)

| share | source | S0b | why |
|---|---|---|---|
| 40 % | flat-canvas singles, uniform over kana + ext (non-small) + kanji + words, ext and kanji at 2× | **all in the font bubble** (`--flat_bubble 1.0`; S0: 60 / 40 bubble / plain) | identity exposure per row; one flat layout so `c_flat` is one direction |
| 40 % | scene composites: singles and phrases inside a generated bubble | `--scene_fill 0.7`, `erase_miss` gate, **174** scenes | the row learns the glyph, not the canvas |
| 20 % | natural phrases on flat canvases (`--natural_frac`) | in the font bubble | the product distribution; T5 contextualises |
| 0 % | random-order strings | out (S1's question; `--strings_frac` keeps the lever) | |
| 0 % | real corpus crops | out (two in three labels wrong, `datacheck.md`) | |

Built 2026-09-15: 15 967 items = 6 400 font + 3 200 phrase + 6 367 scene
(6 072 single / 295 phrase — the 0.7 fill halves bubble capacity, so
composite phrases fell from S0's 851; small kana inside composite words get
≈ ⅓ of S0's exposure). 0 plain captions. Composite single box short side
p10 43 / median 51 px.

## Recipe of record (S0b — rows arm, from scratch)

    Δ_r = f_r + 𝟏[item is flat-canvas] · c_flat        (row-norm units)

- No encoder, no warm start (`f_r` and `c_flat` from zero). Rows-only is
  the S line; see *Not this plan*.
- σ band 0.7–0.9; rows lr 1e-3, cosine; `μ‖f‖²` pull 1e-3
  (`--free_residual`); `c_flat` at the rows lr, **cap 1.5** (S0: 0.75,
  pinned from step 600; 1.5 pins from step 1 100 — the flat layout is a
  large shared direction and the cap is still clipping it).
- Inventory `--kana_ext --kanji 100 --words 120`, held-out words 8; small
  kana only inside words.
- 24 000 steps, batch 4, compile, no grad-ckpt, pool
  `448,512:2,448x512,512x448`; ≈ 2.6 h at 2.53 it/s.
- Loss: rectified flow on the band, `--box_weight 4` inside the swapped box.
- Arm `rows_synth_s0b_s24k_S0b`; controls: S0 (`rows_synth_s0_s24k_S0`,
  two flat layouts, cap 0.75, 186 scenes incl. 12 erase misses) and P0b
  (`encoder_wdsek_w120_s24k_p0b`, flat-only).

Full S0b argv: `reports/synth_s0_s0b_2026_09_15.md` "S0b build + launch".

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

Also read, not gated: `leak` at the end of training and `‖c_flat‖`
(pinned = cap binding), and `--delta_parts f,c` on `native` (`f` alone
should hit — the point of the switch).

## Prediction on record (S0b)

leak ≪ 0.28 and kept ≥ 56 from the single flat layout; hit & kept up
through fewer wipes; singles recover **only if** the plain leak was
costing `f` — P0b held 36/36 with both layouts in `f`, so singles near 20
with the fill and gate in would point at composite glyph size / phrase
share, not layout.

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

Data: composite 0.9 on ≈ 12 rows (6 kana + 6 kanji), captions drawn
from a **frame mix** — the JA clause (`japanese text. Japanese text
reads as "…"`), the EN swap clause (`english text. English text reads
as "…"`), and bare quotes — so identity stops leaning on one clause
(needs a `--frame_mix` lever in the data stage). Two arms, Q off / Q on,
same seed; native on `en` + `swap`.

- **Frame mix alone lifts swap hits (≥ 40/64) with JA hits held (≥ 50)
  and no 目-type identity loss** → Q stays closed; frame mix goes into
  the full-scale S recipe with composite 0.9.
- **Frame mix lifts swap hits only with Q on** → Q is part of the recipe;
  the identity pull is then the open cost (measure on the 6 kanji;
  `--free_residual` and the Q scale 1.0 → 0.5 are the two levers).
- **Neither moves swap hits** → the script decision lives in the DiT's
  reading of the frame, not in the row; that is W3-shaped (DiT-side)
  work, and the row path ships JA-clause-only.
- Either way, **flat 0** (composite only) is the pending data point for
  the remaining seed-0 wipes (`n_flat > 0` assert to lift; identity
  without flat exposure to check).

## Open risks

- **Small glyphs inside composites.** A kana in a 64–100 px bubble at 512
  is 4–6 latent tokens a side; the box-weighted loss and the 32 px floor
  are the mitigation. `--scene_fill 0.7` made this slightly worse (median
  51 px) in exchange for a realistic layout; if singles or dakuten fail,
  raise the bubble bar, not the loss.
- **Erase artefacts as a cue.** Ring-median fill inside a shaded bubble
  can leave a patch the row latches onto. `erase_miss` catches the
  wrong-blob case, not the patch; the sheets are the check.
- **Leak floor.** With one flat layout, leak 0.18 at step 10 k is either
  cap overflow (`‖c_flat‖` pinned) or a glyph-size direction (flat glyphs
  are always big, composite glyphs small — flat-specific, so it lands in
  `c_flat` or `f`). `leak` cannot tell the two apart; the cap-only run
  can.
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
