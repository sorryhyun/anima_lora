# plan_synth — the S line, live plan (rows on self-generated scene composites)

> **Status 2026-09-15 (evening):** S0b **missed** — worse than S0 on every
> ruler (native `f` alone 1 / 50 / 0 vs S0 25 / 48 / 15; `f+c` 25 / 24 / 9;
> singles 15/36). Leak did fall (0.28 → 0.18) but the 1.5 cap let `c_flat`
> take the render trigger, so the rows went silent. Two probes closed the
> same day: a P0b warm start (identity survives, trigger never grows;
> `f` 0/64) and the pretrained quoted-EN direction Q at inference (kills
> the wipes, halves exact hits; S0 15 → 9). Verdicts in
> [`findings.md`](findings.md) *Settled — trigger vs canvas*. **Live arm:
> cap-only isolation** `rows_synth_s0b_s24k_S0b_cap075` (S0b data, cap
> 0.75; train + eval `20260915-152223-870a1d`, native
> `20260915-152223-5b4509`) — attributes S0b's collapse to the cap vs
> `--scene_fill 0.7`.
>
> What the S line is, how the instrument works, the S0 recipe / result and
> the measured budgets moved to [`synth.md`](synth.md); chronology is
> [`history.md`](history.md). The P-line record (`plan.md`) stays the
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

Full S0b argv: `history.md` "S0b build + launch".

## Gates (unchanged from S0)

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

## Decision tree (after S0b's eval + native)

S0b landed in the **wipes remain, sheets show the flat canvas** branch
(kept 24–50 < 56; `fc` draws the font bubble, `f` alone a bare ground +
glyph — note the scene-kept ruler scores the latter as *kept*, read the
sheets). The cap-only run is the pre-registered response, **with one
correction from S0b's data**: the cap is isolated *downward* (0.75, S0's
value), not off — 0.75 → 1.5 moved the trigger from `f` into `c`, so a
larger cap removes the glyph before the canvas.

- **Pass** (hit & kept ≥ 24, kept ≥ 56, singles ≥ 30) → S1 (strings,
  repeat mode) warm-starts from that table on this data.
- **Isolation recovers `f` alone to ≈ S0 (hit ≥ 20)** → S0b's collapse was
  the cap; keep cap 0.75 and S0b's data, and push the only lever that
  puts the trigger in `f` without the canvas: composite share 40 → 60 %
  (`--scene_frac`) and glyph floor `--scene_min_box` 56 → 72 with
  `--scene_n 2000`, one at a time.
- **Isolation stays near S0b (`f` hit < 10)** → the collapse was
  `--scene_fill 0.7` (smaller composite glyphs, phrases 851 → 295);
  revert to `--scene_fill` 0.85 with the erase gate and one layout, cap
  0.75.
- **Either way, the Q-in-training arm is next after that**: rows trained
  with the quoted-EN direction fixed on at the ext positions for every
  item (`OutVec` in the train stage), `c_flat` cap 0.75 for canvas only,
  same data. Prediction on record: `f` alone hits ≥ S0's 25 with kept ≥
  56, because the trigger is supplied and never has to be learned into
  either part; failure = hits stay in the subtitle mode (Q at inference:
  13 / 54 / 9).
- **Wipes remain but the sheets show erase artefacts** (patch, ring) →
  inspect `sheet_scene.png`, redo the erase (inpaint ring, not flat fill).
- **Singles < 30 with kept passing** → the identity budget: `--single_frac`
  0.4 once (flat share 40 → 50 %), or the glyph floor above; not the
  layout, not the cap.
- **Word / small kana fall further** → composite phrases are down to 295;
  raise bubble size (`--scene_min_box`) before touching `--scene_fill`.

The pre-registered hybrid isolation run (`--arm encoder`, same data) stays
available but is not first: the leak split already names a data ×
parametrisation interaction.

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
- `c_flat` cap sweeps beyond the single isolation run above, and any cap
  *above* 0.75: S0b measured that the trigger follows the room.
- Q (the quoted-EN adapter-output direction) as an inference-time
  replacement for a trained `c` — measured, halves exact hits. Its only
  open use is training-time (decision tree).
- Restarting a running arm for a monitor value (leak, `‖c_flat‖`): the
  gates read at the end, and a mid-run change loses attribution.
