# plan_synth — the S line: rows trained on self-generated scenes (2026-09-14 night)

> Opens the **S line** (S0 identity table, S1 strings / repeat mode) and
> retires the P-line's data. The P-line record (P0a canvas shapes, P0b
> singles at scale) stays in [`plan.md`](plan.md) / [`history.md`](history.md)
> as the flat-only control; **no P-line weights are used anywhere in the S
> line**. Target artefact, kill criteria and the P2–P4 phase content in
> `plan.md` stand, re-based on the S0 table.

## Why (what P0b's native probe showed)

P0b (`encoder_wdsek_w120_s24k_p0b`, 24 k steps on the 384–512 pool, full
kana inventory + 100 kanji) passes every singles gate but the new voiced /
small rows: basic 36/36, kanji 27/36, word 11/32 (20/32 at 384×512), EN
24/24, `single_ext` 19/36 against a 24 bar. The `single_ext` miss splits
three ways and only one is exposure: small kana drawn full-size (ゃ→や,
ィ→イ — a lone glyph has no size reference, so *small kana are not a
singles concept*), SFX-reader misses on correct renders (づ ご げ べ; either
reader 22/36), and dakuten drawn as handakuten (び→ぴ, ぼ→ぽ, both seeds).
Table capacity is fine (`table_pr` 2.64, `rel_max` 2.51, P0a's range).

The `native` stage (8 blind-pairs scene prompts × あ か す ぐ × 2 seeds, EN
clause `{scene}, japanese text. Japanese text reads as "{k}"`, delta 0 / 1)
reads **32/64 trained vs 0/64 floor** — and the sheets say the number is
the wrong thing measured. Nearly every hit is the *training canvas
verbatim*: a white bubble on black, or a bare grey canvas with the glyph,
and the classroom / bench / cafe is gone (す: 10 of 11 hits wipe the
scene; the one prompt at 8/8 is "simple background", where a bare canvas
is the right answer anyway). When the scene wins, the text block is a
wall of garbled pseudo-Japanese and the kana is lost or doubled (すす,
かかか, く for ぐ). Same 5/16 and 8/16 on あ か as the W2-era rows arm: the
P0b table neither gained nor lost scene survival.

So the row is an address for the whole training image, not for the
glyph: every item was "glyph on a flat ground", the ground is perfectly
correlated with the glyph, and no loss term can tell the two apart. The
shape pool changed canvas *size*, not canvas *content*, so it could not
remove this.

**Scale probe (`--delta_scale`, trained cond only, 64 images each):** at
0.5 the scene comes back and the kana goes — 2/64 (あ 2/16, rest 0); 0.7
gives 22/64 (`native_x0.7/`). Magnitude is not the lever; a stronger
`‖f‖` pull or a smaller `out_scale` would only slide along the same axis.

**Table-parts probe (2026-09-14 night, `native_parts/`, 64 renders;
`history.md`):** the saved table splits exactly into `g + c + f`. Rendered
one part at a time, **every part is 0/16 hits**: `f`, `g`, `f+g` keep the
scene and draw floor garble; `c` alone draws the training canvas (big
white bubble, a generic ん) and no requested kana. The canvas mode is `c`,
but the identity in `f`/`g` only renders *given* that mode — the rows
learned "glyph | flat canvas", and no linear split separates the two.
Consequences: zeroing `c` at inference is not a fix, and a warm start from
P0b/P0a carries a conditional identity the composites would have to
re-learn. Measured on the new scene-kept ruler, P0b's honest baseline is
**hit & kept 2/64**.
Contrastive terms on native images without text are a trap (the caption
claims text the image lacks → the row learns to be silent); a
preservation term outside the text box needs scene images anyway.

## The idea (user, 2026-09-14): self-generated scenes, bubble swapped

Let Anima draw the scene itself, then replace only what is inside the
bubble:

1. **Scene generation** (GPU, base model, delta off): `{scene tags},
   speech bubble, english text. English text reads as "{anchor}"` with a
   short EN anchor (hi ok no yes wow hey — Latin, every piece pretrained,
   no ext row touched). The output is in-domain by construction, and the
   caption explains everything in the image.
2. **Box filter** (same pass): the `Readers` detector finds the text box;
   keep an image only when exactly one box is found and its read matches
   the anchor. Record `(file, scene tags, box)`.
3. **Swap** (CPU, `--stage data`): fill the box with the bubble's local
   colour (median of a ring around the box), draw the kana / word / phrase
   fitted into the box with the existing font renderer. One scene serves
   several swaps.
4. **Caption**: `{scene tags}, speech bubble, japanese text. Japanese text
   reads as "{text}"` — the native clause shape, with the anchor's EN
   clause swapped for the JA one.

The only difference between image and caption that the row can explain is
the glyph inside the bubble; the scene is already paid for by its tags.
Training and the native eval become the same distribution.

Why EN anchor and not the JA-clause floor garble: generating with the JA
clause would make the caption literally the training caption, but the
floor draws a multi-line garbled block and the swap box is ugly; the EN
anchor gives one clean short bubble. The caption swap is one string.

## Scene prompts

Not the dataset's revised captions: they are character- and
artist-specific and carry `nsfw` / `sensitive` prefixes. Instead a
combinatorial generator in the probe — subject (1girl / 1boy / 2girls /
1boy 1girl …) × setting (classroom, bedroom, park, cafe, street, beach,
rooftop, kitchen, train, …) × pose / action × expression × style
(monochrome manga, colour, screentone, film grain, sketch) → thousands
of clean prompts. The 8 blind-pairs prompts
(`project/cjk_aware_anima/assets/unmask_eval_prompts.txt`) are **held out**
of the generator's vocabulary combinations so `native` stays a held-out
eval.

## Data mix (S0; replaces P0b's flat-only data)

| share | source | why |
|---|---|---|
| 30 % | flat-canvas singles, uniform over kana + ext + kanji + words, ext and kanji at 2× | identity exposure per row stays where P0b left it |
| 40 % | **scene composites**: singles, words, phrases inside a generated bubble | the row learns the glyph, not the canvas; small kana get a size reference inside words |
| 20 % | natural phrases on flat canvases (`--phrases`, fully covered by trained rows) | the product distribution; T5 contextualises |
| 10 % | random-order 2–4-piece strings (strings-arm recipe) | order / count signal with no language prior; keeps `flip` honest |
| 0 % | real corpus crops | two in three labels wrong (`datacheck.md`) |

Composites carry the same text distribution as the flat items (singles /
words / phrases in proportion). Small kana appear only inside words
(きゃ, ちょっと, って) and are scored there, not as singles.

## Instrument (owed, in order)

1. `--stage scenes`: prompt generator (`--scene_n`, `--scene_shapes`
   `512,640,512x640,640x512` — no 768², OOM at batch 4), anchors, generate
   through the daemon, detect + read, write `scenes/<tag>/scenes.jsonl`
   (`file, tags, box, anchor, shape`) + a sheet. Reused by every later arm.
2. `--stage data --scenes <tag> --scene_frac 0.4`: erase + draw + caption;
   composite items keep `src: "scene"`; `shape` from the scene.
3. `--phrases <file>` + `--natural_frac`, filtered by `piece_ok`;
   eval group `phrase_held` (covered phrases never trained).
4. `native` scene-kept ruler (**done**): kept ⇔ PE-Spatial
   cos(img, floor image of the same prompt/kana/seed) − cos(img, mean
   feature of 64 training canvases) ≥ 0 (`--kept_tau`), reported per cond
   as `floor cos | canvas cos | kept | hit & kept`; `--kept_ref` points a
   `--no_floor` run at an earlier run's floor images. Plain cos-to-floor
   does not work (a white bubble on black scores 0.89 against a scene).
   Without the ruler a run can "pass" by wiping scenes harder.
5. `--delta_scale` (done) and `--delta_parts f,c,g,fg` (done): magnitude
   and component axes of the trained table for `native`.
6. **Box-weighted loss** in the train stage (`--box_weight`, composite
   items only): per-item latent weight map from the record's `box`.
7. **Per-source layout vector** (`--c_flat`): the rows arm gains one
   shared vector `c_flat` added to every trained row on **flat-canvas
   items only** (`src` font / corpus / strings / phrases-flat); composite
   items train `f_r` alone. Batches become one-(shape, source) so the
   toggle is per batch (the shape batching already exists). `native` runs
   without `c_flat`; the flat-template `eval` with it. Logged every log
   step: **`leak` = mean cos(f_r, c_flat)** over trained rows (the
   canvas-in-the-rows monitor; P0b's table would read ≈ 1 on the c
   direction) and `‖c_flat‖`. Optional guard `--f_orth λ`: λ ·
   mean_r cos²(f_r, c_flat).

## Recipe of record (S0 — rows arm, from scratch)

Decision (user, 2026-09-14, after the table-parts probe): the identity
table is rebuilt **from scratch on the composite mix**, on the plain rows
arm, with the flat-canvas mode moved into a dedicated switch:

    Δ_r = f_r + 𝟏[item is flat-canvas] · c_flat        (row-norm units)

- **No encoder.** The W2d glyph CNN and its scaffolding (ψ, α, `out_scale`,
  mean-encoder, `enc_pool`, `head_init`, `init_spread`, `lr_enc`, spread /
  max-row kill rules) are dropped: held-out generalisation closed at 0,
  `g` is rank-1, and the parts probe shows it renders nothing alone. The
  rows arm was the original working arm and has never run at scale under
  the 0.8 band — S0 also answers whether the hybrid ever bought anything.
- **No warm start.** `f_r` from zero (the pack rows), `c_flat` from zero.
- **What stays, all measured:** σ band 0.7–0.9 (identity at 0.8); rows lr
  1e-3 in row-norm units, cosine decay; `μ‖f‖²` pull 1e-3 (`--free_residual`,
  the one guard against norm creep); `c_flat` at the rows lr, capped 0.75
  (projected, as `c` was). `s` (inference on/off) and ρ (mean pack-row
  norm) are not knobs.
- **Inventory** as P0b: `--kana_ext --kanji 100 --words 120`, held-out
  words 8; small kana only inside words.
- **Steps** 24 000 matched to P0b (≈ 258 renders per row; ≈ 2.4 h at
  P0b's 2.79 it/s — the rows arm has no CNN forward, expect ≥ that), batch
  4, compile, no grad-ckpt. Pool `448,512:2,448x512,512x448` (light
  shapes; the 640 family is not needed for this gate).
- **Loss**: rectified flow on the band, box-weighted on composites
  (`--box_weight 4` inside the swapped box, 1 outside).
- Arm tag `rows_synth_s24k_S0`; P0b (`encoder_wdsek_w120_s24k_p0b`) is the
  flat-only control. S0 vs P0b differs in data *and* parametrisation; the
  gates are absolute, so a pass settles both. Singles < 30/36 → one hybrid
  run on the same data (`--arm encoder`, same steps) isolates which change
  did it before anything else is touched.

Small glyphs are the one new risk: a kana in a 64–100 px bubble on a 512
canvas is 4–6 latent tokens a side, smaller than the dead 256² case. Hence
the box-weighted loss above and a minimum box of ≈ 96 px short side at
512 (`--scene_min_box`), steering bubble size from the prompt (`large
speech bubble`) rather than accepting what the base draws. Band note: the
strings arm used 0.5–0.9; S0 starts at the singles band, S1 widens only if
`flip` / `str3` sit at 0.

## Budget

| stage | cost |
|---|---|
| scenes: 1 000 images at 512–640, 28 steps | ≈ 1 h GPU once (≈ 3 s/img + detect/read); ~60 % expected to pass the box filter |
| swaps | CPU, minutes |
| train (S0, 24 k steps) | ≈ 2.4 h |
| eval + native (+ scene-kept) | ≈ 15 min |

## Gates

- **native (EN clause, 8 held-out prompts × 4 kana × 2 seeds): hit & kept ≥ 24/64**, from
  P0b's measured baseline of **2/64** (its 32/64 hits are canvas wipes on
  the margin ruler). Scene-kept alone ≥ 56/64 (the delta must stop
  overriding the scene).
- singles ≥ 30/36 (P0b 36; reader noise), `single_ext` non-small ≥ 18/28,
  `single_kanji` ≥ 22/36, word ≥ 8/32, EN 24/24 — nothing P0b holds is
  lost.
- `phrase_held` > 0/16; `flip` order statistic ≥ 24/48 if strings are in.

Also read, not gated: `leak` at the end of training (expect ≪ P0b's ≈ 1;
a leak that climbs with the flat share says the split is not doing its
job) and `--delta_parts f,c` on `native` (expect `f` alone to hit — the
whole point of the switch).

Outcomes: pass → S1 (strings, repeat mode) warm-starts from the S0 table
on this data.
Scene-kept passes, hit falls (glyph too weak inside a real bubble) →
composite share up to 60 %, box-fitted font size floor raised. Hit passes,
scene-kept fails → the swap left an erase artefact the row latched onto;
inspect `sheet_scene.png`, redo the erase (inpaint ring, not flat fill).
Singles fall below 30 → the two layouts compete in `f`; `--single_frac`
0.4 once, then decide.

## Risks named up front

- **Erase artefacts as a cue.** A flat fill inside a shaded bubble leaves a
  visible patch; the row could learn "patch = my glyph". Mitigation: ring-
  median fill, and a share of composites where the erased bubble gets the
  *anchor redrawn in a font* (EN text, EN caption) so the patch is not
  ext-row-specific.
- **Horizontal-only boxes.** The EN anchor bubble is short and horizontal;
  vertical JA layouts are absent from composites. Flat-canvas items keep
  the vertical prior; revisit if `native` with a vertical clause is needed.
- **Detector recall on generated bubbles.** Unknown; the box filter's pass
  rate is the first number the `scenes` stage prints. Below 40 % → anchor
  words / bubble tags adjusted before scaling to 1 000.
- **Tiny bubbles.** A box under ~64 px on the short side cannot hold a
  kana at readable size; filtered out (`--scene_min_box`).

## Not this plan

- Regulariser strength / `out_scale` sweeps (scale probe: direction, not
  magnitude).
- Zeroing or shrinking `c` at inference, or warm-starting any part of the
  P0b / P0a tables (table-parts probe: every part alone is 0/16).
- Inference-time guidance away from a `c`-only branch: `f + g` without `c`
  was 0/16, so that direction removes the glyph before the canvas. A
  64-render curiosity at most.
- The glyph encoder in any form (see *Recipe*); the S line is rows-only.
- Contrastive terms on text-free native images.
- Pasting onto the dataset's real images (off-manifold paste, caption
  style mismatch, nsfw/artist tags) — the self-generated scene replaces it.
