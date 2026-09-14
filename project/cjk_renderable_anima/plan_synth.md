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

1. **Scene generation** (GPU, base model, delta off): a dataset-format
   caption — `{rating}, {count}, {character}, {copyright}, @{artist},
   {generals sorted}. English text reads as "{anchor}"` with `speech
   bubble` and `english text` among the generals and a short EN anchor
   (hi ok no yes wow hey oh huh yay wait — Latin, every piece pretrained,
   no ext row touched). The output is in-domain by construction, and the
   caption explains everything in the image.
2. **Bubble filter** (same pass): the `Readers` detector boxes the *text*;
   every box that reads the anchor is an anchor bubble (the base draws one
   per speaker — all are swapped), any other non-speck box is stray text
   and rejects the image. The bubble itself comes from a flood fill of the
   fill colour around the text box; the **usable region** is its inscribed
   rectangle. Record `(file, head, generals, anchor, boxes, regions,
   bubbles)`.
3. **Swap** (CPU, `--stage data`): fill each region with the bubble's local
   colour (median of a ring around the text box), draw the kana / word /
   phrase fitted into the region with the existing font renderer —
   **vertical when the region is taller than wide** (the base draws tall
   manga bubbles). One scene serves several swaps.
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

A combinatorial generator (`stage/scenes.py`) that writes prompts **in the
dataset's caption format** — the first smoke (generic tag bags, no artist,
no character, unsorted) drew a generic average that was off-distribution
for the base (user, 2026-09-14): `rating` (safe / sensitive) → `count`
(1girl ×5, 1boy ×3, 2girls, 1boy 1girl, 2boys — two speakers draw two
bubbles and far more stray text) → character + copyright (a dataset pair
for 30 % of 1girl prompts, else `original`) → `@artist` (80 %: the
dataset's 83 `@name` rows plus `sincos` / `hews` at 4× weight) → generals
**sorted alphabetically**: 1–2 appearance tags, a setting (library,
rooftop, kitchen, train interior, beach, city street, shrine, …), an
action, an expression, an optional style (screentone, monochrome, sketch,
anime coloring, flat color, watercolor, lineart, halftone), `solo` for
single counts, `looking at viewer` 50 %, a framing (cowboy shot / full
body / from side / none), `speech bubble`, `english text`. Negative prompt
on the uncond branch only (`--scene_negative`; never enters a caption):
`worst quality, lowres, old, bad hands, bad anatomy, sepia, blurry,
glitch, jpeg artifacts`. No positive quality tags — the training caption
mirrors the prompt and the dataset captions carry none. The 8 blind-pairs
prompts (`project/cjk_aware_anima/assets/unmask_eval_prompts.txt`) are
**held out**: none of their setting / action / style / framing tokens
(classroom, bedroom, park bench, cafe interior, windowsill, film grain,
maid, holding a tray, upper body, portrait, simple background, 2koma,
comic, surprised, greyscale) is in the vocabulary, so `native` stays a
held-out eval. Rendering at 2× and downsampling changes nothing (8 % vs
4 % on 24 prompts, same reject profile) and is off.

## Data mix (S0; replaces P0b's flat-only data)

| share | source | why |
|---|---|---|
| 40 % | flat-canvas singles, uniform over kana + ext (non-small) + kanji + words, ext and kanji at 2× | identity exposure per row stays where P0b left it |
| 40 % | **scene composites**: singles and phrases inside a generated bubble | the row learns the glyph, not the canvas; small kana get a size reference inside phrases |
| 20 % | natural phrases on flat canvases (`--natural_frac`, fully covered by trained rows) | the product distribution; T5 contextualises |
| 0 % | random-order 2–4-piece strings | **out for S0** (user, 2026-09-14): order/count is S1's question; `--strings_frac` keeps the lever (adds `flip` / `str3`) |
| 0 % | real corpus crops | two in three labels wrong (`datacheck.md`) |

Built as `data_synth_s0` (2026-09-14 23:49): 16 000 items = 6 400 font +
3 200 phrase + 6 400 scene (5 549 single / 851 phrase — the median region
holds two glyphs at 32 px, so composite phrases skew short) over the
**186** kept scenes (≈ 34 swaps each; scenes re-judged with the
containment rule, 203 → 186). Composites carry the same units as the flat
items, so each of the ~300 rows meets ≈ 20 distinct scenes.

Composites carry the same text distribution as the flat items (singles /
words / phrases in proportion). Small kana appear only inside words
(きゃ, ちょっと, って) and are scored there, not as singles.

## Instrument (owed, in order)

1. `--stage scenes` (**done**, `stage/scenes.py`): prompt generator
   (`--scene_n`, `--scene_shapes` = the S0 pool), `prompts.jsonl` written
   before the first render, batched text-encoder → DiT → VAE per shape
   (`--scene_batch 4`, per-item seeds as one noise tensor each), detector +
   both readers, bubble flood fill (ring-median fill colour, seeds only on
   matching ring pixels, ink dilated 2 px so sketchy outlines close, each
   seed's fill judged alone — a large border-touching fill is a leak, a
   small one an edge-clipped bubble — largest survivor wins, 12× text-box
   plausibility guard), `--scene_min_box 56` on the region's short side,
   `scenes_<tag>/{scenes.jsonl, scenes_all.jsonl, report.md, sheet_kept.png,
   sheet_rejected.png}`. `--scene_rejudge 1` re-applies the filter to an
   existing run from its stored reads on CPU. Reused by every later arm.
2. `--stage data --scenes <tag> --scene_frac 0.4` (**done**, `stage/synth.py`
   + `wake/render.py::render_into_scene` + `wake/bubble.py`): erase + draw +
   caption; composite items keep `src: "scene"`, `shape` from the scene,
   `box` = the drawn text box, `kind` ∈ single / phrase. The erase paints
   the usable region ∪ the text box padded by a quarter, **only inside the
   bubble's flood interior** (letter holes filled) — a rectangle's corners
   poked past round outlines (user, 2026-09-14). Text fitted into the
   region, vertical when taller than wide, per-glyph cell ≥
   `--scene_min_glyph` (32 px: at 40 the mix fell to 82 % singles because
   the median region holds two glyphs); kinds are drawn against the
   region's capacity, singles when the kind never fits. `--scene_stroke`
   0.25 of composites get a thin outline in the fill colour. Fonts: Noto
   CJK only (index 0 = JP); DroidSansFallback drew Chinese-styled kanji
   and is out of every S-line render.
3. `--natural_frac` (**done**, no file needed): covered training-corpus
   lines (296, 276 distinct) in a font on flat canvases; eval group
   `phrase_held` = 16 covered held-out lines whose text never appears in a
   training item (45 available).
4. `native` scene-kept ruler (**done**): kept ⇔ PE-Spatial
   cos(img, floor image of the same prompt/kana/seed) − cos(img, mean
   feature of 64 training canvases) ≥ 0 (`--kept_tau`), reported per cond
   as `floor cos | canvas cos | kept | hit & kept`; `--kept_ref` points a
   `--no_floor` run at an earlier run's floor images. Plain cos-to-floor
   does not work (a white bubble on black scores 0.89 against a scene).
   Without the ruler a run can "pass" by wiping scenes harder.
5. `--delta_scale` (done) and `--delta_parts f,c,g,fg` (done): magnitude
   and component axes of the trained table for `native`.
6. **Box-weighted loss** (**done**, `--box_weight`, composite batches
   only): per-item latent weight map from the record's `box` (VAE 8×),
   normalised by the weight sum so the loss scale matches plain MSE.
7. **Per-source layout vector** (**done**, `--c_flat 1`; `ExtDelta.common`
   set per batch, batches are one *(shape, source)*): the rows arm gains one
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
- Arm dir `rows_synth_s0_s24k_S0`; P0b (`encoder_wdsek_w120_s24k_p0b`) is the
  flat-only control. Jobs `20260914-235607-0ac29b` (train + eval 512²,
  `--with_c_flat 1`, no floor) and `20260914-235621-7e903c` (native: 8
  prompts × あかすぐ × 2 seeds, EN clause, floor + `full,c,fc`; `full` *is*
  `f` on the rows arm). S0 vs P0b differs in data *and* parametrisation; the
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
| scenes: 1 000 images on the S0 pool, 28 steps, batch 4 | **measured** 62 min (≈ 3.6 s/img + 5 min detect/read); **203 kept (20 %)** — the prompt list is a stable prefix, so `--scene_n 2000` on the same tag renders only the missing 1 000 |
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
- **Bubble shapes.** Measured, not a risk: the base draws round *and*
  tall manga bubbles, so composites carry both horizontal and vertical
  layouts (the data stage picks vertical for regions taller than wide).
- **Filter yield — measured 20 % on s0** (1 000 images): stray text 29 %
  (shirts, signs, a second garbled bubble), anchor misread or drawn as JA
  garble 24 %, region under 56 px 17 %, fill leak 8 %. The stray-text and
  read rules are strict on purpose; the size and leak rules were tuned on
  the sheets (user-picked rejects 977 461 704 822 667 585 all pass now).
  Each scene serves several swaps; 2 000 scenes ≈ 400 kept is the target.
- **Small bubbles are the product.** The base draws the bubble at ≈ 1/8 of
  the canvas whatever the framing (region short side median ≈ 65 px at
  512², 57 px for `full body`): a 96 px bar kept 4 %. The bar is **56 px**
  — the inscribed square of a round 76 px bubble is 55 px, and a tall
  65 × 119 region holds one or two kana vertically. Glyphs of 4 latent
  tokens a side are what the box-weighted loss is for; if S0's singles
  gate fails, raise the bar before touching the loss.

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
