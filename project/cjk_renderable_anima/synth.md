# synth — the S line as built

> What the S line is, how its instrument works, and **the recipe as it is
> actually run** (the 2026-09-19/20 `step1_0920` and `step2_0919` argv).
> Chronology — runs, numbers, decisions in order — is in
> [`reports/`](reports/README.md); the forward plan is
> [`plan.md`](plan.md); the settled verdicts are
> [`findings.md`](findings.md). Where a later run changed a number, the
> dated report wins. Superseded design notes (the S0-era mix, `c_flat`, the
> ΔFM plan as proposed) are archived under
> `_archive/cjk_renderable_anima/plan_synth*.md`.

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
かかか, く for ぐ).

So the row is an address for the whole training image, not for the
glyph: every item was "glyph on a flat ground", the ground is perfectly
correlated with the glyph, and no loss term can tell the two apart. The
shape pool changed canvas *size*, not canvas *content*, so it could not
remove this.

**Scale probe (`--delta_scale`, trained cond only, 64 images each):** at
0.5 the scene comes back and the kana goes — 2/64; 0.7 gives 22/64.
Magnitude is not the lever; a stronger `‖f‖` pull or a smaller `out_scale`
would only slide along the same axis.

**Table-parts probe (2026-09-14 night, `native_parts/`, 64 renders;
`reports/wake_canvas_scenes_2026_09_14.md`):** the saved table splits
exactly into `g + c + f`. Rendered one part at a time, **every part is 0/16
hits**: `f`, `g`, `f+g` keep the scene and draw floor garble; `c` alone
draws the training canvas (big white bubble, a generic ん) and no requested
kana. The canvas mode is `c`, but the identity in `f`/`g` only renders
*given* that mode — the rows learned "glyph | flat canvas", and no linear
split separates the two. Consequences: zeroing `c` at inference is not a
fix, and a warm start from P0b/P0a carries a conditional identity the
composites would have to re-learn. On the scene-kept ruler, P0b's honest
baseline is **hit & kept 2/64**.

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

**This premise is the line's load-bearing constraint.** Any build where the
page carries a systematic residual the rows did not cause — a downscaled
panel grid, an off-manifold paste, a small glyph in a bubble the base drew
for a large one — hands that residual to the only free parameters in the
run. Two such builds are named and rejected in `plan.md` S1b.

## Scene prompts

A combinatorial generator (`src/scenes/stage.py`) that writes prompts **in
the dataset's caption format** — the first smoke (generic tag bags, no
artist, no character, unsorted) drew a generic average that was
off-distribution for the base: `rating` (safe / sensitive) → `count`
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
mirrors the prompt and the dataset captions carry none.

The 8 blind-pairs prompts
(`project/cjk_aware_anima/assets/unmask_eval_prompts.txt`) are **held
out**: none of their setting / action / style / framing tokens (classroom,
bedroom, park bench, cafe interior, windowsill, film grain, maid, holding a
tray, upper body, portrait, simple background, 2koma, comic, surprised,
greyscale) is in the vocabulary — except `comic`, which the `ja_comic` pool
(`--scene_extra_tags comic`, 2026-09-17) puts in every caption. **From any
run whose data carries `ja_comic`, native is held out on prompts 1–7 only**;
read prompt 8 separately and never pool it into a gate count. Rendering at
2× and downsampling changes nothing (8 % vs 4 % on 24 prompts, same reject
profile) and is off.

## Scene pools

Pools after the Δ0.9 rejudge and the `s1w` top-up: **s1 233, s1w 380,
sl1w 213, ja_comic 292**, one-glyph pool 586.

`scenes_ja_comic` (2026-09-17): asking the base for *Japanese* text is the
only measured lever that gives tall bubbles. 2 400 generated, **359 kept
(15 %)**, open_bubble 75 % of the rejects; tall (AR ≥ 1.0) 248, AR ≥ 1.3
221, tall height median 135 px. `ja_bubble_reads` keeps 24 %,
`ja_reads_as` 9 %, `ja_saying` 11 %. `comic` alone still pulls most scenes
to greyscale / lineart — the kept sheet carries a colour minority, not a
colour pool. Its one-column caps over `sl1w,ja_comic` at 28 px: short 6
glyphs, sentences one column 65/160.

**No pool has a small bubble.** Region short side p05 / p50 / p95 is
57 / 76 / 145 px on `s1`, 57 / 76 / 133 on `s1w`, 59 / 83 / 133 on `sl1w`,
42 / 62 / 99 on `ja_comic` — **no region under 40 px in 1 118 scenes**.
That is the constraint `plan.md` S1b.1 is built against.

Before any new pool enters a build (the sl1w lesson): read the most-reused
scenes on a 400-item CPU smoke and add bubble-less / text-outside-a-bubble
regions to `--scene_drop`.

**Placement** — text on a subtitle bar or on the scene (as the EN refs
place "hi") is scene-stage work; `s1sfx` (banner) is the pool that has it
and is in no arm yet.

## Instrument (as built, in build order)

1. `--stage scenes` (`src/scenes/stage.py`): prompt generator (`--scene_n`,
   `--scene_shapes`), `prompts.jsonl` written before the first render,
   batched text-encoder → DiT → VAE per shape (`--scene_batch 4`, per-item
   seeds as one noise tensor each), detector + both readers, bubble flood
   fill (ring-median fill colour, seeds only on matching ring pixels, ink
   dilated 2 px so sketchy outlines close, each seed's fill judged alone —
   a large border-touching fill is a leak, a small one an edge-clipped
   bubble — largest survivor wins, 12× text-box plausibility guard),
   `--scene_min_box 56` on the region's short side, `scenes_<tag>/{scenes.jsonl,
   scenes_all.jsonl, report.md, sheet_kept.png, sheet_rejected.png}`.
   `--scene_rejudge 1` re-applies the filter to an existing run from its
   stored reads on CPU. The prompt list is a stable prefix, so
   `--scene_n 2000` on the same tag renders only the missing 1 000.
2. `--stage data --scenes <tag>` (`src/data/synth.py` +
   `src/common/render/scene.py::render_into_scene` + `src/common/bubble.py`):
   erase + draw + caption; composite items keep `src: "scene"`, `shape`
   from the scene, `box` = the drawn text box, `kind` ∈ single / short /
   sentence. The erase paints the usable region ∪ the text box padded by a
   quarter, **only inside the bubble's flood interior** (letter holes
   filled) — a rectangle's corners poked past round outlines. Text fitted
   into the region, vertical when taller than wide, per-glyph cell ≥
   `--scene_min_glyph`; kinds are drawn against the region's capacity,
   singles when the kind never fits. `--scene_stroke` 0.25 of composites get
   a thin outline in the fill colour. Fonts: Noto CJK only (index 0 = JP);
   DroidSansFallback drew Chinese-styled kanji and is out of every S-line
   render.
3. `--natural_frac` (no file needed): covered training-corpus lines in a
   font on flat canvases; eval group `phrase_held` = covered held-out lines
   whose text never appears in a training item. **Flat share is 0 in every
   arm since 2026-09-17** — flat items pull the render toward their own look
   (a large bold glyph, plain canvas or black box, the scene thinned; box
   IoU 0.47 → 0.39).
4. `native` scene-kept ruler: kept ⇔ PE-Spatial cos(img, floor image of the
   same prompt/kana/seed) − cos(img, mean feature of 64 training canvases)
   ≥ 0 (`--kept_tau`), reported per cond as `floor cos | canvas cos | kept |
   hit & kept`; `--kept_ref` points a `--no_floor` run at an earlier run's
   floor images. Plain cos-to-floor does not work (a white bubble on black
   scores 0.89 against a scene). Superseded as the gate by `en cos` /
   `box IoU` (*Rulers* below) but still reported.
5. `--delta_scale` and `--delta_parts f,c,g,fg`: magnitude and component
   axes of the trained table for `native`.
6. **Box-weighted loss** (`--box_weight`, composite batches only): per-item
   latent weight map from the record's `box` (VAE 8×), normalised by the
   weight sum so the loss scale matches plain MSE. **Superseded by
   `--box_share`** (item 12).
7. **Per-source layout vector** (`--c_flat`): one shared vector added to
   every trained row on flat-canvas items only. **Out of the recipe since
   S0b** — raising its cap moved the render *trigger* out of `f` into `c`
   and every flat ruler got worse (`findings.md` *trigger vs canvas*).
   Every arm since runs `--c_flat 0`.
8. **`anchor_residual` / `erase_miss`** (`src/common/render/scene.py`,
   `src/scenes/judge.py`): the erase geometry is `erase_paint`; the judge
   measures the share of the anchor's ink the paint would leave and rejects
   above `--scene_max_residual` (0.33 since Δ0.9). Without it the anchor
   stays under the kana — ≈ 400 S0 composites trained on "hey" + ぐ.
9. **`--scene_fill`** (0.7 in every current arm): fraction of the usable
   region the text block fills. 0.9 filled the bubble edge to edge; 0.7
   leaves manga-like air and scales `region_capacity` with it.
10. **`--flat_bubble`** (1.0 in every current arm): share of flat items
    drawn inside the font bubble.
11. **Δ0.9 judge and routing rules** (2026-09-17 night): specks are recorded
    and erased with the anchor's erase (`--scene_rejudge` re-applies on
    CPU); `bubble_leak` (`--scene_max_offset 1.0`) and `--scene_open_lost
    0.02`; per-kind scene routing in the data stage (`--single_scenes` +
    `--single_max_ar` for one-glyph texts, `--scene_one_bubble`, multi-glyph
    texts routed by fit); `fit_text`'s shrink step fixed (it had sat after a
    `return` since `c434a398`).
12. **Area-independent in-box loss** (`--box_share ρ_g`, 2026-09-19,
    `weighted_fm_loss` in `src/train/stage.py`) — **the current loss
    shape.** Per item `s·mean_in + (1 − s)·mean_out` with
    `s = min(ρ_g · n_glyphs, 0.75)`, batch mean. Per *glyph*, not per box:
    the old loss already kept a row's share flat in glyph count (box area ∝
    count), so a per-box ρ would cut a sentence row to ρ / n.

    Why it exists: `--box_weight` divided by the weight sum over the whole
    canvas, so the row's share of the loss was proportional to the box
    **area**. At `d0`'s ≈ 64-cell box, `--box_weight` 4 = 6.83 % in-box
    share; a jittered small glyph cut it to 2.87 %, as far as dropping the
    weight to 1 does — and the size arms that read 0/24 were measuring the
    share, not the size. `ρ_g 0.25` = `--box_weight` 20 at `d0`'s box.
    Every box-weight and row-norm number recorded before 2026-09-19 is a
    number at that box, and `--free_residual` μ 1e-3 is calibrated to it too.

    The end row norm is where the in-box gradient balances μ, not where
    travel stops: in every arm the norm peaks by step ≈ 400–700 and then
    falls under the cosine schedule, and a smaller box lowers the balance
    point. More steps on the same recipe cannot recover it.

## The paired loss (ΔFM) — as built

`--pair_loss 1 --pair_ref en`. A flag, not the default: it holds the scene
and does not deliver the glyph (`findings.md`), and which loss the seed
table takes at full inventory is open (`plan.md` S1a).

**What it is.** Per composite item B (scene + new glyph, caption `c_B` with
the ext row) the data stage also holds its **sibling A**: the same scene,
same erase, same font / size / position / colour / tilt, a Latin reference
string of the same glyph count in the box, caption `c_A` = the same frame
with only the quoted string swapped. One noise `ε`, one `σ` for the pair:

    L_Δ = ‖ (v_θ(z_σ^B, c_B) − sg v_θ(z_σ^A, c_A)) − (v_B* − v_A*) ‖²_w
        = ‖ r_B − r_A ‖²_w,   r_X = v_θ(z_σ^X, c_X) − v_X*

— plain FM with the sibling's residual subtracted as a control variate. In
the target the noise cancels exactly (`v_B* − v_A* = x_A − x_B`, zero
outside the box); outside the box the term is `‖v_θ(B) − v_θ(A)‖²`, a
scene-preservation term against the wipe; inside the box the shared part of
the residual cancels — the posterior spread of the scene given `z_σ`, and
(because A is a paste by the same renderer) the erase patch, ring-median
fill and font rasterisation both items carry. Measured: ≈ 80 % of the
residual cancels on singles, 86 % on sentences.

**The reference is EN, pasted by our renderer, under the JA frame.** Rules
that matter: same frame in both branches (a frame mismatch adds noise
instead of cancelling it); never the target's romaji letters; one Latin
letter per JA glyph at the same `fs` and column positions, the weighted box
being the **union** of the two drawn boxes; `--pair_ref_pool 4` reference
strings per (scene, glyph count) so reference captions stay ≈ 1 k unique
(the text cache is ≈ 1.3 MB per caption in RAM).

**No warm start is required by the loss** — the one condition is that the
base can already draw the reference, and pasted Latin meets it with no
table, so ΔFM trains new rows from random init.

**The leak to watch**: the target becomes `v_B* + r_A`, so a systematic
error of the base on the reference inside the box is inherited. `ref_bias`
= in-box ‖mean_batch r_A‖ / mean_batch ‖r_A‖ (EMA over 100 steps) is the
free diagnostic; measured 0.26–0.28 and flat, with no Latin-styled strokes
on the hits — the systematic part is the co-text, not the reference.

**As built** (commit `2ffacc78`): `src/data/pair.py`,
`common/render/scene.py` (`ref_text=`, one fit, two draws), `data/synth.py`
(`--pair_ref none|en`, `--pair_ref_pool`, `ref_file` / `ref_text` /
`ref_caption` / `ref_box`, union `box`, `sheet_scene_pair.png`),
`train/stage.py` (`--pair_loss`, `--pair_sigma_min`, `pair_branch` under
`no_grad` with the batch's own `ε` and `σ`, `pair_stats` → `fm_plain` /
`pres` / `ref_bias`). 1.63 it/s against 2.35 plain (0.69×); the second
compiled graph cost nothing. **Plain-FM controls run on the same data dir**
— `--pair_loss 0` ignores the siblings, so no separate build is needed.
`--pair_ref_frame ja|en` rewrites the sibling captions under the scene's EN
frame at train time (measured flat, kept as a flag); `--pair_ref kana` is
parked and not written. Paired `short` / `sentence` siblings are built —
every kind goes through one composite loop and `render_into_scene` takes
the item's own line lengths for the sibling (`ref_lines`), asserting
pixel-identity outside the union box.

## The recipe as run

Both steps train the **rows** arm on a frozen DiT / adapter / text encoder:
`Δ_r = f_r` in row-norm units, no encoder, no `c_flat`, no flat items.
Both ran on the **raw** pack (`ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`,
log sha `7b9fce0bb57b`).

### Step 1 — the seed table (`step1_0920`, 2026-09-20)

Single-glyph composites only (`--scene_mix single=1.0`), one glyph per
bubble, 374 rows ≈ 568 draws/row. Job `20260920-003014-320a05`, arm
`rows_step1_0920_s53k`; data `data_step1_0920` is a symlink to
`data_step1_0919`, so `--data_tag step1_0920 --arm_tag s53k` resolves
against the same build.

```bash
ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \
make daemon-run ARGS="--label step1 --stall-timeout 0 --queue \
    project/cjk_renderable_anima/src/wake_probe.py --stage data train eval --arm rows \
    --scenes s1,s1w,sl1w,ja_comic --scene_one_bubble ja_comic \
    --single_scenes s1,s1w --single_max_ar 2 --data_tag step1_0920 \
    --units kana --units kana_ext*1 --units small --units kanji:200*1 \
    --units 'list:、,。,・,ー,～,〜,！,？,「,」,！！,・・・,・・・・*1' \
    --scene_mix single=1.0 --n_items 10000 --scene_frac 1.0 \
    --natural_frac 0 --strings_frac 0 --flat_bubble 1.0 \
    --scene_fill 0.7 --scene_min_glyph 28 --scene_max_lines 1 --scene_vertical 1 \
    --pair_ref en --pair_ref_pool 4 \
    --shapes 448,512:2,448x512,512x448 \
    --train_steps 53000 --batch 4 --t_min 0.7 --t_max 0.9 \
    --compile 1 --grad_ckpt 0 --aggressive_recompute 0 \
    --lr_rows 2e-3 --lr_decay cosine --free_residual 1e-3 \
    --box_share 0.25 --pair_loss 1 \
    --seeds 2 --no_floor --c_flat 0 --arm_tag s53k"
```

What each block is doing:

- **σ 0.7–0.9** — the singles band; identity is decided at σ ≈ 0.8.
- **`--units small`** (`src/data/inventory.py::small_digraphs`) draws each of
  the 18 small kana inside up to 6 two-glyph digraphs whose Qwen pieces are
  the host row + the small row (あっ きゃ しょ ニャ トゥ; corpus-attested
  first, then the yōon / gairaigo tables), each small kana at the pool mass
  of one weight-1 unit; eval group `single_small`. A lone ゃ renders
  full-size, so small kana are not a singles concept — the digraph is the
  only way to give them a size reference at step 1. The frequent uses
  (って ちゃ った じゃ ック ティ) are single Qwen pieces, i.e. word rows, and
  belong to the loop's vocab step.
- **`--box_share 0.25`** — the current in-box weighting (instrument 12); the
  only variable against `step1_0919`, and it about doubled every ruler.
- **`--pair_ref en --pair_loss 1`** — ΔFM. A plain control needs no new
  build: `--pair_loss 0 --lr_rows 1e-3` on the same data dir.
- **uniform `*1` weights** — no extra draws for kanji.

The native read is a second job on the same arm:

```bash
make daemon-run ARGS="--label step1-native --stall-timeout 0 --queue \
    project/cjk_renderable_anima/src/wake_probe.py --stage native --arm rows \
    --data_tag step1_0920 --arm_tag s53k --native_chars あ,か,す,日 \
    --native_clauses en,swap --seeds 2 --delta_parts full"
```

### Step 2 — the sentence pass (`step2_0919`, 2026-09-19)

The same table warm-started and trained on multi-glyph composites, **plain
FM** (`--pair_loss 0`; ΔFM lost that A/B — `findings.md`). Two jobs: data
`20260919-202821-2392df`, train+eval `20260919-203858-051f61`, arm
`rows_step2_0919_plain_6k`.

```bash
# data — same pools and units as step 1, plus the phrase file and the kind mix
ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \
make daemon-run ARGS="--label step2-data --stall-timeout 0 --queue \
    project/cjk_renderable_anima/src/wake_probe.py --stage data --arm rows \
    --scenes s1,s1w,sl1w,ja_comic --scene_one_bubble ja_comic \
    --single_scenes s1,s1w --single_max_ar 2 --data_tag step2_0919 \
    --units kana --units kana_ext*1 --units small --units kanji:200*1 \
    --units 'list:、,。,・,ー,～,〜,！,？,「,」,！！,・・・,・・・・*1' \
    --phrase_file <manga109s>/derived/dialogue_2_10.tsv \
    --phrase_min_pieces 2 --n_phrase_eval 8 \
    --scene_mix single=0.1,short=0.5,sentence=0.4 \
    --short_pieces 2-5 --short_max_lines 1 \
    --sentence_min_letters 6 --sentence_min_glyph 20 --sentence_fill 0.9 \
    --scene_vertical 1 --n_items 10000 --scene_frac 1.0 \
    --natural_frac 0 --strings_frac 0 --flat_bubble 1.0 \
    --scene_fill 0.7 --scene_min_glyph 28 --scene_max_lines 2 --shapes 512"

# train — one warm source, the step-1 table; no cold row inside a sentence pass
ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \
make daemon-run ARGS="--label step2 --stall-timeout 0 --queue \
    project/cjk_renderable_anima/src/wake_probe.py --stage train eval --arm rows \
    --data_tag step2_0919 --shapes 512 \
    --init_rows output/wake_probe/rows_step1_0919_s53k/trained.pt \
    --init_anchor 0.3 --lr_warmup 500 \
    --train_steps 6000 --batch 4 --t_min 0.5 --t_max 0.9 \
    --compile 1 --grad_ckpt 0 --aggressive_recompute 0 \
    --lr_rows 1e-3 --lr_decay cosine --free_residual 1e-3 \
    --box_weight 4 --c_flat 0 --pair_loss 0 \
    --eval_groups single,single_ext,single_small,single_kanji,short,short_held,phrase,phrase_held,en \
    --seeds 2 --no_floor --arm_tag plain_6k"
```

What each block is doing:

- **σ 0.5–0.9, not the singles 0.7–0.9** — 90 % of the items are multi-piece
  and order / count are decided at σ 0.5–0.8 (strings σ diagnostic). The
  band is one global flag, so a run mixing kinds takes the lower edge.
- **`--init_anchor 0.3 --lr_warmup 500`** — without them the warm start is
  gone by step 50 (Adam at lr 1e-3 in row-norm units; `sent_s24k` ended at
  cos 0.10 to its source, norm 98 → 47 → regrown). The anchor's gradient is
  0 at `f = f₀`, so the warmup is what saves the first steps; `warm_cos` /
  `warm_drift` in `train_log.json` are the read. μ = 0.3 is checked: μ 0.1
  on the same data and seed lost lift (+0.086 vs +0.131, native はい 1 vs
  5/8).
- **One warm source, every row warm.** `--units small` in step 1 is what
  makes this possible — S2b needed `--init_rows <sent>,<Δ1>` for its
  small-kana rows and carried their delta twice as a result.
- **6 k steps, not 24 k** — sized to the pool, not to a step budget. This
  build has 4 564 covered training lines → 480 sentences (≥ 6 letters),
  470 shorts (2–5 pieces), 3 614 in no kind, over 820 of 1 004 scenes:
  ≈ 2.4 epochs. 24 k would show each line 80–100 times.
- **`--box_weight 4`, not `--box_share`** — step 2 ran before the share fix
  reached the sentence step. On this build a sentence box is 1.9 % of the
  canvas, so w 4 is a mean in-box share of 0.07. Round 2's value is open:
  the 2026-09-20 smoke (`reports/step2_0920_box_weight_smoke_2026_09_20.md`)
  found no sentence ruler ordered by the weight, and `--box_share 0.25` caps
  77 % of these items at 0.75 (`plan.md` *Step 2*).
- **`--n_phrase_eval 8`** — flat sentence eval groups are small on purpose;
  `single` and `native` are the rulers, and flat sentences are
  off-distribution for a composite-only arm.
- **`<manga109s>`** is the local Manga109-s derivation; the path stays out
  of the repo (`dialogue_2_10.tsv` = `dialogue_3_10.tsv` + its 2-piece
  lines, `<manga109s>/derived/make_dialogue_2_10.py`).

## Rulers

- **Scene**: **`en cos`** — PE-Spatial cos to the EN reference render of the
  same prompt/seed (`English text reads as "hi"`); floor ≈ 0.93–0.97 is the
  ceiling. **Placement**: **`box IoU`**, the glyph box vs the "hi" box
  (floor 0.36–0.51; every trained cond so far 0.05–0.51 — harsh on small
  boxes, read with the sheets).
- **Native runs both clauses** — `en` (the trained JA frame) and `swap`
  (the EN ref's caption with the word swapped). A row counts as frame-
  independent only when it hits under `swap`, and that is per glyph, not
  per arm (か vs あ). `stage native` renders trained conds only;
  `--native_floor 1` restores the floor renders (needed on katakana
  dakuten, which the pack renders untrained at 10–15/16).
- **Multi-glyph groups are scored by `src/probe/sub_exact.py` pooled lift**,
  not exact match — exact is floor-saturated at 0/32 on every sentence arm
  to date. Lift = glyph recall − the same recall of the group's other refs
  against the same read. Two arm dirs print a bootstrap CI on the
  difference. No GPU: it reads the `eval_reads.json` the eval stage already
  wrote.
- Also read, not gated: end-of-training row norm, per-char hits under
  `swap`, and the `native_swap` sheets before trusting a box IoU on a small
  box.

## Budget (measured)

| stage | cost |
|---|---|
| scenes: 1 000 images on the S0 pool, 28 steps, batch 4 | 62 min (≈ 3.6 s/img + 5 min detect/read); **≈ 20 % kept** |
| swaps (`--stage data`, 10 k items) | CPU, minutes |
| train, 53 k steps batch 4 (step 1, ΔFM) | ≈ 9 h at 1.63 it/s (plain: 2.35 it/s) |
| train, 6 k steps batch 4 (step 2) | ≈ 45 min |
| eval + native | ≈ 15 min each |

Rules of thumb that price a run:

- **≈ 1 000 draws per row.** The exposure curve reads 1 330 / 670 / 490
  draws → 100 / 75 / 36 % of singles. `step1_0920` sits at 568.
- **Pin the share of every kind that must be learned.** A weighted draw is
  not a quota — the 53 k run's word rows got ≈ 8 items each and stayed at
  norm 0.12. Check the items-per-row histogram in the data log before
  training.
- **≈ 10 000 items per data build.** The probe keeps every caption's text
  embedding in RAM at ≈ 1.3 MB each; 30 k items = 33 GB before latents and
  the DiT load, on a 46 GB box. ΔFM adds reference latents (+2.6 GB at 10 k
  items fp32, half in bf16).

## Risks the build carries

- **Small glyphs.** A kana in a 64–100 px bubble at 512 is 4–6 latent
  tokens a side. `--scene_min_glyph` (28 since the sentence line) and the
  in-box share are the mitigation; the open question is `plan.md` S1b.
- **Erase artefacts as a cue.** Ring-median fill inside a shaded bubble can
  leave a patch the row latches onto. `erase_miss` catches the wrong-blob
  case, not the patch; the sheets are the check. (Under ΔFM the patch is
  shared with the sibling and cancels.)
- **The residual wipe is the delta, not the mix.** Seed 0 wipes 4/8 prompts
  at flat 10 % and at flat 0 alike; every plain arm drives the row norm to
  ≈ 125 and that is what overrides the scene. The wipes are seed-shaped, so
  per-char n = 16 hides differences under ≈ 8.
- **The bubble is a canvas.** Every composite puts the glyph inside a round
  white bubble; with flat items at 0 the rows can learn the bubble as their
  unit. Placement diversity in the scene stage (subtitle bar / on-scene
  text) is the lever, not more bubbles.
- **Bubble capacity for sentences.** Wrapping is in (up to 3 columns, min
  glyph 28, vertical first); the fit is bounded by bubble *height* — the
  tall pool (`--scene_tall_ar 1.0`) is the lever, JA-frame scenes the
  fallback.
