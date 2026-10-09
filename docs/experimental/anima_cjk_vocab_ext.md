# CJK vocab ext rows — how preview5's rows are made

How the Japanese rows of the [CJK vocab pack](../methods/cjk_vocab_pack.md)'s
`anima_cjk_vocab_pack_preview5` (and `_preview51` = preview5 plus the mark rows
below; the shipped `_jp_v1` is built on them — the method page's § The shipped
pack) are trained so that the
base model **renders** them as glyphs. This page covers which rows a caption
reaches (per-glyph routing and the encode fold), where the images come from
(a self-generated canvas with pasted text), which σ each item trains at (the
band law and a per-band data mix), how the loss weighs the text (in-box
share), the chain of runs that built the rows, and the one change preview5
makes on top of them (the shared direction × 0.8).

The code and records live in `project/finished/cjk_anima_scale/` (the
trainer, the data builder, the band law in `band_experiment_results.md`, the
seed's floor in `floor_score.md`), runnable by path through its
`scale.py <run> data | train | eval`. There is no `make` target. The line is
finished; cold training and the next pack are in `project/cjk_anima_reseed/`,
which carries the trainer (`reseed/trainer.py`) and its own data table.

## What preview5 is

| | |
|---|---|
| Rows | 2 683 ext rows differ from the raw pack. **1 362 are trained singles**: 174 kana and punctuation + 1 188 kanji and marks (`゙ 々 〇`), trained cold. The rest are the older seed's rows, mostly piece rows that per-glyph routing never reaches. |
| Source | `seed_retrain_0930` (the rows published as `preview4`), with each family's shared direction scaled × 0.8 (§ The shared direction) |
| Bake | `scripts/toolkits/bake_vocab_pack.py --glyph_route` on the raw pack with the `fold` map. Every untouched row is byte-identical to the raw pack. |
| Pack json | `glyph_route: true`, `fold` (§ Addressing) |
| Coverage | Every Japanese text clause in the training set's captions (615 captions) is fully covered by trained singles. On the dialogue corpus, the top 1 000 glyphs cover 99.06 % of occurrences. |
| ComfyUI | ComfyUI-Anima_lora-Adapter ≥ 3.13.0 (3.12.0 added routing, 3.13.0 the fold). Older nodes ignore both keys. |
| Hub | `sorryhyun/anima-vocab-pack-cjk` |

## What preview51 changes (2026-10-05)

preview5's rows, with six mark rows replaced: `～ … ♡ ♥ 、 。`. retrain_kana
had trained `〜` / `～` / `、` / `。` almost only as a lone glyph in a grid cell,
and the `～` row drew a green leaf in dialogue. The new rows train cold inside
words (2–6 char windows around the seed's trained letters, Manga109 dialogue
lines, synthesised heart lines), every other row frozen, on a base pack with
new encode rules: `〜` → `～`, `―` → `ー`, `，` → `、`, and a dot run → an
appended `…` row (ext 69 558, initialised at T5's `...`). Record:
`project/cjk_anima_reseed/reports/ruler_2026_10_05.md` § 5 (the leaf gone and
the page closer to the EN reference; OCR reads of short `…` strings down).

## What trains

Only the pack's ext rows train. The DiT, the LLM adapter and the T5 table stay
frozen. A run stores a delta per row (`raw · row_scale`), summed onto the
pack row at lookup, and the bake folds it into a new pack pair.

A run's vocabs start **cold**: Δ = 0 on the raw pack row. Every other row a
caption touches stays frozen at its context row. Cold is deliberate. On the
same in-word data, rows warmed from the older seed composed less than cold
rows (こんにちは ≤ 1 edit 6 vs 11 / 16) and doubled more (`dup` 13 vs 4)
(`retrain_experiments.md` § 1).

## Addressing: per-glyph routing and the fold

Anima's text path sends each caption through Qwen and through a T5-side query
table. The table's granularity decides whether a row is shared by many words
or is a whole string of its own.

| | T5 side | Qwen | Qwen / T5 |
|---|---|---|---|
| Latin | 28 438 pieces | 68 923 | **2.4** |
| JA, one ext row per Qwen token | 8 734 one-glyph, 18 288 multi-glyph "piece" rows | 27 022 | **1.0** |
| JA, per-glyph routing (shipped) | ≈ 1 500 single rows in use | 27 022 | **≈ 3** |

A JA piece row is the only address of its string, and nothing ever asked the
adapter to read it from its neighbours. **Per-glyph routing**
(`HybridT5Encoder`, `library/anima/ext_vocab.py`) sends every JA Qwen token,
on the T5 side, to its glyphs' single rows. The Qwen text is untouched, and
EN stays bit-exact. Each single row is then shared by every word it appears
in, as EN pieces are. On こんにちは (16 renders), routed beat spelled
(≤ 1 edit 14 vs 11) and the piece row (0).

The flag is `"glyph_route": true` in the pack json.
`ANIMA_VOCAB_GLYPH_ROUTE=1/0` overrides it.

**The fold** (`"fold"` in the pack json) rewrites a few characters on the T5
side before routing. The training set's OCR text writes `! ? ~` half-width,
while the word pool wrote full-width forms:

| typed | encoded as | why |
|---|---|---|
| `！ ？` | `! ?` (base T5 rows) | the house text is half-width; the base knows them |
| `~` | `～` (trained) | half-width `~` has no row |
| `『 【` / `』 】` | `「` / `」` (trained) | their own rows were never retrained |

Qwen reads the text as typed. The fold is one character to one character, so
offsets still index the typed text. It is part of the pack digest, so
changing it re-encodes every TE cache.

## Data: a self-generated canvas with pasted text

**Real images do not train rows.** Warm-training the kana rows on 291 real
kanji-free pages, with the loss box on the quoted lines, collapsed every read
(word official 16 → 0 / 104). The frozen DiT cannot reproduce a real page,
so its FM loss is high everywhere, and the rows absorb the image's whole
mismatch, not just the text's.

So the canvas is the base model's own render, and only the text is foreign
(`project/finished/cjk_anima_scale/src/scenes/stage.py`):

1. **Generate.** The base model draws a scene from a combinatorial tag
   prompt plus an EN anchor clause:
   `<tags incl. speech bubble>, english text. English text reads as "hi".`
   Every token is pretrained, and no ext row is touched.
2. **Judge.** A detector and a reader keep an image only if **exactly one**
   text box is found and the anchor is read back. The box must reach a
   minimum size, and the erase must clear it. The pools kept 13–24 % of
   renders.
3. **Composite.** The data stage erases the anchor's box, draws the JA text
   into the bubble with a font, and swaps only the quote in the caption:
   `English text reads as "hi"` becomes `Japanese text reads as "…"`, and
   the `english text` tag becomes `japanese text`.

Everything outside the box is the DiT's own output, so the loss there is
low. What is left for the rows is the glyph.

The scene pools are 384–640 px shapes at ≈ 1 k tokens: `s1` and `s1w`
(one-word anchors, four caption frames), `sl1w` (EN sentences, the source of
left-to-right items) and `ja_comic` (one bubble). Orientation is drawn, not
fitted: 30 % of multi-glyph items are left-to-right lines, marked in the
caption (`horizontal Japanese text reads as` / `, written horizontally.`);
the rest are columns, the manga default.

## σ per item: the band law and the mix

Every item is stamped with a σ band, and the trainer draws σ inside it (a
batch is split by band). The band comes from the **band law**
(`band_experiment_results.md`, rows with provenance in
`cjk_scale/windows.py`):

- **Glyph count sets the band.** Single-glyph rows train at 0.7–0.9,
  multi-glyph at 0.5–0.7.
- **Rendered px sets the floor** a band may reach: 12–16 px text lives at
  0.2–0.6, 48 px at 0.5–0.7, 128 px at 0.8.
- **Nothing above 0.9.** 0.8–0.95 is dead at 48 px, for kana and kanji
  alike.
- **Ink, stroke density and the bubble ellipse move no band.** Kanji take
  the kana band.

The singles' mix (`builder.TABLE`) has three band groups, a third of the
items each. A tier is named `<form>_<median px>`:

| band | tier | weight | what |
|---|---|---|---|
| 0.7–0.9 | `bubble1_52` | 0.5 | one glyph in a generated scene's bubble, fill 0.7 |
| | `grid_82` / `lone_190` | 0.5 | 1×1 … 3×3 grids, one glyph per cell, half in bubbles; captions name each cell (the 1×1 deals are `lone_190`) |
| 0.5–0.7 | `bubbleN_34` | 0.7 | a 2–6-glyph window of a dialogue line in a bubble, routed caption |
| | `bubble1_32` | 0.3 | one glyph at line px in a bubble it fills 0.2–0.4 of (the count tier) |
| 0.3–0.5 | `bubbleN_18` | 1.0 | windows at 12–24 px in small bubbles |

**Windows** are substrings of manga dialogue lines (plus, for the last kanji
batch, the training set's own JA text) whose glyphs are all trained singles,
with no glyph repeated and none crossing a held-out read word's trigram.
Whole lines would cover kana only; 2–6-glyph windows reach every kanji the
corpus holds. Each draw picks a glyph uniformly, then one of its windows, so
exposure is per row. Every window must encode to exactly its glyphs' single
rows, or it is dropped.

Two reads set the mix: the lone group alone composes nothing (C1), and the
in-word groups are load-bearing.

**Budget** (`cjk_scale/budget.py`): 90 steps per row, scaled by kind and
ink, × 1.5 for the in-word share:

| rows | steps / row |
|---|---|
| cold kana | 135 |
| cold kanji, ink < 10 | 225 |
| cold kanji, ink ≥ 10 | 337 |

## In-box loss

A glyph is a few percent of the canvas. Under plain MSE, a 24–32 px grid cell
is 0.3–0.8 % of the loss. `cjk_scale/loss.py` gives each item an **in-box
share** `s` and averages the in-box and out-of-box cells separately:

```
loss_item = s · mean(se over box cells) + (1 − s) · mean(se over the rest)
s(n) = s1 + (s_cap − s1) · min(1, ln n / ln n_cap)
     s1 = 0.25 (one glyph), s_cap = 0.5, n_cap = 8 glyphs
```

The pixel box maps to latent cells at 8× (`box_mask`). A grid item takes the
**union of its cells** as one box, with the share from the joined glyph
count. Each row learns from its own cell because the caption's position
clause binds it there. A flat 1×1 and any item without a box stay plain MSE.

## Trainer

| setting | value |
|---|---|
| loss | flow matching on the frozen DiT, with the in-box share above |
| lr | 1e-3, cosine, warmup 0.1 |
| batch | 4 |
| anchor μ | 0 (the rest is frozen, not anchored) |
| free residual | 1e-3 · ‖f‖² on the touched rows |
| σ | per item, in its band |

Each constant in `cjk_scale/train.py` names the read that set it. ≈ 2.35 it/s
locally at batch 4, ≈ 6.7 on a Colab G4.

## The chain that built the rows

Each run trains its vocabs cold, with every earlier row frozen at its
context's, and its windows may carry every single trained down the chain.
The last run's `trained.pt` holds the whole chain.

| run | rows | steps | context |
|---|---|---|---|
| `retrain_kana` | 174 kana + punctuation | 23 490 | the older seed |
| `retrain_kanji_b1` | 329 kanji (91 new) | 90 749 | `retrain_kana` |
| `retrain_kanji_b2` | 307 kanji | 90 319 | b1 |
| `retrain_kanji_b3` | 305 kanji | 90 321 | b2 |
| `retrain_kanji_b4` | 247: the training set's JA tail (244 kanji + `゙ 々 〇`), first trained under the fold | 76 662 | b3 |

b1–b3 are the top 1 000 glyphs of the dialogue corpus, ranked by count and
cut at equal steps. b4 adds the training set's own JA glyphs (Chinese-only
strings excluded). b4's rows are `seed_retrain_0930`, published as `preview4`.

## The shared direction (× 0.8)

The trained rows of each family (kana, kanji) share a mean delta, the
"stick", that holds ≈ 30 % of their delta energy; the per-row residuals are
near-orthogonal (`project/cjk_anima_reseed/reports/stick_2026_10_03.md`).
preview5 moves every trained row of a family by `(s − 1) · m_family` with
s = 0.8, leaving the residuals as they are. No other row changes.

On record for this lever is the stick report's sweep on the seed rows
(`sent` keys, 92 renders): ≤ 1 edit 45 → 25 at s = 0.75, with the text region
shrinking toward the EN reference's layout (box 0.149 → 0.142). No scored
read of s = 0.8 itself is on record.

## How the rows read

The seed's floor (`floor_score.md` § New seed), read routed at s = 1 (the
preview4 rows) against the older seed:

| ruler | seed rows | older seed |
|---|---|---|
| acceptance strings (はい おしい やったネ ちょっと来い こんにちは, 80) | **37** | 8 |
| kana words, 7 × 16 | **25** | 0 |
| kanji words (山田太郎 … 愛してる), ≤ 1 edit / 96 | **45** | 3 |
| lone あ / い, native (64) | 33 | 46 |
| `en` (24) | 24 | 24 |

Every acceptance string moves the same way (paired +32 / −3). Exact reads of
kanji words stay low (7 / 96), and lone あ is the one drop.

**A LoRA trained through the pack** (on `preview3`, the kana + b1 rows): two
LoRAs on one artist's 30 images, 12 with Japanese text clauses, `make lora`
defaults, differing only in the text path. By eye, the pack-trained LoRA
shows no visible degradation against the stock one. Nothing was scored.

## Open

- **Repeats.** Glyphs drawn more than once, or a word filling more slots than
  it has (`こんんにちちは`). The base decides the text region and its slot
  count; the rows fill it (`project/finished/cjk_anima_scale/proposal_seed_synthesis.md`).
  Rescaling the stick does not lower them.
- **The reseed** (`project/cjk_anima_reseed/`): cold kana rows on a flattened
  data table aimed at manga-size dialogue. Its arms so far read under the
  seed (≤ 1 edit 18 vs 34 / 64); the gap is in the per-row residuals, not the
  shared direction.
- **Token scaling.** Users render at the 1024 tier (≈ 4 k tokens); the rows
  train at 1 k.
- **An OCR reward on the rows** (`future.md`), since real pages cannot
  supervise a row through the FM loss.
