# plan_synth — the S line, live plan (rows on self-generated scene composites)

> What is still to run. Status, results and the next-steps record through
> the 2026-09-16 evening sentence-arm launch moved to
> [`reports/synth_sentence_launch_2026_09_16.md`](reports/synth_sentence_launch_2026_09_16.md);
> settled verdicts are [`findings.md`](findings.md) /
> [`findings_seed.md`](findings_seed.md); how the S line is built is
> [`synth.md`](synth.md). The P-line record (`plan.md`) stays the flat-only
> control; target artefact, kill criteria and P2–P4 in `plan.md` stand,
> re-based on the S table.

## Starting point for the next arms

Recipe of record (settled; tables and argv in the report): rows only,
`Δ_r = f_r`, no `c_flat`, no Q; frame-mix scenes, composite 0.9 / flat
singles 0.1; σ 0.7–0.9 **for singles** — multi-piece items (short /
sentence) take 0.5–0.9, the strings-arm band (order and count live at
σ 0.5–0.8, absent at ≥ 0.9; the band is one global flag, so a run
mixing kinds takes the lower one); rows lr 1e-3 cosine, `--free_residual 1e-3`,
`--box_weight 4`; batch 4, compile, no grad-ckpt. Canvases are the **512²
family only** (user, 2026-09-16 evening): `--shapes 512` for flat items,
`--scene_min_tokens 900` on the scene pool (drops 448² and 448×512). The
sentence arm on top of this: every item a tategaki composite on sl1w with
hard kind quotas (`--scene_mix`). The full launch line is in
[`README.md`](README.md) *How to run*.

Budgets: **≈ 1 000 draws per row** (steps × batch / rows); **pin the share
of every kind that must be learned** (words, phrase pieces — they train only
through the items that carry them); **≈ 10 000 items per data build** (text
cache in RAM).

## Next steps (one arm at a time)

1. **Sentence arm — LAUNCHED 2026-09-16 20:13** as
   `rows_synth_sent_q_sent_s24k` (jobs `20260916-201323-226f69` data train
   eval, `…-a06a68` native en + swap); build, smoke numbers and the
   deletions in
   [`reports/synth_sentence_launch_2026_09_16.md`](reports/synth_sentence_launch_2026_09_16.md)
   *Relaunch*. **Read owed**: `phrase` vs `phrase_held` (sentences, 20 px
   glyphs — the small-glyph risk taken for this kind), `short` vs
   `short_held` (2–5-piece words, one column), `word`, `swap`, flat singles
   as the regression guard; the composite sheet. **Band caveat:** step 1
   inherited `--t_min 0.7 --t_max 0.9` from the singles recipe, so its
   multi-piece kinds never saw σ 0.5–0.7 — a low `short` / `phrase` there
   is band-or-data, not data; A2 carries the correction. Short items cap at 4
   glyphs and one-column sentences at 8 on sl1w — longer one-column text
   needs Track A's tall pool.
2. **Track A — tall JA colour-comic pool → sentence run 2** (below). A1
   done 2026-09-17 (359 kept); **A2 LAUNCHED 2026-09-17 03:32** as
   `rows_synth_sent2_comic_sent2_s24k` (jobs `20260917-033255-614ff9` data
   train eval, `…-7e0c80` native en + swap). Launched with step 1's flat
   eval unread — its eval job `20260916-203119-c6d4ff` was stopped at
   00:00, so step 1 has `native/` but no `report.md`. Built on
   the 53k + punctuation line; warm-starts from step 1's table.
3. **Punctuation read.** The readers' `norm()` strips punctuation, so the
   punctuation rows are read on the sheets; a CER on the unstripped string
   is the small eval change if punctuation becomes a gate.
4. **Track B — the 53k alternative** (below): punctuation + basic kanji +
   92 kana at 100 k steps → a separate run for the remaining kanji → merge →
   sentence run 2. It replaces the former "92 kana at 23 k" interference
   arm and the 108 k gate run: both asked whether the 53k base is short on
   exposure, and B answers that by rebuilding the base at ≥ 1 000 draws per
   row.
5. **Placement:** text on a subtitle bar / on the scene (as the EN refs place
   "hi") is scene-stage work; `s1sfx` (banner) is the pool that has it and
   is in no arm yet.

## Track A — 53k + punct line: `scenes_ja_comic` render → sentence run 2

Why: sl1w is an EN-anchored pool with wide bubbles, so the sentence arm
caps one-column text at 4 glyphs (short) and 8 glyphs (sentence), and
spills everything longer into two columns at 20 px. Asking the base for
*Japanese* text is the only lever measured to give tall bubbles (smoke:
31 % kept, 24/30 tall, 24 at AR ≥ 1.3, height median 134 px vs sl1w's
116/276 tall).

**A1. Render the pool — queued 2026-09-17 as `20260917-001741-967a1f`.**
Colour comic, not manga pages (user, 2026-09-17): the `ja_manga` argv
(killed at launch 09-16 `a14ef5`) with `monochrome,screentone` replaced by
`comic` and the tag renamed `ja_comic`. The smoke's 31 % kept / 24 of 30
tall were measured on the monochrome prompt, so the kept count below is
the monochrome expectation, not a colour one. All three shapes are ≥ 900
tokens, so the 512²-family rule holds.

**A1 result:** 2400 generated, **359 kept (15 %)** — half the monochrome
smoke's 31 %; open_bubble 75 % of rejects. Tall (AR ≥ 1.0) 248, AR ≥ 1.3
221, tall height median 135 px. `ja_bubble_reads` keeps 24 %, `ja_reads_as`
9 %, `ja_saying` 11 %. `comic` alone still pulls most scenes to greyscale /
lineart; the kept sheet carries a colour minority, not a colour pool.

**A2 smoke (400 items, CPU, `--scenes sl1w,ja_comic`):** missed 0, 169/633
scenes used. One-column caps (≥ 10 scenes, 28 px): **short 4 → 6 glyphs**;
sentences one column **65/160** (step 1: 40). Busiest scenes read on the
composites; dropped as text outside a bubble: `ja_comic:2,440,911,1316,1734`
(open scene / panel edge / beside the bubble) and `sl1w:962` (blank strip,
surfaced once the mix changed).

`comic` is a token of native prompt 8 (`comic, 2koma, 1girl, surprised
expression`), so from any run whose data carries ja_comic, **native is
held out on prompts 1–7 only**; read prompt 8 separately and never pool
it into a gate count.

    make daemon-run ARGS="--label scenes-ja-manga --stall-timeout 0 --queue \
        project/cjk_renderable_anima/src/wake_probe.py --stage scenes \
        --scene_tag ja_comic --scene_n 2400 --seed 2 \
        --scene_shapes 384x640,448x640,448x576 \
        --scene_frames ja_reads_as,ja_bubble_reads,ja_saying \
        --scene_extra_tags comic --scene_min_box 40"

Expect ≈ 740 kept, ≈ 590 tall. Before any data build: read the most-reused
scenes on a 400-item CPU smoke and add bubble-less tall regions to
`--scene_drop` (the sl1w lesson), and check the one-column caps the pool
actually buys (≥ 10 scenes holding a length) — this is the number A exists
for: short kind at 5 pieces in one column, sentences past 8 glyphs in one
column at ≥ 24 px.

**A2. Sentence run 2.** Step 1's argv with the pool changed to
`--scenes sl1w,ja_comic` (the same `--scene_drop sl1w:332,957` plus
whatever A1's smoke adds), **the band corrected to `--t_min 0.5 --t_max
0.9`** (step 1 ran the singles band 0.7–0.9; 90 % of the items are
multi-piece and the order/count window is σ 0.5–0.8 — the strings arm's
band, `plan.md` P1), `--init_rows` step 1's
`rows_synth_sent_q_sent_s24k/trained.pt`, 24 000 steps, one new
`--data_tag` / `--arm_tag`; the native job on both clauses. Only if step 1
reads clean (flat singles held, `short` / `phrase` above zero); if step 1
regresses, A2 warm-starts from `rows_synth_full_fm10k_merge_punct` again.
A2 vs step 1 then moves two things (pool, band); if the read needs them
apart, the cheap split is step 1's argv with only the band changed
(sl1w, 0.5–0.9) — ≈ 2.7 h — run before A2 or beside it. Guard: the wider
band is what dropped the strings arm's singles 34 → 5/36 when singles
were absent from the data; here flat singles stay at 0.1 of the items,
and flat singles ≥ step 1's is the gate that catches it.

Read: `phrase` / `phrase_held` and `short` / `short_held` against step 1
(same eval prompts), column count on the composite sheet, flat singles as
the guard, `swap` — ja_comic is JA-frame only, so frame independence is
what the sl1w half keeps.

Risks: every ja_comic scene carries `comic`, so the rows can take the
comic canvas (panel borders, page layout) as their unit (the flat-0
lesson: rows absorb whatever is constant) — keep sl1w at ≥ half the
composites and read native prompts 1–7 for panelling. Colour pages drop
the screentone erase risk the monochrome pool carried, but a coloured
bubble fill is where the ring-median patch shows; the sheets are the
check. The STYLES draw still gives ≈ 20 % of scenes screentone /
monochrome / halftone — variety, not a constant.

## Track B — the 53k alternative: kana + punct + basic kanji at 100 k → more kanji → merge → sentence run 2

Why: the 53k table fails its own flat gates at ≈ 490 draws per row
(singles 13/36, katakana + small kana gone, words 0/32), and every
sentence run warm-starts on those defects. B rebuilds the base at the
exposure the curve says holds (1 330 → 100 %), then adds kanji in a run
of their own. Two runs is the finest split worth making — see *Not this
plan*, per-row runs.

**B1. Base seed, 100 000 steps (≈ 11 h at 2.5 it/s).** Inventory: `kana`
(92) + the punctuation list (15, step 1's `list:`) + **basic kanji =
`kanji:200`** (the top-200 single-row corpus kanji, the 53k's set) = 307
rows. The 53k recipe otherwise (composite 0.9 / flat singles 0.1,
`--scenes s0,s1`, `--scene_min_tokens 900`, `--shapes 512`, rows lr 1e-3
cosine, `--free_residual 1e-3`, `--box_weight 4`, no `c_flat`, no Q),
**from scratch**, ≈ 10 000 items (RAM).

Exposure check before launch: draws are weighted per source (kana 1,
kanji 2, list 2), so at default weights 400 000 draws give kana ≈ 770 per
row and kanji / punct ≈ 1 530 — kana, the family that failed, would sit
below the planning number. Launch with **`--units kana*2`**: uniform ≈
1 300 per row. Confirm on the data stage's pool counts before training.

Left out on purpose: `kana_ext` (68) and `words:100` — words and small /
voiced kana reach the table through the sentence run's `short` /
`sentence` pieces (`--phrase_pieces`), where their exposure is the phrase
quota, not a singles weight that gave 53k's words 8 items each. If B1's
flat eval should gate `single_ext`, add `kana_ext*2` (375 rows → 125 k
steps for the same per-row count).

Gates (flat): singles ≥ 30/36, `single_kanji` ≥ 22/36, punctuation on the
sheet, EN 24/24; native `swap` per char ≥ step 1's. B1 failing singles at
1 300 draws per row is itself the katakana answer (render-side, not
exposure) — stop B there.

**B2. Remaining kanji, separate run.** The next corpus kanji by rank
(201 → N; N sized so steps = rows × 1 000 / 4 — 200 more ≈ 50 k steps,
400 more ≈ 100 k). Same recipe, same pools, with `chars:あ` as the anchor
the punct-only run used. **Code owed:** `kanji:N` is top-N only
(`kanji_inventory(tok, q, n)`); a rank range (`kanji:201-400`) is the
change, keeping the canonical-order / bit-identity contract.

**B3. Merge.** `src/probe/merge_tables.py --base <B1> --add <B2> --out …`
(`keep-base`, rows rescaled by `row_scale`) — the `merge_punct` precedent.
Before the sentence run, read the merged dir's flat eval (`--stage eval`
on the merge arm): each run grew its own shared trigger direction, and a
caption mixing B1 and B2 rows has not been rendered by either run. A
merged `single` / `single_kanji` below its source run's is the signal to
fold B2 back in as a warm-started joint run instead.

**B4. Sentence run 2 on the merged table** — step 1's argv (or A2's pool
if A1 has run), `--init_rows <B3>/trained.pt`, 24 000 steps; read against
step 1 on the same eval prompts. B vs A is then one comparison: same
sentence data, 53k + punct base vs the B base.

Budget: B1 ≈ 11 h + B2 5–11 h + B4 ≈ 2.7 h + natives, vs A ≈ the render
+ 2.7 h. A runs first on the queue; B1 is the long job behind it.

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
- **RAM.** The probe keeps every caption's text embedding in RAM (≈ 1.3
  MB each): 30 k items = 33 GB before latents and the DiT load, on a 46
  GB box. ≈ 10 k items per data build until the cache is paged.
- **Capacity.** 433 rows at ≈ 490 draws per row fell to 13/36 singles
  (exposure curve 1 330 / 670 / 490 → 100 / 75 / 36 %); plan ≈ 1 000
  draws per row and pin the share of every kind that must be learned.
  Whether katakana's loss is interference on top of exposure is step 4's
  question.
- **Bubble capacity for sentences.** Wrapping is in (3 columns, min glyph
  28, vertical first); the fit is now bounded by bubble *height* — the tall
  pool (`--scene_tall_ar 1.0`) is the lever, JA-frame scenes the fallback.

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
- The glyph encoder in any form; the S line is rows-only.
- Contrastive terms on text-free native images.
- Pasting onto the dataset's real images (off-manifold paste, caption
  style mismatch, nsfw/artist tags) — the self-generated scene replaces it.
- `c_flat` in any form: the micro loop measured cap 0.75 ≡ removed
  (2026-09-15); the S recipe drops the switch.
- Q (the quoted-EN adapter-output direction) as an inference-time
  replacement for a trained `c` — measured, halves exact hits. Q fixed on
  in training — **closed 2026-09-16**: inert on frame-mix data (2×2).
- Normalising fullwidth punctuation to ASCII in the phrase data — the OCR
  line does not, so inference captions carry the ext rows; train the rows,
  do not fold them.
- More items per data build past ≈ 10 k without paging the text cache.
- The floor-based kept margin as a gate — replaced by `en cos` / `box
  IoU`; a bare ground with a bubble scored as kept, and the base itself
  wipes `portrait, simple background` for EN.
- Restarting a running arm for a monitor value (leak, `‖c_flat‖`): the
  gates read at the end, and a mid-run change loses attribution.
- **One run per row (e.g. 50 steps per vocab item, save, merge at the
  end)** — asked 2026-09-16. The S line is rows-only with a batch-mean
  per-item loss, so a joint run already gives every row only its own
  items' gradient and the merge is a plain union: per-row runs buy
  ordering, which a sampler gives for free. What they cost: multi-row
  composites (swap / short / sentence) train against untrained or frozen
  neighbours, and the adapter's self-attn leak makes earlier rows never
  see later ones (B3's warning, one run at a time); model load + compile +
  text cache per run outweighs the training at 300+ rows; and 50 steps at
  batch 4 is 200 draws per row against the ≈ 1 000 the curve needs (490
  → 36 % singles), so the per-row step count is ~300 and the total is
  the joint run's. The part worth keeping is uniform draws per row —
  B1's `--units` weighting — and the modular unit is a family (B1 → B2 →
  B3), gated by B3's merged flat eval, not a row.
