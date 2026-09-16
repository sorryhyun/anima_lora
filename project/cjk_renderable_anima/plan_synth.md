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
singles 0.1; σ 0.7–0.9, rows lr 1e-3 cosine, `--free_residual 1e-3`,
`--box_weight 4`; batch 4, compile, no grad-ckpt, shapes
`448,512:2,448x512,512x448`. The full launch line is in
[`README.md`](README.md) *How to run*.

Budgets: **≈ 1 000 draws per row** (steps × batch / rows); **pin the share
of every kind that must be learned** (words, phrase pieces — they train only
through the items that carry them); **≈ 10 000 items per data build** (text
cache in RAM).

## Next steps (one arm at a time)

1. **Sentence arm — tategaki sentences on `scenes_sl1w` (user, 2026-09-16
   evening).** Every item is a composite on `scenes_sl1w`, drawn vertical
   (columns right-to-left), with the text-length mix pinned:
   - **10 %** — 1 token (one Qwen piece: a single unit / punctuation row);
   - **50 %** — 2–5 tokens;
   - **40 %** — Japanese sentences (Manga109 dialogue lines,
     `--phrase_file`, held by book → `phrase_held`).

   Why the stopped `sent_s24k` had no sentences: the composite draw takes
   the first of 40 random lines that fits the bubble, so of 2 092 phrase
   composites 1 405 were 3–4 glyph interjections (`ハハハ・・・`, `何っ！`),
   and 1 886 of the phrase draws fell back to singles (74 % of composites
   were singles). Owed in `stage/synth.py` before launch: the shares above as
   hard quotas (on a miss, re-pick the scene, never demote the kind); a
   sentence floor so the 40 % are sentences, not interjections; a 2–5-token
   source (`dialogue_3_10.tsv` starts at 3 pieces). Check the composite
   sheet for tategaki sentences before GPU time. Otherwise as the stopped
   arm: 10 000 items, min glyph 28, 3 columns, 24 000 steps, warm-started
   (`--init_rows`); read on `phrase` vs `phrase_held`, `word`, `swap`, flat
   singles as the regression guard.
   After the launch: delete the `output/wake_probe/` `rows_synth_*`,
   `scenes_*`, `encoder_*`, `data_*` dirs that are unneeded, misleading or
   from failed runs; keep a dir only where it is the superset / replacement.
2. **If the tall pool runs short:** the JA-frame scene recipe
   (`ja_reads_as / ja_bubble_reads / ja_saying`, `--scene_ja_anchors`,
   `--scene_extra_tags monochrome,screentone`, `--scene_min_box` 56 → 40),
   smoke-measured, not run at scale.
3. **Punctuation read.** The readers' `norm()` strips punctuation, so the
   punctuation rows are read on the sheets; a CER on the unstripped string
   is the small eval change if punctuation becomes a gate.
4. **92 basic kana at 23 000 steps** (≈ 1 000 draws per row, ≈ 2.5 h) —
   only if katakana still fails after sentence exposure: holds → the 53k
   katakana loss was interference at 433 rows; fails → render-side.
5. **Gate run:** same inventory at ≈ 1 000 draws per row (433 rows → ≈
   108 000 steps ≈ 12 h, or warm-started for the remaining budget), word
   share pinned, re-based on the sentence table.
6. **Placement:** text on a subtitle bar / on the scene (as the EN refs place
   "hi") is scene-stage work; `s1sfx` (banner) is the pool that has it and
   is in no arm yet.

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
