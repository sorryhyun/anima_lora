# plan_synth — the S line (superseded; kept for its verdicts)

> **Superseded 2026-09-18.** The live plans are
> [`plan_synth3.md`](plan_synth3.md) (step 2, the sentence run) and
> [`plan_synth4.md`](plan_synth4.md) (step 1's recipe and the kanji budget);
> how the S line is built is [`synth.md`](synth.md), the settled verdicts are
> [`findings.md`](findings.md) / [`findings_seed.md`](findings_seed.md), and
> the dated record is [`reports/`](reports/README.md). What is left here is
> the budget numbers, the scene-pool results, the rulers and the
> do-not-re-propose list.

## Budgets (measured)

**≈ 1 000 draws per row** (steps × batch / rows; the exposure curve reads
1 330 / 670 / 490 draws → 100 / 75 / 36 % of singles); **pin the share of
every kind that must be learned** — words and phrase pieces train only
through the items that carry them, and a weighted draw is not a quota (the
53 k run's word rows got ≈ 8 items each and stayed at norm 0.12); **≈ 10 000
items per data build** (the text cache is ≈ 1.3 MB per caption in RAM on a
46 GB box).

## The scene pools

`scenes_ja_comic` (2026-09-17): asking the base for *Japanese* text is the
only measured lever that gives tall bubbles. 2 400 generated, **359 kept
(15 %)**, open_bubble 75 % of the rejects; tall (AR ≥ 1.0) 248, AR ≥ 1.3
221, tall height median 135 px. `ja_bubble_reads` keeps 24 %, `ja_reads_as`
9 %, `ja_saying` 11 %. `comic` alone still pulls most scenes to greyscale /
lineart — the kept sheet carries a colour minority, not a colour pool. Its
one-column caps over `sl1w,ja_comic` at 28 px: short 6 glyphs, sentences
one column 65/160.

`comic` is a token of native prompt 8 (`comic, 2koma, 1girl, surprised
expression`), so from any run whose data carries ja_comic, **native is
held out on prompts 1–7 only**; read prompt 8 separately and never pool
it into a gate count.

The sl1w lesson, before any new pool enters a build: read the most-reused
scenes on a 400-item CPU smoke and add bubble-less / text-outside-a-bubble
regions to `--scene_drop`.

**Placement** — text on a subtitle bar or on the scene (as the EN refs place
"hi") is scene-stage work; `s1sfx` (banner) is the pool that has it and is
in no arm yet.

## Rulers (re-based 2026-09-15)

Scene ruler is **`en cos`** (PE-Spatial cos to the EN reference of
the same prompt/seed; floor ≈ 0.93–0.97 is the ceiling), placement ruler
**`box IoU`** (glyph box vs the "hi" box; floor 0.36–0.51, every trained
cond so far 0.05–0.25 — read with the sheets, it is harsh on small
boxes). `stage native` renders trained conds only (`--native_floor 1`
restores the old kept margin). Native runs on **both** clauses — `en`
(the trained JA frame) and `swap` (the EN ref's caption, word swapped) —
and a row counts as a word token only when it hits under `swap`. The
per-arm gates live with the arm (`plan_synth3.md`, `plan_synth4.md`); on
multi-glyph groups the ruler is `sub_exact.py` pooled lift, not exact match.

Also read, not gated: row norm at the end of training (every arm drives
it to ≈ 125–130; the placement / identity trade-off is the delta norm),
per-char hits under `swap` (frame independence is per glyph, not per
arm — か vs あ), and the `native_swap` sheets before trusting a box IoU on
a small box.

## Open risks

- **Small glyphs inside composites.** A kana in a 64–100 px bubble at 512
  is 4–6 latent tokens a side; the box-weighted loss and the
  `--scene_min_glyph` floor (28 px since the sentence line) are the
  mitigation. `--scene_fill 0.7` made this slightly worse (median
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
  Katakana's loss is **not** row-space interference (`findings_seed.md`);
  untrained katakana dakuten rows even render at floor, and ΔFM training
  damages them (`reports/synth_s2_smoke_2026_09_18.md`).
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
- **One *run* per row (50 steps per vocab item, save, merge at the end)** —
  asked 2026-09-16. The S line is rows-only with a batch-mean per-item
  loss, so a joint run already gives every row only its own items'
  gradient and the merge is a plain union; what per-row runs buy is
  ordering, and `--row_blocks N` buys that inside one run (and is the
  better batcher at equal draws — `reports/row_blocks_alpha_2026_09_18.md`).
  What separate runs cost: multi-row composites (swap / short / sentence)
  train against untrained or frozen neighbours; model load + compile +
  text cache per run outweighs the training at 300+ rows; and 50 steps at
  batch 4 is 200 draws per row against the ≈ 1 000 the curve needs (490
  → 36 % singles). The part worth keeping is uniform draws per row (the
  `--units` weighting), and the modular unit is a family, not a row —
  gated by a merged flat eval (`plan_synth4.md` K0).
