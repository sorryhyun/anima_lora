# LoRA blocks: precise MOSS i2i edits assembled from trained parts

Status: proposal (2026-11-07). Nothing below is built except the inference probes it rests
on. Evidence: [`project/moss_te_sae/reports/copy_protocol.md`](../moss_te_sae/reports/copy_protocol.md),
[`i2i_copy_head.md`](../moss_te_sae/reports/i2i_copy_head.md),
[`docs/moss/findings/bbox_grounding.md`](../../docs/moss/findings/bbox_grounding.md),
[`mod_rank_i2i.md`](../../docs/moss/findings/mod_rank_i2i.md). Supersedes steps 1 and 3 of
[`docs/moss/proposal/moss_i2i_ambitious.md`](../../docs/moss/proposal/moss_i2i_ambitious.md).

## Goal

Build a precise edit LoRA (sanitize, swap_text, colorize, and the next one) from parts
that are trained once, on cheap data, and reused: compose them, or init from the
composition and fine-tune briefly on a few real pairs. The target is not a general
instruction editor. It is the class of edits we already ship, each of which today needs
its own curated pair set and a long run:

| task | pairs | steps | what the data cost |
|---|---|---|---|
| sanitize v3 | 277 (39 identity) | 2,216 | in-artist near-twin mining, three versions, a tone term for the brightening |
| swap_text ksplit e2 | 2,109 per epoch (8,082 staged) | 4,218 | edition twins crawled and aligned, OCR captions, the Korean token split |
| colorize v5 | 3,334 | 10,000 | control captions (character colour items, text boxes) |
| uncensor | 66 | 660 | — |

Each of these re-learns how to copy the page. Each also learns where to edit from the pairs
alone, and its failures are mostly failures of that "where": sanitize leaves a faint stamp
as a ghost and removes `usual_case1`'s watermark on some seeds only
([moss_i2i_ambitious.md](../../docs/moss/proposal/moss_i2i_ambitious.md)).

## An edit in three parts

| part | what it does | how it is set |
|---|---|---|
| **copy** | encode the reference and read it back at the same cell | a trained block; a protocol spread over the whole network |
| **gate** | decide where the copy holds | the main-query → reference-key edges, set from a mask (no parameters), or by a detect block |
| **draw** | paint what goes inside the gate | a trained block per kind of content |

What we measured about each:

- **Copy is common.** Every i2i LoRA we trained, and an outside edit LoRA trained on
  different data with a different trainer, builds an own-cell copy head at b39 h19 that
  Base lacks. The i2i LoRAs share a late-block output subspace that t2i runs do not.
- **Copy cannot be lifted out head by head.** b39 h19 alone, its q / k alone, the ten top
  copy heads, or blocks 36–41 leave the copy at Base level; the full LoRA on the reference
  stream without the main rows is worse than Base (the main rows can no longer read the
  reference). The unit of reuse is a whole LoRA, not a slice.
- **The gate has a complete handle.** With `ref_isolate` the reference enters the main rows
  only through main-query → reference-key attention. A logit bias on those edges for the
  queries in a mask moves the copy monotonically (box luma L1 6 → 34 → 41 → 48 at bias
  0 / −2 / −4 / −∞) and leaves the rest of the page where it was (5.9–6.1 outside on the
  horns case). It overrides a wrong decision of the LoRA (the flower it would not remove).
  A hard cut also stops the box from reading the reference, and loses what should stay
  (the eyes under the horns).
- **Base already has a "where" path from text.** A layout bbox becomes a location code in
  the text refiner, read over blocks ~14–35. Sanitize, trained without boxes, edits in the
  same blocks.
- **Drawing is task-specific.** swap_text alone retunes Base's string-read heads
  (b19 h6, b21 h20, b24 h11, b26 h12); colorize's copied value (v / o of the copy head) is
  not the other tasks'.

## The blocks

### G — gate (no parameters)

A per-query logit bias on the reference keys, in every block and head, both CFG branches:

- `hard`: −∞ for queries in the mask;
- `diag`: −∞ only on the own-cell edge and its ±k neighbourhood for queries in the mask;
  the box still reads the rest of the reference (style, font, what lies around);
- `soft`: a finite bias; a schedule over σ is a variant.

Mask sources, cheapest first: the layout bbox items (swap_text has them), the AnimeText
text-block detector in `anime_tools.ocr.animetext` (balloon lines and SFX), SAM3 subject
masks (`anime_tools.masking`), a hand mask, and later the detect block D.

### C — copy / fill

One LoRA trained on clean images only. Pair = (image, same image); a random hole in the
target-side gate (`diag` or `hard`), shaped like real text masks taken from other pages'
AnimeText output, plus boxes and SAM3 blobs. Loss on all tokens. Outside the hole the block
learns the copy protocol; inside it learns to fill from the surroundings and the
reference's non-local context. No pairs to mine; any clean set works
(`image_dataset/`, ~3k images).

Text removal is this fill with text-shaped holes. Whether sanitize needs a separate
removal block beyond C is a phase-2 question.

### D — detect (optional; external first)

The mask that drives G. First as an external detector (AnimeText boxes → mask), which fits
the goal: a precise edit can be a detector plus an edit model. Later, if wanted inside the
DiT: a small block on the reference stream supervised on the gate itself (the copy mass of
the image queries over the reference matches the mask), trained on texted pages with
detector masks, no target images. Watermarks need their own mask source; AnimeText does
not cover them.

### L — letter

swap_text's drawing: letter a given string into a given box in the source's style. Trained
on C + G(`diag`, the layout boxes) so its updates go to the strings, not to the copy.
Candidate data beyond edition twins: synthetic lettering on clean balloons (the fonts in
`project/cjk_anima_reseed/assets/fonts/`).

### K — colour

colorize's drawing. The stress case: its value path differs, so C's RGB copy may brake it
(the copy-head bias study added colour by weakening exactly that copy). C trained on
luma-only references is a variant to try.

Tasks as assemblies:

| task | blocks |
|---|---|
| sanitize | C + G(mask from D or AnimeText) |
| uncensor | C + G(mask of bars / mosaic) + an anatomy drawing block |
| swap_text | C + G(`diag`, layout boxes) + L |
| colorize | C(luma) + K, G open everywhere |

## Composition

1. **Stack, frozen.** Train C; train the next block with C frozen and hooked at multiplier 1
   (the int8 base cannot take a merge). Compatible by construction, sequential not
   modular.
2. **Concat-rank init.** The new LoRA's `down` / `up` are the blocks' factors stacked
   (rank r₁ + r₂ + …), so step 0 is exactly the sum; then a short fine-tune on real
   pairs. All our LoRAs share the seed-0 kaiming `down` init, so the blocks' input sides
   start aligned.
3. **Spatial routing.** A block's delta only on some rows: C everywhere (the reference
   stream must stay encoded, and the hole still reads the reference), a draw block only on
   the queries inside the gate. `LoRAAdapter.row_mask` (today `delta_rows`) already does
   this for text rows.
4. **Instruction assembly** (last). The prompt picks the blocks and the gate: a bbox item
   with a verb ("remove", "letter …") routes to G + the block. Steering an existing edit
   LoRA through G is the cheap version of this and needs nothing new.

Under Muon the early and middle ΔW of two runs of the same task share only 0.06–0.18 of
their output subspace (late blocks 0.61–0.83), so a plain sum of independently trained
blocks may interfere. Phase 4 measures that before anything relies on it.

## Phases

Each phase is a micro arm or an inference sweep, with a gate. Runs past ~10 min are the
user's call.

**P0 — gate on our own LoRAs (inference only, ~10–20 min each).**
- `diag` against `hard` / `soft` on the Edit LoRA cases of copy_protocol.md. Pass: the
  eyes stay under the cat ears and the flower still goes.
- sanitize v3 + G from AnimeText boxes on the five eval pages, seeds 0–2. Pass: the
  misses (faint stamp, `usual_case1`'s watermark) go on every seed, ΔL / Δb outside no
  worse than v3 alone. This tests "detection as a separate part" without training.
- swap_text ksplit e2 + G(`diag`, layout boxes). Pass: nothing outside the boxes moves
  (signed ΔL), letters unchanged.

**P1 — train C (one micro arm, then one full).** Clean images, random text-shaped holes,
`diag` gate in training. Pass: luma L1 outside the hole at or below sanitize v3's on
held-out pages at ≤ 1/4 of its steps; the fill reads as background. Read: the copy head
forms (b39 h19 own-cell mass), and when.

**P2 — sanitize assembled.** C + G(AnimeText) zero-shot on the five eval pages; then
concat-rank init + fine-tune on the 277 pairs, and on 50 of them. Pass: v3's removal and
tone at ≤ 25 % of its steps, or with 50 pairs. Kill: no gain in steps or pairs over
training from kaiming with the same G in training. Then G alone (P0) is the result.

**P3 — reuse on a second task.** swap_text en50_ko150 micro from C (concat init, `diag`
gate on the layout boxes) against the kaiming baseline at equal steps. Pass: signed ΔL
outside the boxes reaches ksplit e2's earlier; letters no worse.

**P4 — interference.** Sum C + L (or C + K) with no fine-tune; measure each block's own
read (copy outside, letters inside) against each block alone. Decides whether
composition 2 needs the fine-tune or whether 1 / 3 are required.

**P5 — optional.** D in the DiT (gate-supervised); instruction assembly.

## Code it needs

- `library/moss/model.py`: a reference gate on the main rows' attention (per-query bias
  or cut against the reference keys). Today a masked call drops from FA2 to SDPA. `hard`
  can stay on FA2 as two calls (gated queries without the reference keys); `diag` and
  `soft` need a mask (SDPA, or flex attention with a block mask).
- `library/moss/requests.py` / `train.py`: `ref_gate` (`off | hard | diag | soft`, `k`,
  bias) and `gate_source` (`region` = the existing target − reference diff, `boxes` = the
  layout items, `masks` = files beside the cache); `init_loras` (files + coefficients,
  concat rank) and `frozen_loras` (hooked at a fixed multiplier); spatial row masks per
  sample on `LoRANetwork` (generalising `delta_rows`).
- A C-block data stage: clean images → identity pairs + hole masks (AnimeText masks from
  other pages, boxes, SAM3), as an i2i staging task under `i2i_tasks/`.
- `generate.py` and the ComfyUI Reference node: a mask input for G.
- Probes exist: `project/moss_te_sae/src/{copy_head,copy_head_ext,copy_transplant,ref_edge_mask}.py`
  (copy mass by block / head / step, gate against the edit region, edge bias at inference).

## Open questions

- Is C one block for every task, or one per copied value (RGB for removal, luma for
  colour)?
- Does a block trained under G keep working when G is off (users without a mask)?
- How much of an existing task LoRA's training goes into the copy? A read on saved
  checkpoints (copy mass against step) bounds what C can save, before P1.
- Reference frame convention: ours (text length) gives more copy than the Edit LoRA's own
  (frame 0) on its weights, 0.81 against 0.66 at step 0. Keep ours for our blocks.
