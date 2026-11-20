# The text refiner turns a layout bbox into the location MOSS draws at

Measured 2026-10-28 on MOSS-Image Base. Probe, ablation and per-condition tables:
`_archive/bench/moss_bbox_grounding/` (`probe.py`, `ablate.py`, `README.md`).

## Setting

Teacher-forced, no sampling. The 36 cached swap_text targets with a single `text` item
(`post_image_dataset/moss/i2i_swap_text`: a page whose box holds pasted lettering, the
caption's item gives the line and its `abs_centralized` `yxyx` box), INT8 TE, fixed noise per
page, one cond forward per caption variant: the staged `bbox`, the box moved off itself
(`bbox_moved`), no box (`none`), and a 3×3 position word for the true or the moved box
(`pos`, `pos_moved`). Regions: the true box `T`, the moved box `M`, the rest.

## Measured

**The box matters at high σ.** In-box flow-matching MSE at σ 0.9: `bbox` 0.620, `none`
0.723, `bbox_moved` 0.723, `pos` 0.698, `pos_moved` 0.725. The moved box also raises the
loss in `M` (0.550 → 0.565): lettering is pushed where there is none. At σ 0.6 the visible
lettering carries the location and the box adds ~2 %; at 0.3, nothing. A position word gives
about a quarter of the box's gain, on the same heads, measured against a tight box it can
only name to a ninth of the canvas.

**Causal: the text refiner builds the location.** Masking the coordinate tokens as keys for
every query (σ 0.9; share of the box's in-box gain lost, 1 = gone; bootstrap 95 % CI):

| masked | gain lost | moved-box push left |
|---|---|---|
| in the two `text_embedder` blocks only (the DiT still sees the tokens) | 0.93 [0.55, 1.35] | 0.02 |
| in all 42 DiT blocks, refiner intact | 0.83 [0.46, 1.24] | 0.14 |
| refiner + all blocks, also the JSON tail and template suffix | 1.00 | 0.02 |
| DiT blocks 0–13 | 0.01 | 0.58 |
| DiT blocks 26–35 (best single range) | 0.39 | 0.40 |
| DiT blocks 9–41 | 0.60 | 0.42 |

- The digits have to read each other in the refiner to become a usable location. The
  refiner is unmodulated, so the location code is σ-independent, and the MOSS adapter
  targets (`layers.N.proj.*`) do not touch it: every LoRA trained so far left it intact.
- The DiT reads the refined coordinate tokens directly, spread over blocks ~14–35; no
  single range carries it. Blocks 0–13 do not need them, although the hidden-state
  difference between `bbox` and `bbox_moved` already concentrates in the two boxes from
  block 7.
- The tokens after the box carry it causally from the TE, but that route adds ≤ 0.04.
- Correlational read (`probe.py`): item-content attention follows the box mostly in blocks
  26–35, over many heads (the content map moves to the moved box on 70–90 % of pages).
  b14 h4 follows the box at every σ.

## The sanitize LoRA does its edit in the same blocks

`_archive/bench/moss_bbox_grounding/sanitize_probe.py`, sanitize v3 (EMA), trained without boxes:
4 eval pages rendered by the LoRA (its output as the target, so the edit region is what it
chose to remove) and 15 training pairs, teacher-forced with the isolated reference. Edit
region `E` = latent RMS diff > 0.6, `K` the rest. Share of the LoRA's gain over Base lost
with the LoRA off in a block range (19 pages):

| blocks | E σ 0.9 | K σ 0.9 | E σ 0.3 | K σ 0.3 |
|---|---|---|---|---|
| each of 0–4 … 21–25 | ≤ 0.03 | ≤ 0.01 | ≤ 0.07 | ≤ 0.02 |
| **26–35** | **0.79** [0.59, 1.01] | 0.34 | 0.30 | 0.46 |
| 36–41 | 0.17 | 0.26 | 0.61 | 0.67 |
| 14–35 | 0.81 | 0.48 | 0.35 | 0.53 |

- At σ 0.9 the edit runs through the LoRA in blocks 26–35 and is edit-specific there:
  the range where Base binds a bbox item's content to its box and reads the coordinates
  most. 14–25 add nothing on top. At low σ, 26–41 carry the edit and the copy alike.
- The LoRA's change to the reference rows (σ-independent: isolated at `ref_timestep 0`)
  separates E from K with AUC rising from 0.51 (block 0) to 0.80 (block 25): a text mark
  built in the reference stream, as the refiner builds the bbox's location code. With the
  LoRA off in 14–25 nothing is lost, so the mark there is redundant, or Base's reference
  features already carry the text and 26–35 read them.
- New caption-reading heads for the edit region (E over K mass on `no sfx, no dialogue, no
  watermark`: e^3–e^5, Base ~e^0) sit late, b35–b41 (b40 h20, b39 h19, b41 h2, b37 h20,
  b35 h3), not on the heads that bind a bbox item in Base.
- The structure takes the full run to form. At 276 steps on a quarter of the pairs
  (`mod_rank_i2i.md` § Sanitize) blocks 26–35 off lose 0.36 of `E`'s gain and 0.41 of
  `K`'s at σ 0.9, the reference-row AUC stays ≤ 0.37 up to block 30 (0.68 at 35), and the
  caption heads are the same ones at about half the strength.
- Whether a page gets cleaned is decided per sample: `usual_case1`'s corner watermark is
  removed at seed 42, half removed at 44, kept at 0, 1, 2 and 43 (CFG 3 and 4 alike).

## Hypothesis (untested)

An i2i task can hand the location to this path instead of learning it: put the edit region
in the layout prompt as an item with a box (from OCR, a detector or a small predictor on the
reference) and let the LoRA learn only what to draw there. The location is decided at high σ,
which is where an i2i LoRA otherwise has to find the region from the reference alone, and
the sanitize LoRA's edit runs through the blocks that read the box (26–35). A box would
also pin the per-seed keep-or-remove call.

## Open

- Labels other than `text` (characters, objects): same path or not.
- A training A/B: an i2i task with and without box items at small data.
