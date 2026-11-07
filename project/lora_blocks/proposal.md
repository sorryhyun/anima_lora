# LoRA blocks: precise MOSS i2i edits assembled from trained parts

Status: proposal (2026-11-07). P0 is done ([reports/p0_gate.md](reports/p0_gate.md)); the
gate exists only as the probe `project/moss_te_sae/src/ref_gate.py`, and no block is
trained. What is measured lives in [findings/](findings/): [copy.md](findings/copy.md),
[gate.md](findings/gate.md). Older evidence:
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

Each of these re-learns how to copy the page (200–300 steps of it, findings/copy.md) and
learns where to edit from the pairs alone. Their failures are failures of that "where",
and P0 fixed two of them from outside the LoRA.

## An edit in three parts

| part | what it does | how it is set |
|---|---|---|
| **copy** | encode the reference and read it back at the same cell | a trained block; a protocol spread over the whole network |
| **gate** | decide where the copy holds | the main-query → reference-key edges, set from a mask (no parameters), or by a detect block |
| **draw** | paint what goes inside the gate | a trained block per kind of content |

Base already has a "where" path from text: a layout bbox becomes a location code in the
text refiner, read over blocks ~14–35, and sanitize, trained without boxes, edits in the
same blocks. Drawing is task-specific: swap_text alone retunes Base's string-read heads
(b19 h6, b21 h20, b24 h11, b26 h12); colorize's copied value (v / o of the copy head) is
not the other tasks'.

## The blocks

### G — gate (no parameters)

A per-query logit bias on the reference keys, in every block and head, both CFG branches:

- `diag<k>`: −∞ on the own-cell edge and its ±k neighbourhood for queries in the mask; the
  box still reads the rest of the reference. The removal mode.
- `boost<v>`: +v on every reference key for queries outside the mask. The lettering mode
  (`boost1` for swap_text).
- `soft<v>`: −v on every reference key for queries in the mask.
- Variants not yet tried: an own-cell boost, σ schedules, cut + boost together.

Mask sources, cheapest first: the layout bbox items (swap_text has them), the AnimeText
text-block detector in `anime_tools.ocr.animetext` (balloon lines and SFX), SAM3 subject
masks (`anime_tools.masking`), a hand mask, and later the detect block D.

### C — copy / fill

One LoRA trained on clean images only. Pair = (image, same image); a random hole in the
target-side gate (`diag`), shaped like real text masks taken from other pages' AnimeText
output, plus boxes and SAM3 blobs. Loss on all tokens. Outside the hole the block learns
the copy protocol; inside it learns to fill from the surroundings and the reference's
non-local context. No pairs to mine; any clean set works (`image_dataset/`, ~3k images).

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
on C + G(`boost` outside the layout boxes) so its updates go to the strings, not to the
copy. Candidate data beyond edition twins: synthetic lettering on clean balloons (the fonts
in `project/cjk_anima_reseed/assets/fonts/`).

### K — colour

colorize's drawing. The stress case: its value path differs, so C's RGB copy may brake it
(the copy-head bias study added colour by weakening exactly that copy). C trained on
luma-only references is a variant to try.

Tasks as assemblies:

| task | blocks |
|---|---|
| sanitize | C + G(`diag`, mask from D or AnimeText) |
| uncensor | C + G(mask of bars / mosaic) + an anatomy drawing block |
| swap_text | C + G(`boost` outside the layout boxes) + L |
| colorize | C(luma) + K, G open everywhere |

## Composition

1. **Stack, frozen.** Train C; train the next block with C frozen and hooked at multiplier 1
   (the int8 base cannot take a merge). Compatible by construction, sequential not
   modular.
2. **Concat-rank init.** The new LoRA's `down` / `up` are the blocks' factors stacked
   (rank r₁ + r₂ + …), so step 0 is exactly the sum; then a short fine-tune on real
   pairs.
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

**P0 — gate on our own LoRAs.** Done: [reports/p0_gate.md](reports/p0_gate.md).

**P1 — train C (one micro arm, then one full).** Clean images, random text-shaped holes,
`diag` gate in training. The copy part is read: 300 steps of sanitize's own pairs on the
int8 plan build the copy head (`probe_sanitize_copy_s300`, saves every 25 steps,
[reports/copy_vs_step.md](reports/copy_vs_step.md)). Not yet run: clean images, holes, a
gate in training (the trainer has no `ref_gate`), so the hole and the gate are what this
arm tests. Pass: luma L1 outside the hole at
or below sanitize v3's on held-out pages; the fill reads as background. Read: the copy head
forms (b39 h19 own-cell mass), and when.

**P2 — sanitize assembled.** C + G(AnimeText) zero-shot on the five eval pages; then
concat-rank init + fine-tune on the 277 pairs, and on 50 of them. Pass: v3's removal and
tone at ≤ 25 % of its steps, or with 50 pairs. Kill: no gain in steps or pairs over
training from kaiming with the same G in training. Then G alone (P0) is the result.

**P3 — reuse on a second task.** swap_text en50_ko150 micro from C (concat init, `boost`
gate outside the layout boxes) against the kaiming baseline at equal steps. Pass: signed ΔL
outside the boxes reaches ksplit e2's earlier; letters no worse.

**P4 — interference.** Sum C + L (or C + K) with no fine-tune; measure each block's own
read (copy outside, letters inside) against each block alone. Decides whether
composition 2 needs the fine-tune or whether 1 / 3 are required.

**P5 — optional.** D in the DiT (gate-supervised); instruction assembly.

## Code it needs

- `library/moss/model.py`: the reference gate on the main rows' attention, ported from
  `RefGate` in `project/moss_te_sae/src/ref_gate.py`. The probe's kernels: `boost` / `soft`
  as two FA2 calls merged by LSE with the per-query bias; `diag` as flex attention with the
  cut in a score_mod, compiled at static shapes outside the block graph (the blocks'
  `dynamic=True` compile cannot lower flex on torch 2.14).
- `library/moss/requests.py` / `train.py`: `ref_gate` (`off | diag | boost | soft`, `k`,
  bias) and `gate_source` (`region` = the existing target − reference diff, `boxes` = the
  layout items, `masks` = files beside the cache); `init_loras` (files + coefficients,
  concat rank) and `frozen_loras` (hooked at a fixed multiplier); spatial row masks per
  sample on `LoRANetwork` (generalising `delta_rows`).
- A C-block data stage: clean images → identity pairs + hole masks (AnimeText masks from
  other pages, boxes, SAM3), as an i2i staging task under `i2i_tasks/`.
- `generate.py` and the ComfyUI Reference node: a mask input for G.
- Probes: `project/moss_te_sae/src/{copy_head,copy_head_ext,copy_transplant,ref_edge_mask,ref_gate,copy_vs_step}.py`.

## Open questions

- Is C one block for every task, or one per copied value (RGB for removal, luma for
  colour)?
- Does a block trained under G keep working when G is off (users without a mask)?
- Reference frame convention: ours (text length) gives more copy than the Edit LoRA's own
  (frame 0) on its weights, 0.81 against 0.66 at step 0. Keep ours for our blocks.
