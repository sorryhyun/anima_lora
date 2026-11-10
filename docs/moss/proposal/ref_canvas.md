# MOSS i2i: the reference as a panel of one layout canvas (`ref_pos = canvas`)

Status: proposal, 2026-11-10. Nothing below is built or run. Context: the character_ref
`ref_pos` A/B (`aligned` against `offset`, `docs/moss/methods/i2i.md`), and the copy block
C of `project/lora_blocks/`.

## Where the reference sits today

The reference latent rides the image's sequence at RoPE coordinates `(r, f, h, w)`
(`library/moss/model.py`, `_RoPE.forward`):

| tokens | r | f | h, w |
|---|---|---|---|
| image | 0 | text length | its grid, centred on the image |
| text | 0 | 0 … S−1 | 0 |
| reference, `aligned` | 1 | text length | its grid stretched onto the image's |
| reference, `offset` | 1 | text length | as `aligned`, w + image width |

- **Base never saw `r ≠ 0`.** The released pipeline puts every token at r = 0
  (`pipeline_moss_image.py:180,183`); the README does not mention the axis. In Base the r
  quarter of each head (32 of 128 dims) is a position-free channel.
- **r = 1 is a weak tag.** The r axis shares the 16 bands `10000^(−j/16)` of the other axes,
  so a step of 1 rotates by 1, 0.56, 0.32, 0.18, 0.10 rad … 2e-4 rad: four or five bands move,
  the rest barely do. A reference token at the image token's own (h, w) is one step away, the
  distance of a neighbouring cell, on an axis Base has no use for. What separates the
  reference in practice is its clean modulation (`ref_timestep 0`) and `ref_isolate`.
- **`aligned` is a copy-the-same-cell prior.** It is why the edit tasks (colorize, swap_text,
  sanitize, uncensor) learn fast; the copy head forms at steps ~175–225
  (`project/lora_blocks/reports/copy_vs_step.md`). A task whose content sits elsewhere in the
  reference (character_ref) gets the prior against it.

## Proposal

Make the reference a panel of the canvas Base already knows how to describe:

1. **Coordinates (`ref_pos = canvas`).** The reference at r = 0, beside the image (as
   `offset`), and both grids re-centred on the centre of the two-panel canvas, so the image's
   own (h, w) move by half a panel width. To Base, image and reference are one wide picture.
2. **Prompt.** The layout JSON describes the canvas: one item per panel with its `bbox` in
   `abs_centralized` `yxyx` pixels of the full canvas (`anime_tools.captions.layout_json.centered_box`
   on the canvas size): the image's panel carries the target's caption, the reference's
   panel a fixed caption ("reference"). The boxes depend on the target and reference sizes,
   so this is a cache-time caption wrap and a separate TE cache.

Why it may pay:

- **Base reads layout boxes.** The text refiner turns a layout bbox into the place MOSS draws
  at; the box matters at high σ, where in-box FM MSE drops 0.723 → 0.620 at σ 0.9
  (`docs/moss/findings/bbox_grounding.md`). A box per panel uses a channel Base has, instead of
  one it does not.
- **Multi-panel consistency is a t2i skill.** Comic pages and character sheets draw one
  character across panels; In-Context LoRA (FLUX) draws on this with short LoRAs (from memory,
  not checked against the paper).
- **It opens the prompt to the reference.** Once the reference is a described panel, an
  instruction can name regions of either panel in the same coordinates; the free-prompt edit
  of `moss_i2i_ambitious.md` needs exactly that.

Token count does not change: the reference tokens are in the sequence already; only their
coordinates and the caption change.

## First experiment: C under canvas

Retrain the copy block C, the init every i2i fine-tune now starts from, under canvas. Not
from C's weights: C learned the copy on `aligned` coordinates. If the copy forms under
canvas, every task built on C can move with it.

Recipe: `c_fill_bw10_mod16_s800` as it is (`project/lora_blocks/configs/`): cache
`post_image_dataset/moss/c_fill_illu_bw10`, 800 steps, rank 32, `mod_rank 16`, lr 5e-5,
`ref_gate cut` at `gate_rate 0.9`, `ref_isolate`, `ref_timestep 0`, swap 36 / budget 0.12
(~33 min). `cut` indexes reference tokens, not coordinates, so it carries over.

| arm | coordinates | caption | new cache |
|---|---|---|---|
| bw10 (exists) | `aligned` | as cached | — |
| canvas-1 | `canvas` | as cached | no |
| canvas-2 | `canvas` | + panel boxes | TE only |

canvas-1 against bw10 measures the cost of giving up the same-cell prior on a copy task;
canvas-2 against canvas-1 what the layout boxes buy back.

Reads: `project/moss_te_sae/src/ref_gate.py --part fill` (the five sanitize pages, seeds
0–2, `ctrl` / `cut`), the numbers of `reports/c_fill_mod16_bw10.md` (L1 outside the mask,
ΔL / Δb), and the copy-head step from `copy_vs_step.py`. Then character_ref from canvas-2's C
against the `offset` run.

## Code it needs

- `_RoPE.forward`: `canvas` mode (r = 0, panel beside, re-centred); the per-sample and
  batched paths; `ref_pos` stamped and read as today.
- Cache: a caption wrap that adds the two panel items from the target and reference sizes
  (`CacheRequest` field, stamped in the TE file like `prompt_tail`), and the same wrap in
  generate.
- ComfyUI MOSS Toolkit: `reference.py::ref_rope` and the prompt helper follow the stamp,
  only once an arm ships.

## Open

- **σ mismatch.** Base's multi-panel images had every panel at one noise level; the
  reference here is clean at `ref_timestep 0`. Feeding the reference at the image's σ
  (repaint-style) is the alternative, at more code.
- **Which side.** The reference on the right is arbitrary; reading order may matter for a
  model trained on comics.
- **Out-of-range coordinates.** At 1024² a two-panel canvas spans w ±64 tokens, inside what
  Base sees at 2048²; a wide target with a wide reference goes past it.
