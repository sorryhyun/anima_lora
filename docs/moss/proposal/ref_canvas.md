# MOSS i2i: the reference as a panel of one layout canvas (`ref_pos = canvas`)

Status: proposal, 2026-11-10, probed the same day (`bench/moss_ref_canvas/`). Base does hold a
character across two panels, and reads a reference panel that is noised with the image and
attends both ways. It does not read the isolated reference of the i2i recipe, on canvas or
`offset` coordinates, and the panel boxes change nothing. What runs from here is the copy
block at `ref_pos offset` ([below](#what-runs-now-the-copy-block-beside-the-image)); the
canvas coordinates and the caption wrap are not built. Context: the character_ref `ref_pos`
A/B (`aligned` against `offset`, `docs/moss/methods/i2i.md`), and the copy block C of
`project/lora_blocks/`.

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

## Proposal (as first written)

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

## Measured (2026-11-10)

`bench/moss_ref_canvas/` (README: conditions, both tables, the run folders). Base, int8 DiT, no
LoRA.

**Renders** (`panels.toml`, a 1728×1216 canvas, 4 prompts × seeds 0–2, no prompt naming hair,
eyes or outfit): with two `panel` boxes, or one sentence naming two panels, all 6 renders
keep hair, outfit and style across the panels. A right panel captioned `reference image of the
character` repeats the left scene, pose included. `character sheet` tags give sheets whose
views change outfit on 2 of 3 seeds.

**Teacher-forced** (`probe.py`: 24 character_ref pairs with a same-grid reference, 24 artists;
flow-matching MSE over the image panel, the pair's reference minus another artist's):

| reference | σ 0.9 | σ 0.6 | σ 0.3 |
|---|---|---|---|
| `joint`: r = 0 beside, noised at the step's σ, attention open | −0.0153 | −0.0112 | −0.0136 |
| `open`: clean at `ref_timestep 0`, attention open | −0.0031 | −0.0034 | −0.0014 |
| `iso`: clean, `ref_isolate`, canvas coordinates | −0.0012 | −0.0013 | −0.0017 |
| `offset`: clean, `ref_isolate`, r = 1 | −0.0009 | −0.0011 | −0.0015 |
| isolated at the step's σ (`joint` + `ref_isolate`) | −0.0016 | −0.0019 | −0.0025 |
| isolated at σ 0.3, `ref_timestep 300` | −0.0036 | −0.0019 | −0.0025 |

For scale, another pair's caption costs +0.0078 at σ 0.9 and nothing below.

- **The premise holds for a joint panel.** The matching reference is worth twice the right
  caption at σ 0.9, at every σ (the 95 % CIs exclude 0).
- **Isolation removes 82–90 % of it**, at a matched σ too; a clean reference with the attention
  open keeps as little. Base's read across panels needs reference rows that share the image's
  noise and read the image and the text. `RefCache` needs the opposite (rows that depend on
  no step), and no configuration it can hold reads more than a quarter of `joint`; the best is
  the reference at σ 0.3 with `ref_timestep 300`.
- **Canvas and `offset` coordinates read alike under isolation**, and the panel-box caption
  moves no condition by more than 0.001. Neither arm of the experiment first planned here
  (C retrained on canvas coordinates, with and without the boxes) starts from more than the
  `offset` run did.
- Every reference condition raises the loss over no reference (`joint` +0.005 to +0.026, the
  clean ones more). Not explained.

**The current C on the character_ref eval inputs** (`output/moss_i2i_eval/character_ref/results/
{c_init,c_init_off}`, seeds 0–2): as trained (`aligned`) it returns the reference on every
seed and ignores the prompt. The same file read at `offset`, untrained there, follows the
prompt and loses the picture (washed, grey figures), with the reference's scene and palette
in it: a park reference gives its hedge, paving and sky at the reference's heights, a red-eyed
one red eyes on one of two seeds. `off_e1`, 1,000 steps of character_ref from kaiming, renders
clean and carries nothing. The read C learned survives the move; the placement does not.

## Two routes

1. **Ride the prior.** The reference noised to the step's σ, its rows at the step's
   timestep, attention open, canvas coordinates. No `RefCache`: twice the tokens on every
   forward. It may need no training; the check is a render with the right panel held to the
   noised reference at every step and the left one generated. Not run.
2. **Keep `RefCache`.** The reference stays isolated at a fixed timestep and a LoRA learns
   the read. Canvas coordinates add nothing here, so `offset` stands in for them.

## What runs now: the copy block beside the image

C retrained at `ref_pos offset`, then character_ref from it (route 2).

- `project/lora_blocks/configs/c_fill_bw10_mod16_off_s800.toml`: the recipe of
  `c_fill_bw10_mod16_s800` (cache `post_image_dataset/moss/c_fill_illu_bw10`, 800 steps, rank
  32, `mod_rank 16`, lr 5e-5, `ref_gate cut` at `gate_rate 0.9`, `ref_isolate`,
  `ref_timestep 0`, swap 36 / budget 0.12, ~33 min) with `ref_pos = "offset"`.
- Read 1: the same character_ref eval renders as `c_init_off`. `offset` moves w and leaves h,
  so an identity copy can be learned as "same row, one image width across", a position copy
  again; the renders show it (the reference returned, prompt ignored). If so, the next C
  takes a target that is a crop, shift or flip of its reference, which no position matches.
- Read 2: character_ref from that C (`ref_pos offset`, `mod_rank 16`, the `e1` recipe, 1,000
  steps, ~28 min) against `off_e1` on the two held-out OCs.

## Open

- Route 1's render check, and its cost in use (no `RefCache`).
- The reference at σ 0.3 with `ref_timestep 300` in place of the clean one at 0: the larger
  read and the smaller loss rise at init, untested in training.
- A panel caption for the reference: `reference image of the character` asks for the scene
  again; the reference's own caption is the candidate.
- **Out-of-range coordinates.** At 1024² a two-panel canvas spans w ±64 tokens, inside what
  Base sees at 2048²; a wide target with a wide reference goes past it.
