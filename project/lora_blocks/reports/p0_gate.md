# P0: the reference gate on trained edit LoRAs

Date: 2026-11-07. Script: `project/moss_te_sae/src/ref_gate.py`. Phase P0 of
[proposal.md](../proposal.md): the gate G (a bias on the main-query → reference-key edges,
set from a mask, no parameters) on LoRAs we already have, inference only.

Result in one line: **G fixes the "where" failures of both shipped task LoRAs we tried,
from the outside, without touching the LoRA.** Cutting the copy inside a detector box
(`diag1`) removes the watermark sanitize v3 leaves on every seed; strengthening the copy
outside the layout boxes (`boost`) stops swap_text ksplit e2 from rewriting regions it
was not asked to. `hard` is unusable on both.

## Setup

Gate modes, for the image queries in a cell mask (16 px cells), every block, head and step,
both CFG branches:

| arm | bias on the reference keys |
|---|---|
| `ctrl` | none (stock path) |
| `hard` | −∞ on every reference key, queries in the mask |
| `soft<v>` | −v on every reference key, queries in the mask |
| `diag<k>` | −∞ on the reference keys within k cells of the query's own cell, queries in the mask |
| `boost<v>` | +v on every reference key, queries **outside** the mask |

Kernels (`--kernel`): `fast` runs `ctrl` on the stock FA2 path, `hard` / `soft` / `boost` as
two FA2 calls (main keys, reference keys) merged by LSE with the per-query bias, and `diag`
as flex attention with the cut in a score_mod, compiled at static shapes outside the block
graph (the blocks' `dynamic=True` compile cannot lower flex on torch 2.14, `FloatTrueDiv`,
and falls back to the dense math path). Against the dense SDPA reference on random tensors:
max abs error 0.008 (FA2 arms) and 0.004 (`diag`), eager and compiled. `dense` runs every
arm, `ctrl` included, through SDPA with a dense `(L, L + n_ref)` mask out of the graph.

| part | LoRA | inputs | mask | kernel | runs |
|---|---|---|---|---|---|
| edit | chinoll/Moss-Image-Edit | the two box cases of `ref_edge_mask.py`, seed 0 | hand box | dense | `output/tests/ref_gate/20261107-1732-edit` |
| sanitize | v3 (`moss_i2i_sanitize_idt_w0003_late_lr5e5_e8`) | `usual_case1`, `usual_case2`, `hard_to_sanitize1`, seeds 0-2 | AnimeText, conf 0.05, 1024 canvas, boxes + 8 px | dense | `…/20261107-1735-sanitize` (usual_case1-2), `…/20261107-1751-sanitize` (hard_to_sanitize1) |
| swap_text | ksplit e2 (`…_en400_ko1709_ksplit_rw2_mod16_ema5e5_int8_e2`) | the four eval pages, seeds 0-2 | layout boxes + 16 px (`copy_head.text_mask`) | fast | `…/20261107-1757-swap_text` |

Shipped generate settings for each task (int8 DiT, fp32 stream, ER-SDE, Spectrum, CFG 3,
skip-22 uncond). Reads, against the reference, with a one-cell ring around the mask left
out: signed ΔL and Δb (Lab) and luma L1 outside, luma L1 inside, luma L1 outside against
the same seed's `ctrl`. Sheets: `sheet_<input>.png` per input, rows = seeds; swap_text
also has `zoom_<page>.png` for the Korean pages (the layout boxes at 2x, columns src,
target edition, arms).

`hard_to_sanitize2` / `3` were not run: their AnimeText boxes cover large parts of the
artwork, the failure the edit part already shows for `hard`. The English swap_text page
`1483610_0007_en` is read by numbers here; the visual verdicts on swap_text are the user's
(2026-11-07).

## Edit LoRA: `diag` against `hard` / `soft`

Luma L1 against the reference, outside the box / inside it. `lora`, `soft*`, `hard` are
the arms of `output/tests/ref_edge_mask/20261107-1549`.

| arm | flower → blue ribbon | horns → cat ears |
|---|---|---|
| lora | 7.8 / 6.4 | 6.0 / 13.6 |
| soft2 | 9.4 / 34.1 | 5.9 / 19.4 |
| soft4 | 11.2 / 41.1 | 5.8 / 33.5 |
| hard | 11.7 / 48.6 | 5.8 / 38.2 |
| diag0 | 10.6 / 44.9 | 5.9 / 23.7 |
| diag1 | 9.2 / 50.4 | 5.9 / 30.0 |
| diag2 | 6.8 / 56.1 | 5.8 / 30.0 |

- **`diag0` is the only arm that keeps the eyes under the cat ears as drawn.** `diag1` /
  `diag2` keep eyes but redraw them; `hard` and `soft4` cover them with hair.
- **No `diag` passes both cases.** On the flower, `diag0` removes it and leaves a white hole
  (as `hard` does); `diag1` / `diag2` read the flower from the neighbouring cells inside the
  box and paint a new flower. A `diag` cut removes what is smaller than its window; an
  object larger than k cells is re-read from its own other cells.

## sanitize v3 + G from AnimeText

| page | v3 alone | + diag1 | + hard |
|---|---|---|---|
| usual_case1 (watermark, top left) | kept on all 3 seeds | **removed on all 3** | removed on 2, a grey wedge on seed 1 |
| usual_case2 (faint URL watermark, mostly not boxed) | already clean | faint red mark on seed 1 | invents objects on the board (seeds 1, 2) |
| hard_to_sanitize1 (two faint stamps) | already clean | clean; a small mark on seed 2 | invents objects (headband, beaded strap, grid patch) |

| page | arm | ΔL out | Δb out | L1 out | L1 out vs ctrl |
|---|---|---|---|---|---|
| usual_case1 | ctrl | −0.29 … −0.12 | −0.64 … −0.59 | 5.55–5.60 | — |
| | diag1 | −0.29 … −0.06 | −0.68 … −0.60 | 5.58–5.65 | 0.42–0.47 |
| | hard | −0.27 … −0.08 | −0.68 … −0.59 | 5.56–5.66 | 0.40–0.43 |
| usual_case2 | ctrl | −0.01 … +0.72 | −0.33 … −0.28 | 6.17–6.59 | — |
| | diag1 | −0.02 … +0.71 | −0.37 … −0.29 | 6.17–6.58 | 0.43–0.47 |
| | hard | +0.05 … +0.69 | −0.39 … −0.31 | 6.16–6.56 | 0.46 |
| hard_to_sanitize1 | ctrl | −0.17 … −0.05 | +0.05 … +0.09 | 6.16–6.19 | — |
| | diag1 | −0.26 … −0.23 | +0.03 … +0.15 | 6.18–6.26 | 0.76–0.86 |
| | hard | −0.32 … −0.15 | +0.02 … +0.07 | 6.25–6.26 | 0.87–0.99 |

- **`diag1` passes P0's sanitize gate on the miss we could reproduce.** The watermark v3
  keeps goes on every seed; outside the boxes ΔL / Δb stay within the seed spread of v3
  alone and the page moves 0.4-0.9 luma L1 against v3's own render.
- **`hard` invents content.** A box that cannot read the reference at all is filled with
  a new object; it is not a removal mode.
- **The faint-stamp ghost in [moss_i2i_ambitious.md](../../../docs/moss/proposal/moss_i2i_ambitious.md)
  did not reproduce**: v3 alone, on the int8 generate plan, clears both stamps of
  `hard_to_sanitize1` on seeds 0-2. On pages v3 already cleans, G adds nothing and
  `diag1` costs an occasional small mark.
- **The mask is the limit.** AnimeText at conf 0.05 boxes `usual_case1`'s watermark and
  both stamps, and misses most of `usual_case2`'s faint URL watermark.

## swap_text ksplit e2 + G from the layout boxes

| page | arm | ΔL out | Δb out | L1 out | L1 in | L1 out vs ctrl |
|---|---|---|---|---|---|---|
| 1483610_0007_en | ctrl | +0.33 … +0.47 | −0.35 … −0.33 | 5.71–6.21 | 17.3–17.6 | — |
| | diag1 | +0.37 … +0.49 | −0.33 … −0.32 | 5.68–5.89 | 30.0–33.2 | 1.10–1.65 |
| | boost2 | −0.44 … −0.38 | +0.18 … +0.21 | 5.98–5.99 | 16.8–17.9 | 3.39–3.92 |
| | boost1 | −0.00 … +0.08 | −0.06 … −0.03 | 5.57–5.60 | 16.9–17.4 | 1.99–2.60 |
| 3124789_0042_en | ctrl | +0.35 … +0.46 | −0.14 … −0.08 | 8.05–8.12 | 26.6–27.7 | — |
| | diag1 | +0.41 … +0.57 | −0.12 … −0.09 | 8.06–8.19 | 44.6–48.6 | 1.21–1.29 |
| | boost2 | −0.17 … −0.14 | +0.34 … +0.35 | 8.29–8.32 | 27.1–28.5 | 3.17–3.24 |
| | boost1 | +0.21 … +0.26 | +0.05 … +0.10 | 7.97–8.02 | 27.0–27.9 | 1.90–1.94 |
| 589883_0004_ko | ctrl | +0.72 … +0.82 | −0.72 … −0.71 | 2.68–2.82 | 16.6–16.7 | — |
| | diag1 | +0.70 … +0.81 | −0.76 … −0.71 | 2.63–2.82 | 21.4–22.5 | 0.58–0.61 |
| | boost2 | −0.16 … −0.09 | +0.76 … +0.83 | 2.60–2.66 | 16.2–16.6 | 3.06–3.27 |
| | boost1 | +0.43 … +0.47 | −0.06 … +0.05 | 2.28–2.32 | 16.2–16.6 | 1.68–1.78 |
| 737486_0019_ko | ctrl | +0.39 … +0.40 | −0.12 … −0.09 | 2.64–2.78 | 20.1–20.3 | — |
| | diag1 | +0.41 … +0.46 | −0.14 … −0.11 | 2.71–2.84 | 29.2–31.1 | 0.60–0.65 |
| | boost2 | −0.13 … −0.12 | +0.13 | 2.61–2.63 | 20.1–20.6 | 1.91–2.10 |
| | boost1 | +0.14 … +0.20 | −0.02 … +0.01 | 2.40–2.41 | 20.2–20.7 | 1.08–1.31 |

"L1 in" is against the Japanese source, so a higher value inside the boxes means less of
the source copied, not better letters.

- **`boost2` fixes ctrl's edits outside the instruction, consistently** (user, all pages,
  seeds 0-2): ctrl rewrites regions that no layout item names; `boost2` leaves them as the
  source has them. Its letters are the best of the three arms. Most of its 1.9-3.9 L1
  against ctrl outside the boxes is that correction.
- **`boost2` shifts tone.** The brightening of ctrl (ΔL +0.33 … +0.82) flips to −0.09 …
  −0.44, and Δb moves +0.1 … +0.8.
- **`boost1` has the lowest L1 outside the boxes on every page** (5.57 / 7.97 / 2.28 / 2.40
  against ctrl's 5.71 / 8.05 / 2.68 / 2.64 at best), with Δb within ±0.1 and ΔL between
  ctrl's and zero. It moves the page 1.1-2.6 L1 against ctrl, against 1.9-3.9 for `boost2`.
  Whether it keeps all of `boost2`'s correction of the unasked regions is the user's read;
  in the Korean zoom sheets its balloons and letters match ctrl and `boost2`.
- **`diag1` breaks the balloons** (user; the Korean zoom sheets show it): stray strokes at
  the box edge, a lost balloon border, an added tail. The mask runs 16 px past the bbox, so
  border cells fall inside it and, cut from their own cell, are redrawn. Outside the boxes
  it matches ctrl (ΔL, Δb within the spread; 0.6-1.6 L1 against ctrl).
- **Letters.** On both Korean pages every arm letters every line right on seeds 0-2
  (`벌거` on `589883_0004_ko` is the caption's own OCR misread of `씻겨`).

Mechanism, as read: under `ref_isolate` an image query outside the boxes splits its mass
between the image being drawn, the layout-JSON tokens and the reference. ctrl's stray edits
are those queries reading a string where no item put one. `boost` multiplies the
reference's share (e² ≈ 7.4 at +2), so the copy wins there. It also raises every reference
key, not only the own cell, which fits the tone shift and the slightly higher L1 outside.

## What it means for lora_blocks

- **G is a result on its own, as [opinion.md](../opinion.md) expected.** Both directions
  work on shipped LoRAs with no training: cut inside the region to edit (sanitize),
  reinforce outside it (swap_text). The LoRA keeps the "what"; the mask supplies the
  "where", which is where both tasks fail.
- **The mode depends on the task.** Removal wants `diag` inside a detector box; lettering
  wants `boost` outside the layout boxes and nothing inside. `hard` invents content and
  should not ship.
- **`diag` has a size rule.** It removes what fits in its window and re-reads anything
  larger from its own cells. Text strokes fit; a flower or a balloon border does not.
- **Product shape.** A mask input on the Reference node with two controls, a cut inside and
  a boost outside, is shippable before any block is trained. For swap_text the layout boxes
  are already in the prompt, so `boost` needs no extra input.

## Not tested

- Whether `boost1` keeps all of `boost2`'s correction of the unasked regions (user read pending).
- An own-cell boost (+v on the own cell ± k only, outside the mask): the copy without the
  neighbours, which may keep the correction without the tone shift. Runs on the flex path.
- `boost` on sanitize, and a cut + boost combination.
- σ schedules (strong early, relaxed late) for either mode.
- `hard_to_sanitize2` / `3`; masks tighter than boxes (text pixels, SAM3).
- Other seeds, other pages; the bf16 generate plan.
