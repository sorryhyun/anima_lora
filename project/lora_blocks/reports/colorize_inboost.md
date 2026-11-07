# colorize v5: a copy boost inside the text boxes

Date: 2026-11-07. Script: `project/moss_te_sae/src/ref_gate.py --part colorize` (daemon job
`20261107-183601-951b64`). Run: `output/tests/ref_gate/20261107-1836-colorize`
(`summary.json`, `sheet_<page>.png` with rows = seeds, `zoom_<page>_s<seed>.png` with the six
largest text boxes at 2x, columns ref, ctrl, inboost1, inboost2). Step 1 of
[plan.md](../plan.md).

Question: colorize v5 writes Korean with the right syllables and soft, uneven strokes, and
the VAE does not cause it (2026-10-27). If the strokes go soft because the page is redrawn
instead of copied, a stronger copy inside the text boxes should keep the source's strokes.

## Setup

- LoRA `output/ckpt_moss_i2i/moss_colorize_v5.safetensors`, `configs/moss/i2i/colorize/generate.toml`
  (int8 DiT, fp32 stream, ER-SDE 30, Spectrum, CFG 3), seeds 0–2.
- Pages 03 / 05 / 09 / 14 of ComfyUI's `input/to_colorize4` (one Korean work, 1359x1920,
  rendered at 864x1216): the pages without explicit content that carry the most balloon text.
- Prompt: v5's bare control caption as the cache builds it, `comic` plus a box-only `text`
  item per OCR line that `keep_line` keeps (7 / 8 / 14 / 11 lines, SFX included). The v5
  eval renders ran on plain `comic`.
- Mask: those boxes padded 4 px, any overlap marks a 16 px cell (571 / 673 / 506 / 480 cells).
- Arms: `ctrl` (stock path) and `inboost<v>`, +v on every reference key for the image
  queries inside the mask, nothing outside (two FA2 calls merged by LSE).
- New read `grad_in`: mean Sobel magnitude of the render's luma inside the mask over the
  source's. 1.0 = edges as strong as the source's.

## Reads

Inside the mask, against the source; ranges over seeds 0–2.

| page | arm | grad_in | L1 in | L1 out vs ctrl |
|---|---|---|---|---|
| 03 | ctrl | 0.93–0.94 | 13.95–14.18 | — |
| | inboost1 | 0.98–0.99 | 13.84–13.99 | 1.45–1.61 |
| | inboost2 | 1.02 | 14.29–14.45 | 1.61–1.73 |
| 05 | ctrl | 0.91–0.93 | 18.13–18.31 | — |
| | inboost1 | 0.97–0.99 | 18.05–18.30 | 1.41–1.51 |
| | inboost2 | 1.01–1.03 | 18.61–18.70 | 1.85–1.96 |
| 09 | ctrl | 0.92–0.94 | 13.94–14.07 | — |
| | inboost1 | 0.99–1.00 | 13.90–14.00 | 1.27–1.44 |
| | inboost2 | 1.04–1.06 | 14.43–14.50 | 1.45–1.64 |
| 14 | ctrl | 0.92–0.95 | 15.47–15.50 | — |
| | inboost1 | 0.99–1.02 | 15.74–15.80 | 1.16–1.23 |
| | inboost2 | 1.05–1.07 | 16.24–16.36 | 1.39–1.60 |

Outside the mask ΔL / Δb stay within ctrl's seed spread on every page.

## Claims

1. **The boost makes the strokes heavier, toward the source's weight and past it.**
   `grad_in` 0.91–0.95 → 0.97–1.02 (`inboost1`) → 1.01–1.07 (`inboost2`) on every page and
   seed. The user sees it as bolder letters.
2. **It does not make the glyphs the source's.** L1 inside the boxes does not fall: flat at
   `inboost1` (−0.1 … +0.3), up at `inboost2` (+0.3 … +0.9). The deformations stay where
   they are (user, zoom sheets): `생` in `학생처럼` on page 05 has the same stray dot under
   every arm, and the other misshapen syllables keep their shapes.
3. The boost moves the page 1.2–2.0 luma L1 against ctrl outside the mask, the spill of a
   bias set per cell on a mask of whole cells.

L1 inside includes the balloon background, which colorize may tint, so it reads glyph
fidelity only together with the sheets.

## What it means

- A stronger copy does not repair colorize's lettering. The deformation is not a shortfall
  in how much the box reads the reference; the strokes the copy delivers are redrawn the
  same way at any copy strength. The gate is the wrong tool here, and step 1 closes.
- The lever is what the DiT draws inside a glyph: the Hangul block of plan step 3, or a copy
  trained to be exact at stroke scale on texted pages.
