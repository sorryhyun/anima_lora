# plan_canvas — does the band table hold on a ~500-token canvas? (plan only, 2026-09-23)

`design.md` § 2 assumes `stage0709` trains on the 448–512 shapes. A canvas
with half the tokens would double throughput if the base still spells there
and the band table (`band_experiment_results.md`: band keyed on glyph
count, rendered px sets the floor, grid +0.1) carries over unchanged. This
plan reads that; nothing here has run.

## 0. What is known

Tokens scale with area (patch grid over the 8× latent); relative to 512²:

| canvas | tokens (÷ 512²) | state |
|---|---|---|
| 512² | 1.00 | every read of record |
| 384×512 / 512×384 | 0.75 | alive: P0a singles 34/36 at 0.82× wall (`wake_canvas_scenes_2026_09_14.md`) |
| 384² | 0.56 | alive: EN 24/24, band rows 21/36, identity peak σ 0.8 = 512² (same report) |
| 320×512 | 0.63 | unread |
| **256×512 / 512×256** | **0.50** | **unread** |
| 256² | 0.25 | dead **under default σ** (EN control 11/24, W2 arm learned layout only); the report says explicitly "256² under a band stays untested" |

So the kill has never been separated into *short edge* vs *token count*,
and no canvas between 0.25 and 0.56 has been looked at. The hypothesis to
read: **H-tokens** — the base spells iff the canvas has ≳ 500 tokens, and
the band table is keyed on absolute glyph px (tokens per glyph), so it is
canvas-free above that floor. The alternative, **H-edge**: the short edge
(≥ 384) is what the base needs, and a 256-short-edge canvas is dead at any
aspect ratio.

## 1. Cells

### D.0 — the canvas gate (eval only, no training, ≈ 10 min a canvas)

The 2026-09-14 gate protocol on new canvases, with an existing table
(`rows_step1_0921_s30k`, or `band_s_0923/bs_hi` for the 24 S rows):

```
eval     … --eval_shape 256x512 --eval_tag c256x512   (vertical text, the tall canvas)
eval     … --eval_shape 512x256 --eval_tag c512x256   (horizontal, the wide one)
eval     … --eval_shape 320x512 --eval_tag c320x512   (the 0.63 rung, only if 256×512 fails)
classify … same shapes, σ grid 0.35 … 0.95           (identity peak σ per canvas)
```

Reads per canvas: EN control (floor and trained; 24/24 at 384², 11/24 at
256²), `single` exact, identity peak σ from `classify`. **Gate**: EN
control ≥ 22/24 *and* the identity peak stays at 0.8. A canvas that fails
the EN control is dead for training whatever its px, as 256² was.

Also **256² under the band** (`--eval_shape 256x256` on the 0.7–0.9 table),
the cell the 09-14 report left open: if the trained rows read there while
the EN control does not, the resolution kill is about the *base* spelling,
not about the rows — a different statement from "dead".

### D.1 — the ceiling on the new canvas (cf_sense EN, ≈ 15 min)

A.1 run 1 (flat, Regular, letter + string2, px 24 / 32 / 48 / 64 / 96,
σ 0.35 … 0.9) repeated with the canvas at 256×512 and 512×256. Tool change:
`cf_sense` takes `--train_size` as an int side; it needs a `WxH` form
(`common.shapes.parse_shape` exists — `render_string` already takes a
`(W, H)` size).

Read: peak σ and live band per px, beside A.1's 512² column. **H-tokens
predicts the same peak at the same px** (a 48 px letter is the same tokens
per glyph on either canvas). A shift down (the canvas has fewer tokens to
spend, so the glyph is "bigger" relative to it) means the window is keyed on
glyph / canvas fraction and `windows.py` needs a canvas term. Strings that
crop at font ≥ 64 on 512² crop earlier on the 256 side — the valid string
cells are 24 / 32 only on the wide canvas.

### D.2 — the training read (2 arms, ≈ 1 h at half the tokens)

B.1 repeated on the tall canvas, same rows, same px, same bands, so the
only change from `bs_lo` / `bs_hi` is the canvas:

```
data  … --data_tag band_s_c256 --shapes 256x512 --units chars:<the 24 S kana> \
        <B.1's scene argv>  + a knob that keeps glyph px absolute (§ 2)
train … <B.1's train argv> --shapes 256x512 --t_min 0.5 --t_max 0.7 --arm_tag cs_lo
                                             --t_min 0.7 --t_max 0.9 --arm_tag cs_hi
native / cf_sense (ja, px 48) as B.1
```

Reads beside B.1 (`band_b1_2026_09_23.md`): exact 25 / 29 of 36, native en
49 / 84, swap 30 / 55, cf_sense peak 0.7 on both arms. **H-tokens
predicts the same winner (0.7–0.9) at ≥ 80 % of B.1's hits**; a canvas
that keeps the winner but loses half the hits is alive-but-worse (the
384² reading, 21 vs 25); a canvas where `cs_lo` wins says the window moved.

Runs only if D.0 passes on 256×512.

## 2. Tool changes the plan needs

- `--eval_shape` already takes `WxH`; `classify` needs the same (check).
- `cf_sense`: canvas as `WxH` (`--train_size` is an int).
- Data stage `--shapes WxH` **scales glyph and bubble sizes by the short
  side** so glyph-to-canvas statistics match 512² (the mixed-shape
  instrument's rule). For D.2 that halves the glyph (≈ 24 px) and turns the
  cell into a small-single read, not a canvas read. Needs a
  `--shape_scale 0` (keep absolute px; the bubble pool's fit then decides
  how many glyphs a bubble holds) — for singles, the 48 px fit still fits
  a 256-wide canvas.

## 3. Verdict rules

| D.0 EN @ 256×512 | D.1 peaks vs 512² | D.2 winner / hits | verdict | design.md |
|---|---|---|---|---|
| ≥ 22/24 | same px → same peak | 0.7–0.9, ≥ 80 % | **H-tokens**: the table is canvas-free above ≈ 500 tokens | `stage0709` (and 0507) may build on 256×512 / 512×256 for 2× throughput; `windows.py` stays px-keyed |
| ≥ 22/24 | shifted | — | tokens are enough to spell, but the window has a canvas term | `windows.py` gains canvas fraction; D.2 re-predicted from D.1 before it runs |
| < 22/24, 320×512 passes | — | — | the floor is between 0.5 and 0.63 of 512² | 384×512 stays the cheap canvas; 320×512 becomes a candidate |
| < 22/24, 320×512 fails | — | — | **H-edge** (or the floor is ≥ 0.63) | 384 short edge is the floor; the 256 family stays closed |

## 4. Cost

D.0 ≈ 10 min a canvas (3–4 canvases), D.1 ≈ 15 min, D.2 ≈ 1 h. Under two
hours end to end; D.0 alone answers the throughput question for
`design.md` and is the only cell that has to run before `stage0709` is
built.
