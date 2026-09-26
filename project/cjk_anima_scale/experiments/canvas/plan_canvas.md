# plan_canvas — does the band table hold on a ~500-token canvas? (2026-09-23; closed 2026-09-26)

> **Closed 2026-09-26 on D.0** (job `20260926-141820-f4afbc`,
> `results/20260926-1418-d0/result.json`; seed rows, no training). D.1 /
> D.2 not run — the half canvases are taken into the grid set instead
> (`recipes.GRIDS` `1x2` / `2x1`, § 2).
>
> | canvas | tokens | EN | あ\|en | あ\|swap | い\|en | い\|swap | id peak σ (move) |
> |---|---|---|---|---|---|---|---|
> | 512² (cache) | 1024 | 24/24 | 14 · 15 · 1 | 15 · 16 · 2 | 10 · 16 · 2 | 8 · 14 · 5 | 0.8 (0.155) |
> | 256×512 | 512 | 23/24 | 6 · 16 · 6 | 13 · 15 · 6 | 11 · 16 · 7 | 12 · 15 · 3 | 0.7 (0.172) ≈ 0.8 (0.170) |
> | 512×256 | 512 | 23/24 | 10 · 15 · 2 | 14 · 15 · 1 | 9 · 16 · 4 | 9 · 14 · 2 | 0.7 (0.207) ≈ 0.8 (0.203) |
> | 256² | 256 | 21/24 | 12 · 15 · 1 | 12 · 14 · 0 | 10 · 13 · 0 | 7 · 14 · 1 | 0.8 (0.180) |
>
> native = official · contained · doubled of 16. **Both half canvases pass
> the gate** (EN ≥ 22/24; the identity peak is a 0.7 / 0.8 tie). The
> identity curve widens downward (σ 0.6 move 0.005 → 0.07–0.10, 0.7
> 0.05 → 0.17–0.21) — the Gate 0 draw scales glyph px by the short side
> (55–100 px vs 110–200), and smaller px sits lower in σ, so this is the
> px row of the law, not yet a canvas term (D.1 would separate them).
> Native identity holds everywhere (contained 13–16 / 16); 256×512's
> あ|en 6 / 16 is doubling (ああ, あある, あいない… — the tall canvas asks
> for a column), 512×256 reads like 512². **H-edge is out** (a 256 short
> edge spells); 256² spells native too, so the "≈ 500 tokens" floor of
> H-tokens is if anything too high — 256² stays one EN hit under the gate
> (21/24). The 09-14 "256² dead" was the default-σ confound.

> **2026-09-26 — re-spec'd on the current tooling.** The plan
> was written against the probe line's stage surface. Since then the stages
> were vendored into `src/` and pruned (2026-09-25): `classify`, the data /
> train stages and `--data_tag` / `--arm_tag` are gone, and runs are
> `scale.py <run> …` with no flags. What changed in the cells:
> - the arm is **the seed rows** (`rows_step1_0921_merged`, every run's
>   floor), not `rows_step1_0921_s30k` / `band_s_0923/bs_hi` — the 512²
>   column then comes from reads the seed dir already holds (floor cache
>   `eval_reads.json` en 24/24, `native/` あ / い, `cf_sense_ja/` identity
>   peak 0.8), so nothing renders at 512²;
> - **identity peak σ = `cf_sense --cf_lang ja`** (`id` pairs, trained
>   cond), in place of `classify`;
> - **`single` exact → `native` あ / い × en / swap** (the seed has no cached
>   512² `single` group; `native` has one, and the rule is to read against
>   the existing floor);
> - one driver, `run_exp.py` beside this file (`--cell d0|d1`), writing to
>   `output/cjk_anima_scale/canvas_<label>/` (a `trained.pt` symlink to the
>   seed; the seed dir is never written).
> The commands in § 1 are the new ones; § 2 records what was changed.

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

The 2026-09-14 gate protocol on new canvases, on the seed rows:

```
X=project/cjk_anima_scale/experiments/canvas/run_exp.py
make daemon-run ARGS="--queue --stall-timeout 0 $X --cell d0 --label d0 \
    --canvas 256x512,512x256,256x256"
# the 0.63 rung, only if 256×512 fails:
make daemon-run ARGS="--queue --stall-timeout 0 $X --cell d0 --label d0_320 --canvas 320x512"
```

Per canvas: `eval --eval_groups en` (12 EN strings × 2 seeds, the floor
cache's captions), `native` あ / い × en / swap (8 prompts × 2 seeds), and
`cf_sense --cf_lang ja` (24 `id` + 24 `order` pairs, the Gate 0 draw — the
same pairs as the 512² read, same rng and rows; its glyph px scales with
the short side, so a 256-side canvas draws 55–100 px where 512² drew
110–200). `result.json` holds each canvas beside the 512² column.

Reads per canvas: EN control (24/24 at 384² and 512², 11/24 at 256² under
default σ), native あ / い per glyph × clause (512²: あ 14 / 15, い 10 / 8
of 16), identity peak σ (512²: 0.8, move 0.15). **Gate**: EN control
≥ 22/24 *and* the identity peak stays at 0.8. A canvas that fails the EN
control is dead for training whatever its px, as 256² was.

Also **256² under the band** (`256x256` in the same job, on the seed rows),
the cell the 09-14 report left open: if the trained rows read there while
the EN control does not, the resolution kill is about the *base* spelling,
not about the rows — a different statement from "dead".

### D.1 — the ceiling on the new canvas (cf_sense EN, ≈ 15 min)

A.1 run 1 (flat, Regular, letter + string2, px 24 … 128, σ 0.35 … 0.9,
16 pairs × 6) repeated with the canvas at 256×512 and 512×256:

```
make daemon-run ARGS="--queue --stall-timeout 0 $X --cell d1 --label d1 \
    --canvas 256x512,512x256 --d1_ref output/wake_probe/rows_step1_0921_s30k"
```

`--d1_ref` is A.1's arm dir, read-only, for the 512² column (per-px peaks
id: letter 24/32 → 0.5, 48 → 0.6, 64/96 → 0.7, 128 → 0.8; string2 one step
higher at 24–48).

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

Runs only if D.0 passes on 256×512. **Not runnable as written
(2026-09-26):** the data / train stages and the B.1 arms (`bs_lo` /
`bs_hi`, probe-line tables) are not in this line. The line's trainer is
`scale.py <run> train` on `builder` items drawn from
`config.DATA["shapes"]` (`448,512:2,448x512,512x448`) with glyph / bubble
sizes scaled by the short side, and `cjk_scale/train.py` pins
`gen_args(512, …)`. D.2 on this line = a singles run (24 S kana) built twice,
once on the default pool and once on `256x512` with absolute px, same
trainer — the builder needs a shape override and a keep-absolute-px knob,
and the comparator becomes the default-pool run, not B.1. Re-spec after
D.0 reads.

## 2. Tool changes

Done 2026-09-26 (`src/`, path plumbing only — the 512² paths are unchanged):

- `native` reads `--eval_shape` (it rendered at `--eval_size²` only;
  `target` already did), and its EN-reference cache is keyed by the canvas
  (`native_enref/<WxH>_28_4/`; 512² stays `512_28_4/`).
- `cf_sense` reads `--eval_shape` (else `--train_size²`); `meta.size` is
  `[W, H]`.
- `classify` is gone — the identity read is `cf_sense --cf_lang ja`.
- `recipes.GRIDS` gains `1x2` (256×512, top / bottom) and `2x1` (512×256,
  left / right), one 256² cell per half, for `grid_single` and
  `grid_string` alike (captions via `grid_cell_header`, no new wording).
  **Not in `builder.TABLE`** — adding them to a tier's `grids` string changes
  that tier's draws for every later build, so it goes in with the read that
  asks for it. The per-shape latent cache and the one-graph-per-token-count
  compile take the new shapes as they are.

Still owed, for D.2 only:

- A builder shape override (the pool is `config.DATA["shapes"]`). The
  renderers **scale glyph and bubble sizes by the short side** so
  glyph-to-canvas statistics match 512² (the mixed-shape instrument's
  rule). For D.2 that halves the glyph (≈ 24 px) and turns the cell into a
  small-single read, not a canvas read. Needs a keep-absolute-px knob
  (the renderers' `sc = 1`; the bubble pool's fit then decides
  how many glyphs a bubble holds) — for singles, the 48 px fit still fits
  a 256-wide canvas.

## 3. Verdict rules

| D.0 EN @ 256×512 | D.1 peaks vs 512² | D.2 winner / hits | verdict | design.md |
|---|---|---|---|---|
| ≥ 22/24 | same px → same peak | 0.7–0.9, ≥ 80 % | **H-tokens**: the table is canvas-free above ≈ 500 tokens | the single groups (and the 0.5–0.7 pieces) may build on 256×512 / 512×256 for 2× throughput; `windows.py` stays px-keyed |
| ≥ 22/24 | shifted | — | tokens are enough to spell, but the window has a canvas term | `windows.py` gains canvas fraction; D.2 re-predicted from D.1 before it runs |
| < 22/24, 320×512 passes | — | — | the floor is between 0.5 and 0.63 of 512² | 384×512 stays the cheap canvas; 320×512 becomes a candidate |
| < 22/24, 320×512 fails | — | — | **H-edge** (or the floor is ≥ 0.63) | 384 short edge is the floor; the 256 family stays closed |

## 4. Cost

D.0 ≈ 10 min a canvas (3–4 canvases), D.1 ≈ 15 min, D.2 ≈ 1 h. Under two
hours end to end; D.0 alone answers the throughput question for
`design.md` and is the only cell that has to run before a run's data is
built on the half canvas.
