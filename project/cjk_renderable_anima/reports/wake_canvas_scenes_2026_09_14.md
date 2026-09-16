# Canvas-shape gate, table-parts probe, scenes s0, S0 build (2026-09-14 night)

> Lifted out of the former `history.md` on 2026-09-15; text unchanged. 384² is
> alive and the mixed-shape pool is the P0a recipe; the table-parts probe shows
> the glyph is conditional on the canvas mode (→ the S line); 1 000 self-generated
> scenes → 186 kept; the S0 data build and launch.
> Index: [`README.md`](README.md). Backticked paths (`src/…`, `plan_synth.md`, `output/…`) are relative to the line root `project/cjk_renderable_anima/` or the repo root, as they were in `history.md`.

## Canvas-shape gate (2026-09-14 night): 384² is alive, the band does not move, mixed shapes are an instrument

**Why it was owed.** The 256² kill (report § W2, "never below 512²") ran on
the *default* σ sampling; the σ 0.8 band was found later the same day. The
verdict "no text competence below 512²" was therefore band-confounded and
one tier too broad. Three jobs, no training:

| job | what | result |
|---|---|---|
| `20260914-151236-404750` (`rows_w24_band/eval_384/`) | the 512² band rows rendered at **384²**, singles + EN | EN **24/24 floor and trained**; singles **21/36** both readers largest box (25/36 at 512² on the same ruler). Misses: repetition (ききき そそそ すすべ), つ→っ, vl misreads of correct こ/し, ち→5 |
| `20260914-152513-e84609` (`rows_w24_band/classify_384/`) | same-noise 24-way classifier at 384² | identity peaks at **σ 0.80, top-1 0.73** (512²: 0.69); chance at 0.65 / 0.95; floor chance everywhere |
| `20260914-152918-40d794` | smoke of the mixed-shape instrument (tiny inventory, 30 steps, 384×512 eval) | per-shape latents, 4 static token families, one-shape batches, non-square renders, EN 2/2 |

**Reading.** The base spells at 384² and the rows transfer *downward*
(1024² showed upward). No σ shift exists in the stack (training draws raw
σ; inference `flow_shift` is a fixed 3.0), and none is needed: the
identity band is the same at 384² and 512², so one hard band 0.7–0.9
serves a mixed pool. 256² itself stays untested under a band; the verdict
now reads "256² under default σ", not "below 512²".

**Instrument (`wake_probe.py`).** `--shapes "384,448,512:2,384x512,512x384"`
on the data stage draws a canvas per font item (corpus crops from the
squares; one shape per balanced group), records `shape` per item, and
scales glyph / bubble sizes by the short side so the glyph-to-canvas
statistics match 512² (512² renders are bit-identical to the old code —
80/80 — so every existing data dir rebuilds unchanged). Train caches
`latents_mixed_<shapes>.pt` per shape and batches one shape per step,
each shape in proportion to its items; the block compile gets one static
graph per distinct token count (384×512 and 512×384 share). Eval takes
`--eval_shape WxH`. `_gen_args` passes (H, W) to the request.

**P0a result (same night, `encoder_wds_w120_s8k_fres_warm_shp`, jobs
`20260914-154123-df61b7` train+eval 47.8 + 12 min, `-8978a0` 384×512
eval).** Run 3's recipe on `wds` (= `wd` rebuilt with `--shapes
"384,448,512:2,384x512,512x384"`; identical inventory, 8992 items).
512²: singles **36/36 sfx / 29/36 both** (Run 3 33 / 26 on the same
ruler), word 10/32, EN 24/24; 384×512: singles **34/36 / 29/36**, word
11/32, EN 21/24 = floor (base writes "Sorry" / "OK." there). 2.79 it/s vs
2.30. Instruments identical to Run 3 through training. **Verdict: better
at cost** (plan gate) — the pool is the recipe of record; B (matched
wall) not run since cheaper-at-parity missed on wall alone (0.82×, not
0.75×). Next = P0b singles at scale (instrument owed: `--kana_ext` +
`single_ext`), then P1 strings on a 512–768 pool.

## Table-parts probe (2026-09-14 night): the glyph is conditional on the canvas mode, nothing separates

**Why it was owed.** P0b's `native` reads 32/64 but the sheets say the hits
are training canvases drawn over the scene (`plan_synth.md`). Before
re-training on scene composites: does the canvas live in the *shared*
parts of the table — the common vector `c` (at its 0.75 cap) and the
rank-1 encoder part `g` — with the glyph in the per-row `f`? If so,
warm-starting `g`/`c` from P0b would carry the canvas forever and zeroing
`c` at inference would already be a fix. The saved table splits exactly:
`raw = g + c + f` (mean row norms 0.968 / 0.750 / 0.718, full 1.574;
`g` centred across rows so `mean(g + c) = c`).

Two jobs on the P0b table, no training:

| job | what | result |
|---|---|---|
| `20260914-213200-f1225f` | scene-kept ruler on P0b's existing `native/` (64 trained images) | plain PE-cos to the floor image fails (a white bubble on black scores 0.89 against a 2-koma scene; real scenes 0.94+; seed-pair p10 0.82). **Margin rule** — cos(img, floor) − cos(img, mean feature of 64 training canvases) ≥ 0 — matches the sheets: every wipe negative (−0.33 … −0.01), every kept scene positive. P0b full table: hit 32, kept 28, **hit & kept 2/64** |
| `20260914-213229-c7e1f5` (`native_parts/`, 64 renders, 4.3 min) | `--delta_parts f,c,g,fg`, 8 prompts × す か × 1 seed, floor reused | **every part 0/16 hits.** `f`, `g`, `fg`: scene kept 16/16, text = floor garble. `c` alone: kept 11/16, and where it wins it draws the *training canvas* — big white bubble, one generic kana (readers say ん at p03 p05 p06) — never す or か |

**Reading.** The canvas mode is `c` (the "big glyph on a blank canvas"
direction the encoder docstring predicted), but the identity in `f` / `g`
does not render without it: the rows learned *glyph given the flat-canvas
layout*, and `f + g` inside a real scene is silent. No linear split of the
table separates glyph from canvas, so (i) zeroing `c` at inference is not
a fix, (ii) a warm start from P0b (or P0a) carries a conditional identity
that the composites would have to re-learn anyway, and (iii) the magnitude
axis (`--delta_scale` 0.5 → 2/64, 0.7 → 22/64 hits) and the parts axis
agree: the P0b table has exactly one working direction and it is the
training image. **Decision (user, 2026-09-14): the S line — the rows
arm from scratch on the scene-composite mix with the canvas mode moved
into a per-source `c_flat` (`plan_synth.md`, S0), no encoder, no warm
start; P0b stays as the flat-only control.**

**Instrument.** `native --delta_parts f,c,g,fg,…` renders each part of an
encoder-arm table as its own cond (`full` keeps the name `trained`);
`_read_native` now reports `floor cos | canvas cos | kept | hit & kept`
per cond, with `--kept_ref` (floor images of an earlier run when this one
is `--no_floor`) and `--kept_tau` (margin, default 0). The canvas
prototype is 64 renders from the arm's own data dir, so a synth-trained
arm measures against *its* flat share.

## Scenes s0 (2026-09-14 night): 1 000 self-generated scenes, 203 kept

**Job `20260914-221635-347544`** (62 min; ≈ 3.6 s/img batched 4 per shape
on the S0 pool `448,512:2,448x512,512x448`, 28 steps cfg 4, negative
prompt on the uncond branch). Three smokes before it (jobs `-220052`,
`-220253`, `-221018`, `-221342`, 16–32 prompts each): the first prompt
generator (plain tag bags, no artist / character, unsorted) drew a
generic off-distribution average (user's read) and 0/16 passed; rewriting
the prompt in the dataset's caption order (`rating, count, character,
copyright, @artist, generals sorted`) with dataset artists + `sincos` /
`hews` made every sheet in-domain; 2× render + downsample did nothing
(8 % vs 4 %) and is off; two anchor bubbles per two-speaker prompt are
normal (both get swapped).

**Filter, re-judged on CPU from the stored reads** (`--scene_rejudge 1`,
no GPU): 45 kept at a 96 px bar → 120 at 64 px → 203 at 56 px with the
bubble fill fixed. The fill's bug was unioning every seed's flood, so one
seed leaking through a sketchy outline (a 20 % fill under the 35 % cap)
opened the whole bubble; per-seed judging (large + border = leak, small +
border = edge-clipped bubble, largest survivor wins) took open bubbles
from 17 % to 3 %, and a 12× text-box plausibility guard removed the fills
that ran into panel-bounded backgrounds (8 % open in the end). Region
short side on the kept set: p10 67, median 83 px. The base draws the
bubble at ≈ 1/8 of the canvas whatever the framing (`full body` median
57 px), so the size bar is the product condition, not a defect.

**Rejects that stay:** stray text 289 (shirts, signs, a second garbled
bubble), read miss 240 (misspelled anchor or JA garble), region < 56 px
174, leak 77, no box 17. Every image has its prompt row
(`scenes_all.jsonl`; future runs write `prompts.jsonl` first).

## S0 build + launch (2026-09-14 night): scene composites, rows arm from scratch

**Instrument** (`src/data/synth.py`, `src/common/bubble.py`, `render_into_scene`,
train-stage `--box_weight` / `--c_flat`, eval `--with_c_flat`, native
`--delta_parts` on rows arms, scene-kept prototype = the flat share only).
Smoke on 400 items caught three data defects before the build:

- **Erase coverage** — a text-box-only erase left the anchor where the
  detector box ran tight (red "Yes" under セックス); erase = region ∪ box
  padded ¼ fixed it, and then the rectangle's corners poked past round
  outlines (user: scene 124). Final: paint only inside the bubble's flood
  interior with the letter holes filled.
- **Leaked bubbles in the kept set** (user: scene 48 — a white kitchen
  wall passed as the bubble, the swap painted a 130 × 240 px white block
  over the character). Rule added to `bubble_mask`: the fill's bbox must
  enclose its text box (¼ tolerance). `--scene_rejudge 1` on s0: **203 →
  186 kept**, all 17 drops verified leaks on the sheet.
- **Chinese-styled kanji** (user): DroidSansFallbackFull was 1 of 15 fonts;
  Noto CJK's ttc index 0 is the JP face. Droid is out of `find_fonts`
  (every S-line render; pre-S0 data dirs had it).

Also from the user: strings out of S0 (`--strings_frac 0`; flip/str3 only
when strings are in), a stroke outline on 25 % of composites. Per-glyph
floor 32 px (40 px left 82 % of composites singles; the median region
holds two glyphs). Composite kind draw is capacity-first (no wasted
renders).

**Data `synth_s0`**: 16 000 = 6 400 font + 3 200 phrase + 6 400 scene
(5 549 single / 851 phrase) over 186 scenes, 6.4 min CPU. Shapes 448²
3 130 / 448×512 3 254 / 512×448 3 071 / 512² 6 545. Eval 150 prompts (+
`phrase_held` 16).

**Train smoke** `20260914-234903-f65608` (40 steps, compile, no ckpt): 1.0
it/s at step 25 (compile warm-up; the un-compiled first try OOMed at
14.8 GB — block compile is the recipe, as ever), `leak` 0.02, `c_flat`
norm 0.12 after 25 steps. **S0 jobs**: train + eval `20260914-235607-0ac29b`,
native `20260914-235621-7e903c` (gates in `plan_synth.md`).
