# plan_synth4 — step 1 again: the seed-table recipe and the kanji budget

> **Status 2026-09-19.** Step 1 is the single-glyph seed table. Every Δ0
> recipe smoke of 2026-09-18 (row blocks, per-block warmup, the α sweep) and
> Δ1 itself trained on the baked preview pack (`plan_synth3.md`, caveat at
> the top); those arm dirs are deleted and the reports are their only
> record. The recipe is re-settled here from the **raw-pack** Δ0 re-run
> ([`reports/s2b_and_raw_pack_rerun_2026_09_19.md`](reports/s2b_and_raw_pack_rerun_2026_09_19.md)).
> Running: `step1_0919` — Δ1's 53 k ΔFM recipe on the raw pack with
> `--units small` (373 rows), job `20260919-100716-22fdca`, arm
> `rows_step1_0919_s53k`. Every launch states its pack
> (`ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack` = raw).

## Where step 1 stands

| table | pack | recipe | `single` | native `en` both / joint / tail | en cos |
|---|---|---|---|---|---|
| `src53k` | raw | plain FM, 434 rows, mixed batches, 490 draws/row | 13/36 | 36 / 24 / 16 | 0.882 |
| Δ1 `d1_s53k` (deleted) | preview | ΔFM residual on the plain sentence table, 356 rows | 12/36 | 22 / 14 / 11 | 0.915 |
| `step1_0919` | raw | ΔFM, 373 rows, 53 k | *running* | | |

Δ0 on the raw pack (12 dakuten rows, 1 500 steps, paired, σ 0.7–0.9, μ 1e-3,
`--box_weight 4` unless stated; floor 0/24):

| arm | `single` /24 | end row norm | native `en` both / joint / tail | en cos |
|---|---|---|---|---|
| plain lr 1e-3 | 7 | 114 | 13 / 10 / 9 | 0.906 |
| ΔFM lr 2e-3 | 10 | 122 | 10 / 8 / 8 | 0.926 |
| **ΔFM lr 5e-3** | **13** | 144 | **14 / 11 / 10** | 0.912 |
| ΔFM lr 2e-3 `--row_blocks 62` (744 steps) | 0 | 116 | — | — |
| ΔFM lr 5e-3 `--row_blocks 62` (744 steps) | 4 | 248 | — | — |
| ΔFM lr 2e-3 `--box_weight 1` | 1 | 68 | — | — |
| ΔFM lr 2e-3, size jitter min 12 px (`d0sz`) | 0 | 70 | — | — |
| ΔFM lr 2e-3, size jitter min 16 px (`d0sz16`) | 1 | 73 | — | — |

Closed, with the record in the report named:

- **Row blocks do not wake a cold row** (0–4/24 on raw; the 16/24 was
  fine-tuning rows the preview pack had already trained). With them go the
  per-block warmup, the block-length question and the per-row trajectory
  read (`reports/row_blocks_alpha_2026_09_18.md`, last two sections). What
  survives from that read: the rerun chaos floor on 12 rows is cos ≈ 0.75
  per row at one seed, so **no schedule knob is decided at one seed**.
- **Geometry under ΔFM** (`reports/table_geometry_2026_09_18.md`): no shape
  structure in any table; a shape neighbour is no better a warm start than
  `--init_anchor μ`. Its "shared direction is per-loss" read used Δ1 as the
  ΔFM table and is unmeasured on a from-scratch one until `step1_0919`.
- **The α re-scores** (old R4.1, R4.3, R4.0's two native rows) pointed at
  preview-pack arms that no longer exist. The α question is re-posed below
  on tables that do.
- **c / Q separation** — the decomposition sits on the norm curve.

## R4.5 — glyph size: the first arm measured the box weight, not size

**Floor (2026-09-18, `src/probe/vae_glyph_floor.py`,
`output/wake_probe/vae_glyph_floor/`):** the VAE round trip is clean from
10 px kana / 16 px kanji and on real 15–17 px corpus crops; kanji at
8–12 px misread *before* the VAE (the readers' floor). Minimum glyph 12 px
(user, 2026-09-18). Under 16 px the native ruler is kana-only.

**Why the item exists:** `fit_text` takes the largest font that fits the
bubble, so the training glyph sat at p10 39 / p50 51 / p90 78 px and was
never varied; the target stage's 768×1344 canvas puts the base's bubbles at
other absolute sizes.

**The size arms (2026-09-19)** — `--scene_size_jitter` (log-uniform fill
scale in [j, 1]), builds `data_synth_pair_d0sz` (min 12, j 0.2: short side
p05 17 / p50 30 / p90 51 px) and `d0sz16` (min 16, j 0.3: 20 / 34 / 53)
against `d0`'s 38 / 54 / 80 — read 0/24 and 1/24 against 10/24. **That is
not yet a size result.** `weighted_fm_loss` (`src/train/stage.py`) weights
the in-box cells 4 and divides by the weight sum over the whole canvas, so
the row's share of the loss is proportional to the box *area*, and the
jitter cut it as far as dropping the box weight does (measured on each
build's `train.jsonl`, ≈ 4 032-cell canvases):

| data / arm | box cells p50 | in-box share of the loss | end norm | `single` /24 |
|---|---|---|---|---|
| `d0`, w 4 | 64 | 6.83 % | 122 | 10 |
| `d0sz16`, w 4 | 30 | 3.27 % | 73 | 1 |
| `d0sz`, w 4 | 25 | 2.87 % | 70 | 0 |
| `d0`, w 1 | 64 | 1.84 % | 68 | 1 |

And the low norm is **not unfinished travel**: in every raw arm the row norm
peaks by step ≈ 400–700 and then falls under the cosine schedule
(`train_log.json`, `delta_norm_mean`):

| step | `d0` w 4 | `d0sz` | `d0` w 1 | `d0` lr 5e-3 |
|---|---|---|---|---|
| 100 | 93 | 75 | 75 | 145 |
| 400 | 130 | 101 | 83 | 189 |
| 800 | 134 | 81 | 82 | 169 |
| 1 500 | 122 | 70 | 68 | 144 |

Weight decay is 0, so the only pull toward 0 is `--free_residual` μ‖f‖²,
which does not scale with the box. The end norm is where the in-box
gradient balances μ; a smaller box lowers that point. More steps on the
same recipe cannot recover it (the report's "travel-matched arm with more
steps" is dropped; lr 5e-3 peaks higher and decays the same way).

Still confounded with the share, and not separated by these arms: a 30 px
glyph is ≈ 4×4 latent cells ≈ 2×2 DiT tokens (p05 17 px ≈ one token) and the
dakuten that separates が from か is under one cell at σ 0.7–0.9; and the
`single` template asks for a large glyph, which may be the wrong ruler for
small-trained rows (the size arms' natives are unread).

- **Next arm (one, ≈ 16 min + eval):** `d0sz` at `--box_weight 12` — in-box
  share ≈ 7.0 %, matched to `d0`'s 6.83 % — ΔFM lr 2e-3, 1 500 steps, raw
  pack. Norm back to ≈ 120 and `single` ≈ 10/24 → the jitter arms measured
  the normalisation and size is still unread; norm back but `single` ≈ 0 →
  resolution or the ruler, read the natives by size bin next (kana only
  under 16 px).
- **If it is the share, fix the loss, not the flag:** make the in-box term
  area-independent — mean over in-box cells and mean over the rest,
  combined at a fixed ratio — so glyph size, glyph count (`short` /
  `sentence` boxes are larger) and canvas shape stop moving the row's
  effective weight. Until then every box-weight and norm number in this
  line is a number at `d0`'s ≈ 64-cell box. The same coupling means
  μ 1e-3 is calibrated to that box too.
- **The free read** (hit rate by detector-box size on existing natives) —
  read 2026-09-19 on `src53k` and `step1_0919`, in R4.6.
- Later levers, unchanged: the band floor 0.7 → 0.5, target-canvas shapes
  in `--shapes`. Not this item: a scale-invariant encoder or any shape
  prior (the W2d verdict).

## R4.6 — glyph size needs scenes that fit it, not a smaller glyph in the same bubble

**The size-binding hypothesis** (user): the rows train at σ 0.7–0.9, where
the sampler is deciding layout and scale, on glyphs that were never varied
(short side p05 36 / p50 51 / p95 88 px on both full tables' data), so a row
may carry "a large glyph" together with its identity.

**The free read supports it (2026-09-19, no GPU).** Every detected text box
of the two raw natives (`native_reads.json`, あ か す 日 × `en,swap`, 128
renders each), binned by approximate glyph size; a hit = both readers exact:

| glyph size | `src53k` boxes / 1-glyph / hits | `step1_0919` boxes / 1-glyph / hits |
|---|---|---|
| < 24 px | 12 / 8 / 0 | 32 / 7 / 0 |
| 24–40 px | 37 / 0 / 0 | 68 / 0 / 0 |
| 40–64 px | 40 / 12 / 7 | 36 / 3 / 2 |
| 64–96 px | 17 / 10 / 7 | 30 / 4 / 0 |
| ≥ 96 px | 45 / 40 / 28 | 28 / 15 / 8 |

No hit under 40 px on either table (0 of 149 boxes), and the 24–40 px bin
holds no single-glyph box at all: when the base lays out small text it
writes its own multi-glyph pseudo-text and the row's glyph appears only
where the layout is one large glyph — most hits are *above* the training
p95. The readers are not the limit here (VAE / reader floor is 10 px kana,
16 px kanji). Not separated by this read: the base's habit of filling a
small box with several glyphs whatever the row says.

**Why `--scene_size_jitter` is the wrong tool** (user, 2026-09-19): it
shrinks the glyph inside the bubble the scene already has, and the pools
have no small bubble to put it in — region short side p05 / p50 / p95 is
57 / 76 / 145 px on `s1`, 57 / 76 / 133 on `s1w`, 59 / 83 / 133 on `sl1w`,
42 / 62 / 99 on `ja_comic`; **no region under 40 px in 1 118 scenes** (the
scenes are 512-class renders of a one-word EN anchor). A 17–30 px glyph in
a 76 px bubble is an image the base never draws, it teaches "small glyph ⇒
mostly empty bubble", and it is the arm that also cut the in-box share
(R4.5). The glyph should be small because the *bubble* is small.

**What the build has to keep** (user, 2026-09-19): the S line exists
because the scene is the base's own output under a caption that explains
all of it (`synth.md`, *The idea*) — so the FM residual outside the bubble
is ≈ 0 and the rows are fitted to the glyph. The rows are the only free
parameters: any systematic residual the page carries is theirs to absorb,
and under plain FM that is the sign-consistent part Adam travels on (the
wipe / m̂, `plan_synth2.md`). Two builds fail that test and are **not** run:

- **Scenes rendered at k × and downscaled** — cost ≈ k² in generation
  (1.5 × is already 2 ×) and 12 px from a 51 px fit is k ≈ 4.
- **n × n panel pages from the existing pools** (written here for an hour on
  2026-09-19) — free and share-preserving, but a page of downscaled panels
  with gutters is not a base output and one panel's tags do not describe
  it: exactly the off-manifold paste the S line replaced
  (`plan_synth.md`, *Not this plan*). Only ΔFM's sibling would cancel it,
  and ΔFM is the weak loss (`reports/step1_0919_2026_09_19.md`).

**The build — make the base draw small, at today's generation cost.** The
base does write small: 149 of the 345 native text boxes above are under
40 px. The pools lack small bubbles because of how they are asked for (one
short EN word, `solo`-class prompts, 512 canvas), so the lever is the
`scenes` prompt and the judge, not the compositor:

1. **Pool smoke (GPU, 512-class, ≈ the `s1w` run's cost per scene):** a few
   hundred scenes per lever, read off `scenes.jsonl` region / letter sizes —
   (a) layout tags that shrink the bubble (`comic` / `4koma` / `multiple
   speech bubbles`, `full body` / `wide shot`, `chibi`); (b) anchors that
   shrink it (`!` `?` `…` `a` `I` — Latin / punctuation pieces, no ext
   row); (c) longer EN anchors (a phrase), which the base letters smaller.
   Number to read: the share of kept scenes with a region under 40 px and
   under 24 px, and the judge's yield there (tiny anchors must still read).
2. **Fit to the anchor's own letter size**, not the largest font the region
   takes: the detector box gives the erased text's letter height, and the
   swap draws at that size (± a small jitter). A single glyph then sits in
   a bubble the base drew for something that small; multi-glyph items from
   (c) get small glyphs with a naturally full bubble.
3. If (1) finds nothing under ≈ 24 px, 12 px is not reachable in-domain on
   a 512-class canvas and the remaining route is the training canvas
   itself (target-shape `--shapes`, R4.5 *later levers*) — at its it/s cost.

**Depends on R4.5's loss fix.** A 26 px glyph is ≈ 16 latent cells against
`d0`'s 64; under today's `weighted_fm_loss` that is the 0–1/24 regime
whatever the scene looks like.

**Arms (12-row `d0` recipe, raw pack, 1 500 steps, on R4.5's loss):** `d0`
vs `d0s` (the small-bubble pool mixed with `s1` / `s1w`). Rulers: `single`
/24 as the does-it-still-learn check, and the read that decides it — native
hits **by glyph-size bin** (the table above, probe to be written as
`src/probe/native_by_size.py`). Pass = hits appear under 40 px without
losing the ≥ 64 px bins; if `d0s` reads like `d0` in every bin, size
binding is not what holds native down and the lever is dropped. Open: a
12 px glyph is under one DiT token (16 px of canvas), so whether σ 0.7–0.9
trains it at all is R4.5's resolution confound — the band floor 0.7 → 0.5
is the second arm only if `d0s` learns on `single` and still misses the
small bins.


## R4.3 — α as a deployment knob (native only, 9 min per point)

The preview-pack α sweep read native hits monotone in row norm and scene
fidelity monotone the other way, crossing ≈ × 0.7. Re-read on raw tables:
`step1_0919` × 0.5 / × 0.7 when it lands (`--native_chars あ,か,す,日
--native_clauses en,swap`) and the same on plain `src53k`. If `en` joint
rises without the tail climbing, α ships as the LoRA-multiplier slot; if
neither table moves, the curve was a 12-row preview-pack artefact and α is
dropped. One `--stage eval` at × 0.7 on the same table says whether the
`single` template is norm-hungry (old R4.1).

## R4.4 — the recipe decision

| knob | value | state |
|---|---|---|
| batching | mixed | row blocks closed on the raw pack |
| lr | **5e-3** under ΔFM | 13 vs 10/24 on raw, no collapse at norm 144 (the 3e-3 off-manifold finding was plain FM); ≥ 2 seeds before K1 |
| `--free_residual` μ | 1e-3 | sets the end norm together with the in-box share (R4.5) — revisit with the loss fix, not alone |
| `--box_weight` | 4 at `d0`'s box size | w 1 reads 1/24 on raw; the weight is area-coupled (R4.5) |
| glyph size | jitter over [12 px, bubble fit] | undecided — R4.5's share-matched arm first |
| row norm | full in training, α at inference | R4.3 |
| loss | ΔFM for singles (`--pair_loss 1 --pair_ref en`) | raw Δ0: ΔFM 10 vs plain 7, scene held; plain on sentences (S2a) |
| unit weights | uniform `*1` | Δ1's exposure read (kanji ≥ kana per draw), preview pack — re-read on `step1_0919` |

**Warm starts convert row units** — `raw` is in units of the run's own
`row_scale` and `_init_rows_one` rescales by `src_row_scale / row_scale`
since 2026-09-18.

**K1 carries `step1_0919`'s inventory as a subset** at the same draws/row,
so its `single` / `single_ext` / `single_kanji` on those rows read against
that table directly and the new kanji ranks are the only new thing.

## K — the kanji budget

**What "more steps per row" costs.** Δ1's rate is 597 draws/row at 355 rows /
53 k steps / batch 4; the exposure curve (`plan_synth.md`) reads 1 330 / 670 /
490 draws → 100 / 75 / 36 % of singles.

| kanji rows | total rows | at 597 draws/row | at 1 194 (×2 kanji) |
|---|---|---|---|
| 200 (Δ1) | 355 | 53 k steps, 8.7 h | 83 k, 13.6 h |
| 400 | 555 | 83 k, 13.6 h | 143 k, 23.4 h |
| 600 | 755 | 113 k, 18.5 h | 203 k, 33.4 h |
| 1 000 | 1 155 | 172 k, 28.4 h | 322 k, 53 h |

**The inventory ceiling is real and close.** `kanji:N` is corpus frequency over
the manga109s bubbles, and there are only **1 037** distinct kanji that are one
Qwen piece with a pack row: top-200 covers 68.3 % of corpus kanji tokens,
top-400 84.3 %, top-600 92.6 %, top-1000 99.5 %. The *pack* holds **8 501**
single-kanji rows, so jōyō 2 136 is addressable — but not through `kanji:N`;
it needs a `jouyou` unit kind or a `list:` file. Decide which target the line
is scaling to before sizing a run: **corpus 600 (92.6 % coverage, 18.5 h)** is
the cheap complete-looking point; jōyō is a different piece of work.

### K0 — is a merged table a table? parked

Training disjoint row blocks and unioning them by ext id
(`src/probe/merge_tables.py`, per-run `row_scale` correction; the shipped
`merge_punct` table is this) would make kanji scaling parallel in
wall-clock. The price of a merge is the cosine between the runs' shared
directions: same-loss blocks agreed at 0.59–0.87
(`reports/table_geometry_2026_09_18.md`; the ΔFM side of that read was
Δ1, a residual — re-read with `step1_0919`). The test itself — a
punct-only block in the same loss as its base (≈ 3 k steps), merged and
evaluated — is not run; it decides whether K is one long run or two or
three parallel-in-time ones.

### K1 — the scaled table

Recipe = R4.4's, with weights set by R's per-type spread and `kanji:400` or
`:600`. **Do not raise the kanji weight above what R measures**: the 53 k run's
kanji 18/36 at weight 2 was read as "kanji is fine" and it was an exposure
artefact; if `step1_0919` shows kanji at parity with kana at weight 1 (Δ1
did, on the preview pack), the extra draws
should go to `kana_ext`/katakana instead (1/12 at equal exposure in the 53 k
run, the standing miss).

- **Gate:** `single_kanji` on the *new* rows (frequency ranks 200–600, which
  are rarer and were never evaluated) not below `step1_0919`'s on ranks
  1–200, at equal
  draws/row. Held-out kanji stay 0 by construction (addresses do not compose,
  `findings.md`) — do not read that as a failure.
- **Guard:** rows in ≈ 0.35 % of batches at 755 rows (Δ1: 1.1 %, Δ0: 8 %).
  AdamW β₂ 0.99 decays `v` between visits; `delta_norm_mean` per draw against
  `step1_0919`'s curve over the first few thousand steps is the early read.
- **Guard:** the 53 k run's word rows sat at norm 0.12 because a weighted draw
  is not a quota. At 755 rows check the items-per-row histogram in the data
  log before training, not after.

## Order

1. R4.5's share-matched arm (`d0sz`, `--box_weight 12`, raw pack).
2. `step1_0919` lands → its eval / native, then R4.3's α points on it and
   on `src53k`, and R4.5's free read.
3. If (1) says share: the area-independent in-box loss, re-run `d0` and
   `d0sz` on it, then μ.
4. R4.6 on that loss: the small-bubble pool smoke, anchor-size fit, `d0s` vs
   `d0`, native by size bin.
5. R4.4 filled in on ≥ 2 seeds; K1 after it.

## Open risks

- **Every recipe number is 12 rows, 24 singles, one seed** — differences
  under ≈ 3 are noise, and the 12-row rerun floor is cos ≈ 0.75 per row.
  `step1_0919` is the only read on a full table.
- **Glyph size is still unread** (R4.5, R4.6 — the free read finds no native
  hit under 40 px). If rows are size-bound, every native number is a number
  at the base's bubble size on a 512-class canvas and K1 needs the size
  lever in its recipe; whether the σ band trains a 1.5-latent-px glyph is
  open, and the readers cannot referee kanji under 16 px.
- **Eval and native may want different norms** (R4.3). Then the artefact
  ships with α as a user knob — a `deploy_plan.md` change.

## Not this plan

- **The sentence step** — `plan_synth3.md` S2.
- **Row blocks, per-block warmup, c / Q separation** — closed above.
- **`--box_weight` as a free knob** — the only open box question is the
  area coupling (R4.5).
- **A bigger `--n_items` build**, **encoder / composition / transplant
  shortcuts**, **kana reference / contrastive ΔFM / cached Jacobians /
  OCR-reward rows**, **jōyō 2 136 in one run** — unchanged from
  `plan_synth3.md`.
