# plan_synth4 — step 1 again: the seed-table recipe and the kanji budget

> **Status 2026-09-18 evening.** Step 1 is the single-glyph seed table
> (Δ1 = `rows_synth_d1_d1_s53k`, 356 rows, ΔFM, 53 k). It missed the Δ1
> gate on singles and on the JA-frame native and its next version has to
> carry more kanji (K, moved here from [`plan_synth3.md`](plan_synth3.md),
> which keeps the sentence step). Before that run is sized, the recipe has
> to be **settled from the 2026-09-18 Δ0 smokes**
> ([`reports/row_blocks_alpha_2026_09_18.md`](reports/row_blocks_alpha_2026_09_18.md)):
> row-block batching, per-block warmup, the row-norm lever (α) and where it
> is applied. This plan is the list of those decisions, each with the read
> that closes it — most of them are native-only or eval-only re-scores of
> tables that already exist. Retraining the seed table is the *last* item.

## R4.0 — geometry re-tests under ΔFM — free rows done 2026-09-18

`src/probe/table_geometry.py`, `reports/table_geometry_2026_09_18.md`:

- **No shape structure under ΔFM either** (neighbour pairs = the same-row
  control in every table) — W2d's premise holds on ΔFM tables.
- **The shared direction is per-loss, not per-run**: ΔFM ↔ ΔFM 0.66–0.87
  across datasets, inventories and sizes, plain ↔ plain 0.59–0.69,
  cross-loss 0.27–0.51. The K0 merge price for **same-loss** blocks is
  ≈ 0.7; the 0.35 that priced chunking was a cross-loss number.
- **Plain rows point 0.41 along the pack row they perturb, ΔFM rows 0.06**
  — α on a plain table scales a pack-row gain ΔFM never had, so R4.3's
  plain sweep is not a control for the ΔFM curve.
- The shared direction's share of row energy is per table: plain 26 %,
  `rb62f` 22 %, **Δ1 11 %** (PR 57 vs 12).

Two native rows are still queued behind R4.1 / R4.3:

| question | re-test | cost |
|---|---|---|
| **"lr 3e-3 → off-manifold at 2.4× row norm, table inert"** (plain FM, `findings.md`) | `rb62f` × 1.5 / × 2.0, native. Inert (singles-style collapse) → the norm boundary is loss-independent; only more wipe → the "off-manifold" was plain FM's scene-residual component, not a manifold edge | 2 × 9 min |
| **"wipe = delta norm"** (flat 0 closed) and the Δ0 "equal norm, different wipe" one-point comparison | the same α sweep (× 0.7 / × 0.4 / × 0.2) on a plain table — `pair0_s750_lr2e-3` (12 rows) and `src53k` (the seed), native. If plain's hits also rise as norm falls, the norm lever is loss-independent and part of ΔFM's no-wipe advantage was Δ1's contraction to norm 79; if plain stays flat or falls, ΔFM's direction is what makes the small row readable | 3–6 × 9 min |

## Where step 1 stands

| table | recipe | `single` | native `en` both / joint / tail | en cos |
|---|---|---|---|---|
| `src53k` | plain FM, 434 rows, mixed batches, 490 draws/row | 13/36 | 36 / 24 / 16 | 0.882 |
| Δ1 `d1_s53k` | ΔFM, 356 rows, mixed, 596 draws/row, uniform weights | 12/36 | 22 / 14 / 11 | 0.915 |

Δ1's `en` loss is co-text: the glyph is there, the base's JA pseudo-text
around it spoils the exact read (`synth_pair_delta1_2026_09_18.md`). The
Δ0 smokes then found, on 12 rows at the s750 budget:

| Δ0 arm (lr 2e-3, paired, `--box_weight 4`) | `single` /24 | `en` both / joint / tail | en cos | row norm |
|---|---|---|---|---|
| mixed batches | 13 | 22 / 20 / 7 | 0.930 | 137 |
| `--row_blocks 62` (frozen) | **16** | 17 / 12 / 9 | 0.921 | 131 |
| `--row_blocks 62` × α 0.7 (inference) | *unread* | 29 / **22** / 10 | 0.910 | 92 |
| `--row_blocks 62` × α 0.2 | *unread* | **38** / 23 / 18 | 0.875 | 26 |
| `--row_blocks 62 --lr_warmup 8` | 10 | 21 / **21** / **6** | **0.928** | 113 |

Settled from those: **`--box_weight 4` stays** (w = 1 weakens the glyph,
scene unchanged); **the row is one direction** — mean / residual
decomposition sits on the same norm curve and the shared direction is ⟂ Q,
so there is no c-style split to train; **native hits are monotone in row
norm, scene fidelity monotone the other way**, crossing ≈ × 0.7.

## What the smokes cannot tell — the reads that settle the recipe

Each item is one read; none needs a new seed-table run.

### R4.1 — does α cost the eval template? (`--stage eval`, ≈ 13 min)

`rows_synth_pair_d0_pairEN_rb62f_lr2e-3_a0p7` has its native (29 / 22 / 10)
and no eval. Its `single` against `rb62f`'s 16/24 says whether the eval
template is norm-hungry. It also decides how the w8 arm reads: w8's natives
sit *above* the norm curve (joint 21 at norm 113 where the curve gives ≈ 16)
and its singles fell 16 → 10 — if × 0.7 also lands near 10, w8 lost singles
to norm and the warmup is a clean direction win; if × 0.7 holds 16, the
warmup traded the template for the natives.

### R4.2 — per-row trajectories — read 2026-09-18 evening (report, last section)

Read: in-box loss is flat (noise-dominated), **row norm is the trajectory**;
one shape for every row (+8/step, saturating by step 32) with **ragged end
norms 80–153, row-intrinsic** (ぐ and ガ lowest in both arms). Warmup 8 only
removes the step-1 jump and costs ≈ 8 % norm. **And the rerun chaos floor is
cos ≈ 0.75 per row at one seed** (singles 16 → 14, 10 → 13) — the w8 native
gain and every other 12-row arm difference sit on it. Consequences for the
recipe table: per-row normalisation has a basis (a norm target per row
instead of a step count is the candidate), and **no schedule knob is
decided at one seed** — R4.1 and any warmup / block-length arm run with
`--seed` × 2. Per-row *loss* variance is uniform (in-box sd 0.02–0.04 for
every row, out-box 5e-4) and says nothing; the row difference is the
displacement rate (steps 9–32: ぐ 1.2, ガ 1.6, the rest 2.2–2.9 norm/step),
i.e. step-to-step gradient-direction consistency, which the loss scalar
cannot show. ぐ is the chaotic case (rerun cos 0.47), ガ the stable small one
(0.90) — "a small natural norm" is a real per-row property. **Log
`cos(Δraw_t, Δraw_{t−1})` per step** (one line in the block log) to read it
directly next time. Instruments: `row_blocks_log.jsonl` (every step: in-box
/ out-of-box residual, row norm, lr per row) and
`src/probe/row_blocks_plot.py` (small multiples, two arms overlaid).

### R4.3 — α on the 53 k table (native only, 2 × 9 min)

`rows_synth_d1_d1_s53k` × 0.5 and × 0.7, `--native_chars あ,か,す,日
--native_clauses en,swap` (the Δ1 read). Δ1's table already contracted under
cosine (161 → 79) and still lost `en` 22 / 14 / 11 vs plain's 36 / 24 / 16; if
× 0.7 recovers `en` joint toward 24 without the tail climbing past 16 the
Δ1 miss was part norm and **α is a deployment knob on the existing table**
(the LoRA-multiplier slot: train at full norm, ship at α). If it does not
move on 356 rows, the Δ0 curve was a 12-row artefact and α is dropped.

### R4.5 — is the row bound to glyph size? (free read first, then one data arm)

> **Floor measured 2026-09-18 (`src/probe/vae_glyph_floor.py`, job
> `20260918-174805-50ea9f`, `output/wake_probe/vae_glyph_floor/`):** the VAE
> round trip is not the size bottleneck. Kana (`どうしたんだよー`, yoko, two
> fonts) read identically before and after from **10 px**; every synth
> condition is CER 0 at 16 px; the three real corpus crops at 15–17 px
> (`channel_(caststation)`, tate and yoko) come back unchanged (0.06 / 0 /
> 0). The kanji string (`くっそ鬱雑えわ`) reads wrong at 8–12 px *before* the
> VAE (髪雅 / 霰雑 / 懲雑 for 鬱雑) — that is the readers' floor, not the
> latent's. **Decision (user, 2026-09-18): the size arm's minimum glyph is
> 12 px** — `--scene_min_glyph` / `--sentence_min_glyph` 12 in the data
> build, the jitter drawn between 12 px and the bubble fit, so the
> distribution spans the corpus's 15–20 px dialogue down to the VAE floor
> instead of sitting at p50 51 px. Two consequences to carry: (1) the
> **data distribution shifts** — `region_capacity` at 12 px lets far more
> multi-line sentences into small bubbles and drops the p50 glyph size,
> so the size bins (below) must be re-binned on the new build and the
> count/size confound loosens on its own; (2) **12–16 px kanji sit under
> the readers**, so the per-bin native ruler is kana-only there, or each
> bin carries its own "before" read as a control (the floor probe's
> before → after column). 12 px is 1.5 latent px at f8 — whether σ 0.7–0.9
> trains anything at that scale is the band-floor lever (second lever
> below), now coupled to the size arm rather than after it.

User's question, 2026-09-18: glyph size has never been varied on purpose.
`fit_text` (`src/common/render/scene.py`) takes the **largest font that
fits the bubble × fill** — the size is a deterministic function of the
bubble, the glyph count and `--scene_fill` (0.7 single / short, 0.9
sentence); the only variation is which bubble the scene draw lands on and
the ±1.5× canvas mix. Δ1's glyph boxes (10 k items, 99 % single): short side
**p10 39 / p50 51 / p90 78 px** at 512-class canvases, i.e. 5–10 latent px,
always at a fixed 70 % of the bubble; multi-glyph texts shrink with the
count (20–28 px floors), so size and count are confounded in `short` /
`sentence`. Under the 0.7–0.9 band only low-frequency structure trains —
silhouette, extent, ink mass at an absolute latent scale — so a row may
well encode "this blob at this size" rather than a size-free identity.
Evidence already consistent with that, never read as size: flat-trained
(large, centred) rows render 0/64 in bubbles, composite-trained rows
render the large flat template only half (Δ1 `single` 12/36 vs native
20–38/64); 512²-trained rows lose 25 → 21/36 at 384². The deployment
exposure is the target stage's 768×1344 canvas, where the base's bubbles
put the glyph at a different absolute latent size than anything trained.

- **Free read (first).** `native_reads.json` keeps every detector box with
  its read (`reads[].box`, `whole=False` rows). For Δ1, `rb62f`, the α sweep
  and `src53k`: bin the *hit* box's short side (the box whose sfx read
  contains the glyph) and the miss renders' JA-box sizes, hit rate per bin,
  against the training distribution above. Hits collapsing outside
  39–78 px = binding confirmed; flat hit rate across bins = not the lever.
  One probe, no GPU; not written yet. Also the eval template's
  glyph size vs native's — R4.1's question is partly this one.
- **If bound, one data arm (Δ0-sized, ≈ 15 min + native):** size
  augmentation inside the same recipe — a random shrink factor on
  `fit_text`'s size (`--scene_size_jitter`, drawn so the glyph lands
  anywhere in **[12 px, bubble fit]**) or `--scene_fill` drawn from a
  range, with `--scene_min_glyph 12`; if the arm is paired the sibling
  takes the same fit, so the pair stays pixel-identical outside the box
  (plain after S2a — `plan_synth3.md`). Rulers: native hits by size
  bin (above), `single`, en cos. Second lever if the first is flat: the band
  floor 0.7 → 0.5 (stroke structure in; the strings arm's band), priced
  against the singles band result. Third: target-canvas shapes in
  `--shapes`.
- **Not this item:** a scale-invariant encoder or any shape prior — the
  W2d verdict; this is exposure, the data shows more sizes or it does not.

### R4.4 — the recipe decision

After R4.1–R4.3, fix the K1 recipe line by line:

| knob | candidate | decided by |
|---|---|---|
| batching | `--row_blocks N` (rows-only, single-glyph) | settled: best singles at equal draws; keep |
| block length N | draws/row ÷ 4 (Δ1: 596 → N ≈ 150), or a per-row norm target | R4.2: norm saturates by step ≈ 32 of 62 under cosine, end norms ragged (80–153) — a norm-target stop is the candidate; sized on ≥ 2 seeds |
| per-block warmup | `--lr_warmup 8` | R4.1 + R4.2, **2 seeds** (one-seed difference is inside the cos ≈ 0.75 rerun floor) |
| lr | 2e-3 (∫lr sets the norm; 3e-3 closed on plain FM only) | unchanged unless R4.2 shows early-block blow-up |
| `--free_residual` μ | 1e-3 | unchanged — norm is handled at α, not in training |
| row norm | full in training, α at inference | R4.3 |
| `--box_weight` | 4 | settled |
| glyph size | jitter over [12 px, bubble fit], `--scene_min_glyph 12` | user 2026-09-18 after the VAE floor probe (round trip clean from 10 px kana / 16 px kanji, corpus 15–17 px unchanged); R4.5 sizes the arm, the free read decides whether the row is size-bound |
| loss | ΔFM (`--pair_loss 1 --pair_ref en`) | settled for singles (tail 15 vs 24); **S2a 2026-09-18: ΔFM loses to plain on sentences** (`reports/synth_s2a_2026_09_18.md`) — the seed table's loss is a singles question only, and the smoke's floor-row damage (katakana) is now a K1 recipe item for ΔFM |
| unit weights | uniform `*1` | settled by Δ1's exposure read (kanji ≥ kana per draw) |

**Warm starts from Δ1 convert row units** — `raw` is in units of the run's
own `row_scale` (Δ1's is 232.9 against ≈ 197 for every other table) and
`_init_rows_one` rescales by `src_row_scale / row_scale` since 2026-09-18
(`plan_synth3.md` S2a). Warm starts before that were all inside one
inventory family, ratio 0.996–1.001, so no recorded result moves.

A recipe that changes batching, warmup and unit count at once is not
comparable to Δ1. **K1 therefore carries Δ1's inventory as a subset** (the
356 rows at the same draws/row) so its `single` / `single_ext` /
`single_kanji` on those rows read against Δ1 directly, and the new kanji
ranks are the only new thing.

## K — the kanji budget (moved from `plan_synth3.md` unchanged, 2026-09-18)

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

### K0 — is a merged table a table? (a) read, (b) parked

The budget above wants the shortcut `plan_synth2.md` parked in *Chunking*:
train disjoint row blocks and union them by ext id (`src/probe/merge_tables.py`
already does this with the per-run `row_scale` correction; the shipped
`merge_punct` table is exactly this). That would make kanji scaling parallel in
wall-clock instead of linear.

**(a) is read** — `reports/table_geometry_2026_09_18.md`. The price of a
merge is the cosine between the runs' shared directions, and it is set by
the **loss**, not the run: same-loss blocks agree at 0.59–0.87 (the shipped
`merge_punct` is 0.590), cross-loss pairs at 0.27–0.51, and Δ1 ↔ punct-only
at **0.043**. So chunking within one loss merges at ≈ 0.7; the 0.35 that
looked like the price was a plain-vs-ΔFM number.

**(b) is parked.** Δ1 already trains the punct rows, so `merge_tables.py`
Δ1 ⊕ punct-only is a cross-loss *override* at the floor (0.04), not the
same-loss chunking K1 would use; the merged table exists
(`rows_synth_d1_merge_punct`) and its eval was cancelled (user,
2026-09-18). A same-loss K0(b) needs a punct-only block in the same loss as
its base (≈ 3 k steps), and it decides whether K is one long run or two or
three parallel-in-time ones.

### K1 — the scaled table

Recipe = Δ1's, with weights set by R's per-type spread and `kanji:400` or
`:600`. **Do not raise the kanji weight above what R measures**: the 53 k run's
kanji 18/36 at weight 2 was read as "kanji is fine" and it was an exposure
artefact; if Δ1 shows kanji at parity with kana at weight 1, the extra draws
should go to `kana_ext`/katakana instead (1/12 at equal exposure in the 53 k
run, the standing miss).

- **Gate:** `single_kanji` on the *new* rows (frequency ranks 200–600, which
  are rarer and were never evaluated) not below Δ1's on ranks 1–200, at equal
  draws/row. Held-out kanji stay 0 by construction (addresses do not compose,
  `findings.md`) — do not read that as a failure.
- **Guard:** rows in ≈ 0.35 % of batches at 755 rows (Δ1: 1.1 %, Δ0: 8 %).
  AdamW β₂ 0.99 decays `v` between visits; `delta_norm_mean` per draw against
  Δ1's curve over the first few thousand steps is the early read, as in Δ1.
- **Guard:** the 53 k run's word rows sat at norm 0.12 because a weighted draw
  is not a quota. At 755 rows check the items-per-row histogram in the data
  log before training, not after.


## Order

R4.0's free rows and R4.2 are read; K0(a) is read and K0(b) is parked.
Still to run: R4.0's two native rows with **R4.1** and **R4.3**, all
re-scores of existing tables (≈ 1.5 h of GPU together), and **R4.5's free
read** (size bins on the existing natives, no GPU), which decides whether a
size-jitter arm joins the recipe table. Then R4.4 is filled in. K1 runs
after S2b (`plan_synth3.md`), on the recipe above, not Δ1's.

## Open risks

- **Glyph size was never varied** (R4.5). If the rows are size-bound, every
  native number above is a number at the base's bubble size on a 512-class
  canvas, and the 768×1344 target stage reads on a size the table never
  saw; K1 would then need the size lever in its recipe, not after it. The
  floor is known (VAE clean from 10–16 px, min 12 px decided); what is not
  is whether the σ band trains a 1.5-latent-px glyph, and the readers
  cannot referee kanji under 16 px.

- **Δ0 is 12 rows.** Every recipe number above comes from 12 dakuten rows at
  744 steps, and Δ0's 12 rows never showed the `en`-clause loss that Δ1's
  356 did. R4.3 is the only item that reads on the full table; the batching
  and warmup choices are 12-row results until K1 confirms them on its
  Δ1-subset rows.
- **Eval and native want different norms.** If R4.1 shows the template
  needs full norm while natives want × 0.7, the two rulers stop agreeing on
  one table and the artefact ships with α as a user knob — that is a
  `deploy_plan.md` change, not a training one.
- **Row blocks do not fit multi-row items.** A sentence pass on a
  row-block seed table goes back to mixed batches; whether the seed's
  directions survive that (the anchor sweep's question, in a new setting)
  is S2b's `warm_cos` read.
- **n.** 24 singles, 64 natives, 2 seeds: differences under ≈ 3 are noise.
  The α curve is read as monotone across four points and four glyphs, not
  from any one cell.

## Not this plan

- **The sentence step.** `plan_synth3.md` S2 — including whether ΔFM
  reaches multi-glyph items at all.
- **c / Q separation.** Closed 2026-09-18: the decomposition sits on the
  norm curve, the shared direction is orthogonal to Q on every frame.
- **`--box_weight` ≠ 4**, **lr 1e-3 as a norm lever** (same ∫lr, less
  converged direction — α does the same thing on a converged one).
- **A bigger `--n_items` build**, **encoder / composition / transplant
  shortcuts**, **kana reference / contrastive ΔFM / cached Jacobians /
  OCR-reward rows**, **jōyō 2 136 in one run** — unchanged from
  `plan_synth3.md`.
