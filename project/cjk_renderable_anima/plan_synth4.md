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

## R4.0 — geometry re-tests under ΔFM (first; mostly free)

The line's geometry claims were all measured on plain-FM tables. ΔFM rows
are trained on a different residual (the scene part cancelled), row-block
rows on a different optimizer path, and today's α curve is a norm result
the old claims never saw. Re-ask each on the tables that now exist before
the recipe is fixed on top of them:

| claim (where it was closed) | re-test | cost |
|---|---|---|
| **"lr 3e-3 → off-manifold at 2.4× row norm, table inert"** (plain FM, `findings.md`) | `rb62f` × 1.5 / × 2.0, native. Inert (singles-style collapse) → the norm boundary is loss-independent; only more wipe → the "off-manifold" was plain FM's scene-residual component, not a manifold edge | 2 × 9 min |
| **"wipe = delta norm"** (flat 0 closed) and the Δ0 "equal norm, different wipe" one-point comparison | the same α sweep (× 0.7 / × 0.4 / × 0.2) on a plain table — `pair0_s750_lr2e-3` (12 rows) and `src53k` (the seed), native. If plain's hits also rise as norm falls, the norm lever is loss-independent and part of ΔFM's no-wipe advantage was Δ1's contraction to norm 79; if plain stays flat or falls, ΔFM's direction is what makes the small row readable | 3–6 × 9 min |
| **"trained addresses are near-orthogonal random directions, context-free"** (`wake_rows_geometry`) | on `rb62f`, `pairEN_s750`, `pair0_s750`, Δ1: pairwise row cos distribution; shape-neighbour pairs (が·ぎ·ぐ·げ·ご share the dakuten, が/ガ hiragana–katakana) vs unrelated pairs; cos of each row to the pack's pretrained row for the same glyph. Structure appearing under ΔFM would reopen a premise W2d closed on | free |
| **shared trigger direction = 17–29 % of row energy, per-run** (K0) | already moved: plain 26 %, `rb62f` 22 %, **Δ1 11 %**. Add participation ratio / spectrum per table and cos of shared directions across today's arms (mixed / rb62f / w8) — the K0 merge price at 0.35 may be a ΔFM-vs-plain artefact rather than a per-run one | free |
| "composite residuals transfer, flat ones don't" (transplant closed) | covered by K0's merge eval; no separate item | — |

The free rows go in one probe (`src/probe/table_geometry.py`, reads
`trained.pt` + the pack) and run today; the native rows queue behind R4.1 /
R4.3. What they change: a loss-independent norm lever makes α a property of
the *table*, not of ΔFM, and the recipe table below loses a ΔFM-specific
argument; shape structure in the rows changes what K1's new kanji rows can
be expected to share with their neighbours.

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
directly next time. Original question kept below for the record.


`row_blocks_log.jsonl` (every step: in-box / out-of-box residual, row norm,
lr per row) and `src/probe/row_blocks_plot.py` (small multiples, two arms
overlaid). The question is whether rows are **uniform or ragged**: same
in-box curve and same end norm for every glyph → one schedule fits all and
the per-row lever is only the block length; ragged (some rows still falling
at step 62, some at their floor by 20, end norms spread) → per-row
normalisation is on the table (block length or lr from the row's own
in-box slope, or a norm target per row instead of a step count). The w8
overlay shows what the first 8 steps do to that.

### R4.3 — α on the 53 k table (native only, 2 × 9 min)

`rows_synth_d1_d1_s53k` × 0.5 and × 0.7, `--native_chars あ,か,す,日
--native_clauses en,swap` (the Δ1 read). Δ1's table already contracted under
cosine (161 → 79) and still lost `en` 22 / 14 / 11 vs plain's 36 / 24 / 16; if
× 0.7 recovers `en` joint toward 24 without the tail climbing past 16 the
Δ1 miss was part norm and **α is a deployment knob on the existing table**
(the LoRA-multiplier slot: train at full norm, ship at α). If it does not
move on 356 rows, the Δ0 curve was a 12-row artefact and α is dropped.

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
| loss | ΔFM (`--pair_loss 1 --pair_ref en`) | settled for singles (tail 15 vs 24); S2 decides it for sentences in `plan_synth3.md` |
| unit weights | uniform `*1` | settled by Δ1's exposure read (kanji ≥ kana per draw) |

**Before any warm start from Δ1** (K1 seeded from it, transplants,
S2a): fix the loader's row-unit conversion — `raw` is in units of the run's
own `row_scale`, Δ1's is 232.9 against ≈ 197 for every other table, and
`_init_rows_one` copies without converting (`plan_synth3.md` S2a
pre-condition has the details; past warm starts were all inside one
inventory family, ratio 0.996–1.001, so no recorded result moves).

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

### K0 — is a merged table a table? (free, no GPU)

The budget above wants the shortcut `plan_synth2.md` parked in *Chunking*:
train disjoint row blocks and union them by ext id (`src/probe/merge_tables.py`
already does this with the per-run `row_scale` correction; the shipped
`merge_punct` table is exactly this). That would make kanji scaling parallel in
wall-clock instead of linear. **It is not free**, and the price is now
measured — cosine between run's mean row directions (the shared trigger,
17–29 % of row energy):

| pair | cos |
|---|---|
| 53 k ↔ punct-only (the shipped `merge_punct`) | **0.590** |
| 53 k ↔ plain-FM Δ0 (`pair0_s3000`) | 0.686 |
| 53 k ↔ **ΔFM** Δ0 (`pairEN_s1500_lr2e-3`) | **0.348** |
| plain-FM Δ0 ↔ ΔFM Δ0 (same rows, same data) | 0.449 |
| 53 k ↔ `sent_s24k_a1_s05` (warm off it, μ 0.3) | 0.999 |

So the shared direction is **per-run, not per-line**, and the loss rotates it
more than the dataset does (0.449 on identical rows and data). The shipped
merge worked at 0.59 for 17 punctuation rows; nothing says a 400-row kanji
block merged at 0.35 into a ΔFM kana block does. Caveat: the Δ0 means are over
12–17 rows and are noisy estimators; the 53 k ↔ punct row is the load-bearing
one.

K0 is therefore: (a) the cos table above extended to Δ1's table once it lands,
(b) `merge_tables.py` Δ1 ⊕ punct-only and eval the merged table on both blocks
— if a 0.35–0.59 merge costs the donor block its singles, the shortcut is dead
and K is one long run; if it does not, K1 is two or three parallel-in-time
runs. Both are eval-only.

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

R4.0's free rows run first (one probe, no GPU); its native rows queue with
R4.1 and R4.3, all re-scores of existing tables (≈ 1.5 h of GPU together);
R4.2 lands on its own. Then the recipe table
is filled and K0 (free) answers whether K1 is one run or several. K1 itself
waits on `plan_synth3.md`'s S2 verdict for the box: if S2 dies, the seed
table is the whole artefact and K1's size is a shipping decision; if S2
passes, S2b goes first. Either way K1 is the recipe above, not Δ1's.

## Open risks

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
