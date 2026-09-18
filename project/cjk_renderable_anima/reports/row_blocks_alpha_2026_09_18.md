# Δ0 smokes after Δ1 — box weight, row blocks, the norm curve (2026-09-18)

> Six Δ0-sized arms (12 dakuten rows, `synth_pair_d0`, ΔFM, lr 2e-3, 744–750
> steps) plus seven native-only re-scores, all off the question Δ1 left: why
> does ΔFM lose the `en` clause. **`--box_weight 4` stays** (w = 1 weakens the
> glyph and buys nothing on the scene). **Row-block training** (one row at a
> time, own Adam state, own schedule) gives the best singles of the line at
> this budget (16/24) — but its native gain turned out to come from a bug
> that *shrank* rows, and a no-training α sweep on the fixed table showed
> why: **native hits rise monotonically as the row norm falls (17 → 38 of 64
> at ×1.0 → ×0.2) while scene fidelity falls with it (en cos 0.921 →
> 0.875)**. A mean / residual decomposition does not break that curve — the
> row is one direction and its norm is the whole lever, with the crossing at
> ≈ ×0.7. The next read is the same α on Δ1's 53 k table (native only).

## What ran

All arms `--stage train eval --arm rows --data_tag synth_pair_d0 --units
chars:がぎぐげござガギグゲゴザ --shapes 512 --batch 4 --t_min 0.7 --t_max 0.9
--compile 1 --grad_ckpt 0 --aggressive_recompute 0 --lr_rows 2e-3 --lr_decay
cosine --free_residual 1e-3 --seeds 2 --no_floor --c_flat 0 --pair_loss 1`,
natives `--native_chars が,ガ,ご,ゴ --native_clauses en,swap --seeds 2
--delta_parts full`. Baseline is `pairEN_s750_lr2e-3` (2026-09-17, jobs
`20260917-221722-34514c` / `…-ebe2e0`).

| arm (`rows_synth_pair_d0_<arm>`) | what moves | train + eval | native |
|---|---|---|---|
| `pairEN_s750_lr2e-3_bw1` | `--box_weight 1` | `20260918-115726-5c567c` 14.9 m | `…-a7bcac` 9.3 m |
| `pairEN_rb62_lr2e-3` | `--row_blocks 62 --train_steps 744` (leaky, see below) | `20260918-124953-e1a48c` 14.7 m | `…-a18a43` 9.1 m |
| `pairEN_rb62f_lr2e-3` | same, non-block rows frozen | `20260918-131506-c3ec1f` 14.8 m | `…-691eb0` 9.1 m |
| `pairEN_rb62f_lr2e-3_a0p2 / _a0p4 / _a0p7` | the `rb62f` table × 0.2 / 0.4 / 0.7, no training | — | `20260918-134302-da8eb7 / -558f8e / -550113` ≈ 9 m each |
| `pairEN_rb62f_lr2e-3_mean1_res0p2` | `rb62f` = mean + residual; mean × 1, residual × 0.2 | — | `20260918-141200-5d99fe` 8.9 m |
| `pairEN_rb62f_lr2e-3_mean0p2_res1` | mean × 0.2, residual × 1 | — | `…-82b6be` 9.1 m |
| `pairEN_rb62f_lr2e-3_w8` | `rb62f` + `--lr_warmup 8` (per block) | `20260918-143653-0abe2d` | `…-98108f` |

Native columns below: `both` = sfx ∧ vl hit, `joint` = both ∧ en cos ≥ 0.85,
`tail` = en cos < 0.85, of 64 renders per clause (4 glyphs × 2 seeds × 8
prompts); `en` is the JA frame (`japanese text. Japanese text reads as`),
`swap` the EN frame. Co-text = renders with more than one text box.

## Box weight under ΔFM: keep 4

Every paired arm so far ran `--box_weight 4`, inherited from plain FM where it
keeps the out-of-box noisy-target MSE (≈ 0.1 per cell over ≈ 94 % of the
latent) from swamping the glyph. Under ΔFM that term is already cancelled
(`pres` ≈ 5e-4), so the weight looked redundant — its only remaining job
would be to de-weight the preservation term 4×.

| s750, lr 2e-3, paired | `single` | `en` both / joint / tail | `swap` | en cos | `pres` | delta norm |
|---|---|---|---|---|---|---|
| `--box_weight 4` | **13/24** | **22 / 20 / 7** | **9 / 9 / 0** | 0.930 | 4.8e-4 | 137 |
| `--box_weight 1` | 9/24 | 15 / 14 / 6 | 5 / 5 / 1 | 0.932 | 4.3e-4 | 93 |

Dropping it weakens the glyph everywhere and changes nothing on the scene —
en cos, `pres`, tail, co-text (28 → 31) all flat; the table is 30 % smaller at
the same steps. Under ΔFM the weight's job is the in-box share of the
*gradient direction*: at w = 1 the `pres` noise takes 4× more of it and the
row grows less per step. Not a lever for the `en` clause. Stays at 4.

## Row blocks: `--row_blocks N`

Rows share no parameter (rows-only training; a row's gradient comes only from
its own items), so mixed batches are a habit from LoRA training, not a need.
`--row_blocks N` (`src/cli/train.py`, `src/train/stage.py::Batcher`) trains
one ext row at a time for N steps, rows in a shuffled cycle; every batch is
`--batch` items of that row from one shape, drawn with replacement; the lr
schedule (`--lr_warmup`, `--lr_decay cosine`) is measured inside the block and
the row's Adam slice (`exp_avg`, `exp_avg_sq`) is zeroed at block start. What
that removes from the mixed scheme: a global warmup / cosine that lands on
each row at a different phase (at 356 rows a row is in ≈ 1.1 % of batches);
Adam moving absent rows on stale momentum and `v` decaying (β₂ 0.99) between
visits; one item per row per visit (a block batch is four scenes of one
glyph). Same wall-clock. Single-glyph items only — a multi-row item has no
block, so it does not fit the sentence step.

At 12 rows × 62 steps = 744 ≈ the s750 budget:

| s750-equivalent, lr 2e-3, paired | `single` | `en` both / joint / tail | `swap` | en cos | co-text (en) | row norms |
|---|---|---|---|---|---|---|
| mixed (`pairEN_s750_lr2e-3`) | 13/24 | 22 / 20 / 7 | 9 / 9 / 0 | 0.930 | 28 | 126–151 |
| row blocks, **leaky** (`rb62`) | 15/24 | **31 / 24 / 9** | **17 / 17 / 0** | 0.920 | 23 | 24–153 by block age |
| row blocks, **frozen** (`rb62f`) | **16/24** | 17 / 12 / 9 | 10 / 10 / 0 | 0.921 | 25 | 81–148 |

**The leak.** The first version let the other rows keep taking Adam steps
after their block — from the `--free_residual` μ‖f‖² pull and their leftover
momentum, at the block-restarted lr — so rows shrank with block age: block 0
→ norm 24, 1 → 25, 2 → 53, 3 → 67, 4 → 79, … 11 → 150. The fix restores every
non-block row after `opt.step()` (`stage.py`, the `raw_before` block).

**Singles improve with the fix (13 → 15 → 16 of 24) — the native gain does
not survive it.** Per native glyph, `en` both of 16:

| glyph | mixed (norm) | leaky (norm) | frozen (norm) |
|---|---|---|---|
| ガ | 7 (132) | **15 (25)** | 4 (115) |
| が | 10 (140) | 10 (153) | 8 (145) |
| ゴ | 4 (150) | 4 (150) | 3 (148) |
| ご | 1 | 2 (79) | 2 (140) |

ガ at norm 25 renders 15/16 on the JA frame; the same row direction at 115
renders 4/16 and mixed's at 132 renders 7/16. The leaky run was an accidental
norm sweep, and what the natives liked was the small row.

## The α sweep: norm is the lever (no training)

`rb62f`'s `trained.pt` with `raw` × α, native only. Mean row norm 131 → 92 /
52 / 26.

| `rb62f` × α (norm) | `en` both / joint / tail | en cos | `swap` both / joint / tail | en cos | が/ガ/ご/ゴ (`en`) |
|---|---|---|---|---|---|
| × 1.0 (131) | 17 / 12 / 9 | 0.921 | 10 / 10 / 0 | 0.970 | 8 / 4 / 2 / 3 |
| × 0.7 (92) | 29 / **22** / 10 | 0.910 | 12 / 12 / 0 | 0.961 | 9 / 9 / 4 / 7 |
| × 0.4 (52) | 34 / **22** / 15 | 0.890 | 14 / 11 / 5 | 0.939 | 12 / 13 / 1 / 8 |
| × 0.2 (26) | **38** / 23 / 18 | 0.875 | **16** / 14 / 10 | 0.926 | 11 / 14 / 1 / 12 |

Hits rise monotonically as the norm falls, on both clauses and on three of
the four glyphs (ご is at floor at every norm, as on every Δ0 arm); scene
fidelity falls monotonically with it (en cos, tail). `joint` peaks at
× 0.4–0.7 and the tail is still 10 at × 0.7 — that is the crossing. Sheets:
`rows_synth_pair_d0_pairEN_rb62f_lr2e-3_a0p2/native/cross/x_<glyph>_en.png`
(prev = × 1.0, curr = × 0.2).

Two readings were on the table. (i) A big row pushes the scene toward the
wipe / pseudo-text and spoils the read while the glyph is there (Δ1's
`en`-clause finding, `synth_pair_delta1_2026_09_18.md`); shrinking it lets the
base's JA-frame render through, which reads the glyph but drifts from the EN
reference. (ii) The ΔFM row carries two jobs at different norms — the glyph
(light) and a scene-pull back toward the EN-reference render (full) — and
they could be separated. The decomposition below tests (ii).

## Mean / residual decomposition: does not break the curve

`rb62f` rows = mean (the shared direction, norm 61, 22 % of row energy) +
per-row residual (mean norm 116). The shared direction is orthogonal to the
pretrained EN quoted-frame direction Q on every frame (cos −0.04 … +0.04 for
`rb62f`, `pairEN_s750`, `pair0_s750` and Δ1's table alike; per-row energy on
Q 0.1 %) — so whatever it is, it is not Q, and it is learned.

| table (row norm) | `en` both / joint / tail | en cos | curve at that norm |
|---|---|---|---|
| × 1.0 (131) | 17 / 12 / 9 | 0.921 | — |
| **(b) mean × 0.2 + residual × 1 (117)** | 14 / 13 / 8 | 0.924 | ≈ 21, ≈ 0.917 |
| × 0.7 (92) | 29 / 22 / 10 | 0.910 | — |
| **(a) mean × 1 + residual × 0.2 (65)** | 24 / 20 / 10 | 0.896 | ≈ 32, ≈ 0.897 |
| × 0.4 (52) | 34 / 22 / 15 | 0.890 | — |

Both decomposed tables land exactly where their *total norm* puts them on en
cos — keeping the shared component at full norm (a) does not hold the scene
any better, dropping it (b) does not hurt it. On hits both sit *below* the
curve: removing either component costs more glyph than shrinking both. So
reading (ii) is wrong: the row is one direction, the two behaviours are one
vector's norm function, and there is no c-style split to train. Sheets:
`…_mean0p2_res1/native/cross/x_<glyph>_en.png` (prev = (a), curr = (b)).

## Reading

1. **`--box_weight 4` is settled** for every paired arm; not the `en` lever.
2. **Row blocks are the better batcher for the singles table** (16/24 at the
   s750 budget, the line's best; direction is what improved — bw1 and the
   leak both say displacement follows direction consistency). It does not fit
   multi-row items, so it is a seed-table / K1 recipe, not an S2 one.
3. **Row norm is the native lever, and it is non-monotone in what we want:**
   hits want small rows, scene fidelity wants big ones, and the crossing on
   this table is ≈ × 0.7 (joint 22, tail 10, cos 0.91). Eval singles were
   read at × 1.0 only — whether α < 1 costs the eval template is unread.
4. **This is not a training-schedule question at first order.** lr sets ∫lr
   → norm (1e-3 × 1500 ≡ 2e-3 × 750, `synth_pair_2026_09_17.md`); a
   trained-then-scaled row and a trained-smaller row differ only in how
   converged the direction is. The cheapest deployment is train at full norm,
   infer at α ≈ 0.7 — the LoRA-multiplier slot. Training-side equivalents
   (`--free_residual` μ up, shorter blocks) change the direction too and are
   not yet measured.
5. **Δ1's `en`-clause loss is probably part norm.** Δ1's table contracted
   under cosine (161 → 79) and still lost `en` 22 vs 36; the same α on the
   53 k table, native only, is the free test and the one that matters — 356
   rows, where Δ0's 12 never saw the `en` loss.

## Owed

- α ∈ {0.5, 0.7} on `rows_synth_d1_d1_s53k` — native only, the real read.
- `rb62f` × 0.7 **eval singles** (`--stage eval` on the `_a0p7` arm dir, ≈ 13
  min) — now load-bearing twice: whether the deployment α costs the template,
  and which way the w8 result below reads.
- `plan_synth3.md`: row blocks + α go in as the K1 recipe change and a
  deployment knob; S2-smoke order unchanged.

## `pairEN_rb62f_lr2e-3_w8` (14:36 → 15:00): natives up, singles down, undecidable without the × 0.7 eval

`rb62f` + `--lr_warmup 8`, measured inside each block (jobs
`20260918-143653-0abe2d` 14.8 m / `…-98108f` 9.1 m). The question was whether
the block's first Adam step — ± lr on every coordinate from a fresh state, a
jump of ≈ lr·√dim in one batch's gradient sign — costs direction.

| `rb62f` | `single` | `en` both / joint / tail | en cos | `swap` both / joint / tail | が/ガ/ご/ゴ (`en`) | row norm |
|---|---|---|---|---|---|---|
| warmup 0 | **16/24** | 17 / 12 / 9 | 0.921 | 10 / 10 / 0 | 8 / 4 / 2 / 3 | 131 [81–148] |
| warmup 8 | 10/24 | 21 / **21** / **6** | **0.928** | 13 / 13 / 0 | 7 / 6 / 3 / 5 | 113 [62–132] |

Natives improve on every column and, unlike the α arms, *above* the norm
curve: at norm 113 the curve predicts ≈ 23 both / ≈ 16 joint / cos ≈ 0.916,
and w8 gives 21 / 21 / 0.928 with the lowest tail of the day — every hit kept
the scene. That is a direction-quality signal, the one the warmup was meant to
buy. Eval singles fall 16 → 10 of 24 at the same time.

The two readings cannot be separated from what ran: the α sweep never
re-scored eval singles, so it is unknown whether the template is simply
norm-hungry (then w8's 10 is the curve at × 0.86 and the warmup is a clean
win) or whether the warmup moved the direction off the template (then it is a
trade). `rb62f` × 0.7 eval singles decides it and is the first thing to run.

## `_log` reruns (15:10 → 15:40): per-row norm is the trajectory, and the 12-row chaos floor

`rb62f` and `rb62f + w8` re-run with `row_blocks_log.jsonl` (every step:
in-box / out-of-box residual, row norm, lr; jobs `20260918-151049-5da276` /
`-5c897c`), plotted by `src/probe/row_blocks_plot.py` →
`rows_synth_pair_d0_pairEN_rb62f_lr2e-3_w8_log/row_blocks.png`.

- **In-box loss is not a per-row signal.** Every row 0.21–0.24 → 0.19–0.24
  over its block, minima ≈ 0.17: at σ 0.7–0.9 the in-box paired residual is
  batch noise with the glyph term ≈ 10 % on top. Row norm is what moves.
- **One shape, ragged end points.** All 12 rows: norm 16 after step 1 (the
  fresh-Adam ± lr jump), 28 / 46 / 70 at steps 2 / 4 / 8 (≈ +8 per step),
  saturating by step 32 under the cosine tail. End norms 80–153, and the two
  low rows are the same in both arms — ぐ 80 / 63, ガ 104 / 81 — so the
  spread is row-intrinsic (gradient consistency of that glyph), not the
  schedule. ガ is also the glyph the natives read best at norm 25.
- **Warmup 8** replaces the step-1 jump (16 → 2), catches up by step 16 and
  ends ≈ 8 % lower; the trajectory is otherwise the same curve.
- **Chaos floor.** The rerun of `rb62f` with the same argv and seed lands
  its rows at **cos 0.47–0.92 (mean ≈ 0.75)** to the original (kernel
  non-determinism); eval singles 16 → 14 and w8's 10 → 13. Warmup 0 vs 8
  differ at cos 0.17–0.74 — only somewhat below that floor. **Every
  12-row arm comparison above sits on this floor**: w8's native gain (joint
  12 → 21) is not separable from rerun noise at one seed, and a recipe
  decision (warmup, block length) needs ≥ 2 seeds per arm.
