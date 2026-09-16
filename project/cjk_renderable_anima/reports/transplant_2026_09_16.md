# transplant probe (2026-09-16): a composite-trained residual renders on the 53k shared direction; a flat-trained one does not

> Follow-up to [`rows_manifold_2026_09_16.md`](rows_manifold_2026_09_16.md),
> which split the 53k table into one shared direction m̂ (the self-built
> render trigger, 18 % of the energy, hit tracks it) plus near-orthogonal
> per-row residuals. Question (user): flat font data teaches identity
> cheaply but a flat-only table wipes the scene; can a new glyph be
> trained flat-only if it inherits the composite-trained m̂? No-training
> test (`src/probe/transplant_table.py` + the `native` stage, 128 renders per
> cond, あ か す 日 × 8 held-out scene prompts × 2 seeds, `en` clause,
> both readers): **a flat-only donor's residual (P0b 24 k, Run 3 8 k) on
> the 53k m̂ renders 0/64 — indistinguishable from m̂ alone (scene kept,
> floor garble in the bubble) — while the composite-trained micro6 donor's
> residual on the same m̂ renders 23/64 with the scene kept (en cos 0.889;
> micro6 original 54/64).** So the m̂ + residual decomposition is modular
> across tables, but only composite training produces a modular residual;
> flat-trained identity is conditional on its own flat trigger (P0b as-is
> 32/64 with its trigger, 0 without). The recipe for new glyphs is
> therefore **composite data with m̂ inherited and frozen, residual only
> trained** — the training arms `m6fm_s2k_pin` / `m6fm_s500_pin` (+ the
> `m6fm_s500_qoff` from-scratch control) test whether pinning buys steps.
> By-product: m̂ alone gives the best scene / placement numbers of any
> cond (en cos 0.920, box IoU 0.31 vs the 53k table's 0.882 / 0.13) — the
> 53k residuals cost scene, not the trigger.

## Construction

`Δ_r = a_r · m̂_fam(53k) + f_r^{donor}`, with `m̂_fam` the 53k table's mean
direction over basic kana rows (あ か す) or over the rest (日; the two
have cos 0.69), `a_r` the 53k row's own coefficient along it (あ 0.115,
か 0.179, す 0.264, 日 0.328), and `f_r^{donor}` the donor row with the
*donor's own* family direction projected out. `matched` rescales the
residual to the 53k row's residual norm (あ 0.34, か 0.38, す 0.43, 日 0.56).
Tables under `output/wake_probe/rows_{transplant,tm6,twd}_*/`, coefficients
in each dir's `transplant.json`; native settings identical to the 53k
native (`rows_synth_full_fm10k_full_s53k_qoff/native/`).

Donor facts that matter: P0b's shared direction has cos 0.18 (kana) /
0.46 (other) to the 53k one, Run 3's 0.14 / 0.42, micro6's 0.65 / 0.53;
same-glyph residuals across tables have cos ≈ 0.1 (addresses are per-table
random directions, as W2d found). P0b residual norms are 2–3× the 53k's
(hence `matched`); micro6's match already.

## Results (`en` clause, both-reader hits of 64; en cos = PE-Spatial cos to the `English text reads as "hi"` render of the same prompt/seed, box IoU vs that render's word box)

| cond | donor | donor training | hits | en cos | box IoU | per-char hits あ / か / す / 日 |
|---|---|---|---|---|---|---|
| 53k table as is (reference) | — | composite 0.9, 53 k | 36 | 0.882 | 0.13 | 11 / 10 / 10 / 5 |
| `p0b_full` (donor as is) | P0b | flat-only, 24 k | 32 | 0.851 | 0.19 | 5 / 9 / 12 / 6 |
| `m53k_p0bres` | P0b | flat-only | **0** | 0.935 | 0.39 | 0 / 0 / 0 / 0 |
| `m53k_p0bres_matched` | P0b | flat-only | **0** | 0.923 | 0.36 | 0 / 0 / 0 / 0 |
| `twd_m53k_res` | Run 3 (`encoder_wd_w120_s8k_fres_warm`) | flat-only, 8 k warm | **0** | 0.931 | 0.37 | 0 / 0 / 0 / (no row) |
| `twd_m53k_res_matched` | Run 3 | flat-only | **0** | 0.927 | 0.31 | 0 / 0 / 0 / (no row) |
| `m53k_only` (trigger, no residual) | — | — | 0 | 0.920 | 0.31 | 0 / 0 / 0 / 0 |
| `tm6_m53k_res` | micro6 (`rows_synth_micro6_fm_m6fm_s2k_qoff`) | composite frame-mix, 2 k on 6 rows | **23** | 0.889 | 0.19 | 6 / 3 / 12 / 2 |
| `tm6_m53k_res_matched` | micro6 | composite | 18 | 0.887 | 0.17 | 2 / 0 / 11 / 5 |
| micro6 table as is (reference, its own native) | — | composite | 54 | 0.868 | 0.12 | — |

`swap` clause (the EN caption with the word swapped) follows the same
order: micro6 residual 3–4 / 64, every flat donor and m̂-only 0 / 64,
P0b as is 10 / 64, 53k 18 / 64.

Sheets: the flat-donor and m̂-only conds draw the base's floor — a
multi-line block of pseudo-Japanese in the bubble, scene intact (that is
why en cos and box IoU beat every trained table). The micro6-residual
hits are the glyph in the bubble with the scene kept, す nearly at the
donor's own rate, か and 日 far below it.

## Reading

- **Identity is conditional on the trigger it was trained with; a
  composite-trained trigger makes it modular.** The parts probe
  (`wake_canvas_scenes_2026_09_14.md`: P0b's `f` alone 0/16) said flat
  identity renders only with its own canvas mode. This adds: it does not
  render with *another* trigger either, at any residual scale — while a
  residual trained on composites transfers to a trigger it never saw
  (micro6's own m̂ and the 53k's have cos 0.65, not 1). The difference is
  the data, not the amount: P0b had 234 draws/row, Run 3 130 + warm
  start, micro6 1 143 — but the 53k's own rows (529 draws/row, composite)
  are what the micro6 residual is imitating.
- **So flat-only training for new glyphs is closed**, even with the
  trigger supplied: the flat items teach "glyph given flat canvas" and
  that conditional does not factor. The user's efficiency question moves
  to *how many composite draws a new row needs when the trigger is
  given* — the pinned arms below.
- **The trigger does not need the residual to keep the scene**; it is the
  residuals that cost scene (53k en cos 0.882 vs m̂-only 0.920). A
  regulariser that keeps residuals ⟂ m̂ and small is the lever for the
  wipes, not the trigger (`--pin_orth`).
- Per-glyph transfer is uneven (す 12/16, か 3/16, 日 2/16 from a donor
  that renders all four): the donor's residual for か / 日 partly
  encodes *its* trigger's context. Expect the pinned arms to be the
  cleaner version of this.

## Pinned arms (launched 10:25, `--pin_dir`; results below when in)

`src/train/trainables.py::_pin_from` + `ExtDelta.pinned`: every trained row
gets a fixed `a_fam · m̂_fam` from the 53k table (kana 0.132, other 0.333,
`--pin_coef fam` — the honest new-row setting, no per-row leak from the
53k), the trainable `raw` is the residual only and is re-projected ⟂ m̂_fam
after every step (`--pin_orth 1`); the saved `trained.pt` folds the pin
into `delta.raw` (parts kept as `pinned` / `resid`). Data and recipe are
the micro6 frame-mix arm's (`synth_micro6_fm`, 6 rows 日す人出あか, batch
4, lr 1e-3 cosine, μ 1e-3, box 4, σ 0.7–0.9).

| arm | steps | draws/row | native `en` hits (both readers) / en cos / IoU | `swap` hits | flat singles | per-char `en` あ / か / す / 日 |
|---|---|---|---|---|---|---|
| `m6fm_s2k_qoff` (from scratch, reference) | 2 000 | 1 143 | 54 / 0.868 / 0.12 | 46 | 12/12 | 13 / 14 / 15 / 12 |
| `m6fm_s2k_pin` | 2 000 | 1 143 | **54** / 0.868 / 0.12 | **32** | 12/12 | 14 / 14 / 15 / 11 |
| `m6fm_s500_qoff` (from scratch control) | 500 | 286 | 39 / 0.901 / 0.18 | 9 | 8/12 | 11 / 10 / 12 / 6 |
| `m6fm_s500_pin` | 500 | 286 | **41** / 0.896 / 0.19 | **20** | 10/12 | 11 / 10 / 12 / 8 |

Training curves (loss, residual norm) of pin and scratch overlap at
every logged step (500: loss 0.0874 vs 0.0868, rel 0.413 vs 0.421; 2 k:
0.1006 vs 0.1005, 0.563 vs 0.605). Caveat on record: the six rows are in
the 53k table, so m̂ was fit with them present (a 433-row mean; the
per-row coefficient is *not* used).

**Verdict — pinning the trigger buys no steps.** On the trained clause
the pinned arm equals scratch at 2 k (54 = 54) and at 500 (41 vs 39,
inside seed noise), with the same scene score. The step budget is spent
on the per-row residual — the identity — and the trigger is not what a
new row is waiting for: a from-scratch row builds its share of the
shared direction for free while it learns identity (the 53k table's m̂
is the *mean* of rows each of which grew it). On the `swap` clause
(frame independence) the pin helps early (20 vs 9 at 500) and hurts at
convergence (32 vs 46 at 2 k): the frozen family direction plus the ⟂
projection deny the row the trigger component the EN frame wants, which
a free row grows by 2 k. So `--pin_dir` is not a training recipe; it
stays as an instrument (it is the cleanest way to read a residual
against a fixed trigger).

## What this settles for "new glyphs cheaply"

- Flat-only exposure cannot teach a scene-renderable glyph even with the
  composite trigger supplied (four donors, 0/64 each); composite items
  are the exposure that counts, and the data side of them is CPU (the
  scene pool is glyph-independent, the swap is a paste).
- Inheriting the trigger does not cut the composite steps; identity
  is the cost, ≈ 300 draws/row for 40/64, ≈ 1 000 for 54/64 on six rows,
  and the 53k run's 529 draws/row over 433 rows is on that curve.
- There is no shared identity to amortise (`rows_manifold_2026_09_16.md`:
  near-orthogonal residuals, no manifold). The remaining draws-per-row
  levers are in the *item*, not the table: more trained rows per
  composite (phrase / multi-slot swaps, once the wrapping fix lands), a
  larger glyph in the box (a stronger gradient per draw; `--scene_fill`,
  sl1's wider bubbles), and not spending draws on rows the readers
  cannot score. Do not re-propose flat-heavy mixes, trigger transplants,
  or Q / m̂ seeding as a shortcut.
