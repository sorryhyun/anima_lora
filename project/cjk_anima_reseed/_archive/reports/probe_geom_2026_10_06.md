# probe_geom: what the loss terms and AdamW do to a row (2026-10-06)

From the user (10-06, after `structure_candidate.md`): does the box-share
loss move a row's norm or turn it where no glyph asks, and is the turn
`rand_turn` priced (`sent_ball_2026_10_05.md` § 6) the loss's or the
optimizer's? `probe_geom.py`: the rows at `seed_fixed_1005_stick080`
(f0's start), f0's data dir and its first 900 batches in its own order;
per batch the in-box term `mean(s · mean_in)` and the out-box term
`mean((1 − s) · mean_out)` of `box_share_fm_loss` taken apart and each
one's gradient on the rows read. No step is taken. Read in
`structure_candidate.md`'s coordinates (stick `s`, init `q_i`, glyph
`e_i`, the row's radius); AdamW replayed on the saved gradients against
SGD. Job `20261006-162314-61f80a` (daemon, 10.3 min, 0.69 s / batch eager) →
`results/20261006-1639-probe_geom-s080/` (`read.json`, `tiers.json`).

- **The out-box term does nothing to the rows.** Its gradient per draw is
  2 % of the in-box term's (kana and kanji alike), its split halves agree
  at 0.01–0.07 and it sits at cos 0.00–0.01 to the move f0 made. Every row
  move is the in-box term's.
- **No norm push.** A descent step's cos with the row's own direction is
  +0.01–0.02 (a random direction: ± 0.03). The one lean: frequent kana step
  against their stick (cos −0.10).
- **A draw's gradient is ~97 % noise.** The split-half cos follows the draw
  count (kana Q4 0.80 at ≥ 167 draws, kana median 0.26 at 32, kanji 0.07 at
  6), and Spearman–Brown puts the per-draw signal share at **≈ 2.5 %** for
  every group. A row's summed gradient is half signal at ~40 draws; before
  that its turn is a random walk — the turn `rand_turn` reproduced.
- **AdamW is not the source.** Replayed on the same gradients, AdamW's
  displacement sits at cos 0.78 (rare kanji) – 0.96 (frequent kana) to
  SGD's, with the same split-half agreement. The noise is in the gradient.
- **For kana, the windows and the lines pull opposite ways**: per row, the
  bubbleN draws' summed gradient against the sent draws' at cos **−0.18**
  (median, 46 rows); within a tier, a bubbleN draw agrees with its row's
  other bubbleN draws half as well as a sent draw does. Kanji +0.06 on 8
  rows. A candidate for the short kana loss every warm pass shows
  (`sent_kanji_2026_10_06.md` Open), not tested.

## 1. The terms

Medians over rows; `step` = the direction of a descent step (−mean
gradient). cos baselines: ± 0.03 for a random direction.

| family / group | draws | \|out\| / \|in\| per draw | split-half in | split-half out | cos(step, row) | cos(step, s) | cos(step, e_i) |
|---|---|---|---|---|---|---|---|
| kana all (153) | 32 | 0.02 | 0.262 | 0.024 | +0.024 | −0.040 | +0.029 |
| kana rare Q1 (40) | 16 | 0.02 | 0.070 | 0.018 | +0.009 | −0.000 | +0.009 |
| kana frequent Q4 (38) | 167+ | 0.02 | 0.797 | 0.069 | +0.008 | −0.096 | +0.009 |
| kanji all (459) | 6 | 0.02 | 0.069 | 0.008 | +0.019 | +0.011 | +0.022 |
| kanji rare Q1 (181) | 5 | 0.02 | 0.016 | 0.018 | +0.016 | +0.035 | +0.011 |
| kanji frequent Q4 (108) | 11+ | 0.02 | 0.161 | 0.026 | +0.018 | −0.029 | +0.028 |

- The step's share on the family's top-64 `e_i` PCs (the glyph subspace;
  isotropic 0.063): in-box 0.10–0.12, out-box 0.10–0.11. The gradient
  leans into the glyph subspace twice over chance and still lies 90 %
  outside it.
- cos(mean in, mean out) per row +0.07 – +0.15: the two terms do not fight;
  the out-box one is too small to matter either way.

## 2. Signal and noise

Spearman–Brown on the split halves (cos of two halves of n / 2 draws each
= (n/2)ρ / (1 + (n/2 − 1)ρ)): ρ ≈ 0.027 (kana Q4), 0.022 (kana median),
0.024 (kanji median) — one per-draw signal share across draw counts. The
first 900 steps' direction against the move f0 ended with: kana 0.27–0.31,
kanji 0.07–0.11 — what that share predicts at these draw counts.

By the item that holds the glyph (draws whose glyph sits in one item of the
batch; rows with ≥ 20 such draws; leave-one-out cos with the row's other
draws):

| | σ 0.2 | 0.3 | 0.4 | 0.5 | 0.6 |
|---|---|---|---|---|---|
| kana | −0.056 (183) | +0.107 | +0.153 | +0.154 | +0.147 |
| kanji | +0.007 (37) | +0.096 | +0.103 | +0.114 | +0.097 |

σ < 0.3 carries nothing, but those draws are mostly `bubbleN_18`'s (band
0.2–0.5): σ and tier are not apart here. The other-draws mean is ~90 % sent,
so the per-tier rows below read inside each tier:

| | within-tier LOO cos, bubbleN | within-tier LOO cos, sent | cos(bubbleN sum, sent sum) per row |
|---|---|---|---|
| kana | +0.066 (1 028 draws) | +0.161 (12 283) | **−0.181** (46 rows) |
| kanji | +0.041 (84) | +0.119 (2 277) | +0.056 (8 rows) |

Per draw, ρ ≈ 0.017 for a kana bubbleN draw against ≈ 0.034 for a sent one.

## 3. AdamW against SGD

The saved gradients through AdamW's recursion (betas 0.9 / 0.99, the row's
gradient where drawn, 0 elsewhere), f0's peak lr 2e-4 and kana row_lr 0.12,
no warmup (the turn degrees are an upper bound):

| group | cos(AdamW, SGD) | AdamW split-half | spike turn |
|---|---|---|---|
| kana all | 0.896 | 0.251 | 1.3° |
| kana frequent | 0.958 | 0.804 | 3.2° |
| kana rare | 0.810 | 0.077 | 0.9° |
| kanji all | 0.813 | 0.056 | 7.4° |
| kanji rare | 0.781 | 0.015 | 6.6° |

AdamW's split-half agreement is SGD's: it scales what the gradient carries,
signal and noise alike. Its own share — the per-element normalisation on
a rarely drawn row — is the 0.2 of the direction it does not share with SGD.

## Read

The box weighting is not a lever on row geometry: the out-box term is
inert on the rows and neither term pushes the norm. What turns a warm row
is the gradient's own noise, ~97 % of every draw, summed over too few draws
— a kanji drawn 6 times in 900 steps has turned before its direction is
known. Lowering lr shortens the walk without changing its signal share at
a given draw count (`sent_ball_lr2`: rows held, short strings lost the
same).

## Open

- Where the noise comes from: the same item at several (σ, ε) against
  other items — if (σ, ε) dominates, several noise draws per item per step
  are a cheap variance cut.
- The kana window / line conflict against the short kana loss: the run
  with the kana held at preview51 and the kanji free (`sent_kanji` Open),
  or the bubbleN / sent shares moved.
- A linearised read at f0's start, its first 900 of 53 920 steps; kanji at
  ~6 draws a row rest on the per-draw share extrapolated.
