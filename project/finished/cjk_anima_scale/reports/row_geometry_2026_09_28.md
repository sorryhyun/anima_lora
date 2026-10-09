# row_geometry — the retrain runs' rows in row space (2026-09-28)

The question (user): do the trained rows of `retrain_kana` (174 cold kana,
135 / row) and the C3 kanji arms (36 cold kanji, 225 and 450 / row)
correlate with each other, and is the structure anything beyond what the
seed already had? CPU only, no render. **Verdict: the rows correlate
through one shared direction and a thin layer of glyph relations. Neither
is new in kind, and neither predicts a glyph's read.** The shared direction
grows in every run: kana rows put 32 % of their energy on it (seed 28 %),
kanji 450 put 38 % (seed 30 %). It is new: ⟂ the raw pack (0 % of its
energy along the pack mean). And it is common across runs (kana vs kanji 450
mean directions cos 0.83). Centered on it, the rows are as spread as the
seed's (PR 70.0 vs 70.2 of 169). Seed → trained is not a rotation. What
survives the centering is glyph relations: dakuten and small-kana pairs,
built beyond the pack. None of this is a lever: shared-direction init is
`--pin_dir`, closed 2026-09-16. The row-geometry features have |ρ| ≤ 0.38
against the 450 read.

Script `experiments/row_geometry/run_exp.py`; envelope
`experiments/row_geometry/results/20260928-1519-rg1/result.json`.
Row = the trained delta `raw × row_scale` (the pack row is under it;
effective = pack + delta). A run's rows = the ext rows that differ from its
`context` rows (kana: the seed; kanji: `p1_mix`). The seed has rows for 170
of the kana and 20 of the kanji; every same-row comparison uses those.

## 1. One shared direction carries the correlation

| rows | pairwise cos | shared (mean) energy | PC1 | cos after centering |
|---|---|---|---|---|
| kana, trained delta | 0.315 | **0.32** | 0.34 | −0.006 |
| kana, seed delta (same rows) | 0.277 | 0.28 | 0.29 | −0.005 |
| kana, pack | 0.271 | 0.27 | 0.28 | −0.006 |
| kana, effective (pack + delta) | 0.377 | 0.38 | 0.39 | −0.006 |
| kanji 225, trained delta | 0.338 | 0.36 | 0.36 | −0.028 |
| **kanji 450, trained delta** | 0.368 | **0.38** | 0.39 | −0.028 |
| kanji, seed delta (20 rows) | 0.262 | 0.30 | 0.31 | −0.052 |

- PC1 ≈ the mean energy, and centering sends the pairwise cos to zero:
  rank one, the mean. Training grows it over the seed and the pack, and
  225 → 450 grows it further.
- Per row, the delta leans against its own pack row (cos −0.18 kana,
  −0.21 kanji 450) and keeps part of its seed direction (cos 0.42, 0.29
  centered; kanji 0.38 / 0.24).

**The direction.** Cos of the trained mean delta to:

| | kana | kanji 225 | kanji 450 |
|---|---|---|---|
| the other runs' mean | kanji 225 0.81, kanji 450 **0.83** | 450: 0.95 | kana 0.83 |
| its own seed mean | 0.73 | 0.58 | 0.65 |
| (trained − seed mean) vs the seed mean | −0.36 | −0.60 | −0.42 |
| its rows' pack mean · whole-pack mean | 0.04 · 0.15 | 0.03 · 0.11 | 0.01 · 0.09 |
| T5 table mean | 0.25 | 0.21 | 0.19 |
| `u_S` (count twin's `u_twin`) | 0.51 | 0.45 | 0.47 |
| `u_S`, same convention (Δ vs seed, own-row part removed) | 0.63 | 0.45 | 0.51 |

- **Not from the raw pack**: energy along the rows' pack mean is 0.2 %
  (kana; kanji ≈ 0). The effective mean turns off the pack mean (cos 0.59)
  as its norm goes 98 → 177. A lean toward the T5 table's mean, ≈ 5 % of
  the energy.
- **Common across runs**, more than to each run's own seed mean: kana and
  kanji 450 train on different glyphs from different context rows and meet
  at 0.83. p1_mix's 36 donors meet `retrain_kana` at 0.83 too (§ 4).
- **Not the seed's direction scaled up**: what each run adds is partly
  against the seed mean (−0.36 … −0.60). The runs rebuild a similar
  direction, rotated.
- **Related to `u_S`, not equal to it** (0.45–0.63; the twin and Stage B
  met at 0.956). The seed's own mean is ⟂ `u_S` (0.05–0.06).

## 2. Effective rank: the seed's

Participation ratio (Σλ)² / Σλ², same rows:

| | trained delta | seed delta | pack | trained eff | seed eff | trained − seed |
|---|---|---|---|---|---|---|
| kana 170, centered | **70.0** | **70.2** | 76.8 | 78.6 | 78.9 | 80.9 |
| kana 170, uncentered | 8.2 | 10.9 | 12.0 | 6.3 | 8.8 | 29.0 |
| kanji 450, 20, centered | 16.4 | 16.1 | 17.0 | 17.0 | 16.7 | 16.5 |
| kanji 450, 20, uncentered | 5.6 | 8.1 | 10.2 | 4.9 | 7.3 | 12.2 |

- Centered, kana is unchanged (random 36-row subsets: 26.7 vs 25.9, pack
  27.2). The uncentered drop is § 1's shared energy, not a collapse.
- The kanji sets sit at the ceiling (20 rows → 19; all 36 → 29 of 35),
  so PR cannot separate 225 from 450 or from the seed. Kana at 20 rows
  (16.2) is at the ceiling too: only the 170-row read carries information.
- The retrain's change (trained − seed) is not low-rank (80.9 centered).

## 3. Seed → trained is not a rotation

Procrustes, rows unit-Frobenius after centering. Full space, in-sample (a
fit with rows ≪ dim; shuffled rows are the null). With ≥ 100 rows also
inside top-k PC subspaces fitted on half the rows and scored on the other
half, against a no-rotation fit (one scale) and the share of held-out
energy the k-subspace can hold at all:

| kana | full in-sample (shuffled) | held-out k = 8 / 16 / 32: rot | shuffled | no rotation | ceiling (k = 32) |
|---|---|---|---|---|---|
| seed delta → trained delta | 0.78 (0.66) | 0.02 / 0.02 / 0.02 | −0.04 … −0.10 | **0.08** | 0.22 |
| seed eff → trained eff | 0.83 (0.70) | 0.02 / 0.03 / 0.03 | −0.04 … −0.10 | **0.16** | 0.20 |
| pack → trained eff | 0.79 (0.61) | 0.02 / 0.02 / 0.02 | −0.03 … −0.10 | **0.22** | 0.20 |

- A rotation fitted on half the rows does not carry to the other half, and
  falls below no rotation at all. The top 32 PCs hold ≈ 20 % of held-out
  energy, so a low-dimensional rotation could not explain the change even
  in principle (the PR 70 of § 2).
- Full-space in-sample sits 0.12–0.18 above shuffled: the pair-cos
  structure the seed and trained rows share (centered pair-cos Pearson
  0.38), not a map.
- Kanji: seed → trained 0.95 vs shuffled 0.93 (20 rows, no power);
  225 → 450 0.98 vs 0.91, and no rotation already fits (same-row cos 0.77,
  centered 0.66, norm × 1.14): 450 is 225 pushed further.

## 4. What the centered rows keep: glyph relations

Kana, centered trained delta, pair means vs a permutation null
(all pairs −0.006):

| pairs | n | trained | perm p | pack | seed |
|---|---|---|---|---|---|
| dakuten / handakuten (か ↔ が) | 52 | **0.24** | < 5e-4 | 0.06 | 0.19 |
| small ↔ full (ゃ ↔ や) | 18 | **0.22** | < 5e-4 | 0.11 | 0.07 |
| hiragana ↔ katakana (あ ↔ ア) | 81 | 0.11 | < 5e-4 | 0.11 | 0.06 |

- Dakuten and small-kana pairs are built beyond the pack (the seed had
  dakuten, not small kana). Hiragana ↔ katakana is the pack's, inherited.
  The script blocks (hira–hira, kata–kata) carry nothing (≈ 0.01).
- The strongest pairs are punctuation (。・ 0.67, ～〜 0.64, 、～ 0.60)
  and べ / ペ / ベ (0.52).

**p1 arms on the same 36 donors** (shared energy / delta norm):
`p1_mix` 0.28 / 184, `p1_cold` 0.23 / 161, `p1_lone` 0.26 / 165;
`retrain_kana` on those rows **0.43 / 272**. Mean direction to
`retrain_kana`'s: 0.83 / 0.75 / 0.61; same-row cos 0.57 / 0.49 / 0.40.

## 5. Row geometry vs the read

Kanji, the 24 read glyphs of `c3_kanji/results/20260928-1145-c3s450`,
Spearman (|ρ| ≥ 0.41 for p < 0.05 at n = 24):

| feature | official 450 | contained | repeat | official 225 |
|---|---|---|---|---|
| shared share of the row | 0.15 | 0.13 | 0.38 | −0.17 |
| row norm | 0.38 | 0.27 | 0.19 | 0.34 |
| residual norm (shared part removed) | 0.29 | 0.23 | 0.04 | 0.31 |
| ink | 0.04 | −0.15 | **−0.49** | −0.18 |

Nothing in the row predicts a glyph's identity read. Row norm is the
closest, short of significance. Ink predicts repeats (dense glyphs repeat
less). Rows are judged by render, as before (`spell_2026_09_26.md`).

## 6. What it settles

- **Shared-direction init: no.** The direction is what `--pin_dir` pinned
  (`../finished/cjk_renderable_anima/reports/transplant_2026_09_16.md`: curves
  overlap scratch, 54 = 54 at 2 k, `swap` hurt at convergence). A free row
  builds it while learning identity. It is ≤ 40 % of a row's energy, and a
  run rebuilds it rotated even from a seed that has it.
- **No low-dimensional map between tables** (rotation, subspace):
  retrain rows are the seed's shared direction regrown plus per-row
  identity that keeps ≈ 0.3 of its seed direction.
- **Open, for the doubling rise** (`retrain_experiments.md` § 5, `dup` 42 vs
  `p1_mix`'s 31): on p1_mix's donors `retrain_kana` carries more of the
  shared direction (0.43 vs 0.28) and more norm (272 vs 184), and
  `transplant_line_2026_09_26.md` found spell_b's doubling mostly in its
  shared part. Across the C2 arms the share does not order `dup`
  (`p1_cold` 0.23 → 60, `p1_lone` 0.26 → 40, `p1_mix` 0.28 → 55 / 128),
  so this is a candidate, not a reading. The read that settles it is
  training-free: `transplant_line`'s strip mode on `retrain_kana`'s rows
  (shared part removed or scaled), C2's eight words against the cached
  floor.
