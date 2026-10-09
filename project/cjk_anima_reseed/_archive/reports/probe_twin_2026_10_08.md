# probe_twin: the twin difference (2026-10-08)

`idea3.md`. One `sent` item drawn twice — A with glyph u in slot k, B with
v there (`render_into_scene(ref_text=…)`: the record's scene, columns and
fill, one font, one rng; pixel-identical outside one glyph's box) — under
A's caption (row u), one σ, one ε, the same Gaussian probes on both. Per
(twin, row): D = g_A − g_B (the part of the row's sensitivity that depends
on the glyph drawn) and S = (g_A + g_B) / 2 (the shared part); M_D, M_S
per fit, family, own row (u) / cross rows (the item's other once-only
glyphs), in-box (dilated 2 cells) / out.

`probe_twin.py render`: f0's `sent` records in a seeded order, u kana or
kanji (200 each), v of u's script and size class from the glyphs f0's
texts hold ≥ 20 times (80 hiragana, 79 katakana, 907 kanji — a first pass
drew v from the whole pack and got rare kanji), absent from the item, not
u's dakuten / small-kana sibling; every twin's A / B difference one
glyph-sized box (15–36 px). `fit`: rows at f0's start (stick080), fp32 /
SDPA / eager as `probe_jl`, σ uniform on 0.45–0.8, 8 probes per mask,
A and B as two forwards, twins alternating between fits A / B within a
family (100 each). Job `20261008-012301-aa95d6`, 52.8 min (7.9 s / twin)
→ `output/cjk_anima_reseed/probe_twin/t1/` (`fit.pt`, `read{,_pair}.json`).

## f — how much of a row's sensitivity changes with the glyph

f = E|D|² / E(|g_A|² + |g_B|²) (1 − cos for equal norms; 1 = the two
gradients orthogonal), pair-weighted, ratio of sums:

| | σ 0.45–0.55 | 0.55–0.65 | 0.65–0.8 |
|---|---|---|---|
| kana own, in-box | 0.39 | 0.44 | 0.22 |
| kana cross, in-box | 0.33 | 0.32 | 0.18 |
| kanji own, in-box | 0.35 | 0.32 | 0.36 |
| kanji cross, in-box | 0.22 | 0.20 | 0.31 |
| own, out-box | 0.82–0.90 | 0.86–0.90 | 0.67–0.80 |
| cross, out-box | 0.89–0.90 | 0.88–0.98 | 0.69–0.84 |

- In the box a third of the Jacobian's energy turns with the glyph (63–89
  own pairs per cell; `grad_identity`'s 0.68 is the FM-loss gradient's, a
  residual-weighted quantity, not this one). The neighbours' rows turn
  0.6–0.9 as much as the swapped glyph's own — `grad_identity`'s f_cross ≈
  f_own, again.
- **Off the box the row's sensitivity all but decorrelates** (0.67–0.98)
  when one glyph in the box changes, own and cross rows alike: what a row
  does to the page depends on the drawn text, near-orthogonally.

## The subspace — the stop rules fire

| | kana | kanji | bar |
|---|---|---|---|
| M_D own in-box A / B @16 (pair; none) | 0.27; 0.25 | 0.25; 0.23 | 0.6 |
| … by fit A's first 10 / 25 / 50 / 100 twins | 0.20 / 0.21 / 0.22 / 0.27 | 0.18 / 0.22 / 0.23 / 0.25 | |
| M_S own in-box @16 (pair) | 0.48 | 0.46 | |
| r = tr(M_D) / tr(M_S) in-box | 0.75 | 0.62 | |
| cross-fit λ / r top-1 / 16 (pair) | 0.96 / 1.02 | 1.28 / 1.11 | 2 |
| high-λ subspace A / B @16 | 0.017 | 0.017 | isotropic 0.016 |
| own / cross @16: own–own, cross–cross, own–cross | 0.27, 0.53, 0.36 | 0.25, 0.52, 0.31 | own–cross < own–own |

- **M_D is not estimable at this budget and barely grows with it**: 0.20 →
  0.27 from 10 to 100 twins (kana), far under 0.6; M_S, the shared part,
  reaches 0.46–0.48.
- **No identity-only directions.** Cross-fit λ / r ≈ 1 at every k (none
  0.86–1.41): the directions where the row's effect turns with the glyph are
  the directions where it acts at all, in the same proportion (r 0.6–0.8).
  The high-λ subspace sits at the isotropic floor.
- **Identity is the window's, not the row's.** M_D of the own row overlaps
  the cross rows' M_D (0.31–0.36) more than it overlaps itself across fits
  (0.25–0.27); the cross rows' own floor is 0.52–0.53 on 6× the pairs.
- The moves: f0's shared vector puts 0.22 (kana) / 0.31 (kanji) of its
  energy in M_D's top 16 (13–19× isotropic) — but M_D's top 16 does not
  repeat between fits, so this is read on a lens that is not there.

## Verdict

idea3 stops on all three rules, at both weightings, and more twins would
not save it: M_D's overlap is flat in the sample and λ ≈ r does not depend
on the budget. Glyph identity is not a subspace of the row's
sensitivity — it rotates the sensitivity the row already has (a third of
it in the box), the same for the neighbours' rows as for the swapped
glyph's own. With probe 0 (`probe_jl_2026_10_07.md`) and its PE twin
(`probe_jl_pe_2026_10_08.md`), the Jacobian-lens line has no estimable
row subspace to project onto: not for text against page, not for identity
against layout.
