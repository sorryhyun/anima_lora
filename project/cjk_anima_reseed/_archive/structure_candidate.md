# structure_candidate — how to read a trained row (2026-10-06)

A candidate decomposition of the JA pack's trained rows, read on
`sent_kanji_f0` (163 kana + 1 185 kanji) with the cold seed
(`seed_retrain_0930`) and `seed_fixed_1005_stick080` beside it. CPU numbers
plus one short render probe; no training. The user asked (10-06) whether
`pack init + offset stick` can be read as an absolute stick, whether the
offset stick + ball reading is the best one, and whether the raw init helps
or hurts the reading.

## The convention

`trained.pt` `delta` is an **offset**, not the row: offset = `raw ×
row_scale`, and **the row the model sees = pack row + offset** (checked
against `anima_cjk_vocab_pack_preview51.safetensors`). The pack row
is `anima_cjk_vocab_pack_punct` `ext_embed` — Qwen rows mapped by
procrustes-mix or a contextual char init (`preview51.json` `stats`). Every
"stick / spike" in the reports before this file is read on the offset.

## The decomposition

```
row_i = m_pack + s + (1 − α_i)·q_i + e_i
```

| term | what | size (kana / kanji) | read it with |
|---|---|---|---|
| `m_pack` | the family's mean pack row; frozen | \|99\| / \|101\| | — |
| `s` | the trained stick = the family's mean offset; a free bias | \|121\| / \|127\| | — |
| `q_i` | the row's pack spike (pack row − `m_pack`): Qwen semantics | \|159\| / \|183\| | meaning pairs |
| `α_i` | how much training shrank `q_i`: `−(o_i·q_i)/\|q_i\|²` | 0.38 / 0.32 (p5–p95 0.20–0.58) | — |
| `e_i` | the offset spike with `q̂_i` projected out: **the glyph** | ≈ \|209\| / \|211\| | bitmap similarity, look-alike pairs |

The absolute centre is `m_pack + s` — two sticks at right angles
(cos 0.02 / 0.01). Norms are read on the whole row, shape on `e_i`,
semantics on `q_i`.

## What each term is

**`m_pack` is T5's mean.** cos to the T5 table's mean 0.72 / 0.81, to the
EN caption direction 0.73 / 0.76. Along the T5-mean unit the rows carry
97–99 against T5 rows' own 105; the init supplies ~75 % of it. Render probe:
taking it out changes the composition and breaks the text on 7 / 7 strings.

**`s` is a free bias.** cos 0.13–0.23 to T5's mean, 0.05–0.09 once that
component is out; 0.31 of its energy on T5's top-256 PCs (isotropic 0.25,
T5 rows 0.56); its nearest T5 rows are number tokens and `<extra_id_*>`.
It is set by training and moved by scaling: cold 149 / 143 → × 0.8 → 119 /
114 → f0 121 / 127 (f0 regrew the kanji's, not the kana's), cos 0.99 to
f0's at every step.

**`q_i` carries meaning, not shape.** Spike cos on 68 meaning pairs (左右,
話語 …) and 46 look-alike pairs (日目, 土士 …), random 0.00 ± 0.07:

| spike | meaning pairs | look-alike pairs |
|---|---|---|
| pack `q_i` | **+0.21** (z 2.8) | +0.09 (z 1.1) |
| offset | +0.08 (z 1.1) | **+0.21** (z 2.9) |
| row (pack + offset) | +0.10 | +0.19 |

Mirror images. Training shrank every `q_i` by a third (α 0.32–0.38, 0.62–0.68
of it left) — 4–6 % of the offset's energy, even on and off T5's top PCs
(α_on 0.27 / α_perp 0.26 at 256), so a uniform shrink, not a targeted
erase. It was done in the cold seed (cos(offset, pack) −0.18 there); f0's
own increment sits at −0.04 to the init. Render probe: `q_i` projected out
(−30 in norm) keeps the composition and layout on 7 / 7 strings, with some
glyphs changed (r77 私有除聖 → 私自筆, r62's large line); swapped for
another row's `q`, 7 / 7 break. Read by eye, not by OCR.

**`e_i` carries the glyph.** Bitmap Spearman / visual-NN rank / look-alike
cos by view:

| view (family-centred) | kana ρ | kana NN rank | kanji ρ | kanji NN rank | kanji look-alike |
|---|---|---|---|---|---|
| pack only | 0.046 | 47 | 0.011 | 439 | 0.085 |
| row | 0.264 | 1 | 0.160 | 25 | 0.186 |
| offset | 0.273 | 1 | 0.174 | 16 | 0.211 |
| **offset − own `q̂`** (`e_i`) | **0.279** | 1 | **0.177** | 16 | **0.220** |
| cold seed offset | 0.256 | 1 | 0.167 | 15 | 0.216 |

(NN rank: where the bitmap's nearest glyph falls among a row's spike
neighbours, of 165 / 1 184; chance 82 / 592.) The shape structure is the cold
seed's; f0 adds a little to it. Cross-family look-alikes agree (カ力 ニ二
タ夕 ー一: spike cos 0.24–0.27, random 0). Radicals are weak as families
(within-radical spike cos 0.09 vs 0 ± 0.07, z 1.3); 門 is the one strong
group (0.31).

## The shell

The whole row sits on an **origin-centred shell**: |row| 275 / 289, cv
**0.051 / 0.065** — tighter than the offset (0.099 / 0.123), the offset
spike about its stick (0.086 / 0.114), a shuffled pack ↔ offset pairing
(0.073 / 0.080) and T5's own table (0.165). It is made by compensation:
corr(|p|² + |o|², 2 p·o) −0.82 / −0.86 — a longer init gets a more
negative offset along it. No better centre: a least-squares centre on the
mean / pack mean / T5 mean / stick directions gives cv 0.051 / 0.064; a scan
along the mean only improves by running off to infinity. The cold seed has
the same shell (0.051 / 0.066).

Energy of |row|²: pack 0.47 / 0.52, stick 0.19, spike 0.58 / 0.54, the
pack·spike cross term −0.25 / −0.26 (the shrink). Pairwise inner product
between rows (mean row cos 0.33 / 0.32): stick·stick 0.59 / 0.61, pack·pack
0.39 / 0.38, spike·spike ≈ 0 — the row cos every report quotes is the two
sticks.

## Not there

- No low-rank layout: spike participation ratio 79 / 196; top PC 3–5 %.
- Not on T5's manifold: rows put 0.28–0.35 of their energy on T5's top-256
  PCs (T5 rows 0.56); nearest-T5 cos 0.29–0.31; 37 % of rows have a
  `<extra_id_*>` sentinel as nearest T5 row (offsets 0 %).
- The hiragana / katakana axis is the init's more than training's
  (split-half d′ pack 2.98, offset 2.51, row 2.70).

## Use

- **Reading rows**: norms on the row, glyph identity on `e_i`, meaning on
  `q_i`. A report's "row cos" should say which; the uncentred row cos is
  mostly `s` and `m_pack`.
- **The init**: keep it in the row (the model needs `m_pack`; `q_i` swapped
  breaks the render), project `q̂_i` out when measuring shape. A linear
  regression on the pack explains ≤ 4 % of the offset and does not help.
- **New rows (`plan.md`)**: a cold row's `m_pack` and `q_i` come with its
  pack row and `s` is the family's; only `e_i` has to be learned. A
  shape-neighbour init (start `e_i` near the `e` of the existing row whose
  bitmap is nearest) is defined on this decomposition — untested.

## Open

- The render probe was 7 strings × 4 arms read by eye
  (`fable/probe_init/sheet.png`, job `20261006-155024-420db4`, 4 min); `q_i`
  out vs in has not been read by OCR or on the ruler.
- Whether `q_i`'s semantics help anything (content conditioning, a word's
  read) is not measured.
- Frequency was read only through Qwen ids (|f0 increment| ρ −0.39 with it:
  exposure, not the init).

Scripts and full outputs (not in the repo): the session scratchpad's
`fable/absgeom{,2,3}.py` / `.out`, `probe_init.py`; the offset / row
first read `geom.py`.
