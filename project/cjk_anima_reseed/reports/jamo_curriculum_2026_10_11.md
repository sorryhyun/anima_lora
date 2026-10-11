# The jamo curriculum — 2026-10-11

User 10-11: train the standalone jamo (ㅎ ㅏ ㄱ) first, then the J64 factor
model on top of them. Phase 1 (`jamo_phase1_2026_10_10.md`) carried glyph
form and jamo positions to held-out syllables but not their identity, and
its misses were shape neighbours (ㄱ↔ㅋ, ㅅ↔ㅈ, ㅇ↔ㅎ, ㅏ↔ㅓ↔ㅕ). A jamo drawn
alone at full size is a dense identity signal the factors get only
indirectly from 64 syllables.

(§ 5 of the phase-1 report had this idea down as "jamo identity in T5 rows"
and a jamo-stream render probe; that was a misreading — corrected there.)

## 1. Runs

| run | what | steps |
|---|---|---|
| `jamo_lone` | the 51 compatibility jamo (U+3131–U+3163: every cho / jung / jong letter, each its own ext row) cold free rows on seed_1008; lone / bubble1 / grid tiers, Hangul only | 56 a row, 2 856 (23 min) |
| `jamo_j64_lone` | `jamo_j64` (its data, 56 a row, 3 584) with `factor_init = {from = "jamo_lone", scale = 0.577}`: C[cho, every cls] / V[jung] / F[jong] ← 0.577 × the letter's full row (pack + Δ) centred over the 51; b at 0 | 3 584 (25 min) |

A compatibility jamo holds less ink than a block at the same font px, so its
√(ink area) runs ~30 % under a syllable's (grid_29 22 px vs J64's 30); the
table is unchanged, and grid_44's 40 px gate kept 132 / 580 (the dense
letters). Init sizes (`train_record.json`): lone row 224, centred 204 (little
shared component), one factor ~118 (J64's trained C / V / F: 109 / 102 / 96),
composed Δ 185 at step 0 (J64 ends at 300). Final losses equal (in-box 0.149
vs J64's 0.151).

## 2. Stage 1: the jamo alone

`results/20261011-1151-jamo_lone-render/` (each letter alone in a KO bubble,
seed 0; by eye): **31 / 51** draw their letter as the main glyph.

| group | right | misses |
|---|---|---|
| cho letters (19) | 15 | ㄲ → "TI", ㅁ (tiny), ㅃ, ㅉ |
| vowels (21) | 16 | ㅓ, ㅕ, ㅙ, ㅚ, ㅛ |
| final clusters (11: ㄳ ㄵ ㄶ ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ ㅄ) | 0 | text |

seed_1008 draws text for nearly all 51.

## 3. Stage 2: H (32 held out), J64 vs J64-lone

`results/20261011-1221-jamo-read/` (J64's renders are phase 1's, cached;
re-counted here by the same eye, one seed, ±1–2):

| | J64 | J64-lone |
|---|---|---|
| one glyph drawn | 17 | **10** |
| exact | 1 (할) | 2 (고 그) |
| initial right (of one-glyph) | 6 / 17 | 5 / 10 |
| vowel right (of one-glyph) | 5 / 17 | 2 / 10 |

(Phase 1 counted J64's one-glyph renders as 19; the same images read 17
today.) J64-lone falls back to text or two-glyph words on more of H; the
exact count is inside the noise. The words: J64-lone draws one block for
none of 정리 / 소리 / 아래 / 문제 / 시장 except 아래 (저+ㅐ).

## 4. Where the factors went (CPU)

- The init half-survives: J64-lone's final factors vs their lone start, mean
  cos C 0.51 / V 0.59 / F 0.55, norms × 1.13–1.27.
- **J64 cold's factors are orthogonal to the lone directions**: mean cos
  C −0.00 / V 0.03 / F −0.01. Trained from Δ 0, the factor for "ㄱ as the
  initial of a block" lands nowhere near the row that draws ㄱ standing alone.
- b ends the same (145 vs 147).

## 5. Reading

The curriculum as an init does not carry identity into the block. The
standalone row of a jamo and the factor that places it in a syllable are
different directions; starting there costs H its one-glyph rate (the
factors keep half of a "draw this letter alone" meaning) and buys no exact
syllables. The tied form (C = e[cho] + P[cls], e shared with the lone rows)
was gated on this init helping; it did not, and § 4 says the two uses would
pull e apart.

Not tested: a smaller scale (0.577 was fixed in advance), training the lone
items inside the J64 run, more than one seed.

## Code

`reseed/config.py` `factor_init`; `reseed/jamo.py` `COMPAT`, `compat_rows`,
`Jamo.init_from`; `reseed/trainer.py` loads the `from` run's rows (pack + Δ)
and records the init norms; `probes/kozh_render.py` registers its run as an
arm (any `lang` run reads). `src/data/inventory.py`: the JA render corpus
(`post_image_dataset/render/ja/`) was deleted 10-10 — `corpus_lines` returns
`[]` without it (only the eval strings read it, and `build_pools` drops
them), `word_inventory(n=0)` no longer reads it; a JA build now draws its
pools from another rng state than the builds before.

## 6. The reader (`probes/jamo_vl.py`, 10-11)

v4 = anime_tools 0.7.10's `SfxReader` (the `ocr-reader` line's manga
reader). Calibration on font-drawn lone glyphs (4 faces, 64 px, in a bubble;
`results/20261011-1352-jamo-vl-cal/`), free read:

| set | v4 | stock |
|---|---|---|
| H (32 common) | **71 %** (print faces 27–29 / 32, Nanum Pen 8 / 32) | 57 % |
| J128 − J64 | 55 % | 45 % |
| J64 (rare blocks) | 32 % | 28 % |
| the 51 jamo | 0 % (ㄱ → フ) | 0 % |

v4 pulls a rare syllable to a common neighbour (붙→분, 폐→페, 묶→뭐); it
cannot read a lone jamo. The forced read (a candidate-constrained beam) added
71 → 78 % on H at twice the time and is off by default.

H renders, v4 free (seed 0):

| arm | exact | read as one syllable | cho / jung / jong right (of those) |
|---|---|---|---|
| J64 | 1 | 17 | 4 / 5 / 7 |
| J96 | 2 | 15 | 6 / 5 / 7 |
| J128 | 3 | 10 | 5 / 5 / 7 |
| F64-reg | 0 | 2 | 1 / 0 / 0 |
| J64-lone | 1 | 7 | 4 / 2 / 3 |

Same order as the eye counts; the words 0 / 5 everywhere.
