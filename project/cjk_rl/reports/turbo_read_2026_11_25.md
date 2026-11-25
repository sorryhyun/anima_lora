# The string read on the turbo itself — merged15 + the turbo v1.1 delta (2026-11-25)

Fresh renders on the 4-step stack the turbo line ships next (`m15v11`), read by today's reader:
the dialogue ruler with four seeds a string and its EN references, and a kanji grid (singles and
words in a bubble). The user's call (11-25): measure on the turbo directly instead of the reads
of record.

## Facts

Probe `../probes/turbo_read.py`, daemon job `20261125-084840-a2aa0d`; result
`../results/20261125-0930-turbo_read-m15v11/` (local): `result.json`, `alignments.json`,
`sheets/`. Renders `output/cjk_rl/turbo_read/m15v11/` (local).

- Stack: `anima_merged15` + `models/extracted/anima_turbo_v1.1_delta_r96_asvd_adaln.safetensors`
  at 1.0 (the ruler's `MODELS["m15v11"]`), 4 Euler steps, cfg 1, shift 3, bf16 DiT with the fp32
  stream, VAE decode, the punct pack, routed.
- Arms: `en` (the EN line, rows off), `seed_1008` (jp_v1's rows), `seed_retrain_0930` (the old
  floor's rows).
- Reader: the tuned reader as `common.readers.Readers` loads it today. Alignment, pick and null
  as in `position_read_2026_11_25.md`; the null is taken inside each seed.
- Cost: 2 248 renders in 33.4 min (0.89 s a render, 0.30 s of it sampling), the reads in about
  9 min (26 crops a second with detection on `ruler/seed_1008`).
- These numbers do not compare with `position_read_2026_11_25.md`: another string set (the
  rebuilt ruler), another reader, another stack.

### The ruler (96 strings × seeds 0 – 3, captioned prompts; EN on the 66 strings with a line)

| arm | renders | cer | hit (null) | exact | ≤ 2 edits | hit per seed |
|---|---|---|---|---|---|---|
| en | 264 | 0.83 | 0.28 (0.12) | 3 | 12 | 0.27, 0.35, 0.26, 0.24 |
| seed_1008 | 384 | 0.93 | 0.14 (0.10) | 2 | 44 | 0.12, 0.13, 0.15, 0.15 |
| seed_retrain_0930 | 384 | 0.92 | 0.16 (0.11) | 1 | 42 | 0.15, 0.15, 0.16, 0.17 |

By token count, cer / hit (null):

| tokens | en | seed_1008 | seed_retrain_0930 |
|---|---|---|---|
| 3 – 4 | 0.91 / 0.24 (0.10), n 52 | 0.91 / 0.24 (0.12), n 92 | 0.89 / 0.27 (0.15) |
| 5 – 9 | 0.83 / 0.29 (0.12), n 108 | 0.94 / 0.17 (0.10), n 128 | 0.94 / 0.19 (0.12) |
| 10 – 20 | 0.77 / 0.29 (0.14), n 72 | 0.93 / 0.11 (0.09), n 152 | 0.92 / 0.13 (0.10) |

By position, hit / null, all strings; then head third above tail third : the reverse, and the
null's split:

| arm | first | head third | middle | tail third | last | head : tail, null |
|---|---|---|---|---|---|---|
| en | 0.40 / 0.11 | 0.37 / 0.13 | 0.25 / 0.13 | 0.22 / 0.10 | 0.23 / 0.06 | 97 : 33, 205 : 126 |
| seed_1008 | 0.21 / 0.06 | 0.14 / 0.08 | 0.14 / 0.12 | 0.13 / 0.10 | 0.16 / 0.11 | 98 : 85, 172 : 239 |
| seed_retrain_0930 | 0.20 / 0.08 | 0.16 / 0.09 | 0.16 / 0.13 | 0.15 / 0.11 | 0.22 / 0.14 | 107 : 105, 208 : 258 |

- seed_1008, short bin: head third 0.34 / 0.08, tail third 0.17 / 0.15; read / asked 1.92, 1.09
  glyphs inserted per asked glyph. Long bin: hit 0.10 against null 0.09, read / asked 0.97, 0.70
  substituted.
- seed_1008 by class, hit / null: hiragana 0.16 / 0.12 (1 724), kanji 0.03 / 0.01 (444), katakana
  0.10 / 0.03 (180), small kana 0.18 / 0.18 (272).
- Substitution kinds follow the null's on both row arms (kana → another kana 0.61 against 0.62);
  voicing, size and script swaps together are at or under 1 %.

### The kanji grid (768², `<plain prompt>, speech bubble, japanese text. Japanese text reads as "<key>".`, p00 / p01 × seeds 0 / 1)

Singles: the key read alone in a box (exact), or anywhere in a box's read (on the page). The null
is the key on the page of another key of its group, same prompt and seed.

| group (renders) | arm | exact | on the page | null |
|---|---|---|---|---|
| hiragana, 24 (96) | seed_1008 | 2 | 44 | 0.22 |
| | seed_retrain_0930 | 0 | 48 | 0.22 |
| kanji, the 32 frequent of the 1 185 (128) | seed_1008 | 0 | 12 | 0.003 |
| | seed_retrain_0930 | 0 | 14 | 0.005 |
| kanji, the 32 rare of the 1 185 (128) | seed_1008 | 0 | 0 | 0 |
| | seed_retrain_0930 | 0 | 1 | 0 |
| kanji, 32 of the 225 (128) | seed_1008 | 0 | 0 | 0 |
| | seed_retrain_0930 (no trained row: the pack's) | 0 | 1 | 0 |

- Every group and arm letters 32 – 37 glyphs a page in 2.1 – 2.5 boxes.
- The frequent kanji found on the page, seed_1008: 入 3, 生 2, 太 2, 行, 二, 安, 王, 走.
- A single kanji read in the place of the asked one: 3 – 5 pairs a group, too few for the bitmap
  comparison (overlap 0.33 – 0.41 against 0.29 – 0.36 shuffled).

Words (32 kanji words, 128 renders, 296 glyphs):

| arm | exact | hit (null) | kanji hit (null) |
|---|---|---|---|
| seed_1008 | 0 | 0.17 (0.04) | 0.14 (0.02) |
| seed_retrain_0930 | 2 | 0.23 (0.04) | 0.20 (0.02) |

Claude's read of the sheets, arm labels visible:

- `sheets/kanji_kanji_frequent_seed_1008.png` (p00, seed 0, 32 kanji): every render draws the
  same two bubbles filled with several columns of small pink text. The page and the lettering
  change little with the asked kanji.
- `sheets/ruler_mid.png` (8 strings, seed 0): the bubbles of all three arms hold several columns
  of dialogue; the EN arm letters Japanese or mixed script on about half of the rows.

## Interpretation

- **On this stack the quoted string barely steers what is lettered, in either script.** The row
  arms sit 0.04 – 0.05 above their null overall and at the null on long strings; the EN reference
  is at cer 0.83. What the pages letter is the stack's own dialogue.
- What gets through is the start of short strings (first glyph 0.21 against 0.06) and, for
  kanji, the frequent ones inside words.
- The old floor's rows read a little above jp_v1's on every hit number (overall 0.16 against
  0.14; the seeds' ranges, 0.15 – 0.17 and 0.12 – 0.15, do not overlap). The gap is small, in the
  direction `position_read` found on the base.
- Kanji singles are not a usable identity meter here: the stack fills a bubble with a sentence
  whatever is asked, so a lone glyph is almost never drawn. Within that limit, jp_v1's 225 added
  rows read as the untrained pack rows do (0 and 1 of 128); the grid gives no sign that they act
  on this stack. Words are the better kanji read (0.14 – 0.20 against 0.02).
- For the RL line: a reward on this policy starts at the null for mid and long strings, and the
  policy does not follow an EN line either. Rows alone are an unlikely lever here; what has to
  learn to follow the quote is on the DiT side of this stack (the soup or the turbo delta).
