# Where the asked string is lost — position, substitutions and the EN level (2026-11-25)

A CPU read of the dialogue ruler's reads of record: each asked string aligned to the best read
of one bubble, on seven row tables on the base, on jp_v1's rows on the soup and the turbo, and
on the EN references' own text. No render, no new read.

## Facts

Probe `../probes/position_read.py` (its docstring holds the pick and alignment rules); result
`../results/20261125-0829-position_read/` (local): `result.json`, `alignments.json`.

Sources, all from `../../cjk_anima_reseed/results/` (local):

- `20261008-2025-ruler-sensitive-seed_1008` and `20261006-1502-ruler-sensitive-glyph_f1`: seven
  tables on the base, 96 strings (32 a bin), seed 0, sensitive prompts.
- `20261113-1417-ruler-sensitive-turbo_m14v3_only64`: jp_v1's rows on the base, merged14
  (28 steps) and merged14 + turbo v3 (4 steps), 64 of the 96 strings.
- `20261113-2057-ruler-captioned-rl_p1_only64`: the captioned set, 64 strings, on the turbo.
- The EN references' reads (`output/cjk_anima_reseed/ruler/sensitive/en*/reads.json`, local)
  against their hand-written lines (`en.json`).

`ruler.json` was rebuilt on 11-13 and keeps 19 of the 96 strings; the strings here come from
each result's own records. The sensitive and the captioned reads are two string sets.

A hit is an asked glyph matched in the Levenshtein alignment to the picked bubble read. The null
aligns each string to the page of another string of its bin (three shifts). Readers: `sfx` is
the tuned reader of the read's date, `vl` stock VL16 (kept on the reads up to 11-13 14:17). On
the base tables the two agree on 90 – 93 % of positions (both hit 0.25 – 0.37, one only
0.06 – 0.10); on merged14 on 86 % (both 0.16, one only 0.14). Tables below are `sfx` unless
marked. Spread between seeds: unknown (one seed a string).

### Position

Hit rate / null, all strings of the arm. The last column is the strings whose head third reads
above their tail third : the reverse, then the same split for the null, and Fisher's p between
the two splits.

| arm (n) | first | head third | middle | tail third | last | head : tail, null, p |
|---|---|---|---|---|---|---|
| retrain_kana (96) | 0.46 / 0.04 | 0.34 / 0.06 | 0.30 / 0.09 | 0.46 / 0.10 | 0.64 / 0.10 | 20 : 40, 35 : 57, 0.61 |
| seed_retrain_0930 (96) | 0.65 / 0.05 | 0.42 / 0.06 | 0.34 / 0.07 | 0.51 / 0.10 | 0.70 / 0.10 | 21 : 33, 30 : 54, 0.72 |
| preview51 (96) | 0.52 / 0.07 | 0.35 / 0.08 | 0.24 / 0.10 | 0.47 / 0.12 | 0.58 / 0.12 | 19 : 37, 39 : 72, 1.0 |
| sent_whole (96) | 0.45 / 0.07 | 0.33 / 0.06 | 0.30 / 0.12 | 0.33 / 0.12 | 0.55 / 0.13 | 25 : 28, 31 : 75, 0.034 |
| sent_kanji_f0 (96) | 0.49 / 0.04 | 0.37 / 0.05 | 0.28 / 0.11 | 0.33 / 0.12 | 0.51 / 0.13 | 31 : 28, 30 : 76, 0.0025 |
| sent_kanji_pres (96) | 0.44 / 0.07 | 0.32 / 0.06 | 0.26 / 0.08 | 0.28 / 0.12 | 0.38 / 0.15 | 39 : 30, 38 : 87, 0.0004 |
| jp_v1 (96) | 0.46 / 0.08 | 0.32 / 0.07 | 0.29 / 0.09 | 0.29 / 0.13 | 0.43 / 0.15 | 35 : 28, 36 : 79, 0.0023 |
| jp_v1, the 64 | 0.53 / 0.08 | 0.36 / 0.06 | 0.27 / 0.07 | 0.31 / 0.13 | 0.48 / 0.15 | 25 : 17, 23 : 51, 0.0034 |
| jp_v1 on merged14 (64) | 0.41 / 0.09 | 0.26 / 0.11 | 0.18 / 0.11 | 0.22 / 0.17 | 0.23 / 0.18 | 23 : 14, 33 : 58, 0.010 |
| jp_v1 on the turbo (64) | 0.28 / 0.10 | 0.21 / 0.11 | 0.22 / 0.09 | 0.21 / 0.13 | 0.28 / 0.16 | 19 : 17, 44 : 54, 0.44 |
| captioned: jp_v1 on the turbo (64) | 0.44 / 0.08 | 0.29 / 0.10 | 0.12 / 0.14 | 0.15 / 0.13 | 0.20 / 0.16 | 31 : 13, 36 : 47, 0.0049 |
| captioned: rl_p1 on the turbo (64) | 0.41 / 0.06 | 0.28 / 0.10 | 0.20 / 0.13 | 0.18 / 0.13 | 0.17 / 0.17 | 27 : 12, 39 : 50, 0.012 |

- Long bin, jp_v1 (32 strings): thirds 0.24 / 0.24 / 0.17 against null 0.04 / 0.11 / 0.10; head :
  tail 16 : 9 against the null's 11 : 37 (p 0.0009). seed_retrain_0930: 0.35 / 0.25 / 0.47
  against 0.05 / 0.07 / 0.08; 7 : 16 against 11 : 22 (p 1.0).
- Hit share per long string, counts in 0 / under 0.2 / 0.2 – 0.4 / 0.4 – 0.6 / 0.6 – 0.8 / over
  0.8: seed_retrain_0930 0 / 4 / 16 / 8 / 3 / 1, jp_v1 1 / 14 / 16 / 0 / 1 / 0.
- Adjacent hit pairs inside a string over the count expected from its hits placed at random
  (strings of 5 or more glyphs): seed_retrain_0930 1.08, jp_v1 1.23, merged14 1.22, the turbo
  1.43, captioned 1.30. No interval computed.

### Length

Per asked glyph, the picked read (null in brackets for the hit).

| arm, bin | read / asked | hit (null) | deleted | substituted | inserted |
|---|---|---|---|---|---|
| seed_retrain_0930, short | 1.20 | 0.72 (0.09) | 0.04 | 0.24 | 0.24 |
| jp_v1, short | 2.09 | 0.59 (0.15) | 0.07 | 0.35 | 1.15 |
| seed_retrain_0930, long | 0.82 | 0.36 (0.07) | 0.28 | 0.37 | 0.09 |
| jp_v1, long | 0.62 | 0.22 (0.08) | 0.40 | 0.38 | 0.02 |
| jp_v1 on merged14, long (22) | 1.23 | 0.16 (0.12) | 0.08 | 0.76 | 0.31 |
| jp_v1 on the turbo, long (22) | 1.04 | 0.14 (0.09) | 0.17 | 0.68 | 0.21 |

Short-bin read / asked for the seven tables in order: 1.15, 1.20, 1.85, 1.57, 1.37, 1.76, 2.09.

### Glyph class and substitutions

Hit / null by the asked glyph's class:

| arm | hiragana | voiced hiragana | small kana | katakana | kanji |
|---|---|---|---|---|---|
| seed_retrain_0930 | 0.49 / 0.10 (446) | 0.34 / 0.05 (59) | 0.55 / 0.11 (60) | 0.36 / 0.00 (45) | 0.22 / 0.01 (125) |
| jp_v1 | 0.35 / 0.12 | 0.20 / 0.10 | 0.35 / 0.14 | 0.27 / 0.04 | 0.16 / 0.01 |
| jp_v1 on merged14 | 0.27 / 0.17 (266) | 0.27 / 0.15 (34) | 0.27 / 0.17 (34) | 0.09 / 0.03 (34) | 0.07 / 0.01 (77) |
| jp_v1 on the turbo | 0.26 / 0.15 | 0.27 / 0.11 | 0.29 / 0.16 | 0.23 / 0.02 | 0.03 / 0.00 |

Substitution pairs by kind, share of the arm's substitutions, own page against null:

| kind | seed_retrain_0930 (280) | its null (1 553) | jp_v1 (279) | its null (1 199) |
|---|---|---|---|---|
| kana → another kana | 0.52 | 0.61 | 0.59 | 0.63 |
| kana → kanji | 0.17 | 0.18 | 0.15 | 0.15 |
| kanji → kanji | 0.15 | 0.05 | 0.10 | 0.03 |
| kanji → kana / other | 0.09 | 0.12 | 0.11 | 0.15 |
| voicing, size, the same kana in the other script | 0.05 | 0.00 | 0.02 | 0.00 |

No pair occurs more than twice in either arm (seed_retrain_0930: あ→な, さ→き, き→さ, ち→と …;
jp_v1: と→な, と→か, い→た, っ→て, ろ→て).

### The EN references

The EN line as the asked string, by its T5 token count, beside the JA strings of that count (one
row a glyph). cer is the smallest edit distance over the bubble candidates; EN read by `vl`, JA
by `sfx`. Base, 96 strings:

| tokens | EN: n, chars | EN cer | EN hit (null) | EN exact / ≤ 2 | 0930: n, cer, hit (null), exact / ≤ 2 | jp_v1: cer, hit (null), exact / ≤ 2 |
|---|---|---|---|---|---|---|
| 3 – 4 | 24, 7.4 | 0.52 | 0.73 (0.30) | 8 / 8 | 21, 0.42, 0.76 (0.09), 7 / 18 | 0.61, 0.59 (0.16), 3 / 12 |
| 5 – 9 | 36, 15.2 | 0.44 | 0.67 (0.21) | 6 / 11 | 31, 0.60, 0.54 (0.09), 1 / 10 | 0.68, 0.41 (0.09), 0 / 4 |
| 10 – 20 | 25, 26.8 | 0.50 | 0.56 (0.22) | 1 / 3 | 39, 0.71, 0.35 (0.07), 0 / 0 | 0.77, 0.23 (0.09), 0 / 0 |
| all | 96, 15.1 | 0.46 | 0.63 (0.23) | 21 / 31 | 96, 0.60, 0.43 (0.08), 10 / 32 | 0.70, 0.30 (0.10), 5 / 20 |

The same on the three generators (the 64 strings; EN `vl`, JA `sfx`):

| generator | EN cer | EN hit (null) | EN exact | EN at 10 – 20 tokens: cer, hit (n 13) | jp_v1 cer | jp_v1 hit (null) |
|---|---|---|---|---|---|---|
| base, 28 steps | 0.46 ¹ | 0.63 (0.23) ¹ | 21 / 96 ¹ | 0.50, 0.56 ¹ | 0.69 | 0.32 (0.09) |
| merged14, 28 steps | 0.56 | 0.64 (0.23) | 9 / 64 | 0.43, 0.64 | 0.84 | 0.22 (0.13) |
| merged14 + turbo v3, 4 steps | 0.74 | 0.32 (0.12) | 7 / 64 | 0.80, 0.30 | 0.87 | 0.21 (0.11) |

¹ all 96 strings (n 25 at 10 – 20 tokens).

EN by position, long bin (`vl`), first / thirds / last: base 0.75 / 0.72, 0.45, 0.47 / 0.41 (null
0.09 – 0.23); merged14 0.82 / 0.75, 0.64, 0.60 / 0.64; the turbo 0.50 / 0.37, 0.23, 0.32 / 0.27.

## Interpretation

- Position gives no clean tail loss on the base. The cold-era tables (retrain_kana,
  seed_retrain_0930, preview51) hit both ends above the middle and split head against tail as
  the null does. The tables after a warm dialogue pass split toward the head where the null
  leans to the tail, and long strings come out short (0.62 of the asked length, 0.40 deleted),
  so those tables do show a sequence-level loss along the string, a mild one.
- The substitutions are not near misses. Their kinds follow the null's, look-alike kinds are
  2 – 5 %, and no pair repeats: the bubble mostly holds other text, not a damaged copy of the
  asked string. A per-glyph stage aimed at confusable kana has little to fix. Kanji is the
  exception: its null is near 0, its hit rate is the lowest of the classes, and a missed kanji
  is drawn as another kanji two to three times as often as on a page not asked for it.
- The frozen adapter and DiT write a quoted EN line at about the old floor's level on 3 – 4
  tokens and above it from 5 tokens on (the EN null is the higher one, 0.21 – 0.30 against
  0.07 – 0.09, so the gap in hit rate overstates it). So the rows have room up to the EN level. That level is
  itself 1 exact line of 25 at 10 – 20 tokens: exact long lines are outside the frozen model for
  either script at this bubble size.
- The warm passes moved the string read down at every length against seed_retrain_0930 (hit
  0.43 → 0.30, exact 10 → 5, twice the asked text in short bubbles). `status.md` § 2 has the same
  on cer; this read adds that it holds in order and inside one bubble.
- merged14 keeps the quoted EN line (hit 0.64, as the base) and loses jp_v1's string to near
  the null. The loss is specific to the rows, not to the soup's ability to follow a quoted line.
- The turbo loses the EN line too (hit 0.64 → 0.32). `ruler_turbo_2026_11_13.md` called the
  turbo a fair proxy of its teacher on JA cer; on JA both sit near the null, so that read could
  not show it. On the RL policy the EN reference is at cer 0.74.

Not separated: reader error from render error on any single glyph (the two readers agree on
about 90 % of positions on the base, less on the soup); one seed a string.
