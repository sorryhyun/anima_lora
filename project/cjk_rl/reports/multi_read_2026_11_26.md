# Multi read — two or three short strings in one clause, on kmix's rows and in EN (2026-11-26)

Two parts: the glyph stage's read scenes, then the pool's templates of two dialogue strings.

# On the read scenes

Forward reads only, no training: `wmulti_1126_1252` (kana strings on `kmix_1126_1158`'s rows)
and `enmulti_1126_1258` (the same captions with EN words). Opened on the user's suggestion for
the sentence run (11-26): "짧은 단어 여러개 들어가는 케이스도 있으면 어떨까 해. 예를 들어 "ああっ"
"こんいちは" 식으로?" Counted by `probes/multi_read.py`.

## Facts

- A caption is `<scene, bubble frame>. Text reads as "A", "B".` (or three strings), the form the
  training captions have for several bubbles. In the prompt pool
  (`output/cjk_rl/pool/pool.json`, local) 634 of the 1 108 items with a dialogue clause hold two
  strings or more.
- Kana: `wread_1126_1207`'s 36 strings of 2 – 4 glyphs as 18 pairs in both orders and 12 triples,
  8 pages a caption (read scenes 6, 12, 45, 15, seeds 0 and 1). EN: `enbubble_1125_2258`'s 16
  words as 8 pairs in both orders and 5 triples, the same scenes and seeds.
- "Own": a string read exactly by some box. Each string is also aligned to a box of its own
  (the assignment with the highest summed sequence score) for its positions hit. Tuned reader,
  strings as written; hayai's counts and the folded ones are within 3 strings of these in every
  kana cell (`multi_read.json` in each result).
- Spread unknown (no twin). Results (local):
  `results/20261126-1256-row_line-wmulti_1126_1252/`,
  `results/20261126-1300-row_line-enmulti_1126_1258/`; a labelled sheet a caption in each
  `sheets_kmix/`.

Pages:

| | pages | every string own | one string own or more | boxes a page | pages with 0 / 1 / 2 / 3+ boxes |
|---|---|---|---|---|---|
| kana, 2 strings | 288 | 0 | 40 | 1.24 | 2 / 223 / 56 / 7 |
| kana, 3 strings | 96 | 0 | 9 | 1.60 | 1 / 50 / 31 / 14 |
| EN, 2 words | 128 | 0 | 64 | 1.02 | 17 / 92 / 18 / 1 |
| EN, 3 words | 40 | 1 | 20 | 1.55 | 5 / 13 / 17 / 5 |

Strings, own over the pages that ask for them (positions hit in the assigned box):

| | alone | in a pair, first | in a pair, second | in a triple, first / second / third |
|---|---|---|---|---|
| kana | 86 / 288 (0.61) | 25 / 288 (0.31) | 15 / 288 (0.21) | 3, 4, 2 of 96 (0.26, 0.12, 0.12) |
| EN | 30 / 96 (0.55) | 45 / 128 (0.46) | 19 / 128 (0.26) | 16, 4, 6 of 40 (0.52, 0.23, 0.22) |

- The kana alone are `wread_1126_1207`'s kmix arm on the same strings, scenes and seeds. The EN
  words alone are `enbubble_1125_2258` on the scenes and seeds the two runs share (6, 45, 15 at
  seeds 0 and 1); on those three scenes the EN pairs read 29 and 10 of 96 in the first and
  second place.
- Kana by length, alone → in a pair: 2 glyphs 52 / 96 → 33 / 192, 3 glyphs 21 / 96 → 4 / 192,
  4 glyphs 13 / 96 → 3 / 192.
- Pair pages with one box. Kana, 223 pages: the box is 2.97 glyphs long on average (the
  strings 2.99 and 3.01); it is closer to the first string on 114, to the second on 72, level
  on 37; it holds glyphs that only the first string has on 94, only the second on 62, of both
  on 39, of neither on 28. EN, 92 pages: 4.76 letters; closer to the first word on 49, the
  second on 23, level on 20; letters only the first has on 48, only the second on 23, of both
  on 5, of neither on 16.
- Claude's read, arm and caption labels visible, four sheets viewed (`たぷん+んぃ`,
  `はぁ+ハァ+ピッ`, `ちゃ+しめたっ`, EN `stop+sale`): the kana pair pages show one bubble holding
  a short string made of both strings' glyphs (`たん`, `ぷん`, `たぃ` for `たぷん`, `んぃ`; `ちめっ`,
  `しゃ` for `ちゃ`, `しめたっ`); the EN pages show one bubble with `stop` and no `sale`; the
  triple of three 2-glyph strings draws two or three bubbles on 5 of its 8 pages, one page
  reading `はぁ` / `ハ` / `ピッ`.

## Interpretation

- On the glyph stage's scenes the frozen model draws one bubble for a clause of two strings,
  for EN words as for kana, so no row training on these scenes can put two strings in two
  boxes: the second string has nowhere to be read. The case needs scenes that draw a bubble a
  string before it is a question about the rows.
- What differs between the scripts is what the one bubble holds. EN mostly writes one of the
  two words whole, the first twice as often as the second. The kana rows write a string of one
  string's length that mixes the two: a row is not held to its own quote. A run on single
  strings never shows the rows this, which is a reason to have such captions in the run once
  the scenes allow them.
- In a single-string run, the same looseness would show only as a wrong order or a dropped
  glyph inside the string, which the plan already reads by position.

# On the pool's two-string templates

`tread_1126_1306` (`probes/templ_read.py`, forward only), on the user's call (11-26, of the
two sources of scenes the first part left open): "다중 문자열 템플릿이 모델도 텍스트를 잘 쓸수
있지 않을까 싶긴 해. 1번 ㄱㄱ"

## Facts

- 24 templates drawn from the pool's 52 stems with two dialogue strings and no SFX clause, at
  the two commonest shapes (672 × 896, 640 × 928). The caption is the template's, its two
  strings swapped: a kana pair of `wmulti_1126_1252` or an EN pair of `enmulti_1126_1258`, as
  `"A", "B"`, `"B", "A"`, `"A"` alone and `"B"` alone, seeds 0 and 1; kmix's rows. The clause
  stays `Japanese text reads as` for the EN words too. 96 pair pages and 96 alone pages a
  kind. Spread unknown. `results/20261126-1310-templ_read-tread_1126_1306/` (local; a labelled
  sheet a template in `sheets/`).
- Tuned reader, strings as written; hayai's counts are within 1 string of these.

| | pair pages | every string own | one string own or more | boxes a page | 0 / 1 / 2 / 3+ boxes | boxes a page, the string alone | glyphs read over glyphs asked, pair / alone |
|---|---|---|---|---|---|---|---|
| kana | 96 | 0 | 2 | 2.27 | 9 / 30 / 19 / 38 | 2.05 | 1.63 / 3.21 |
| EN | 96 | 0 | 2 | 1.92 | 26 / 14 / 23 / 33 | 1.71 | 1.34 / 2.63 |

| own (positions hit in the assigned box) | alone | in a pair, first | in a pair, second |
|---|---|---|---|
| kana | 8 / 96 (0.33) | 1 / 96 (0.18) | 1 / 96 (0.16) |
| EN | 1 / 96 (0.08) | 2 / 96 (0.09) | 0 / 96 (0.05) |

- The same kana strings alone on the read scenes: 86 / 288 (0.61). The same EN words alone
  on the read scenes under `Text reads as`: 30 / 96 (0.55).
- Templates whose four pair pages all hold two boxes or more: 5 of 24 for kana, 7 for EN.
- Claude's read, labels visible, two sheets viewed (`t21`, `t14`): `t21` writes small
  vertical lines of handwriting beside the figure and no bubble, a line longer than the
  string asked and partly following it (`どうやかしるか` for `どうかな`); `t14` draws two to
  four small bubbles of one or two glyphs with hearts, some near a string asked (`ヘッ` for
  `ヘヘッ`, `エッホ` once alone) and the rest not asked for.

## Interpretation

- The templates do put more than one box on a page, but the count is the template's and
  not the clause's: about as many boxes when one string is asked as when two are, and two to
  three times the glyphs asked. What they hold is the lettering the template's tags and
  artist bring (moans with hearts, side notes, a signature), which is the invented text the
  line's goal is to be rid of.
- The asked kana string is read far less often here than on the text-free scenes, alone as
  well as in a pair. As scenes for a row's loss they are worse than what the run has: the
  assignment would mostly hand a string a box of text nobody asked for.
- The EN arm says nothing about the bubble count here: EN words are not written under
  `Japanese text reads as` even alone. The run cannot say which of the clause's wording, the
  templates' text tags, the shape and the artist makes the difference from the read scenes;
  they changed together.
