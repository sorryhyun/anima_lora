# piece coverage — what a row budget buys: single kanji vs multi-glyph pieces (2026-09-21)

> **Question (user):** for the scaled JA table, is covering the multi-glyph
> strings that are a *single token* (こんにちは is one Qwen piece, one row) more
> important than scaling `kanji:N`? **Answer: yes, by about 12 × per row.**
> Adding all 1 452 remaining corpus kanji to today's inventory covers 13.5 →
> 17.6 % of dialogue lines; adding 1 000 multi-glyph pieces covers 63.9 %.

`src/probe/piece_coverage.py --phrase_file <manga109s>/derived/dialogue_2_10.tsv`
(CPU, ≈ 1 min; raw pack `7b9fce0bb57b`). 42 557 lines, `--phrase_norm 1`. A
line is *covered* when every Qwen piece of it has a warm row; a *sentence* is a
covered line of ≥ 4 pieces. "Today" = every single kana with a row +
punctuation + `kanji:200` (382 rows; `plan.md` § V counted 5 211 covered lines
on `step1_0920`'s exact inventory, this count is 5 745 — same picture). Ranked
cold pieces: [`piece_coverage_ranked_2026_09_21.tsv`](piece_coverage_ranked_2026_09_21.tsv)
(4 132 pieces; extends `vocab_step_candidates_2026_09_20.tsv`'s 300).

## Where the text mass is

Piece tokens by class: single kana 42.6 %, **multi-glyph pieces 35.0 %**, single
kanji 15.7 %, other (punctuation, Latin, digits) 6.6 %, no pack row 0.0 %.
2 681 distinct multi-glyph pieces, 1 652 distinct single kanji. Multi-glyph
mass by length: 2 glyphs 65.5 %, 3 glyphs 26.8 %, 4 glyphs 6.4 %, 5 + 1.3 % —
**98.7 % of it is ≤ 4 glyphs**. Top-100 multi pieces hold 47.7 % of the
multi-glyph mass, top-500 79.7 %, top-1000 92.2 %. Pack-wide there are 18 252
JA-script multi-glyph single-piece rows; the corpus uses 2 681.

## Line coverage

| inventory | rows | covered lines | sentences |
|---|---|---|---|
| today | 382 | 5 745 (13.5 %) | 3 709 |
| kanji only → `kanji:600` | 782 | 6 808 (16.0 %) | 4 417 |
| kanji only → all 1 652 corpus kanji | 1 834 | 7 503 (17.6 %) | 4 916 |
| multi only + 100 | 481 | 12 166 (28.6 %) | 8 699 |
| multi only + 200 | 581 | 15 502 (36.4 %) | 11 378 |
| multi only + 500 | 881 | 21 853 (51.3 %) | 16 665 |
| multi only + 1 000 | 1 381 | 27 207 (63.9 %) | 21 294 |
| **joint + 400** (8 kanji, 392 multi) | 782 | 20 056 (47.1 %) | 15 165 |
| joint + 600 (84 kanji, 516 multi) | 982 | 23 525 (55.3 %) | 18 096 |
| **joint + 800** (176 kanji, 624 multi) | 1 182 | 26 357 (61.9 %) | 20 592 |
| joint + 1 000 (260 kanji, 740 multi) | 1 382 | 28 931 (68.0 %) | 22 848 |
| joint + 1 500 (453 kanji, 1 047 multi) | 1 882 | 33 771 (79.4 %) | 27 146 |
| **joint + 2 000** (659 kanji, 1 341 multi) | 2 382 | 36 941 (86.8 %) | 30 013 |
| joint + 3 000 (1 045 kanji, 1 955 multi) | 3 382 | 40 491 (95.1 %) | 33 230 |

*Joint* = cold pieces of either class by corpus frequency. At equal rows the
joint ranking beats both pure orders (782 rows: joint 47.1 % vs `kanji:600`
16.0 %; 1 382 rows: joint 68.0 % vs multi-only 63.9 %), and it does not reach
for a kanji until rank ≈ 300. Glyph lengths inside joint + 1 000's 740 multi
pieces: 451 two-glyph, 207 three, 70 four, 12 five or longer.

## What frequency misses

Manga dialogue is not where greetings live: **こんにちは is rank 1 426** (13
lines), ありがとう 250, お願い 214, ください 106. The target stage's own strings
(`assets/target_prompts.txt`) and whatever a user types first have to be pinned
into the inventory by hand, not left to the ranking. はい is は + い (two single
rows, a sequence — step 2's job, not a row); おはよう is お + は + よう, すみません
す + み + ません, ごめん three singles, 大丈夫 大 + 丈夫.

## Verdicts

- A scaled JA table is sized in **joint-frequency cold pieces**, not in
  `kanji:N`. The sentence pool is vocabulary-limited (`plan.md` § V) and the
  vocabulary is multi-glyph pieces first, kanji second.
- Whether a multi-glyph piece trains as one unit under the step-1 / grid recipe
  is **still unmeasured** — V's 12-row micro arm never ran; the only evidence is
  W2d Run 3 (します してる きた from one row each, 9/32 at 40 renders per row, the
  encoder arm) and the punctuation rows (！！ ・・・). It is the gate in front of
  any such table.
- A pinned list of deploy strings rides on top of the ranking.
