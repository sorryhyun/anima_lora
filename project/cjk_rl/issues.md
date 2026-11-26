# Issues — the cjk_rl line

Problems found that are not fixed and that no plan item covers yet. One entry each: what was
seen, where it was counted, what it touches. An entry leaves this file when a plan takes it
up or when it is fixed.

## A kana read as the kanji of the same shape scores as a miss (2026-11-26)

- `src/s1.py` compares glyphs by `key()`, which is NFKC. NFKC leaves 二 (U+4E8C) and ニ
  (U+30CB) apart, so 二 read where ニ was asked is not a hit. `soft()` (`g1.shape`) only
  treats a kana as a look-alike of the same kana with another voiced mark, and へ as one of ヘ.
  A kanji falls outside that kana range, so the read is not `near` either. It scores 0, not
  `near_score` (0.5 in `sents_1126_1856`).
- With `skip_hit = true` the position is not a hit, so it gets a window and a backward. The
  window loss adds the tuned reader's cross-entropy of ニ on that window, and that is high
  wherever the tuned reader reads 二. On a render that draws ニ correctly, this pushes the row
  toward the reader's habit, not toward the glyph.
- `split_1126_0910` (glyph stage, single kana) gives the size of it. Among the crops where
  hayai read the asked glyph and the tuned reader did not, the tuned reader read ニ → 二 on
  44, ろ → the digit 3 on 18, ミ → 三 on 13, and ユ and ュ → the Hangul 그 on 12 each
  ([g2b](archive/reports/g2b_2026_11_26.md), second part). hayai goes
  the other way and reads a 二 as ニ. Nobody has counted how often this happens in the
  sentence stage, where a string's context can steer the reader.
- Other pairs drawn the same way: ロ / 口, エ / 工, カ / 力, タ / 夕, ト / 卜, ハ / 八, ー / 一.
  Hangul is in it too: ユ / ュ against 그 (a kana pack row read as a KO syllable; `key()` and
  `soft()` leave Hangul as it is). A digit as well: ろ against 3. The user (11-26): folding
  ハ / 八 looks fine, and ユ / 그, ろ / 3 and ヲ / ㅋ go in. No read of ㅋ is in
  `split_1126_0910` (either reader); ヲ / ㅋ is in the list on its shape. ヲ's misses there
  are kana: the tuned reader reads ヲ → ヨ on 107 crops. ー / 一 is less clear: ー has its own row,
  and `FOLD` already maps the long dashes (―, —, ─) to ー.
- Possible fixes: count the pairs as `near` in `soft()`; fold them into one glyph in
  `key()` so they count as hits; or, on these pairs, keep hayai's term in the window loss and
  drop the tuned reader's.
- Touches: a row's score for the guards (`taper`, `[back]`), the hit / exact counts in the
  read, and the loss on the window for the rows of these kana. When the results of
  `sents_1126_1856` are read, the ニ / ろ / ミ / ユ / ュ / ヲ / ロ / エ / カ / タ / ト / ハ rows carry
  this effect.
