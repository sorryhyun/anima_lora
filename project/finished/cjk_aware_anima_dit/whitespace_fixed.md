# The target normalizer deleted whitespace (fixed 2026-09-09)

`crop_dataset.normalize_target` was `"".join(NFKC(s).split())` — it joined the
whitespace out of every training target. Now `" ".join(...)`: runs collapse to
one ASCII space, edges strip. `crop_dataset.TARGET_NORM` records which rule a
run trained under (1 = deleted, 2 = collapsed).

## Why it matters

Japanese does not space words, so v1 read as a harmless fold for the whole
O1/O2 line. Korean does — `알고 있었어` and `알고있었어` are not the same
string — so a KO pseudo-label arm on v1 would have its spacing deleted at the
last step, after the teacher took the trouble to produce it. It was never
purely a Korean bug: **3.80 %** of the 87,124 COO/Manga109 targets carry U+3000
or a newline (`あ　麻美さんそれ`), and v1 deleted those too.

## What is dirty

Every checkpoint under `output/ocr/` predates the fix; each now carries a
`WHITESPACE_DIRTY` file. They cannot emit a space, and their Korean reads are
unspaced by construction.

**No measured number moves.** `eval_manga109.exact_key` is whitespace-blind, so
every gate on this line — sincos SFX ♡-blind 402/617, COO SFX 2189, COO speech
2260 — scores identically under either rule. What is dirty is the output
surface, not the score: the fix does not re-baseline the line, and a
TARGET_NORM = 2 arm stays comparable to `vl16_pl_20k` on the Japanese gates.
The one claim that must not cross the boundary is a claim about spacing itself.

Pseudo-label parquets and `manifest_pseudo_*.parquet` hold raw teacher strings,
not targets — the normalizer runs at `load_split` time, so none need recutting.
