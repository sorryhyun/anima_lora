# CJK-aware Anima — done (glyph line)

What is finished on the [`plan2.md`](plan2.md) line. The vocab-pack line's
ledger is [`done.md`](done.md).

## Phase 0 — ceilings (resolved 2026-08-17)

Planned as three measurements; two were skipped by user call, one was run and
produced a fix:

- **0.1 VAE reconstruction ceiling — SKIPPED** (user call 2026-08-17):
  qwen-image demonstrates the VAE family carries legible glyphs; not worth a
  measurement pass. Revisit only if Phase 2 renders fail in a way that looks
  like reconstruction (blurred strokes at sizes the OCR corpus reads fine).
- **0.3 coverage + register census — SKIPPED** (user call 2026-08-17): the
  method is already established (`gates/coverage.py` analog over OCR'd char
  histograms); run it when Phase 1 data actually exists rather than as a
  pre-ceiling.
- **0.2 char-row separability — MEASURED, one defect found and FIXED** (below).

## Phase 0.2 — char-row separability (2026-08-17)

**Instrument**: [`gates/separability.py`](gates/separability.py) (CPU-only) —
full pairwise cosine over every single-char glyph row in the ext table:
39,632 rows = 11,615 qwen 1-char surfaces (where the common kanji/kana live)
+ 28,017 char-map byte-fragment rows, measured on the Qwen-init table and the
trained `cjk_vocab_pack_item2` pack. A char's row can sit in either map
(水 = qwen row, 氷 = char row), so the union — not the char map alone — is
the measured set. JSON: `assets/separability_phase02.json` (pre-fix, incl.
the pack comparison) and `assets/separability_phase02_fixed.json` (post-fix
init).

**Verdicts:**

- **The common-kanji class is cleanly separated, and the `param=global`
  correction does not collapse it — it *widens* it.** All 15 handpicked
  visually-confusable pairs (鎧/錯 0.30, 日/曰 0.38, 未/末 0.48 …) sit at
  cos 0.25–0.55 in the pack; 13/15 are *more* separated post-training.
  Pack-wide, kanji-kanji pairs >0.9 drop 93,786 → 51,588 and >0.99
  801 → 658 vs init. Init→pack row movement is large (median row cos 0.46,
  NN-identity churn 78%) but separation is preserved — the property that
  matters. The favorable prior plan2 stated for the direct arm holds as
  measured.
- **One real collision class existed — an init artifact, not training.**
  Char-map byte-fragment rows were init'd as the *mean* of UTF-8 byte-token
  embeddings; mean is order-invariant, so the **527 char pairs whose byte
  triples are permutations of each other had bit-identical vectors**
  (䛮/䮛 class). 508 pairs were Ext-A/B tail noise; 19 involved a URO kanji,
  at least one both-members-real: **鯰(catfish)/鰯(sardine)** (also 鶴/鴶,
  鶯/鯶, 鰈/鈰 — one real member each). No gradient or attention map can ever
  distinguish bit-identical rows, so this was worth fixing at the source.
- Context: NN-cos median 0.85 (kanji), 0.94 (hangul — a dense tail cluster,
  irrelevant under ja-only scope).

**Fix (landed 2026-08-17)**: `ext_vocab.build_ext_table` now detects chars
sharing a fragment-id *multiset* with another char and pools those with
position weights 1..n instead of the plain mean — a targeted tie-break; the
non-colliding rows keep the plain-mean formula unchanged. Assets rebuilt
(`bench/cjk_adapter/build_ext.py`; row count and id mapping unchanged).
Post-fix verification: exact-duplicate groups **527 → 0**, kanji-kanji pairs
>0.99 **801 → 273** (the survivors are genuinely-near neighbors, max 0.9982,
none identical); former ties land at cos ≈ 0.97 (鯰/鰯 0.977) — distinct
direction, still close, which is the honest ceiling of linear pooling over
shared byte embeddings. Real separation for these rows, if ever needed, comes
from training, not init.

**Carry-forward notes:**

- The trained packs (`cjk_vocab_pack_*`) are baked tables distilled from the
  *old* init — their 1,054 collided rows stay collided until the next retrain
  (already scheduled in plan.md 2c remaining work), which starts from the
  fixed asset automatically.
- The rebuild refit W on the same anchors/seed; holdout cos moved 0.7516 →
  0.7563 (environment drift since the original build, not the fix — the fix
  touches pooling only, after W).
