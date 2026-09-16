# datacheck — corpus bubble labels are mostly wrong (2026-09-14)

Trigger: building `diagram.html`, the corpus crop labelled **うがいっ** is a
hand-lettered **じゅぱっ**-type SFX. A by-eye pass says this is the rule for
the corpus slice, not an outlier.

## Where the labels come from

`wake_probe.py --stage data` adds corpus crops through `corpus_lines` (`src/data/inventory.py`)
(`post_image_dataset/render/ja/{resized,heldout}/boxes.jsonl`, built by
`project/cjk_aware_anima_dit/render/corpus_boxes.py`):

- text boxes: AnimeText detector → read by PaddleOCR-VL-1.6 (`stock` sweeper)
- `bubbles` guards drop `sfx` (1 599), `short_kana` (1 520), `no_balloon`
  (604), `script`, `short`, `repeat` … → 3 607 kept bubbles in `resized`,
  631 in `heldout`
- `_corpus_lines` then keeps **kana-only lines of ≤ N chars**

The guards judge the *read*, not the image. A garbled read of a hand-lettered
SFX no longer looks like SFX (`line_kind`) and is long enough to pass
`short_kana`, so it survives — and the kana-only ≤ N filter selects exactly
that tail. Typeset speech mostly carries kanji or runs longer and is filtered
out. (Mechanism inferred from the sheet, not traced per box.)

## Hand check

60 of the 279 distinct P0b corpus lines (`data_wdsek`, first 60 distinct),
label vs crop, by eye:

| verdict | n | examples (label → drawn) |
|---|---|---|
| match | ~12 | なんだ, ほんとムリ, もうやだっ, おおおっ, よぉし, おおきに |
| partial / near | ~8 | ぬちゃ → ぬちゅ, ねばあ → ねばぁ, ちさと (+ あっ missing) |
| **wrong** | **~40** | うがいっ → じゅぱっ, えいえい → ごそごそ, いやんっ → じゅぷっ, ちょっと → びくん, はちゃっ → ぬちゅっ, ふうへラ → ムラムラ |

Error types: handakuten / dakuten dropped (ぱ→は, ぷ→ふ/う, ぴゅ→えゆ),
stylised strokes misread as other kana (ぬ→は, ち↔う), and free garbage.
So roughly **two in three corpus labels are wrong**, and the wrong ones point
basic-kana rows (は ふ う ち) at handakuten / other glyphs — the rows P0b's
`kana_ext` is adding separately.

## Scope

| data | items | corpus crops | share |
|---|---|---|---|
| `wd` / `wds` (Run 3, P0a) | 8 992 | 132 (118 distinct) | 1.5 % |
| `wdsek` (P0b) | 15 880 | 300 (279 distinct) | 1.9 % |

Per character (P0b, items containing the char): corpus share is 8–24 %
(お 0.23, ふ 0.24, あ 0.18, ち 0.14, は 0.14) — small overall, not small for
the rows the SFX errors land on.

**Eval sets built from the same reads** (text-only T2I prompts, so the harm is a
nonsense target, not an image/label mismatch):

- `line` (16): そオニ ほらそれ さわって はっちゃっ さむん たはは ちゃわは くちゃん
  いたちょ なにそれ ちゃく すゅう ヌナちゃ めちゃ いいふっ いやいい — by eye only
  ~6 read as real speech (ほらそれ さわって なにそれ いやいい めちゃ たはは).
- `corpus` (10): ぬごっか はげしっ フフフ ぬ〜ちゃ ちゃく フルルっ なにそれ ふろふ
  ヘッケン あ…ひる — mostly garbled.

## What this does and doesn't touch

- **Not affected**: font renders (98 % of items) and the `single` /
  `single_ext` / `single_kanji` / `word` / `combo` / `en` eval groups — their
  targets are font strings or the inventory, not OCR reads.
- **Weakened**: `line` 0/32 (README "one unit" row) and `corpus` 0/20 — part of
  the denominator is unreadable-by-design targets. The one-unit reading still
  holds on the real lines (ほらそれ→ら, なにそれ→な, いやいい→いい), but the
  count should be re-stated on a clean set.
- **Training**: ~2/3 of 1.9 % mislabelled — mild label noise, concentrated on
  basic-kana rows. P0b keeps running; not a reason to stop it.

## Options (not done)

1. `--n_corpus 0` for the next data build — font-only; cheapest, loses the
   only real-manga layouts.
2. Re-read the corpus crops with the SFX reader and keep only lines where
   both readers agree (the `eval` stage already runs both).
3. Rebuild `line` / `corpus` eval from the agreed subset or hand-picked
   speech, then re-score Run 3 / P0a / P0b on it before P1 leans on `line`.

The by-eye sheet is a scratch artefact (adult corpus content, not kept in the
repo); regenerate from `data_wdsek/train.jsonl` rows with `src == "corpus"`.
