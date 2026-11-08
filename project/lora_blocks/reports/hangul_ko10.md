# Hangul block v2 on ten held-out Korean pages, and stacked on ksplit e2

Date: 2026-11-08. Follow-up to [hangul_v2_s500.md](hangul_v2_s500.md). Sheets (`output/tests/lora_blocks/`):
`ko10/zoom_<page>_s0.png` (cut into `ko10/parts/`), `hangul_v2_stack/zoom_<page>_s<seed>.png`.

Result in one line: **the v2 block alone makes many more syllable errors than ksplit e2 on held-out pages
(short vowel strokes lost, lines duplicated or scrambled in long text); the two eval pages hid
this. The exact concat v2 + e2 fails on every page, as v1's did. e2's own weak spot is long
dense narration, where it draws pseudo-glyphs. Byte-split syllables are drawn right by both
when the training data held them even once; the one never seen comes out as a neighbour.**

## Setup

- **ko10** (`output/moss_i2i_eval/swap_text/inputs_ko10/`): ten Korean pages of the two eval
  works (589883, 737486), out of every training set, with their full staged captions. Five
  dense or small-text pages (589883_0002 / 0015 / 0032, 737486_0012 / 0026), five holding
  byte-split syllables (589883_0010 / 0013 / 0021, 737486_0011 / 0020). Seed 0, generate
  default (int8 DiT, fp32 stream). The captions carry OCR errors of their own (`낦습니다`,
  `에잊`, `누넘`); a render is read against its caption.
- **Arms**: ksplit e2, the v2 block alone (`hangul_v2_s500`), and their exact rank-64 concat
  (`stack_ksplit_e2_hangul_v2_s500`, `src/concat_lora.py`). The concat also on the two eval
  pages, seeds 0–2, bf16 + swap 10 as before. `src/hangul_eval.sh <arm> <lora> ko10`,
  `src/hangul_read.py OUT --ko10 <arms>`.
- `grad_in` is not read here: the bold faces raise it without making the strokes crisper
  (v2 2.0–2.4 on pages where it misspells), and e2 sits above 1 on most editions.

## Reads (per glyph, seed 0)

**Stack (v2 + e2).** Fails on all 12 pages: strokes doubled or smeared vertically, pink / grey
plates behind the text, white-outlined glyphs; dense pages (0021, 0015, 0002) illegible. On
the eval pages, seeds 0–2, the same doubling and tint (`좀…더`, `질 속이`, `벌거 주기?`).

**v2 alone**, wrong syllables against the caption:

- A short vowel stroke lost, ㅏ / ㅓ / ㅔ read as ㅣ: `저`→`지`, `정마알`→`정미알`, `실전`→`실진`,
  `정말로`→`정밀로`, `싸버릴`→`씨비릴`, `제길`→`지길`, `연장`→`연징`, `현재`→`현제`.
  Also `반응`→`반앙`, `공격`→`공걱`, `둘러싸`→`둘더씨`, `존재`→`존애`, `않았지만`→`넓았지만`.
- `스윽스윽`→`스윅스욱` (윽: byte-split, once in v2's data, 28 times in e2's).
- Long text: an extra `여탕도` line (0032), `몸모는 몸모는` (0015), `매끈매끈하가게` (0002); the
  four-line balloon of 737486_0026 split into two columns in scrambled order.
- Right where e2 is right on the short balloons of 589883_0021, 737486_0020; keeps
  uncaptioned Japanese.

**ksplit e2**, wrong syllables:

- `반응`→`반영`, an inserted `만` (`시작하 만 겠습니다`), `에잊`→`에입`, `존재`→`존에`,
  `용돈`→`용든`.
- 589883_0015 (long narration, ~26 px glyphs): pseudo-glyphs mid-line (`잘못ᄐ`, the
  `이 일이 마음에 들어서라는 이유로 계속` stretch garbled). The soft / broken strokes the
  plan looked for show here, not on the eval pages.

**Both**: `변태`→`변배`, `끜`→`끔` (`미끈미끜`).

**Byte-split syllables** (count in e2's data / v2's): `뜩` 156/18, `엣` 23/3, `뚫` 5/1,
`낦` 4/1, `뭉` 2/0, `짖` 1/0, `젯` 1/0 — right in both arms (`윽` wrong in v2 only, above).
`끜` 0/0 — both draw `끔`, the same neighbour. `륜` 1/0: not resolvable at render size.

## What it means

- The v2 block is not a swap_text model, and the two eval pages are too easy to tell the
  arms apart. ko10 is the eval for Korean from now on.
- Exact concat is out for both v1 and v2: composition needs a block that leaves the reference
  path alone, or a fine-tune.
- The byte-split tail costs only the syllables the data never held; one exposure was enough
  for `짖` / `젯`. Covering the tail is a data-coverage job, not a tokenizer one.
- e2's target for improvement is long, dense narration boxes, the region type v2's builder
  rejects (rectangles, not closed balloons).

## Addendum: e2 at 1.5x (inference only)

Is e2's breakdown on dense narration a token budget per glyph? e2 rendered 589883_0015 / 0002 /
0032 at 1824 × 1296 (bboxes × 1.5; `results/ko10_ksplit_e2_x15/`, sheets
`ko10/parts/x15_<page>.png`, rows edition / e2 / e2 1.5x). No: on 0015 and 0002 the layout
breaks (lines out of their box, reordered) and pseudo-glyphs stay or grow (`낮습니다` line on
0002); on 0032 the strokes are thinner and clean but `용든` stays and `연장`→`연쟝` appears.
Outside the training size the layout fails before the strokes improve, so resolution is not
a free inference lever.
