# OvisOCR2 — an outside page parser on the crop gates (2026-10-12)

OvisOCR2 run as a crop reader on K3 and sincos, and as a whole-page parser on the 146 labelled sincos pages.

## Facts

### Setup

[`ATH-MaaS/OvisOCR2`](https://huggingface.co/ATH-MaaS/OvisOCR2) (commit
`1fc9221`; 0.9 B, Qwen3.5-0.8B base, Apache-2.0) parses whole document pages
to Markdown. It is first on OmniDocBench v1.6 (96.58). The card does not
mention Japanese, Korean, Chinese, vertical text or manga. Reader:
`eval_manga109.OvisReader` (`--reader ovis`), weights in `models/ovis_ocr2`.

The card's recipe is used verbatim: its single page-parse prompt, thinking off,
greedy, `min_pixels` 448² / `max_pixels` 2880², with `<img src="images/bbox_…">`
blocks dropped. Two harness fixes were needed:

- EOS. The repo ships no `generation_config.json`, and `config.json`'s eos
  is `<|endoftext|>`. The chat turn ends at `<|im_end|>`, which is the
  tokenizer's eos and what vLLM stops on. Without it, `generate` runs past the
  answer and repeats `<think></think>` blocks until `max_new_tokens`. The first
  K3 / sincos run was invalid for this reason and was re-run.
- Markdown. Short crops come back as headings (`## 잠깐만`), so the reader
  strips `#` markers from each line. Line breaks cost nothing because
  `exact_key` is whitespace-blind.

`fla` / `causal-conv1d` are not installed, so the GDN layers run on the
transformers torch path: 10.5 crops/s on sincos.

### Crop mode

| gate | stock VL-1.6 | v4 (P3 ep2) | OvisOCR2 | OvisOCR2 vs stock |
|---|---|---|---|---|
| K3 KO exact (n = 48) | 21 | 26 | 12 | 2 / 11, z −2.5 |
| K3 ZH exact (n = 48) | 31 | 30 | 32 | 9 / 8, z +0.2 |
| sincos SFX strict / ♡-blind (n = 597) | — | 346 / 387 | 2 / 6 | — |
| sincos speech strict (n = 272) | — | 123 | 35 | — |

- SFX: on 597 SFX crops it gets 2 right (sim 0.098) and runs away on 40. Kana
  bursts come back as digits or as unrelated glyphs: `おっ` → `1`, `ぢゅぽ` →
  `170`, `へっ？` → `77`.
- ZH: 32 vs stock 31. KO: Korean dialogue reads right when the lettering is
  clean (`최근 제게 보내는 관심이 부족하지 않습니까?`); stylised Hangul comes back
  as Han (`조로` → `圣呈`). Several KO crops come back as digit runs.
- Sincos speech: 35 / 272. The speech labels are PP-OCRv6 text (see the
  `eval_sfx.py` docstring).

Verdict: OvisOCR2 is not a candidate for the crop reader.

### Page mode

`ocr/eval_page.py` feeds the 146 labelled sincos pages whole, which is what the
model was built for. A label row counts as hit when its `exact_key` is a
substring of the page output's `exact_key` (position-blind recall). It is split
at key length 3, because 1–2 char keys can hit by accident. It also counts rows
that are hit only once the `<img>` blocks are kept, i.e. SFX the parser filed as
a picture.

`--bs 16`, 146 pages in 7.6 min (3.1 s / page; `--bs 2` was 8–14 s / page).
Report: [`ocr_eval_page_ovis.md`](ocr_eval_page_ovis.md); raw output per page
in `output/ocr/eval/page_ovis.jsonl`.

| kind | rows | hit | hit, key ≥ 3 | hit only with `<img>` kept |
|---|---|---|---|---|
| sfx | 597 | 3 (0.5 %) | 3 / 558 | 0 |
| speech | 272 | 49 (18.0 %) | 47 / 266 | 0 |

- SFX recall is 3 / 597. Only 19 `<img>` blocks appear over 146 pages, and
  keeping them recovers no row. The page reads in the sample are the balloon
  text.
- Speech recall is 18 %. 77 of the 146 pages end in one block repeated ≥ 8
  times (`うろろろ` × 40 on `10506513`). The card's `_clean_truncated_repeats`
  is not applied: it only acts on output ≥ 8 000 chars and would not change any
  hit.

## Interpretation

- Setup: the torch path is fast enough for crops.
- Crop mode: it does not read hand-lettered SFX. It reads printed balloon text.
  The sincos speech labels being PP-OCRv6 text means 35 / 272 measures
  agreement, not accuracy; even so, it is far below every manga-trained reader.
- Page mode: page-mode hit is an upper bound, and not comparable to crop exact.
  Reading the whole page does not help. The SFX are not filed as pictures; the
  parser just does not transcribe the lettering on the art. Speech page reads
  run away, and as with crop mode the speech number is agreement, not accuracy.
- Page mode confirms the crop verdict: OvisOCR2 is a document parser, and manga
  lettering — the SFX especially — is outside what it reads.
