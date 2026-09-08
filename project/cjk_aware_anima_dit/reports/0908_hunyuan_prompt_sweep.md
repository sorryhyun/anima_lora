# HunyuanOCR-1.5 — instruction sweep (2026-09-08)

`ocr/hunyuan_prompt_sweep.py`, one model load, 6 arms. sincos SFX n=617 (the gate); COO test subset 400/kind (in-domain control). `no-kana` = predictions containing no kana at all — the Chinese-prior failure.

| arm | sincos exact | sincos sim | no-kana | COO sfx | COO speech | COO no-kana (sfx) |
|---|---|---|---|---|---|---|
| `ja_plain` | 21 / 617 (3.4 %) | 0.336 | 5 % | 15.2 % | 60.5 % | 14 % |
| `ja_plain_nolang` | 14 / 617 (2.3 %) | 0.319 | 20 % | 11.5 % | 58.8 % | 33 % |
| `ja_manga` | 9 / 617 (1.5 %) | 0.339 | 15 % | 15.0 % | 62.0 % | 28 % |
| `ja_manga_lang` | 19 / 617 (3.1 %) | 0.325 | 6 % | 16.2 % | 63.2 % | 13 % |
| `ja_short` | 14 / 617 (2.3 %) | 0.343 | 20 % | 14.8 % | 57.8 % | 34 % |
| `ja_short_lang` | 21 / 617 (3.4 %) | 0.336 | 9 % | 17.5 % | 62.3 % | 17 % |

## Prompts

- `ja_plain` — 画像中の日本語のテキストを抽出してください。
- `ja_plain_nolang` — 画像中のテキストを抽出してください。
- `ja_manga` — 日本の漫画の一行です。描かれている文字をそのまま書き写してください。
- `ja_manga_lang` — 日本の漫画の一行です。画像中の日本語のテキストをそのまま書き写してください。
- `ja_short` — 画像の文字をそのまま書き写してください。
- `ja_short_lang` — 画像中の日本語のテキストをそのまま書き写してください。
