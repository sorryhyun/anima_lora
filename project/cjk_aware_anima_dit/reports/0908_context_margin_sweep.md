# Context sweep — margin / marker / page text (2026-09-08)

`ocr/context_margin_sweep.py`, readers vl16,vl16_bprime,hunyuan, pads 0.12,0.35,0.7,1.5. sincos SFX n=617 (the gate); COO test subset 400/kind. `pad` = margin per side as a fraction of the box's long edge (0.12 = today's crop); `_box` = the 12 % box drawn in red (+ a read-inside-the-box instruction on Hunyuan); `pagetext` = the page's Manga109 `<text>` lines quoted in the prompt (oracle). `contains` = label ⊂ prediction on `exact_key` — the fair metric for a reader that reads the whole frame; `no-kana` = predictions with no kana at all.

| reader | arm | sincos exact | contains | sim | runaway | no-kana | COO sfx exact | contains | COO speech exact | contains | COO runaway | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `vl16` | `pad0.12` | 21 | 23 | 0.366 | 10 | 6.6 % | 31.2 % | 32.5 % | 94.8 % | 95.5 % | 52 | 65 s |
| `vl16` | `pad0.35` | 16 | 17 | 0.351 | 15 | 4.9 % | 19.8 % | 26.2 % | 82.5 % | 92.0 % | 54 | 73 s |
| `vl16` | `pad0.35_box` | 14 | 15 | 0.361 | 16 | 4.2 % | 26.0 % | 29.8 % | 80.8 % | 90.0 % | 54 | 72 s |
| `vl16` | `pad0.7` | 3 | 11 | 0.265 | 57 | 5.7 % | 11.8 % | 24.0 % | 35.2 % | 73.8 % | 76 | 177 s |
| `vl16` | `pad0.7_box` | 5 | 14 | 0.285 | 46 | 7.0 % | 13.8 % | 25.0 % | 35.5 % | 71.8 % | 68 | 165 s |
| `vl16` | `pad1.5` | 3 | 13 | 0.158 | 113 | 8.6 % | 5.0 % | 18.5 % | 13.8 % | 59.0 % | 98 | 282 s |
| `vl16` | `pad1.5_box` | 4 | 11 | 0.181 | 87 | 9.9 % | 5.2 % | 18.5 % | 14.0 % | 61.5 % | 78 | 253 s |
| `vl16_bprime` | `pad0.12` | 316 | 330 | 0.853 | 2 | 0.0 % | 84.2 % | 84.8 % | 97.2 % | 97.5 % | 32 | 47 s |
| `vl16_bprime` | `pad0.35` | 294 | 307 | 0.817 | 11 | 0.2 % | 80.0 % | 84.5 % | 96.5 % | 97.0 % | 33 | 56 s |
| `vl16_bprime` | `pad0.35_box` | 294 | 306 | 0.823 | 11 | 0.2 % | 82.5 % | 85.2 % | 96.8 % | 97.0 % | 32 | 57 s |
| `vl16_bprime` | `pad0.7` | 179 | 246 | 0.690 | 23 | 0.5 % | 59.5 % | 79.2 % | 76.5 % | 90.8 % | 36 | 106 s |
| `vl16_bprime` | `pad0.7_box` | 218 | 243 | 0.732 | 12 | 2.4 % | 70.0 % | 78.0 % | 78.0 % | 88.0 % | 38 | 105 s |
| `vl16_bprime` | `pad1.5` | 36 | 102 | 0.353 | 80 | 1.3 % | 30.2 % | 61.5 % | 15.5 % | 35.0 % | 95 | 205 s |
| `vl16_bprime` | `pad1.5_box` | 45 | 93 | 0.342 | 31 | 8.3 % | 35.2 % | 58.8 % | 11.8 % | 27.3 % | 70 | 177 s |
| `hunyuan` | `pad0.12` | 21 | 21 | 0.336 | 0 | 5.0 % | 15.2 % | 15.2 % | 60.5 % | 67.0 % | 9 | 92 s |
| `hunyuan` | `pad0.35` | 18 | 22 | 0.352 | 3 | 3.1 % | 14.0 % | 15.8 % | 57.0 % | 68.0 % | 5 | 96 s |
| `hunyuan` | `pad0.35_box` | 13 | 14 | 0.353 | 0 | 8.9 % | 17.5 % | 18.0 % | 69.5 % | 73.0 % | 2 | 93 s |
| `hunyuan` | `pad0.7` | 9 | 18 | 0.313 | 7 | 3.1 % | 13.0 % | 17.0 % | 45.2 % | 70.8 % | 6 | 181 s |
| `hunyuan` | `pad0.7_box` | 14 | 14 | 0.377 | 0 | 7.9 % | 20.2 % | 22.0 % | 71.0 % | 75.8 % | 0 | 146 s |
| `hunyuan` | `pad1.5` | 4 | 18 | 0.228 | 28 | 2.4 % | 7.0 % | 13.8 % | 28.2 % | 66.0 % | 15 | 265 s |
| `hunyuan` | `pad1.5_box` | 16 | 17 | 0.346 | 0 | 5.5 % | 21.0 % | 22.8 % | 67.2 % | 72.0 % | 0 | 179 s |
| `hunyuan` | `pad0.12_pagetext` | — | — | — | — | — % | 13.8 % | 14.0 % | 54.5 % | 75.0 % | 75 | 363 s |

## Prompts

- Hunyuan base — 画像中の日本語のテキストを抽出してください。
- `_box` — 赤い枠の中の日本語のテキストだけを、そのまま書き写してください。
- `pagetext` — 'これは日本の漫画の一部です。同じページのセリフ: {lines}\n赤い枠の中の日本語のテキストだけを、そのまま書き写してください。'
