# HunyuanOCR-1.5 vs PaddleOCR-VL-1.6 as the manga reader (2026-09-08)

**Question.** `tencent/HunyuanOCR` (HunyuanOCR-1.5, 1 B, Tencent Hunyuan
Community licence, native `HunYuanVLForConditionalGeneration` in transformers
≥ 5.13 — no remote code) shipped as a lightweight end-to-end OCR VLM. Is it a
better reader — or a better *fine-tune base* — than the 0.9 B PaddleOCR-VL-1.6
this line picked at O2b?

**Setup.** Both evals of the line, both readers stock, same crops, same
scorer: the in-domain **COO test split ∩ Manga109-s** (2,558 SFX + 2,559
speech crops, `ocr/eval_manga109.py`) and the out-of-domain **sincos doujin
gate** (617 hand-labelled SFX, `ocr/eval_sfx.py`). Hunyuan is wired as
`eval_manga109.READERS["hunyuan"]`, bf16 / sdpa / greedy, decoded with the
card's locked recipe (`repetition_penalty=1.08`) and its **official** crop
prompt — upstream fixes one prompt per task and refuses to expose a free-form
one ("users pick a task, not a prompt"), so the crop task is
`structured_parse` = `提取图中的文字。` verbatim. Jobs `20260908-1532{24,24}`,
`…-1539{22,22}`, `…-154112`, `…-154723`. Weights `models/hunyuan_ocr`.

> **Every row below is on the current `exact_key`.** The ellipsis fold (each
> dot run → one `…`) landed in `acd41d72`, 2026-09-08 00:12; every stored
> prediction file older than that was re-scored on the new key by
> `ocr/rescore_eval.py` before it entered these tables. The correction is
> large — stock VL-1.6's COO speech row is 82.8 %, not the 63.4 % its report
> file still prints. See § "Two keys" below.

## In-domain — COO test split ∩ Manga109-s

| reader | SFX exact | SFX sim | sim ≥ 0.8 | runaway | speech exact | speech sim |
|---|---|---|---|---|---|---|
| PP-OCRv6 rec (retired) | 7.7 % | 0.194 | 11.8 % | 0 | 16.1 % | 0.297 |
| **Hunyuan-1.5, official zh prompt** | **9.5 %** | 0.251 | 14.3 % | 18 | **37.7 %** | 0.814 |
| **Hunyuan-1.5, Japanese prompt** | **13.0 %** | 0.320 | 18.8 % | 35 | **43.3 %** | 0.857 |
| stock manga-ocr | 28.9 % | 0.478 | 34.6 % | 0 | 81.0 % | 0.975 |
| **stock VL-1.6** | **31.6 %** | **0.545** | 39.5 % | 91 | **82.8 %** | 0.976 |
| hayai v2.1 (outside, zero-shot) | 78.2 % | 0.917 | 86.5 % | 28 | 88.0 % | 0.987 |
| B′ `vl16_tower_lr1e-5` (ours) | 83.2 % | 0.927 | 87.6 % | 25 | 88.3 % | 0.986 |

## Out-of-domain — the sincos doujin gate (617 hand-labelled SFX)

| reader | SFX exact | ♡-blind | sim | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|
| stock manga-ocr | 6 (1.0 %) | 13 | 0.312 | 17.2 % | 0 |
| Hunyuan, en prompt | 5 (0.8 %) | — | 0.299 | 6.8 % | 1 |
| Hunyuan, official zh prompt | 12 (1.9 %) | 32 | 0.300 | 10.5 % | 0 |
| **stock VL-1.6** | **19 (3.1 %)** | **52** | **0.366** | 15.2 % | 10 |
| **Hunyuan, Japanese prompt** | **21 (3.4 %)** | 46 | 0.336 | 12.2 % | 0 |
| hayai v2.1.5 (outside) | 316 (51.2 %) | — | 0.792 | 68.1 % | 5 |
| B′ (ours) | 312 (50.6 %) | 375 | 0.852 | 75.2 % | 2 |

**Wall**, batch-matched (same 600 crops, bs 32, RTX 5070 Ti): Hunyuan **20.2
crops/s**, VL-1.6 **26.6 crops/s** — ~1.3× slower, not the 2.3× the two full
runs suggest (those ran at different batch sizes). Hunyuan's image processor
sets `min_pixels` 262144, so a 40×60 SFX crop is upscaled to ≥ 256 visual
tokens; its cost per crop is nearly flat in the crop's size where VL's
shortest-edge rule shrinks with it.

## Four readings

1. **Hunyuan-1.5 is not a Japanese manga reader.** In-domain it lands
   **below stock manga-ocr on every column** and at a third of VL-1.6's SFX
   exact; only on the doujin gate is it level with VL — and both sit on the
   ~3 % floor there, which is the floor this whole line exists to lift, not a
   result. Against the reader of record (B′, 83.2 % / 50.6 %) it is not close.
2. **Half of its miss is the prompt naming no language.** Under the official
   Chinese instruction **51 % of its COO SFX reads contain no kana at all**
   (840 of 2,558 are pure Han: `ドドド` → `咚咚`, `ビッ` → `砰！！`) against 4 %
   for VL-1.6. A Japanese instruction (`画像中の日本語のテキストを抽出して
   ください。`) drops that to 18 % and buys +3.5 SFX / +5.6 speech points — the
   single largest lever found here, and one upstream's own client will not let
   a user pull. An English instruction is *worse* than the Chinese one
   (0.8 % on the gate). This is a language prior, not a legibility limit.
3. **What survives the prompt fix is small-kana blindness.** Folding
   `っゃゅょぁぃぅぇぉ` to their full-size forms on both sides rescues
   **+6.8 points of Hunyuan's COO speech** (1107 → 1282) against **+0.8 for
   VL-1.6** — it writes `言っちゃった` as `言っちやった`, `帰ろっかな` as
   `帰ろつかな`, `起床ーーッ` as `起床——ツ`. It also splits furigana into its
   own line and interleaves it with the base text (`全員起床ーーッ` →
   `ぜん いん き しょう / 全員 起床——ツ`), and inserts spaces between kana
   runs. Small kana, `ー` and `♡` are exactly what VL-1.6 was picked for at
   O2b; they are the fine detail of this domain, and Hunyuan drops them.
4. **No reason to switch bases, and a weak prior for a fine-tune arm.** The
   O2b lesson was that the *frozen tower* was the doujin gap, so a
   tower-unfrozen Hunyuan LoRA could close a lot of distance in principle
   (~90 GPU-min + peft plumbing on an untried architecture). But it would
   start 18 SFX points and 39 speech points behind where B′ started, on a
   decoder whose kana errors are systematic rather than noisy, and B′ +
   hayai v2.1.5 already bracket the gate at ~50 %. Not run; the arm is
   available if the line ever wants a third base, and 108 COO SFX lines that
   Hunyuan-ja reads and VL-1.6 misses (against 547 the other way) say the
   complementarity is real but one-sided.

## Two keys

Setting up these tables surfaced a comparability bug in the record, not in
either model. `exact_key` gained the ellipsis fold in `acd41d72`
(2026-09-08 00:12, the eval half of the O4e guard fix), so **every eval row
measured before that date scores `・・・`, `...` and `…` as three different
reads**, and every row after scores them as one. On COO speech, where a manga
line pauses and the label and the reader spell the pause differently, this is
worth up to **+495 lines**:

| row (COO speech exact) | as its report file prints it | on the current key |
|---|---|---|
| stock manga-ocr | 1588 (62.1 %) | 2072 (81.0 %) |
| stock VL-1.6 | 1623 (63.4 %) | 2118 (82.8 %) |
| hayai v2.1 | 2006 (78.4 %) | 2252 (88.0 %) |
| B′ `vl16_tower_lr1e-5` | 2120 (82.8 %) | 2259 (88.3 %) |

The SFX columns move less (+37 to +154) and the sincos gate less still (+4 to
+9), and **no ranking in the record flips** — which is why this is a footnote
and not a retraction. But a Hunyuan row measured today against a VL row
measured on 09-06 would have credited Hunyuan ~19 speech points it never
earned. `ocr/rescore_eval.py` re-derives `exact` for every stored
`output/ocr/eval/*.jsonl` from its own columns (CPU, no model re-run) and
prints the delta table; `--write` regenerates the reports. **It has not been
run with `--write`** — re-issuing twenty historical report files is the user's
call, and `findings.md`'s § O0 / § O2 / § O2b / § hayai tables would then
disagree with them until they are edited too.

*Reproduce:* `ocr/eval_manga109.py --reader hunyuan` and `ocr/eval_sfx.py
--reader hunyuan`; the prompt arms are `ANIMA_HUNYUAN_PROMPT=…` (off-recipe by
construction) and `ANIMA_HUNYUAN_TASK=<key>` for the other official tasks.
Reports `reports/ocr_eval{,_sfx}_hunyuan{,_ja_prompt,_en_prompt}.md`; both
stock rows are in `ocr/reeval_sfx_all.py`.
