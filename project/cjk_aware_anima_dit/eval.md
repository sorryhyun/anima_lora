# OCR reader scoreboard

Every reader in the O2 / O2b / O4 line, on **one** scoring basis. This file exists
because the per-run reports under `reports/` and the tables in `findings.md` were
each written on whatever basis was live that day, so their headline numbers cannot
be read down a column — two of them are off by up to 9 points for reasons that have
nothing to do with the reader. Read this file, not those, when comparing two runs.

Regenerate with `python project/cjk_aware_anima_dit/ocr/eval_table.py --write`
(CPU, no model — every cell is re-derived from the stored predictions).

## How rows are scored

**♡-blind exact is the headline.** Hearts are stripped from both prediction and
label before comparing. Two reasons:

1. **80.6 %** of sincos SFX labels carry a `♡`, against **0.3 %** of COO labels. On
   the sincos gate, strict exact is mostly a heart-agreement test.
2. The gate's `text_hand` came from PP-OCRv6, which reads no hearts — so a reader
   that gets `♡` *right* scores as a miss. And the captions these reads feed strip
   symbols anyway, so heart agreement is not a property we are buying.

Strict exact is kept in its own column because `findings.md` quotes it, and because
the two columns rank the readers differently — see below.

`sim` is untouched by either fold (it already strips punctuation and symbols).

## Comparability — the two re-bases

Two changes broke cross-day comparison. Both are *already applied* to every cell
here; the note matters when reading any **other** file.

1. **Label basis, 2026-09-06.** The sincos SFX label set was re-based from PP-OCRv6
   boxes onto `deepghs/AnimeText_yolo` boxes. Old basis scored a 71-row subset, new
   basis scores 617. There is no conversion between them: the row *set* differs, so
   `n/71` is not `n × 617/71`. Any gate threshold written in `/71` units is dead.
2. **Key basis, 2026-09-08 (`acd41d72`).** `exact_key` gained an ellipsis fold, so
   `・・・` / `...` / `…` stopped counting as three different reads. Anything scored
   before that date reads low: B′'s gate number moves **304 → 312**, its COO speech
   **2120 → 2259**. `reports/*.md` and `findings.md` still print the old-key values.

The practical trap: `vl16_tower_ssl` was evaluated after the fold and B′ before it,
so their stored headlines (307 vs 304) look like a win that is really a loss.

Per-run reports also carry ±1-line batching jitter on the VL readers. **A move of
1–2 lines is noise.**

Speech and chrome rows on the *sincos* set are not an accuracy metric at all — their
`text_hand` is PP-OCRv6 record text. Only the SFX column is a gate. COO speech is a
real metric.

## The table

<!-- TABLE -->
| reader | sincos SFX ♡-blind | strict | COO SFX ♡-blind | COO speech ♡-blind | COO spaced | in-domain val | note |
|---|---|---|---|---|---|---|---|
| `vl16_stock` | **52** / 617 (8.4 %) | 19 | — | — | — | — | stock VL-1.6, no fine-tune |
| `manga_ocr_stock` | **15** / 617 (2.4 %) | 6 | — | — | — | — | stock manga-ocr |
| `mocr_lr5e-5` | **153** / 617 (24.8 %) | 127 | — | — | — | 74.9 % | manga-ocr fine-tuned |
| `vl16_lr1e-4` | **157** / 617 (25.4 %) | 110 | 1704 / 2558 (66.6 %) | 2207 / 2559 (86.2 %) | 0 / 198 | 66.2 % | arm B — LoRA, tower frozen |
| `vl16_tower_lr1e-5` | **375** / 617 (60.8 %) | 312 | 2127 / 2558 (83.2 %) | 2259 / 2559 (88.3 %) | 1 / 198 | 86.2 % | **arm B′** — LoRA + tower unfrozen; the published reader |
| `vl16_tower_col100` | **377** / 617 (61.1 %) | 323 | 2171 / 2558 (84.9 %) | 2246 / 2559 (87.8 %) | 0 / 198 | 85.6 % | B′ + 1.6 % colorized append |
| `vl16_tower_col1500sw` | **341** / 617 (55.3 %) | 281 | 2188 / 2558 (85.5 %) | 2263 / 2559 (88.4 %) | 1 / 198 | 86.1 % | B′ + 22.3 % colorized swap |
| `vl16_pl_20k` | **402** / 617 (65.2 %) | 334 | 2189 / 2558 (85.6 %) | 2260 / 2559 (88.3 %) | 0 / 198 | 86.1 % | B′ + 20.6 % pseudo-label append (P1, cross-reader agreement) |
| `vl16_pl_kozh` | **391** / 617 (63.4 %) | 331 | 2167 / 2558 (84.7 %) | 2273 / 2559 (88.8 %) | 14 / 198 | 86.9 % | B′ + 27.5 % pseudo append (P1 JA 20k + K2 KO 5 930 / ZH 3 300) |
| `vl16_b2_norm2` | **350** / 617 (56.7 %) | 297 | 2138 / 2558 (83.6 %) | 2255 / 2559 (88.1 %) | 19 / 198 | 87.0 % | B′ recipe verbatim under TARGET_NORM 2 (plan_vl_respace R2) |
| `vl16_tower_ep3` | **360** / 617 (58.3 %) | 306 | 2234 / 2558 (87.3 %) | 2282 / 2559 (89.2 %) | 1 / 198 | 88.3 % | B′ × 3 epochs |
| `vl16_lpft` | **345** / 617 (55.9 %) | 264 | 2157 / 2558 (84.3 %) | 2242 / 2559 (87.6 %) | 0 / 198 | 86.8 % | LP-FT — arm B then B′ |
| `vl16_tower_ssl` | **380** / 617 (61.6 %) | 307 | 2165 / 2558 (84.6 %) | 2256 / 2559 (88.2 %) | 0 / 198 | 87.2 % | B′ from an SSL tower (draw20k, 4.4k steps) |
| `vl16_tower_ssl_all` | **378** / 617 (61.3 %) | 301 | 2194 / 2558 (85.8 %) | 2259 / 2559 (88.3 %) | 0 / 198 | 87.8 % | B′ from an SSL tower (manifest_all, 12k steps) |
| `vl16_tower_ssl_all_lr5e5` | **378** / 617 (61.3 %) | 288 | 2188 / 2558 (85.5 %) | 2262 / 2559 (88.4 %) | 1 / 198 | 87.8 % | same tower, LoRA lr 5e-5 |
| `hayai_v2_1_5` | **351** / 617 (56.9 %) | 316 | — | — | — | — | hayai v2.1.5 sidecar (~1/6 the parameters) |
| `sfx_pkg` | **374** / 617 (60.6 %) | 311 | — | — | — | — | shipped `anime_tools.ocr.sfx` (B′ + decode guard) |
<!-- /TABLE -->

## What it says

- **♡-blind reorders the top.** Strict has `col100` (323) > `hayai_v2_1_5` (316) >
  B′ (312) > `vl16_tower_ssl` (307). ♡-blind has `vl16_tower_ssl` (380) >
  `ssl_all` (378) > `col100` (377) > B′ (375) > `sfx_pkg` (374), with hayai falling
  to 351. hayai was winning on hearts; the SSL arms were losing on them.
- **The top is a six-line band across five readers** (374–380) and VL readers carry
  ±1–2 lines of batching jitter. Everything in that band is tied. No gate decision
  should rest on a move inside it.
- **SSL corpus scale bought nothing on the gate** (2026-09-09). Growing the tower's
  SSL corpus from draw20k / 4.4k steps to `manifest_all` / 12k steps — 100× the
  crops — moved the gate 380 → 378. Both LoRA learning rates (1e-4, 5e-5) landed on
  exactly 378, so the SFT lr is not a lever here either. Against B′ that is +3,
  short of the +15 the line needed. **The SSL tower line closes at "tied with B′".**
- **In-domain val does not predict the gate.** `vl16_tower_ep3` gains 4.1 points of
  val over B′ and *loses* 15 lines on the gate. Both `ssl_all` runs post the highest
  val of any B′ variant (87.8 %) and sit mid-band on the gate.
- **The gate and COO disagree, again.** `ssl_all` is the best SFX reader on COO of
  the whole B′ family bar `ep3` (2194 vs B′'s 2127, +67 on n = 2558 — outside noise)
  while being flat on sincos. Grey Manga109 SFX and colour doujin SFX are not the
  same task.
- **More SSL costs hearts.** Strict falls monotonically with SSL scale — B′ 312,
  `ssl` 307, `ssl_all` 301, `ssl_all_lr5e5` 288 — while ♡-blind stays flat. Tower
  adaptation is quietly trading away heart emission, which is invisible on the
  metric we score and would matter to anyone reading strict.
- **Unfreezing the tower is still the one big lever** — arm B 157 → B′ 375.
- **Colorized data**: a 1.6 % append is neutral-to-positive, a 22.3 % swap costs 34
  lines. Do not scale the swap.
