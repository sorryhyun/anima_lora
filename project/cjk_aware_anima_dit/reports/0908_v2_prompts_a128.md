# The α128 pair re-rendered on the hard (v2) prompt set (2026-09-08)

**Status: RENDERED, grading open.** Blind set `s18_OCR128_vs_PLAIN128_v2`
pushed, **32 pairs**, unfilled.

## Why

`s17_OCR128_vs_PLAIN128` came back **OCR128 10 / PLAIN128 12 / 2 ties** — a
tie, and the direction even flipped from s16's 14–9 lean toward OCR. That
closes the α argument of `0908_alpha128.md`: raising α/r to 4 moved the two
arms measurably further apart in PE-Spatial (0.9862 → 0.9794, past
LoRA-vs-base) and the blind read *still* did not resolve. So the missing
dynamic range was never only in the adapter — it was also in the **prompts**.

`assets/unmask_eval_prompts.txt` is 8 generic rows (`1girl, solo, blonde
hair, school uniform, classroom…`), all deliberately text-free. Nothing in it
asks for the thing the two arms disagree about. `assets/unmask_eval_prompts_v2.txt`
(`probes/make_eval_prompts.py`, built 2026-09-05 for the s11–s14 sets, never
used on this pair) is 16 rows carrying training characters + series names,
the `@sincos` trigger on some rows and not others, a 2-character crossover,
rare mid-frequency training tags, seeded random-tag rows — and **two rows that
ask for text**: r10 `comic, 4koma, 1girl, japanese text, speech bubble,
monochrome, screentone…` and r11 `1girl, japanese text, holding sign, sign,
street…`.

## Design

Render-only; both checkpoints are the ones s17 graded, unchanged. New driver
`project/cjk_aware_anima/run_grid.py` (multi-arm, render-only, sampler flags
copied verbatim from `run_unmask_r2.py` so grids stay comparable across both).

| | |
|---|---|
| arms | `cjk_unmask_ocr_a128`, `cjk_unmask_plain_a128` (α=128, the s17 pair) |
| prompts | `assets/unmask_eval_prompts_v2.txt`, 16 rows |
| seeds | 42, 7 |
| grids | `output/tests/cjk_unmask_evalv2/arm{OCR128,PLAIN128}_s{42,7}` |
| budget | **64 images** → 32 blind pairs |

No base reference this time — the cos→base ruler is already on the table from
s17, and the 64-image budget buys more pairs instead (32 vs s16/s17's 24).

## Automated readouts — still flat, and the one apparent signal is noise

### Adherence (dbv4, `--rows_json` v2)

| arm | n | adherence prob | recall |
|---|---:|---:|---:|
| OCR128 | 32 | 0.7055 | 0.8486 |
| PLAIN128 | 32 | 0.7158 | 0.8443 |

Flat, as between every arm this line has run. Worth noting the v2 rows *are*
harder in the intended way — adherence falls 0.75 → 0.71 vs the v1 set, with
the character/rare-tag rows carrying it (r12 chiyoko 0.57/0.53, r14 0.54/0.56,
r15 0.52/0.59 vs r5 `@sincos` classroom 0.87/0.87).

### PE-Spatial, arm vs arm, matched (row, seed)

| contrast | n | mean cos | min |
|---|---:|---:|---:|
| same arm, different render seed (floor) | 32 | 0.9738 | 0.8324 |
| **OCR vs PLAIN @ α128, v2 prompts** | 32 | **0.9884** | 0.8945 |

**The v2 prompts shrink the arm contrast rather than opening it.** On the v1
set the same pair sat at 0.9794 against a 0.9347 seed floor — the arm gap was
*wider* than the seed spread. Here it is **narrower** (0.9884 vs 0.9738). The
mechanism is that v2 rows are heavily specified (12–15 tags, named
characters), so they pin the composition and leave both the seed *and* the
adapter less room; the seed floor rising 0.935 → 0.974 is the same effect.
Per row, the only two that move are r14 (0.945) and r10 (0.976) — the random
mid-frequency-tag row and the 4koma row.

### Text boxes (AnimeText detector), split by whether the prompt asked for text

r10 and r11 *request* `japanese text`, so a box there is adherence, not spam;
the other 14 rows are the spam tally.

| arm | text-free cells w/ box | lines | glyph% | r10/r11 cells w/ box | lines |
|---|---:|---:|---:|---:|---:|
| OCR128 | 11 / 28 | 33 | 0.72 | 4 / 4 | 47 |
| PLAIN128 | 13 / 28 | 34 | 0.92 | 4 / 4 | 35 |

Spam is equal (11 vs 13 cells of 28, 33 vs 34 lines) — the α128 finding holds
on harder prompts. The "OCR writes 47 lines vs PLAIN's 35 where text was
asked for" reading **does not survive the per-cell dump** and should not be
quoted:

| arm | seed | row | lines | glyph% |
|---|---|---|---:|---:|
| OCR128 | 42 | r10 | 3 | 2.37 |
| OCR128 | 42 | r11 | 10 | 4.98 |
| OCR128 | 7 | r10 | 5 | 9.16 |
| OCR128 | 7 | r11 | **29** | 13.32 |
| PLAIN128 | 42 | r10 | 8 | 6.84 |
| PLAIN128 | 42 | r11 | 7 | 3.22 |
| PLAIN128 | 7 | r10 | 4 | **25.61** |
| PLAIN128 | 7 | r11 | 16 | 14.99 |

One cell (OCR s7 r11, 29 lines) carries the whole OCR total, and by glyph
*area* the sign flips — PLAIN 12.7 % vs OCR 7.5 % on the same four cells,
driven by a different single cell (PLAIN s7 r10, 25.6 %). n = 4 per arm.
This is noise; **two text-requesting rows at two seeds cannot measure the
text-rendering axis.** If that axis is the question, it needs its own grid of
text-requesting prompts, not two rows borrowed from a general set.

## Next

Grade `s18_OCR128_vs_PLAIN128_v2` (32 pairs, `sets/s18_OCR128_vs_PLAIN128_v2/verdicts.tsv`).
The prior is now weak: two graded sets on this pair have tied, the automated
readouts are flat again, and the PE ruler says the harder prompts made the
arms *more* alike, not less. If s18 also ties, the line has three independent
reads saying the shipped OCR clauses are **neutral for image quality on this
shard** — which is still a shippable result (they cost nothing) but closes the
"captions are load-bearing" question in the negative for the render axis, and
the remaining case for them rests on the 0901 unmask A/B/C, where they were
measured against *spam*, not against quality.
