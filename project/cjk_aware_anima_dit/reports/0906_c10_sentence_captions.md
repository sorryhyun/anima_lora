# B3 verdict — arm C10 (sentence captions on hybrid records) vs C9ISOQ (2026-09-06)

C10 = C9ISOQ's recipe with the two plan_base1 changes (hybrid OCR records, B0;
speech-only sentence captions, B1/B2). Three training seeds (42 / 7 / 1234),
grids `output/tests/cjk_unmask_eval2/armC10{,s7,s1234}_s{42,7,1234}`. Control
C9ISOQ: same pack, same latents, tags format on v1 records.

## Blind set s14_C10_vs_C9ISOQ (48 pairs, seeds 9/10/11, 16 v2 rows)

Graded 2026-09-06, user pasted a 48-char a/b string with `-` = tie
(private repo commit a07494e; `reports/blind_s14_C10_vs_C9ISOQ.md` in
`project/cjk_aware_anima/`).

| | C10 | C9ISOQ | tie |
|---|---:|---:|---:|
| pairs | 21 | 15 | 12 |
| rows (net) | 8 | 6 | 2 |

Pair sign test p = 0.41, row sign test p = 0.79; seed-twin floor is s02
15–9 on 24. **Flat.** Gate half 2 ("blind ≥ C9ISOQ inside the floor") is
met — C10 is not below.

**Caveat — side bias.** The grader chose B in 31 of the 36 decisive pairs
(p ≈ 1e-5 against a fair coin); s11–s13 were balanced (34/38, 18/22,
22/21). Sides were shuffled 24/24 per arm, so the bias does not tilt the
arm total in expectation, but it means the 36 decisions carry less
content than usual. If the string was graded with the opposite
convention (`b` = worse) the totals flip to 15–21, still inside the floor
— the flat verdict does not depend on it.

## Spam tally, 8-row grids × 3 render seeds (gate half 1)

Instrument: lenient PP-OCRv6 over every grid png
(`probes/grid_spam_tally.py`, `reports/grid_spam_tally_c10.json`) to flag
cells, then an eyeball read of the 25 flagged cells with the C-series
ledger's convention (non-diegetic text: pseudo-Latin banner, pseudo-JA
band, bubbles filled with text outside the `comic` row).

| arm (train seed) | events | cells |
|---|---|---|
| C10 (42) | ~2 | s42 r6 pseudo-Latin banner; s7 r3 hug with two JA speech bubbles (+ s42 r7 small pink `nふ` mark, marginal) |
| C10s7 | ~2 | s42 r6 banner; s7 r3 JA speech bubbles (+ s7 r8 comic bubbles filled with JA, comic row) |
| C10s1234 | ~2 | s42 r1 chalkboard pseudo-JA band; s42 r6 banner |
| C9ISOQ | ~2 | s42 r1 pseudo-Latin band; s42 r6 banner (+ s7 r8 comic bubbles filled with JA) |

C10 ≤ C9ISOQ on all three seeds, at equality (~2 = the C8 / C9 ledger
value). The r6 maid-cafe banner at render seed 42 appears in every arm
(C8 → C10) and is a base-model habit, not a caption effect; the s1234
render seed is clean for every arm. Lenient OCR "images with any box"
(10 / 14 / 12 of 24 for the C10 seeds vs 11 for C9ISOQ) is dominated by
signatures, tiny ASCII and false boxes and is not the ledger — kept only
as the flagging pass.

## Other readouts

- Tagger adherence recall (`reports/unmask_grid_judge_c10.md`): C10 0.912
  / 0.896 / 0.886 vs C9ISOQ 0.844 (C9 0.912). C10 recovers the recall
  C9ISOQ had lost, mostly on r8 (`comic, 2koma`: C9ISOQ 0.50 vs C10 0.83).
  PE cos columns saturated as before.
- Sentence-shape spam-direction probe (`reports/sentence_spam_probe.md`,
  4 prompts × 3 seeds): plain prompts — C10 25 % of images carry text vs
  C9ISOQ 50 %, glyph area 0.98 % vs 5.32 %; with a text address (tags or
  sentence) both arms render text at 100 % as intended, C10/sentence
  7.1 % glyph area vs C9ISOQ/sentence 12.7 %. n = 12 per cell, direction
  only.

## Verdict

Gate: **PASS at the floor.** Spam not up on any seed (kill condition not
met), blind flat inside the seed-twin floor, adherence recall back to
C9's level. Per plan_base1 B3 the sentence caption shape on hybrid
records becomes D2's default caption shape. What this does NOT show: any
blind-visible *gain* from the shape — the hybrid records survive on B0's
floor number, the shape survives on "does no harm + recall". B4 (fold
into plan.md / findings) is next.
