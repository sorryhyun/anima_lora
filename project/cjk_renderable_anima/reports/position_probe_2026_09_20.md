# position probe — the base model binds quoted text to position clauses (2026-09-20)

> Base model, no delta, EN controls, 512², 28 steps, cfg 4, block-compiled;
> `src/probe/position_probe.py`, output `output/wake_probe/position_probe/`.
> **Position clauses route quoted text to cells, and the clause wins over the
> sequence.** Clean at k = 4 for letters and words, clean at k = 9 for words in
> reading order, weaker at k = 9 when the clauses are shuffled or the units are
> single letters. The native list grammar (`reads as "A", "B", …`) carries count
> but not position. n = 4 renders per cell of the table: direction, not size.

Gate for the grid step-1 idea (k single-unit glyphs drawn mechanically in a
2×2 / 3×3 grid, k rows paid per item): a grid only pays k draws if the frozen
model already sends clause i to cell i.

`lift` = `bound` − `perm` (unit read in its cell, minus the same under random
unit → cell assignments). `order` = bound scored against the clause sequence.

| k | kind | style | bubble lift | flat lift | `order` (bubble / flat) |
|---|---|---|---|---|---|
| 4 | letter | list | +0.17 | +0.17 | |
| 4 | letter | pos | +0.41 | **+0.67** | |
| 4 | letter | pos_shuf | +0.21 | **+0.57** | 0.44 / 0.25 |
| 4 | word | list | +0.26 | +0.42 | |
| 4 | word | pos | **+0.72** | **+0.71** | |
| 4 | word | pos_shuf | **+0.71** | **+0.65** | 0.12 / 0.12 |
| 9 | letter | list | +0.25 | +0.14 | |
| 9 | letter | pos | +0.25 | +0.23 | |
| 9 | letter | pos_shuf | +0.02 | +0.29 | 0.28 / 0.19 |
| 9 | word | list | +0.22 | +0.06 | |
| 9 | word | pos | **+0.65** | **+0.71** | |
| 9 | word | pos_shuf | +0.23 | +0.54 | 0.06 / 0.06 |

Reads, with the sheets:

- **Clause beats sequence.** On `pos_shuf`, k = 4 words: bound 0.88–0.94 with
  `order` 0.12 — the unit goes where its clause says, not where it sits in the
  caption. Shuffling clause order is safe at k = 4.
- **k = 9 in reading order is a real 3 × 3** (`sheet_bubble_k9_word.png` `pos`
  #0 s0: HELP SORRY STOP / WHAT OK GO / YEY YES RUN, found 0.86–0.92). Shuffled
  at k = 9 it degrades (bubble +0.23, found 0.64): units go missing or repeat
  (GO GO, RUN RUN) before they go to the wrong cell.
- **`In the center` binds**: both k = 9 flat letter assignments draw the centre
  unit (S, G) large in the middle, every seed, shuffled or not.
- **Letters at k = 9 are the weak cell, and partly the ruler's.** The base
  model draws them as ≈ 15 px glyphs hugging the canvas border (a ring, not a
  grid), where found is 0.7–0.9 and the detector returns 3–7 boxes for 9 units.
  This is the base model's own layout habit and says nothing about a
  mechanically composited training image.
- **`list` is count, not place**: k bubbles or a line of k letters along the
  top edge, in caption order. It is the native grammar and it does not address
  cells.

Verdict: the gate passes for **position clauses in reading order, k = 4 and
9**; clause-order shuffling is supported at k = 4 and is a risk at k = 9. The
flat frame binds at least as well as the bubble frame.
