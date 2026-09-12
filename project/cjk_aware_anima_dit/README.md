# cjk_aware_anima_dit — CJK semantics on the DiT side (frozen 2026-09-08)

Successor line to [`../cjk_aware_anima/`](../cjk_aware_anima/) (encoder side,
frozen 2026-09-05). Premise: ext rows are content-free addresses; the DiT /
LoRA learns what CJK means from image data.

**The line is frozen.** Its OCR half shipped, its two DiT goals were never
tested at scale. [`plan.md`](plan.md) carries the freeze note (what ran, what
did not, where the headroom is), **§ Still open** — the open items of the three
lines that outlived the freeze and were archived 2026-09-12 (`plan2.md`,
`plan_vl_respace.md`, `plan_render.md`) — and **§ Do not re-propose**, their
anti-re-proposal lists lifted out of the gitignored tree.
[`findings.md`](findings.md) is every settled verdict with its evidence
pointer; read its § Label basis before comparing any two numbers in it, because
the sincos gate changed units twice.

- [`findings.md`](findings.md) — the ledger. D0/D1, the OCR-reader verdicts
  (O0–O6, the AnimeText detector flip, the shipped SFX reader), the closed
  directions (page context, VL prompting, HunyuanOCR, LP-FT, tower SSL), and
  the plain-vs-OCR caption ties.
- [`plan.md`](plan.md) — freeze note, § Still open, § Do not re-propose, and
  the pointer to the archived plans.
- `ocr/`, `probes/`, `render/`, `assets/` — this line's code, still runnable.
  The hand labels (`assets/sfx_labels_sincos.tsv`, 975 rows / 617 SFX scored)
  are the gate's ground truth and the one asset a reopening would start from.
- Archived (gitignored, private mirror): the eight plan files →
  `_archive/cjk_aware_anima_dit/plans/`, the dated reports →
  `_archive/cjk_aware_anima_dit/reports/`. Every `reports/…` link below or in
  `findings.md` resolves there.
- The paired-edition manga corpus lives in an external private checkout that
  is never named or pathed here (plan principle 9); it never entered the
  training work.
