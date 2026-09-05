# cjk_aware_anima_dit — CJK semantics on the DiT side

Successor line to [`../cjk_aware_anima/`](../cjk_aware_anima/) (encoder side,
frozen 2026-09-05). Ext rows are content-free addresses; the DiT / LoRA
learns what CJK means from image data.

- [`plan.md`](plan.md) — goals (G-A unmask health at corpus scale, G-B
  semantics on isotropic rows), principles, phases D0–D6 with gates and kills.
- `findings.md` — starts with D0's verdict (ISO1 vs C9 direct blind set);
  then the OCR-reader verdict (PaddleOCR-VL-1.6 vs PP-OCRv6) that fixes D2/D3.
- `reports/`, `probes/` — this line's own; older material stays in the old tree.
- D2's second source is **the paired-edition manga corpus** (same page
  lettered in JA / EN / KO). It lives in an external private checkout that is
  never named or pathed here (plan principle 9); this tree carries only its
  numbers and consumes an accepted-pairs manifest.
- Code, probes, blind-pair tooling and the shipped packs stay in the old
  line's tree until this one needs its own; new scripts land here.
