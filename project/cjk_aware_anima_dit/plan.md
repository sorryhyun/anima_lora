# CJK-aware Anima, DiT side — plan (archived 2026-09-08)

The DiT-side line is **frozen**. Every plan file this directory carried —
`plan.md`, `plan_base1.md`, `plan_ocr.md`, `plan_det.md`, `plan_ssl_tower.md` —
moved verbatim to `_archive/cjk_aware_anima_dit/plans/` (gitignored tree,
preserved in the private mirror; the pre-move versions are in git history).
The dated `reports/` moved with them to
`_archive/cjk_aware_anima_dit/reports/`, so a `reports/…` link from
[`findings.md`](findings.md) resolves against that archive, not against this
directory. Each plan carries a status banner written at archive time, so the
archived copy says what ran and what did not.

What stays live here: [`findings.md`](findings.md) (every settled verdict, and
the only file to read before reopening anything), `ocr/`, `probes/`, `assets/`.
The shipped surfaces are untouched — the SFX reader
(`sorryhyun/paddleocr-vl-1.6-manga-lora`, wired as `anime_tools.ocr.sfx`), the
AnimeText detector defaults, and the D1 quote-partitioned pack.

## Why frozen

The line was two goals and a side line; the side line is what delivered.

**Shipped (the OCR stack).** A reader that reads hand-lettered onomatopoeia —
arm B′, a PaddleOCR-VL-1.6 LoRA with the **vision tower unfrozen**, which took
the sincos SFX gate from stock 6 to ~50 % and COO test to 81.7 % against the
published 81.2 %. Plus `deepghs/AnimeText_yolo` as the detector (floor 38 → 4
on sincos' masked pages), the O4c–O4e caption rules (SFX/speech dedupe, glyph
floor, the `…`/`♡` normalisation that stopped the guard eating the longest line
on the page), and PP-OCRv6 retired to the explicit torch-free pair. D1 shipped
too: the deterministic isotropic table, the pack's quote-route partition, and
`ss_ext_pack_sha` on every LoRA.

**Not answered (the DiT goals).** D2–D6 never ran. **G-A** — manga trains
healthily unmasked at *corpus scale* — was measured only on `sincos` (351 of
the 873 text-masked images), where the shipped caption clauses cost nothing and
win nothing: three blind reads, three ties, pooled 35–36 on 71 decisive pairs.
**G-B** — a LoRA learns CJK semantics for isotropic addresses — is
**unmeasured**; D5a, the decisive experiment of the hypothesis, was never
executed. The paired-edition corpus never entered: alignment stopped at 2 of
240 works, so arm T and the contrastive address are designs, not results.

**Where the remaining headroom is, measured.** On the reader: the label side,
not the representation side. Five arms in a row moved in-domain COO and left
the doujin gate flat — colorized-COO at 1.6 %, the col1500 swap, ×3 epochs,
LP-FT, and label-free tower SSL — while B′'s misses are 55/313 ♡-only against
♡ in 497 of 617 gate rows. On the DiT: nothing was refuted, only unrun; a
reopening starts at D2's census, not at a new instrument.

Read [`findings.md`](findings.md) § Label basis first — the sincos gate changed
units twice (PP-box → AnimeText, then 619 → 617 rows), and **no `/ 71` figure
converts to a `/ 617` one.**

## The encoder-side predecessor

[`../cjk_aware_anima/`](../cjk_aware_anima/findings.md), frozen 2026-09-05 —
same shape: `findings.md` live, plans and reports archived.
