# cjk_aware_anima — native CJK prompting through extra T5-side vocab rows

Encoder-side CJK line. Frozen 2026-09-05, moved to `finished/` on 2026-09-24
(from its old home directly under `project/`; the pack-recipe builders left first, to
`scripts/distill_cjk/corpus/`). Premise: give Japanese / Korean / Chinese
spans their own rows in the LLM Adapter's T5-side query table (ids ≥ 32128)
and distill those rows against the adapter's own output on the EN
translation, so EN prompts stay bit-exact and the artefact is a vocab pack,
not a LoRA.

## Verdicts

[`findings.md`](findings.md) §1–§14 is the record, read-only. The short form:

- **Shipped.** `synthja_v4` (JA tag tier, 2026-09-01) and
  `synthjakozh1sym_r256` (JA+KO+ZH, 69,558 rows incl. the symbol block,
  2026-09-06) — the latter is the base of the public
  `sorryhyun/anima-vocab-pack-cjk` repo; the `_preview*` packs there are it
  plus a baked wake-line delta. JA tags behave like their EN spelling in
  same-seed grids; KO / ZH rows are trained but were never grid-validated.
- **Ceiling.** Rare kanji character names never compose at any corpus size or
  lever (§4, §12–§14); the coverage and geometry refinements are inert; twelve
  blind sets found no table property that beats the trained pack, and
  content-free tables (HOT / ISO1 / COLLAPSE) tie or beat it for the unmask
  goal. The one confirmed requirement is that rows must exist (C9 > P).
- **Continued elsewhere.** The DiT-side question (what an ext row *means*)
  went to [`../cjk_aware_anima_dit/`](../cjk_aware_anima_dit/README.md),
  then the wake line [`../../cjk_renderable_anima/`](../../cjk_renderable_anima/README.md)
  and the production line `../../cjk_anima_scale/`.

## What is where

| Piece | Where |
|---|---|
| Pack recipe (ext table → corpus → cache → distill) | `bench/cjk_adapter/build_ext.py`, `scripts/distill_cjk/{corpus,cache,distill}` — `docs/methods/cjk_vocab_pack.md` § Rebuilding a pack |
| Shipped surface (loader, hooks, cache stamps) | `library/anima/{vocab_pack,ext_vocab}.py`, `docs/methods/cjk_vocab_pack.md` |
| Code ledger, packs, ship contract | [`deliverables.md`](deliverables.md) |
| Freeze note and why | [`plan.md`](plan.md) |
| Exploration-side data tooling (D2 commentary, Love Hina set, manga text pilot, glossary audit, OCR record builders) | `datasets/` ([README](datasets/README.md)); `datasets/assets` is a symlink to `post_image_dataset/cjk_distill/assets/` |
| Gates, probes, eval prompt sets | `gates/`, `probes/`, `assets/` — run by path (`python project/finished/cjk_aware_anima/gates/g34.py`) |
| Dated reports | `reports/` (gitignored) here for the blind sets s19–s28; the rest under `_archive/cjk_aware_anima/{plans,reports}/` |
| Bench envelopes | `bench/cjk_distill/results/`, `bench/cjk_adapter/results/` |

## Open remainder

None owned here. KO / ZH grid validation and the symbol register (`tags_sym`,
the `u5-sym-r256` arm) were never shipped; a pack that trains them is the
same recipe with `cache_sym` appended.
