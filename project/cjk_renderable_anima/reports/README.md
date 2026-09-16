# reports — the dated run record of cjk_renderable_anima

What ran, in order, with the numbers and the decision each run forced. One
file per stretch of related runs; text inside a report is the voice of its
date and is not rewritten when a later run changes a number (the later
report wins). Settled verdicts live in [`../findings.md`](../findings.md), the
state digest in [`../README.md`](../README.md), the forward plans in
[`../plan.md`](../plan.md) (P line) and [`../plan_synth.md`](../plan_synth.md)
(S line). Until 2026-09-15 everything from W2d on was one file, `history.md`;
it was split here without edits.

## Index (chronological)

| report | dates | what it holds | headline |
|---|---|---|---|
| [`krzh16_2026_09_16.md`](krzh16_2026_09_16.md) | 09-16 | 16 Korean / simplified-Chinese rows on the micro recipe, read against the 53k table; the `~` tag-leak row; concat feasibility | the DiT holds Hangul and simplified units; KO/ZH rows share the kana manifold (trigger + orthogonal identity), hanzi neighbour JA rows by radical; row addresses are reproducible across runs (own char 0.5 / shape 0.3 / random 0.03) |
| [`wake_w0_w2_2026_09_13.md`](wake_w0_w2_2026_09_13.md) | 09-13 | the wake hypothesis, Probe 0/1, address geometry, 256² / 24-kana / balanced / σ-band arms, native-rendering and kanji probes, the case for W2d | the frozen DiT holds kana units; a rows-only delta draws them; identity is decided at σ ≈ 0.8 |
| [`wake_plan_2026_09_13.md`](wake_plan_2026_09_13.md) | 09-13 (status notes to 09-14) | the plan as written: where the line stood, target artefact (a vocab pack), W3 / W4, kill criteria, shelved W2 levers, **EN safety**, instruments & gotchas | superseded as decision state by `plan.md` / `plan_synth.md`; the EN-safety and gotcha lists still apply |
| [`wake_w2d_encoder_2026_09_13_14.md`](wake_w2d_encoder_2026_09_13_14.md) | 09-13 → 09-14 | W2d glyph encoder: Run 1 (ten launches), 1b / 1b amended, 1c decorrelation, 1d free-residual hybrid, Run 2 kanji + IDS composites | table is rank-1; `g + f` renders every trained single (24/24), held-out flat; addresses do not compose → kanji is an exposure budget |
| [`wake_words_strings_2026_09_14.md`](wake_words_strings_2026_09_14.md) | 09-14 pm | Run 3 word addresses, base-model order probe, `classify_str` σ diagnostic, strings arm | a row can be a word; order reading is pretrained (σ 0.5–0.8); a static table carries order + count but absorbs the unit-count prior |
| [`wake_canvas_scenes_2026_09_14.md`](wake_canvas_scenes_2026_09_14.md) | 09-14 night | canvas-shape gate + P0a, table-parts probe, scenes s0, S0 build + launch | 384² alive, mixed pool is the recipe; glyph is conditional on the canvas mode → the S line; 186 kept scenes |
| [`synth_s0_s0b_2026_09_15.md`](synth_s0_s0b_2026_09_15.md) | 09-15 morning → afternoon | S0 result, S0b build + result, P0b warm-start probes, quote-direction (Q) probes, cap-only launch | composites decouple canvas from glyph but miss the gates; cap 1.5 gave `c_flat` the trigger; Q is a canvas-free trigger, not a drop-in |
| [`synth_micro_loop_2026_09_15.md`](synth_micro_loop_2026_09_15.md) | 09-15 16:18 → 20:15 | 6-row micro arms (cap / no `c_flat` / Q), composite share 0.9, EN-reference ruler, swap-clause natives | cap ≠ lever, composite share is; rulers = `en cos` / `box IoU` vs `English text reads as "hi"`; rows are JA-frame-bound; Q is glyph-dependent |
| [`transplant_2026_09_16.md`](transplant_2026_09_16.md) | 09-16 | no-training transplant of donor residuals onto the 53k shared direction (`probes/transplant_table.py`), then the `--pin_dir` arms (inherited m̂ frozen, residual only trained) | flat-trained residuals (P0b, Run 3) render 0/64 on the 53k m̂ = m̂ alone; the composite-trained micro6 residual renders 23/64 with the scene kept → flat-only for new glyphs is closed, composite + pinned m̂ is the arm |
| [`rows_manifold_2026_09_16.md`](rows_manifold_2026_09_16.md) | 09-16 | plan_synth branch C: the 53k full-inventory table read in row space and at the adapter output under the five frames (`probes/rows_manifold_probe.py`) | one shared direction (18 % energy, = the S0/S0b row mean, ⟂ Q) + near-orthogonal residuals; weak small-mark shape locality (ば↔ぱ) only; no shared representation with `She is saying "…"`; katakana miss is not row-space crowding; word rows untrained |
| [`synth_sentence_launch_2026_09_16.md`](synth_sentence_launch_2026_09_16.md) | 09-15 night → 09-16 evening | `plan_synth.md`'s status notes, recipe of record + exposure curve, frame-mix 2×2 / micro tables and three-branch next-steps record, moved out unchanged; tategaki wrapping, tall-bubble pool, punctuation and sentence arm launches; sentence arm stopped | ≈ 1 000 draws per row and pinned shares are the budget; wrap fit is bounded by bubble height (tall pool); `sent_s24k` stopped at ≈ 10.3 k — its data held no sentences |

## Conventions

- **Adding a run:** append a dated `## <run> (<date time>): <one-line verdict>`
  section to the latest report while it is the same stretch of work; start a
  new `<line>_<topic>_<YYYY_MM_DD>.md` when the question changes, and add its
  row above. Keep the title + one-paragraph blockquote summary at the top.
- Paths in backticks are relative to `project/cjk_renderable_anima/` (probes,
  plans) or the repo root (`output/…`, `make …`), as in the original record.
- Job ids, arm dirs and argv go in the report; the verdict also goes to
  `../findings.md` once it is settled.
