# product_criteria — what a table has to do to ship (2026-09-25)

The pack's job on a manga page: **the requested Japanese text comes out, and
the page still looks like the page.** Two axes, read together on one contact
sheet; neither alone accepts a table. The 24-row series
(`reports/conflict_joint_2026_09_25.md`) showed why both are needed — hit
counts reward a row that pastes a white-box glyph over the scene, and the
eye reads the μ 0 / lr 1e-3 joint table as the most *harmonized* while its
rows had walked off identity (い → `u`, 日 → 目).

## Axis 1 — the text (`native_sent`)

The ruler of record is **`native_sent`**: the target strings rendered in
native scene prompts (`Japanese text reads as "…"`, `en` clause), 8 prompts ×
2 seeds = 16 per string, scored by both readers (sfx + VL16, `hit_sfx and
hit_vl`). Not `single` / `word` exact (flat one-bubble templates, floor-
saturated on multi-glyph rows) and not `native` on lone glyphs — the product
draws words.

Strings, and the rows they need (Qwen pieces; a target word needs every row
— `sent_run.md`):

| string | pieces | note |
|---|---|---|
| はい | は · い | two singles |
| おしい | お · しい | single + piece |
| やったネ | や · った · ネ | single + piece + katakana single |
| ちょっと来い | ちょっと · 来 · い | piece + kanji + single |
| こんにちは | こんにちは | one piece |
| + `target` | はい / こんにちは in the user's verbatim captions (`target/`, 8 + 6) | the shipped prompt shape |

Add strings only with their rows in the run's inventory. Report per string
(of 16) and the total; a table's Axis-1 number is the **total both-reader
hits over the string set**, with the per-string row beside it.

Noise: 16 renders per string is ±3 **as a rule of thumb, not a confidence
interval** — the 16 are 8 prompts × 2 seeds, so they are not 16 independent
draws, and two same-config tables differed by 9 of 128 on `native`. Read a
difference only when it repeats across strings (sign over strings), never
off one string or the total alone. Strings are not independent either:
はい / おしい / ちょっと来い share い, and a total over the set moves with
that one row — count the sign over strings that share no row before
calling a direction. Power comes from more strings, not more seeds
(`project_cjk_blind_pairs_protocol`).

### Dev vs acceptance (2026-09-25)

The five strings + `target` above are the **acceptance set**: read once
per candidate table, never used to pick shares, μ, lr or a checkpoint. A
table tuned on them is accepted on its own training signal. Choices
between arms are made on a **dev set** of the same shape and disjoint
strings — native prompts, the small-px / mixed-string cases, pieces the
acceptance strings do not contain (e.g. ありがとう / それを / すごい in
`run0925_72`'s inventory) — read with the same both-reader rule and the
same sheet. The dev set is what a gradient-bank surrogate (`idea.md`) is
calibrated against; the acceptance set stays untouched until one arm is
chosen. Both sets are listed in the run file (`[eval] sent_strings` for
acceptance, a `dev_strings` key when the eval grows it), never chosen
per run.

## Axis 2 — the page (EN-reference similarity)

The same prompt and seed rendered with the EN clause (`English text reads
as "hi"`, `native_enref/`) is what the base draws for this page; the JA
render should differ from it only inside the text box. Ruler: **token-wise
PE-Spatial cosine outside the text boxes** — for every spatial token outside
the union of the JA box and the EN box, cos(token, ref token), averaged; not
the pooled mean. The pooled `en_cos` in `native_reads.json` (mean of tokens,
then one cosine) is **not usable** for this: the whole series sat at
0.916–0.930, below the 0.93–0.97 floor, and the worst tables scored highest
(0507 μ 0.01 0.930, 0309 ctx 0.928 — less text drawn, more like the EN page).
`en_cos_out` (same pooling, outside the boxes) is flat for the same reason.
`box_iou` (0.17–0.23 everywhere) is placement, kept as a side column.

Until the token-wise ruler is in `eval/enref.py`, Axis 2 is read on the
contact sheet by eye against the seed and the EN ref, and the two known
failure shapes are named: **paste** (a flat white box or black block with the
glyph, layout overridden — the seed's 日/en p4 s0, あ/en p4 s1) and **wipe**
(the scene's own bubble with the base's guess in it — the μ 0 joint's
なるめら, `uu`, 目).

## The sheet

One PNG per comparison, `contact_native_<arms>.png`: a block per string ×
clause, the arms as rows (seed always the last row), the 8 prompts × 2
seeds as columns, so a column reads straight down as *same page, N tables*.
Green = both readers, orange = one, red = none, the per-row count at the
left. Both axes are read off this one sheet; `native_sent` and `target`
blocks first, lone glyphs after.

## Accept

A table replaces the current one when, on the same string set and prompts:

1. Axis 1 total is up and no string is down by more than its own noise (3);
2. Axis 2 is not below the seed's — no new paste, no new wipe — on the sheet
   (or, once it exists, the token-wise outside-box cosine within the seed's
   band);
3. `en` (24 EN strings) stays 24/24.

Everything else — `single` / `word` exact, `cf_sense`, `warm_cos`, loss — is
diagnosis, not acceptance.
