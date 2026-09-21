# sent_run — the sentence pass, every attempt on one page (as of 2026-09-20)

Index: [`README.md`](README.md). The forward plan is [`plan.md`](plan.md) § 2;
this file is the record of what the sentence pass has been asked to do, what
each run returned, and what is left to try. Numbers are `exact (sfx)` and
`src/probe/sub_exact.py` pooled lift unless said otherwise; **lifts from
different eval sets are not comparable** (marked † where that applies).

## What the pass is for

Step 1 gives a table where one ext row draws one glyph. The sentence pass
warm-starts that table and trains multi-glyph items so the same rows compose —
`はい`, a short line, a sentence — without losing the single-glyph function.
The rule since 2026-09-18 (user): **no cold row inside a sentence pass**; new
vocabulary is a step-1 job, merged in by ext id.

What is already known to be possible (`findings.md` *Settled — sequences*): the
frozen adapter + DiT read order, and rows trained on strings carry order and
count (strings arm, 8 k steps: first rendered glyph = caption's first piece
28/48 vs its last 5/48). Unit count is a **data-distribution prior that lands
in the rows** — singles-only rows draw one unit, strings-only rows draw several
even for a one-kana caption.

## The runs

| date | arm | seed → recipe | sentence read | singles / native | note |
|---|---|---|---|---|---|
| 09-16 | `sent_tall_sent_s24k` | 53 k table + punct, 24 k | — | — | **stopped at 10 k**: the data build held no sentences |
| 09-16 | `sent_q_sent_s24k`, `_a1_s05` | same seed, quotas + tategaki, 24 k | +0.070 † (`a1_s05`) | — | the shipped **preview pack**; `sent2_s24k` (no anchor) +0.161 † |
| 09-17 | anchor sweep | — | — | — | warm starts were being erased by Adam (row cos 0.10); `--lr_warmup 500 --init_anchor μ` fixes it. Every warm-start verdict before it is suspect |
| 09-18 | S2a plain / pair, 1.5 k | Δ1, μ 0.3 | plain **+0.131** vs ΔFM +0.081 † | native はい 5 vs 2 of 48 | **ΔFM killed on sentences.** Multi-glyph words collapse to one big glyph in both |
| 09-18 | S2a plain μ 0.1 | same | +0.086 † | はい 1/8 | looser anchor *worse* at 1.5 k — on the preview pack |
| 09-18 | S2b plain 6 k | Δ1, μ 0.3, warm rows only | +0.089 † (`short` +0.235, held ≈ +0.045) | singles = seed (12/17/18) | first pass that did not pay in singles; sentences did not move |
| — | — | — | — | — | **09-19: every run 09-17 → 09-18 trained on the baked preview pack, not the raw pack.** Arms deleted; the verdicts above are direction only |
| 09-19 | `step2_0919` 6 k | `step1_0919` (weak ΔFM seed), μ 0.3, w 4, **raw pack** | **+0.086** (held +0.141 / +0.049) | 10/4/2 = seed; はい 2/16 | table is its seed to cos 0.994 |
| 09-20 | weight smoke ×6, 2 k | `step1_0920`, μ 0.3 | +0.08 … +0.15, no order (n = 16) | singles 6 → 4, native 4 → 2 of 8 as weight rises | **in-box weight axis closed**; `short_held` falls as long items get more share |
| 09-20 | Round 2 6 k (`…bs05c25_6k`) | `step1_0920`, μ 0.3, ρ_g 0.05 cap 0.25 | +0.099 (held +0.108 / +0.033) | 15/7/6 (seed 20/8/8); native **9 / 4** (seed 19 / 8) | gate failed; a 2× better seed bought nothing |
| 09-20 | Round 2 μ 0.1 (`…mu01_6k`) | same, μ 0.1 | **+0.153** (held +0.165 / +0.067) | 14/7/6; native **5 / 2**, あ 0/32 | first arm above `step2_0919`'s CI (+0.066, [+0.003, +0.129]); all four groups rise |
| 09-20 | `step2_0920b` 6 k (`rows_step2_0920b_plain_bs05c25_6k`) | same seed, μ 0.3; data rebuilt: `--short_lexical 0 --phrase_norm 1 --text_draw balanced` (497 sentences, 3 097 shorts, every string ± 1 item) | +0.102 † (new eval strings; held +0.037 / +0.108) | **16 / 17 / 6** (seed 20 / 8 / 8); native 8 / 5 | first pass on this seed that does not pay in eval singles — `single_ext` doubles (kana-only shorts now exist); native and multi-glyph native (+0.101, 0/64) unchanged |
| 09-20 | `step2_0920b` μ 0.1 (`…0920b_plain_bs05c25_mu01_6k`) | same, μ 0.1 | — | — | **stopped at 21 min** (user): the row-exposure read below predicts its result. Dir holds `eval_coverage.json` only |
| 09-20 | seed read (`rows_step2_0920_seedread`) | `step1_0920` table copied in, `--stage eval` only, the step-2 multi-glyph eval strings | the per-piece baseline for every `step2_0920` arm (4 min) | — | keep: it is what makes a paired per-row read possible |
| 09-20 | boost 8 rows, 6 k (`…bs05c25_boost8_6k`) | Round 2 argv (μ 0.3) + `--row_boost` ろ 事 カ そ ら め 知 も → ≥ 1 000 draws each | +0.095 (held +0.075 / +0.037); the 8 rows **−0.023 [−0.07, +0.00]**, held −0.042 — the partner's numbers | 16 / 7 / 7; native 9 / 4 | **gate failed.** The rows travelled 2× further (drift 0.03–0.14 → 0.09–0.15) and gained nothing; A/B partner is the Round 2 μ 0.3 row |

Exact match on every multi-glyph eval group of every arm above: **0**.
Detail for the 09-20 rows: `reports/step2_0920_box_weight_smoke_2026_09_20.md`.

## What 2026-09-20 measured

1. **The anchor μ is the trade knob, on both sides.** μ 0.3 → 0.1 lets the
   table end twice as far from its seed (drift 0.042 → 0.089) and buys
   +0.054 pooled lift with the `_held` groups moving — and single-glyph native
   halves each time (19 → 9 → 5 of 64). Eval-frame singles do not show it
   (15 vs 14): the loss is on scene prompts.
2. **The pass changes the habit before the content.** Same prompt, same seed:
   the step-1 table answers `はい` with one large glyph (は alone 4×, one clean
   はい); after the pass it answers with a smaller multi-glyph line carrying
   the pieces in the wrong company (はかい, ばい, 笺こたをりはい). This is the
   unit-count prior of `findings.md` again — the step-2 mix is 10 % singles.
3. **The eval-frame lift does not reach a scene prompt.** Four strings
   (はい おしい やったネ ちょっと来い) × 8 scene prompts × 2 seeds: sub-exact lift
   +0.093 / +0.084 / +0.084 for seed / μ 0.3 / μ 0.1, exact 1 / 0 / 0 of 64,
   nothing past 3 glyphs even starts right. The eval frame is the bare
   template (`manga, speech bubble, japanese text. …`); the 8 native prompts
   carry no bubble / sign tag (the clause adds `japanese text` only) and
   training captions are full scene captions in several
   clause shapes — three different frames, and only the easiest one moves.
4. **The logged `loss` cannot see a sentence run.** Out-of-box is ≈ 80 % of
   it and batch noise is ±0.013. Split (`BoxSplit`, new): `in_box`
   0.153 → 0.142 over 6 k and **still falling**, `out_box` flat after
   step 1 000, σ ≥ 0.7 and σ < 0.7 falling by the same fraction (the 0.5–0.9
   band is not wasting its lower half). "Flat loss" in every earlier sentence
   report is a statement about the log.
5. **The seed does not matter yet.** `step1_0920` reads 2× `step1_0919` on
   singles and native; the sentence pass on top of each lands at the same lift.
6. **The pool is small.** `data_step2_0919`: 4 000 sentence items from
   **471** distinct strings, 5 000 shorts from **454**. 6 k steps × batch 4 is
   ≈ 25 looks at each string, in different scenes and fonts.

## What the row-exposure read measured (2026-09-20, evening)

CPU reads over the tables and `eval_reads.json` already on disk, plus one
4-minute seed eval. Ruler: `src/probe/row_dose.py`.

7. **The pass's drift is Zipfian per row.** Step-2 table − `step1_0920` seed,
   369 rows: 50 % of the drift energy sits in **8–12 rows**, 90 % in 54–83;
   spearman(|Δ|, occurrences in `train.jsonl`) = **0.94–0.95**. The heavy rows
   are the frequent function glyphs (っ ー だ あ い が は ん ！！ ・・・・),
   moving 30–100 % of their seed norm and *against* it (cos −0.2); the median
   row moves 1–5 %. In 10 k items the median row is carried by 27–40, the top
   row by 2 000–2 800: over 6 k × 4 that is ≈ 65–100 draws vs ≈ 6 000.
   `--text_draw balanced` balances strings, not rows. Δ is **not** a shared
   low-rank subspace — participation ratio 12 against 19 for a random-direction
   null with the same row norms; the low number was norm concentration.
8. **Sentence content is gated per row at ≈ 400 multi-glyph items per 10 k —
   ≈ 1 000 draws, step 1's planning number.** Paired gain over the seed on the
   same 64 eval items (piece string in the item's read, item-bootstrap 95 %):

   | row's multi-glyph items in `train.jsonl` | pieces / rows | seed hit | μ 0.3 gain | μ 0.1 gain |
   |---|---|---|---|---|
   | < 150 | 100 / 43 | 0.080 | **+0.000** [−0.06, +0.06] | **+0.000** [−0.04, +0.04] |
   | 150 – 400 | 68 / 19 | 0.059 | +0.000 [−0.08, +0.08] | +0.000 [−0.08, +0.08] |
   | 400 – 1 000 | 88 / 20 | 0.080 | **+0.148** [+0.06, +0.24] | **+0.227** [+0.13, +0.32] |
   | ≥ 1 000 | 64 / 7 | 0.266 | +0.031 [−0.09, +0.16] | **+0.172** [+0.03, +0.31] |

   `_held` strings alone give the same shape (400 – 1 000: +0.095 / +0.167;
   below 400: 0). The seed is flat at ≈ 0.07 below 1 000 and 0.27 above (っ ー
   う render from step 1), so the slope is step 2's and not "frequent kana are
   easy". 62 of the 89 rows in the eval strings, and most of the table, got
   nothing from any sentence run to date.
9. **This is what μ was trading.** μ 0.3 → 0.1 adds its lift in the ≥ 400 bins
   only — ≈ 27 rows — and those are the rows whose drift it doubles. Native
   あ か す are among them (あ 0/32 at μ 0.1); eval singles average over the
   rows that did not move and hide it (item 1).
10. **Count habit is global, content is per row.** Read length on the same
    items: 4.3 chars (seed) → 11.7 (μ 0.3) / 9.4 (μ 0.1), for every item; the
    right glyphs appear only for the exposed rows. Item 2 above, measured.
11. Ruler gaps: ！！ ・・・・ ！ read 0 hits at ≈ 2 000 occurrences — most likely
    the sfx reader dropping punctuation, not a render verdict. や (≈ 500 items)
    0/10 is the one real exception.

**Budget arithmetic.** Items carry 5.2 ext pieces on average, so 6 k × 4 is
≈ 124 k row draws — ≈ 335 per row over 369 rows *under perfect balance*. No
sampler lifts the whole table past ≈ 1 000 inside a 6 k pass; on today's
distribution the median row would need ≈ 25× the steps.

## Is "train longer" the only lever left?

Superseded by items 7–10: longer on this distribution buys the frequent rows
more drift and nothing for the rest. What is live:

- **The exposure hypothesis test — failed (2026-09-20 night).** `--row_boost`
  (mixed-shape batcher) repeated the items of ろ 事 カ そ ら め 知 も (ext 1101,
  192, 831, 145, 237, 672, 521, 220) until each row expected 1 007 – 1 612
  draws (from 259 – 787; 10 000 items → 12 865 slots, item repeats ≤ 5). Those
  8 rows (44 eval pieces, 24 held) read **−0.023 [−0.07, +0.00]** against the
  seed, held −0.042 — the unboosted partner's numbers to the digit, and the
  μ 0.1 arm's too. The boost reached them: their drift doubled (0.03–0.14 →
  0.09–0.15 of the seed norm), in the partner's direction (cos 0.53–0.76).
  Pooled lift, eval singles, native and multi-glyph native are all the
  partner's. **Draws per row are not what gates content**, and neither is the
  anchor on those rows (μ 0.1 moves them as far for the same −0.023).
- **What item 8's slope can still be.** The boost repeats a row's own strings:
  the 8 rows ride 13–51 distinct strings (median 30) where the rows that gained
  ride 22–123 (median 60) and the ≥ 1 000 rows a median of 106. Frequent rows
  differ from rare ones in *variety of company*, not only in draws — that is
  the pool bullet below, now first in line. Also not excluded: the gaining rows
  are the function kana, which sit in every position of every string.
- **`plan_step1.md` (archived) is dropped as written** — its gate read
  "≈ 0 everywhere". Its budget arithmetic (1 000 multi-glyph draws per row
  inside step 1) rested on draws being the unit. A row-drawn word mix would
  still raise variety per row as a side effect, so M0 is not refuted, but it
  no longer has a measured reason to work.
- **A per-row drift cap in place of the uniform μ** is no longer motivated:
  the rows that moved further (boost, μ 0.1) did not gain.

Still open from the earlier list, unchanged by the above:

- **The pool.** 925 strings; a boosted rare row sees its 13–50 strings up to
  5× more often, so string memorisation is the boost arm's failure mode — read
  `_held`. A larger covered pool (S2a's wider inventory passed 11 119 lines /
  6 142 sentences against today's 4 564 / 471) widens every row's strings.
- **The singles share.** The mix is 10 % singles and the count habit moves
  globally (item 10). A 30–50 % singles share may hold native singles at a
  given μ; item 9 says the native loss is the heavy rows' drift, so read it
  per row. One smoke arm, `--scene_mix` only — after the boost gate.
- **The frame.** Training never shows a caption without a bubble / sign
  tag, and the native prompts have none. Either the native ruler should
  carry the tag the card will tell users to write, or the data should include
  tag-less captions. Not a training lever until that is decided.
- **Not levers** (do not re-propose): ΔFM on sentences, the in-box weight,
  a better seed by itself, row blocks, a shape-neighbour warm start, **row
  oversampling of an unchanged string pool** (`--row_boost`).

## Reading rules for the next run

- Sentence arms are ordered on `sub_exact.py` pooled lift with the `_held`
  groups, never on exact match (0 everywhere) and never on the mixed `loss`.
- Always read single-glyph native (あ か す 日 × `en,swap`, of 64) beside it:
  that is where the pass pays.
- Read multi-glyph native (`--eval_tag sent`) before calling a lift a result.
- Read `src/probe/row_dose.py` against `rows_step2_0920_seedread` (or a seed
  read made the same way for a new eval set): a pooled lift can be ≈ 27 rows.
- Launch with `ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`
  (raw); a preview-pack base invalidated two days of runs.
