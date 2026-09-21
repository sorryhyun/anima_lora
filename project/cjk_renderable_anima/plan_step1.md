# plan_step1 — multi-glyph exposure inside step 1 (proposed 2026-09-20, **gate failed the same night**)

> **Status: dropped as written.** The boost arm read ≈ 0 on the boosted rows
> (−0.023 [−0.07, +0.00], held −0.042 — identical to the unboosted partner)
> although their drift doubled. Draws per row are not what gates sentence
> content, so the budget arithmetic below has no measured basis. Record:
> [`sent_run.md`](sent_run.md) *Is "train longer" the only lever left?* and
> `reports/step2_0920_box_weight_smoke_2026_09_20.md` *The boost gate*. Kept
> for the inventory table and M0's design; the standing reading is variety of
> strings per row (the 8 rows ride a median 30 distinct strings, the gaining
> rows 60).

Index: [`README.md`](README.md). The live plan is [`plan.md`](plan.md); this
file is one proposed change to its *Step 1* recipe and **does not start until
the gate below passes**. The measurements it rests on are
[`sent_run.md`](sent_run.md) items 7–10.

> Every launch states its pack:
> `ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack` (raw).

## Why

- A row gains sentence content only after ≈ 1 000 draws **inside multi-glyph
  items** (`sent_run.md` item 8): rows under ≈ 400 multi-glyph items per 10 k
  gain +0.000 over the seed at μ 0.3 and at μ 0.1; rows above it gain
  +0.15 … +0.23. That is step 1's own exposure number, paid a second time.
- Step 2 cannot pay it. Real text is Zipfian — the median row is carried by
  27–40 of 10 k items — and 6 k × 4 is ≈ 335 row draws per row even under
  perfect balance. The step-2 budget is 6 k (user, 2026-09-20).
- Step 1 is where the step budget already is (53 k, ≈ 9 h), it is
  row-balanced by construction, and it spends all of it on "this one glyph,
  large, alone" — the habit step 2 then moves the frequent rows *against*
  (cos(Δ, seed) −0.2 on the heavy rows).

So: let a row meet its neighbours inside step 1's budget, and leave step 2
the part that is global and fast — count and layout (read length 4.3 → 11.7
chars on every item within 6 k, item 10).

## The gate this plan waits on

Arm `rows_step2_0920_plain_bs05c25_boost8_6k` (jobs `20260920-203327-b24831`
train + eval + native, `…-5521aa` sentence natives): Round 2's μ 0.3 argv plus
`--row_boost 1101,192,831,145,237,672,521,220` (ろ 事 カ そ ら め 知 も),
each lifted from 259–787 to 1 007–1 612 expected draws.

```bash
.venv/bin/python project/cjk_renderable_anima/src/probe/row_dose.py \
    --data output/wake_probe/data_step2_0919 \
    --seed output/wake_probe/rows_step2_0920_seedread \
    output/wake_probe/rows_step2_0920_plain_bs05c25_6k \
    output/wake_probe/rows_step2_0920_plain_bs05c25_boost8_6k \
    --rows 1101,192,831,145,237,672,521,220
```

| read on the `--rows (8)` bin (44 pieces, 24 held) | verdict |
|---|---|
| gain ≥ ≈ +0.15 with the interval off 0, held moving with it (partner arm: −0.023 [−0.10, +0.05]) | exposure is causal → M0 below |
| gain on the trained strings only, held ≈ 0 | the boost memorised strings (a rare row has 13–50 of them, repeated ≤ 5×) → the pool, not the mix, is the limit; this plan waits for V (`plan.md`) |
| ≈ 0 everywhere | the slope of item 8 is something the frequent rows share besides exposure → this plan is dropped and recorded in `sent_run.md` |

Also read beside it: `sub_exact.py` pooled lift vs Round 2 (+0.099), eval
singles (15 / 7 / 6), native あ か す 日 (9 / 4) — the boost shrinks every
other item's share × 0.78, so a small drop there is the price of the probe and
not a verdict.

## The change

Step 1's data goes from `--scene_mix single=1.0` to singles **and** short real
words, with the short items drawn **by row**: cycle the rows, and for the
current row take the least-used word that carries it. Every row then gets a
known number of multi-glyph draws, the way it already gets a known number of
single draws.

**Budget, same 53 k steps × 4.** Today: 212 k single items ÷ 374 rows = 567
draws per row, all single. At `single=0.5,short=0.5` with 3-piece words:
106 k singles = 283 single draws per row, plus 106 k words × 3 pieces = 850
multi-glyph draws per row — **1 130 draws per row for the same steps**,
because a word pays every row in it. Whether a multi-glyph draw is worth a
single draw for *identity* is the thing M0 measures: the glyph is a third of
the height (S1b: hits fall under 64 px) and the row shares the box.

## What the inventory can carry

Real words only (standing rule, 2026-09-16). From the dialogue file
(`dialogue_2_10.tsv`, 42 973 distinct lines), lines of 2–4 Qwen pieces whose
every piece has a row in `step1_0920`: **2 971 words**. Rows by how many of
those words carry them:

| class | rows | 0 words | 1–4 | 5–19 | ≥ 20 |
|---|---|---|---|---|---|
| hiragana | 81 | 0 | 9 | 15 | 57 |
| katakana | 83 | 2 | 13 | 41 | 27 |
| punctuation units | 10 | 1 | 2 | 0 | 7 |
| kanji (`kanji:200`) | 200 | **78** | **81** | 40 | 1 |

Kana can carry the mix today (140 of 164 rows have ≥ 5 words). Kanji cannot:
a kanji word needs its okurigana or partner piece, and those are the cold
multi-glyph pieces of `plan.md` *V* (好き 言って 先生 …). So the plan is staged
on the kana block, and the kanji rows join when V's tiers — or a dictionary
source filtered to the inventory — give them words. A row with < 5 words stays
singles-only rather than seeing one word 200 times.

## Decisions taken here, and the ones M0 takes

| knob | value | why |
|---|---|---|
| loss | **plain** (`--pair_loss 0 --lr_rows 1e-3`) for the whole mix | ΔFM is killed on multi-glyph items (S2a) and is the weak loss at full inventory (`plan.md` S1a). A per-kind loss (ΔFM on singles only) needs batches keyed by kind; owed only if S1a keeps ΔFM |
| σ band | 0.5–0.9 | order and count are decided at 0.5–0.8, identity at 0.7–0.9; a run mixing both takes the lower edge (`findings.md`). M0 has the 0.7–0.9 arm if singles pay |
| in-box loss | `--box_share 0.05 --box_share_cap 0.25` | step 2's value; `--box_share 0.25` caps 77 % of multi-glyph items at 0.75. Singles inside the mix read under it for the first time — M0 |
| mix | `single=0.5,short=0.5`, `--short_pieces 2-4`, one column | count must be predictable from the caption's ext-token count in **one** distribution (`findings.md`); 2–4 keeps the glyph ≥ a third of full fit |
| word draw | by row, least-used word, **per-word cap 25 items** | 25 is what `step2_0919` gave a string; below 5 words the row is left to singles |
| start | cold, all rows together | this is a step-1 table. The no-cold-row rule (2026-09-18) is about sentence passes; M0 reads whether cold rows sort out who draws what when half the items are singles |
| `--short_lexical` | 0 | `step2_0920b`'s setting; kana-only words are what the kana block is made of |

## M0 — the mechanism, micro (≈ 40 min per arm)

24 kana rows chosen greedily to maximise the 2–4-piece words they cover among
themselves (the data log prints the word count; want ≥ 150 words, ≥ 32 held
out by string). Cold, plain, 3 000 steps = 500 draws per row in arm A.

| arm | data | band |
|---|---|---|
| A | `single=1.0` — the recipe | 0.7–0.9 |
| B | `single=0.5,short=0.5`, row draw | 0.5–0.9 |
| B′ (only if B loses singles) | B's data | 0.7–0.9 |

Reads, all three on every arm: eval singles on the 24 rows; **piece hit on the
held words** (`row_dose.py`, A as `--seed`); native singles on four of the
rows × `en,swap`; the sheets.

- **Pass:** B's held-word gain over A ≥ +0.15 with the interval off 0, and
  B's singles and native within A's rerun floor. Then the 1 000 multi-glyph
  draws can be paid inside step 1.
- **B wins words and loses singles:** the multi-glyph draw is not an identity
  draw at a third of the height. Try `single=0.7`, then stop — that is the
  answer that the two exposures do not share a budget.
- **B ≈ A on words:** cold rows do not learn context in the mix; the
  fallback below is what is left.

12–24-row arms are ceilinged (83 % at 250–500 draws per row) — M0 settles the
mechanism, never the budget.

## M1 — the kana block at scale (≈ 4.5 h)

164 kana + 10 punctuation rows, cold, B's recipe, 174 × 600 ÷ 4 = **26 k
steps**. No control run: the same rows inside `step1_0920` (567 single draws
each) are the singles-only reference, read on the same eval.

Then the read the plan is for — **step 2 unchanged** (`step2_0920b`'s data
argv, Round 2's train argv, 6 k) on
`--init_rows rows_step1_0920_s53k,<M1 table>` (M1's kana override, the kanji
stay `step1_0920`'s):

- `row_dose.py` against a seed read of the merged table: the < 400 bins, which
  are +0.000 today, move for kana rows;
- `sub_exact.py` pooled lift above Round 2's +0.099 with `_held` moving;
- native あ か す not below the seed's — they no longer need step 2 to move
  them far;
- any multi-glyph exact > 0 would be the line's first.

The merge is cross-recipe (plain kana on a ΔFM table: shared directions agree
at cos 0.27–0.51, `reports/table_geometry_2026_09_18.md`). If the merged
table's singles read below both parents, run M1's recipe over all 374 rows
instead (kanji rows singles-only inside it) — that is M2 early, not a new
question.

## M2 — the full table (53 k, the budget that exists)

M1's recipe over the whole inventory, kanji joining the word draw as V's tiers
land. It replaces `step1_0920` as the seed only if it holds that table's
singles / native (20 / 8 / 8, 19 / 8) — `plan.md` S1a's plain control is the
same run with `single=1.0`, so the two are read against each other and S1a
should run first or beside it.

## Code owed (none of it before the gate)

1. `src/data/`: `--text_draw row` — cycle rows, least-used word carrying the
   row, `--word_cap` items per word, rows under `--row_min_words` left to the
   singles quota; the data log prints per-row single / multi-glyph item counts
   (min, median, the rows under target) so an under-exposed row is seen before
   the GPU is.
2. `src/data/`: `--units` selection for M0 — the greedy 24-row set, printed
   with its word count.
3. A `short_held` eval group drawn from the held-out words, so
   `src/probe/row_dose.py` reads a step-1 table the way it reads a step-2 one.
4. Tests beside the existing data-stage ones; CLI golden regenerated for the
   new flags.

Already in: `--row_boost` / `--row_boost_draws` (train stage) and
`src/probe/row_dose.py`, both from the gate arm.

## Fallback, if M0 says cold rows cannot learn context in the mix

A warm, row-drawn word pass on `step1_0920`'s kana block — 174 rows × 1 000 ÷
(3 pieces × 4) ≈ **14.5 k steps**. It pays the second 1 000 draws instead of
saving them, but evenly, and it is the only other way every kana row crosses
the threshold. Not run without the user's call: it is over the 6 k step-2
budget.

## Not this plan

- A whole-table row-balanced step 2 (≥ 18 k steps at perfect balance).
- Random strings to cover the kanji rows (standing rule: real words).
- ΔFM on any multi-glyph item; the in-box weight axis; a better singles seed
  by itself (`sent_run.md` *Not levers*).
