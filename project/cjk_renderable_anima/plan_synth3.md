# plan_synth3 — after Δ1: the sentence step

> **Status 2026-09-18 evening.** This plan is **S2** — the sentence step
> (step 2). Step 1 (the seed-table recipe and the kanji budget K) is
> [`plan_synth4.md`](plan_synth4.md).
> R, S2-smoke, S2a and **S2b** are read — S2b was round 1 of the vocab
> step → merge → sentence step loop (*S2b* and *The loop* below).
> **The loss is decided: plain.** S2a (`reports/synth_s2a_2026_09_18.md`)
> put plain above ΔFM on every sentence ruler that moves (sub-exact pooled
> +0.131 vs +0.081, native `en` 5 vs 2 / 48); ΔFM is killed on sentences
> (user decision). The ruler that decided it is `src/probe/sub_exact.py`:
> exact match is floor-saturated on every multi-glyph group.

> **Caveat found 2026-09-18 night — every run since 2026-09-17 17:36 trained
> on the baked preview pack, not the raw pack.** `configs/base.toml`
> `vocab_pack` was pointed at `anima_cjk_vocab_pack_preview` when that pack
> was built (17:04; committed `b1028bd4` 23:20), and the preview pack **is**
> the raw pack + `sent_s24k_a1_s05`'s delta (502 rows, mean |Δ| 105;
> preview − raw equals that table to 4e-6). `wake_probe` reads the
> configured pack, so per the job logs' pack sha (`7b9fce0bb57b` raw,
> `5f52aefce82a` preview): raw = `sent2_s24k`, `smk3k_*`,
> `sent_s24k_a1_s05`, and the first ΔFM arm `pairEN_s1500`; **preview =
> every other `pair_d0` arm, Δ0b, the row-block arms, Δ1 53 k, S2-smoke,
> S2a, the KR / ZH blocks, S2b.** What that does to this plan's numbers:
>
> - **Runs on the same pack still compare** (ΔFM vs plain in S2a, μ 0.3 vs
>   0.1, the row-block arms), and each arm's eval is the table it trained
>   (pack + delta), so the scores are real scores of that stack.
> - **Δ1 is not a ΔFM-from-scratch table.** 355 of its 356 rows were already
>   trained in the preview pack (|S| 137 vs |Δ1| 79 on them, cos 0.137): Δ1
>   is a 53 k ΔFM residual on top of the plain sentence table, and its
>   12 / 17 / 18 singles sit beside that table's own 11 / 19 / 19. Its
>   `trained.pt` means nothing on the raw pack; "v1's source is D1" reads
>   preview + Δ1.
> - **"Δ1 ↔ plain 0.14" is the residual's cosine to the table under it**
>   (the same 0.137), not evidence that the shared direction is per-loss.
> - **Δ0's verdict compared across packs:** `pairEN_s1500` 10/24 ran on raw,
>   `pair0_s1500` 20/24 on preview, where the 12 dakuten rows were already
>   trained `kana_ext` rows. "ΔFM saves no draws" is unmeasured.
> - **The smoke's floor finding** ("untrained katakana dakuten render at
>   floor 10/12", "ΔFM damages floor-positive rows") read the preview pack's
>   trained rows as a floor. It may still hold as "ΔFM wears down trained
>   plain rows"; on the raw pack it is unmeasured.
> - **S2b's 15 small-kana rows carry their delta twice** — baked in the
>   preview pack and added again by `--init_rows <sent_s24k_a1_s05>`.
>
> **Δ0 re-run on the raw pack — read 2026-09-19**
> (`reports/s2b_and_raw_pack_rerun_2026_09_19.md`, `ANIMA_VOCAB_PACK=…/anima_cjk_vocab_pack`,
> `base.toml` untouched): floor **0/24**; plain 7, ΔFM lr 1e-3 / 2e-3 10,
> **ΔFM lr 5e-3 13** of 24 at 1 500 steps (native `en` 13 / 10 / 14, ΔFM
> holding the scene as before); `--row_blocks 62` **0/24** (4/24 at 5e-3 with
> row norm 248) — its 16/24 was fine-tuning trained rows; glyph-size jitter
> 0–1/24. The katakana floor finding is retired; "ΔFM saves no draws" and
> "2e-3 × 1 500 ≡ 1e-3 × 3 000" are not reproduced. S2b itself is read in
> the same report (singles 12 / 17 / 18 held, pooled lift +0.089, はい 1/8).
> The body below is not yet re-derived; every launch from here states its
> pack.
>
> **Preview-pack arm dirs deleted 2026-09-19 (user).** All 31
> `output/wake_probe/rows_synth_*` arms whose job log shows only the preview
> sha are gone — every `pair_d0` arm without a `_raw` suffix except
> `pairEN_s1500`, `pair_d0f`, the row-block arms, `pair_kr` / `pair_zh`,
> `s2_smoke_*`, `s2a_*`, `d1_d1_s53k` (Δ1), `d1_merge_punct`,
> `s2b_plain_d1_6k`. The reports are the only record; paths to those arms
> below and in `reports/` are dead. Kept: the raw arms, every `data_synth_*`
> dir (images do not depend on the pack), the daemon job logs, and
> `models/vocab_packs/vocab_pack_test` (S2b's baked table).

**`step1_0919` — Δ1 re-run on the raw pack, small kana added — launched
2026-09-19 10:07**, job `20260919-100716-22fdca`, data
`data_step1_0919`, arm `rows_step1_0919_s53k`. Δ1's argv (R below) with two
changes: `ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack` (log sha
`7b9fce0bb57b`) and `--units small`. ΔFM, lr 2e-3 cosine, 53 k × batch 4,
373 rows ≈ 568 draws/row. `--units small` (`src/data/inventory.py`
`small_digraphs`) draws each of the 18 small kana inside up to 6 two-glyph
digraphs whose Qwen pieces are the host row + the small row (あっ きゃ しょ
ニャ トゥ; corpus-attested first, then the yōon / gairaigo tables), each small
kana at the pool mass of one weight-1 unit; eval group `single_small`. The
frequent uses (って ちゃ った じゃ ック ティ) are single Qwen pieces — word rows,
the loop's vocab step. With it a sentence step needs no second warm source
for the small-kana rows (S2b's `--init_rows <sent>,<Δ1>` and its
double-delta defect).

**`step1_0919` read 2026-09-19** (`reports/step1_0919_2026_09_19.md`):
singles 10 / 5 / 3 of 36 against the plain seed's 13 / 18 / 18 — ΔFM from
scratch is weak at full inventory, no bug, and `--delta_scale 1.7` does not
recover it.

**`step2_0919` — the sentence step on that table, launched 2026-09-19
20:38** (user: run it as it is), job `20260919-203858-051f61`, data
`data_step2_0919` (job `20260919-202821-2392df`), arm
`rows_step2_0919_plain_6k`, raw pack (log sha `7b9fce0bb57b`). S2b's data
argv + `--units small`, S2b's train argv with one warm source:
`--init_rows rows_step1_0919_s53k/trained.pt` (369/369 rows warm, the 18
small kana included), plain, μ 0.3, warmup 500, lr 1e-3 cosine, σ 0.5–0.9,
6 k. The pool is S2b's — 4 564 covered lines, 480 sentences, 470 shorts:
the small kana were already covered through `kana_ext`, so `small` makes
them warm, not more numerous. Eval adds `single_small`.

## Where the line stands going in

Every number below is `exact (sfx)` off the arm's `report.md`, seed-0/1 pooled.

| table | what it is | single | ext | kanji | word | short/phrase/line | en |
|---|---|---|---|---|---|---|---|
| `src53k` (`full_s53k_qoff`) | 53 k plain FM, 434 rows, 490 draws/row | 13/36 | 18/36 | 18/36 | 0/32 | — | 24/24 |
| `sent_s24k_a1_s05` | + sentence pass, warm, `--lr_warmup 500 --init_anchor 0.3` | 11/36 | 19/36 | 19/36 | 0/32 | **0/32** each | 24/24 |
| `sent2_s24k` | a *second* sentence pass warm-started off that | 0/36 | 0/36 | 0/36 | 0/32 | **0/32** each | 24/24 |
| Δ1 `d1_s53k` | ΔFM, 356 rows, 596 draws/row, uniform weights | 12/36 | 17/36 | 18/36 | (no word rows) | 0 exact; sub-exact pooled **+0.178** vs `src53k` +0.098 (P = 0.016) | 24/24 |

(`src53k` ext was printed here as 20/36 on 2026-09-18 morning; the arm's
`report.md` and the Δ1 gate both say 18/36.) Δ1's native, by clause (both /
joint / tail of 64): `en` 22 / 14 / 11 vs plain 36 / 24 / 16; `swap` 19 / 15 /
4 vs plain 18 / 12 / 8. か and す carry −13 of the −14 on `en` and both gain
on `swap`. The cross sheets (`src/probe/cross_sheet.py`, `x_か_en.png`) show
the mechanism: plain FM writes one large か on a wiped canvas; ΔFM keeps the
scene and か sits *inside the base's free-running JA pseudo-text*
(`きじたと(か)`) — the exact read is lost to co-text, not to a missing glyph.
That is the wipe the Δ0 report said ΔFM removes, now measured on the full
table — and the failure S2's premise claimed to fix and did not (S2a).

## The ruler this plan is built on

**Exact match cannot order two sentence arms.** Every multi-glyph group of
every arm to date is 0/32, so the ruler that decided the S and ΔFM lines has
no resolution exactly where step 2 lives. `src/probe/sub_exact.py` (written
2026-09-18, no GPU — it reads the `eval_reads.json` the eval stage already
wrote) supplies one that does:

    glyph recall  |{c ∈ set(ref) : c ∈ read}| / |set(ref)|
    perm control  the same recall of the group's *other* refs against this
                  same read, averaged
    lift          recall − control

The control shares the read, so read length and the arm's general JA-glyph
habits are held; what is left is whether the read carries *this* item's
glyphs. Measured:

| arm | short | short_held | phrase | phrase_held | line | corpus | word | pooled |
|---|---|---|---|---|---|---|---|---|
| `sent_s24k_a1_s05` | +0.113 | +0.071 | +0.053 | +0.012 | +0.100 | +0.165 | +0.009 | **+0.070** |
| `sent2_s24k` | +0.171 | **+0.233** | +0.124 | +0.128 | +0.131 | +0.203 | +0.152 | **+0.161** |

Pooled difference **+0.091, bootstrap 95 % CI [+0.051, +0.129]**, n = 212 each.

Three things follow, and they are the plan:

1. **Both sentence tables carry real multi-glyph content** — lift > 0 on every
   group, including `short_held` and `phrase_held`, strings the table never
   saw. That is not what "0/32 everywhere" said.
2. **The second pass is the better sentence table, and it is the one that lost
   every single.** `sent2_s24k` doubles the lift and reads 0/36 on singles.
   Read together with the anchor sweep (μ ≥ 0.1 keeps ≥ 80 % of the source's
   singles, and `sent2_s24k` ran with **no anchor at all**), the trade looks
   like a knob, not a cliff: the anchor buys singles back and, on this
   evidence, spends sentence lift. **μ was the candidate lever** — and the
   first sentence-ruler comparison says it is not one at micro scale: S2a's
   plain μ 0.3 +0.131 vs μ 0.1 +0.086 on the same data and seed (−0.046,
   CI [−0.118, +0.026]; native はい 5 → 1/8). The anchor stays at 0.3.
3. **Sentence work must be gated on lift, with exact match as a secondary.**
   A gate written as "short ≥ 1/32" is a coin flip; "pooled lift above the
   baseline's CI" is a measurement.

## R — read Δ1 — done 2026-09-18

`reports/synth_pair_delta1_2026_09_18.md` discharges the gate, the sheets and
the exposure read:

- **Gate:** missed on singles (12 vs 13) and native joint (29 vs 36), passed
  on the tail (15 vs 24). ΔFM is not a drop-in for the seed table.
- **The uniform-exposure read:** at weight 1 and 596 draws/row, kana 12/36,
  `kana_ext` 17/36, kanji 18/36. **Per draw, kanji are not harder than kana**
  — the 53 k run's kanji 18 vs kana 13 was its 2× exposure. K's weights follow
  from this: no extra draws for kanji; if anything, plain kana / katakana is
  the weak kind.
- **Flat-0 drift on kanji:** 日 is the one glyph that does *not* drop on
  `en` (5 → 6 both); it moves the scene most (lowest en cos, lowest box IoU)
  but is not drawn as Latin. Risk retired for Δ1's recipe.
- **`sub_exact.py` on Δ1 is not meaningless** — the eval set carries
  `line` / `combo` / `corpus` even though training had no multi-glyph items,
  and Δ1 orders *above* `src53k` there (+0.178 vs +0.098, P = 0.016). That is
  the ruler's second comparison (see *Open risks*).
- **Δ1 vs `src53k` is not a clean A/B on the loss**: the pool (rebuilt `s1w`),
  row count (356 vs 434), weights (uniform vs ×2), layout (vertical
  one-column ≥ 28 px) and lr (2e-3 vs 1e-3) all moved with it. The control is
  `--pair_loss 0` on `data_synth_d1` (`src/cli/data.py` names it) — no
  separate arm: a plain-FM run on Δ1's data *is* that control.

**K0(a)** — the cos table is `reports/table_geometry_2026_09_18.md`
(`src/probe/table_geometry.py`): **the shared direction is per-loss** —
ΔFM ↔ ΔFM 0.66–0.87 across datasets and sizes, plain ↔ plain 0.59–0.69,
cross-loss 0.27–0.51; Δ1 ↔ punct-only **0.043**. **K0(b) is parked**: Δ1
already trains the 13 punct rows, so "Δ1 ⊕ punct-only" is a cross-loss
override at cos 0.04, not the same-loss chunking K1 would use; the override
table exists (`rows_synth_d1_merge_punct`) and its eval was cancelled (user,
2026-09-18). A same-loss K0(b) needs a ΔFM punct-only block (≈ 3 k steps).

### box weight under ΔFM — settled 2026-09-18, keep `--box_weight 4`

Measured on the s750 smoke (`reports/row_blocks_alpha_2026_09_18.md`): w = 1
weakens the glyph (singles 13 → 9, native `en` 22 → 15) and changes nothing
on the scene. Every arm below keeps 4.

## S2 — the sentence step

The premise was [`plan_synth2.md`](plan_synth2.md) Δ2's: on a single-glyph
native the base free-runs a sentence's worth of JA text and the table addresses
one glyph of it, so the rest comes out as pseudo-text; plain FM hides that by
wiping the scene, and a caption that is *entirely* trained rows would leave
nothing unaddressed to invent. **S2a measured it and it does not hold** at
this exposure: the caption was fully addressed and the base still free-ran
around the glyph.

### S2-smoke — read 2026-09-18, `reports/synth_s2_smoke_2026_09_18.md`

The paired loss works mechanically on sentences (86 % of the residual
cancels, the 2-column siblings included) and the cold A/B could not separate
the arms (lift +0.030 [−0.010, +0.070]; native `en` 31 vs 29, co-text
22 = 22). Two findings from it outlive the arm:

- **The katakana dakuten rows render at floor** (untrained ガ 15/16, ゴ 14/16
  native; 10/12 flat), so every Δ0 `single` and native number was part floor
  — a dakuten probe uses hiragana, or renders the floor
  (`--native_floor 1`) and subtracts it.
- **ΔFM damages floor-positive rows and plain FM does not** (Δ0 paired arms
  5–7/12 katakana vs floor 10/12, plain 11/12) — the geometry read as a
  behaviour: plain rows grow along the pack row, ΔFM rows rotate away from
  it. A K1 table under ΔFM needs a pack-row anchor or an exclusion of
  floor-positive rows.

### S2a — the A/B, read 2026-09-18, `reports/synth_s2a_2026_09_18.md`

Two arms on `data_synth_s2a` (2 000 items, 446 rows, `--pair_ref en`), both
warm from Δ1, μ 0.3, 1 500 steps, lr 1e-3
(`rows_synth_s2a_s2a_{plain,pair}`). Sub-exact pooled plain **+0.131** vs
pair **+0.081** (pair − plain −0.050, CI [−0.123, +0.022], P 0.09), every
group plain; exact 1/16 vs 0/16; native `en` (はい …, 6 strings) 5 vs 2 / 48,
sub-exact tie. The sheets: pair holds the scene (en cos 0.931 vs 0.922) and
drops はい into pseudo-JA at subtitle size; plain writes one large はい. The
gate's kill clause fired — **ΔFM does not reach sentences; S2b runs plain**,
and the row goes to `findings.md` *What does not move it*. The one unpursued
caveat (the anchor is ≈ 6× heavier on the pair arm at equal μ; the pair arm
ran at lr 1e-3, not 2e-3) is in the report.

What S2a leaves standing for S2b:

- **Warm-start row units are converted** — `_init_rows_one` rescales by
  `src_row_scale / row_scale` (test `test_init_rows_converts_row_scale`).
  Before the fix a Δ1 row (row_scale 232.9) started 1.18× too large in a
  ≈ 197 inventory and `--init_anchor` pinned it there. Warm starts before
  2026-09-18 all stayed inside one inventory family (ratio 0.996–1.001), so
  no recorded result moves.
- **A warm start across losses is the right size and still the source
  loss's direction** — Δ1's rows have cos 0.14 to `src53k`'s on the same ids
  (`table_geometry`).
- **Paired siblings for `short` / `sentence` are built** — every kind goes
  through one composite loop and `render_into_scene` takes the item's own
  line lengths for the sibling (`common/render/scene.py` `ref_lines`),
  asserting pixel-identity outside the union box.
- **≈ 24 draws/row at 3 k steps over ~500 rows measures transfer onto an
  already-trained table, not identity** — a flat micro result does not price
  a 24 k run.

### S2b — the run, round 1 (launched 2026-09-18 night: 6 k, plain, μ 0.3)

**The warm start's job is sentence composition on rows that are already
trained — no cold vocab inside a sentence pass** (user, 2026-09-18). That
fixes the seed and the inventory together, and replaces the recipe this
section carried until the evening (`merge_punct` seed, `words:100/held=8`,
`list:はい,こんにちは`, 24 k):

- **Seed Δ1 (`rows_synth_d1_d1_s53k`), loss plain.** This is S2a's plain arm
  — the one measured configuration on the rejudged pools (+0.131, native
  はい 5/8). The cross-loss objection (Δ1 ↔ plain 0.14 per row) was answered
  by that arm winning; the katakana-damage objection is a Δ0 smoke finding,
  not measured on Δ1, and `--native_floor 1` reads it here.
- **Inventory = Δ1's units only** (`kana`, `kana_ext*1`, `kanji:200*1`, the
  punct `list:`): no `words:`, no `--phrase_pieces`, no こんにちは. In S2a
  those were 143 of 446 rows starting cold with no anchor, so S2a's +0.131
  is not a pure composition number. `はい` is は + い (two pieces, both Δ1
  rows), not a `list:` unit.
- **The 15 small-kana rows** (`っ ッ ゃ ょ ィ ぁ ェ ゅ ォ ぅ ァ ぇ ぃ ぉ ャ`)
  cannot be drawn as singles, so Δ1 never had them, and they sit in 2 693 of
  the 8 998 multi-glyph items (っ 1 630). They warm-start from
  `rows_synth_sent_q_sent_s24k_a1_s05` (plain FM, sentence-trained, has all
  15): `--init_rows <sent_s24k_a1_s05>,<Δ1>` — the later table overrides by
  ext id, so every Δ1 row is Δ1's.
- **Data `data_synth_s2b`** (rebuilt 22:00, no `--pair_ref`): 9 998 items,
  single 1 000 / short 5 000 / sentence 3 998, rejudged pools. Without piece
  rows the covered pool is small — 4 564 covered lines, **475 distinct
  sentences, 454 distinct shorts** (S2a's inventory: 6 142 sentence lines);
  365 rows drawn, median 42 occurrences per row in the 10 k items.
- **6 k steps, not 24 k** — sized to that pool (2.4 epochs, ≈ 100 draws on
  the median row; 24 k would show each line 80–100 times). `--lr_warmup 500
  --init_anchor 0.3`, lr 1e-3 cosine, σ 0.5–0.9, box weight 4, batch 4. μ 0.3
  is checked: the μ 0.1 arm on S2a's data and seed lost lift (+0.086 vs
  +0.131, native はい 1 vs 5/8; addendum of
  `reports/synth_s2a_2026_09_18.md`).
- **Rulers:** `single,single_ext,single_kanji,short,short_held,phrase,
  phrase_held,en` (no word groups — no word rows); native `あ,か,す,日,はい`
  on `en,swap`, katakana only with `--native_floor 1`. Sub-exact pooled lift
  is the primary number. こんにちは has no row this round and is not gated.

Job `20260918-222047-984df3`, arm `rows_synth_s2b_s2b_plain_d1_6k` — **read
2026-09-19**: singles 12 / 17 / 18 (= Δ1's), pooled lift +0.089 with the lift
on `short` only (held groups +0.03–0.05), native はい 1/8 on S2a's prompts;
the small pool memorises its shorts and does not generalise. Not
step-comparable to `sent2_s24k` (+0.161) or S2a (+0.131): different
inventory, pool and step count — the round-over-round numbers of the loop
below are the comparison this line makes from here.

Owed before the native read: pin the target stage's prompt frame to the
clause the card claims — `Japanese text reads as "…"` (`deploy_plan.md`
*What gets baked*) — so the gate is read on the clause that ships.

### The loop — vocab step, merge, sentence step (user, 2026-09-18)

S2b is round 1. After it the line runs a three-part cycle, twice:

1. **Vocab step.** Train the rows the sentence pool needs and the table does
   not have — the cold set S2b left out (the `words:` pieces, the phrase
   file's frequent pieces, こんにちは) — as a step-1 table (the seed recipe,
   `plan_synth4.md`), not inside a sentence pass.
2. **Merge** those rows into the round's table by ext id (`--init_rows a,b`
   or `merge_tables.py`; `row_scale` is converted).
3. **Sentence step.** Warm-start the merged table and train sentences again
   on the pool the new rows open up (S2a's inventory covers 11 119 lines /
   6 142 sentences against this round's 4 564 / 475), every row warm.

Then the same cycle once more. Each sentence step is sized to its own pool
the way S2b is and read on the same rulers; the word groups come back with
the word rows. Open for the vocab step: which loss trains a multi-glyph
piece as a unit (Δ1's ΔFM is a singles recipe, and the smoke's floor finding
applies to any row the pack already renders), and whether a merged table's
two shared directions (`table_geometry`: ΔFM ↔ plain 0.14) cost the sentence
step anything — round 1's small-kana rows are the first read of that.

## K — the kanji budget

Moved to [`plan_synth4.md`](plan_synth4.md) with the step-1 recipe decisions
(lr, the area-coupled box weight, glyph size, the row-norm lever α). K0 is
parked there.

## Order

R, S2-smoke, S2a and S2b are read (S2b:
`reports/s2b_and_raw_pack_rerun_2026_09_19.md`). Next is the pack question
at the top of this file, then the loop's vocab
step, merge and second sentence step, then the cycle once more. K1
(`plan_synth4.md`) shares the vocab step's recipe and is ordered there. One deployment fact stands from R: under the EN frame
(`swap`) Δ1 beats `src53k` on every native column (19 / 15 / 4 vs
18 / 12 / 8); under the JA frame it loses.

## Open risks

- **The sub-exact ruler is new and has two comparisons behind it** (`sent`
  vs `sent2`, and `src53k` vs Δ1 — the latter *disagrees* with exact match by
  construction: fewer exact hits, more of the right glyphs). It is a
  bag-of-glyphs measure: it cannot see order or count, the two things the
  strings arm showed a table *can* carry. It orders arms; it does not say the
  output is readable. Sheets stay the second read, per-glyph — a pooled lift
  that rises while the sheets show the same garbage with more JA glyphs in it
  is a reward-hacked ruler, and the check is the `_held` groups plus the
  first-glyph rate (`sent2` 0.219 vs 0.156 — moving, weakly).
- **The `sent2` result is one run, unreplicated, and confounded**: it changed
  the anchor (none), the pools (`sl1w,ja_comic` vs `sl1w`) and the warm source
  (a sentence table, not the 53 k) in one step. The μ arms are what separate
  the anchor from the rest; the pool change is not reproduced by design (the
  rejudged pools supersede both).
- **Singles and sentences may be one budget, not two.** Every sentence pass so
  far has cost singles, and the anchor recovers singles by pinning `f` — i.e.
  by refusing the update the sentence step is asking for. If lift and
  `single` move in opposite directions at every μ, the artefact needs two
  tables, not one, and the shipping question (`deploy_plan.md`) changes shape.
- **Δ1's flat 0 on kanji** — read (R): 日 holds on `en` and is not drawn as
  Latin. The lever, if a later inventory drifts, is still small paired flat
  glyphs, not 10 % at 110–200 px (`plan_synth2.md` Δ1).
- **RAM** (46 GB usable): captions are pool-bounded, but the loop's later
  sentence steps add word and phrase-piece rows on top of S2b's inventory,
  and K1 at 755 rows raises the text cache. The 10 k-item build stays the cap.

## Not this plan

- **A bigger `--n_items` build.** Draws per row, not distinct items, is the
  measured budget; 10 k items is the RAM cap and 21 epochs over them is not
  where the line is losing.
- **Reviving the encoder / composition / transplant shortcuts for kanji.**
  Closed (`findings.md` *Do not re-propose*); K is an exposure budget, which is
  why it is a wall-clock table here and not a research phase.
- **A kana reference, contrastive ΔFM, cached Jacobians, OCR-reward rows.**
  Unchanged from `plan_synth2.md` *Not this plan*.
- **lr 3e-3**, and **lr as a plain-FM lever** (L0, closed 2026-09-17).
- **The seed-table recipe and the kanji budget** — `plan_synth4.md` (lr,
  box weight / glyph size, α, K0 / K1, jōyō).
