# plan_synth3 — after Δ1: the sentence step

> **Status 2026-09-18 afternoon.** Δ1 (`20260918-015823-5a6c3f`, `d1_s53k`)
> landed and is read — `reports/synth_pair_delta1_2026_09_18.md`: gate missed
> on singles (12/36) and native joint (29/128), passed on the tail (15/128);
> the whole joint loss is the `en` clause (JA frame), `swap` (EN frame) gains.
> This plan is **S2** — the sentence step (step 2), *including whether the
> pair loss is effective there at all*. Step 1 (the seed-table recipe and the
> kanji budget K) moved to [`plan_synth4.md`](plan_synth4.md) on 2026-09-18
> evening after the Δ0 smokes changed its recipe
> ([`reports/row_blocks_alpha_2026_09_18.md`](reports/row_blocks_alpha_2026_09_18.md)).
> Order: R (free, mostly done) → **S2-smoke** (≈ 1 h, cold 2-arm) → S2a
> (micro A/B, ≈ 3 h, trimmed by the smoke) → S2b. One GPU line at a time, as
> in [`plan_synth2.md`](plan_synth2.md).
> **No paired arm has ever trained on a multi-glyph item** — Δ0's 2 000 items
> and Δ1's 10 000 are all `single`. Whether ΔFM reaches sentences is untested,
> not just unmeasured, which is what the smoke is for.
> Two measurements made while writing this plan changed its shape:
> **(1)** exact match is floor-saturated on every multi-glyph group, and under a
> sub-exact ruler the run that was recorded as a collapse (`sent2_s24k`) is the
> **best** sentence table we have; **(2)** the shared trigger direction is
> **not** shared across runs (cos 0.35–0.69), which prices the merge-by-ext-id
> shortcut that the kanji budget wants to lean on.

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
table, and it is the failure S2's premise claims to fix.

`sent_s24k_a1_s05`'s report has been owed since 2026-09-17; the table above
and the ruler below discharge it — the full record goes to
`reports/synth_sentence_step2_2026_09_18.md` with Δ1's read.

## The finding this plan is built on

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
   evidence, spends sentence lift. **μ is the sentence step's main lever and
   has never been swept against a sentence ruler** — the 2026-09-17 sweep read
   singles only.
3. **Sentence work must be gated on lift, with exact match as a secondary.**
   A gate written as "short ≥ 1/32" is a coin flip; "pooled lift above the
   baseline's CI" is a measurement.

## R — read Δ1 (free) — done 2026-09-18, two items left

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
  `--pair_loss 0` on `data_synth_d1` (`src/cli/data.py` names it). It is not a
  separate arm: if S2 kills ΔFM, K1 runs plain FM on Δ1's data and *is* that
  control; if S2 keeps it, the sheets already assign the `en` loss to the
  loss.

Still owed, both free:

- **`single_extra`** — the 13 punctuation rows are in the inventory and
  rendered, but the group is missing from `EVAL_GROUPS` (`src/common/prompts.py`),
  so neither the table nor the sheet is written. One line + re-run the eval
  stage on `rows_synth_d1_d1_s53k`.
- **K0** — cos table extended to Δ1's table, and Δ1 ⊕ punct-only merged and
  evaluated on both blocks.

### box weight under ΔFM — settled 2026-09-18, keep `--box_weight 4`

Measured on the s750 smoke (`reports/row_blocks_alpha_2026_09_18.md`): w = 1
weakens the glyph (singles 13 → 9, native `en` 22 → 15) and changes nothing
on the scene. Every arm below keeps 4.

## S2 — the sentence step, and whether ΔFM is effective there

The premise stays [`plan_synth2.md`](plan_synth2.md) Δ2's: on a single-glyph
native the base free-runs a sentence's worth of JA text and the table addresses
one glyph of it, so the rest comes out as pseudo-text; plain FM hides that by
wiping the scene. A caption that is *entirely* trained rows leaves nothing
unaddressed to invent — if that holds, ΔFM's one measured cost disappears
where the target lives and its scene-holding stays.

What is new here is that the question is now askable: there is a ruler, and
there is a prior (both existing sentence tables carry lift; the anchor trades
singles against it). And Δ1 made the premise concrete: the `en`-clause loss
*is* co-text around the glyph (R above), so "fully addressed caption → no
co-text" is now a prediction with a number behind it, not a story.

### S2-smoke — does the pair loss reach a multi-glyph item at all (≈ 1 h, first)

Every paired arm to date trained on singles only. Before S2a spends 3 h and a
Δ1 warm-start on four arms, a Δ0-sized cold A/B on sentence items:

- **Data** (one build): Δ0's `chars:がぎぐげござガギグゲゴザ` plus a small word
  inventory (`--units words:30`), `--scene_mix short=0.5,sentence=0.5
  --phrase_file dialogue_2_10.tsv`, `--pair_ref en --pair_ref_pool 4`,
  `--shapes 512`. **Read `sheet_scene_pair.png` first** — multi-column and
  multi-line siblings have only ever passed the pixel-identity assert.
- **Train**, two arms on that dir, cold, 1 500 steps, the Δ0 argv
  (`--box_weight 4`, `--t_min 0.7 --t_max 0.9`, `--free_residual 1e-3`):
  `--pair_loss 0` vs `--pair_loss 1 --lr_rows 2e-3`.
- **Read:** (a) how much of the sentence residual cancels — paired loss vs
  `fm_plain` (Δ0 singles: ≈ 80 %); if the cancelled share collapses on
  multi-line items the sibling is not doing its job; (b) `sub_exact.py`
  pooled lift over `short` / `phrase` / `line`, plain vs paired, bootstrap CI;
  (c) native `がガ`, **`en` clause**, co-text count (renders with more than
  one text box) — the clause Δ1 lost.
- **Decision:** paired lift inside plain's CI *and* co-text not below plain →
  S2a drops its paired arms and becomes a μ sweep (2 arms, plain). Paired lift
  above plain's CI → S2a keeps the paired arms and may drop the plain μ 0 arm.
  Either way S2a shrinks.
- **Limit:** cold 1 500 steps at ~60 rows is under the identity budget; this
  answers "does the paired residual point somewhere on sentences, relative to
  plain", which is what Δ0 answered for singles — not "are sentences readable".

### S2a — the A/B, micro (≈ 3 h, after the smoke)

**Pre-condition (found 2026-09-18, not yet fixed): the warm-start loader
does not convert row units.** `raw` is in row-norm units (delta = `raw ×
row_scale`) and `row_scale` is the mean pack-row norm of *that run's*
inventory (`src/train/trainables.py:30`); `_init_rows_one` (`:103`) copies
the source `raw` as is. Every warm start so far stayed inside the 53 k
inventory family (`src53k` 196.4 → `sent_s24k` 197.1 → `sent2` 197.0, a 0.4 %
error — the anchor-sweep and `sent2` reads stand). S2a warm-starts **Δ1
(row_scale 232.9, 356 rows) into a ~500-row inventory (≈ 197)**: uncorrected,
every Δ1 row starts 1.18× too large and `--init_anchor` pins it there — and
norm is the native lever (`plan_synth4.md`). Fix before launch: `row = row *
src_row_scale / self.row_scale` in the loader (source without `row_scale` →
1.0), the same correction `src/probe/merge_tables.py` already applies;
one unit test on a two-inventory fixture; CLI golden untouched.


Four arms on **one** data dir, 3 k steps each, warm-started from Δ1's table,
everything else the `sent_s24k_a1_s05` argv:

| arm | loss | anchor μ | asks |
|---|---|---|---|
| `s2_plain_a03` | `--pair_loss 0` | 0.3 | the baseline, same data, same steps |
| `s2_pair_a03` | `--pair_loss 1 --lr_rows 2e-3` | 0.3 | **is ΔFM effective on sentences** |
| `s2_plain_a0` | `--pair_loss 0` | 0 | reproduce the `sent2` trade at 3 k |
| `s2_pair_a0` | `--pair_loss 1 --lr_rows 2e-3` | 0 | the two levers together |

- **Data.** The `sent_s24k_a1_s05` recipe (`--scene_mix single=0.1,short=0.5,
  sentence=0.4`, `--phrase_file dialogue_2_10.tsv`) over the **rejudged** pools
  (s1 233 / s1w 380 / sl1w 213 / ja_comic 292 — Δ0.9 shrank and cleaned them
  after that run was built), `--pair_ref en --pair_ref_pool 4`. Word rows are
  needed and Δ1 has none (`--scene_mix single=1.0`): add
  `--units words:100/held=8` to the data stage, plus `--units list:はい,
  こんにちは` (the anchor sweep's two target strings — こんにちは is one Qwen
  piece with no row at all, so it is untrainable without a `list:` unit).
- **Paired siblings for `short` / `sentence` are already built** — the Δ0.9
  rework put every kind through one composite loop and `render_into_scene`
  takes the item's own line lengths for the sibling (`common/render/scene.py`
  `ref_lines`), asserting pixel-identity outside the union box. plan_synth2's
  "code owed in `data/synth.py`" is stale. **Read `sheet_scene_pair.png` for
  multi-column items before launch** — that assert is the only thing checked
  so far.
- **Ruler:** `sub_exact.py` pooled lift (primary), per-group lift, exact match
  and `single` (secondary), `en` 24/24 (invariant). Native `がガゴ` on the
  **`en` clause** (the JA frame — the trained and deployed frame, and the one
  Δ1 lost) for the co-text count; `swap` is the control clause.
- **Gate — ΔFM is effective on sentences iff** `s2_pair_a03` pooled lift is
  above `s2_plain_a03`'s bootstrap CI **and** `single` is not below it **and**
  the `en`-clause co-text count is below the plain arm's. The third clause is
  the mechanism itself: Δ1 showed ΔFM's cost is co-text around the glyph, and
  a fully addressed caption is the only proposed fix. Lift without a co-text
  drop means the ruler moved and the failure did not.
- **Kill:** paired lift inside plain's CI at equal `single` → ΔFM does not
  reach sentences; it stays a flag, S2b runs plain, and the row goes to
  `findings.md` *What does not move it*. This is the decision the whole ΔFM
  line has been heading toward since Δ0 — do not soften it.
- **Guard:** the `--init_anchor` recipe was tuned on a **plain-FM** source, and
  Δ1 is ΔFM. Their shared directions differ (below), so `warm_cos` on a paired
  arm warm-started from Δ1 is not comparable to the sweep's numbers; read
  `warm_cos` per arm, not against 2026-09-17.
- **Guard:** 3 k steps over ~500 rows is ≈ 24 draws/row — far under the
  identity budget. S2a measures *transfer onto an already-trained table*, not
  identity; a flat result at 3 k does not price S2b's 24 k. If every arm is
  flat, the next step is 8 k on the two surviving arms, not a verdict.

### S2b — the run (≈ 6 h, after S2a picks loss and μ)

The winning arm at 24 k on the same data. Gates against `sent2_s24k`'s pooled
lift (+0.161) with `single` ≥ 11/36, and the target stage (`はい` / `こんにちは`
at 768×1344, the user's ComfyUI captions) against the anchor sweep's 1/8 and
0/6.

## K — the kanji budget

Moved to [`plan_synth4.md`](plan_synth4.md) with the step-1 recipe decisions
(row blocks, per-block warmup, the row-norm lever). K0 stays free and is
listed there.

## Order

R is read; `single_extra` and K0 are free and run off Δ1's output. Then the
**S2-smoke** (≈ 1 h), then **S2a** trimmed to what the smoke leaves, because
S2 is the branch that can close: every multi-glyph group has been 0 since
2026-09-14, the ΔFM line exists to fix exactly that, and the smoke + A/B either
give the line a result or end it. One deployment fact decides whether Δ1 is
already a usable seed: under the EN frame (`swap`) Δ1 beats `src53k` on every
native column (19 / 15 / 4 vs 18 / 12 / 8); under the JA frame it loses. The
target stage's prompt frame (`deploy_plan.md`) has to be fixed before S2b so
the gate is read on the clause that ships. K1 (`plan_synth4.md`) waits on
this line's verdict: if S2 dies, a bigger singles table is the whole ceiling
of the line and its size is a shipping decision; if S2a passes, S2b goes
first (it is what the artefact is for).

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
  (a sentence table, not the 53 k) in one step. S2a's μ arms are what separate
  the anchor from the rest; the pool change is not reproduced by design (the
  rejudged pools supersede both).
- **Singles and sentences may be one budget, not two.** Every sentence pass so
  far has cost singles, and the anchor recovers singles by pinning `f` — i.e.
  by refusing the update the sentence step is asking for. If S2a shows lift and
  `single` moving in opposite directions at every μ, the artefact needs two
  tables, not one, and the shipping question (`deploy_plan.md`) changes shape.
- **Δ1's flat 0 on kanji** — read (R): 日 holds on `en` and is not drawn as
  Latin. The lever, if a later inventory drifts, is still small paired flat
  glyphs, not 10 % at 110–200 px (`plan_synth2.md` Δ1).
- **The S2-smoke is cold and small**, like Δ0 — and Δ0's 12 rows did not
  expose the `en`-clause loss that Δ1's 356 did. A smoke pass is a licence to
  run S2a, not a sentence verdict; a smoke kill is a kill only for the paired
  arms of S2a, and S2b's plain run still measures sentences.
- **RAM** (46 GB usable): reference latents + captions are pool-bounded, but
  S2a adds word and `list:` rows on top of the sentence recipe's phrase pieces,
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
- **The seed-table recipe and the kanji budget** — `plan_synth4.md` (row
  blocks, warmup, α, K0 / K1, jōyō).
