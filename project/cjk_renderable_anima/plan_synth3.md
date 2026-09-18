# plan_synth3 — after Δ1: the sentence step

> **Status 2026-09-18 evening.** This plan is **S2** — the sentence step
> (step 2). Step 1 (the seed-table recipe and the kanji budget K) is
> [`plan_synth4.md`](plan_synth4.md).
> R, S2-smoke and S2a are read; **S2b is the next launch and the only live
> item here**, its recipe below.
> **The loss is decided: plain.** S2a (`reports/synth_s2a_2026_09_18.md`)
> put plain above ΔFM on every sentence ruler that moves (sub-exact pooled
> +0.131 vs +0.081, native `en` 5 vs 2 / 48); ΔFM is killed on sentences
> (user decision). The ruler that decided it is `src/probe/sub_exact.py`:
> exact match is floor-saturated on every multi-glyph group.

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

### S2b — the run (≈ 6 h, plain, μ 0.3 — both decided 2026-09-18)

Plain FM at 24 k. Gates against `sent2_s24k`'s pooled lift (+0.161) with
`single` ≥ 11/36, and the target stage (`はい` / `こんにちは` at 768×1344, the
user's ComfyUI captions) against the anchor sweep's 1/8 and 0/6.

Recipe = the `sent_s24k_a1_s05` argv (`20260917-131126-da1e0a`: σ 0.5–0.9,
lr 1e-3 cosine, `--lr_warmup 500`, `--free_residual 1e-3 --box_weight 4`,
`--scene_mix single=0.1,short=0.5,sentence=0.4`, `words:100/held=8`, the
punct `list:`, `--n_items 10000`) with these changes, each one a measured
reason:

- **Seed `rows_synth_full_fm10k_merge_punct`, not Δ1.** Same loss as the
  run (plain ↔ plain shared direction 0.59–0.69; Δ1 ↔ plain 0.14 per row —
  `table_geometry`), and the smoke showed ΔFM tables damage the pack's
  floor-positive katakana rows, which an anchor to Δ1 would pin in.
- **Data without `--pair_ref`.** Halves the latent store and the build; no
  sibling is read by a plain arm.
- **Rejudged pools** (S2a's build), `--units list:はい,こんにちは` added (the
  target strings; こんにちは is one Qwen piece with no row).
- **`--init_anchor 0.3` — μ checked and kept.** The μ 0.1 arm on S2a's
  data and seed (`rows_synth_s2a_s2a_plain_a01`, read 2026-09-18 evening,
  addendum of `reports/synth_s2a_2026_09_18.md`) moved the rows twice as far
  and lost lift (+0.086 vs +0.131, native はい 1 vs 5/8). The no-anchor
  `sent2_s24k` lift (+0.161) is a 24 k-from-a-sentence-table result, not a
  μ result at this scale; if S2b's lift lands under the gate, μ is swept on
  S2b's own table then, not before.
- **Rulers:** `single,single_ext,single_kanji,word,word_held,short,
  short_held,phrase,phrase_held,en` (S2a had no `single` — the anchor's
  singles cost went unmeasured); native `あ,か,す,日` + `はい,こんにちは`
  on `en,swap`; katakana only with `--native_floor 1` (the smoke's floor
  finding). Sub-exact pooled lift is the primary number.
- **Not changed:** lr (1e-3 is plain's recipe of record; "lr is a lever"
  was an ΔFM finding), σ 0.5–0.9, box weight 4, batch 4, 24 k.

Owed before launch: pin the target stage's prompt frame to the clause the
card claims — `Japanese text reads as "…"` (`deploy_plan.md` *What gets
baked*) — so the gate is read on the clause that ships.

## K — the kanji budget

Moved to [`plan_synth4.md`](plan_synth4.md) with the step-1 recipe decisions
(row blocks, per-block warmup, the row-norm lever). K0 stays free and is
listed there.

## Order

R, S2-smoke and S2a are read. **S2b is next**, then K1
(`plan_synth4.md`). One deployment fact stands from R: under the EN frame
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
- **RAM** (46 GB usable): captions are pool-bounded, but S2b adds word and
  `list:` rows on top of the sentence recipe's phrase pieces, and K1 at 755
  rows raises the text cache. The 10 k-item build stays the cap.

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
