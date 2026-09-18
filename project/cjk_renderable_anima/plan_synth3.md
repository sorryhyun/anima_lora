# plan_synth3 — after Δ1: the sentence step and the kanji budget

> **Status 2026-09-18 morning.** Δ1 (`20260918-015823-5a6c3f`, `d1_s53k`) is at
> ~40 k / 53 k, train ends ≈ 11:06, job ≈ 11:30. This plan is what runs after
> it. Two branches, from the user's note of 2026-09-18: **S2** — the sentence
> step (step 2), *including whether the pair loss is effective there at all* —
> and **K** — kanji scaling, at more draws per row than a kana row gets.
> Order and why: R (free) → S2a (micro A/B, ≈ 3 h) → branch. One GPU line at
> a time, as in [`plan_synth2.md`](plan_synth2.md).
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
| `src53k` (`full_s53k_qoff`) | 53 k plain FM, 434 rows, 490 draws/row | 13/36 | 20/36 | 18/36 | 0/32 | — | 24/24 |
| `sent_s24k_a1_s05` | + sentence pass, warm, `--lr_warmup 500 --init_anchor 0.3` | 11/36 | 19/36 | 19/36 | 0/32 | **0/32** each | 24/24 |
| `sent2_s24k` | a *second* sentence pass warm-started off that | 0/36 | 0/36 | 0/36 | 0/32 | **0/32** each | 24/24 |
| Δ1 `d1_s53k` | ΔFM, 355 rows, 597 draws/row, uniform weights | *running* | | | | (no multi-glyph groups) | |

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

## R — read Δ1 (free, today)

No new GPU beyond the eval already inside the job.

- **Gate (from `plan_synth2.md`):** beat `src53k` on singles (13/36) and on the
  joint native count with the tail under plain FM's.
- **The uniform-exposure read, which only Δ1 can give.** The 53 k run drew
  kanji and `kana_ext` at weight 2 against kana's 1 — its kanji 18/36 vs kana
  13/36 is *2× the exposure*, not evidence about strokes. Δ1 forces every
  source to `*1`, so **Δ1's `single` vs `single_ext` vs `single_kanji` vs
  `single_extra` spread at 597 draws/row is the first clean measurement of
  per-draw difficulty by unit type.** It sets K's weights; do not pick them
  before this number exists.
- Sheets to read: the first kanji rows for the flat-0 drift risk
  (`plan_synth2.md` Δ1, last bullet — 日 as Latin "a" was a plain-FM flat-0
  failure and Δ0's 12 rows were all kana), and `single_extra` (punctuation at
  weight 1 for the first time).
- Also free: `sub_exact.py` on Δ1 is meaningless (no multi-glyph groups) —
  Δ1's contribution to S2 is the seed table, not a ruler.

## S2 — the sentence step, and whether ΔFM is effective there

The premise stays [`plan_synth2.md`](plan_synth2.md) Δ2's: on a single-glyph
native the base free-runs a sentence's worth of JA text and the table addresses
one glyph of it, so the rest comes out as pseudo-text; plain FM hides that by
wiping the scene. A caption that is *entirely* trained rows leaves nothing
unaddressed to invent — if that holds, ΔFM's one measured cost disappears
where the target lives and its scene-holding stays.

What is new here is that the question is now askable: there is a ruler, and
there is a prior (both existing sentence tables carry lift; the anchor trades
singles against it).

### S2a — the A/B, micro (≈ 3 h, first)

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
  and `single` (secondary), `en` 24/24 (invariant). Native `がガゴ` for the
  co-text count.
- **Gate — ΔFM is effective on sentences iff** `s2_pair_a03` pooled lift is
  above `s2_plain_a03`'s bootstrap CI **and** `single` is not below it. The
  mechanism claim predicts something sharper and should be checked separately:
  the co-text count on the native sheets falls for the paired arm where the
  caption is fully addressed, which is the only reason this line survived Δ0.
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

**What "more steps per row" costs.** Δ1's rate is 597 draws/row at 355 rows /
53 k steps / batch 4; the exposure curve (`plan_synth.md`) reads 1 330 / 670 /
490 draws → 100 / 75 / 36 % of singles.

| kanji rows | total rows | at 597 draws/row | at 1 194 (×2 kanji) |
|---|---|---|---|
| 200 (Δ1) | 355 | 53 k steps, 8.7 h | 83 k, 13.6 h |
| 400 | 555 | 83 k, 13.6 h | 143 k, 23.4 h |
| 600 | 755 | 113 k, 18.5 h | 203 k, 33.4 h |
| 1 000 | 1 155 | 172 k, 28.4 h | 322 k, 53 h |

**The inventory ceiling is real and close.** `kanji:N` is corpus frequency over
the manga109s bubbles, and there are only **1 037** distinct kanji that are one
Qwen piece with a pack row: top-200 covers 68.3 % of corpus kanji tokens,
top-400 84.3 %, top-600 92.6 %, top-1000 99.5 %. The *pack* holds **8 501**
single-kanji rows, so jōyō 2 136 is addressable — but not through `kanji:N`;
it needs a `jouyou` unit kind or a `list:` file. Decide which target the line
is scaling to before sizing a run: **corpus 600 (92.6 % coverage, 18.5 h)** is
the cheap complete-looking point; jōyō is a different piece of work.

### K0 — is a merged table a table? (free, no GPU)

The budget above wants the shortcut `plan_synth2.md` parked in *Chunking*:
train disjoint row blocks and union them by ext id (`src/probe/merge_tables.py`
already does this with the per-run `row_scale` correction; the shipped
`merge_punct` table is exactly this). That would make kanji scaling parallel in
wall-clock instead of linear. **It is not free**, and the price is now
measured — cosine between run's mean row directions (the shared trigger,
17–29 % of row energy):

| pair | cos |
|---|---|
| 53 k ↔ punct-only (the shipped `merge_punct`) | **0.590** |
| 53 k ↔ plain-FM Δ0 (`pair0_s3000`) | 0.686 |
| 53 k ↔ **ΔFM** Δ0 (`pairEN_s1500_lr2e-3`) | **0.348** |
| plain-FM Δ0 ↔ ΔFM Δ0 (same rows, same data) | 0.449 |
| 53 k ↔ `sent_s24k_a1_s05` (warm off it, μ 0.3) | 0.999 |

So the shared direction is **per-run, not per-line**, and the loss rotates it
more than the dataset does (0.449 on identical rows and data). The shipped
merge worked at 0.59 for 17 punctuation rows; nothing says a 400-row kanji
block merged at 0.35 into a ΔFM kana block does. Caveat: the Δ0 means are over
12–17 rows and are noisy estimators; the 53 k ↔ punct row is the load-bearing
one.

K0 is therefore: (a) the cos table above extended to Δ1's table once it lands,
(b) `merge_tables.py` Δ1 ⊕ punct-only and eval the merged table on both blocks
— if a 0.35–0.59 merge costs the donor block its singles, the shortcut is dead
and K is one long run; if it does not, K1 is two or three parallel-in-time
runs. Both are eval-only.

### K1 — the scaled table

Recipe = Δ1's, with weights set by R's per-type spread and `kanji:400` or
`:600`. **Do not raise the kanji weight above what R measures**: the 53 k run's
kanji 18/36 at weight 2 was read as "kanji is fine" and it was an exposure
artefact; if Δ1 shows kanji at parity with kana at weight 1, the extra draws
should go to `kana_ext`/katakana instead (1/12 at equal exposure in the 53 k
run, the standing miss).

- **Gate:** `single_kanji` on the *new* rows (frequency ranks 200–600, which
  are rarer and were never evaluated) not below Δ1's on ranks 1–200, at equal
  draws/row. Held-out kanji stay 0 by construction (addresses do not compose,
  `findings.md`) — do not read that as a failure.
- **Guard:** rows in ≈ 0.35 % of batches at 755 rows (Δ1: 1.1 %, Δ0: 8 %).
  AdamW β₂ 0.99 decays `v` between visits; `delta_norm_mean` per draw against
  Δ1's curve over the first few thousand steps is the early read, as in Δ1.
- **Guard:** the 53 k run's word rows sat at norm 0.12 because a weighted draw
  is not a quota. At 755 rows check the items-per-row histogram in the data
  log before training, not after.

## Order

R and K0 are free and run today, off Δ1's output. Then **S2a**, because it is
the branch that can close: every multi-glyph group has been 0 since 2026-09-14,
the ΔFM line exists to fix exactly that, and a 3 h A/B either gives the line a
result or ends it. K1 is 14–19 h of GPU that produces a bigger *singles* table;
if S2 dies, a bigger singles table is the whole ceiling of the line and its
size should be a shipping decision, not a research one. If S2a passes, S2b and
K1 compete for the box and S2b goes first (it is what the artefact is for).

## Open risks

- **The sub-exact ruler is new and has one comparison behind it.** It is a
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
- **Δ1's flat 0 on kanji** is unverified — the flat-0 drift failures (日 as
  Latin "a") were plain-FM and Δ0's rows were all kana. R's sheets are the
  first look; the lever if it drifts is small paired flat glyphs, not 10 %
  at 110–200 px (`plan_synth2.md` Δ1).
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
- **Jōyō 2 136 in one run.** 8 501 single-kanji rows exist in the pack, so it
  is reachable, but at Δ1's rate it is ~60 h and it needs a unit kind that does
  not exist. Revisit only after K1 prices the 400–600 band.
