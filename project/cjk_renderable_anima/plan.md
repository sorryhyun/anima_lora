# plan — the live plan (step 1, step 2, the kanji budget)

> **Rewritten 2026-09-20.** This is the one forward plan for the line. The
> method as built is [`synth.md`](synth.md), settled verdicts are
> [`findings.md`](findings.md) and [`findings_seed.md`](findings_seed.md),
> the dated record is [`reports/`](reports/README.md), and publishing is
> [`deploy_plan.md`](deploy_plan.md). Only open items live here; everything
> closed has moved to `findings.md` or to the report that closed it.
>
> **Every launch states its pack.** `ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`
> is the raw pack (log sha `7b9fce0bb57b`); `configs/base.toml` still points at
> `anima_cjk_vocab_pack_preview` (= raw + `sent_s24k_a1_s05`'s 502-row delta,
> sha `5f52aefce82a`), which silently based every run between 2026-09-17 17:36
> and 2026-09-19 — see `reports/s2b_and_raw_pack_rerun_2026_09_19.md`.

## Where the old plan files went

The four `plan_synth*.md` files are archived under
`_archive/cjk_renderable_anima/`, kept verbatim for the arms they record.
Source docstrings that cite them resolve here:

| old pointer | now |
|---|---|
| `plan_synth.md` (budgets, pools, rulers, do-not-re-propose) | `synth.md` *Budget* / *Scene pools* / *Rulers*, `findings.md` |
| `plan_synth2.md` Δ0–Δ2 (the ΔFM loss as built) | `synth.md` *The paired loss (ΔFM)*; verdicts in `findings.md` |
| `plan_synth2.md` Δ0.9 (scene-judge rules) | `synth.md` *Instrument* item 11 |
| `plan_synth3.md` S2, the loop | *Step 2* below |
| `plan_synth4.md` R4.3 / R4.4 / R4.5 / R4.6 / K | *Step 1* and *K* below |

## The target artefact (unchanged since 2026-09-13)

A vocab pack: a static table over ext rows, loaded by the existing
`llm_adapter.embed` hook, shipped as safetensors + mapping json under the
same `vocab_pack` key (training, TE caching, `inference.py`,
`GenerationRequest`, the register node). Rows are Qwen pieces; a new pack
changes the digest → `make preprocess-te ARGS=--overwrite` for CJK
captions. Anything that leaves this form (DiT/adapter LoRA, runtime gate)
is a fallback, not a phase. Packaging and the pre-upload gates are
[`deploy_plan.md`](deploy_plan.md).

The line is built in two steps over that one table:

- **Step 1 — the seed table.** One ext-row delta per single-glyph unit
  (kana, `kana_ext`, small kana, kanji, punctuation), trained on scene
  composites with `--scene_mix single=1.0`. Recipe open, *Step 1* below.
- **Step 2 — the sentence pass.** The same table warm-started and trained
  on multi-glyph composites (`short` / `sentence`), plain FM. Loop open,
  *Step 2* below.

---

## Where the line stands (2026-09-20)

Raw pack, full-inventory tables. `exact (sfx)` off each arm's `report.md`,
seeds 0/1 pooled; native = both readers exact over あ か す 日 × `en,swap`.

| table | recipe | `single` / `_ext` / `_kanji` / `_small` | native `en` / `swap` both of 64 | en cos |
|---|---|---|---|---|
| `src53k` (`full_s53k_qoff`) | plain FM, 434 rows, 490 draws/row, `--box_weight 4` | 13 / 18 / 18 / — | 36 / 18 | 0.882 |
| `step1_0919` | ΔFM, 374 rows, 53 k, `--box_weight 4` | 10 / 5 / 3 / 0 | 8 / 4 | 0.931 |
| **`step1_0920`** | the same argv and data, **`--box_share 0.25`** | **20 / 8 / 8 / 0** | **19 / 8** | 0.934 |
| `step2_0919` | step 2 on `step1_0919`, plain, 6 k | 10 / 4 / 2 / 0 | はい 2 of 16 (sentence natives) | 0.931 |

**`step1_0920` is the current seed table** (2026-09-20, job
`20260920-003014-320a05`, arm `rows_step1_0920_s53k`; `data_step1_0920` is a
symlink to `data_step1_0919`). One variable against `step1_0919` — the
area-independent in-box loss — and every ruler about doubles with the scene
*better* held (en cos 0.931 → 0.934). It closes about a third of the native
gap to plain `src53k`, not all of it. Kanji is still the weak group
(8/36, 日 1/32 native), `single_small` is 0/36 on both, and the back-half
norm contraction is unchanged in proportion (250 → 160 against 149 → 76) —
the share raised the curve, it did not remove the pull.

**`step2_0919` is on record and not read out** (2026-09-19, jobs
`20260919-202821-2392df` data / `20260919-203858-051f61` train+eval). It is
step 2 on the *weak* `step1_0919` seed, so it prices the loop's mechanics,
not the artefact. Sub-exact pooled lift **+0.086** (n = 64) with the lift on
both trained and held groups — `short` +0.125, `short_held` **+0.141**,
`phrase` +0.031, `phrase_held` +0.049 — against S2b's +0.089 that sat on
`short` alone. Singles held at the seed's level (10 / 4 / 2 vs 10 / 5 / 3);
sentence natives 2 of 96 (はい 2/16, the other five strings 0). **Owed:** the
same step on `step1_0920`, which is the comparison the loop is for.

---

## Step 1 — the seed-table recipe

### The recipe as it stands

| knob | value | state |
|---|---|---|
| batching | mixed | row blocks closed on the raw pack (`findings.md`) |
| in-box weighting | **`--box_share 0.25`** | the decision of 2026-09-20; area-independent, replaces `--box_weight` |
| lr | **2e-3** under ΔFM (1e-3 plain) | what both full-table runs used, `step1_0920` included. **5e-3 is a candidate, not the value**: it read 13 vs 10/24 with the best native on raw Δ0 and no collapse at norm 144 (the 3e-3 off-manifold finding was plain FM), but that is 12 rows at 1 500 steps and it has never run at full inventory or at 53 k. Open, below |
| `--free_residual` μ | 1e-3 | sets the end norm together with the in-box share; calibrated to `d0`'s ≈ 64-cell box, so it is owed a re-read on `--box_share` |
| glyph size | half full-fit, half jittered [12 px, fit] (`d0mix`) | native 18 vs 11 of 128 with hits under 64 px, scene held; jitter alone reads 0/24 and native 4. Micro only; ≥ 2 seeds and a full-table arm before K1 |
| loss | ΔFM (`--pair_loss 1 --pair_ref en`) for singles | raw Δ0: ΔFM 10 vs plain 7 at 12 rows, scene held — **but at full inventory ΔFM is the weak loss** (`step1_0920` 20/36 vs plain `src53k` 13/36 on `single` and 19 vs 36 on native). Open, below |
| row norm | full in training, α at inference | R4.3, below |
| unit weights | uniform `*1` | kanji are not harder than kana per draw (Δ1, preview pack) — re-read on `step1_0920` |
| pack | raw | `ANIMA_VOCAB_PACK=…/anima_cjk_vocab_pack` on every job |

**Warm starts convert row units** — `raw` is in units of the run's own
`row_scale` and `_init_rows_one` rescales by `src_row_scale / row_scale`
(since 2026-09-18, test `test_init_rows_converts_row_scale`).

### S1a — which loss the seed table takes, at full inventory

The one unresolved recipe question, and it now has evidence on both sides:

- **ΔFM wins at 12 rows** (raw Δ0: 10–13 vs plain 7 of 24, scene held).
- **Plain wins at 374–434 rows** (`src53k` 13/18/18 and native 36 against
  `step1_0919`'s 10/5/3 and 8) — and `--box_share` lifts ΔFM to 20/8/8 and
  native 19 without closing the native gap. Plain at ρ_g 0.25 on the 12-row
  Δ0 data is the best raw native read of the line (26 of 64 `en`, up from
  13) at the cost of the scene (en cos 0.906 → 0.893).
- Same split at both scales: **plain is ahead on native, ΔFM on the scene.**

**The arm:** `step1_0920`'s argv and data with `--pair_loss 0 --lr_rows 1e-3`
(the plain control on the same build — no separate data dir needed). Rulers:
`single` / `_ext` / `_kanji` / `_small`, native `en` / `swap`, en cos.
Pass = the loss that wins native without giving up more than ≈ 0.01 en cos.
Until it runs, every "ΔFM vs plain" sentence in this line is a statement
about 12 rows or about two runs that also differed in lr.

**lr rides with it.** Every full-table arm has run ΔFM at 2e-3 and plain at
1e-3 — the same pairing as `src53k` vs `step1_0919`, so the loss and the lr
have never been separated at scale. The 12-row read that ΔFM at **5e-3** is
better (13 vs 10/24, best raw native at 20 both) is the one recipe knob
still decided entirely at micro scale, and a 53 k run at 5e-3 has never been
tried. If S1a's control keeps ΔFM, the second arm is `step1_0920`'s argv at
`--lr_rows 5e-3` before anything is frozen for K1; if it kills ΔFM, the
5e-3 question dies with it. Two seeds either way — the 12-row rerun floor is
cos ≈ 0.75 per row.

### S1b — glyph size

**Status.** The share fix is in and the small-glyph arms are re-read under
it (`--box_share` makes the row's weight per *glyph*, not per box area, so
the old jitter arms' 0/24 was mostly normalisation). What survives:

- **Small glyphs at σ 0.7–0.9 do not teach identity by themselves.**
  `d0sz` at an equal 25 % share is the worst small arm on both rulers
  (`single` 0/24, native 4) at a norm above `d0` w 4's — the rows travel,
  not toward identity (misreads are voiced-but-wrong: が → ず, ご → ど).
  A ≈ 30 px glyph is ≈ 4 × 4 latent cells and what survives the band is
  "a dakuten kana".
- **Mixing large and small in one build works.** `d0mix` (half `d0`'s
  items at 38/54/80 px, half `d0sz`'s at 17/30/50): native 18 of 128
  against 11 for both `d0` arms at the same lr, 7 of the 18 under 64 px
  (`d0` w 4: 1 of 11), scene *better* held (0.933 vs 0.921). Large items
  carry identity, small ones carry size.
- **Size arms are read on native, not `single`** — the `single` template
  asks for a large glyph and under-reads size-trained rows on every such
  arm (`d0mix` 5/24 with 18 native; w 12 5/24 with 9).
- **The free read** (no GPU): no native hit under 40 px on either full
  table (0 of 149 boxes), and the 24–40 px bin holds no single-glyph box at
  all — when the base lays out small text it writes its own multi-glyph
  pseudo-text. Most hits are *above* the training p95. The readers are not
  the limit (VAE / reader floor is 10 px kana, 16 px kanji; minimum glyph
  12 px, user 2026-09-18).

**Open — S1b.1, the small-bubble pool.** `--scene_size_jitter` shrinks the
glyph inside the bubble the scene already has, and no pool has a small
bubble to put it in (region short side p05/p50/p95 = 57/76/145 px on `s1`,
57/76/133 `s1w`, 59/83/133 `sl1w`, 42/62/99 `ja_comic`; **no region under
40 px in 1 118 scenes**). The glyph should be small because the *bubble* is
small, so the lever is the `scenes` prompt and the judge, not the
compositor:

1. **Pool smoke** (GPU, 512-class, ≈ the `s1w` run's cost per scene): a few
   hundred scenes per lever, read off `scenes.jsonl` region / letter sizes —
   (a) layout tags that shrink the bubble (`comic` / `4koma` / `multiple
   speech bubbles`, `full body` / `wide shot`, `chibi`); (b) anchors that
   shrink it (`!` `?` `…` `a` `I` — Latin / punctuation, no ext row);
   (c) longer EN anchors (a phrase), which the base letters smaller.
   Number to read: share of kept scenes with a region under 40 px and under
   24 px, and the judge's yield there.
2. **Fit to the anchor's own letter size**, not the largest font the region
   takes: the detector box gives the erased text's letter height and the
   swap draws at that size (± a small jitter).
3. If (1) finds nothing under ≈ 24 px, 12 px is not reachable in-domain on
   a 512-class canvas and the remaining route is the training canvas itself
   (target-shape `--shapes`), at its it/s cost.

**Arms:** 12-row `d0` recipe on `--box_share`, `d0` vs `d0s` (the
small-bubble pool mixed with `s1` / `s1w`). Rulers: `single` /24 as the
does-it-still-learn check, and native hits **by glyph-size bin**
(probe to be written as `src/probe/native_by_size.py`). Pass = hits appear
under 40 px without losing the ≥ 64 px bins; if `d0s` reads like `d0` in
every bin, size binding is not what holds native down and the lever drops.
Open inside it: a 12 px glyph is under one DiT token (16 px of canvas), so
whether σ 0.7–0.9 trains it at all is unresolved — the band floor 0.7 → 0.5
is the second arm only if `d0s` learns on `single` and still misses the
small bins.

**Not run, and why** (both fail the S line's own premise — the scene is the
base's own output under a caption that explains all of it, so the FM
residual outside the bubble is ≈ 0):

- *Scenes rendered at k × and downscaled* — cost ≈ k² in generation
  (1.5 × is already 2 ×) and 12 px from a 51 px fit is k ≈ 4.
- *n × n panel pages from the existing pools* — free and share-preserving,
  but a page of downscaled panels with gutters is not a base output and one
  panel's tags do not describe it: the off-manifold paste the S line
  replaced.

### S1c — α as a deployment knob (native only, 9 min per point)

The preview-pack α sweep read native hits monotone in row norm and scene
fidelity monotone the other way, crossing ≈ × 0.7. Re-read on raw tables:
`step1_0920` × 0.5 / × 0.7 (`--native_chars あ,か,す,日 --native_clauses
en,swap`) and the same on plain `src53k`. If `en` joint rises without the
tail climbing, α ships as the LoRA-multiplier slot; if neither table moves,
the curve was a 12-row preview-pack artefact and α is dropped. One
`--stage eval` at × 0.7 on the same table says whether the `single`
template is norm-hungry.

---

## Step 2 — the sentence pass and the vocab → merge → sentence loop

**The loss is decided: plain.** S2a put plain above ΔFM on every sentence
ruler that moves (sub-exact pooled +0.131 vs +0.081, native `en` 5 vs 2 of
48); ΔFM is killed on sentences (`findings.md` *What does not move it*).

**The ruler is `src/probe/sub_exact.py` pooled lift**, not exact match:
every multi-glyph group of every arm to date is 0/32, so exact match has no
resolution exactly where step 2 lives. Lift = glyph recall − the same
recall of the group's *other* refs against the same read, so read length and
the arm's general JA-glyph habits are held. Sheets stay the second read,
per-glyph — a pooled lift that rises while the sheets show the same garbage
with more JA glyphs in it is a reward-hacked ruler, and the check is the
`_held` groups plus the first-glyph rate.

**The warm start's job is sentence composition on rows that are already
trained — no cold vocab inside a sentence pass** (user, 2026-09-18). That
is what makes this a loop rather than one run:

1. **Vocab step.** Train the rows the sentence pool needs and the table does
   not have — the `words:` pieces, the phrase file's frequent pieces,
   こんにちは — as a *step-1* table on the step-1 recipe, not inside a
   sentence pass.
2. **Merge** those rows into the round's table by ext id (`--init_rows a,b`,
   the later table overriding, or `src/probe/merge_tables.py`; `row_scale`
   is converted).
3. **Sentence step.** Warm-start the merged table and train sentences on the
   pool the new rows open up, every row warm. Size the run to its own pool
   (`step2_0919`: 4 564 covered lines, 480 sentences, 470 shorts →
   6 k steps ≈ 2.4 epochs), read it on sub-exact pooled lift.

Then the same cycle once more.

**Round 2 — the owed run.** `step2_0919` ran this on the weak
`step1_0919` seed. The comparison the loop is for is the same step on
**`step1_0920`**: same data argv (`data_step2_0919`'s build), same train
argv, one warm source `rows_step1_0920_s53k/trained.pt`. Gate: pooled lift
above `step2_0919`'s +0.086 CI with the `_held` groups moving, and singles
not below the seed's 20 / 8 / 8.

**Open inside the loop:** which loss trains a multi-glyph piece as a unit
(ΔFM is a singles recipe, and ΔFM damages rows the pack already renders);
whether a merged table's two shared directions cost the sentence step
anything (same-loss blocks agree at cos 0.59–0.87, cross-loss 0.27–0.51 —
`reports/table_geometry_2026_09_18.md`).

**Owed before the next sentence native read:** pin the target stage's prompt
frame to the clause the card claims — `Japanese text reads as "…"`
(`deploy_plan.md` *What gets baked*) — so the gate is read on the clause
that ships.

---

## K — the kanji budget

**What "more steps per row" costs.** The exposure curve reads
1 330 / 670 / 490 draws/row → 100 / 75 / 36 % of singles.

| kanji rows | total rows | at ≈ 600 draws/row | at ≈ 1 200 (×2 kanji) |
|---|---|---|---|
| 200 (today) | 355–374 | 53 k steps, 8.7 h | 83 k, 13.6 h |
| 400 | 555 | 83 k, 13.6 h | 143 k, 23.4 h |
| 600 | 755 | 113 k, 18.5 h | 203 k, 33.4 h |
| 1 000 | 1 155 | 172 k, 28.4 h | 322 k, 53 h |

**The inventory ceiling is real and close.** `kanji:N` is corpus frequency
over the manga109s bubbles, and there are only **1 037** distinct kanji that
are one Qwen piece with a pack row: top-200 covers 68.3 % of corpus kanji
tokens, top-400 84.3 %, top-600 92.6 %, top-1000 99.5 %. The *pack* holds
**8 501** single-kanji rows, so jōyō 2 136 is addressable — but not through
`kanji:N`; it needs a `jouyou` unit kind or a `list:` file. Decide which
target the line is scaling to before sizing a run: **corpus 600 (92.6 %
coverage, 18.5 h)** is the cheap complete-looking point; jōyō is a different
piece of work.

### K0 — is a merged table a table? parked

Training disjoint row blocks and unioning them by ext id
(`src/probe/merge_tables.py`, per-run `row_scale` correction; the shipped
`merge_punct` table is this) would make kanji scaling parallel in
wall-clock. The price of a merge is the cosine between the runs' shared
directions: same-loss blocks agreed at 0.59–0.87. The test itself — a
punct-only block in the same loss as its base (≈ 3 k steps), merged and
evaluated — is not run; it decides whether K is one long run or two or
three parallel-in-time ones.

### K1 — the scaled table

Recipe = the step-1 table above (S1a's loss verdict, `--box_share 0.25`,
`d0mix` sizes) with `kanji:400` or `:600`. **Do not raise the kanji weight
above what the uniform-exposure read measures**: the 53 k run's kanji 18/36
at weight 2 was read as "kanji is fine" and it was an exposure artefact. At
uniform weight on the preview pack kanji were *not* harder than kana per
draw — but `step1_0920` reads kanji 8/36 against kana 20/36, so re-read the
per-type spread on the raw table before setting weights, and if kanji are
the weak kind the extra draws go there rather than to `kana_ext` /
katakana (the other standing miss, 8/36).

- **Gate:** `single_kanji` on the *new* rows (frequency ranks 200–600, rarer
  and never evaluated) not below `step1_0920`'s on ranks 1–200, at equal
  draws/row. Held-out kanji stay 0 by construction (addresses do not
  compose, `findings.md`) — do not read that as a failure.
- **Guard:** rows appear in ≈ 0.35 % of batches at 755 rows (374 rows: 1.1 %;
  12 rows: 8 %). AdamW β₂ 0.99 decays `v` between visits; `delta_norm_mean`
  per draw against `step1_0920`'s curve over the first few thousand steps is
  the early read.
- **Guard:** the 53 k run's word rows sat at norm 0.12 because a weighted
  draw is not a quota. At 755 rows check the items-per-row histogram in the
  data log **before** training, not after.

---

## Order

1. **S1a** — the plain control on `step1_0920`'s build, then (if ΔFM
   survives) the 5e-3 arm. It decides the loss *and* the lr for both K1 and
   every later vocab step, and nothing else should run first.
2. **Round 2 of the loop** — step 2 on `step1_0920` (independent of 1; can
   queue behind it).
3. **S1c** — α points on `step1_0920` and `src53k` (9 min each, CPU-cheap
   native re-renders).
4. **S1b.1** — the small-bubble pool smoke, anchor-size fit, `d0s` vs `d0`,
   native by size bin.
5. The recipe filled in on ≥ 2 seeds; **K1** after it.
6. **Publishing** — the Hub v2 layout and gates G1–G4 / G6 are unrun
   (`deploy_plan.md`).

## Open risks

- **Most recipe numbers are 12 rows, 24 singles, one seed.** Differences
  under ≈ 3 are noise and the 12-row rerun chaos floor is cos ≈ 0.75 per
  row, so **no schedule knob is decided at one seed**. `step1_0919` /
  `step1_0920` are the only full-table reads of the ΔFM recipe, and they
  differ by one flag.
- **The loss verdict is scale-dependent and unresolved** (S1a). Every
  12-row ΔFM-beats-plain number is contradicted by the full-table reads,
  and no full-table arm has separated the loss from the lr — ΔFM has only
  ever run at 2e-3 and plain only at 1e-3.
- **Glyph size is still unread at full scale** (S1b): no native hit under
  40 px on either table. If rows are size-bound, every native number is a
  number at the base's bubble size on a 512-class canvas and K1 needs the
  size lever in its recipe. Whether the σ band trains a 1.5-latent-px glyph
  is open, and the readers cannot referee kanji under 16 px.
- **Singles and sentences may be one budget, not two.** Every sentence pass
  so far has cost singles, and the anchor recovers singles by pinning `f` —
  i.e. by refusing the update the sentence step asks for. If lift and
  `single` move in opposite directions at every μ, the artefact needs two
  tables and the shipping question changes shape.
- **Eval and native may want different norms** (S1c). Then the artefact
  ships with α as a user knob — a `deploy_plan.md` change.
- **Sub-exact is a bag-of-glyphs ruler.** It cannot see order or count, the
  two things the strings arm showed a table *can* carry. It orders arms; it
  does not say the output is readable.
- **RAM** (46 GB usable): captions are pool-bounded, but later sentence
  steps add word and phrase-piece rows and K1 at 755 rows raises the text
  cache. The 10 k-item build stays the cap.
- **`configs/base.toml` still defaults to the preview pack** (`NOTE.md`).
  Anything that forgets `ANIMA_VOCAB_PACK` trains on a pre-delta'd table.

## Not this plan

- **Row blocks, per-block warmup, c / Q separation, `c_flat` in any form,
  `--box_weight` as a free knob** — closed (`findings.md`).
- **A bigger `--n_items` build.** Draws per row, not distinct items, is the
  measured budget; 10 k items is the RAM cap.
- **Encoder / composition / transplant / warm-start shortcuts** for kanji or
  for new glyphs — closed (`findings.md` *Do not re-propose*); K is an
  exposure budget, which is why it is a wall-clock table here and not a
  research phase.
- **A kana reference, contrastive ΔFM, cached Jacobians, OCR-reward rows**
  — the reasons are in `_archive/cjk_renderable_anima/plan_synth2.md`
  *Not this plan*.
- **lr 3e-3 for plain FM**, and lr as a plain-FM exposure lever (L0, closed
  2026-09-17: the budget is draws, not the lr integral).
- **jōyō 2 136 in one run** — K's inventory note decides the target first.

## Fallbacks (not phases)

- **Slot rows** — tokenizer routes the i-th piece of a quoted string to
  row (piece, i); train `Δ(piece, i) = Δ_piece + P_i`, bake the sum. Still
  a pack. Only if order and count cannot live in one table — the strings
  arm says they can.
- **W3 DiT-side ext-gated cross-attn LoRA** — only if slot rows also fail;
  carries the EN-safety list (ext gate on ext-free sequences, position
  mask, EN replay on mixed prompts — `reports/wake_plan_2026_09_13.md`)
  and leaves the native pack form.
