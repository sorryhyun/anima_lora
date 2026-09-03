# CJK-aware Anima — plan_zh2: coverage-aware distillation (2026-09-03)

*Follows `plan_zh.md` (joint JA+KO+ZH pack, `synthjakozh1_r256` is the
current best). Starts from the coverage audit in
`docs/experimental/cjk_ext_vocab_coverage.md`: the training pool visits
9,831 of 58,968 rows, 1,455 of those fewer than five times, and under
`--param global` every row — visited or not — is pushed through the same
map (gain 0.30, diag rms 0.75). The symbol block (rows 58,968–69,557,
routed by the pack's `route`) exists as of today but no cache has staged it
and no teacher speaks for it. This plan refines the *training* so that (a)
rows the corpus never shows are not silently degraded, (b) rows it shows
once do not steer the shared map, and (c) the new symbol rows get a signal.*

## Premise (measured, `visit_stats.py`)

| fact | number |
|---|---|
| rows visited / total (JA+KO+ZH pool, 200,744 pairs) | 9,831 / 58,968 |
| visited rows seen 1–4× | 1,455 |
| visits carried by the top-100 rows | 34.8 % (median visited row: 62) |
| trainable per-row parameters in the shipped recipe | 0 (`param=global`) |
| trained pack `global_gain` / `diag rms` | 0.299 / 0.749 |
| structurally unreachable rows | 2 (fullwidth `１０` `２０`) |
| symbol rows / visited | 10,590 / 0 (caches predate the block) |
| `<unk>` left in the student stream after the symbol block (1/20 sample) | 0 |

Two things follow. **Unvisited rows are not neutral today** — they receive
the full shared correction, fitted on a Han-heavy visited set, and nothing
in the eval looks at them. **Singletons are full-weight teachers** — a row
seen once contributes to the span loss at the same weight as `、` (425k
visits), and the map has no per-row slack to absorb it.

## Principles

1. **A row the corpus never shows should leave the pack as close to its
   init as the shared map allows** — its init is the only evidence we have
   about it. Corollary: the map's job on unvisited rows is to be *gentle*,
   not to be *right*; correctness is only defined where there is a teacher.
2. **A row seen once is not a teacher.** Below a visit floor a span
   contributes no gradient; the row still rides the map (like an unvisited
   one) and is tagged so in provenance.
3. **Measure generalization on rows, not on pairs.** The holdout is a pair
   split from the same distribution, so it cannot see unvisited rows. Hold
   out *rows*.
4. **Symbols enter as a register with a real teacher**, never through the
   EN caption (where they are `<unk>`).

## Phases

### U0 — probe: what the shared map does to unvisited rows (½ day, CPU)

Materialize `synthjakozh1_r256` and compare per row against the v2 init,
split by visit band (0 / 1–4 / 5–49 / 50+): cos(init, trained), norm ratio,
participation ratio of each band's rows, and kNN overlap (k=10) among
unvisited rows before/after (does the map preserve their neighbourhood or
fold them onto the visited manifold?). Same for `_fdiag`. Cheap: one
`ExtTable.materialize()` per pack, `probes/spread_probe.py` already has the
PR/kNN code.

*Landed 2026-09-03* as `probes/map_bands_probe.py` (bands + unvisited-by-
script + "nearest neighbour is a visited row" before/after; JSON to
`reports/u0_map_bands.json`). The render read is comfy-path only (no
in-tree inference loads a pack) and is still owed.

*Gate:* if unvisited rows move ≤ visited rows (cos to init higher, kNN
overlap ≥ 0.5) the map is already gentle and U2 shrinks to a regularizer
check. If unvisited rows are scaled to ~0.3 norm and lose neighbourhood
(kNN overlap < 0.3), U2 is load-bearing. **Also read the render side:** a
JA prompt made of unvisited-but-common characters (Ext-A never, but e.g.
shinjitai the ZH corpus does not share — 髪 顔 獣 are *visited*; pick from
the `qwen/han` unvisited band that the JA wiki uses) rendered on the
trained pack vs the v2 init.

### U1 — visit floor on the span loss (`--span_min_visits k`, 1 day)

New distill flag: a span whose student tokens contain any ext row with
`visits < k` (computed once on the training pool, as `compute_visits` does
now) gets weight 0 (or `--span_min_visits_bg b` to keep a background
weight, mirroring `--span_focus_bg`). `k=2` drops singletons; `k=5` matches
`--min_visits`. Rows below the floor are tagged `mapped-unseen` in the
pack's provenance so a reader can tell "trained on" from "carried along".

*Why not just drop the pairs:* a pair with one rare row still carries 20
common spans; only the rare span should go quiet.

*Read:* Z3 distill metrics (cos/discrimination on the standard holdout must
not move — the floor removes < 1 % of span weight) + U4's row-holdout
metric (the actual target) + the JA/ZH grids on the same prompts as
`reports/0903_rank_armC.md`.

*Landed 2026-09-03* (`--span_min_visits k`, `--span_min_visits_bg b`;
`result.json` carries `span_floor.weight_dropped_frac` measured on a 4k-pair
sample, and the pack's `provenance` gains the `mapped-unseen` tier for every
row with `visits < k`, unvisited rows included). Arms queued the same
evening, all on the U4 holdout (5 %, cap 500) so they share one pool:
`u4-base-r256` (floor off), `u1-k2-r256`, `u1-k5-r256` — r256 recipe
otherwise verbatim.

*Verdict 2026-09-04: **inert, CLOSED.*** k=2 drops 0.0015 % of span weight,
k=5 0.030 %; both match the base arm to three decimals on every metric
(`reports/0903_coverage_arms.md`, findings §12). Keep the flag for its
`mapped-unseen` provenance, not as a lever.

### U2 — visit-gated correction (2 days)

*Revised by U4 (2026-09-04):* the base arm's held rows gain **+0.18 cos**
from the shared map with no direct supervision (0.405 → 0.587, findings
§12), so `α0 = 0` freezes rows at a *worse* point and `α0 = 0.25` forfeits
most of the gain — both arms are predicted to lose on U4 and should not be
run first. What survives is the identity-anchor arm (`--unseen_anchor`,
α ≡ 1) and the question of whether the gain transfers below the 5–49
band (`--holdout_rows_max_visits 50` probe) and to the 49k truly unvisited
`char` rows.

Make the correction strength a function of evidence, per row, inside
`ExtTable.rows()`:

```
out = init + α(v) · Δ_global(init) + r_row        # r_row only in global_row
α(v) = α0 + (1 − α0) · min(1, log1p(v) / log1p(v_sat))
```

`α0` (default 0.25) is what an unvisited row receives; `v_sat` (default
50) is where a row gets the full map. `visits` is already a per-row buffer;
`α` ships in the pack json as a per-row float next to `provenance`, so the
ComfyUI node needs nothing — the exported table is materialized. The scalar
gain and the diagonal are folded into `Δ_global` (they are part of the
correction, not of the init), so an unvisited row keeps ~its init norm
instead of 0.3×.

Two arms against `synthjakozh1_r256` (rank 256, same corpus, same steps):
`α0=0.25` and `α0=0` (unvisited rows frozen at init). Add an **identity
anchor** term as the ablation of the ablation: `λ · E_{r∼unvisited}
‖Δ_global(init_r)‖² / ‖init_r‖²` sampled 256 rows/step (`--unseen_anchor
λ`), which keeps `α ≡ 1` but asks the map to be small off-corpus.

*Gate:* U4's row-holdout metric ↑ with standard-holdout metrics flat, and
the render grids not regressing on names (the name failure in `findings.md`
§4 is *not* expected to move — names are a composition problem).

### U3 — `global_row` revisited with a gentle global (1 day, after U2)

`global_row` was set aside when `global` alone won on the JA-only corpus.
With U2 in place the split becomes principled: a small shared map (rank 64,
`α0` gated) carries the systematic anchor-map error to every row, and
per-row residuals (`--min_visits 5`) carry what the corpus actually shows.
Row residuals get `--wd` toward init. This is the arm that can finally
separate the degenerate char-row pairs `plan_zh.md` says the global map
cannot (random char-row pairs at cos 0.5+): a residual is per row.

*Gate:* the char-row separability probe (`gates/separability.py`) on the
heavy JA kanji + U4 metric not worse than U2's best.

### U4 — row-disjoint holdout (½ day; do this first, it gates everything)

Add `--holdout_rows p` to `distill.py`: pick p (default 5 %) of the
*visited* rows at random (seeded), remove from the training pool every span
that contains one of them (spans, not pairs — keep the pair's other spans),
and score at eval the held-out spans only: `cos_student_vs_teacher` and the
attn-readout discrimination restricted to those spans. That number is the
first direct measurement of "how does the map do on a row it never saw
trained". Report it in `result.json` as `eval.row_holdout.*` and in the
`0903_rank_armC.md`-style tables.

Rows the corpus shows only once cannot be held out meaningfully (one
occurrence); sample the holdout from rows with ≥ 5 visits and stratify by
script (han / hangul / kana) so KO does not dominate by row count.

*Landed 2026-09-03* (`--holdout_rows`, `--holdout_rows_{min,max}_visits`,
`--holdout_rows_eval`; `scripts/distill_cjk/rows.py`). One amendment
measured on the way in: the 500+ band is ~140 rows carrying a third of all
visits, and a 5 % stratified draw picks 139 of them — that strips **4.8 %
of all visits** (≈ every span containing 髪/目/の…) out of the pool, for a
question those rows do not pose (the map fits them regardless). With
`--holdout_rows_max_visits 500` (default) the draw is 297 rows from the
5–499 band at **0.17 %** of visits. The metric is `eval.row_holdout.held`
(cos / disc / top1 over up to 2,048 stripped spans) next to
`eval.row_holdout.control` (the same on trained spans from the same pairs);
`gap_cos` is the generalization number. Known leak: a held-out row is still
*looked up* (its pair stays in the pool), so its embedding reaches the map
through the neighbours' context — the holdout removes direct supervision,
not presence.

### U5 — symbol register `tags_sym` (2 days, data first)

The symbol rows have no teacher: the EN caption gives `<unk>`. Build the
register the way the glossary registers were built:

- **Teacher text = the tag's wiki definition, EN**, one short clause
  (`^^^` → "surprise lines above the head", `:<` → "frowning mouth", `^ ^`
  → "closed eyes smiling", `☆` → "star symbol", `×` → "x-shaped", `♪` →
  "musical note"; Danbooru wiki `other_names`/description, hand-verified —
  the list is ~40 tags with > 20 occurrences in `image_dataset`). Provenance
  `via=wiki_verified` (1.0).
- **Student text = the symbol verbatim**, in a real caption slot (compose
  with the tag-register recipe, `datasets/synth_tags.py`, so the symbol sits
  among ordinary tags on both sides; only the symbol span differs).
- Separators (`·` in zh names, `~` in titles) get **no** entry: the name /
  title span already covers them and the surrounding characters carry the
  teacher; U0's probe on `sym` rows after one distill tells whether the
  separator row drifted somewhere harmful.
- Kaomoji: no teacher exists beyond "emoticon"; leave the rows at the map
  (they are routed now, which is the whole ask: the prompt no longer
  collapses them into one `<unk>` shared with `^^^`).

Then **re-stage the four caches** (`scripts/distill_cjk/cache.py`) so the
existing `^^^`/`<`/`·`/`~` positions land on symbol rows instead of `<unk>`
— this is a full restage (~5 h GPU over 4 caches; keyed on ids so nothing
else changes). Until then a distill on the old caches trains 0 symbol rows,
which is harmless (they stay at init through the map).

*Gate:* on a 20-prompt symbol grid (`^^^`, `:<`, `^ ^`, `☆` in JA/ZH
captions) the trained pack renders the described attribute at ≥ the rate
the EN prompt with the *description* does; today's rate with the symbol is
the `<unk>` baseline.

### U6 — corpus-side visit balancing (optional, 1 day)

The map is fit under a visit distribution where 100 rows carry a third of
the gradient. Two knobs, both distill-time (no restage): `--span_weight_pow
p` scaling each span's weight by `(1 / mean_row_visits)^p` (p=0.5 =
inverse-sqrt frequency, the usual choice); and `--register_sampling` per
language so KO's 6.5 M visits on 1,609 rows do not out-vote ZH's 5.8 M on
6,329. Read on U4.

## Not in scope / anti-goals

- **No new rows for scripts T5 can spell.** The `<unk>` test is the
  criterion; Cyrillic/Thai/Devanagari are not in the candidate ranges on
  purpose (they would add ~50k rows for no anime-tag user).
- **Do not prune unvisited rows from the pack.** Lookup needs the row; a
  73 MB pack is fine. Prune only if U0 shows the map actively harms them
  *and* U2 cannot fix it — then ship them at init (α=0), not absent.
- **No per-row LR / per-row optimizer tricks** on `global` mode: there is
  no per-row parameter to schedule. U2's α is the only per-row knob.
- **Build determinism** (`docs/experimental/cjk_ext_vocab_coverage.md`, last
  section): store `W` in the asset json or fit on CPU single-thread so a
  rebuild is a strict superset by bytes, not by construction. Small, do it
  with U4's code change.

## Order and budget

*Original:* U4 (½ d) → U0 (½ d) → U1 (1 d) → U2 (2 d) → U5 data (1 d) +
restage (GPU, overnight) → U5 distill (½ d) → U3 (1 d) → U6 if U4 says the
map is still frequency-bound. Roughly 7 working days; every arm is a
distill (~40 min at 12k steps, rank 256) plus grids.

*Revised 2026-09-04 (U4/U0/U1 done in one evening, findings §12):*
**next is U5**, the symbol register — the only phase that adds a teacher
where none exists. U1 is closed (inert), U6 is pre-empted by the same
number, and U2 is deferred: its α-gate arms are predicted to lose on U4
and the surviving identity-anchor arm is a regularizer question, not a
coverage one. U2 (`--unseen_anchor` + the `--holdout_rows_max_visits 50`
probe) and U3 come back after U5 lands, if the symbol grid says the map
side is what limits it. The restage U5 needs has already happened (the
four caches sit on the 69,558-row asset; 0 symbol rows visited only
because no pair carries a symbol teacher yet), so U5 is data → distill →
grid, no GPU overnight. Every U5 arm carries `--holdout_rows 0.05` so
`eval.row_holdout.gap_cos` stays comparable with the base arm.

## Deliverables

- `distill.py`: `--holdout_rows`, `--span_min_visits[_bg]`, `--alpha0` /
  `--alpha_sat` (U2), `--unseen_anchor`, `--span_weight_pow`; pack json
  gains per-row `alpha` and the `mapped-unseen` provenance tag.
- `datasets/build_pairs_sym.py` + `assets/tag_glossary_sym.json` (U5).
- `reports/09xx_coverage_arms.md`: one table, the U4 row-holdout metric
  next to Z3's, per arm.
- `findings.md` §11 entry with the verdicts.
