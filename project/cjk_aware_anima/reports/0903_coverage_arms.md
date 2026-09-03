# plan_zh2 U4 / U0 / U1 — coverage-aware distillation, first arms (2026-09-03)

*Follows `plan_zh2.md`. Everything here is the `synthjakozh1_r256` recipe
(`param=global`, rank 256, span loss, `--trust provenance`, 12k steps, bs 32,
lr 1e-3) on the four restaged caches (69,558-row asset, 200,444 pairs). The
new distill flags are `--holdout_rows` (U4), `--span_min_visits[_bg]` (U1);
the probe is `probes/map_bands_probe.py` (U0). Code: `scripts/distill_cjk/rows.py`.*

## U0 — what the shared map does per visit band

`map_bands_probe.py` on `synthjakozh1_r256` (58,968 rows) against the v2
init, visits from the restaged caches (9,830 / 58,968 visited under the
training registers), 4,000 rows sampled per band, k = 10. The
`synthjakozh1sym_r256` pack reads the same to the second decimal (JSON:
`reports/u0_map_bands.json`).

| band | rows | cos(init, trained) | p10 | ‖trained‖/‖init‖ | PR init → trained | kNN overlap |
|---|---|---|---|---|---|---|
| 0 (unvisited) | 49,138 | **0.343** | 0.219 | **0.969** | 10.5 → 15.3 | **0.463** |
| 1–4 | 1,457 | 0.315 | 0.260 | 0.992 | 299 → 148 | 0.303 |
| 5–49 | 3,178 | 0.314 | 0.258 | 0.998 | 342 → 160 | 0.348 |
| 50–499 | 2,746 | 0.321 | 0.264 | 0.991 | 335 → 161 | 0.308 |
| 500+ | 2,449 | 0.331 | 0.268 | 0.952 | 334 → 174 | 0.315 |
| 0 / han | 30,765 | 0.312 | 0.246 | 1.002 | 147 → 127 | 0.462 |
| 0 / hangul | 10,468 | 0.275 | 0.168 | 0.930 | 46 → 46 | 0.723 |
| 0 / kana | 651 | 0.294 | 0.236 | 0.983 | 115 → 87 | 0.595 |

Unvisited rows whose nearest neighbour (in a mixed 4k unvisited ∪ 4k
visited sample) is a *visited* row: **0.171 → 0.216**; mean cos to that
nearest visited row 0.626 → 0.591.

Reading:

- **The map is not gentle anywhere, and it is not selectively harsh on
  unvisited rows either.** Every band lands at cos ≈ 0.31–0.34 to its init
  — the trained row is mostly the low-rank term's output, not the init plus
  a nudge — while the norm is preserved (≈ 1.0, *not* the 0.3× the plan
  feared: `global_gain 0.30` rescales a vector the rank-256 term has already
  made ~3× longer). So the "unvisited rows are scaled to ~0.3 norm" branch
  of the U0 gate is false, and the "unvisited rows move ≤ visited rows"
  branch is true only marginally (cos 0.343 vs 0.31–0.33).
- **Neighbourhoods are lost about equally.** Unvisited rows keep 46 % of
  their k = 10 neighbourhood (hangul 72 %, kana 60 %, han 46 %); visited
  bands keep 30–35 %. The visited bands also lose half their effective
  dimensionality (PR 335 → 160, the learned-diagonal collapse §9/§10 already
  named), while the unvisited band's PR is dominated by the v2 init's own
  clustering (10.5) and barely moves.
- **Mild folding onto the visited manifold**: the share of unvisited rows
  whose nearest neighbour is visited rises 17 % → 22 %. Not a collapse.

Verdict against the gate: neither branch cleanly. The map applies one
large, uniform rotation (cos ≈ 0.3) to every row; on visited rows the
teacher paid for it, on the 49k unvisited rows nothing did. Whether that
rotation *helps* a row the loss never saw is exactly U4's number
(`row_holdout.gap_cos`), so U2's α-gate is decided there, not here — U0
only rules out the "already gentle" shortcut. The render-side read (an
unvisited-han JA prompt on the trained pack vs the v2 init) is comfy-path
only and still owed.

## U4 / U1 — arms

All three on the same U4 holdout (5 % of rows with 5 ≤ visits < 500,
seeded, script-stratified: **297 rows** — han 213 / hangul 52 / kana 24 /
mixed 8 — at 0.17 % of visits; **29,135 spans stripped from 24,420 pairs**,
every pair kept) so they share one pool. `held` = 2,048 of the stripped
spans, `control` = 2,048 trained spans from the same pairs (ext-touching
only); `gap = control − held`. Standard holdout = the 1,200-pair pair-split
eval every earlier pack reports. Jobs `20260903-212835-{f1675b,9c14ee,9388f0}`,
results `bench/cjk_distill/results/20260903-{2129,2227,2325}-u*`.

| arm | floor | span weight dropped | final span | recovery | cos(s,t) | disc_far | held cos / disc / top1 | control cos / disc / top1 | gap cos |
|---|---|---|---|---|---|---|---|---|---|
| (init, step 0) | — | — | — | — | — | — | 0.405 / 0.253 / 0.218 | 0.378 / 0.241 / 0.276 | −0.027 |
| u4-base-r256 | off | — | 0.0761 | 0.015 | 0.137 | 0.056 | **0.587** / 0.405 / 0.458 | **0.876** / 0.701 / 0.917 | **0.289** |
| u1-k2-r256 | k=2 | 0.0015 % (362 visited rows) | 0.0761 | 0.015 | 0.137 | 0.056 | 0.586 / 0.404 / 0.458 | 0.876 / 0.701 / 0.917 | 0.290 |
| u1-k5-r256 | k=5 | 0.030 % (1,470 visited rows) | 0.0761 | 0.015 | 0.137 | 0.056 | 0.586 / 0.404 / 0.455 | 0.876 / 0.701 / 0.920 | 0.290 |

Held-span trajectory (base arm; k2 and k5 identical to the third decimal):

| step | 0 | 250 | 1,000 | 3,000 | 6,000 | 9,000 | 12,000 |
|---|---|---|---|---|---|---|---|
| held cos | 0.405 | 0.560 | 0.583 | 0.591 | 0.589 | 0.589 | 0.587 |
| control cos | 0.378 | 0.790 | 0.841 | 0.862 | 0.870 | 0.874 | 0.876 |

Reading:

- **The shared map does generalize to rows it never supervised — by a lot,
  and early.** A held-out row's span goes from cos 0.405 at init to 0.587,
  most of it inside the first 250 steps (0.560); rank-1 retrieval of its own
  teacher among 2,048 held spans doubles (0.22 → 0.46). So "a row the
  corpus never shows should stay at its init" (plan principle 1) is the
  wrong prior *for the 5–499 band*: the map is worth +0.18 cos to a row
  with no direct evidence. U2's `α0 = 0` arm (unvisited rows frozen at init)
  is predicted to lose on this metric, and `α0 = 0.25` to lose most of the
  gain; the arm that survives the prediction is the identity anchor
  (`--unseen_anchor`, α ≡ 1), if any.
- **The gap is where the capacity goes.** Held saturates at ~0.59 by step
  1,000 and drifts down 0.004 over the next 11k steps while control climbs
  0.841 → 0.876: after the first ~1k steps the rank-256 term is fitting
  visited rows, not learning a better shared correction. That is the same
  statement as §10's "rank tightens the fit, holdout unchanged", now with
  the row-level number attached (gap 0.29 in cos, 0.46 vs 0.92 in top1).
  A held-out row ends at two-thirds of a trained row's cos.
- **U1 is a no-op at the metric level — CLOSED.** k=2 silences 362
  singleton rows' spans (0.0015 % of span weight), k=5 silences 1,470 rows'
  (0.030 %); both match the base arm to three decimals on every number,
  held and standard alike. Rows seen 1–4× are not steering the map — the
  visit distribution is so top-heavy that they have no measurable purchase
  on a rank-256 map trained for 12k steps. Plan principle 2 ("a row seen
  once is not a teacher") was true and irrelevant. The floor stays as a
  hygiene flag: its `mapped-unseen` provenance tier is the useful part (a
  reader of `u1k5_r256.json` sees 61,165 rows carried along vs 8,393
  trained on), it is not a lever. Do not spend a U6 arm on
  `--span_weight_pow` for the same reason unless U4 says otherwise.
- **Caveats on what "held out" means here.** (i) The held rows are visited
  rows whose *direct* supervision was removed; their pairs stay in the
  pool, so the row is still looked up and shapes its neighbours' outputs —
  a second-order path into the map the holdout does not close. (ii) They
  are drawn from the 5–499 band, i.e. the same distribution the map was
  fit on; the 49k truly unvisited rows are mostly rare `char`-block
  entries the v2 init clusters at PR ≈ 10 (U0), and the +0.18 may not
  transfer to them. A `--holdout_rows` draw with `--holdout_rows_max_visits
  50` (5–49 band only) is the cheap next probe of (ii).
