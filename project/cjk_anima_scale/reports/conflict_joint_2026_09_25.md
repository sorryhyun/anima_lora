# conflict + joint — stage chain vs one run, and what a row's budget actually is (2026-09-24 → 25)

The question after `micro_chain_result.md` (30 / 30 / 30 on 24 warm rows: pieces
+1–3 of 16, singles held only by μ) was whether **stage chain** (one band per
stage, warm chain, anchor) or **band per item, one run** is the design — and
how to set the budget. Two things were done: a training-free gradient read of
the question (`cjk_scale/conflict.py`), then ten trained arms on the *same*
15 000 renders (`data_stage*_run0923_micro_b30`) crossing chain / joint ×
steps × μ × lr. Every arm on the raw pack (sha `7b9fce0b…`), 24 rows
(`configs/runs/run0923_micro.toml`), through the daemon; job ids in
`runs/ledger.jsonl`. Rulers: `single` 16 × 2 seeds, `word` 8 × 2, `en` 12 × 2
(exact, sfx reader), native あ い 日 願 (both readers, 8 prompts × 2 seeds per
clause, of 16).

## 1. The conflict probe (job `…-230856-b86a5b`)

`output/cjk_anima_scale/conflict_run0923_micro_b30/report.md`. For each of the
three band stages, 600 items of its data dir, σ drawn in the stage band, the
trained loss backpropped onto the rows at the seed table
(`rows_step1_0921_merged`), no optimizer step. Per row and stage: `‖ḡ‖` (mean
per-draw gradient norm), `coh` = ‖Σg‖ / Σ‖g‖, `half` = split-half cosine (the
noise floor); per stage pair: cos(ḡ_A, ḡ_B) and `keep` = ‖ḡ_A + ḡ_B‖ /
(‖ḡ_A‖ + ‖ḡ_B‖); and, from the `b30` chain's tables, cos(Δ_A, −ḡ_B) — what
stage A moved the row by, against stage B's descent.

| kind | rows | 0709 ‖ḡ‖ / coh / half | 0507 ‖ḡ‖ / coh / half | 0305 ‖ḡ‖ / coh / half | pair cos / keep | Δ_A · −ḡ_B |
|---|---|---|---|---|---|---|
| single | 16 | 0.051 / 0.11 / +0.23 | 0.031 / 0.29 / +0.31 | – | 0709×0507 **+0.21 / 0.79** | Δ0709·−g0507 +0.15, Δ0507·−g0709 +0.12 |
| piece | 8 | – | 0.077 / 0.25 / +0.66 | 0.045 / 0.25 / +0.73 | 0507×0305 **+0.62 / 0.91** | Δ0507·−g0305 +0.40, Δ0305·−g0507 +0.22 |

**No band fights any other.** The cross-stage cosine sits at the within-stage
reliability ceiling (disattenuated ≈ 0.8 singles, ≈ 0.9 pieces); the Δ
columns are positive (a later band's descent runs *along* what the earlier
one bought). No single row is negative beyond its own `half` (い −0.12,
精 −0.04). Chain ≡ joint run with per-item bands.

Two more things it says. `coh` 0.1–0.3: the per-draw gradient is mostly
noise, the coherent part a tenth to a third of it, so the μ ≤ 0.01 "wipe" of
`micro_chain_result.md` was Adam's per-coordinate walk, not a band conflict,
and μ 0.1 was a spring against that walk. And `‖ḡ‖` falls ~40 % per band
step down (singles 0.051 → 0.031, pieces 0.077 → 0.045): a lower-px recipe
costs ≈ 1.6 × the draws for the same move — the exchange rate for shares.

## 2. The arms

All on the `b30` data (the joint dir = the three stage dirs concatenated,
each item carrying its stage's band; `cjk_scale/joint.py`,
`train.py::noisy_by_band`). "drift" = `warm_drift` at the end, mean
‖f − f₀‖ / ‖f₀‖ over the warm rows (`warm_cos` beside it).

| arm | steps/row | μ | lr | drift (cos) | single /32 | word /16 | native en / swap (of 64) | jobs |
|---|---|---|---|---|---|---|---|---|
| seed | – | – | – | – | 24 | 2 | – | `…-seed` |
| chain 30/30/30 (first) | 30 | 0.1 | 1e-3 | 0.07 (0.993) | 26 | 5 | 41 / 31 | `micro_chain_result.md` |
| chain 30/30/30 (`b30`) | 30 | 0.1 | 1e-3 | 0.07 (0.993) | 27 | 3 | 34 / 29 | `184036-{…}`, `213439-a9eafa` |
| chain 0/45/45 (`b0045`, no 0709) | 45 | 0.1 | 1e-3 | – | 27 | 4 | 41 / 39 | `184036-{e61273,fcc668}` |
| chain 30/30/30 | 30 | 0.02 | 1e-3 | – | 24 | 3 | – | `_mu002` |
| chain 30/30/30 | 30 | 0.01 | 1e-3 | 0.31 (0.952) | 23 | 0 | – | `194145-{7cb951,da29db,319b8d}` |
| chain 0/45/45 | 45 | 0.01 | 1e-3 | – | 22 | 3 | – | `194232-{d670bc,725470}` |
| **joint 90** | 90 | 0.1 | 1e-3 | 0.085 (0.994) | 25 | 3 | 38 / 35 | `233420-b87667` |
| **chain 100/100/100** | 100 | 0.1 | 1e-3 | 0.074 (0.992) | 27 | 4 | 41 / 32 | `000814-{92b018,60f130,6a1957}` |
| **joint 300** | 300 | 0.1 | 1e-3 | 0.082 (0.993) | 28 | 3 | 40 / 32 | `000814-5228c5` |
| joint 90 | 90 | 0 | 1e-4 | 0.11 (0.9925) | 26 | 4 | 41 / 31 | `025803-f6e43f` |
| **joint 90** | 90 | 0 | 1e-3 | **0.55 (0.877)** | 24 | 4 | **33 / 23** | `090100-49f260` |

(A μ 0.01 / lr 2e-4 chain + joint pair, `011049-*`, was cancelled after its
0709 stage read drift 0.078 — re-pinned; the pair below replaced it.)

Per-stage drift at μ 0.1, 30 vs 100 steps/row: 0709 0.045 → 0.037, 0507
0.13 → 0.107, 0305 0.07 → 0.074 — **3.3 × the steps, the same
displacement**. At μ 0 / lr 1e-4 the drift curve flattens at step ~1 800
with the cosine lr (0.11).

Contact sheets (`output/cjk_anima_scale/contact_seed_b30_joint.png`,
`contact_native_seed_b30_joint.png`: seed × `b30` chain × joint 90, same
compositions per seed): every table renders the seed's picture — bubble,
layout, strokes — and the ✓/✗ differences are marginal cases flipping (お s1,
動 s1). The failures are the seed's: 口 drawn as a box in all six, 聞 s0
missing the 耳, and every piece miss is the run / doubling signature (それを →
a fake sentence, はじ → はじし, プロ → ププロ, アン → アアン, メン → メンメ,
すご → すごご) — right glyphs, wrong count — untouched by any arm.

## 3. The comparator: `micro_warm_0923`

The only arms on record that moved pieces on this seed table
(`output/wake_probe/rows_micro_warm_0923_mw_warm_lo{_40,_80,}`: 16 piece rows
alone, scene-only 0.5–0.7, **μ 0, lr 1e-3**, batch 4, cosine):

| steps/row | drift (cos) | pieces exact /32 |
|---|---|---|
| 40 | 0.52 (0.907) | 6 |
| 80 | 0.72 (0.845) | 8 |
| 188 | 1.04 (0.743) | 13 |

Pieces gain monotonically with **displacement of half to a full row norm**.
The joint μ 0 / lr 1e-3 arm above reached 0.55 and read 4/16 ≈ the 40/row
point (6/32) — on the curve, not off it — while the singles in the same table
rode the same displacement and paid native (swap 23, い 1, 願 2 of 16).

## 4. Verdicts

1. **Stage chain vs joint is a non-question.** Read training-free (§ 1) and
   confirmed by every pinned pair (30/30/30 ≈ joint 90, 100/100/100 ≈ joint
   300, within the ±2 floor with the same rows flipping). Stages remain the
   data builder's gate unit; the trainer needs no chain. A per-stage anchor
   buys nothing the bands don't already agree on.
2. **Steps per row is not the budget variable.** μ (spring) or lr (Adam path
   length) sets the displacement; past that equilibrium more steps re-sample
   noise. Every μ 0.1 or lr 1e-4 arm is the seed table ± 0.1 and reads as
   the seed on both sheets. `coh` 0.1–0.3 says why the walk wins at μ ≤ 0.01
   / lr 1e-3 on a mixed table.
3. **The seed's singles and pieces want opposite regimes.** Singles (24/32
   on the seed) have nothing to gain from displacement and lose native to
   it; pieces (2/16) need drift ≈ 1.0 at lr 1e-3, μ 0, ~200 steps/row of
   piece draws (§ 3). One table, one μ, one lr cannot serve both — this
   series tried every corner of that box.

What follows: **freeze the singles, train the pieces** — a run whose
inventory is the pieces with `context = "seed"` (the singles ride at their
seed value, zero gradient — `cjk_scale/rows.py`), the 0507 + 0305 piece data
as one joint dir, μ 0, lr 1e-3, ≥ 200 steps/row; single refinement, if ever,
is a separate pinned pass. Not per-row anchors or drift caps (closed,
`sent_run.md`). No further mixed-table micro arms of this shape — they cannot
read anything past what is here. Unread: whether sentence draws (the 0309
data, 28 px median → the 0507 window by the law) fight single identity —
the same probe with `stage0309` as a fourth stage.

## 5. Tools this added

- `cjk_scale/conflict.py` + `scale.py --steps conflict [--conflict_stages …]
  [--probe_items N] [--warm_from …]` — the gradient read, at any table point.
- `configs/joint.toml` (`joint_from = [...]`, no mix) + `cjk_scale/joint.py`
  — the merged data dir; `train.py::noisy_by_band` draws σ per item from its
  `band`; `run0923_micro.toml [budget] joint = 90`.
- `scale.py --lr_rows` override (beside `--init_anchor`, `--steps_per_row`).
- Data dirs under a new tag can be symlinks to an existing build
  (`data_*_run0923_micro_b100 → …_b30`), so an arm re-uses renders,
  latents and the TE cache; the ledger's argv records the tag.

## 6. Addendum (2026-09-25 evening) — the drift column re-read against the seed

The `drift` column in § 2 is `warm_drift` = ‖f − f₀‖ / ‖f₀‖ with f₀ **the
warm-from table** (`cjk_scale/rows.py::log_record`): for a chain stage that
is the previous stage, for a joint arm the seed. The two are not one
ruler. Recomputed from `trained.pt` against `rows_step1_0921_merged`, per
kind (16 singles / 9 piece rows of the conflict inventory; `row_scale`
applied):

| arm | logged | singles vs seed | pieces vs seed | pieces vs prev stage |
|---|---|---|---|---|
| chain `b30` 0709 | 0.045 | 0.069 | 0.002 | – |
| chain `b30` 0507 | 0.112 | 0.083 | 0.241 | 0.241 |
| chain `b30` 0305 (end) | 0.072 | 0.083 | **0.385** | 0.200 |
| chain `b100` 0305 (end) | 0.074 | 0.067 | 0.401 | 0.206 |
| joint 90, μ 0.1 | 0.085 | 0.033 | 0.177 | – |
| joint 300, μ 0.1 | 0.082 | 0.026 | 0.180 | – |
| joint 90, μ 0, lr 1e-4 | 0.110 | 0.064 | 0.191 | – |
| joint 90, μ 0, lr 1e-3 | 0.554 | 0.365 | **0.890** | – |
| joint 90, μ 0, lr 1e-3, `grid_box` | 0.605 | 0.376 | **1.014** | – |
| stage0709 alone, μ 0 | 0.182 | 0.254 | 0.054 | – |

What changes:

- **The chain did not stay within 0.1 of the seed.** Its pieces are at
  0.24 after 0507 and 0.39–0.40 at the end (the per-stage anchor holds each
  stage to its predecessor, and the stages add up). "Chain ≡ joint" in § 4
  holds on the rulers, but the two tables are not at the same displacement:
  the μ 0.1 joint's pieces sit at 0.18.
- **The μ 0 / lr 1e-3 joint's pieces are at 0.89 (1.01 under `grid_box`),
  not 0.55.** § 3 put that arm "on the `micro_warm_0923` curve at the
  40/row point" by matching the mixed-table mean 0.55 to the comparator's
  0.52; per kind, the pieces got the displacement of the 80–188/row points
  (0.72–1.04) and read 4/16 — the comparator read 8–13 / 32 there. The
  displacement was bought; the hits were not. So `next_2026_09_25.md` § 3's "A flat at
  drift ≈ 0.5 → rerun at 200/row" is not a supported branch: more
  displacement is not what that table lacked. What differs from the
  comparator is the inventory (singles present), the recipe mix (grid
  draws), and the loss (`grid_box`), not the path length.
- **The μ 0 norm pull is real but weak.** At μ 0 the regularizer switches
  to `free_residual` (1e-3) · ‖f‖² on every touched row (`rows.py::regularized`,
  the `elif` branch) — so a μ > 0 arm is a different objective, not "the same
  plus a spring". Empirically it did not shape these tables: under μ 0 / lr
  1e-3 the piece norms **grew** to 1.34 × the seed (1.40 with `grid_box`),
  singles 0.94 ×; at lr 1e-4, 1.03 × / 0.98 ×.
- **Exposure per row is grid-dominated.** In the `b30` data dirs a piece row
  appears in 313 `scene_piece` items and 622 (0507) / 1 664 (0305)
  `grid_string` items; a single in 156 `scene_single` + 813 `grid_single`
  (0709). A recipe share is not a row-exposure share; any per-draw price has
  to be multiplied by these before it says anything about a mix.

The verdicts in § 4 stand as read on the rulers; the *displacement* story
under them (2, 3) is corrected as above.
