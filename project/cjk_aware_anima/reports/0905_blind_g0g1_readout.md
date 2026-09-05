# Blind pairs s01–s08 — G0 calibration + G1 geometry bracket (graded 2026-09-05)

User graded all 8 sets double-blind (24 pairs each, 8 rows × seeds 42/7/1234,
no ties, no skips). Per-set tables: `blind_s0N_*.md`. Verdicts committed to the
private repo (`ee91dd3`, `9561689`). p = two-sided exact binomial on 24 pairs.

## Results

| set | pair | wins | p | read |
|---|---|---:|---:|---|
| s02 | C9 vs C9s2 (same recipe, seed 2) | 9–15 | 0.31 | **grader/seed noise floor: 15–9 is nothing** |
| s01 | C9 vs P (presence tag only) | **19–5** | 0.007 | C9 > P. The user's eye was right; OCR count and my read were wrong |
| s07 | C9 vs ROTATE (r256 under random orthogonal rotation + rotated mean) | **19–5** | 0.007 | rotation hurts as much as removing the rows |
| s03 | C9 vs R (Gaussian, spectrum/norm/mean matched) | 7–17 | 0.06 | R ≥ C9, borderline; non-transitive with s04 |
| s06 | C9 vs COLLIDE (native EN T5 rows) | 15–9 | 0.31 | noise |
| s05 | C9 vs HOT (norm × 5) | 9–15 | 0.31 | noise — **HOT did not hurt** (plan's one "expected to hurt" arm) |
| s08 | C9 vs INIT (v2 anchor init, no distill) | 14–10 | 0.54 | noise |
| s04 | P vs R | 10–14 | 0.54 | noise |

Per-seed splits (from the per-set reports): s01 is carried by seed 42 (8–0)
and 7 (7–1), seed 1234 is 4–4; s07 is 8–0 / 5–3 / 6–2; s03 R is 5–3 / 5–3 / 7–1.
Row sweeps (3–0): s01 C9 on r1 r4 r6 r7; s07 C9 on r1 r2 r4 r6 r7; s03 R on r4 r5 r6.

## What moved and what did not

Two arms fall to the P level (19–5 against C9): **no rows** and **rotated
rows**. Everything else is inside the seed floor. So on this grid:

- The span rows are **not inert** for the LoRA (contradicts findings §14's
  "P ≈ C9" — that read came from the PP-OCR floor and an unblinded eye).
- The property that matters is **not** content (R ≥ C9), **not** scale (HOT ≈ C9),
  **not** native collisions (COLLIDE ≈ C9), **not** distillation (INIT ≈ C9), and
  **not** row-to-row relational structure (ROTATE keeps it and still loses).
- **Correction (measured after writing the first draft):** R's mean direction is
  *also* random — `make_random_pack.py --mode matched` draws a random basis
  and a random mean direction (cos to the trained mean: R −0.005, ROTATE
  +0.040, HOT +0.028; COLLIDE +0.81). So "native mean alignment" does **not**
  separate R from ROTATE. On every geometry the script measures (spectrum /
  PR, per-row norm distribution, mean norm, orientation) R and ROTATE are
  the same pack class; the only thing ROTATE has that R lacks is the trained
  pack's exact row-to-row structure (clusters, 2.2 % near-duplicate pairs,
  per-char norm assignment) carried into a wrong orientation. That is a
  weak mechanism, and each arm is one training run — the s02 floor is one
  seed pair. The cheaper reading is that one of s03 / s07 is a training-run
  fluke. **Before G2, retrain ROTATE with a second rotation seed (or R with a
  second draw), ~50 min GPU, and re-pair against C9.** If ROTATE loses again,
  the property is "trained relational structure in the wrong frame hurts";
  if it does not, the bracket is flat and only P (no rows) is confirmed bad.
- R vs C9 (17–7) should not be read as "random beats trained": R vs P is only
  14–10, so the three sets are not transitive at this n. Treat R = C9.

## Instrument calibration (judge vs blind)

Tagger recall (`unmask_grid_judge_all.md`): C9 = C9s2 0.912 > R 0.893 >
COLLIDE 0.885 > P 0.884 > HOT 0.880 > ROTATE 0.875 > INIT 0.855.

- Agrees in direction on C9 > P, C9 > ROTATE, C9 > INIT, C9 > COLLIDE.
- Wrong on HOT (recall gap 0.03 = the C9–P gap, blind says noise) and on R.
- So recall is a usable **sign** readout for a P-sized gap but its 0.03
  margin is not a ranking margin; the seed pair (C9/C9s2 both 0.912) makes
  the recall floor look tighter than the blind floor is. PE cos columns
  stay saturated.

## Consequence for the plan

Nothing beat C9 beyond the floor, so G3/G4 have no positive target. The
only confirmed property is *rows must exist* (P). ROTATE is the one
candidate for a second property and it needs a replication run before G2
is spent on it. Scale (HOT), collisions (COLLIDE), distillation (INIT) and
content (R) are inert — do not add norm / collision terms to the distill.

## Addendum — s09 / s10 at fresh seeds 1/2/3 (graded 2026-09-05, later the same day)

Both arms of each new set had already been seen by the grader at 42/7/1234,
so the grids were re-rendered at seeds 1/2/3 (`probes/regrid_set.py`,
render-only, same LoRAs) before pairing.

| set | wins | p | read |
|---|---:|---:|---|
| s09 HOT vs COLLIDE | 15–9 HOT | 0.31 | noise, as predicted from s05/s06 |
| s10 ROTATE vs R | 11–13 R | 0.84 | **flat** — the direct test of s03 + s07 (which together implied R ≫ ROTATE) |

s10 kills the ROTATE story: with fresh render seeds ROTATE and R are
indistinguishable, so s07's 19–5 (and/or s03's 17–7) was a one-off of the
42/7/1234 grids, not a property of the pack.

### The pair count overstates the power

Pairs are not independent: the grader's preference clusters by **row**
(same prompt, three seeds). Re-scoring every set as rows-won (majority of
the 3 seeds) and counting 3–0 sweeps:

| set | pairs | rows won | 3–0 sweeps | row sign-test p |
|---|---|---|---|---:|
| s01 C9 vs P | 19–5 | 7–1 | C9 4 / P 0 | 0.07 |
| s02 C9 vs C9s2 (floor) | 9–15 | 4–4 | C9 0 / **C9s2 3** | 1.00 |
| s03 C9 vs R | 7–17 | 2–6 | R 3 | 0.29 |
| s04 P vs R | 10–14 | 2–6 | 0 | 0.29 |
| s05 C9 vs HOT | 9–15 | 2–6 | HOT 1 | 0.29 |
| s06 C9 vs COLLIDE | 15–9 | 5–3 | C9 2 | 0.73 |
| s07 C9 vs ROTATE | 19–5 | 6–2 | C9 5 | 0.29 |
| s08 C9 vs INIT | 14–10 | 4–4 | C9 2 | 1.00 |
| s09 HOT vs COLLIDE | 15–9 | 6–2 | HOT 1 | 0.29 |
| s10 ROTATE vs R | 11–13 | 4–4 | R 2 / ROTATE 1 | 1.00 |

The seed-control set itself has three one-sided 3–0 sweeps, so a sweep is
noise, and the effective n is 8 rows, not 24 pairs. On the row sign test
nothing reaches 0.05; the only set that comes close is **s01 C9 vs P (7–1
rows, p 0.07)**. Everything else in the bracket is flat.

### State of the line after ten sets

- Confirmed at the floor: nothing. Candidate: C9 > P (rows exist vs not).
- Refuted by direct test: ROTATE hurts (s10). Inert within the floor: scale
  (HOT), collisions (COLLIDE), content (R), distillation (INIT).
- **Next: replicate C9 vs P at fresh seeds (s11).** If it holds at ≥ 7–1
  rows again, "rows must exist" is the one result and G2 asks *how* an
  unused-content row helps (the adapter self-attn side effect, §9). If it
  does not, the bracket is flat and findings §14's original reading (rows
  inert, ship `japanese text`) stands — the eye-vs-OCR disagreement was
  grader noise on one grid.
- Power fix for any further set: **more prompts (rows), not more seeds.**
