# suggestions — ideas worth a probe, not yet run (2026-09-22)

Index: [`README.md`](README.md). Ideas that came out of reading the
`step1_0921z` table (the Z8 cold table, 1 900 rows) and a second opinion
(Codex, `gpt-6-astra`, 2026-09-22, prompted with the measured geometry and
`findings_seed.md`). Each entry: what it claims, which recorded fact it has to
get past, the cheapest probe, and the odds as judged today. An idea leaves this
file when its probe runs — the read goes to `reports/`, the verdict to
`findings.md`, and the entry here becomes one line under *Done* at the bottom.
The forward plan stays [`plan.md`](plan.md); this file never schedules anything.

Numbers cited: `step1_0921z` read (`rows_step1_0921z_s152k/`, kanji 15/48,
multi-glyph 3/144 as strings; `eval_build.py` beside it), merged-table
geometry (`table_geometry.py`, 2026-09-22), and the exposure audit below.

## 0. What the exposure audit found (CPU, 2026-09-22) — the fact the rest sits on

| | `step1_0921` (374 rows) | `step1_0921z` (1 900 rows) |
|---|---|---|
| draws per row over the run | 963 (scene 133 + grid 830) | 1 001 (scene 160 + grid 841) |
| P(row in a batch of 4) | 3.9 % | 0.66 % |
| **median revisit interval** | **26 steps** | **152 steps** |
| Adam `v` left between visits (0.99^T) | 0.77 | 0.22 |
| ‖row‖ peak → end | 298 → 282 (−5 %) | 199 → 149 (**−25 %**) |
| spearman ‖row‖ ~ draws (draws vary ± 5 %) | 0.17 | **0.46** |
| spearman ‖row‖ ~ glyph px | 0.39 | 0.16 |
| box short side, median, scene / grid | 50 / 97 px | 40 / 72 px |
| eval hit vs miss (z) | — | px 96 vs 63; ‖row‖ 164 vs 142; draws 1 010 vs 999 |

Exposure is equal — draws per row and hit-vs-miss draws are the same. What
differs is the **revisit interval**: the pull `μ · mean_r ‖f_r‖²`
(`train/trainables.py:358`) is on every row every step, and under Adam its
normalised step grows while `v` decays between a row's visits. At 374 rows the
next FM gradient arrives 26 steps later; at 1 900 rows, 152 steps later. The
z table's norm falling 25 % from its peak and its norm ranking following a
± 5 % draw spread (not saturated — every draw is pushing against the pull)
both fit that; `step1_0921`'s norm is draw-independent (saturated) and follows
glyph size instead. Multi-glyph columns also shrink the box (72 vs 97 px in
the grid half), and hit rows are 1.5 × the px of misses. **The low norm reads
as regularisation that does not scale with table size plus smaller glyphs, not
as harder rows.**

## Ideas

### 1. Norm as the cause: split the gain into `a·m̂` and residual — HIGH odds of a useful read, MED that misses come back

Uniform `--delta_scale` conflates trigger strength (along m̂) with identity
strength (residual). On the 12 hit and 12 miss rows of the z read, sweep the
m̂ coefficient and the residual gain separately (×1, ×1.5, ×2), render the
bare template, read exact / repeats / scene. Has to get past: fact 2 of
`findings_seed.md` (norm predicts a hit, ρ +0.21) may be effect not cause.
Probe: ≈ 15 min GPU on the existing table, no training. If the residual gain
alone recovers misses, the z table is rescued without retraining and the next
table's recipe changes at zero cost. Falsified if neither axis moves the
misses without breaking the hits or the scene.

### 2. Make the pull scale with the table: per-visit pull or μ ∝ 1/N — MED odds, the mechanism arm for § 0

Two forms, one variable each: (a) apply the pull only to rows present in the
batch (per-visit, so a row's pull time equals its FM time whatever N is); (b)
keep the always-on pull but set μ · 374 / 1 900 ≈ 2e-4. Same z data, 15 k
steps, compare ‖row‖ at 15 k against the z log (193 at 15 175) and the peak
height. ≈ 1.8 h GPU. If the norm tracks `step1_0921`'s curve the 1 100-row
table trains on the fixed recipe; if not, § 0's reading is wrong and glyph
size is the lever. Note the sentence run's anchor `μ · mean_r ‖f − f₀‖²` has
the same 1/N form at 2 271 rows.

### 3. Mean-init on the *effective* embedding, not the delta — MED for picture safety, LOW for rendering

`plan.md` § 1b adds a class-mean Δ on top of the untrained pack row `e_r`,
leaving `e_r`'s own variation in place. Alternative: `Δ_r = mean_c(e + Δ) −
e_r`, so the T5-side row *is* the class centroid. Compare global mean, class
mean and m̂-only targets, at the mean's own norm (do not scale a mean to a
row's norm). Has to get past: transplant fact 3 (m̂ alone keeps the scene) was
measured on trained ids, not on unseen `e_r`; the Qwen-side stream still
differs. Probe: 1 / 2 / 4 unknown pieces in the same scene prompts, both
inits, `native` en cos + IoU, ≈ 15 min. Folds into § 1b's probe as one more
condition. Rejected if scene preservation is no better than the delta mean.

### 4. Repair common pieces before adding rare ones — MED, a budget question for the user

"85 % covered" is token coverage; readable coverage with multi-glyph at 3/144
is far lower. Repairing the 1 317 pieces already in the table (they cover the
frequent lines) may buy more readable dialogue than 1 100 rows at corpus
counts 2 – 7. Probe: matched cohorts at 20 / 40 / 80 steps per row from the
z table (optimizer state kept), fresh composites, scored on exact strings and
incremental *readable* lines, ≈ 30 min per cohort. Does not amortise
identity; it stops spending on rows that are not moving. Conflicts with the
coverage-first goal, so it is a decision, not a default.

### 5. Two independently captioned bubbles per scene — MED for compute, conditional on layout

Two units in two bubbles of one real scene, each with its own clause, keeps
composite exposure per row (unlike grid cells) and halves DiT passes per
draw. Has to get past: fact 7's closed grid-share ceiling (grid costs native)
— a second bubble is still a scene. Probe: two-bubble vs single-bubble at
equal GPU minutes on ≈ 100 rows, judged as correct rows per minute, not
updates per minute, plus the singles read. ≈ 30 min. Rejected if native drops
or correct rows per minute do not rise.

### 6. One radical / stem neighbour's residual at λ = 0.15 – 0.3 as warm start — LOW–MED for fewer steps, LOW for rendering

Fact 1: radical-sharing kanji are the nearest neighbours (cos 0.25 – 0.33), and
the IDS-*sum* failure (0/16) does not test a single donor as an optimisation
start. But at cos 0.3 the optimal scalar projection explains 9 % of the
residual's squared norm — an edge, not a shortcut. Init `mean_c Δ + λ ·
f_donor`, donor chosen by IDS structure or shared stem without looking at the
target's trained row. **Random donor matched for norm and donor accuracy is
the control**; without it fact 4 (`--pin_dir` saved no steps) makes any gain
unreadable. 12 – 24 targets, matched exposure, ≈ 30 min. Rejected if early
exact reads do not beat the random donor, or the donor glyph persists.

### 7. Feature → residual regression (Qwen embedding / pack row / positional IDS) — LOW for identity, MED for predicting the safe amplitude

Pack rows are a ridge transform of Qwen embeddings (holdout cos 0.71), so the
two are one feature, not two. Predict the m̂ coefficient and the residual
separately; encode IDS as component *position, multiplicity, layout*, not
sums. CPU first: cross-validate within-family and held-out-family against
class means and shuffled labels; only a survivor gets ≈ 15 min of rendering.
Rejected if the gain vanishes after norm matching.

### 8. Narrow the "categorical address" reading — HIGH that it clarifies the mechanism, no direct win

Fact 5 records failed transfers, not impossibility. Same-glyph residuals from
two tables sit at cos ≈ 0.1 and both render — if they converge *after* the
adapter or in the DiT's response at σ ≈ 0.8 (against different-glyph
controls), addresses are many-to-one and regressing toward their mean is the
wrong target, while function-space predictors become worth a look. ≈ 10 min.
Absence strengthens the current pessimism.

### 9. PCA of residuals as a compression test only — LOW

Fit PCA on trained residuals (m̂ out), reconstruct *known* rows at rising rank
against random subspaces of equal rank, render. ≈ 10 min. A basis gives no
coordinates for an unseen glyph; this only tells whether "variance retained"
is the right objective for any later basis-coordinate training.

## Done

(none yet)
