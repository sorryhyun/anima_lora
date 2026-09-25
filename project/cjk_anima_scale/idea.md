# idea — a gradient bank in place of trained arms (2026-09-25, from an outside review)

Not planned, not scheduled. Written down so the next person who wants to
price a data mix without a 50-minute arm starts here instead of at
`‖ḡ‖·coh`. Source: a gpt-6-astra review of this line's reports on
2026-09-25; the claims below are its, checked against the code where a
line number is given.

## What the current price is, and what it lacks

`reports/*` price a recipe by `‖ḡ‖·coh` (`cjk_scale/conflict.py:298`). With
`‖ḡ‖` = mean per-draw norm and `coh` = ‖Σg‖ / Σ‖g‖, the product is
‖(1/n) Σ g‖ — **the norm of the mean gradient, conditional on the draws
that touched the row**. Sound as a movement-per-draw number; not a utility,
because it lacks:

- **direction** — a coherent gradient that walks a row off identity is
  priced the same as one that buys a hit (`grid_string` at 0.7–0.9: 1.8 ×
  the signal, cos 0.27 to the low band, Δ·−ḡ 0.06 — `grid_box` report § 4);
- **exposure** — the price is per touching draw; a row's actual gradient
  per step is price × P(row in the batch), and grid items touch many rows
  per draw (`b30` 0305: `grid_string` 1 664 items per piece row vs
  `scene_piece` 312 — `conflict` addendum § 6);
- **bias** — a finite-sample mean norm is biased up, and `coh` 0.1–0.3
  does not mean 70–90 % of the draw is useless noise;
- **the optimizer** — Adam's second moment and momentum, and the
  regularizer, reshape the update; a per-kind gradient scale is mostly
  cancelled by Adam (a per-kind lr needs a real optimizer group; `raw` is
  one Parameter today — `cjk_scale/rows.py`).

Split-half `half` is a reliability, not a significance threshold: with
independent equal-variance halves the full-mean reliability is
≈ 2h / (1 + h), and the singles' cross-band cosine ceiling comes out near
0.5, not the ≈ 0.8 the report disattenuated to. The singles' "same
direction" verdict is the shaky one; the pieces' (cos 0.62, half ≈ 0.7)
stands.

## The bank

One probe, on GPU (training-free is not GPU-free — the frozen DiT still
runs backward), whose output is reusable on CPU for every mix / budget
question after it. Per cell c = recipe × px stratum × σ bin, per touched
row:

1. the gradient vector (or a fixed random projection of it, if 2 274 rows
   × dim is too much — it is not for a 72-row run);
2. the unconditional mean and diagonal second moment — a draw that does not
   touch the row contributes **0**, so exposure is inside the number;
3. item / scene id, the σ actually drawn, the latency per item;
4. all of the above at **two table points**: the seed and a moved table
   (e.g. the μ 0 / lr 1e-3 joint), so the gradient's drift with position
   is observable.

`conflict.py` stores scalar summaries only (`conflict.py:346`); nothing in
the existing `rows.json` can be recombined into a new mix's direction or
noise scale, and `trained.pt` carries no optimizer moments. The bank is
the missing artifact.

## What to compute from it

**Validation influence (the price that points somewhere).** A dev set of
product-shaped composites (native prompts, small px, mixed strings —
separate from the acceptance strings, `product_criteria.md` § Dev vs
acceptance) with the right text rendered into a clean latent; v_s = the FM
loss gradient on the rows for string s. For a candidate batch c, the
expected Adam-shaped update δ_c (from the bank's mean and second moment
and the current moments), then

    I[c, s] = − v_sᵀ δ_c

is the predicted validation-loss change on s. Use I / cost per draw, the
sign per string, and its uncertainty (bootstrap over scene ids, not over
draws — repeated noise draws of one scene and the cells of one grid are
not independent samples). This is LESS-style optimizer-aware influence
(arXiv 2402.04333), not gradient norm.

**Page preservation as a separate constraint.** At the seed the squared
outside-box discrepancy to the EN-ref velocity has zero gradient, so
alignment reads nothing there; read the **outside-box JVP norm** (or a
small finite difference) of a candidate update instead.

**Budget from the noise scale, not steps/row.** With P the Adam
preconditioner held fixed, B_noise ≈ tr(P Σ Pᵀ) / ‖P m‖² per cell says how
much another draw of that cell still sharpens the direction (arXiv
1812.06162). It does not give a hit count or a total step budget; ESS
(Σw)² / Σw² of any importance weights is a separate number and does not
stand in for scene diversity.

**Calibrate before trusting.** One-step denoising influence cannot see the
sampling trajectory, glyph count, or a reader's binary hit. The surrogate
earns a place only if it ranks the existing good / bad tables (the ten
`b30` arms, `micro_warm_0923`, `grid_box 0 / 1`) the way the rulers did.
If it does not, do not use it to choose shares. Linearizing at the seed
and integrating to drift ≈ 1.0 is not trustworthy — read the two bank
points, then confirm a shortlisted candidate with a short real trajectory.

**Mix design that fits the bank.** Not DoReMi (a high FM loss here may be
synthetic-data difficulty, not learnable signal) and not a RegMix
regression over the ten arms (they were not a mix factorial — μ, lr, chain
/ joint moved together). A 2 × 2 over scene-vs-grid share × small-vs-medium
px, fixed optimizer, two checkpoints, read for rank stability. Do not run a
bandit that kills arms before their drift threshold.

## The one cheap discriminating read for scheduling (Q3)

Matched-item σ sweep: 8 representative pieces, scene / grid × two px strata,
the **same item and the same noise** at σ ≈ 0.4 / 0.6 / 0.8, at the seed and
at a moved table — validation influence, direction, noise, page JVP.
4 strata × 32 items × 3 σ × 2 noises × 2 tables = 1 536 backwards (the 2 400-item
direction probe took 15.6 min, so ~10 min of compute plus loading). Reads:

- same direction and validation sign across σ, only norm / noise differ →
  loss weighting or importance sampling over the window, band is not the lever;
- validation sign differs by σ at matched norm → the band choice itself matters;
- the preferred σ moves with px → per-item windows (a row's window is
  kind × the item's px / layout, not the row's; one latent has one σ, so a
  mixed-row item cannot give each row its own — `joint.py` attaches the
  source stage's band and does not resolve this);
- the preferred σ moves seed → moved table → a curriculum candidate; the
  order's causal effect then needs a matched-budget order comparison.

Two things to state before any of it: the in-band draw is not uniform but
the trainer's sigmoid-normal affine-mapped into the band
(`library/runtime/noise.py:204`), so the target density p(σ | x) has to be
written down; and changing the sampler without a p/q weight changes the
objective, not just its variance. The variance-optimal proposal at fixed P
is q ∝ p · √E‖P g(x, σ)‖², which is not "draw where the norm is largest".
