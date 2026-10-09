# idea2 — the rows' Jacobian lens (2026-10-07)

Borrowed from JLD (Saini, Adsumilli, Bovik, arXiv:2610.05967, "Perceptual
Distance Through A Jacobian Lens"). The paper weights a frozen encoder's
block-1 patch features by how much the encoder's output moves when they move:
M = E[JᵀJ], J = ∂z/∂hₜ, keep the top-k eigenvectors as a fixed projection
(the lens). The transferable part is the construction, not the metric:
**an early representation, read through the sensitivity of a late output,
fitted label-free from random output probes.**

Here the early representation is the row and the late output is the DiT's
prediction. Nothing below has been run.

## What carries over from the paper

- **The fit needs no target.** A Gaussian probe v ~ N(0, I) on the output,
  backpropagated once, gives g = Jᵀv at every input position; E_v[g gᵀ] = JᵀJ.
  Averaging over inputs is an unbiased estimate of M (their Prop. 2).
- **Which directions, not how many.** On DINOv2-S block 1: all 384 raw
  directions 0.772 mean SRCC, 64 random directions 0.772, 64 PCA 0.851, the 64
  lens directions 0.905. Euclidean distance on the raw features counts the
  directions the output ignores at full weight.
- **The sample is a weak knob, the output target a strong one.** 10 fit images
  already overlap the 100-image lens 0.877; a one-image lens overlaps 0.72,
  close to two fits of the same image (0.69) — mostly estimation noise. Moving
  the target from the final output to the next block drops the overlap to 0.510.
- **It is a linearisation.** The lens is blind to 320 of 384 directions; the
  paper warns that optimising against it exploits that null space.

## The mapping

| JLD | here |
|---|---|
| hₜ, block-1 token at patch t (384) | a row occurrence in the caption (`ext_embed`, 1024) |
| z, final CLS | the DiT's prediction at (x_σ, c_JA; rows), latent cells |
| E over images and patches | E over items, σ in a band, row occurrences of a family |
| probe v on z | v on the latent cells, optionally masked |

Within one σ the subspace is the same for v and x̂₀ (x̂₀ = x_σ − σ·v, so the
Jacobians differ by the factor σ); across a band the choice sets how the
σ's are weighted.

**Not the FM loss.** The FM gradient on the rows is Jᵀ(v_θ − v_target): first
order, needs the target, says which way to move. M = E[JᵀJ] is second order,
needs no target, and says which row directions the image responds to at all.
For an MSE loss it is the Gauss–Newton curvature of FM. `probe_geom` /
`probe_pres` / `probe_scene` read first moments (gradients of a loss); this
reads the second moment of the model itself.

## The in-box / out-box split

Mask the probe: v on the text box (dilated as `probe_pres`'s `out_mask`, 2
latent cells) gives M_in, v on the rest gives M_out. The cells are disjoint, so
M_in + M_out = M.

The generalised eigenproblem **M_in u = λ M_out u** ranks row directions by
how much they move the text against how much they move the page. Its top
directions are the subspace L_pres is trying to reach with a penalty.

Why this matters for the line now: L_pres trades text for page. h32 p10 moved
the page (`en_tok_out` +0.039, `en_match` +0.097) and paid in text (CER +0.06,
`le2` 13 → 5); p10c09's σ ≤ 0.9 cap got the text back by dropping σ 0.95 —
the trade sits on the σ axis. A row update projected onto the high-λ subspace
would hold the page by construction, not by λ, and M fitted per σ band would
show where the trade comes from.

**What it cannot replace.** M is local sensitivity around the current rows. It
says which updates leave the page alone; it does not see a page gap the rows
already have against the EN render — that gap is what L_pres measures against
the EN teacher. The two are complementary: the subspace constrains the move,
L_pres corrects the standing gap.

## Other uses

- **A metric on rows.** ‖U_kᵀ(row_a − row_b)‖ — how differently two rows render
  — in place of the Euclidean norms `structure_candidate.md` reads with. The
  share of a trained offset that lies inside the lens says how much of it the
  image sees (the paper's distortions keep ~10 % of their energy in the lens;
  an isotropic move would keep k/d).
- **Reading existing moves.** Where Δ(p10 − plain)'s shared vector (29 % of the
  move) and the stick sit: high M_out, high M_in, or the null space.

## Risks

- **The operating point moves.** Cold-reseeded rows travel far (offset |~200|
  against the pack spike's |~160|). M at f0's start may not be M at the end:
  check the overlap of M fitted at start rows and at `sent_kanji_f0`'s rows
  before relying on a fixed lens.
- **σ dependence.** The text and the page settle at different σ; fit per band,
  never pooled across the whole schedule.
- **Sharing.** One M per family (kana / kanji), as the paper shares one lens
  over patches. A per-row M is mostly noise at any budget the line can afford.
- **Cross-row leakage.** `grad_identity` found that changing glyph k moves the
  other rows' regions almost as much as its own. J of a row occurrence covers
  the whole box, so M_in mixes "my glyph" with "my neighbours'"; per-glyph
  boxes (idea.md § 1) would separate them.
- **Null-space exploitation.** If a training arm projects updates or adds a
  lens loss, keep the data term beside it.

## Probe 0 — rows held, no step (`probes/probe_jl.py`)

Same form as `probe_geom` / `probe_pres`: f0's start rows
(`seed_fixed_1005_stick080@punct`), h32's build (5 677 batch items + 400 held).

1. **Fit.** Per band (σ 0.5–0.7 / 0.8 / 0.9 / 0.95) and per family: M_in, M_out
   from P = 8 probes per draw over ~100 items, one forward per draw, the probes
   on retained graphs. Two independent fits (disjoint items, fresh probes).
2. **Stability.** Subspace overlap ‖UᵀŨ‖²_F / k between the two fits, k ∈
   {16, 64}.
3. **Spectrum.** Trace share of the top-k of M_in and of M_out; the generalised
   eigenvalues λ against the trace ratio tr(M_in) / tr(M_out).
4. **Reading.** The share of h32 `plain`'s and `p10`'s trained offsets, of
   Δ(p10 − plain)'s shared vector and of the stick that lies in the top-k of
   M_in, of M_out, and of the high-λ subspace.
5. **Drift.** Refit band 0.8 at `sent_kanji_f0`'s rows; overlap with the start.

**Stop if** the two fits overlap below 0.8 at k 16 (too noisy at this budget),
or no direction's λ exceeds 2× the trace ratio (no text-only subspace to
project onto).

**If it passes**, the training arm: h32's recipe with each row step projected
onto the top-k high-λ subspace of its band, against p10c09 on the same draws,
read on the ruler's 84 live strings.

**Ran 10-07 — stopped** (`reports/probe_jl_2026_10_07.md`): A / B overlap
0.21–0.50 at k 16 (pair-weighted, kana 0.69–0.80 and still climbing; kanji
0.30–0.55); a cross-fit text-only λ only at σ 0.5–0.7, ~1–2 at L_pres's
σ 0.8–0.9 — no direction there moves the box without the page, so the
projection arm is not run. Read in fp32: bf16's per-position row gradient
is at cos 0.3–0.6 to fp32's.

## The page in PE's terms

Probe 0's M_out sits on the prediction's latent cells: every channel of
every out-box cell counts, a tone drift or a texture jitter as much as a
character gone. λ ~1 at σ 0.8–0.9 says no row direction moves the box
without moving the page's *cells*. The ruler scores the page on what the
scene is (`en_match`), not on its cells — and JLD's point is the late
output's meaning (the CLS), not raw pixels.

The probe: x̂₀ = x_σ − σ·v from the prediction, the VAE decode, the text
box masked (or cropped out), PE's features z of the rest; Gaussian probes
on z give M_scene = E[J_zᵀ J_z], J_z = ∂z / ∂row. The box side stays
latent (M_in). M_in u = λ M_scene u, cross-fit as probe 0: a direction
with λ ≫ r at σ 0.8–0.9 moves the text and leaves what the page *is* —
the subspace the projection arm needs, which the cell metric did not
show.

- **Blur.** At σ 0.8–0.9 the one-step x̂₀ is soft; PE on a soft page may
  read the blur, and J_z then measures that. Check first: PE's features of
  x̂₀ at σ 0.8 against the clean image's, per scene.
- **Cost.** A VAE decode and PE in the backward, fp32, per probe — several
  times probe 0's 3.85 s / item; not measured.
- **The mask.** A grey box and a crop read different things; the EN page
  (`probe_pres`'s teacher) is the reference either way.
- The training-side twin of it is L_pres on PE features against the EN
  render, not on cells.

**Ran 10-07/08 — stopped** (`reports/probe_jl_pe_2026_10_08.md`): PE
reads the scene from x̂₀ at σ 0.8 (top-1 0.8), half of it at 0.9; u is
local — it turns under a 1e-3 white-noise δ, holds (cos 0.72–0.96) along
the rows' own moves at that size, and needs the VAE in fp32. M_pe is less
stable than the cells' M_out (A / B 0.57–0.58 pair-weighted at σ 0.8–0.9,
kana), its cross-fit λ sits where the cells' did (~2 r at 0.8, none at
0.9), its high-λ subspace does not repeat (0.02–0.04), and it predicts
the page reads no better than the move's size. Δ(pres − f0)'s shared
vector sits in its top 16 (28× isotropic). No projection arm.
