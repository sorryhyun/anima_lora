# sigma_lowres — mechanism hypothesis: the normalized two-term account

Status: **living account, v2.1 (2026-07-27).** v1 (the two-term account) was
pre-registered 2026-07-24, survived its kill switch the same day, and has
since absorbed four refinements (route-uniform residual shape, J-side
amplitude/Floor, gradient-norm denominator, and the v2.1 Floor
decomposition Floor_e = RoPE_e + Resid_e from the G10 origin-side probe). This file states the current
account *cleanly* — every claim cites a grounding **G1–G8 in
`groundings.md`**, which holds the test designs, pre-registration
provenance, run pointers, tables, and falsifiers. Nothing here is
un-grounded; nothing there is un-cited.

## The claim

The demotion gap measured by the σ-probe (`bench/run_sigma_probe.py`;
gap = split-half floor cosine − cross-grid cosine, i.e. **cosine units:
scale-free, a mismatch *fraction*, SEM ~0.02**) decomposes as:

```
gap_e(σ)  ≈  S1_e(σ)           +  Floor_e
              └ input branch ┘      └ graph term ┘

S1_e(σ)  ≈  A_e · m(σ) / G(σ)^p          (p ∈ [1, 2], schematic)
```

Four objects, each independently measured:

| term | what it is | σ-shape | lives in | grounded by |
|---|---|---|---|---|
| `m(σ)` | universal residual-mismatch shape — how much the model's mean prediction error `r̄` differs across grids | monotone ↓ (0.89 → 0.36) | prediction side, **route-uniform** | G7 |
| `G(σ)` | total gradient scale ‖g‖ — the cosine's denominator | **U-shaped** (2.6 → 0.4 @ σ≈0.3 → 9.3) | checkpoint anatomy | G8 |
| `A_e` | route amplitude — how strongly the universal r-mismatch transmits into gradient mismatch | σ-independent | **J (graph)**; governed by route **ratio**, not target capacity | G5, G7, G9 |
| `Floor_e` | σ-independent graph term (0 / 0.06–0.13 / ~0.3 for 896/768/512 from 1024); **decomposes as RoPE_e + Resid_e** — a coordinate-system share erasable by PI position alignment (≈ all of 768's, ~30% of 512's) plus a non-PE graph residue | flat | **J (graph)**, early-block | G1, G2, G4, G6, **G10** |

Term by term:

- **m(σ) — the universal numerator.** The mean FM residual
  `r̄ = E_ε[v̂ − (ε − x)]` decorrelates across grids with a strong σ-shape
  that is the *same for every route* (±0.02 across routes spanning Floors
  0 → 0.3; G7). All route identity therefore lives in J, not in the
  prediction. m decays smoothly — Wiener-like posterior shrinkage, **no
  hard gate** at the spectral crossover, which is why gap curves read as
  "collapsed" only where S1 sinks below the instrument's reenc band
  (±0.04), not at the RAPSD σ\* ≈ 0.14.
- **G(σ) — the denominator.** Because gap is a cosine, it measures the
  mismatch *fraction*, not its magnitude. Total gradient norm is U-shaped:
  at σ ≳ 0.85 the model is pulling composition out of the prior and the
  residual is image-scale ("the high-σ residual *is* the image"); at
  σ → 0 the residual is dominated by irreducible ε plus low-frequency
  reconstruction both grids share (Anima latents are HF-quiet, RAPSD
  P(f) < 1 above f ≈ 0.16). In between the model is merely refining and
  ‖g‖ bottoms out — so a monotone absolute mismatch becomes an
  **interior-peaked** cosine curve (peak σ ≈ 0.2–0.44, dip at the lowest
  bin), which is exactly the measured Phase-0 shape (G8). Interpretive
  cross-reference: the σ ≳ 0.85 "plan-writing window" is independently
  located by the cross-attn drive measurement (`docs/inference/xattn_boost.md`).
- **A_e — the route amplitude, in J.** The transmission strength of the
  universal r-mismatch scales with route severity (0.80 is harsher than
  0.875 → S1 stays above the band to higher σ), which retrodicts all
  measured crossovers: 1024→896 σ\* ≈ 0.5; 1280→1024 σ\* ≈ 0.75;
  896→768 moderate S1 atop a Floor already > band → never floors (G5).
  The governor is **ratio, not target capacity** (G9, iso-severity
  probe): 1280→1120 (ratio 0.875 matched to 1024→896, target capacity
  1.6× larger) reproduces 1024→896's curve and σ\* ≈ 0.5, while the
  same-native 1280→1024 (ratio 0.80) floors later. With Floor_1120 ≈ 0
  this completes the hybrid's division of labor: **ratio sets A_e;
  absolute target capacity sets Floor_e.**
- **Floor_e — the graph term.** σ-independent, grows with demotion
  severity, and **structurally exempt from noise masking**, for two
  reasons that make the account principled rather than curve-fitted:
  1. **Expected gradients are never noise-masked.** "Noise power exceeds
     signal power in band f" is a statement about a *single sample* of
     z_σ. The trained quantity is E_ε[∇θL] — an expectation that averages
     noise out and retains signal at any SNR, merely attenuated.
     Sufficiency of demotion for the *input* licenses inference
     equivalence, never training-dynamics equivalence.
  2. **The target carries the clean image at unit amplitude at every σ.**
     The FM target is v = ε − x; the input gets noise-masked as σ→1 but x
     sits in the target at coefficient 1 even at σ = 0.94. The gradient
     g = Jᵀr then differs across grids through J itself: attention
     softmax over N, RoPE phase density, seq-dependent normalization.
  Measured properties: graph-dominated, not content (G1); real at σ=1
  where the input is pure ε by construction (G2); localized in **depth**
  (early blocks ~3×, all module types uniformly) (G4); **dissociated from
  the checkpoint's forward prior** — prior distances are flat across
  routes whose Floors span 0 → 0.3, so the Floor's route-ordering lives
  in the J factor, not in x̂_prior (G6). Floor_e is also σ-independent
  *in cosine units* — a transmission fraction of whatever residual
  exists — which is why it reads as a flat plateau rather than being
  divided by G like S1.

  **Decomposition (v2.1, G10, qualified by G11): `Floor_e = RoPE_e +
  Resid_e`.** An origin-side intervention (PI-stretched fractional RoPE
  positions on the demoted grid, matching native relative phase geometry
  exactly) removes the large majority of 768's Floor at the σ=1 endpoint
  (+0.080 → −0.001 absolute; paired residual over reenc ≈ +0.05–0.07
  across the two runs) and ~30% of 512's (+0.320 → +0.224). So the
  mild-route Floor is mostly a **coordinate-system artifact** (RoPE
  phase-density mismatch), while the harsh-route bulk is the genuine
  graph residue (softmax-over-N / seq-normalization / capacity). Three
  corollaries: (1) G4's "RoPE refuted" was a *landing-side* inference — a
  PE-originated perturbation propagates through the block and lands
  uniformly across module types, so landing uniformity never localized
  origin (the depth profile is a property of both components); (2) the
  capacity governor attaches to **Resid_e**, not to Floor_e as a whole;
  (3) **RoPE_e is a noise-regime artifact only (G11)** — with content in
  the input the stretched forward is off-manifold and *adds* error
  (768pi worse than 768 through σ 0.56–0.81, better only at σ ≥ 0.94),
  so the removal does not transfer into the training window and is a
  mechanism finding, not a lever. In the commutator reading, PI alignment
  zeroes the RoPE component of [D, J] exactly — but only where the
  residual it transmits is position-geometry-dominated.

The **latent-space quirk** (HF-quiet Qwen-VAE latents, non-scale-equivariant
encoder) shapes m only — exonerated as the Floor's cause by gap_reenc ≈ 0
(encoder round-trip harmless) and the σ = 0.94 persistence (G2).

## What the account retrodicts

1. **σ\* ≈ 0.5, not the RAPSD 0.14** — the spectral prediction is
   numerator-only; the measured crossover is where A·m/G^p sinks below the
   reenc band, a property of the *ratio* (and the denominator recovers
   above σ ≈ 0.44, which helps push curves into the band). The 3.5×
   discrepancy is expected, not anomalous.
2. **Interior peak + low-σ dip** of every gap curve (G8: renormalizing by
   G restores a low-σ-maximal, monotone-into-noise numerator; the dip at
   σ = 0.06 flips to the maximum in every arm of both runs).
3. **Tier ordering at every σ-bin** — A_e and Floor_e are both
   severity-ordered.
4. **High-σ persistence** — 768 ≈ 0.06–0.12 and 512 ≈ 0.3 at σ = 0.94
   where the latent is ~97% noise: that's Floor alone (G2). 896's plateau
   0.03–0.05 sits inside instrument noise → "safe".
5. **Phase-1a FAIL** — 896→768's Floor is outside the band on its own, so
   no σ-gate can rescue the route. (G10 briefly reopened a PI-aligned
   variant; **G11 closed it same-day** — the stretch is off-manifold with
   content in the input, and S1(1024→768) is fatal regardless. The FAIL
   stands for every 768 route.)
6. **1280→1024** — bigger A (harsher ratio) with Floor ≈ 0 → floors
   later (σ\* ≈ 0.75) but cleanly (G5); breaks any pure-capacity
   route-ordering story, hence the hybrid split of S1 (severity) vs Floor
   (graph approximation quality).
7. **Route-uniform σ-shape with route-ordered floors** (G7) — impossible
   if the σ\* ordering were prediction-side; forced if m is universal and
   A_e, Floor_e live in J.

Corollary: **1024→896 safety is an empirical smoothness statement about
Anima's function across nearby token counts, not an information
statement** — there was never a reason to expect a universal ratio
invariant, and "Floor = how well the coarse graph approximates the fine
graph's computation" is the natural reading.

## What the account does NOT claim (open edges)

- **The formula is structural, not a quantitative fit.** m(σ) is measured
  forward-side in rel-L2 units; the exponent p is fixed only to [1, 2] by
  cosine geometry (2 in the small-orthogonal-mismatch limit); A_e absorbs
  units. G8 shows the *shape* facts (dip removal, gap ∝ 1/G tracking),
  not a fitted curve, and the renormalized proxy is meaningless at
  σ ≳ 0.8 where (gap − Floor) ≈ 0 ± SEM gets multiplied by G² ≈ 86.
- **A_e's governor — RESOLVED (2026-07-26, G9): A ~ ratio.** The
  iso-severity 1280→1120 probe discriminated it (ratio-matched routes
  coincide despite 1.6× capacity difference). Residual: the coarse bins
  localize σ\*(1280→1120) only to (0.375, 0.625); a
  `--sigma_window 0.375,0.625` run would pin it. Mechanism value only.
- **RoPE_e/Resid_e — route question RESOLVED (G11), mechanism edges
  remain.** The σ-resolved gate closed the PI-768 route (off-manifold
  in-window + fatal S1 at ratio 0.75). Still open, mechanism-value only:
  why the pi stretch is content-hostile (candidate: attention content
  lobes calibrated to integer phase spacing per grid — would predict the
  penalty grows with content share, i.e. toward low σ, as measured); the
  1/40 scaled-rope outlier's grid dims.
- **Floor checkpoint-dependence through J** — untested. The carving test
  (fine-tune on the 1280 cache via `bench/prep_1280_probe.py`, re-probe;
  if the 1280→1024 gap *opens*, the safety map is
  operating-point-specific) is the discriminator. The prior-side form of
  checkpoint-dependence ("base Anima never learned 1280") is already
  refuted (G6).
- **G(σ)'s anatomy is interpretive** — the plan-window / refinement /
  ε-floor reading of the U-shape is consistent with xattn_boost's
  independent measurement but nothing measures plan-commitment and the
  gap in the same run.
- **Single operating point** — all gradient probes ran `anima_soup_sincos`
  trained at native tiers; a mixed-res-trained adapter might equalize its
  own gradients (Q3's warning stands).

## Paper framing (Q6)

Not just "a measured counterexample to spectral sufficiency" but *why it
must be one*: the target branch of the FM objective is structurally exempt
from noise masking (the two Floor reasons above), and the observable is a
cosine — so spectral (numerator-only, magnitude-blind) reasoning
mispredicts both the crossover location and the curve shape. The
safety-boundary map over (route, σ) with the J-side Floor is the stronger
paper shape.
