# AsymFlow asymmetric parameterization — assessed, not worth reviving for Anima

This records why AsymFlow's headline contribution — the rank-asymmetric velocity
parameterization `u_A = Pε − x_0` — was never ported into Anima, and whether it is
worth reviving. Short version: the parameterization is a remedy for a high-dimensional
/ pixel-space bottleneck that Anima, a compressed-latent DiT, does not have. The
paper says this in its own introduction, and Anima's numbers confirm it quantitatively:
the per-patch noise dimension is `D_patch = 64` against a `2048`-wide residual stream
(ratio 1/32). On top of that it is a base-model parameterization change, not a
LoRA-compatible one, so it sits outside this repo's fine-tuning scope (Tier 3). Keep
it shelved. A cheap necessary-condition probe is defined below if anyone wants to
close it empirically rather than by argument.

Method reference: Chen, Ackermann, Kim, Wetzstein, Guibas, *Asymmetric Flow Models*
(Stanford, arXiv:2605.12964, 13 May 2026). PDF in repo root (`2605.12964v1.pdf`).

## First: don't conflate the paper's two ideas

AsymFlow is two separable contributions that we have historically blurred together:

1. §4 — the asymmetric parameterization (this doc). A new training target for
   high-dimensional flow models. The benefit is an optimization/capacity effect that
   exists when the per-patch noise dimension `D` is large relative to network width.
2. §5 — latent→pixel finetuning, of which §5.2's variance-reduced loss is a
   sub-component. This is the only part of the paper we ever implemented (as
   Anima's VR-loss, with an FEI low-pass standing in for the paper's Procrustes lift).

The VR-loss line (the §5.2 sub-component) was benched and is MARGINAL on Anima —
the corrected v1.0 run came back MARGINAL (high-band ρ²≈0.23, ~12% loss-variance
reduction; the VR-loss headroom rig itself is no longer in-tree).
That result says nothing about §4. The asymmetric parameterization has
never been tested here. This doc evaluates only §4.

## What the asymmetric parameterization is

Standard flow matching (paper Eq 1, linear schedule `α_t = 1−t`, `σ_t = t`):

```
u := (x_t − x_0) / σ_t = ε − x_0          # full-rank velocity target (Anima uses this)
```

Predicting `u` means the network must reproduce the full noise component `ε`. AsymFlow
restricts only the noise term to a rank-`r` subspace via an orthonormal basis `A ∈ ℝ^{D×r}`,
projector `P = AAᵀ` (Eq 3):

```
u_A := P ε − x_0                          # noise low-rank, data full-rank
```

Decomposed onto the subspace `Im(P)` and its complement `Im(I−P)` (Eq 4):

```
P u_A      =  P u            → behaves like u-prediction inside the subspace
(I−P) u_A  = −(I−P) x_0      → behaves like x_0-prediction in the complement
```

So the parameterization is a family interpolating x_0-prediction (`r=0`) ↔ u-prediction
(`r=D`), with both endpoints recovered exactly. The full velocity is recovered
analytically for loss and sampling (Eq 5), so architecture and sampler are unchanged:

```
u = P u_A + (I−P) (x_t + u_A) / σ_t
```

Two design choices the paper stresses:

- Patch-wise projection. `P` is applied independently per DiT patch token, shared
  across tokens — it reduces the per-patch noise dimension, not the token count.
- The subspace must be meaningful. `A` is a data-dependent PCA basis of image
  patches. Their ablation (Fig 5) shows a random subspace performs no better than the
  `x_0`-prediction baseline — the gain comes from aligning `P` with the data's principal
  directions, not from merely reducing rank.

Results (ImageNet 256², pixel JiT-H/16, per-patch `D = 16²×3 = 768`): optimal `r = 8`
(`r/D ≈ 1/96`), FID `1.90 (x_0-pred) → 1.76`, and ~40% faster convergence. The gain
is real but modest, and it is measured exactly where the bottleneck is severe.

## Why it is a pixel-space remedy — and Anima isn't pixel space

The paper is explicit that the entire motivation is high `D`. From the introduction:

> "To predict it accurately, the network must extract the noise from the input and pass
> it through its internal features. This is straightforward in latent spaces, where the
> noise dimension is small relative to the network width. In pixel space, however, the
> per-patch noise dimension can pollute the network's internal states, creating a
> bottleneck."

And §3:

> "u-prediction … is widely used in modern latent flow models, where the representation is
> compressed. When moved to pixels … the target `u = ε − x_0` requires predicting a
> high-dimensional noise component in addition to structured data."

AsymFlow exists to make `u`-prediction affordable when `D` rivals the network width. The
decisive question for Anima is therefore the ratio `D_patch / width`.

### Anima's numbers (the crux)

| quantity | value | source |
|---|---|---|
| DiT spatial patch size | `2` | `library/anima/weights.py:154` (`patch_spatial=2`) |
| VAE latent channels | `16` | `library/anima/weights.py:152`; `models.py:1169` (`LATENT_CHANNELS=16`) |
| per-patch noise dim `D_patch = 2²·16` | `64` | derived |
| DiT residual width | `2048` | `library/anima/weights.py:148`; `models.py:1181` (`model_channels=2048`) |
| ratio `D_patch / width` | `1/32` | derived |
| target parameterization | `u = ε − x_0` (`r = D`, full u-pred) | `train.py:934` (`target = noise − latents`) |

Anima's entire per-patch noise dimension (64) is 12× smaller than the pixel case the
paper studies (768) and sits at 1/32 of the network width. The "burden of predicting
full-rank noise" — routing 64 noise dimensions through a 2048-wide stream — is not a
bottleneck by any reading. Anima is precisely the "straightforward latent" regime the
paper carves out as not needing AsymFlow. The original one-line rejection
("per-patch noise/hidden ratio too small to matter") was correct; this is the number
behind it.

## Steelman — what could still make it interesting

To be fair to the revival idea:

1. Anima latents are low-rank. The SPD/spectral work found a power-law latent
   spectrum (`β ≈ 2.26`) and a collapsed manifold (participation ratio ≈ 6.2). A
   patch-PCA subspace `P` would therefore be meaningful (the paper's necessary
   condition C2, below, would pass). But this is moot: a low-rank data manifold makes
   full-rank noise prediction even less of a capacity burden, so it strengthens the
   "no bottleneck" conclusion rather than weakening it.

2. The loss-geometry view is LoRA-shaped. Via Eq 5, predicting `u_A` and converting
   reweights the complement component's velocity loss by `1/σ_t²` relative to predicting
   `u` directly — i.e. AsymFlow is, in part, a `σ_t`-and-subspace-dependent
   reconditioning of the FM loss. That reweighting is expressible without touching
   architecture, so a "poor-man's AsymFlow" loss weighting is conceivable on the existing
   u-pred base. But the benefit of reconditioning is an optimization-dynamics effect
   (it buys convergence speed from scratch); a finished, converged u-pred base has
   already paid that cost, and a rank-r LoRA cannot re-derive a global parameterization
   change. Expected payoff on a LoRA fine-tune → ~0.

3. Convergence speed, not just quality. The 40% speedup is the most transferable
   claim. But it is a from-scratch pretraining number, and adopting it means changing
   the base model's target — see scope below.

## The scope blocker

Even if the ratio were favorable, AsymFlow changes the base model's prediction target.
It is not an adapter. To adopt it you would either (a) pretrain a new Anima base with the
asymmetric target, or (b) full-finetune the existing base under it. Both are
Tier 3 (new base-model support, not accepted) under `CONTRIBUTING.md`, and neither is
something `anima_lora`'s LoRA/adapter pipeline is built to do. The §5 latent→pixel lift —
the other half of the paper — also does not apply: Anima is latent and stays latent.

## If you insist: a cheap necessary-condition probe

The argument above is decisive on paper, but a one-afternoon empirical kill is available.
AsymFlow helps only if both hold:

- C1 — full-rank noise prediction is a burden. Proxy = `D_patch / width`. Anima = 1/32
  → FAILS by inspection. An empirical version: ablate the DiT's effective rank usage
  on the noise-prediction component (e.g. SVD of the per-patch output Jacobian w.r.t. `ε`);
  if it's already low-rank-using, there is no capacity to reclaim.
- C2 — a small-`r` PCA subspace captures the latent patch energy. Fit per-patch PCA on
  cached latents (`post_image_dataset/lora/**/*_anima.npz`), plot cumulative energy vs `r`,
  and measure the frozen base's `ε`-prediction error projected onto `Im(P)` vs `Im(I−P)`.
  Anima's low-rank spectrum suggests this would PASS — but C1 already fails, so C2 is moot.

A probe mirroring the (now-removed) VR-loss headroom rig (same
cached-latent loader, frozen base, FEI-style banding swapped for PCA banding) would
formalize C1+C2 in <100 lines. Priority: low — write it only if someone wants the
empirical record rather than the dimensional argument.

## Verdict

**Do not revive the asymmetric parameterization for Anima.** It solves a high-`D` pixel
bottleneck Anima does not have (`D_patch/width = 1/32`, the paper's own "straightforward
latent" regime), and it is a base-model change outside this repo's adapter scope. The
only piece of AsymFlow that ever applied to a latent LoRA pipeline was §5.2's VR loss,
which we did port and which benches MARGINAL. The genuinely *untested* idea is now
*assessed* — the dimensional case against it is strong enough that the burden of proof is
on a future advocate to clear C1 above, not on us to keep it on the table.

## References

- AsymFlow paper — arXiv:2605.12964, §4 (parameterization), Fig 2–3 (decomposition),
  Fig 5 / Table 1 (rank + PCA ablation), §5–5.2 (latent→pixel + VR loss).
- The §5.2 VR-loss headroom rig (no longer in-tree) — its `proposal.md` (lines 94–97)
  is where the "not the AsymFlow parameterization" boundary was first drawn.
- `[[project_vr_loss_status]]` — VR-loss (§5.2) status: v1.5 ships, v2/v3 falsified.
- `[[project_spd_spectrum_precondition]]` — Anima latent power-law `β≈2.26` (relevant to C2).
- `[[project_pe_feature_diagnostics]]` — collapsed-manifold / participation-ratio evidence.
- Anima dims: `library/anima/weights.py:148–158`, `library/anima/models.py:1169,1181`.
