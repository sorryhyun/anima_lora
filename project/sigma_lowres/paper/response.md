*Pruned 2026-07-30: the review's fully-absorbed sections were removed
after being folded into the manuscript and `action.md` — §1
(interventional B/C split, scalar-probe conflation, κ_eff validity
domain → manuscript §4.6 + E9 design), §3 (Goldilocks
amplitude-matching → pre-registered in `additional_question.md` AQ4),
§4 (exact affine target read → E10, manuscript §4.3; the t_b
Fourier-mask lever → `action.md`), §6 (κ_eff² refit, floor-law family,
secant discriminator, normalization-recalibration drop → manuscript
§4.6 + `action.md` ladder registration). Kept below: what is still
live — the σ-resolved RoPE derivation and band gate (AQ2), the
posterior-covariance transfer-matrix probe (AQ6), and the batch/
shadow-Adam instrument (E3, AQ5). Original section numbering retained.*

## 2. Q1: a workable (\sigma)-resolved RoPE derivation

For RoPE band (b), an attention logit can be written as

[
\ell_{ij}
=========

\frac{1}{\sqrt d}
\sum_b
q_{i,b}^{\top}R(\phi_{ij,b})k_{j,b}.
]

Under a phase error (\delta\phi_{ij,b}),

[
\delta \ell_{ij}
================

\frac{1}{\sqrt d}
\sum_b
\delta\phi_{ij,b},
q_{i,b}^{\top}R(\phi_{ij,b})\Omega k_{j,b}
+O(\delta\phi^2),
]

and therefore

[
\delta A_i
==========

\left[\operatorname{Diag}(A_i)-A_iA_i^\top\right]\delta\ell_i.
]

This can be propagated through the network to the gradient itself. If (\lambda_b) continuously controls alignment of band (b),

[
\frac{\partial \bar g}{\partial\lambda_b}
=========================================

\mathbb E_\epsilon
\nabla_\theta
\left[
r^\top
\frac{\partial \hat v}{\partial\lambda_b}
\right].
]

That mixed derivative is computable with autograd or paired finite differences. Crucially, its (\sigma)-dependence comes from (q,k,r) and all downstream sensitivities—not from the phase mismatch itself. This gives a proper operator-level positional tangent.

For plain, PI-aligned and re-promoted gradients, define

[
C_{\rm plain}=\bar g_{\rm plain}-\bar g_R,\qquad
C_{\rm rest}=\bar g_{\rm PI}-\bar g_R,
]

[
C_{\rm rope}=C_{\rm plain}-C_{\rm rest}.
]

Then report all three pieces:

[
\frac{|C_{\rm rope}^{\perp}|^2}{2G^2},
\quad
\frac{|C_{\rm rest}^{\perp}|^2}{2G^2},
\quad
\frac{\langle C_{\rm rope}^{\perp},C_{\rm rest}^{\perp}\rangle}{G^2}.
]

The present scalar “PI erasure” is actually

[
F_{\rm plain}-F_{\rm PI}
========================

F_{\rm rope}+I_{\rm rope,rest},
]

not necessarily a pure RoPE share. Consequently, the proposed claim that banded alignment can erase “at most (0.10)” on 512 is not a valid bound without a monotonicity/orthogonality assumption. A subset of bands can outperform full PI through cross-terms.

A plausible derived band gate is a sensitivity-weighted noise posterior:

[
\lambda_b(\sigma)
\approx
\frac{\sigma^2}
{\sigma^2+(1-\sigma)^2P_b^{\rm sens}},
]

where (P_b^{\rm sens}) is measured in the RoPE-gradient tangent, not from raw image power. This predicts gradual, band-specific activation and naturally explains why uniform PI is harmful while banded alignment works.

## 5. Q5: posterior covariance supplies a principled cross-band theory

Let (a=1-\sigma), (b=\sigma), (z=ax+b\epsilon), and

[
\mu(z,c)=\mathbb E[x\mid z,c].
]

The Bayes-optimal flow velocity is

[
v^*(z,\sigma,c)=\frac{z-\mu(z,c)}{b}.
]

For Gaussian corruption, Tweedie’s identity gives

[
D_z\mu
======

\frac{a}{b^2}
\operatorname{Cov}(x\mid z,c).
]

Therefore, for different Fourier bands (\omega\neq\omega'),

[
(D_zv^*)_{\omega,\omega'}
=========================

-\frac{1-\sigma}{\sigma^3}
\operatorname{Cov}
(x_\omega,x_{\omega'}\mid z,c).
]

This is the missing principled cross-band mechanism. The diagonal Gaussian null sets these off-diagonal posterior covariances to zero; natural images do not, because edges, textures and semantics couple frequencies. The connection between a flow velocity Jacobian and posterior covariance is now available explicitly in recent flow-matching theory. [Divergence is Uncertainty](https://arxiv.org/html/2605.00941v1)

Measure the matrix

[
M_{\sigma}[a,b]
===============

\mathbb E
\left|
P_aD_z\hat v,P_b
\right|_F^2
]

using Fourier-band JVPs and Hutchinson probes on the native grid. Then predict each route by masking the columns corresponding to its destroyed bands. If (M_\sigma) is low-rank and all tested cutoffs remove the same dominant conditional-covariance mode, route-uniformity follows naturally.

Before that expensive probe, simply stack the aligned (\Delta\bar r_e) vectors and examine their pairwise cosines/SVD. Uniform norms with different directions would undermine the “universal (m)” interpretation; a rank-one common direction would strongly support it.

*[2026-07-30 note: the cheap SVD pre-check ran as E11 — verdict
norm-only, rank-one excluded; the full transfer-matrix probe is
deprioritized accordingly, and the sharper question is AQ6 in
`additional_question.md`.]*

## 7. Q6: training cares about drift and covariance, not one cosine estimand

For paired per-example perturbations,

[
\delta_i=g_{{\rm dem},i}-g_{{\rm src},i}
=b+\eta_i,\qquad \mathbb E[\eta_i]=0.
]

At batch size (B),

[
\mathbb E[d_B]
\approx
\frac{|P_\perp b|^2}{2|\mu|^2}
+
\frac{
\operatorname{tr}(P_\perp\Sigma_\eta P_\perp)
}{
2B|\mu|^2
}.
]

So the batch-size curve should approximately be intercept (+;1/B):

* intercept: coherent training drift that does not cancel;
* (1/B) term: example-specific disagreement;
* covariance changes: optimizer noise, which can matter even when mean drift vanishes.

The per-example estimand is therefore not automatically a formal lower bound on safety; monotonic improvement with batch size requires an iid zero-mean disagreement model.

For Adam, the closest proxy is the gap after applying the same frozen optimizer state and preconditioner to both gradients. I would run a paired “shadow optimizer replay” at the actual batch and accumulation size, measuring both mean update drift and update covariance, before the expensive full training A/B.

### My experiment order (remaining items)

1. **768 demote–re-promote, retaining vectors**: separates (F(\sigma)) from (I(\sigma)) and resolves the central logical ambiguity. *(= E9, relaunch pending)*
3. **x-zero token-count ladder plus entropy matching**: tests Q2’s family and Resid mechanism. *(registered, not scheduled)*
4. **Fourier JVP transfer matrix**: tests the posterior-covariance account of Q5. *(deprioritized after E11; see AQ6)*
5. **Batch-size (a+b/B) fit and shadow-Adam replay**. *(= E3 follow-on)*

*(Items done and pruned: 2 — α-vector reanalysis = E10; 4's SVD
pre-check = E11. The 512-banded run guidance — intervention-effect
measurement, not a falsification test — is folded into `action.md`.)*
