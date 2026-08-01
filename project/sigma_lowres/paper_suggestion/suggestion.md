# Suggestion: derive the residual-mismatch curve from the posterior operator

## Summary

The paper currently treats the route-uniform residual-mismatch curve
\(\|\Delta\bar r(\sigma)\|\) as a measured ingredient of the predictive
account. It can be grounded one step more deeply. Under the Bayes-optimal
flow-matching predictor, the fixed-image, noise-averaged residual has an exact
posterior-operator form. This does not produce a parameter-free numerical
curve for natural-image latents, because the conditional latent distribution
is unknown, but it does show precisely what the measured curve estimates and
which additional assumptions would turn it into a closed-form prediction.

The recommended manuscript position is therefore:

> \(\Delta\bar r(\sigma)\) is derivable in principle from the pair of
> grid-conditional clean-latent distributions. We measure it because those
> distributions, including their cross-frequency and cross-grid dependence,
> are not analytically available. The measured curve is an empirical closure
> of an exact posterior identity, rather than an unconstrained black-box
> regressor.

This suggestion does **not** revive E12's refuted posterior-budget or
lossy-code saturation hypothesis. E12 attempted to predict the curve from a
single posterior-uncertainty scalar. The derivation below depends on the
posterior **mean operator**, and a practical second-order closure would depend
on the full paired cross-grid covariance, including off-diagonal structure.

## Exact Bayes identity

Fix a grid \(e\), caption \(c\), and clean latent \(X_e\sim p_e(\cdot\mid c)\).
Write

\[
a = 1-\sigma,\qquad Z_e = aX_e + \sigma\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),
\]

and define the posterior mean

\[
m_e(z,c) := \mathbb E[X_e\mid Z_e=z,c].
\]

For the squared flow-matching objective with target
\(v=\epsilon-X_e=(Z_e-X_e)/\sigma\), the population-optimal velocity field is

\[
v_e^*(z,\sigma,c)
= \mathbb E[v\mid Z_e=z,c]
= \frac{z-m_e(z,c)}{\sigma}.
\]

The paper's residual is averaged over noise while holding the query image
fixed. For a particular clean latent \(x_e\), its Bayes-optimal value is

\[
\boxed{
\bar r_e^*(\sigma;x_e)
= \mathbb E_{\epsilon}
  \left[v_e^*(a x_e+\sigma\epsilon,\sigma,c)
        -(\epsilon-x_e)\right]
= \frac{x_e-
  \mathbb E_{Z_e\mid x_e}[m_e(Z_e,c)]}{\sigma}.
}
\]

Let \(A_e\) be the paper's alignment operator that places the source-grid
residual on the demoted grid. Up to the sign convention, which disappears
under a norm, the exact Bayes cross-grid mismatch is then

\[
\boxed{
\Delta\bar r_e^*(\sigma;x)
= \bar r_{\mathrm{dem}}^*(\sigma;x_{\mathrm{dem}})
  - A_e\bar r_{\mathrm{src}}^*(\sigma;x_{\mathrm{src}}).
}
\]

Thus the object is the difference between two posterior reconstruction-bias
operators. The noise schedule determines how those operators are smoothed;
it does not determine them without a clean-latent distribution.

### Score form

Let \(p_{\sigma,e}(z\mid c)\) be the noised marginal. Tweedie's identity for
the linear interpolant gives, for \(\sigma<1\),

\[
m_e(z,c)
= \frac{z+\sigma^2\nabla_z\log p_{\sigma,e}(z\mid c)}{1-\sigma}.
\]

Substitution yields

\[
\boxed{
\bar r_e^*(\sigma;x_e)
= -\frac{\sigma}{1-\sigma}
  \mathbb E_{Z_e\mid x_e}
  [\nabla_z\log p_{\sigma,e}(Z_e\mid c)].
}
\]

At \(\sigma=1\), where \(Z=\epsilon\) is independent of \(X\), the
well-defined endpoint is

\[
\bar r_e^*(1;x_e)=x_e-\mathbb E[X_e\mid c].
\]

This form makes the missing ingredient explicit: deriving the numerical
curve requires the grid- and caption-conditional smoothed score, or an
equivalent model of the clean-latent distribution.

## Important correction to the present interpretation

The probe averages over \(\epsilon\) at fixed \(x\), forms a residual
distance for each image, and only then aggregates over images. MSE optimality
implies

\[
\mathbb E[v^*(Z)-v\mid Z,c]=0
\]

and hence zero residual after a joint average over \((X,\epsilon)\). It does
**not** imply

\[
\mathbb E_\epsilon[v^*(a x+\sigma\epsilon)-v\mid X=x,c]=0.
\]

The latter is exactly the paper's per-image object and is generally nonzero,
even for the Bayes-optimal predictor. Consequently, the statement in
`bench/run_prior_distance.py` that "a perfect model has r-bar = 0 on every
grid" is too strong. The measured mismatch contains:

1. intrinsic fixed-image posterior reconstruction bias; and
2. approximation error of the trained velocity network relative to the
   Bayes field.

Writing \(\hat v_{\theta,e}=v_e^*+q_{\theta,e}\) makes the split exact:

\[
\bar r_{\theta,e}(\sigma;x_e)
= \bar r_e^*(\sigma;x_e)
  + \mathbb E_{Z_e\mid x_e}[q_{\theta,e}(Z_e,\sigma,c)].
\]

The first term is derivable from \(p_e(X\mid c)\); the second is genuinely
model-specific. The manuscript should therefore call the probe a
**mean-residual mismatch** or **posterior-bias-plus-model-error mismatch**, not
purely a model-error difference.

## A closed-form second-order closure

The cheapest nontrivial closure is a full Gaussian approximation

\[
X_e\mid c\sim\mathcal N(\mu_e,C_e).
\]

With

\[
Q_e(\sigma)=(1-\sigma)^2C_e+\sigma^2I,
\]

the exact posterior calculation gives

\[
\boxed{
\bar r_e^*(\sigma;x_e)
= \sigma Q_e(\sigma)^{-1}(x_e-\mu_e).
}
\]

For paired source and demoted latents, define

\[
L_s=\sigma Q_s^{-1},\qquad L_d=\sigma Q_d^{-1},
\]

and covariance blocks \(C_{ss},C_{dd},C_{ds}\). Then the image-population
mean squared mismatch is

\[
\begin{aligned}
\mathbb E\|\Delta\bar r_e^*\|^2
={}&\operatorname{tr}(L_dC_{dd}L_d^\top)
+\operatorname{tr}(A_eL_sC_{ss}L_s^\top A_e^\top)\\
&-2\operatorname{tr}(L_dC_{ds}L_s^\top A_e^\top).
\end{aligned}
\]

This predicts a \(\sigma\)-curve from paired **clean latents only**, with no
DiT forward pass. The source--demote cross-covariance is load-bearing: it is
what distinguishes a derivation of cross-grid mismatch from two unrelated
posterior-variance estimates.

A diagonal Fourier covariance reduces to the existing spectral/Wiener null:
the mismatch is confined to destroyed bands and ordered by their power. The
paper has already falsified that closure. The remaining theoretically
interesting question is whether a structured non-diagonal covariance
(wavelet blocks, low-rank cross-band modes, or another tractable operator)
recovers the observed amplitude law without consulting the denoiser.

## Match the derivation to the actual measured scalar

The paper writes \(\|\Delta\bar r(\sigma)\|\), but the G7 instrument reports a
split-half-corrected **relative** \(L_2\) distance between aligned residuals,
and the fitted curve averages that scalar over routes. A theory comparison
must reproduce the same estimand. In a high-dimensional concentration
approximation, the Gaussian closure would predict

\[
m_e(\sigma)
\approx
\frac{
\sqrt{\mathbb E\|\bar r_d-A_e\bar r_s\|^2}}
{\tfrac12\left(
\sqrt{\mathbb E\|\bar r_d\|^2}
+\sqrt{\mathbb E\|A_e\bar r_s\|^2}
\right)},
\]

followed by the same split-half floor correction. A raw mismatch norm should
not be silently substituted for this normalized quantity.

## Proposed paper change

Add a short proposition after the introduction of
\(\Delta\bar r(\sigma)\) in the theory section:

> **Posterior-operator interpretation.** For the population-optimal velocity
> field, the fixed-image mean residual is
> \(\bar r_e^*(\sigma;x)=
> [x-\mathbb E_{Z\mid x}\mathbb E[X\mid Z,c]]/\sigma\). Hence the cross-grid
> mismatch is determined by the difference of the source- and demoted-grid
> posterior reconstruction operators. The noising path fixes the smoothing
> but not the posterior operators; a numerical curve therefore requires a
> model of the conditional latent distribution. Our measured
> \(\|\Delta\bar r(\sigma)\|\) is an empirical closure of this identity and
> additionally includes the frozen denoiser's approximation error.

Then adjust the measurement paragraph to avoid describing the quantity as
model error alone. The paper can retain the measured curve in every current
fit: the derivation changes its interpretation and theoretical status, not
the reported map or held-out results.

## Optional validation experiment

Before adding another GPU probe, test the closure entirely on stored paired
latents:

1. Project source and demoted latents into a common low-dimensional wavelet
   or patch-PCA basis.
2. Estimate \(C_{ss},C_{dd},C_{ds}\) on a fit split with shrinkage.
3. Predict the normalized mismatch curve across \(\sigma\) from the Gaussian
   formulas above.
4. Score its shape and route transfer against the existing G7/E11 residual
   curves on held-out images.
5. Compare three nested closures: diagonal spectral, within-band block
   covariance, and low-rank cross-band covariance.

The experiment is informative in either direction. Success replaces part of
the empirical curve with a data-only derivation. Failure establishes that
higher-order/non-Gaussian posterior structure or model approximation error is
essential, which justifies retaining the measured curve as the minimal honest
closure.

## Recommendation

Adopt the exact posterior identity and the fixed-image averaging correction
now. Keep \(\Delta\bar r(\sigma)\) measured for the paper's quantitative
predictions unless the paired-covariance closure passes held out. This gives
the account a first-principles foundation without overstating what can be
computed from the noise schedule alone.
