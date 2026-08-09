# SVD-Down LoRA — a principal input basis without OrthoInit's cold-start bottleneck

Status: **SHIPPED** (Phase 0 passed; archived 2026-06-28). The `down_init =
"weight_svd"` option is live on the plain-LoRA path — see the ops doc
`docs/methods/svd-down-lora.md`. This file is kept as the original theory /
motivation write-up. Turbo is the observation that motivated the proposal, not
the only intended use case.

## Executive claim

The current OrthoInit parameterization has a sound goal—start from singular
directions of the pretrained weight while retaining full rank-$r$
expressivity—but it pays for that prior with a very narrow initial optimization
geometry. At zero delta, only its $r$ singular-value parameters receive a
gradient. Standard LoRA exposes an entire $d_{out}\times r$ up-projection on the
first step.

We should test a simpler initialization that keeps the useful half of both:

\[
\Delta W = sBA,\qquad
A_0 = cV_r^\top,\qquad
B_0 = 0,
\]

where $W_0=U\Sigma V^\top$, $V_r$ contains the top-$r$ right singular vectors,
and $c$ matches the scale of LoRA's current Kaiming-initialized down projection.

This **SVD-Down LoRA** is still ordinary LoRA after initialization. It starts
with exactly zero $\Delta W$, leaves the pretrained $W_0$ untouched, exposes the
same wide first-step tangent as plain LoRA, and can rotate away from the SVD
basis because both $A$ and $B$ remain trainable. It requires no new forward,
optimizer, checkpoint, merge, or inference format.

The proposal is deliberately narrower than “SVD initialization is better.” The
claim worth testing is:

> For short or data-limited adaptation, replacing LoRA's random input basis with
> a normalized principal input basis can improve early conditioning without
> imposing OrthoInit's diagonal cold start or OrthoLoRA's frozen-subspace cap.

## Why this proposal exists

The prior question comes from **StelLA** (Li et al., NeurIPS 2025,
[arXiv:2510.01938](https://arxiv.org/abs/2510.01938)), read the same day. Its
three-factor $USV^\top$ with $U,V$ on the Stiefel manifold **is** the OrthoInit
parameterization critiqued below, and its Table 5 initialization ablation shows
the SVD seed washes out (SVD-major ≈ SVD-minor ≈ random) once the subspace is
trainable — which is what makes “what is the SVD seed still worth on a *free*
LoRA?” the question this proposal answers. The chimera branch of the same
reading is `docs/proposal/stella_chimera.md`. The probe below is what made it
urgent, not where it came from.

`bench/turbo/probe_ortho_init_step.py` was written around a “hot OrthoInit
start” hypothesis. Its results falsify that hypothesis:

| Probe | plain LoRA step-1 $\lVert\Delta W\rVert_F$ | OrthoInit | Ortho / plain |
|---|---:|---:|---:|
| $W_0$-aligned input | $5.606\times10^{-3}$ | $2.192\times10^{-4}$ | **0.0391** |
| isotropic input | $5.911\times10^{-3}$ | $2.248\times10^{-4}$ | **0.0380** |

Plain LoRA already exceeds OrthoInit's first update on step 1. This is not a
small scaling discrepancy: at the probed width and rank, the OrthoInit update is
about **25× smaller**. The same effect appears in live Turbo dynamics. With all
other resolved settings fixed, increasing only the OrthoInit student's LR from
$10^{-5}$ to $10^{-4}$ changed the late GAN generator/discriminator losses from
roughly $3.05/0.44$ (discriminator wins) to $0.72/1.37$ (balanced) and restored
movement in the diversity term. That is consistent with an under-responsive
student, not an excessively hot one.

Turbo makes the problem obvious because the fake network and discriminator are
moving opponents. The underlying geometry is not Turbo-specific, however. A
short supervised character/style LoRA also spends a finite optimizer budget
escaping its initialization; making the initial tangent both informed and wide
could benefit those tasks without requiring a larger LR.

## The load-bearing theory: the tangent at zero delta

For one adapted Linear, write standard LoRA as

\[
W = W_0 + sBA,
\quad A\in\mathbb{R}^{r\times d_{in}},
\quad B\in\mathbb{R}^{d_{out}\times r}.
\]

Let $G=\partial\mathcal{L}/\partial\Delta W$. With the normal zero-up
initialization $B_0=0$,

\[
\frac{\partial\mathcal{L}}{\partial B}=sGA^\top,
\qquad
\frac{\partial\mathcal{L}}{\partial A}=sB^\top G=0.
\]

Only the down projection is dormant on step 1. The up projection is not. The
reachable first-order update set is

\[
\mathcal{T}_{LoRA}(A_0)=\{\delta B A_0:\delta B\in
\mathbb{R}^{d_{out}\times r}\},
\]

with dimension up to $d_{out}r$ when $A_0$ has full row rank. At
$d_{out}=2048,r=64$, that is up to 131,072 active coordinates.

Current OrthoInit instead uses

\[
\Delta W=sP\,\mathrm{diag}(\lambda)Q,
\qquad \lambda_0=0.
\]

At initialization,

\[
\frac{\partial\mathcal{L}}{\partial P}=0,
\qquad
\frac{\partial\mathcal{L}}{\partial Q}=0,
\qquad
\frac{\partial\mathcal{L}}{\partial\lambda_i}
=s\,p_i^\top Gq_i.
\]

Its first-order update is restricted to the $r$ fixed dyads
$\{p_iq_i^\top\}_{i=1}^r$. “Full LoRA expressivity” remains true eventually,
after nonzero $\lambda$ activates gradients into $P$ and $Q$, but it is not true
of the cold-start tangent. AdamW makes the difference especially visible: it
normalizes updates per parameter, so updating $r$ scalar amplitudes does not
match updating $d_{out}r$ entries of $B$ merely because the scalar gradients are
well aligned.

SVD-Down keeps standard LoRA's tangent dimension. It changes only which
$r$-dimensional input row-space that tangent initially sees.

## What the weight SVD does—and does not—justify

Let $W_0=U\Sigma V^\top$. Among all orthonormal $r$-dimensional input bases,
$V_r^\top$ captures the maximum pretrained-weight energy:

\[
V_r=\arg\max_{R^\top R=I_r}\lVert W_0R\rVert_F^2.
\]

This is the useful prior: the initial LoRA branch reads the input directions to
which the pretrained Linear is most responsive. For in-domain adaptation, where
the pretrained representation is already useful and needs a controlled
correction, those directions are plausible high-signal coordinates. Unlike a
random down projection, they are deterministic and layer-specific.

That statement is **not** a proof that top singular vectors of $W_0$ maximize
the task gradient. They do so only under additional alignment assumptions about
the activation covariance and downstream gradient. The proposal therefore does
not borrow the stronger guarantees of data-aware methods:

- [EVA](https://arxiv.org/abs/2410.07170) initializes from activation SVD and
  proves that its selected directions maximize expected gradient signal.
- [LoRA-GA](https://arxiv.org/abs/2407.05000) uses gradient information to align
  the low-rank first step with full fine-tuning.

Those methods support the general premise that LoRA initialization materially
affects convergence, but they also expose the main risk here: weight-SVD is a
cheap proxy for task-relevant directions, not the task-relevant measurement
itself. If SVD-Down fails specifically on novel concepts or large domain shifts,
activation-SVD is the principled next arm rather than increasingly elaborate
weight-only parameterizations.

## Relation to PiSSA and current OrthoInit

[PiSSA](https://arxiv.org/abs/2404.02948) also begins from the principal singular
components of $W_0$, but it initializes both adapter factors nonzero and places
the remaining components in a modified frozen residual weight. The total
function is preserved, yet the stored base/adapter decomposition is no longer
the ordinary “unchanged $W_0$ plus zero LoRA delta” decomposition.

SVD-Down is intentionally more conservative:

| Method | Base weight at init | Adapter delta | First-step trainables | Initial prior |
|---|---|---:|---|---|
| Plain LoRA | unchanged $W_0$ | zero | full $B$ | random input basis |
| PiSSA | residualized | nonzero component | both factors | paired principal component |
| Current OrthoInit | unchanged $W_0$ | zero | $\lambda$ only | paired SVD dyads |
| **SVD-Down** | unchanged $W_0$ | zero | full $B$ | principal input basis |

SVD-Down gives up OrthoInit's paired left/right singular-vector prior. That is
the point: the task gradient chooses the output directions in $B$ immediately,
instead of being forced to change only the amplitudes of pre-paired dyads. It
also gives up any claim of maintaining orthogonality during training; this is an
initialization for free-form LoRA, not an orthogonal parameterization.

## Fair initialization scale

The current `LoRAModule` uses
`kaiming_uniform_(lora_down.weight, a=sqrt(5))`. For a Linear with large
`fan_in`, each row has expected squared norm approximately $1/3$. A row of
$V_r^\top$ has norm 1. Copying it directly would therefore make SVD-Down's
initial down projection about $\sqrt{3}$ larger than the current LoRA basis and
would confound “better direction” with “larger effective step.”

The v0 initialization should be

\[
A_0=\frac{1}{\sqrt{3}}V_r^\top,\qquad B_0=0.
\]

The probe must still report actual $\lVert A_0\rVert_F$ and step-1
$\lVert\Delta W\rVert_F$; the analytic match is an expectation, not a guarantee
under AdamW and a particular gradient.

## Minimal implementation surface

Add a plain-LoRA initialization option, tentatively:

```toml
down_init = "kaiming"       # existing default
# down_init = "weight_svd"  # proposal
```

For v0, `weight_svd` applies only to Linear layers:

```python
W = org_module.weight.data.float().to(init_device)
_, _, V = torch.svd_lowrank(W, q=min(rank + 6, min(W.shape)), niter=2)
lora_down.weight.copy_(V[:, :rank].T / math.sqrt(3))
lora_up.weight.zero_()
```

The repo already uses this randomized SVD construction in
`networks/lora_modules/ortho.py`; this is not new numerical machinery. The
existing channel-scale absorption runs after initialization exactly as it does
for Kaiming LoRA. Conv2d remains Kaiming in v0.

No new module class is needed. The runtime remains `LoRAModule`, and saved
weights remain the standard `lora_down.weight` / `lora_up.weight` / `alpha`
layout. T-LoRA masking composes automatically because it acts on the rank
bottleneck after `lora_down`. Hydra/Chimera are out of scope until the plain
two-factor result earns extension.

## Evidence plan and decision gate

### Phase 0 — parameterization probe

Extend `bench/turbo/probe_ortho_init_step.py` into three arms:

1. Kaiming-down plain LoRA;
2. normalized SVD-Down LoRA;
3. current $P\,\mathrm{diag}(\lambda)Q$ OrthoInit.

Run both random targets and targets with controlled alignment to $V_r$. Record
stepwise $\lVert\Delta W\rVert_F$, loss, factor-gradient norms, and cosine of
$\Delta W$ with the target update.

Phase 0 passes if SVD-Down:

- is exactly zero-output at initialization;
- has nonzero gradients only in `lora_up` on step 1, like standard LoRA;
- remains within $0.5\times$–$2\times$ plain LoRA's first effective-update norm
  after normalization; and
- improves update alignment in the aligned arm without materially hurting the
  random arm.

The norm gate matters. A “win” caused only by taking a larger step is an LR
change disguised as initialization research.

### Phase 1 — ordinary LoRA training, where the proposal must earn its name

Run Kaiming LoRA versus SVD-Down with identical rank, alpha, LR, scheduler,
dataset order, and sampling seeds. Use one in-domain/style task and one
novel-character or stronger-domain-shift task; the theory predicts the first is
more favorable and the second is the failure case. Screen with one seed, then
confirm any apparent win with at least three training seeds.

Measure:

- loss-versus-step AUC and steps to the baseline's final loss;
- aggregate and per-layer $\lVert\Delta W\rVert_F$;
- drift of the learned down row-space away from $V_r$;
- fixed-prompt, fixed-seed sample grids at matched steps and matched wall time;
- the repo's current distribution metric where available, treated as supporting
  evidence rather than a substitute for rendered quality.

**Ship the option** only if it produces a repeatable convergence improvement
(target: at least 20% fewer steps to a matched loss/quality point) without a
fixed-budget quality or diversity regression. Keep Kaiming as the default until
that result survives reseeding. **Kill or narrow it to in-domain tasks** if it
only helps the style task, if the effect disappears after step-size matching,
or if the stronger domain-shift task consistently learns worse details.

Turbo can be rerun only after the normal-LoRA result. It is a useful stress test
of early responsiveness, but its moving fake/discriminator makes it a poor sole
judge of an initialization intended for general adaptation.

## Risks and honest limits

- **Principal directions can be conservative.** The top singular directions
  describe what $W_0$ already represents strongly. A new concept may need
  directions in the spectral tail. Trainable $A$ allows escape, but does not
  make the initial bias free.
- **Weight SVD ignores the data.** EVA's activation basis or LoRA-GA's gradient
  basis is theoretically closer to the actual task. SVD-Down earns its place
  only if its zero-calibration simplicity is enough in practice.
- **SVD initialization has startup cost.** It is paid once per adapted Linear,
  as in current OrthoInit. It should be reported separately from per-step
  throughput.
- **This is not orthogonality.** The basis is orthonormal only at initialization
  and may drift immediately. The benefit sought is conditioning, not a
  structural constraint.
- **One successful run proves little.** Initialization changes the training
  lottery. The decision must survive multiple seeds and an explicit
  domain-shift control.

## Recommendation

Build Phase 0. The implementation cost is small, checkpoint compatibility is
free, and the proposal directly fixes a measured optimization defect without
adding another loss or optimizer heuristic. The theory is strong enough to
justify the experiment but narrow enough to be falsifiable: SVD-Down should
retain plain LoRA's early update capacity and help most when pretrained
principal input directions overlap the downstream task.

If that prediction holds, SVD-Down is a better general-purpose interpretation
of “SVD-informed LoRA initialization” than the current diagonal OrthoInit path.
If it does not, the negative result is still decisive: pretrained weight energy
is not a useful proxy for task gradient signal on Anima, and future work should
move to activation- or gradient-informed bases instead of tuning OrthoInit's LR.

## References

- Li et al., [StelLA: Subspace Learning in Low-rank Adaptation using Stiefel
  Manifold](https://arxiv.org/abs/2510.01938), NeurIPS 2025 (Spotlight) — the
  paper this proposal reacts to. Code: <https://github.com/SonyResearch/stella>.
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language
  Models](https://arxiv.org/abs/2106.09685), 2021.
- Meng et al., [PiSSA: Principal Singular Values and Singular Vectors
  Adaptation](https://arxiv.org/abs/2404.02948), NeurIPS 2024.
- Wang et al., [LoRA-GA: Low-Rank Adaptation with Gradient
  Approximation](https://arxiv.org/abs/2407.05000), 2024.
- Paischer et al., [Parameter Efficient Fine-tuning via Explained Variance
  Adaptation](https://arxiv.org/abs/2410.07170), NeurIPS 2025.
- Local mechanism evidence:
  `bench/turbo/results/20260621-1420-ortho-init-step/result.json` and
  `bench/turbo/results/20260621-1422-isotropic/result.json`.
