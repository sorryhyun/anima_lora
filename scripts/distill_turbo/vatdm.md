# VA-TDM vs the turbo loop (arXiv:2611.06796)

Source: Luo, Pang, Hu, *VA-TDM: A Unified View for Improving Few-Step Diffusion Models by
Velocity Adversarial Distillation* (2026-11-04). Code: <https://github.com/Luo-Yihong/VA-TDM>.
Read 2026-11-06, main text pp. 1–13; the appendix (derivations, training cost,
hyperparameters, whether the student is a LoRA) and the code are unread. Idea 1 has one run
(below); the rest has not been run. The current loop is described in
`docs/anima/methods/turbo.md`, MFD in `README.md`.

## The paper

A follow-up to TDM (Luo et al., 2025). Few-step distillation in the DMD family: student,
frozen teacher, and an online field that tracks the student's distribution.

**Student transition.** The student predicts `x̂0 = x_t − t·v_θ(x_t, t)` and takes a DDIM step
to `s`: `x_s = (1−s)x̂0 + ηs·ε̂ + s√(1−η²)z` (Eq. 4). The default η = 1 is deterministic and is
the Euler step of the flow ODE. The output is an ordinary velocity model on an Euler sampler.
Matching is imposed on `x_s`, not on `x̂0`, after perturbing it to `x_τ`, s < τ < 1, with the
teacher-compatible kernel `α_{τ|s} = (1−τ)/(1−s)`, `β²_{τ|s} = τ² − α²_{τ|s}s²` (Eq. 5).
Distillation runs along the student's own K-step trajectory, one transition `(t_k, t_{k−1})` at a
time.

**TDM's mismatch (Remark 1).** TDM learns the online score by clean-target denoising: it
perturbs `x̂0` straight to τ. The distribution the generator update needs is `x_s` perturbed to
τ, and `x_s` keeps the residual inherited from the student transition. The two differ.

**Two velocity targets.**
- A-form, rectified: `v^A = (x_τ − x̂0)/τ` (Eq. 8). Low variance; matching it matches the
  conditional moment `E[x̂0 | x_τ]`.
- B-form, Gaussian: the velocity of the s→τ path (Eq. 9). Higher variance; matching it matches
  the distribution, `p_θ,τ = q_τ`.
- At s = 0 the two coincide.

**Velocity Adversarial Distillation.** A vector critic f plays
`max_f E[fᵀ(u − v̄_θ) − ½‖f‖²]` against the student (Eq. 12). Its optimum is the velocity residual
`f* = u − v̄_θ` (Lemma 2), and equilibrium is `v̄_θ = u` (Prop. 3). The critic is not a
classifier, and the game uses no real data. Two generator updates:
- pathwise: `−E[fᵀ v^F]`, differentiated through the whole sampling chain;
- flow-product: `−E[fᵀ(v^F − h₀)]` (Lemma 4). It needs input derivatives of the frozen field,
  that is a backward through the fake.

Theorem 5 maps form × update onto known methods:

| Form | Pathwise, detached critic | Flow-product, non-detached critic |
|---|---|---|
| A | MMD practical update (Salimans et al., 2024) | exact moment matching |
| B | reverse KL (DMD / Diff-Instruct / TDM) | Fisher divergence (SiD / FGM) |

**The practical recipe (Sec. 3.3).**
- Stop-gradient on the whole critic output. Backpropagating through it collapsed (Fig. 3).
- `L_mix = (1−λ)·L_path,sg + λ·L_fp,sg` (Eq. 20), and `λ_m·L^A + (1−λ_m)·L^B` (Eq. 22), with
  λ_m = 0.8.
- Field: `h_ψ` regresses `v^F(x_τ | x_s, x̂0)` (Eq. 21), with a teacher regulariser
  `λ_r‖h − u‖²`. The minimiser becomes `(v̄ + λ_r u)/(1+λ_r)`, so the critic is corrected to
  `f = (1+λ_r)(u − h)`.
- The A and B fields share one backbone, with two heads.
- **Transition-aware field**: the field takes the transition's target timestep as input,
  `h_ψ(x_τ, τ, s)`. Student transitions at different step indices produce different noisy
  distributions, and one shared auxiliary score cannot track all of them. TDM gave each step its
  own disjoint τ interval, which the paper says gives up part of the supervision.
- Warm-up: ODE regression on the student's own trajectory (teacher ODE over each
  `(t_k, t_{k−1})`), then switch to VAD.

**Results, 4 NFE (Table 1).**

| | GenEval | OCR | Aest. | HPSv3 | LPIPS |
|---|---|---|---|---|---|
| SD3.5-M teacher, 50×2, CFG 4.5 | 0.65 | 0.58 | 5.37 | 9.78 | 0.61 |
| DMD2 | 0.60 | 0.53 | 5.55 | 11.18 | 0.53 |
| DPDMD | 0.62 | 0.33 | 5.28 | 8.13 | 0.59 |
| TDM | 0.61 | 0.57 | 5.58 | 11.54 | 0.57 |
| **VA-TDM** | **0.67** | **0.59** | **5.65** | **11.89** | 0.58 |
| FLUX.2-Klein 4B, official 4-step | 0.78 | 0.64 | 5.28 | 11.27 | 0.47 |
| **VA-TDM** on FLUX.2-Klein-Base 4B | **0.80** | **0.69** | **5.43** | **11.58** | **0.56** |

MiniMax-H3 (33B audio-video): 61% human preference against Fast-H3 at 4 NFE.

The DPDMD row is the authors' run on SD3.5, with no GAN, CDM or dynamic grid as far as the main
text says. It is not our configuration.

**Ablation, SD3.5-M (Table 2).**

| Configuration | GenEval | OCR | HPSv3 | LPIPS |
|---|---|---|---|---|
| B-form only, no transition-aware field | 0.58 | 0.53 | 11.13 | 0.56 |
| B-form only | 0.61 | 0.57 | 11.47 | 0.59 |
| A-form only | 0.64 | 0.55 | 11.20 | 0.58 |
| λ_m 0.8, separate backbones | 0.62 | 0.56 | 11.33 | 0.57 |
| λ_m 0.8, shared backbone | 0.63 | 0.58 | 11.47 | 0.58 |
| + path / flow-product mix, λ 0.7 | 0.65 | 0.57 | 11.39 | 0.59 |
| + field regularisation, λ_r 1 | 0.66 | 0.58 | 11.44 | 0.57 |
| + corrected critic | 0.67 | 0.59 | 11.89 | 0.58 |

## Where our loop sits in this frame

Our DM term (`steps.dmd_surrogate`) is the s = 0 case: the signal lands on the clean estimate
`x_pred = x_g − σ_g·v_g`, so A- and B-form coincide. The `(grad · x_pred)` surrogate with a
detached signal is the pathwise update with a detached critic, i.e. reverse KL. Only the time
weight differs (`τ`, plus `dm_x0_norm`).

The three methods move different axes:

| | Ours (DP-DMD + GAN, CDM) | MFD (`--mfd_dtau`) | VA-TDM |
|---|---|---|---|
| Rollout | deterministic Euler, dynamic grid | same | DDIM η 1 = Euler |
| Matched state | `x_pred` of grad step g ~ U{1..N−1} | same | each transition's `x_s`, perturbed to τ > s |
| Signal | `τ·(v_real − v_fake)` at one τ | `(σ_a−σ_b)·(U_real − U_fake)`, K-step mean velocity | `sg(f)ᵀ v^F`, A/B mix, path/fp mix |
| Fake | flow matching on `x_pred` (`ε − x_pred`, `steps.fake_update`); blind to g and σ_g | same | shared backbone, A/B heads, takes s, teacher-regularised |
| Real data | `[gan]` term reads dataset latents (`weight_gen` 0.03) | GAN optional; the MFD paper has none | none in the main text |
| Step 0 | DP-DMD diversity anchor, no critic signal | same | no anchor; ODE-regression warm-up |

MFD changes how the signal is estimated along τ. VA-TDM changes which distribution is matched,
and how the field is built. They are orthogonal: `mfd_surrogate`'s `_v_fake` can take a
transition input like any other fake call.

VA-TDM's "adversarial" critic is not our discriminator. It has an analytic optimum and no real
data, so it does not stand in for the `[gan]` term or for DMAD's head R.

## Ideas, cheapest first

### 1. Transition-aware fake

Our fake trains on a mixture of `x_pred` from every grad step g, with σ_g also varying under
`dynamic_schedule`, and nothing tells it which step a sample came from. This is the setup the
paper calls overlooked, and its largest single ablation gain (B-form: GenEval 0.58 → 0.61, OCR
0.53 → 0.57, LPIPS 0.56 → 0.59).

Arm: feed σ_g of the grad step into the fake, as a zero-init embedding added on its timestep
path, in both `fake_update` and the DM/MFD query. The fake is discarded at save, so the student
output stays a plain LoRA. How the fake's timestep path can take the extra input has not been
checked. Run it on top of the current MFD recipe, same seed and init, against an MFD-only twin.

**Result, 2026-11-10: no harm, no verdict at 500 steps.** Implemented as `network.fake_transition`
(`FakeTransitionEmbed` in `networks/methods/turbo_dmd.py`: σ of the source transition → a
zero-init delta on the fake's t-embedding, fake view only). Run on DP-DMD rather than MFD:
`anima_turbo_m14_dpdmd_svd_500_tafake` against `anima_turbo_m14_dpdmd_svd_500` (merged14 int8
teacher, 500 steps, same seed and init; the one config difference is `fake_transition = true`).

- Losses: identical within noise over the whole run. Paired per-step difference at steps
  350–500: fake −0.002 ± 0.003, dm +0.002 ± 0.003, rel_gap +0.001 ± 0.002, cos 0.978 in both.
  The fake loss did not drop.
- Renders (4 steps, cfg 1, anima-sign prompt seeds 1–4 and the nozomi/monster prompt seeds 42–45,
  against the teacher at 28 steps, cfg 4): tafake is sharper and closer to the teacher in
  detail, hair colour and sign layout. The split and duplicated signs it draws are also in the
  teacher. The plain run drifts to brown hair and one centred sign. One tafake-only trait: two
  cans in hand on the monster prompt (3 of 4 at step 500, present at 250 too; the teacher
  draws one). Contrast is a little above the teacher's.
- Both runs are still moving between steps 250 and 500 (both oversaturated at 250), so 500
  steps does not settle the comparison. The next test is a 1,000-step tafake run.

### 2. Teacher-regularised fake and corrected critic

Add `λ_r‖h − u‖²` to the fake loss and scale the signal by `(1+λ_r)`, with λ_r = 1. The code
change is small. The cost is a teacher forward in every fake inner step
(`fake_steps_per_student_step` 4). It is unclear from the main text whether `u` here is
CFG-guided; ours would have to be, since the DM real score is (`dmd_surrogate` docstring).

### 3. Intermediate-state matching (TDM proper)

Match `x_{g+1}` perturbed to τ > σ_{g+1}, not `x_pred`. Structural: the fake models the
perturbed `x_s` distribution, and the DP-DMD anchor and L_CDM have to be re-placed around it.
The output is still a normal LoRA on Euler, unlike data-space iteration (`docs/anima/proposal/turbo_data_space.md` §5).

### 4. B-form, flow-product mix

Both need a backward through the fake. Last on a 16 GB card.

## Open before porting

- Appendix B.2 (the TDM mismatch derivation), the time weights of Theorem 5, the training cost
  and the student parameterisation are unread.
- Whether the warm-up is needed when the student starts from the v1.1 delta, as ours does.
