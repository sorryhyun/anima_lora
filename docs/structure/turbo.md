# DP-DMD: diversity-preserved few-step distillation

A few-step distillation that compresses the 28-step, CFG=4 Anima teacher into an
N-step student LoRA (`student_steps=4` ships; was 2 until `5ef128d`). Both the student and the
auxiliary "fake" score model are plain LoRAs on one frozen DiT — the base
weights do the heavy lifting for all three roles (teacher / student / fake), and
only the rank-$r$ adapters train.

This is a port of Wu, Li, Zhang, Ma — *"Diversity-Preserved Distribution Matching
Distillation"* (arXiv:2602.03139). The output is a normal LoRA file: no
inference-side code, you just run it at `--infer_steps <student_steps> --cfg 1.0`.

> This doc is the structural walkthrough — the diversity-anchor / DMD gradient
> split, the velocity↔x0 conversion that makes it work on a flow-matching DiT, the
> per-step schedule, and the co-LoRA capacity argument. For the usage / ops /
> decision-log reference (config knobs, `make` targets, metrics to watch,
> current status), see `docs/methods/turbo.md`.
>
> History. This replaced the CA-decoupled DMD2 objective ("CFG-as-Spear,
> Distribution-Matching-as-Shield", Liu et al. arXiv:2511.22677) on 2026-05-30.
> The CA branch never reached a fixed point — the whole turbo program had been
> spent managing its standing CFG bias — and every CA-side lever came back inert or
> harmful. DP-DMD removes the CA branch entirely and recovers diversity with an
> explicit anchor instead. The CA-era decision log survives at
> `_archive/proposals/dmd2_decoupled_improvements.md`; the migration proposal at
> `_archive/proposals/dpdmd.md`.

---

## 1. Why distill at all, why DMD, and why "diversity-preserved"

The teacher is good but slow: 28 sampling steps × 2 forwards per step (cond +
uncond for CFG=4) = 56 DiT forwards per image. A distilled student that matches it
at 4 steps × 1 forward (CFG baked in) is a ~14× inference cut.

The naive distillation target — "regress the student's few-step trajectory onto the
teacher's many-step trajectory" — is brittle: it locks the student to one solver and
one step count, and small errors compound across the compressed trajectory. DMD
(Distribution Matching Distillation) instead matches distributions: it asks the
student's output distribution to look like the teacher's, scored by a moving "fake"
model that learns what the student currently produces.

But DMD's gradient is a reverse-KL, and reverse-KL is mode-seeking. Left to
itself it collapses the student onto the teacher's dominant modes: faces, poses, and
compositions converge across seeds, and fine off-mode structure (text, texture
tails) gets pulled to the mean. The CA-era code fought this collapse at the
symptom with a per-sample gradient normalization (`dm_x0_norm`, §7) — it helped,
but the collapse was structural, not a scaling artifact.

DP-DMD removes the collapse at the source. The first step of a few-step rollout
is what fixes the *mode* — pose, layout, global composition — out of pure noise; all
later steps refine within that mode. So DP-DMD supervises only the first step
toward the teacher's own diverse trajectory (a mode-preserving map), detaches it
from the DMD gradient, and lets DMD do mode-seeking refinement on the remaining
steps where mode-seeking is exactly what you want (sharper, on-manifold detail). The
diversity lives in step 1; the quality lives in steps 2..N; the two never share a
gradient.

---

## 2. The two gradients (the whole idea)

The student carries two losses with disjoint graphs:

$$
\nabla_\theta \mathcal{L}
= \underbrace{\lambda\,\nabla_\theta \big\|v_\text{first} - v_\text{target}\big\|^2}_{\text{diversity — first step only, detached}}
\;+\;
\underbrace{\nabla_\theta \mathcal{L}_\text{DMD}(x_\theta)}_{\text{quality — steps }2..N}
$$

- Diversity (first step). $v_\text{target}$ is the teacher's own CFG-guided
  K-step trajectory velocity (§5). Regressing the student's first-step velocity onto
  it is a plain MSE — a mode-preserving map that inherits the teacher's seed-to-
  seed spread of poses and compositions. This term is backwarded and severed
  before the DMD chain is built (`detach_after_first`, load-bearing — their Fig 5:
  preference rises while diversity falls if the DMD reverse-KL is allowed to flow
  back into the first-step mapping).

- Quality (steps 2..N). Standard DMD: a reverse-KL on the rollout endpoint
  $x_\theta$, expressed as a real−fake score gap and applied via the DMD2 gradient
  trick (§4). Mode-seeking, but now confined to refinement within the mode step 1
  already chose.

This is the structural difference from the retired CA-decoupled DMD2: there the two
terms were CA (CFG-augmentation, the "spear") and DM (distribution-matching, the
"shield"), both acting on the same single-call generator endpoint and scheduled to
different renoise levels. DP-DMD has no CA branch — CFG is folded back into the
single DMD real score (§5) — and its split is along the rollout step axis (first
vs. rest), not the gradient-term axis.

---

## 3. Three roles, one frozen DiT

```
                          frozen Anima DiT  (no grad, ~5 GB bf16)
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
   teacher view              student view                fake view
   both LoRAs OFF            student ON, fake OFF         fake ON, student OFF
   → base velocity           → v_student → x_θ            → v_fake_cond_dm
   (CFG'd at α externally)   (the N-step rollout)         (the score tracker)
```

`networks/methods/turbo_dmd.py::TurboDMDNetwork` owns two ordinary `LoRANetwork`
instances. Both `apply_to(unet)`, chaining onto every targeted Linear's forward:

```
linear(x)  →  fake.forward  →  student.forward  →  original_linear.forward
```

Each `LoRAModule` short-circuits at `not self.enabled`, so a view switch is just
`set_enabled(bool)` on each network — an O(num_modules) Python flag flip, negligible
beside a DiT forward. `set_view` flips only what changed, and short-circuits when the
target view is already active (consecutive teacher forwards — the CFG cond+uncond
pair — don't repay the loop).

`freeze_dit` runs after both `apply_to`'s and walks only the base params (names
not prefixed by a LoRA module), so it freezes the backbone without zeroing the LoRA
grads. The DiT is never an optimizer target; there are two AdamW states, one per LoRA.

### Co-LoRA capacity: why $r_\text{fake} \ge r_\text{student}$

This is the counter-intuitive bit. Usual LoRA intuition says smaller rank =
better-regularized. Here the fake is not a generator — it's a score tracker. Its
DM term $(s^\text{real}_\text{cond} - s^\text{fake}_\text{cond})$ is the corrective
signal; if the fake under-fits the student's current output distribution, DM
produces noisy gradients and the quality refinement stops landing on-manifold (the
paper's high-frequency checkerboard that compounds over training). The student LoRA
at rank $r_s$ defines a manifold of perturbed scores; the fake needs to be at least
$r_s$-expressive to track it pointwise. So $r_\text{fake} \ge r_\text{student}$ is a
capacity floor on the regularizer, not a stability prior — the config validator
warns when it's violated. (Defaults ship `student_rank = fake_rank = 48`.)

---

## 4. The velocity↔x0 conversion (what makes it work on Anima)

The paper is written for score/ε-prediction. Anima predicts velocity
$v = \varepsilon - x_0$ on the flow-matching path $x_t = (1-t)\,x_0 + t\,\varepsilon$
(see `docs/findings/asymflow_parameterization.md`; $t$ and $\sigma$ are the same
noise level in $[0,1]$ here — $\sigma=1$ pure noise, $\sigma=0$ clean). Everything
has to be re-derived in velocity/x0 space or the signs and scales are wrong.

The student is a genuine N-step Euler rollout from pure noise — not the DMD2
single-call generator the CA-era code used. From $x = \varepsilon$ at $\sigma = 1$,
each step takes one Euler stride down a static σ-grid (`get_timesteps_sigmas`, shifted
by `flow_shift` to match inference):

$$
x \leftarrow x - (\sigma_i - \sigma_{i+1})\,G_\theta(x, \sigma_i, c),
\qquad i = 0 \dots N-1
$$

The endpoint $x_\theta$ (= the clean image) is what DMD scores. We do not regress
onto the teacher's full trajectory — only the first step is teacher-anchored (§5);
steps 2..N are shaped by distribution-matching alone.

Re-noising primitive. To score $x_\theta$ at a fresh noise level $\tau$:

$$
x_\tau = (1-\tau)\,x_\theta + \tau\,\varepsilon, \qquad \varepsilon \sim \mathcal N(0, I)
$$

— the same forward path, applied to the predicted clean image instead of a dataset
latent (`renoise()` in `scripts/distill_turbo/primitives.py`). $x_\theta$ is
detached into the renoise so the teacher/fake forwards build no graph; the
student gradient enters through the surrogate below, not through the renoise input.

Sign / scale of the update. The DM delta is a velocity gap; the
distribution-matching update acts in x0 space. Converting a velocity gap to its
x0 gap at level $\tau$ picks up a $-\tau$ factor:

$$
\Delta_\text{dm} = s^\text{real}_\text{cond}(x_\tau) - s^\text{fake}_\text{cond}(x_\tau),
\qquad
x_0^\text{real} - x_0^\text{fake} = -\tau_\text{dm}\,\Delta_\text{dm}
$$

We want $x_\theta$ to move toward $x_0^\text{real}$, so the surrogate-loss
gradient on $x_\theta$ must be the positive combination $\tau_\text{dm}\Delta_\text{dm}$;
gradient descent then steps along its negative — the desired direction:

$$
\texttt{grad\_signal} = \tau_\text{dm}\,\Delta_\text{dm}
$$

and the DMD2 grad trick assembles a dummy scalar whose $\partial/\partial x_\theta$
equals it:

```python
loss_dmd = (grad_signal.detach() * x_theta).mean()   # ∂/∂x_θ = grad_signal
loss_dmd.backward()                                   # walks x_θ → steps 2..N → θ
```

> **The sign of this term is load-bearing and was once inverted.** Before the
> 2026-05-27 fix the student gradient pointed the wrong way (anti-distill). The tell
> is subtle: inverted-sign runs look like *"base few-step blur / never trained,"*
> not a blow-up. See [[project_turbo_dmd_sign_fix]].

---

## 5. The DP-DMD per-step schedule

Each training step is: teacher anchor → diversity-supervised first step → DMD-refined
rollout → fake update.

### 5.1 Teacher K-step CFG anchor → $v_\text{target}$ (no grad)

From the shared start noise $\varepsilon$, roll the teacher (CFG-guided,
$v_u + \alpha\,(v_c - v_u)$, `teacher_cfg`$=\alpha=4$) for `k_anchor` Euler steps on
the `teacher_anchor_steps` grid to an intermediate latent $z_{t_k}$ at continuous
time $t_k$. The first-step diversity target is the average velocity over
$[t_k, 1]$:

$$
v_\text{target} = \frac{\varepsilon - z_{t_k}}{1 - t_k}
$$

(Euler integrates $z_{t_k} = \varepsilon - (1-t_k)\,\bar v$, so $\bar v$ is exactly
this.) $t_k$ is read from the teacher grid, not the student grid — a σ mismatch
silently mis-scales $v_\text{target}$. This anchor is the diverse landing point: the
teacher's CFG-guided trajectory preserves the seed-to-seed spread of poses and
compositions, so regressing onto it transfers that spread to the student's first
step.

### 5.2 Student first step — diversity-supervised, then detached

Step 0 rolls from $\varepsilon$ at $\sigma = 1$:

$$
v_\text{first} = G_\theta(\varepsilon, 1, c), \qquad
\texttt{div\_loss} = \big\|v_\text{first} - v_\text{target}\big\|^2
$$

Under `detach_after_first` (load-bearing, default true) the diversity term is
backwarded immediately and the step-0 graph is severed before the DMD chain
builds its own:

```python
(div_weight * div_loss).backward()    # accumulates into student .grad now
x = x.detach().requires_grad_()        # fresh grad-ckpt root for steps 2..N
```

Two reasons this is correctness work, not just memory hygiene:

1. Diversity must not see the DMD reverse-KL. If the mode-seeking gradient from
   steps 2..N flows back into the first-step mapping, it un-does the very diversity
   the anchor installed (their Fig 5).
2. Peak activations stay at one forward. The two losses share no graph, so a
   single combined backward would pay 2× activation memory for nothing. Severing lets
   grad-ckpt stay *optional* at small $N$ — the unroll it tames is the $N{-}1$-step DMD
   chain, not the freed step-0 graph (see [[project_custom_down_autograd_distill_lever]]).

With the detach off (A/B only) the graphs are entangled and the diversity term
must ride the single combined backward at assembly time.

### 5.3 DMD on $x_\theta$ — CFG-guided real score (the un-decoupling)

Steps 2..N roll on with grad to $x_\theta$. Then DMD scores it at $\tau_\text{dm}
\sim \mathcal U[0,1]$ (§4). The one un-decoupling versus the CA-era code: the real
score is CFG-guided

$$
s^\text{real}_\text{cond} = v_u + \alpha\,(v_c - v_u)
$$

— not cond-only. The old DM branch was deliberately unguided because CFG lived in
the separate CA branch; with CA gone, guidance has to ride the single DMD real score
(matching the reference `compute_dmd_loss`). Without it $v_\text{real} \approx
v_\text{fake}$ (`dm_cos ≈ 0.9999`) and the quality gradient is noise. The fake
stays cond-only. The teacher uncond is the T5("") sidecar
(`library/anima/uncond.py`), not a zero tensor — a zero crossattn is
fed-out-of-distribution and the resulting $v_\text{real,uncond}$ amplified at
$(\alpha-1)=3\times$ drives the student off-manifold (saturated white output).

---

## 6. The fake update (keeping the shield sharp)

The fake learns to denoise the student's current output distribution by plain
flow-matching regression:

$$
\tau_\text{fake} \sim p(\tau), \quad
x_\tau^\text{fake} = (1-\tau_\text{fake})\,x_\theta^\text{detach} + \tau_\text{fake}\,\varepsilon,
\quad
\mathcal L_\text{fake} = \big\|\,\text{fake}(x_\tau^\text{fake}, \tau_\text{fake}, c) - (\varepsilon - x_\theta^\text{detach})\,\big\|^2
$$

Run `fake_steps_per_student_step` (default 1) inner steps against the same
$x_\theta^\text{detach}$, resampling $(\tau, \varepsilon)$ each time. Standard DMD2
practice is to give the fake extra iterations (the reference uses 5): its target
distribution is moving as the student sharpens, and a lagging critic produces noisy
DM gradients — raise this knob when `dm_cos` drifts high. The fake's cosine LR schedule
anneals over `iterations × fake_steps_per_student_step + fake_warmup_steps`, not the
student's count.

Fake (critic) head-start. A `fake_warmup_steps` block of fake-only updates runs
before the main loop, calibrating the zero-init fake against the student's
(init ≈ teacher) $x_\theta$ distribution so the critic is ready before the student LR
warmup ramps. This kills the early `grad_signal_rms` spike (~step 50). The student is
untouched during it. The fake LR warmup span is sized over the same total, so the
fake enters the main loop already calibrated and at full LR.

---

## 7. DM-grad normalization: two policies, not additive

There are two ways to scale the DM term, and they are alternatives, not a stack.

- (a) τ-damping — use $\tau_\text{dm}\,\Delta_\text{dm}$ directly.
- (b) DMD per-sample x0-norm — original DMD normalizes the DM gradient per-sample
  for scale-invariance. The DM x0-gap is $-\tau\,v^\text{real}_\text{cond,dm}$, whose
  per-sample magnitude is $\text{denom} = \tau\cdot\overline{|v^\text{real}|}$:

  ```python
  denom   = (tau_dm_e * v_real_cond_dm).abs().mean(dim=(1,2,3), keepdim=True).clamp_min(norm_floor)
  grad_dm = (tau_dm_e * delta_dm) / denom        # ≈ Δ_dm / mean|v_real|
  ```

The subtlety: because $\text{denom} \approx \tau\cdot\overline{|v^\text{real}|}$, the
$\tau$ cancels across the bulk of the range — (b) is therefore $\approx$ "drop the
$\tau$-weight and magnitude-normalize." The `clamp_min(norm_floor)` only bites for
$\tau < \texttt{norm\_floor}/\overline{|v^\text{real}|}$ (a thin sliver).

| Policy | What it is |
|---|---|
| (a) | $\tau$-damping only — shipped default (`dm_x0_norm=false`) |
| (b) | DMD magnitude-normalization ($\tau$ roughly cancels) — opt-in via `dm_x0_norm=true` |
| (c) | both ≈ (b) with $\tau$ re-multiplied in — do not ship believing it composes |

How to think about the choice. (b)'s per-sample normalization gives every sample
a unit-scale direction → even distribution-matching pressure; in the CA-era A/B it
was the clearest win for fine structure (text rendering especially), and it was once
over-credited as the diversity fix. Under DP-DMD the diversity-collapse story
moved to the anchor (§1, §5) — first-step supervision now carries pose/composition
spread — which demoted `dm_x0_norm` to a pure scale-invariance lever on the quality
term, and the shipped default reverted to plain $\tau$-damping. Reach for (b) when
fine detail / text degrades late in a run; never combine it with (a) expecting the
two to stack.

---

## 8. Masked loss (student-only)

With `use_masked_loss=true` the per-image foreground mask multiplies the student
DMD gradient so distribution-matching focuses on the subject:

```python
loss_dmd = (grad_signal * x_theta * mask).mean()   # background latents → zero student push
```

The fake/critic regression is left full-frame (it still needs to model the whole
distribution), and the mean-variance reg (§5.4) stays full-frame too (it's a global
clamp). Normalization stays `/numel` (no renorm by mask area, matching
`apply_masked_loss`), so a masked run sees a lower effective gradient by design.

---

## 9. What's frozen at inference, and why it ships as a plain LoRA

`TurboDMDNetwork.save_student` serializes only the student in the standard
plain-LoRA layout (`save_variant="standard"`); the fake is training scaffolding and
never shipped. (The off-by-default `per_step_expert` variant — a shared-down student
with one up-head per rollout step — saves a bespoke step-expert layout instead: not
a plain LoRA, kept live at inference, refused by merge.) The default student is an
ordinary rank-$r$ LoRA with CFG=4 baked in — load
it through the normal inference path and run `--infer_steps <student_steps> --cfg 1.0`
(currently 4). No turbo code runs at inference.

Step count: the student is trained at `student_steps` (currently 4), so that step
count is the matched schedule. But an under-trained student behaves like a continuous
velocity field (the DMD quality loss is trained at random $\tau$; only step 0 is
grid-anchored), so it can integrate better at more Euler steps than its trained
grid. If a checkpoint looks better at more steps than it was trained for, that's the
tell that distillation hasn't reached a true N-step map yet — train longer or raise
`student_steps`. Always keep `--cfg 1.0` (CFG is baked; don't
double-guide). See [[project_sigma_signal_resolves_by_045]] for why the σ≈0.5 band is
where the extra evaluation matters.

Consequences of the plain-LoRA bake (the load-bearing constraint):

- Composes with concept LoRAs linearly (ranks add), same model surgery as
  LCM-LoRA + style LoRA.
- Cannot carry anything needing a step-size/per-t input at inference (Shortcut /
  MeanFlow Δt-conditioning; timestep-conditioned T-LoRA whose mask is training-only,
  [[project_tlora_inference_full_rank]]). A plain LoRA must average antagonistic
  per-$t$ corrections — true multi-stride robustness is out of scope.
- Incompatible with Spectrum (Chebyshev cache assumes ≥16 steps).

---

## 10. Minimal mental model

1. Two LoRAs on one frozen DiT. Student = generator $G_\theta$ (an N-step Euler
   rollout); fake = score tracker. View-toggle by enabling/disabling each per forward.
2. DMD's reverse-KL is mode-seeking → it collapses seed diversity (pose,
   composition). DP-DMD fixes the mode in step 1 and lets DMD refine steps 2..N.
3. First step = diversity, detached. Supervise $v_\text{first}$ toward the
   teacher's CFG-guided K-step anchor velocity, backward it, then sever the graph so
   the DMD reverse-KL can't leak into the diversity map.
4. Steps 2..N = DMD quality. Renoise $x_\theta$ to a fresh $\tau$; the x0-space
   update picks up a $\tau$ factor; its sign must point $x_\theta$ toward $x_0^\text{real}$.
5. The DMD real score is CFG-guided ($v_u + \alpha(v_c-v_u)$) — the one
   un-decoupling now that the CA branch is gone; the fake stays cond-only.
6. Fake stays ahead via the pre-loop head-start (and extra inner steps when the
   critic lags); $r_\text{fake} \ge r_\text{student}$.
7. DM grad scaling: τ-damping (a) is the shipped default; per-sample x0-norm (b)
   is the opt-in scale-invariance lever — the two are alternatives, never stacked
   (the diversity job lives in the anchor, not here).
8. Ships as a plain student LoRA, CFG baked in, run at `student_steps` (4) with
   CFG=1.

---

*(A schematic for `docs/structure_images/dpdmd.png` — the three-role frozen-DiT
diagram plus the diversity-anchor / DMD step split — is still to be drawn; the ASCII
diagram in §3 is the interim reference. The retired CA-era schematic lives at
`docs/structure_images/dpdmd.png`.)*
