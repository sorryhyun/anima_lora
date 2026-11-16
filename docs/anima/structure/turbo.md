# DP-DMD: diversity-preserved few-step distillation

Distills the 28-step, CFG=4 Anima teacher into an N-step student LoRA (`student_steps=4`
ships). The student and the auxiliary "fake" score model are plain LoRAs on one frozen
DiT; only the rank-$r$ adapters train. Port of Wu, Li, Zhang, Ma, *Diversity-Preserved
Distribution Matching Distillation* (arXiv:2602.03139).

This page covers the structure and the math. Knobs, `make` targets, metrics and
inference settings: `docs/anima/methods/turbo.md`.

---

## 1. Goal and the collapse it has to avoid

The teacher costs 28 steps × 2 forwards (cond + uncond) = 56 DiT forwards per image. A
student matching it at 4 steps × 1 forward (CFG baked in) is a ~14× inference cut.

Regressing the student's few-step trajectory onto the teacher's locks it to one solver
and step count, and errors compound across the compressed trajectory. DMD (Distribution
Matching Distillation) matches distributions instead: the student's output distribution
is scored against the teacher's by a "fake" model that learns what the student currently
produces.

DMD's gradient is a reverse-KL, which is mode-seeking: on its own it collapses faces,
poses and compositions across seeds and pulls off-mode detail (text, texture tails) to
the mean.

The first step of a few-step rollout fixes the mode (pose, layout, composition) out of
pure noise; later steps refine within it. DP-DMD therefore supervises only the first step,
toward the teacher's own diverse trajectory, detaches it from the DMD gradient, and lets
DMD's mode-seeking act on the refinement steps, where it sharpens on-manifold detail.

---

## 2. The two gradients

The student carries two losses with disjoint graphs:

$$
\nabla_\theta \mathcal{L}
= \underbrace{\lambda\,\nabla_\theta \big\|v_\text{first} - v_\text{target}\big\|^2}_{\text{diversity: first step only, detached}}
\;+\;
\underbrace{\nabla_\theta \mathcal{L}_\text{DMD}(x_\theta)}_{\text{quality: refinement steps}}
$$

- Diversity (first step). $v_\text{target}$ is the teacher's CFG-guided K-step trajectory
  velocity (§5.1). Regressing the student's first-step velocity onto it is a plain MSE that
  inherits the teacher's seed-to-seed spread. It is backwarded and severed before the DMD
  chain is built (`detach_after_first`, load-bearing: in the paper's Fig 5, preference
  rises while diversity falls when the DMD reverse-KL flows back into the first step).
- Quality (refinement steps). DMD on the student's x0-prediction $x_\theta$, expressed as
  a real − fake score gap and applied through the DMD2 gradient trick (§4).

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

Each `LoRAModule` short-circuits at `not self.enabled`, so a view switch is
`set_enabled(bool)` on each network, an O(num_modules) flag flip. `set_view` flips only
what changed and returns early when the view is already active (the CFG cond + uncond
teacher pair).

`freeze_dit` runs after both `apply_to` calls and walks only the base params (names not
prefixed by a LoRA module), so the LoRA params keep their grads. There are two AdamW
states, one per LoRA.

### Co-LoRA capacity: $r_\text{fake} \ge r_\text{student}$

The fake is a score tracker, not a generator. Its gap
$(s^\text{real}_\text{cond} - s^\text{fake}_\text{cond})$ is the corrective signal; if the
fake under-fits the student's current output distribution, the DM gradient turns noisy and
refinement stops landing on-manifold (the paper's high-frequency checkerboard, compounding
over training). A rank-$r_s$ student spans a manifold of perturbed scores, and the fake
needs at least that rank to track it pointwise. The config warns when
$r_\text{fake} < r_\text{student}$.

---

## 4. The velocity↔x0 conversion

The paper is written for score / ε-prediction. Anima predicts velocity
$v = \varepsilon - x_0$ on the flow-matching path $x_t = (1-t)\,x_0 + t\,\varepsilon$ ($t$
and $\sigma$ are the same noise level in $[0,1]$: $\sigma=1$ pure noise, $\sigma=0$ clean),
so signs and scales are re-derived in velocity / x0 space.

The student is an N-step Euler rollout from $x = \varepsilon$ at $\sigma = 1$ down a σ-grid
shifted by `flow_shift` to match inference:

$$
x \leftarrow x - (\sigma_i - \sigma_{i+1})\,G_\theta(x, \sigma_i, c),
\qquad i = 0 \dots N-1
$$

Backpropagating through the whole rollout would cost N grad-bearing forwards. Each
iteration instead draws one grad step $g \sim \mathcal U\{1..N-1\}$, runs steps $1..g-1$
under no_grad, and takes step $g$'s one-step x0-prediction with grad:

$$
x_\theta = x_g - \sigma_g\,G_\theta(x_g, \sigma_g, c)
$$

Memory stays at one grad-bearing forward and every refinement grid point is supervised
over the run. $x_\theta$ is what DMD scores.

Re-noising primitive. To score $x_\theta$ at a fresh level $\tau$:

$$
x_\tau = (1-\tau)\,x_\theta + \tau\,\varepsilon, \qquad \varepsilon \sim \mathcal N(0, I)
$$

(`renoise()` in `library/training/forward/renoise.py`). $x_\theta$ is detached into the
renoise, so the teacher and fake forwards build no graph; the student gradient enters
through the surrogate below.

Sign and scale. The DM delta is a velocity gap; the update acts in x0 space. Converting a
velocity gap to its x0 gap at level $\tau$ picks up a $-\tau$ factor:

$$
\Delta_\text{dm} = s^\text{real}_\text{cond}(x_\tau) - s^\text{fake}_\text{cond}(x_\tau),
\qquad
x_0^\text{real} - x_0^\text{fake} = -\tau_\text{dm}\,\Delta_\text{dm}
$$

$x_\theta$ must move toward $x_0^\text{real}$, so the surrogate's gradient on $x_\theta$ is
the positive combination, and gradient descent steps along its negative:

$$
\texttt{grad\_signal} = \tau_\text{dm}\,\Delta_\text{dm}
$$

The DMD2 grad trick builds a scalar whose $\partial/\partial x_\theta$ equals it:

```python
loss_dmd = (grad_signal.detach() * x_theta).mean()   # ∂/∂x_θ = grad_signal
loss_dmd.backward()                                   # walks x_θ → step g → θ
```

> **The sign of this term is load-bearing and was once inverted.** An inverted-sign run
> looks like base few-step blur that never trained, not a blow-up.

---

## 5. The per-step schedule

Teacher anchor → diversity-supervised first step → rollout to the grad step → DMD →
fake update.

### 5.1 Teacher K-step CFG anchor → $v_\text{target}$ (no grad)

From the shared start noise $\varepsilon$, the teacher rolls (CFG-guided,
$v_u + \alpha\,(v_c - v_u)$, `teacher_cfg` $=\alpha$) for `k_anchor` Euler steps on the
`teacher_anchor_steps` grid to $z_{t_k}$ at time $t_k$. The first-step target is the mean
velocity over $[t_k, 1]$:

$$
v_\text{target} = \frac{\varepsilon - z_{t_k}}{1 - t_k}
$$

Euler integrates $z_{t_k} = \varepsilon - (1-t_k)\,\bar v$, so $\bar v$ is exactly this.
$t_k$ is read from the teacher grid, not the student grid; a σ mismatch silently
mis-scales $v_\text{target}$.

### 5.2 Student first step: diversity-supervised, then detached

Step 0 rolls from $\varepsilon$ at $\sigma = 1$:

$$
v_\text{first} = G_\theta(\varepsilon, 1, c), \qquad
\texttt{div\_loss} = \big\|v_\text{first} - v_\text{target}\big\|^2
$$

Under `detach_after_first` (default true) the diversity term is backwarded at once and
the graph severed before the grad step builds its own:

```python
(div_weight * div_loss).backward()     # student .grad now holds the diversity term
x_g = x.detach().requires_grad_()      # step g starts from a fresh leaf
```

This keeps the mode-seeking DMD gradient out of the first-step mapping (Fig 5), and
frees step 0's activations before step $g$ runs. With the detach off (A/B only), or when
$g = 0$ under `dm_first_step`, the diversity term rides the single final backward.

### 5.3 DMD on $x_\theta$ with a CFG-guided real score

DMD scores $x_\theta$ at $\tau_\text{dm} \sim \mathcal U[0,1]$ (§4). The real score is
CFG-guided,

$$
s^\text{real}_\text{cond} = v_u + \alpha\,(v_c - v_u),
$$

because it is the only place guidance enters (as in the reference `compute_dmd_loss`).
With a cond-only real score $v_\text{real} \approx v_\text{fake}$ (`dm_cos ≈ 0.9999`) and
the quality gradient is noise. The fake stays cond-only. The uncond input is the T5("")
sidecar, not zeros (`docs/anima/methods/turbo.md`).

---

## 6. The fake update

The fake learns to denoise the student's current output distribution by flow-matching
regression:

$$
\tau_\text{fake} \sim p(\tau), \quad
x_\tau^\text{fake} = (1-\tau_\text{fake})\,x_\theta^\text{detach} + \tau_\text{fake}\,\varepsilon,
\quad
\mathcal L_\text{fake} = \big\|\,\text{fake}(x_\tau^\text{fake}, \tau_\text{fake}, c) - (\varepsilon - x_\theta^\text{detach})\,\big\|^2
$$

`fake_steps_per_student_step` inner steps run against the same $x_\theta^\text{detach}$,
resampling $(\tau, \varepsilon)$ each time. The fake's target moves as the student
sharpens, and a lagging critic gives noisy DM gradients, so it gets more updates than the
student (the reference uses 5). A `fake_warmup_steps` head-start trains the fake alone
before the main loop (`scripts/distill_turbo/warmup.py`).

---

## 7. DM-grad normalization: two policies, not additive

- (a) τ-damping: use $\tau_\text{dm}\,\Delta_\text{dm}$ directly.
- (b) Per-sample x0-norm, as in the original DMD. The DM x0-gap is
  $-\tau\,v^\text{real}_\text{cond,dm}$, whose per-sample magnitude is
  $\text{denom} = \tau\cdot\overline{|v^\text{real}|}$:

  ```python
  denom   = (tau_dm_e * v_real_cond_dm).abs().mean(dim=(1,2,3), keepdim=True).clamp_min(norm_floor)
  grad_dm = (tau_dm_e * delta_dm) / denom        # ≈ Δ_dm / mean|v_real|
  ```

Because $\text{denom} \approx \tau\cdot\overline{|v^\text{real}|}$, $\tau$ cancels over most
of the range: (b) is roughly "drop the $\tau$-weight and magnitude-normalize". The
`clamp_min(norm_floor)` bites only for $\tau < \texttt{norm\_floor}/\overline{|v^\text{real}|}$.

| Policy | What it is |
|---|---|
| (a) | $\tau$-damping only: the code default (`dm_x0_norm=false`) |
| (b) | magnitude-normalization ($\tau$ roughly cancels): `dm_x0_norm=true`, what `configs/methods/turbo.toml` ships |
| (c) | both ≈ (b) with $\tau$ re-multiplied in; the two do not stack |

(b) gives every sample a unit-scale direction, which helped fine structure (text
especially). Diversity is carried by the anchor (§5), not by this choice.

---

## 8. What ships

`TurboDMDNetwork.save_student` writes only the student, in the standard plain-LoRA layout;
the fake (and the GAN disc) are discarded. The result is an ordinary rank-$r$ LoRA with
CFG=4 baked in, run at `--infer_steps <student_steps> --cfg 1.0` through the normal LoRA
path. Step count and composition limits: `docs/anima/methods/turbo.md`.
