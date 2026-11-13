# Ideas outside the arms in `plan.md`

Written 2026-11-12, trimmed 2026-11-13 after the v1 run and the step-0 fit probes
(`reports/step0_fit.md`). None is built or run unless it says so. Since the step-0 fit turned out to
be what carries the layout, and Hub v3 already has it, ideas 1 and 2 only matter if a student has
`dm_first_step` on.

## 1. The anchor's weight against the DM signal on step 0

With `dm_first_step` on, step 0 takes two signals: the anchor regression, which ties the layout to
the noise, and the DM gradient, which pulls toward one mode. S-c.1 moved the guitarist's pose on
every seed and drew one apron colour on all four café seeds (user's read: improved, less diverse).
The anchor enters at `div_weight 0.1` on every iteration, including the ones where step 0 is the DM
step. That ratio is the dial between the two reads.

- Arm: `div_weight` 0.3 or 0.5 with `dm_first_step` on. No code for the global value; a weight that
  applies only on the iterations where step 0 takes the DM gradient needs a switch.
- Cost: 250 iterations, about 25 min, read against S-c.1; or v1's length against v1.
- Worth running if v1 gains composition and loses diversity. If v1 keeps its diversity, it is not
  needed.
- DP-DMD's λ table (0.01–0.10, detached first step) shows diversity flat above 0.05. It says nothing
  about the weight once the DM gradient reaches step 0.

## 2. DM queries on step 0 at high σ only

Step 0 moves the state from σ 1 to 0.9 and only has to place the layout. Its one-step x̂₀ is blurred.
A DM query at low σ_a asks that blurred prediction for detail, which step 0 cannot give and steps
1–3 redraw anyway. The guess: those queries add mode-seeking pressure without a layout gain, and
restricting step 0's queries to high σ_a keeps the composition gain at a lower cost in diversity.

- Arm: on iterations where step 0 is the DM step, draw σ_a from the upper range only. Needs a
  switch beside `dmd.dm_tau_distribution`.
- No measurement supports the guess yet. On arm 0 the pre-clip norm is flat below σ_a 0.75 and
  higher above it (`reports/arm0.md`), so the high-σ_a queries are also the noisier ones.
- Second to idea 1: it needs code and idea 1 does not.

## 3. Handoff probe on a 2-step student

The 2-step grid's first step ends at σ 0.75, which is where the anchor target lands, so a step 0 at
zero anchor loss puts the 2-step student exactly on the teacher's σ 0.75 state. On the 4-step
students a handoff at σ 0.75 keeps 0.86–0.93 of the teacher's layout, against 0.70–0.80 at σ 0.9
(`reports/handoff.md`). No handoff probe was run on the 2-step line, so where it loses the layout is
not known.

- Probe: `bench/turbo/probe_handoff.py --student_steps 2` on `1111_1115` at 250, arms `S`, `A>S@1`,
  `T>S@1`. Forward only, about 7 min.
- A high `A>S@1` with a low `S` puts the 2-step loss in step 0, as on 4 steps, and whatever fixes
  step 0 there carries over. A low `A>S@1` puts it in the single step from σ 0.75 to 0, and the line
  stays closed.
- The 2-step students were trained with `rollout_grad`; their step 1 has seen their own step-0
  states, not the teacher's.

## 4. Low-frequency weighting of the anchor loss — closed

Measured 2026-11-13 (`reports/step0_fit.md`). The layout is in the low band of the step-0 residual
(the 8× band alone gives 85–96 % of the anchor's gain), but that band is already 59–68 % of the
students' error and is the part training lowers. A pooled loss has little to redirect.

## 5. A lower-resolution anchor (σ-demoted training applied here)

Looked at on 2026-11-12 on the user's question. It does not carry over as it stands; one narrow
variant is left.

- **The premise is missing.** `--sigma_lowres` trains a high-σ step on a smaller latent of the same
  image: same content, fewer tokens. The turbo student path has no image. A smaller ε gives the
  teacher a different sample with its own composition, not a smaller twin of the native one.
- **The map does not cover it.** The certified routes (896 above σ 0.5; 768 in σ 0.65–0.95, off-map
  at σ 1) certify the LoRA gradient of an FM step on re-encoded latents of the base model. The turbo
  terms at high σ are a teacher rollout, a regression at exactly σ 1, and DM / fake queries on
  generated samples.
- **DM, fake and GAN terms are out.** They act on the student's x̂₀. A smaller copy of it is a latent
  resize, not a VAE re-encode, and a decode → resize → encode per iteration costs more than it saves.
- **What is left: the anchor alone.** It runs at σ 0.75–1 and its gradient is already cut from the
  rollout (`detach_after_first`). The regression could take its own smaller ε, with the rollout
  keeping the native one. The anchor is about 2.0 s of a 5.75 s iteration (12 teacher forwards and
  one student forward and backward), so the saving is about 8 % of the iteration at 896 and 15 % at
  768, less one extra no-grad student forward.
- **The risk sits on the open problem.** A smaller canvas composes differently (larger, more centred
  subjects), and the gap this line is working on is exactly a large centred figure where the teacher
  draws a small one from below. Step 0 learning its layouts from smaller canvases may push the same
  way.
- **Check, if throughput becomes the limit:** the anchor-only loop at a smaller ε (it needs no
  sibling latents), 250 iterations, then the usual probe at 768 × 1344: held-out anchor MSE and
  `P>S@1` against S-b's 0.157 and 0.472 / 0.543. About 10 min. Throughput is not the limit now: the
  step-0 fit follows the LR, not the number of pairs (`reports/step0.md`).

## 6. DAVE on the student's step 0 at inference

Parked on the user's call. DC attenuation in blocks 8–18 during step 0 is aimed at the same symptom
(every seed drawing one composition) and is training-free, so `--dave auto` at strength 0.3 / 0.8 on
an existing EMA is a render-only check. It is a hook: it does not bake into the LoRA, and the mask
was calibrated on base Anima at CFG 4. On 4 steps the default window covers step 0, which spans the
first quarter of the teacher's steps, wider than the window the DAVE sweeps found safe for text and
hands.
