# Ideas from data-space iteration (arXiv:2610.12102)

Source: Lin, Peng, Wang, Fan, *Few-Step Generation via Data-Space Iteration*
(ByteDance Seed, 2026-10-08). Read 2026-10-13. Nothing here has been run. The current
loop is described in `docs/anima/methods/turbo.md`.

## The paper

Few-step DMD generators iterate in flow space. They either re-noise each x0
prediction to the next schedule point (DMD2, consistency models), or map
`t_r → t_{r+1}` directly (Phased DMD, MeanFlow). In both cases a hand-chosen
schedule sets the states that self-rollout training sees, so trying another
schedule means training another generator.

Data-space iteration drops the schedule. It starts from `x̂0⁽⁰⁾ = 0` and repeats
`x̂0⁽ʳ⁺¹⁾ = G(x̂0⁽ʳ⁾, z, r)`, with the same noise z at every step and no
re-noising. The generator is the pretrained velocity net with three changes:
one new input projection for `x̂0⁽ʳ⁾`, summed with z's; an r embedding on the
timestep path; and t fixed at 1. The output is `z − g(x̂0, z, r, t=1)`. The DMD
loss applies to every r in `0..R−1`, so every prefix of the rollout is a valid
sample. Training is otherwise plain DMD: real and fake flows, fake model
conditioned on r, the same compute as a matched DMD baseline.

ImageNet 256, SiT-XL/2 teacher, R = 4, 40 epochs, no GAN term:

| | 1 NFE | 2 NFE | 4 NFE |
|---|---|---|---|
| DMD (re-noise, uniform schedule) | 4.45 | 3.22 | 2.75 |
| Phased DMD (flow-map, uniform / best of 3 schedules) | – | – | 2.55 / 2.48 |
| data-space | **3.85** | **2.68** | **2.46** |
| data-space + CFG 1.1 on τ ∈ [0, 0.6] | 3.31 | 2.47 | **2.29** |

Fig. 2: the data-space rollout keeps the composition and refines it. The re-noise
rollout redraws content between steps. The best schedule shift γ was different
for each method and for each CFG setting, with no pattern across them.

## Ideas, cheapest first

### 1. τ importance sampling on the DM query (Table 1a, 2a)

The paper's best result samples the DMD noising time τ from **Beta(2.5, 1)**
(density ∝ τ^1.5, weighted toward high noise; t = 1 is noise) with no loss
weight. This beat every loss weight it tried under uniform τ, including the
standard DMD weight (dynamic normalisation: 4.24 / 3.22 at 1 / 4 NFE, against
3.85 / 2.46). The paper's explanation is that sampling puts more of the batch on
the τ that carry the objective, where a weight scales down part of a uniform
batch. Sampling τ also gives `G` and `v_fake` the same τ distribution.

Ours: the DM query draws τ ~ U(0, 1) on purpose (`steps.py` DM term:
`distribution="uniform"`, independent of `t_distribution`), with
`dm_x0_norm = true` (the per-sample normalisation, i.e. the "DMD weighting" row).

Arm: the query τ ~ Beta(2.5, 1), `dm_x0_norm` off, the fake critic's τ drawn from
the same Beta, everything else as shipped. Watch:
- the `norm_floor` / `dm_x0_norm` interaction. Turning norm off changes the
  gradient scale, so the student lr may need retuning;
- `fake_tau_banks = 2`: under Beta(2.5, 1) the 0.5 boundary no longer splits
  updates evenly.

### 2. Adam β1 = 0 for the student (Table 1c)

`(0.0, 0.95)` beat `(0.9, 0.95)` at every NFE (4 NFE 2.46 vs 2.66). The official
DMAD H3 recipe also runs its student at β (0, 0.99) (`_archive/turbo_dmad/difference.md`, Student
table). That is two independent sources. Ours: student and fake at the AdamW
default (0.9, 0.999). The disc is already at (0, 0.99).

Arm: student β (0, 0.99), with the fake left alone first. The fake runs
4 updates per student step, so its momentum works differently. One flag, and it
can run alongside any other arm.

### 3. Guidance interval on the real score (Table 2b)

With CFG on, applying it only on τ ∈ [0, 0.6] (the low-noise side) gave the best
result for every method in the paper. Full-range CFG 1.2 hurt data-space
iteration at 4 NFE (3.27 vs 2.46 without CFG).

Ours: `teacher_cfg = 4` on every τ, and that is the guidance baked into the
student. Cutting the range changes how much guidance gets baked, so the
inference look moves too, not only the score's quality. Lower priority than 1–2,
and it needs a render read, not a scalar read. If tried: CFG 4 on τ ≤ 0.6, and
cond-only above.

### 4. The OT regulariser does not help (Table 1d)

`λ_ot‖G(x̂0⁽ʳ⁾, z, r) − z‖²` at 1e-6 was slightly worse at every NFE. This counts
against adding a "stay near the noise / stay near the previous step" anchor as a
consistency term. Our step-0 diversity anchor is a different thing (it pins step
0 to the teacher's K-step anchor) and is not affected.

### 5. Data-space iteration itself (structural; not now)

Porting it means a second patch-embed input for `x̂0⁽ʳ⁾`, an r embedding in place
of t, and a rollout that passes `x̂0` and not `x_t`. Costs:
- **The output stops being a normal LoRA.** It needs the extra input projection
  and its own sampler loop. ComfyUI / `make merge` / the published
  `anima-turbo-4step` path all assume a LoRA on the stock DiT with an Euler
  sampler.
- The paper tests only class-conditional ImageNet 256. Text-to-image is left as
  future work.
- Our Euler rollout is the flow-map family, and `dynamic_schedule` (CDM, random
  grid per iteration) already removes the fixed-schedule dependence the paper is
  trying to get rid of. The paper's best-of-3 flow-map trails data-space by only
  0.02 FID without CFG. The gap that is left is mostly the 1–2 NFE prefixes,
  which we do not ship.

The paper does back one current design choice: schedule shift had no
predictable effect across methods. That is the reason for `dynamic_schedule`
over tuning a fixed `flow_shift` grid.

## Suggested order

Ideas 1 and 2 are each a single flag, and they touch different parts of the
loop. Run them as two 500-step arms against the p0b DP-DMD 500 renders
on `bench/turbo/real_prompts.txt` at seed 42, and add a
`val/div_*` read for diversity. Try 3 only if 1 or 2 moves the result. Leave 5
alone unless a non-LoRA turbo is ever on the table.
