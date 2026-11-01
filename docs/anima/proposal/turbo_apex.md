# Turbo APEX — one-LoRA self-adversarial student with velocity + endpoint regression

Status: **closed 2026-11-01**, Phases 0b, 1 and 2 ran and Phase 2 hit its kill criterion:
[findings](../findings/turbo_apex_no_quality.md). Phase 0a was not run. Sources: Liu et al., *Self-Adversarial One
Step Generation via Condition Shifting* (APEX, arXiv:2604.12322); Liu et al., *Intrinsic
Distribution Matching* (IDM, arXiv:2610.39633), for the regression only. IDM's `L_pop` stays
out (`scripts/distill_turbo/report_gaps.md`). Dataset: the turbo set, `post_image_dataset/lora`
(3,242 TE caches).

## Motivation

The DP-DMD loop keeps three parties: student LoRA, fake LoRA, frozen teacher. Most of a step
goes to the teacher anchor (12 forwards), the DM term and 4 fake updates. L_CDM's extra graph
is what keeps cdm grad ckpt on: off, step 1 runs out of memory at 14.2 GB allocated, swap 0 or
14 (2026-11-01).

APEX puts the fake inside the student: one network under `c` is the generator and the
reference, and under `c_fake` it tracks the generator's own output. Real data enters as a
regression term, where DP-DMD uses it only in the 0.03 GAN term. The APEX paper reports a
LoRA at r32/64 on Qwen-Image reaching GenEval 0.83 in 400 steps at batch 64.

The question: on 3.2k images at batch 1, does this reach DP-DMD's 4-step quality with fewer
parties and a cheaper step?

## What endpoint regression adds

The x and v losses share one direction. `ℓ_x(t) = t²·ℓ_v(t)` (IDM Eq. 12, App. B.4; APEX
Eq. 18–19), so adding them changes the time weighting only. IDM's table (Z-Image, 2 NFE,
GenEval): v 76.64, x 77.26, v+x 77.81.

Our surrogate is already endpoint-shaped. `(g·x_pred).mean()` with `x_pred = x_g − σ_g·v_g`
puts a factor `σ_g` on `∂v_g`, before `g`'s own `τ` and `dm_x0_norm`. On the static 4-step
grid (σ_g = 0.9, 0.75, 0.5):

| Weight on the DMD correction | 0.9 | 0.75 | 0.5 | max/min |
|---|---|---|---|---|
| v only | 1 | 1 | 1 | 1.0 |
| v+x, `1 + σ²` | 1.81 | 1.56 | 1.25 | 1.45 |
| **ours, `σ`** | 0.9 | 0.75 | 0.5 | **1.8** |
| x only, `σ²` | 0.81 | 0.56 | 0.25 | 3.24 |

So we sit between v+x and x only. The change we can expect from "endpoint regression" alone is
small, and Phase 0a tests it on the existing loop. What Phase 2 tests is the branch structure.

## Objective

Branches: `F(x, t, c)` and `F(x, t, c_fake)` with `c_fake = −c` on the `crossattn_emb` the loop
passes to the DiT. `a = −1, b = 0` keeps zero padding at zero, which the max-padded TE invariant
needs; a bias `b ≠ 0` (APEX Table 7's best cell, 0.81 at a = −0.5, b = 1) would break it.
`a = −1, b = 0` read 0.76 there.

Each iteration runs three terms, each backwarded alone, then one optimizer step:

1. **`L_sup` on the batch latent `x`.** `x_t = (1−t)x + tε`, target
   `v_sup = (ε − x) + γ·(v_c − v_u)`, with `v_c`, `v_u` from the frozen teacher view (no grad).
   `E[ε − x | x_t] ≈ v_c`, so `γ = 3` targets CFG 4. Loss `‖F(x_t, t, c) − v_sup‖²` + the same in
   x space.
2. **`L_score` on a student point.** Backward-simulate to a random grid step `g ~ U{0..N−1}`
   under no_grad, as `dmd_grad_step = "random"`, step 0 included. With `F = F(x_g, σ_g, c)` and
   `x̂ = x_g − σ_g·F`: renoise `sg(x̂)` at `τ ~ U(0,1)` to `x'`, then
   `d = clamp(F(x', τ, −c) − F(x', τ, c), ±1)` under no_grad. Targets `v* = sg(F − d)` and
   `x* = x_g − σ_g·v*`. Loss `‖F − v*‖² + ‖x̂ − x*‖²` (IDM Eq. 11–12).
3. **`L_fake` on `sg(x̂)`.** `‖F(x_τ, τ, −c) − (ε − sg(x̂))‖²` (IDM Eq. 112).

`L = L_sup + λ_score·L_score + L_fake`, all weights 1, `λ_score` as IDM.

Decisions behind this form:

- **Guidance from the frozen base.** The papers do not describe guidance; the public APEX code
  adds `ratio·(v_c − v_u)` to the target (`enhance_target`, 4.0 in its 1-order LoRA config with
  the adapter-disabled base as `v`). We need it: base Anima at CFG 1 renders washed out, and the
  dataset sits about 60% of the way from CFG 1 to CFG 4 in pixels (`report_gaps.md`). We use
  `γ = 3`, since 4.0 on top of `ε − x` is CFG 5.
- **`L_fake` detached, from IDM.** APEX Eq. 12 lets the fake loss flow back into the `c` branch
  through `x_fake`. The detached form is a separate backward with one graph alive.
- **Clamp ±1 on `d`,** from the public APEX `dist_match`. IDM's normalization is not specified.

## Not ported

| Piece | Why |
|---|---|
| TwinFlow branch split by time sign (`t` vs `−t`), which the public APEX code still uses | Anima's time embedding stays unchanged; this is the reason APEX moved to condition space |
| Target-time input `tt` and the `t − tt/2` LoRA approximation | Anima has one `t_embedder`, and the shift would alter every forward |
| RCGM (IDM's `L_base`) | Built on `tt`. The APEX paper's objective has no RCGM: `L_sup` + `L_cons` + `L_fake` |
| Batch partition (`assert bsz >= 4`) | Replaced by the three per-iteration terms at B = 1 |
| `L_pop`, EMA features, queues | Held out; at B = 1 a 16,384 queue outlasts the run |
| DP-DMD anchor, L_CDM, GAN | Off in Phase 2, so the arm reads the structure alone |

## Cost

Grad forwards 3, no-grad at most 7 (3 rollout, 2 teacher, 2 branch queries), one graph alive at
a time. The fake stack is not built. Estimate from the DMAD phase timings: 3–4 s/iteration
against 6.7–7.0 on the shipped recipe. Not measured. It runs on `int8 = "stream"`, `stream =
"fp32"`: no stack is built after quantization, which is what keeps DMAD on bf16.

## Risks

- **Shared capacity.** One r64 LoRA holds the generator, the reference and the fake. Upstream
  ran this at r32/64 on Qwen-Image 20B; nothing has been shown on Anima.
- **`−c` reaches the DiT through cross-attention only.** The loop runs with
  `skip_pooled_text_proj=True` and the adaLN LoRA sees only `t`, so branch separation depends on
  how the cross-attention responds to inverted keys. Phase 0b reads this.
- **Data volume.** APEX trained on 800K+ images plus BLIP-3o / ShareGPT-4o, and its Table 5
  moves from 0.19 to 0.83 at 400 steps with the dataset alone. Here 10k iterations at B = 1 are
  about 3 passes over 3.2k images. `L_sup` may imprint the set's style.
- **Noise.** `d` is a single draw at B = 1 against upstream's 64. Our DM estimate agrees with a
  second draw at cos ≈ 0.12.
- **Diversity.** No anchor. Neither paper reports pose diversity, and the official v1.1 delta
  used as the warm start is mode-collapsed.

## Phases

### 0a — endpoint weighting on DP-DMD

Shipped `turbo.toml`, with `cdm.weight = 0`, `gan.weight_gen = 0`, and the DM term reweighted
from `σ_g` to `1 + σ_g²` (a one-line flag). 500 iterations, about 1 h. Control: the same recipe,
unchanged weight, seed 42. Read at 250 / 500. This is the direct test of the endpoint hypothesis
and has no APEX code in it.

### 0b — does `−c` separate on the base

Forward-only, under a minute. On a handful of (x_t, t, c) from the dataset read `cos(v(c), v(−c))`
and `cos(v(−c), v_u)` per τ. If `v(−c)` equals `v(c)` at any τ band, the branches share a path
there and the fake cannot be an independent estimator; try `a = −0.5` before building.

### 1 — smoke

30 iterations on the shipped base: peak VRAM, s/iteration, no NaN, `F` under `c` and `−c`
diverging.

### 2 — the arm

1500 iterations, seed 42, checkpoints every 250, warm start from the shipped v1.1 delta (IDM also
started from a Turbo checkpoint), with 200 `L_fake`-only iterations first so the `−c` branch
tracks the student. Estimated 1.5 h.

Read 4-step grids at `--cfg 1.0` on `bench/turbo/real_prompts.txt` prompts 0 / 1 / 8 / 13 and
whichever text and pose prompt set replaces the other 12. Controls are existing checkpoints:
`anima_turbo_v3_longer`, `anima_turbo_p0b_dpdmd` (bf16 base, stated next to the read).

- **Log:** `L_sup`, `L_fake`, `d` RMS, `cos(d, d')` on a second draw every 50 steps, `x_pred` std,
  `ℓ_v` / `ℓ_x` per σ_g. Scalars are for health; `fm_mse` is not a quality signal here.
- **Kill:** at 1500 the grid loses to both controls on pose or text; or `x_pred` std leaves 0.5–0.8
  for 50+ iterations; or `L_fake` does not fall during the 200 head-start iterations.
- **Next, one knob each, only on a pass:** cold start (zero LoRA); `γ` 2 / 4; anchor back on;
  10k iterations.

## Config and code

`base_loss = "apex"` with an `[apex]` section, off by default and byte-identical when off:

| Key | Default | Notes |
|---|---|---|
| `cond_scale` | `-1.0` | `a` in `c_fake = a·c` |
| `guidance_ratio` | `3.0` | `γ`; 0 = data target only |
| `lambda_score` | `1.0` | |
| `lambda_fake` | `1.0` | |
| `v_weight` / `x_weight` | `1.0` / `1.0` | 0 reproduces x only / v only |
| `d_clamp` | `1.0` | 0 = off |
| `fake_warmup_steps` | `200` | |

- **`scripts/distill_turbo/apex.py`** (new): the three terms.
- **`distill.py`:** an `apex` branch in `run_loop`, on the existing rollout, resume and save paths.
- **`networks/methods/turbo_dmd.py`:** skip the fake stacks; the `teacher` and `student` views are
  unchanged.
- **`config.py` / `setup.py`:** the section and flags; no fake optimizer. Stamp `ss_turbo_base_loss`.
- **`metrics.py`:** the rows above.
- **`tests/test_turbo_apex.py`:** `ℓ_x = t²·ℓ_v` on the targets; `c_fake` keeps zero padding at
  zero; `γ = 0` gives `ε − x`; off leaves the DP-DMD loop unchanged.
- **`bench/turbo/`:** the Phase 0b read, per `CONTRIBUTING.md` Tier 2.

The output stays a plain LoRA. `make test-turbo` reads it at 4 steps, CFG 1.0; the `−c` behaviour
it carries is unused at inference.

## Open questions

1. **Warm start and the reference.** With the v1.1 delta, the `c` branch starts as a student and is
   also the DMD reference. `L_sup` has to pull it back to a velocity field at low σ. The cold arm
   answers this, at a sample budget far below upstream's.
2. **Sample time for `L_sup`.** Uniform t, or the student's σ grid only. Uniform is the paper's;
   the grid spends the LoRA on the points inference uses.
3. **The `L_mix` mixing ratio.** APEX mixes data and fake targets with one `λ` inside one regression
   (Eq. 22–24). Separate terms are simpler to read, and the guided target has no clean place in
   `T_mix`, but this differs from the paper.
