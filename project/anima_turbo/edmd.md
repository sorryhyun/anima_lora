# eDMD on the 4-step loop — an endpoint fake for steps 1–3

Status: PROPOSED (2026-11-23). Nothing is built and nothing has run. It is a candidate for the
"objective for steps 1–3" entry of [plan.md](plan.md) § To measure again, and it is not on the
plan's item order (item 1 is preference RL); the first pair (§ 6) is queued on the user's word.

Paper: Zhao et al., *eDMD: Endpoint Distribution Matching for Few-Step Diffusion Distillation*,
arXiv 2611.17194v1 (code `github.com/Westlake-AGI-Lab/eDMD`, not read). Section, table and
appendix numbers below are the paper's.

## TL;DR

The fake of our loop is trained on the one-step prediction of whichever refinement step took the
gradient, so two thirds of what it learns are intermediate predictions, not delivered images. The
paper shows that such a fake adds a term to the DM gradient that does not vanish when the delivered
images already match the teacher, and that training the fake on delivered images alone removes it.
The proposal is three changes to `scripts/distill_turbo/`, each behind its own knob and tried one
at a time: the endpoint fake (E), a floor on the query level of intermediate steps, and the precise
teacher (PT). The first pair trains the warm v1.1 delta on merged15 with DM as the loop stands and
with E, and the user reads the two against each other and against the untrained delta.

## 1. What the paper measures

### Facts

- Decomposition (Eq. 3, Prop. 1): with a fake fitted on a reference distribution ν and the
  student's delivered distribution q, the correction is (guidance term) + (D_p − D_q) + (D_q − D_ν).
  The last term is τ² / (1 − τ) · ∇ log(q_τ / ν_τ), zero at every τ only when ν = q.
- Zero-correction sets (Prop. 2): endpoint fake, {q = p}; a fake on a mixture of steps
  (DMD2, D-DMD), {Σ π_k q_k = p}; a stage-wise fake (TDM), {q = p and every intermediate state
  distribution equal to the teacher's marginal}.
- The method (Alg. 1, Table A.2): the fake trains on the final output of a detached 4-step Euler
  rollout only, 5 fake updates per student update, each on a fresh prompt and rollout. The student
  step k is uniform over all 4 steps. Steps before the last are queried at τ ≥ σ_{k+1} (their
  landing level) from a calibrated density (Q); the last step at τ ~ U[0.02, 0.99). For the last
  step at τ < σ_{K−1} the teacher target is PT: the guided ODE integrated from τ to 0 over four
  Euler nodes at τ · {1, ¾, ½, ¼}.
- Its setup: rank-32 LoRA student and fake (alpha 64) on the frozen teacher, student LR 6e-6, fake
  LR 2e-5, batch 16 at 1024², 2000 updates, no real images. SD3-Medium runs on σ 1 → 0.9 → 0.75 →
  0.5 → 0.
- Scores are HPSv3++ on DrawBench-1000 (200 prompts × 5 seeds), at the checkpoint with the highest
  HPSv3++ on those prompts. Rerun spread (App. G): standard deviation 0.074 on SD3-Medium and
  0.11 on FLUX.2-klein-4B.

  | | SD3-Medium | FLUX.2-klein-4B |
  |---|---|---|
  | full | 11.171 | 12.699 |
  | without E (TDM's stage fake) | 10.122 | 11.095 |
  | without PT | 11.026 | 11.667 |
  | without Q (uniform on [σ_{k+1}, 0.99)) | 10.798 | 12.507 |
  | DMD2, their run | 10.270 | 11.337 |

- A trained FLUX.2-klein-4B student switched to the mixture fake at 1.4k updates loses 1.71 within
  100 updates; with the mixture fake pre-fitted for 1000 updates on the frozen student first, 2.07
  (App. G). The paper describes the switched renders as "smoother textures and simpler
  backgrounds".
- Norm of the intermediate predictions over the final one, ‖x̂_k‖ / ‖x̂_3‖, steps 0 and 1, on
  FLUX.2-klein-4B: 2.93 and 2.35 under the endpoint fake, 1.43 and 1.24 one hundred updates after
  the switch, within 1.3 when trained on the mixture fake from scratch. On SD3-Medium under the
  endpoint fake all three stay within 11% of the final norm.
- The endpoint fake put on TDM, everything else of TDM kept, raises its peak by 0.61 / 1.45 / 0.53
  (SD3-Medium / klein-4B / klein-9B; Table A.7).
- What the paper tried and dropped (App. C, G): PT on every query of every step (collapse on
  klein-4B, −1.23 at 1k); the endpoint correction back-propagated through the rollout (−0.28 on
  SD3-Medium, −1.74 on klein-4B); updating the last step only (plateau at 7.5–8.4 on klein-4B
  against 12.1).
- Calibration (App. D, Table A.3): on SD3-Medium the density concentrates near the landing level
  (mean query level 0.913 against 0.945 uniform at s = 0.9, effective support 0.39); on FLUX and
  Qwen-Image it stays near uniform (support 0.92–0.99).
- No diversity or seed-to-seed measure is reported anywhere in the paper.

## 2. Our loop against it

### Facts

| | Our loop | eDMD |
|---|---|---|
| Fake's training data | `x_pred`, the one-step x̂ of the grad step g ~ U{1, 2, 3}, taken at σ 0.9 / 0.75 / 0.5 (`distill.py:153`, `:168`; `steps.py:259`). Only g = 3 is the delivered image | the delivered image of a full rollout |
| Fake head-start | 200 updates on the student's one-step x̂ from a renoised dataset latent at a random τ (`warmup.py`) | none; fake and student both start from the teacher |
| Fake updates | 4, all on the one `x_pred` of the iteration | 5, each on a fresh prompt and rollout |
| Fake loss | velocity MSE, τ uniform | x̂ MSE weighted min{max{SNR, 1}, 10}, τ ~ U[0.02, 0.99) |
| Steps that take the DM gradient | 1–3; step 0 takes the anchor regression or nothing (`div_weight` 0) | 0–3 |
| Query level | τ ~ U[0, 1) for every g (`steps.py:159`) | τ ≥ σ_{k+1} before the last step, calibrated |
| Teacher target | one CFG 4 prediction at the query | the same; PT for the last step below σ_{K−1} |
| Normaliser (`dm_x0_norm`) | mean \|τ · v_real\| (`steps.py:173`) | ω = mean \|x̂ − A\|, which in our terms is τ · mean \|v_real − (ε − x̂)\| |
| Grid | σ 1 → 0.9 → 0.75 → 0.5 → 0 | the same on SD3-Medium |

## 3. Records on this line that it touches

### Facts

- v1.1 S sameness 0.397 on `judge_hard` against the teacher's 0.314, and 0.572 against 0.504 on
  `judge_random_1118` ([reports/step0_k7of28.md](reports/step0_k7of28.md) §§ 5, 6).
- `turbo_1118_1818` (step 0 alone on the 7 of 28 anchor) won every probe number and lost the
  user's blind read 0 / 12; its step 0 on the untouched delta lost 2 / 10 (same report, §§ 2, 3).
- The user's blind reads put the v1.1 delta untrained level with the 28-step teacher on
  `judge_hard` (T 5 / S 7) and ahead on three random dataset captions (T 3 / S 9) (same report,
  §§ 5, 6).
- No run since the restart has put a DM gradient on steps 1–3.
- [pref_rl/proposal.md](pref_rl/proposal.md) § 1 cites a pre-restart record: 250 iterations of the
  v1 recipe on merged14 + Hub v3 raised sameness from 0.326 to 0.465. That run used the pre-restart
  anchor and another teacher and student; it carries no verdict here.

## 4. Interpretation

- Our fake is the paper's mixture reference, with the mixture over steps 1–3. The step-1 prediction
  comes from σ 0.9, so a third of the fake's data is a one-step guess from near-noise.
- Our query level has no floor. A step-1 prediction is asked at τ 0.1 to carry what a finished
  image carries there, which is the pressure the paper's norm ratios show under a mixture fake:
  intermediate predictions pulled onto the final one.
- If the residual is what a warm student pays under DM, the control arm of § 6 should lose to the
  untrained delta and the E arm should not. Whether the residual has anything to do with sameness
  or with the haze of the pre-restart students is a guess; the paper measures neither.
- The paper's evidence is a preference score on photographic prompts, selected at its own peak.
  On this line probe numbers have not predicted the user's read, so its tables rank the three
  changes for us (E first) and decide nothing.
- The untrained delta already reads level with the teacher, so on `judge_hard` the E arm has
  little room to read above it. What the first pair can show is whether DM stops costing; a gain
  would have to show in sameness or on prompts the delta handles worse.
- The method does not touch step 0 or the question of what the line distills toward.

## 5. The three changes

Each is a knob defined in `scripts/distill_turbo/config.py`, off by default, so the loop as it
stands is the control.

1. E, `dmd.fake_reference = "endpoint"` (default `"mixture"`). `student_rollout` continues from
   the grad step to σ 0 under no_grad and returns `x_final`; for g = 3 that is `x_pred` detached.
   `fake_update` and `run_fake_warmup` (a full no-grad rollout per warm-up step) train on it. Cost:
   one more student forward per iteration on average, four per warm-up step. The paper's fresh
   prompt and rollout per fake update is left out of this knob (it multiplies the fake's cost by
   about 4) and is a later arm if E reads well.
2. Query floor, `dmd.query_floor = "landing"` (default `"none"`): τ_dm ~ U[σ_{g+1}, 1) for g < 3,
   unchanged for g = 3. No extra forwards. The calibrated density on top of it needs teacher
   samples per landing level (256 prompts × 2 draws in the paper) and is not proposed until the
   uniform floor has been read.
3. PT, `dmd.precise_teacher_nodes = 4` (default 0): for g = 3 and τ_dm < 0.5 the teacher target is
   the clean latent A after four CFG 4 Euler nodes from the query y, and the signal uses
   v_real = (y − A) / τ_dm. Three more CFG pairs (6 forwards) on about one iteration in six
   (1/3 · 0.5). At batch 1 the branch is per iteration, with no masking.

Order: E, then the floor, then PT. The paper's ablation puts E first on both teachers, and E and
the floor cost almost nothing per iteration.

Left as they are in every arm: the fake's loss form, the normaliser, the grid, the teacher and
its CFG. The normaliser differs from the paper's (§ 2), and `docs/anima/structure/turbo.md` § 7
describes ours as the original DMD's; that is a separate check.

## 6. The first pair

This section becomes the two config headers when the pair is queued
(`project/anima_turbo/configs/`); from then on the settings live there.

- Question: does DM with an endpoint fake hold the warm v1.1 delta where DM with the mixture fake
  does not?
  - E arm at or above the untrained delta on the read: eDMD is the steps 1–3 objective; the floor
    arm runs next on the E arm's recipe.
  - E arm above the control and below the untrained delta: the reference matters and DM still
    costs; the floor arm runs once, then the user decides whether the line stays on it.
  - E arm level with the control: the fake's reference is not what DM costs here; nothing else of
    this file is built.
- Control: merged15 teacher (CFG 4, int8 `stream`, fp32 stream), the v1.1 delta init at rank 64 /
  alpha 180 (adaln 16 / 90), the base loop of `configs/methods/turbo.toml` with a fresh fake
  (rank 64, 200 warm-up updates, 4 fake updates per student update, `fake_lr` 5e-5),
  `student_lr` 2e-5, `div_weight` 0, 250 iterations, seed 42.
- Arm: the control with `dmd.fake_reference` `"mixture"` → `"endpoint"`, nothing else changed.
- Read: EMA of each arm, `bench.turbo.probe_handoff` on `judge_hard` seeds 1–4 (12 cells), S
  column. Two blind sheets with `bench/turbo/blind_sheets.py`: arm against control, arm against
  the v1.1 delta untrained. The user's read decides. Reported beside it: S sameness, S cos, and
  ‖x̂_k‖ / ‖x̂_3‖ for k = 0, 1, 2.
- Bars: spread unknown (no seed twin of a DM run since the restart).
- Cost: 250 iterations an arm. The pre-restart DM configs record 0.8–1.0 it/s for a loop with
  more terms (`_archive/configs/anima_turbo/turbo_1118_1122.toml`, `_1215`, local), so about 5 min
  of loop an arm plus load, compile and the warm-up; the control's log gives the real figure.
- Preconditions: for g = 3, `x_final` equals `x_pred` (an assert in the smoke); the endpoint
  fake's loss per τ bin is logged from the warm-up on, as the mixture fake's is now.

Open calls, the user's:

- `div_weight` 0 in both arms. Step 0 then takes no loss of its own and moves only through the
  shared LoRA. The alternative is the anchor on from the `m15_k7of28_p500` cache, which the 0 / 12
  read of `turbo_1118_1818` speaks against.
- `student_lr` 2e-5 is the value carried for step-0 fitting; no DM run on this student has set one.
- The fresh fake against a fake initialised from the v1.1 delta.
