# Turbo DMAD — discriminator-carried distribution matching inside DP-DMD

Status: **Phase −1 closed without a verdict; Phase 0 DMAD arm trained (750 steps);
Phase 0b (paper-style raw signal, no disc warmup, λ = 100) and a matched 500-step
DP-DMD arm trained and rendered: no quality gain read, DMAD moved less**
(2026-10-10). Branch `turbo_dmad`. Source: Yu et al., *DMAD: Distribution Matching
as Adversarial Distillation for Fast Visual Generation*, arXiv:2610.02188
(ByteDance, 2026-10-01).

## Motivation

The turbo loop pays most of its step time for the DM term: 4 fake-critic FM updates
per student step plus a CFG teacher and a fake forward per DM estimate, and a
`fake_warmup_steps = 200` head start. The estimate it buys is noisy: two
independent single-draw DM estimates on the same `x_pred` agree at cos ≈ 0.12
(Phase −1). DMAD replaces the critic with a classifier whose logit gradient is the
score difference at the classifier's optimum, and adds a real-data target the
teacher-only DM term lacks. The paper reports both a quality win over DMD2 and a
4–5× cheaper step. If either holds on Anima, it is worth having.

## The method

DMD's student gradient is `s_teacher − s_fake`, two separately estimated scores.
DMAD trains a balanced-BCE classifier `h` between target and student samples; its
optimum is `h* = log p_t − log p_θ,t`, so `∇ₓh* = s_p − s_θ` (Prop. 1), and the
linear generator loss `−E[h]` has the DMD reverse-KL gradient at that optimum.

- **Two heads, one backbone.** Head T: teacher samples vs student. Head R: real data
  vs student. Generator loss `−E[λ_T·h_T + λ_R·h_R]`. At the optimum the update is
  reverse KL to a per-noise-level geometric mixture of the two targets (App. A).
- **Linear loss, not non-saturating.** `−log D` reweights each sample by `1 − D`;
  the linear loss carries the logit gradient unweighted.
- **Gap reweighting.** Per noise band, an EMA of `mean h_R(real) − mean h_R(teacher)`;
  bands with a lower gap get a larger teacher weight, `σ((median − Δ_b)/τ)`.
- **Recipe (App. B).** One disc update per student update. AdamW β = (0, 0.99).
  η_G = η_D. Teacher samples cached offline. No gradient penalty is mentioned.
  - Wan: MLP heads on final-block tokens, logits averaged over the video, batch 8,
    LR 2e-6 (1.3B) / 1e-6 (14B), λ = 1.
  - MiniMax-H3: student and disc backbone both LoRA r128 on the teacher; RMS-normed
    final-block features into 2-layer MLP heads; batch 8, LR 4e-5, λ = 1, 800 steps.
  - SDXL: pretrained U-Net encoder + mid block, conv heads, batch 64, LR 5e-7.
- **Evidence for Prop. 1.** Only a 2D toy (Fig. 2): fixed student, disc trained with
  batch 256 + 256; the disc gradient's NMSE against the analytic field is 0.14,
  DMD's score difference 1.54. At scale the paper reports outcomes only.

| Setting | DMAD | Baseline |
|---|---|---|
| SDXL, 4 steps, COCO-10K FID / patch FID | 14.47 / 19.88 | DMD2 19.32 / 20.86; teacher 19.36 |
| Wan2.1-1.3B, 4 steps, VBench total | 84.70 | DMD2 84.56 |
| MiniMax-H3, 4 steps, human preference | 79.1 % over DMD2 (ties excluded) | |
| Time per generator update (H200) | Wan-14B 38.5 s, MiniMax-H3 233 s | DMD2 162.9 s, 818 s |

Ablations (Table 4): removing head R costs the most FID (SDXL 14.47 → 21.37);
removing head T improves SDXL FID (13.85) but costs patch FID, CLIP and VBench
semantic (−4.19), so head T carries detail and prompt adherence. Gap reweighting
and the linear loss are each a smaller step. The paper does not measure diversity;
DMAD is still reverse KL at the optimum, so DMD's mode-seeking is expected to
remain, and the DP-DMD anchor stays.

## Mapping onto the turbo loop

The proposal keeps DP-DMD's step-0 diversity anchor and replaces only the **DM term
and the fake critic**. The anchor's graph is detached from the DM steps
(`detach_after_first`), so the two objectives do not interact.

| DMAD piece | Now | Under DMAD |
|---|---|---|
| Student gradient | `steps.dmd_surrogate` → `grad_signal` (CFG teacher + fake, `dm_x0_norm`) | `∂(−λ_T h_T − λ_R h_R)/∂x_pred`, same grad-trick slot |
| Fake critic | `fake_update`: 4 FM steps/student step + 200 warmup | removed |
| Head T | — | new; teacher samples vs `x_pred` |
| Head R | the `[gan]` disc scores real `latents` vs `x_pred` (hinge, frozen-teacher features, ~2 M head) | new head on the DMAD backbone, balanced BCE |
| Disc backbone | — | teacher + its own LoRA stack (`turbo.make_aux_stack`, built for the probe) |
| CFG | baked into the DM real score (α = 4) | only through head T's teacher samples, rendered at CFG 4 |
| L_CDM | real − fake surrogate at an off-trajectory `x0_off` | `∂(−λ_T h_T − λ_R h_R)/∂x0_off` (`student_signal` at the same point); on when `cdm.weight > 0` |
| Gap reweighting | — | later arm |

f-distill ([[project_turbo_rollup]]) reweights the score-difference gradient by disc
logits; DMAD removes the score difference, so it is not a re-proposal of that
closed line.

## Phase −1 — premise probe (closed, no verdict)

**What it measured.** `--dmad_probe` (`scripts/distill_turbo/dmad_probe.py`) trains a
side disc (head T on a cold or warm LoRA stack over the teacher) beside an unchanged
DP-DMD run, and every step compares `g_T = ∂(−h_T)/∂x_pred` with that step's DM
`grad_signal` at the same (τ_dm, ε_dm): cosine and agree-energy, each against a
permutation null, plus a ceiling `cos(DM, DM')` from a second DM draw. Teacher
samples finish the step-0 anchor rollout on the 12-step CFG-4 grid, so teacher and
student share ε. Pre-registered read (`bench/turbo/dmad_probe_read.py`), window =
second half of the rows: UNCONVERGED if disc acc < 0.75; KILL if `cos − null` is
within 3 SEM of 0; PASS if `cos ≥ 0.5 × ceiling` and agree-energy beats its null by
3 SEM; WEAK otherwise. The reader also reports the window split at |margin| ≤ 1.

| Arm | Disc setup | Outcome | Rows (`output/logs/turbo/…`) / read |
|---|---|---|---|
| P1 | warm, token BCE, B = 1, LR 5e-5, clip 1 | plateau at acc 0.58; cos − null +0.029 ± 0.010 (not a read) | `20261009-093847`; `bench/turbo/results/20261009-1016-dmad_probe_p1` |
| P2 | LR 2e-4, unclipped | separates in bursts (acc 0.95), grad norm 14–24, chance from step 47 | `20261009-155349`; `…/20261009-1631-dmad_probe_p2` |
| P3 | LR 1e-4, unclipped | acc 0.999 at step 18, grad norm 46, chance; stopped at 80 | `20261009-163511`; `…/20261009-1656-dmad_probe_p3` |
| P4 | P3 + clip 5 | cycles between margin 0.8 and chance; pre-clip norms 23–38 | `20261009-170245`; `…/20261009-1739-dmad_probe_p4` |
| P5 | P4 + approx-R1 (w 1) + cold start | same cycle (norms 25, 64); R1 term < 0.01, never bit; stopped at 39 | `20261009-174446`; no read |
| P6 | paper recipe: cold, **scalar logit, replay window 4**, LR 4e-5, unclipped, no R1 | stable 150 steps; UNCONVERGED by `acc`, at the null after step ~18 (below) | `20261009-180548`; `…/20261009-1849-dmad_probe_p6` |

**P1–P5: a disc set up unlike the paper's.** P1 stepped too little; P2–P5 learned
and then blew up. Rereading the paper, the probe's disc differed on more than step
size: a BCE per token instead of one logit per sample (Prop. 1 is a statement about
the sample's log-ratio), batch 1 at one τ per update instead of 8–64, and a student
that never trains against the disc, so teacher vs student is a fixed, separable
problem on which logistic loss drives the logits up without bound. That fits the
separate → spike → chance cycle. The positive alignment P3 / P4 showed in their
first ~20 steps was first read as Prop. 1 holding while unsaturated (P3), then as
an artifact of the warm start, since the disc stack started from the student's own
init (`fake_init_weights` and `student_init_weights` are the same file).

**P6: a stable disc, aligned only early.** New knobs `dmad_probe.scalar_logit`
(mean the token logits before the BCE) and `dmad_probe.window` (each update
accumulates over the newest N teacher/student pairs, a fresh (τ, ε) each); rows
gain `acc_window`, `n_pairs` and the teacher − student statistics `dc_share` (share
of ‖x_t − x_s‖² in the per-channel means) and `ch_std_logratio`. 150 steps,
~17 s/step (disc update 3.5 s), peak 14.1 GiB.

| Steps | cos − null | rank acc (h_t > h_s) | mean margin | max grad norm | ceiling |
|---|---|---|---|---|---|
| 8–18 | +0.103 ± 0.027 | 1.00 | +0.54 | 10.4 | +0.061 |
| 19–40 | −0.000 ± 0.011 | 0.95 | +0.82 | 12.2 | +0.218 |
| 41–75 | −0.024 ± 0.012 | 0.86 | +0.68 | 24.4 | +0.153 |
| 76–150 (read window) | +0.003 ± 0.009 | 0.85 | +1.00 | 29.5 | +0.139 |

- **Verdict by the pre-registered rule: UNCONVERGED** (window `acc` 0.673 ± 0.029;
  `acc_window` 0.713). Under rank accuracy (0.85) the gate would pass and the read is
  **KILL**: paired cos − null +0.003 ± 0.009, agree-energy Δ +0.011 ± 0.007. The rank
  metric was proposed after seeing steps 1–36, so it is reported, not voted.
- **Why the two disagree.** With one logit per sample at B = 1, `acc` scores the two
  logits' signs separately, so a correct ranking with both logits on one side of 0
  reads 0.5.
- **The disc held.** No collapse in 150 steps (P2–P5 collapsed by step 20–50); the
  grad norm crept up from ~10 to a max of 29.5, at chance never. Window split: 45
  unsaturated rows read cos − null −0.006 ± 0.009, 30 saturated rows +0.015 ± 0.016.
- **The early alignment appears again with a cold start**, then goes. The cold
  backbone is the teacher DiT itself, whose input gradient may carry teacher-score
  directions that DM (= teacher − fake) shares; untested.
- **By τ_dm (post hoc, n = 6–12 per bin):** negative at low noise (−0.050 ± 0.011 at
  τ < 0.125, −0.046 ± 0.009 at 0.125–0.25), positive in the middle (+0.031 ± 0.008
  at 0.5–0.625, +0.038 ± 0.023 at 0.625–0.75), against nulls within ±0.003.
  Opposite-signed bins average to the null; not read further at this n.
- **Teacher − student cue.** Window `dc_share` 0.086 ± 0.013: the disc is not
  separating on a global offset. `ch_std_logratio` 0.22 in the window (0.6–0.85 in
  the first 20 steps): a per-channel contrast gap, a candidate cue, and one DM
  itself should be correcting.

**Why Phase −1 stops here.** Its yardstick is the DM estimate, which the paper
argues is the inaccurate quantity (toy NMSE 1.54), and which agrees with a second
draw of itself at cos ≈ 0.12. Noise alone does not explain a zero: if `g_T` were
the true field and DM = truth + independent noise, the expected cos would be
≈ √0.12 ≈ 0.35. So either `g_T` is not the true field, or the part of the DM
estimate that repeats across draws is the critic's systematic error rather than
the true field. A measure-only probe has no ground truth to separate the two, and
it cannot test the one mechanism the paper depends on and the probe lacks: the
student training against the disc. Phase 0 tests the outcome, as the paper does.

## Phase 0 — DMAD as the DM term (plan)

One arm against a matched DP-DMD arm, read on rendered grids.

### Decisions

- **Student: warm start, as shipped** (`anima_turbo_v1.1_delta_r96_asvd_adaln`, the
  official turbo v1.1 extracted against base v1.0, not a DP-DMD output). The
  matched DP-DMD arm moves the student off that init (re-expanding the official
  turbo's collapsed modes, `docs/methods/turbo.md`), so the two arms have something
  to differ on.
- **Disc backbone: cold.** Teacher + a zero-init LoRA stack (`make_aux_stack`,
  `fake_rank`), as the paper's MiniMax setup. The only stable Phase −1 arm was cold;
  a warm stack starts from the student's own weights.
- **Disc: P6's recipe.** Scalar logit (mean of token logits), replay window 4,
  AdamW β = (0, 0.99), LR 4e-5, unclipped, no R1, tap = middle block
  (`return_features_early`), heads = `TeacherFeatureDiscriminator` token heads
  (LayerNorm → Linear → LeakyReLU → Linear), one per target.
- **Teacher samples: online.** Finish the step-0 anchor rollout from `k_anchor` to
  σ = 0 on the 12-step CFG-4 grid (the probe's `_teacher_sample`, ~2.5 s/step). An
  offline cache only if speed becomes the question.
- **Real samples:** the step's `latents` (the caption's own image), as the `[gan]`
  disc already uses them.
- **Signal scale: per-sample RMS normalization.** The disc's input gradient is
  1e-5 to 8e-4 of the DM signal's RMS in P6 and drifts 30× over a run, so a fixed λ
  cannot hold its weight against the diversity anchor. Normalize
  `g = ∂(−λ_T h_T − λ_R h_R)/∂x_pred` per sample to `dmad.signal_rms` (default
  0.18, the median DM `grad_signal` RMS in P6), then use it as `grad_signal`. This is
  the analogue of `dm_x0_norm`; the paper has none, and its appendix says the
  equivalence does not cover per-sample normalization. λ_T = λ_R = 1 sets the head
  mix only.
- **Disc warmup.** Normalization gives an untrained disc's gradient full weight, so
  the disc trains alone for `dmad.disc_warmup_steps` (default 50; P6 ranked
  consistently from step ~8) before the student uses it: student untouched, no-grad
  rollouts, the `run_fake_warmup` pattern. The replay window fills during it.
- **Held off:** gap reweighting (fixed λ_T = λ_R), L_CDM and the `[gan]` side term
  (`cdm.weight = 0`, `gan.weight_gen = 0` in **both** arms), soft-rank.

### Config — new `[dmad]` section, off by default

| Key | Default | Notes |
|---|---|---|
| `enabled` | `false` | off → byte-identical loop, no RNG drawn, nothing built |
| `lambda_t` / `lambda_r` | `150` / `150` | head mix; 0 drops a head (and its disc branch). Sets the signal's size only when `signal_rms = 0` |
| `signal_rms` | `0.18` | per-sample RMS of the student signal; `0` = raw (λ sets the size) |
| `lr` | `4e-5` | disc stack + heads, constant |
| `grad_clip` | `0` | disc only; 0 = unclipped |
| `window` | `4` | replay pairs per disc update |
| `disc_warmup_steps` | `50` | disc-only updates before the main loop |
| `feature_block_idx` | `-1` | −1 = middle block |

Every key gets a `--dmad_*` CLI flag. Guards at resolve: requires `base_loss =
"dpdmd"`; refuses `gan.weight_gen > 0`, `f_distill`, `dmad_probe`,
`fake_tau_banks > 1`, `blocks_to_swap > 0`, and `--resume` (the disc is not in the
resume bundle in this phase).

### Code

- **`scripts/distill_turbo/dmad.py`** (new): `DmadDisc` — the stack, head T, head R,
  optimizer, replay window and the update. `DmadProbe` shares its helpers (tap
  resolution, disc view, generator draws, the anchor finish) but keeps its own
  update, so P1–P6 replay with the same RNG order. Update per pair: three backbone
  passes (teacher sample → h_T; real → h_R; student → both heads from one pass),
  each backwarded one at a time under `selective_block_grad_ckpt`, losses scaled
  1/n, one optimizer step.
  `student_signal(x_pred, τ, ε, c)`: one grad-bearing disc forward on a detached leaf,
  `autograd.grad`, per-sample normalization → `grad_signal`.
- **`steps.py`**: `dmad_signal(...)` returning a `DmdResult`-shaped record (no
  `v_real`/`v_fake`), so the assembly code and masking are untouched.
- **`distill.py`**: under `dmad`, `dmd_surrogate` → `dmad_signal`, `fake_update` →
  `DmadDisc.update`, `run_fake_warmup` → disc warmup. The anchor and the student
  backward are unchanged.
- **`setup.py`**: build `DmadDisc` before compile (it is a LoRA on the DiT); do not
  build the fake banks or their optimizer (frees VRAM).
- **`config.py`**: the section, flags and guards above.
- **`metrics.py`**: a DMAD row group — disc BCE per head, margin and rank accuracy
  per head, the real − teacher gap of h_R (the gap-reweighting signal, logged now so
  that later arm has data), raw `g_T` / `g_R` RMS before normalization, and
  `cos(g_T, g_R)`.
- **Save metadata**: `ss_turbo_dmad` and the `[dmad]` keys, so an arm can be checked
  from the file as well as the `.snapshot.toml`.

### Tests (`tests/test_turbo_dmad.py`)

- Config: default off, CLI/TOML precedence, each guard.
- Normalization: per-sample RMS equals `signal_rms`, direction preserved, a zero
  gradient stays zero.
- Disc update on a toy backbone: the loss is the sum of the two balanced BCEs, the
  window accumulates n pairs into one step, a head with λ = 0 adds no branch.
- The probe's existing tests pass unchanged on the shared helpers.

### Runs and verdict

- **Arms.** Both: shipped `configs/methods/turbo.toml` with `cdm.weight = 0`,
  `gan.weight_gen = 0`, 750 iterations (the warm-start length in
  `docs/methods/turbo.md`), same seed, checkpoints at 250 / 500 / 750. DMAD arm:
  `[dmad] enabled = true`. Daemon jobs (`make turbo ARGS="… --queue"`).
- **Smoke first.** 30 steps of the DMAD arm: peak VRAM, step time, the disc reaching
  rank accuracy ≥ 0.8 in warmup, no NaN.
- **Read.** 4-step grids at `--cfg 1.0`, the same prompt × seed set (fixed before
  the runs) for the v1.1 init, the DP-DMD arm and the DMAD arm at each checkpoint.
  Read pose diversity, text and saturation. Not scalars: `fm_mse` is
  anti-correlated with quality ([[project_turbo_rollup]]).
- **Kill.**
  - The DMAD grid loses to the matched DP-DMD arm on pose or text at both
    `signal_rms` values (0.18, then 0.09); or
  - a disc runaway: rank accuracy at chance for 20+ steps after warmup, at
    `grad_clip` 0 and again at 5.

### Results so far

**Smoke (30 steps, `anima_turbo_dmad_smoke`).** ~12 s/step after the disc warmup
(10.5 s/step). Both heads rank at warmup end; by step 30 both saturate (margin T
+2.9, R +3.3; disc grad norm 34). Renders at 4 steps / cfg 1.0 against the v1.1
init (`bench/turbo/real_prompts.txt`, seed 42; `output/tests/dmad_smoke30`,
`output/tests/v11_init`): nothing broken (people, poses, anatomy and prompt
content kept in all 16), but flat backgrounds become busy scenes, skies pick up a
streaky high-frequency texture, colour turns duller, and compositions vary more.
No same-step DP-DMD reference was rendered, so how much of that change is DMAD's
own is unknown.

**DMAD arm (`anima_turbo_p0_dmad`, 750 steps, job `20261009-191035-30614c`).**
~12.5 s/step, 2 h 45 min including warmup; checkpoints at 250 / 500 / 750. Rows:
`output/logs/anima_turbo_p0_dmad.progress.jsonl` (`dmad_*` keys); TensorBoard
`output/logs/turbo/20261009-191043`. Means over 150-step bins:

| Steps | rank acc T / R | margin T / R | BCE T (chance 1.39) | disc grad norm | raw g_T RMS | cos(g_T, g_R) | gap_r |
|---|---|---|---|---|---|---|---|
| 1–150 | 0.81 / 0.89 | 1.9 / 2.0 | 1.08 | 24 | 2.6e-4 | 0.85 | 0.22 |
| 151–300 | 0.88 / 0.89 | 3.0 / 3.2 | 0.94 | 35 | 1.3e-3 | 0.94 | 0.34 |
| 301–450 | 0.91 / 0.91 | 2.1 / 2.4 | 1.01 | 32 | 9.3e-4 | 0.78 | 0.40 |
| 451–600 | 0.85 / 0.90 | 1.7 / 2.2 | 1.09 | 30 | 7.8e-4 | 0.74 | 0.63 |
| 601–750 | 0.90 / 0.91 | 2.1 / 2.4 | 1.01 | 28 | 9.3e-4 | 0.62 | 0.70 |

- **No runaway, no collapse.** Rank accuracy never fell to chance; the disc grad
  norm peaked at 136 (step 270) and settled near 28.
- **The disc wins throughout.** Margins of ~2 from step 100 on: the student does
  not erase what the disc separates on, which the game should do.
- **The heads start as one detector and separate late.** cos(g_T, g_R) 0.85–0.94
  over the first 300 steps with gap_r (h_R on real minus on teacher) ~0.2–0.3:
  both heads scoring "is a student sample". From step ~300, cos falls to 0.62 and
  gap_r rises to 0.70, so head R starts telling real data from teacher samples.
- **The raw disc gradient sharpened ~4–5×** (2.6e-4 → ~1e-3); per-sample
  normalization keeps the student's update size fixed regardless.
- **Student scalars calm:** `xpred` 0.62–0.66, `v_student` ~1.14; the diversity
  anchor loss drifts down (0.114 → 0.085).

**Matched DP-DMD arm** (`anima_turbo_p0_dpdmd`, job `20261009-191035-6e1aeb`):
stopped by hand at start, not run. The grid read below needs it, or a stand-in
the owner names.

### Phase 0b — raw signal, no disc warmup, against a matched DP-DMD arm

The paper has no disc head start (App. B: one disc update per generator update
from the first step) and no signal normalization; an untrained disc's gradient
is small, so the student is not pushed by it early. Phase 0b drops both:
`dmad.signal_rms = 0` (new: the raw `∂(−λ_T h_T − λ_R h_R)/∂x_pred` is used as
`grad_signal`) and `disc_warmup_steps = 0`.

- **λ = 100 (T and R).** The loop's DM loss is the element mean of
  `grad_signal · x_pred`, the convention the DP-DMD signal (RMS ~0.2) lives in.
  The paper's literal `−λ·E_b[h_b]` with λ = 1 is ~10³× that and would make the
  diversity anchor negligible and pin the student clip. λ = 100 puts a disc at
  P0's mid-run sharpness (raw RMS ~1e-3 per head, cos ~0.8) near 0.18.
- **Arms.** Shipped `turbo.toml`, `cdm.weight = 0`, `gan.weight_gen = 0`, seed 42,
  500 iterations, checkpoints at 250 / 500. DMAD `anima_turbo_p0b_dmad_raw` (job
  `20261009-220442-b34a0e`, 11.9 s/step, 1 h 39 min); DP-DMD
  `anima_turbo_p0b_dpdmd` (job `20261009-220400-b13d79`, 200 fake-warmup steps
  then 6.6 s/step, 57.5 min). Rows: `output/logs/anima_turbo_p0b_*.progress.jsonl`.

| Steps | DMAD `grad` RMS | DP-DMD `grad` RMS | raw g_T / g_R RMS | cos(g_T, g_R) | rank acc T / R | margin T / R | gap_r | DMAD `div` | DP-DMD `div` |
|---|---|---|---|---|---|---|---|---|---|
| 1–25 | 0.016 | 0.29 | 7.4e-5 / 9.0e-5 | 0.79 | 0.92 / 0.88 | 0.8 / 0.8 | −0.04 | 0.132 | 0.159 |
| 26–100 | 0.051 | 0.23 | 2.9e-4 / 2.3e-4 | 0.97 | 0.91 / 0.88 | 1.9 / 1.9 | 0.02 | 0.116 | 0.120 |
| 101–200 | 0.071 | 0.22 | 3.5e-4 / 3.6e-4 | 0.97 | 0.86 / 0.85 | 1.8 / 2.0 | 0.13 | 0.112 | 0.112 |
| 201–300 | 0.093 | 0.21 | 4.6e-4 / 4.7e-4 | 0.97 | 0.88 / 0.89 | 1.6 / 1.9 | 0.29 | 0.114 | 0.105 |
| 301–400 | 0.076 | 0.19 | 3.9e-4 / 3.8e-4 | 0.93 | 0.82 / 0.83 | 1.6 / 1.8 | 0.26 | 0.100 | 0.098 |
| 401–500 | 0.117 | 0.21 | 6.0e-4 / 5.9e-4 | 0.91 | 0.91 / 0.90 | 2.0 / 2.4 | 0.50 | 0.085 | 0.095 |

- **The self-warmup happens.** With no head start the disc ranks from the first
  bin and its signal grows 7× as it sharpens, as the paper's setup implies.
- **But the signal stays 2–18× below DP-DMD's** for the whole run (0.016 →
  0.12 against ~0.2), so the anchor carries more of the student update than
  under DP-DMD. λ = 100 was calibrated on P0's sharper disc (raw ~1e-3 by step
  ~450); this disc reached ~6e-4.
- **Disc healthy, as in P0:** no runaway, margins ~2, heads one detector early
  (cos 0.97) and separating late (cos 0.91, gap_r 0.50). Student scalars calm
  (`xpred` 0.56–0.66, `v_student` 1.13–1.16).
- **Step time.** DMAD 11.9 s/step against DP-DMD 6.6 s/step plus its 200-step
  fake warmup (2.7 min): at window 4 DMAD is ~1.8× slower per step, not the
  paper's 4–5× faster.

**Read (4 steps, cfg 1.0, `bench/turbo/real_prompts.txt`, seed 42).** Renders
for v1.1 init, both arms at 250 / 500, and the P0 arm at 500
(`output/tests/{v11_init,p0b_dpdmd_250,p0b_dpdmd_500,p0b_dmad_raw_250,p0b_dmad_raw_500,p0_dmad_norm_500}`).
Read on prompts 0, 1, 8 and 13 only; the other 12 are not suitable for the grid
read, and a full read needs a replacement eval set (with text and pose prompts).

- Nothing broken in either arm at 250 or 500.
- DMAD raw stays closer to the init than DP-DMD: compositions and poses kept
  (prompts 8, 13), where DP-DMD re-poses, adds props and brightens sea / sky.
- DMAD raw adds side clutter in places (prompt 1), like the P0 smoke.
- No quality gain over DP-DMD read; on this subset DMAD reads as "moved less",
  which the signal scale accounts for. Not a kill under the Phase 0 criteria.

**Next, one knob:** λ = 300–500 so the late-run signal lands near 0.2, or 1000
steps at λ = 100; both after an eval prompt set is fixed.

### Smokes after Phase 0b (30 steps each, p0b raw recipe, seed 42)

- **Disc update without checkpointing** (`anima_turbo_dmad_nockpt_smoke`, λ 100):
  the window update runs through the compiled blocks. 10.6 s/step against p0b's
  12.2 over steps 10–30, peak 14.3 GiB. `student_signal` keeps the unsloth
  checkpoint: the student graph is alive there, and dropping it OOMs at step 1.
- **L_CDM under DMAD + λ 300 default** (`anima_turbo_dmad_cdm_smoke`, `cdm.weight`
  1.0 from the TOML): L_CDM takes `student_signal` at `x0_off` in place of
  teacher − fake. Runs, 12.1 s/step (+1.5 s for CDM), peak 14.3 GiB. The raw
  signal grows fast: `grad` 0.05 → 0.25 → 0.89 at steps 10 / 25 / 30 (λ 100
  without CDM: 0.015 → 0.038), `cdm` 0.54 at step 30. 5-step means at B = 1, so
  noisy, but already above DP-DMD's ~0.2 by step 25. Raw mode has nothing that
  caps the signal as the disc sharpens.

### Cost (unmeasured)

- Removed per step: 4 fake FM forward + backward; the DM term's 2 CFG teacher
  forwards and 1 fake forward.
- Added per step: the teacher finish (6 CFG steps, ~2.5 s in Phase −1); a disc
  update over 4 pairs × 3 backbone passes (P6 measured ~0.9 s per pass pair, so
  ~5 s); one grad-bearing disc forward for the student signal (~0.4 s).
- Plausibly a wash at window 4. VRAM should drop (no fake banks); Phase −1 peaked at
  14.1 GiB with both the fake and the probe's stack resident.

### Later arms, one knob each, only if Phase 0 passes

1. Gap reweighting (the gap is already logged).
2. L_CDM under DMAD.
3. NFE = 2 (superturbo).
4. Offline teacher cache, if speed matters.
5. Final-block tap, as the paper's Wan / MiniMax heads.

## Open questions

1. **Real-data pull.** Head R pulls the student toward whatever `data_dir` holds. Is
   the turbo dataset broad enough to improve fidelity rather than imprint a style?
2. **Mid-rollout x0 predictions.** `grad_step = "random"` puts the signal on a
   one-step x0 prediction at a random refinement step, blurrier than the teacher
   endpoints head T was trained on. Phase −1 binned alignment by grad step but never
   had a converged disc to read it on.
3. **Diversity.** The paper does not measure it; the anchor is expected to carry it,
   as under DP-DMD.
