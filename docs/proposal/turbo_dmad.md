# Turbo DMAD — discriminator-carried distribution matching inside DP-DMD

Status: PROPOSAL, no code or GPU work done. Source: Yu et al., *DMAD: Distribution
Matching as Adversarial Distillation for Fast Visual Generation*, arXiv:2610.02188
(ByteDance, 2026-10-01). Wiring claims below were checked against
`scripts/distill_turbo/` and `configs/methods/turbo.toml` as of `3079e415`.

## TL;DR

DMD's student gradient is the difference of two separately estimated scores,
`s_teacher − s_fake`. DMAD gets that difference from a discriminator logit instead.
The balanced-BCE optimum is `h* = log p_t − log p_θ,t`, so `∇_x h* = s_p − s_θ`
(their Prop. 1), and the linear generator loss `−E[h]` reproduces the DMD
reverse-KL gradient at the optimum. One discriminator (shared backbone, two heads:
**teacher samples vs student** and **real data vs student**) replaces the fake
critic. It runs 1 disc update per student update, against DMD2's 5 critic updates.

The proposal: keep DP-DMD's step-0 diversity anchor unchanged, and replace only the
**DM term + fake critic** with DMAD's two-head linear loss. The step-0 diversity
graph is detached from the DM steps (`detach_after_first`), so the two objectives
don't interact.

## What the paper reports (for scale, not as a prior for Anima)

| Setting | DMAD | Baseline |
|---|---|---|
| ImageNet-64, 1 step, FID | 1.24 (1.04 with a projected D) | DMD2 1.51 |
| SDXL, 4 steps, COCO-10K FID / patch FID | 14.47 / 19.88 | DMD2 19.32 / 20.86; teacher 19.36 |
| Wan2.1-1.3B, 4 steps, VBench total | 84.70 | DMD2 84.56 |
| MiniMax-H3, 4 steps, human preference | 79.1 % over DMD2 (ties excluded, 33 % ties) | |
| Time per generator update (H200) | Wan-14B 38.5 s, MiniMax-H3 233 s | DMD2 162.9 s, 818 s |

Ablations (their Table 4):
- **Removing the real head** costs the most FID (SDXL 14.47 → 21.37).
- **Removing the teacher head** improves SDXL FID to 13.85 but costs patch FID,
  CLIP and VBench semantic (−4.19). The teacher head carries detail and prompt
  adherence.
- **Gap reweighting** and the **linear loss** (vs non-saturating) are each worth
  a smaller step.

The paper does not measure diversity. DMAD is still reverse-KL at the optimum, so
DMD's mode-seeking is expected to remain. That is why the DP-DMD anchor stays.

## Why it fits this loop

The current loop is already "DMD + GAN with real data". DMAD promotes the GAN to
carry the DM gradient and deletes the score difference. Inventory:

| DMAD piece | What exists now | Change |
|---|---|---|
| Generator term through a disc on renoised student output | `steps.gan_generator_term`: grad-bearing renoise of `x_pred` → teacher features → `turbo.disc` | Replace `softplus(−h)` with a linear `−h`; weight on the order of the DM term, not 0.03 |
| Real head | `fake_update` disc branch already scores renoised real `latents` vs `x_pred` | Keep as head R |
| Teacher head | — | New head T, trained on teacher samples vs `x_pred` |
| Disc backbone | Frozen teacher block features + ~2 M MLP head (`TeacherFeatureDiscriminator`, `disc_head="token"`) | See open question 1 |
| Fake critic | `fake_update`: 4 FM steps per student step + `fake_warmup_steps` | Removed under DMAD |
| DM term | `steps.dmd_surrogate`: CFG teacher + fake, no-grad, grad trick, `dm_x0_norm` | Removed under DMAD |
| L_CDM | `steps.cdm_off_trajectory_loss`: same real−fake surrogate at an off-trajectory `x0_off` | Same substitution: `−h(renoise(x0_off))` |
| Gap reweighting | `f_div_weighting_h` already keeps per-τ EMA bins over disc logits | New: per-band EMA of `mean h_R(real) − mean h_R(teacher)` → teacher weight `σ((median − Δ_b)/τ)` |

The **f-distill** lever is closed ([[project_turbo_rollup]]). DMAD is not a
re-proposal of it: f-distill reweights the score-difference gradient using disc
logits, whereas DMAD removes the score difference entirely.

## Where CFG comes from

Today CFG is baked through the DMD real score (`teacher_cfg_velocity`, α = 4). Under
DMAD it comes only through **head T's teacher samples**, which must therefore be
generated at CFG 4. Head R pulls toward the training-set distribution, which has no
guidance. The paper's no-teacher ablation losing CLIP / semantic score is the
expected failure shape here: head T is what keeps prompt adherence.

## Open questions

1. **Disc capacity.** Prop. 1 needs a near-optimal disc. The paper trains the disc
   backbone: a pretrained U-Net encoder for SDXL, LoRA r128 for MiniMax. A ~2 M head
   on frozen mid-block features is probably too weak to stand in for a fake critic.
   The natural candidate is to **reuse the fake LoRA stack as the disc backbone**:
   the fake view, plus block features, plus two MLP heads. Its parameter and memory
   footprint is roughly the critic's, and it can keep the warm-start init.
2. **Teacher-sample source.**
   - (a) Finish the per-step anchor rollout online, `k_anchor = 6` → 12 of the
     12-step grid. This gives a teacher sample on the same ε as the student, but
     costs ~6 more CFG steps per iteration, which is about what the 4 fake updates
     cost now. The speed win would disappear.
   - (b) Cache offline as the paper does: one CFG-4 teacher render per caption,
     paid once, with a full step count.

   Phase 0 uses (b).
3. **Backprop through the disc.** The DMD grad trick never backprops through
   teacher or fake. DMAD has to backprop through the disc backbone into `x`, and
   does so twice per step (DM term + L_CDM). The current GAN path already needs
   half-depth early exit plus `gan.grad_ckpt` because full depth OOM'd. Measure
   peak VRAM before committing to a tap depth. The paper taps final-block tokens
   for Wan.
4. **Stability.** The disc becomes the *only* quality gradient, and the linear loss
   does not saturate. The NFE=2 token-disc runaway (`superturbo_div01`,
   [[project_turbo_rollup]]) was a side-term failure; under DMAD the same failure
   would wreck the student. Keep R1 on from the start, and read `gan_disc_margin` /
   `gan_logit_spread`, not the hinge means.
5. **No `dm_x0_norm` analogue.** The paper's appendix states its equivalence does
   not cover DMD's per-sample normalization. λ also varies by 300× across their
   setups (0.003 ImageNet → 1 Wan/MiniMax). Expect a λ sweep.
6. **Real-data distribution.** Head R pulls the student toward whatever
   `data_dir` holds. Is the turbo dataset broad enough that this improves fidelity
   rather than imprinting a style?
7. **Grad routing.** DMAD's generator term applies to the final student output. Our
   `grad_step = "random"` puts the DM grad on a one-step x0 prediction at a random
   refinement step. Both are just "renoise an x0 estimate, score it", so the
   substitution is direct. But it has not been checked whether head T's ratio is
   meaningful for mid-rollout x0 predictions, which are blurrier than teacher
   endpoints.

## Phase 0 — one knob: the DM term

Base: the shipped NFE=4 warm-start recipe (`configs/methods/turbo.toml`). New
section `[dmad]`, off by default. Off must be byte-identical to the current loop,
following the same rule `weight_gen = 0` and `softrank.weight = 0` already obey.

- **On:**
  - disc backbone = fake LoRA stack, two heads, linear generator loss, R1 on;
  - teacher samples from the offline cache;
  - `dmd_surrogate` and the fake FM updates skipped; 1 disc update per student step.
- **Held off for this arm:** gap reweighting (fixed `λ_T = λ_R`), L_CDM
  (`[cdm] weight = 0` in **both** arms), `[gan]` side term.
- **Verdict:** rendered 4-step grids at `--cfg 1.0` vs a matched DP-DMD arm. Read
  pose diversity, text and saturation. Do not use scalars
  ([[project_turbo_rollup]]: fm_mse is anti-correlated). Confirm the arm's knobs
  in `ss_turbo_*` metadata / `.snapshot.toml`, since misplaced TOML sections fail
  silently.
- **Kill:**
  - the grid loses to the matched DP-DMD arm on pose or text, at 2 λ values; or
  - a disc runaway that R1 does not damp.

Later arms, one knob each and only if Phase 0 passes:

1. L_CDM on under DMAD.
2. Gap reweighting.
3. NFE=2 (superturbo).
4. Online teacher samples (2a), only if the cache becomes the bottleneck.

## Cost expectation (unmeasured)

- **Removed per step:**
  - 4 fake FM forward+backward;
  - 2 CFG teacher forwards + 1 fake forward in the DM term;
  - the same three again in L_CDM.
- **Added per step:**
  - one disc update over three sources (student, teacher, real);
  - grad-bearing disc forwards in the DM term and in L_CDM.
- **Net:** likely faster, since the fake inner loop is the bulk. VRAM is the open
  item (question 3).
