# Turbo DMAD — discriminator-carried distribution matching inside DP-DMD

Status: **PARKED** (2026-10-09). Phase −1 (premise probe) gates Phase 0; arms P1–P5
all failed to read (P1: head T plateaued at accuracy 0.58; P2–P5: the disc
separates teacher from student in bursts, then its gradient blows up and it falls
back to chance). No evidence for Prop. 1 on Anima; the one untested lever is a
strong R1 (see § Phase −1 results, "Where this leaves Phase −1").
Source: Yu et al., *DMAD: Distribution
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
   The one measurement we have points the wrong way: the shipped GAN head's
   generator gradient is elementwise orthogonal to the DM signal (agree-energy
   0.489 vs permutation null 0.488, cos 0.006;
   `docs/findings/turbo_gan_dm_grad_orthogonal.md`). That disc is a ~2 M hinge
   head on real vs student, not a balanced-BCE teacher head, so it does not refute
   Prop. 1 — but Prop. 1 holding on Anima is unmeasured. Phase −1 measures it.
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

## Phase −1 — does ∇ₓh_T point along the DM signal? (measure-only)

Prop. 1 is the whole premise, and it is checkable on the current loop without
building DMAD. `--dmad_probe` (`[dmad_probe]`, `scripts/distill_turbo/dmad_probe.py`)
rides an ordinary DP-DMD run — no resume bundle survives, so a fresh run from the
shipped warm-start init — and leaves the DM term, the critic and the student
untouched: every probe draw comes from its own generator, so the training RNG
stream is the same as with the probe off.

- **Disc:** its own fake-shaped LoRA stack (warm-started from `fake_init_weights`,
  like the critic) + head T (`TeacherFeatureDiscriminator`, token head) on the
  middle block, trained one update per step with balanced BCE, constant LR 5e-5.
  The teacher and student branches backward one at a time under
  `selective_block_grad_ckpt` (a batched pair OOM'd at step 9 on 16 GB); the
  renoised input requires grad, else the unsloth checkpoint drops the LoRA grads.
- **Teacher samples:** online, not cached — the step-0 anchor rollout is finished
  from `k_anchor` to σ=0 on the 12-step CFG-4 grid (open question 2a), so teacher
  and student share ε. Costs 6 more CFG steps per iteration; fine for a probe.
- **Probe:** every step, `g_T = ∂(−h_T)/∂x_pred` at the DMD's own (τ_dm, ε_dm),
  against `grad_signal` (pre f-distill): cosine and agree-energy, each with a
  permutation null on the same tensors — the telemetry of the closed sign-gate
  line (`docs/findings/turbo_gan_dm_grad_orthogonal.md` § What was measured), whose
  code was never committed. **Ceiling:** a second DM estimate at an independent
  (τ', ε') gives `cos(DM, DM')` — how far two DM draws agree with each other —
  the scale `cos(g_T, DM)` is read against.
- **Read** (`bench/turbo/dmad_probe_read.py`, pre-registered): window = second
  half of the rows.
  - UNCONVERGED: window disc accuracy < 0.75 — no read; run longer.
  - KILL: paired `cos − cos_null` within 3 SEM of 0. The disc cannot carry the
    DM gradient at this capacity; Phase 0 does not start.
  - PASS: mean `cos` ≥ 0.5 × mean `cos(DM, DM')` AND `agree − agree_null` > 3 SEM.
  - WEAK: between the two — aligned above the null but well short of DM's own
    draw-to-draw agreement. Owner call.
  - Also reported: τ-binned and per-DMD-grad-step `cos` (open question 7 — is head
    T's ratio meaningful on blurry mid-rollout x0 predictions?), peak VRAM.

### Phase −1 results

**P1 (2026-10-09) — UNCONVERGED.** Shipped recipe (`configs/methods/turbo.toml`,
GAN + L_CDM on) from the warm-start init, 150 steps, `--dmad_probe` defaults;
~13.8 s/step, peak 14.1 GiB on a 16 GB card. Rows:
`output/logs/turbo/20261009-093847/dmad_probe.jsonl`; read:
`bench/turbo/results/20261009-1016-dmad_probe_p1/result.json`.

- Head T stalled, not converging: disc accuracy 0.583 ± 0.017 in the read window
  (steps 76–150), flat from step ~15 on (deciles 0.55–0.64). Split by τ_d it stays
  ≤ 0.62 even at τ_d < 0.25, so the gate is not failing on the high-τ bins where
  teacher and student are indistinguishable by construction — the disc is weak.
- Likely cause, not yet tested: the run's `grad_clip = 1.0` is applied to the disc
  too, and its grad norm sat at 3–6, so each update was clipped to a fraction of
  an already small LR (5e-5), over 150 updates in total.
- Alignment, **not a verdict** (disc unconverged): cos(g_T, DM) = +0.030 ± 0.010 vs
  null +0.001; paired Δ +0.029 ± 0.010 (just under 3 SEM). Ceiling cos(DM, DM') =
  0.119 ± 0.014, so cos is ~0.25× ceiling. Agree-energy Δ +0.015 ± 0.009. By τ_dm:
  0.5 ≤ τ < 0.875 gives cos 0.057–0.076 (0.3–0.6× its ceiling); τ < 0.5 is near
  zero. Not the orthogonality of the shipped GAN head, but unreadable as it stands.
- The DM signal itself is noisy: two independent single-draw DM estimates agree at
  cos ≈ 0.12 (0.01–0.19 by τ). Any per-step alignment with DM is capped near there.

**P2 (2026-10-09) — UNCONVERGED.** P1 with the disc's two step-size knobs: LR 2e-4
and the disc exempt from the run's grad clip (`--dmad_probe_lr 2e-4
--dmad_probe_grad_clip 0`; the new knob defaults to the run's `optim.grad_clip`, so
P1 is unchanged by it). 150 steps, ~13.9 s/step, peak 14.1 GiB. Rows:
`output/logs/turbo/20261009-155349/dmad_probe.jsonl`; read:
`bench/turbo/results/20261009-1631-dmad_probe_p2/result.json`.

- Not a plateau like P1's: the disc separates in short bursts (acc 0.81 at step 9,
  0.95 at step 41), and each burst is followed by a grad-norm spike (14–24; P1's sat
  at 3–6) and a BCE jump (2.1 at step 11, 4.1 at step 47). After step 47 it has
  collapsed: margin 0.000 ± 0.000, acc 0.500, BCE 1.396 ≈ 2 ln 2 for the rest of the
  run. `h_probe` still moves (±0.4), so the head is not dead — it gives teacher and
  student the same score and has stopped separating them.
- Alignment in the window is at the null (cos −0.007 ± 0.005, null −0.001; agree Δ
  −0.004 ± 0.003), as expected from a sample-blind head; not a reading. The ceiling
  cos(DM, DM') = 0.128 ± 0.014 reproduces P1's 0.119.
- Read against the pre-registered P2 rule ("if head T still plateaus below 0.75 …
  evidence against the cheap-disc premise"): accuracy is below 0.75, but by
  diverging rather than plateauing. P1 stepped too little and P2 too much, so neither
  run shows how far a tuned head T can go at this capacity. Whether that counts
  against the premise, or a P3 between the two is run first (e.g. LR 2e-4 with a disc
  clip of ~5, or LR 1e-4 unclipped), is the owner's call.

**P3 (2026-10-09) — UNCONVERGED, stopped at step 80 on collapse.** P2 at LR 1e-4
(`--dmad_probe_lr 1e-4 --dmad_probe_grad_clip 0 --dmad_probe_stop_on_collapse 10`).
Rows: `output/logs/turbo/20261009-163511/dmad_probe.jsonl`; read:
`bench/turbo/results/20261009-1656-dmad_probe_p3/result.json`. Two probe knobs are
new with this arm: `dmad_probe.stop_on_collapse` (end the run after N consecutive
steps of |margin| < 1e-2; P1 never has 10 such steps, P2 reaches 10 at step 56) and
`dmad_probe.disc_steps` (k disc updates per step on the same pair at fresh (τ, ε);
unused so far). Rows now carry per-stage wall time.

- Head T can separate teacher from student at this capacity: acc 0.83 at step 10,
  0.997 / 0.999 at steps 17–18 (margin 2.4 / 5.8). Saturated, it diverges (grad norm
  46 at step 19, 43 at step 22) and settles at chance, as in P2. Window (steps
  41–80): acc 0.540 ± 0.019, cos −0.010 ± 0.010 vs null +0.004, ceiling 0.138.
- So P1's plateau was not a capacity limit, and the P2 rule's "evidence against the
  cheap-disc premise" branch, which was written for a plateau, does not apply. The
  failure is stability once the disc saturates.
- **Post hoc, not a read** (window chosen after seeing the deciles, n = 8): over
  steps 9–16, while the disc was learning but not saturated (margin 0.37 ± 0.08),
  cos(g_T, DM) = +0.179 ± 0.048 against the null +0.000 and a ceiling of 0.120 ±
  0.031. Over steps 17–24 (saturated, then collapsing) it is +0.008 ± 0.014. This is
  consistent with Prop. 1 holding in the unsaturated regime, where its optimum
  h* = log p_teacher − log p_student applies, and failing once a perfectly
  separating head's logit no longer tracks the density ratio. The pre-registered
  gate (acc ≥ 0.75) cannot tell the two regimes apart.
- Cost per step (rows' stage timings): teacher finish 2.46 s, one disc update 0.90 s,
  probe grad 0.43 s, ceiling 0.64 s — ~4.4 s of the 13.9 s step; the DP-DMD loop is
  the rest. An extra disc update per step costs ~0.9 s.

**P4 (2026-10-09) — UNCONVERGED.** P3 with the disc clipped at 5
(`--dmad_probe_grad_clip 5`), collapse stop 10 (never fired: 150 steps). The reader
now reports the window split at |margin| ≤ 1 vs > 1 beside the verdict (reported,
not voted; the verdict rule is unchanged). Rows:
`output/logs/turbo/20261009-170245/dmad_probe.jsonl`; read:
`bench/turbo/results/20261009-1739-dmad_probe_p4/result.json`.

- The clip bounds each update but not the gradient: pre-clip norms still reach 23–38
  (20 of 150 steps above 5). Instead of one collapse, head T cycles — margin up to
  0.8 (steps 31–45), down to 0.02 (76–90), up to 0.2 (121–135) — and never holds
  separation: window acc 0.556 ± 0.013, best decile 0.65.
- Window alignment at the null: cos −0.011 ± 0.012 (null +0.004), ceiling 0.106.
  Saturation split: 73 unsaturated rows read −0.015 ± 0.012; 2 saturated rows.
- **The P3 post-hoc reading does not hold up.** In P4 the positive alignment is
  confined to the first steps (steps ≤ 20: cos − null = +0.082 ± 0.031), and later
  steps with a learning, unsaturated disc (margin 0.3–1) read +0.004 ± 0.023
  (n = 17). P3's steps 9–16 are also early. An early-steps-only effect fits the
  warm start better than Prop. 1: the disc stack starts from the critic's weights
  (`fake_init_weights`), and DM = teacher − fake. Untested.

**P5 (2026-10-09) — stopped by hand at step 39, no read.** P4 plus the approximate
R1 of `gan.r1_weight` on the teacher branch (`--dmad_probe_r1_weight 1`, α 0.1; the
probe's backbone trains, so the MSE gradient reaches the LoRA stack, split exactly
across the two one-at-a-time branches) and a cold-started disc stack
(`--dmad_probe_cold_start`: zero-init LoRA instead of `fake_init_weights`). Rows:
`output/logs/turbo/20261009-174446/dmad_probe.jsonl`. Disc update 1.45 s (0.90 s
without R1), peak 14.0 GiB.

- Same cycle as P2–P4: acc 0.965 / 0.999 at steps 12–13 (margin 3.5), grad norm 25
  at step 14 and back to chance; 0.98 at step 20, grad norm 64 at step 22 and back;
  steps 31–39 at acc 0.50.
- **R1 at weight 1 does not bite:** the term ran 0.0001–0.09 (mostly < 0.01) against
  a BCE of ~1.4, so this arm says nothing about whether R1 stabilizes head T.
- Cold start, early alignment: cos − null +0.028 (steps 1–10) and +0.034 (11–20),
  against P4's warm-started +0.082 over steps ≤ 20. Small n, but the direction the
  warm-start explanation predicts.

**Where this leaves Phase −1.** Five arms, no read. Step size alone (LR 5e-5 → 2e-4,
clip 1 → off → 5) moves head T between too slow to learn and unstable once it
learns, and the only positive alignment seen (early steps of P3 / P4) shrinks once
the disc stack starts cold. Untested and the obvious next arm if the line is picked
up: P5 with R1 strong enough to compete with the BCE (weight ~100 from the logged
term's scale), or a squared-logit penalty. If head T still cannot hold acc ≥ 0.75
over the read window with that, close the line: Phase 0 makes this disc the
student's only quality gradient.

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
- **Net:** unknown. The fake inner loop is the bulk today, but with the fake LoRA
  stack as the disc backbone each grad-bearing disc forward is a full fake-DiT
  forward + backward into `x`, twice per step, plus a disc update over three
  sources — plausibly close to even. VRAM is the open item (question 3); Phase −1
  records both.
