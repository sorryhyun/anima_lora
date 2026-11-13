# anima_turbo — a few-step DP-DMD student on the merged14 soup

Opened as a project 2026-11-12. The goal is a 4-step turbo whose teacher is the soup
`output/ckpt/anima_merged14.safetensors` (8-shard concat_exact LoRA, rank 256, adaln averaged). It
ships as **one LoRA that loads on plain anima-base-v1.0**: `scripts/toolkits/merge_loras.py
anima_merged14.safetensors <turbo>.safetensors --normalize off --adaln sum` gives exact ΔW14 + ΔT.
The defaults (global normalise, adaln average) break exactness. The turbo LoRA alone means nothing
on base, and the merged LoRA is used at multiplier 1.0 only.

The shipped base-teacher turbo (`huggingface.co/sorryhyun/anima-turbo-4step`) is documented in
`docs/anima/methods/turbo.md`; this line is the work after it.

## Layout

- [1112_v1.md](1112_v1.md): the setup of the first 1000-iteration run (`1112_1512`, shipped as
  `v4_preview`): the four changes against arm 0's control with the evidence for each, and what is
  kept. The 250-iteration control of the current arms is this recipe.
- [idea.md](idea.md): candidates that are not arms: the anchor's weight against the DM signal on
  step 0, high-σ DM queries on step 0, a 2-step handoff probe, a lower-resolution anchor, DAVE at
  inference. Low-frequency weighting of the anchor loss is closed there.
- [plan.md](plan.md): the open plan: the A/B protocol, the next run (a warm start from Hub v3)
  and what follows it. The closed 2-step plan and its results are
  `91d476e0:scripts/distill_turbo/plan.md`.
- [configs/](configs/): one TOML per run, named by its run id (`mmdd_hhmm`). Launch with
  `make turbo ARGS="--config project/anima_turbo/configs/<id>.toml --queue"`.
- [deploy/](deploy/): `deploy.sh` writes the release forms of a student: one LoRA for base, the
  baked DiT, and its 4-step / cfg 1 int8 GPTQ.
- [reports/](reports/): one report per run or probe.
  - [arm0.md](reports/arm0.md): arm 0 control and twin. The two runs share only 9–10 % of ΔW,
    both random-walk from step 125, and the σ_a norm profile gives arm 3 = adaptive clip alone.
    The control's raw 250 sits inside a divergence burst. The render noise floor is large, and
    composition is the gap both seeds share. GAN-off: GAN is not the walk.
  - [handoff.md](reports/handoff.md): teacher / student handoff at each step boundary on the arm 0
    students. Step 0 (σ 1 → 0.9) is the largest single loss of the teacher's layout; with the
    teacher drawing step 0 the student's steps 1–3 keep the view from below. Render read by Claude,
    the user's read pending.
  - [step0.md](reports/step0.md): the step-0 arms. The anchor target is right (S-a). At 250
    iterations no trained step 0 gets past the teacher's instantaneous velocity (S-b, S-b2, S-c.1),
    and four anchor draws buy a small gain (S-c.2). The fit follows the LR at the end of training
    and not the length (`1112_1336`, `1112_1410`); at a held-out MSE of 0.12 the layout is still
    the students' own.
  - [step0_fit.md](reports/step0_fit.md): how good the step-0 fit has to be, and Hub v3 as a
    fitted step 0. The layout is in the low band of the step-0 residual and comes back in
    proportion to the fit; Hub v3 (10k it) sits at 0.3 of the reference where the 1k students sit
    at 0.85, and its step 0 carries over to merged14 untrained (held-out MSE 0.063 against 0.146).
  - [walk.md](reports/walk.md): the measurements taken before arm 0. It holds what the 2-step line
    measured (the batch-1 random walk), the 4-step ΔW overlap and the 2-step gradient norms.
  - [mfd.md](reports/mfd.md): MFD in the DM slot (2-step 100 / 500-step arms, 2026-11-04), the
    anchor-free MFD collapse (4-step m14, 2026-11-10) and arm 1 on the v1 recipe (2026-11-13):
    cleaner, and the seeds of a prompt draw one image.
  - [warm_v3.md](reports/warm_v3.md): the control warm-started from Hub v3. It passes the gate
    against the control and regresses from v3 untrained on merged14, which sits at the teacher's
    seed diversity.
- [vatdm.md](vatdm.md): VA-TDM (arXiv:2611.06796) against the loop and MFD; idea 1 is
  `fake_transition`.
- [proposal/](proposal/): turbo proposals; the closed ones moved from `docs/anima/proposal/` on
  2026-11-12.
  - [pref_rl.md](proposal/pref_rl.md): preference RL on the student (open, 2026-11-13). The user
    picks 0–8 of 16 renders a round; the picks and gelcrawl's retrieved / selected split train a
    preference head, and the student follows its gradient on step 0. Open: the head's feature
    basis (tagger, PE / DINOv3, DiT head). The round UI is [pref_rl/](pref_rl/) (`make
    exp-turbo-pref`): a render server keeps the policy loaded as one daemon job, and the window
    logs train rounds and blind A/B grids to `output/turbo_pref/<session>/picks.jsonl`.
  - [superturbo.md](proposal/superturbo.md): the 2-step `rollout_grad` line (closed 2026-11-12).
  - [turbo_apex.md](proposal/turbo_apex.md): APEX one-LoRA student (closed 2026-11-01).
  - [turbo_dmad.md](proposal/turbo_dmad.md): DMAD inside DP-DMD (no gain read, 2026-10-10).
  - [turbo_data_space.md](proposal/turbo_data_space.md): ideas from data-space iteration (not run).
- [findings/](findings/): settled turbo findings, moved from `docs/anima/findings/` on 2026-11-12
  (still indexed in `docs/anima/findings/README.md`).
  - [turbo_apex_no_quality.md](findings/turbo_apex_no_quality.md): APEX unstable as DMD, blurred as
    regression.
  - [turbo_fei_band_deficit_falsified.md](findings/turbo_fei_band_deficit_falsified.md): FEI
    band-deficit lever, dead twice.
  - [turbo_gan_dm_grad_orthogonal.md](findings/turbo_gan_dm_grad_orthogonal.md): the GAN gradient is
    elementwise DM-orthogonal.
  - [turbo_tau_critic_interference_lr_artifact.md](findings/turbo_tau_critic_interference_lr_artifact.md):
    the τ-split critic's interference is an LR artifact.
  - [turbo_moment_gaps.md](findings/turbo_moment_gaps.md): dataset / CFG 1 / CFG 4 / student
    moment gaps.

## Code and data

- Loop: `scripts/distill_turbo/` (`make turbo`); method doc `docs/anima/methods/turbo.md`,
  structure `docs/anima/structure/turbo.md`.
- Benches and probes: `bench/turbo/`, including `dw_overlap.py` (ΔW share between runs),
  `probe_grad_noise.py` (2-step only), `probe_handoff.py` (teacher / student handoff per step
  boundary), `moment_gap.py` and `judge_hard.txt`.
- Teacher: `models/diffusion_models/anima_merged14.safetensors` (baked at 1.0, bf16) with its own GPTQ
  `anima_merged14_int8_gptq.safetensors`. The base GPTQ is refused on the merged DiT.
- Checkpoints in `output/ckpt/<id>/` (125-step saves, `_ema`, `resume.pt`). TB and
  `grad_steps.jsonl` in `output/logs/turbo/<stamp>/`. Renders in `output/tests/turbo4/` (2-step:
  `output/tests/superturbo/`).
- Older material: `_archive/turbo_dmad/` (the removed DMAD path), retired proposals under
  `_archive/proposals/turbo_*`.

## Runs

| Id | Arm | Change against control | State |
|---|---|---|---|
| `1112_0833` | 0, control | the baseline in `1112_v1.md`, seed 42 | done, rendered |
| `1112_0834` | 0, null twin | seed 43 | done, rendered |
| `1112_0917` | GAN-off probe (measurement only) | `gan.weight_gen = 0` | done: GAN is not the walk |
| `1112_1044` | S-b, anchor alone | `dpdmd.anchor_only`, no CDM / GAN / fake warmup | done: step 0 at the teacher's instantaneous level (`reports/step0.md`) |
| `1112_1110` | S-b2 | `1112_1044` at student LR 2e-4 | done: worse. User's read: 2e-4 steps are too large for the noise, 5e-5 steps too small to get past the instantaneous level in 250 it |
| `1112_1143` | S-c.1 | `dpdmd.dm_first_step` on (anchor kept) | done: probe `S` 0.445 / 0.441 (EMA 250 / 125), at the twin's level; guitarist pose moves to the jump, camera does not; raw 250 broken (`reports/step0.md`). User's read: improved, less diverse |
| `1112_1230` | S-c.2 probe | S-b + `dpdmd.anchor_draws 4` | done: small consistent gain, every paired number favours it (anchor MSE 0.157 → 0.147, `P>S@1` +0.01 / +0.03); passes by the repo gate rule, size thresholds in its header not met (`reports/step0.md`) |
| `1112_1336` | S-b long | S-b resumed (from step 125) to 1000 iterations, EMA span kept at 50 updates | done: held-out anchor MSE 0.165 / 0.143 / 0.155 / 0.140 at 250 / 500 / 750 / 1k, flat; length on the constant LR is not the lever (`reports/step0.md`) |
| `1112_1410` | S-b long, annealed tail | `1112_1336` from step 750 with `lr_schedule cosine` | done: 0.123 raw / 0.127 EMA; the anneal is what the 500-iteration students had. `P>S@1` 0.495 / 0.587, the layout does not come with it |
| `1112_1512` | v1 (`1112_v1.md`) | 1000 it, cosine, EMA 0.2 of summed LR, `dm_first_step` on, saves every 250 | done; 1k EMA shipped 11-12 as Hub `anima_turbo_4step_v4_preview`. Against Hub v3: the multi-character layout holds one scene; the solo guitarist draws one frontal pose on every seed, raw and EMA |
| `turbo_1113_1105` | control for the post-v1 arms | the v1 recipe at 250 it (cosine over the 250), saves every 125 | done, rendered; held-out anchor MSE 0.149 |
| `turbo_1113_1104` | 1, MFD | `turbo_1113_1105` + `mfd.dtau 0.1` / `max_steps 4` / `to_zero` | done: fails on the renders. Cleaner lines and all five characters on every seed, but the four seeds of a prompt draw one image and the sign text is worse (`reports/mfd.md`) |
| `turbo_1113_1221` | warm start from Hub v3 | `turbo_1113_1105` with the student seeded from Hub v3 | done: passes the `plan.md` gate against the control, but regresses from v3 untrained on merged14 on every number (sameness 0.326 → 0.465, `S` cos 0.639 → 0.593, anchor MSE 0.063 → 0.070) and in the user's read; the 1k run is dropped (`reports/warm_v3.md`) |
| `turbo_1113_1309` | warm start, `dm_first_step` off | `turbo_1113_1221` with `dpdmd.dm_first_step = false` | done: not the cause. Anchor MSE stays at v3's 0.063, but step 0 still raises the sameness (cross cell 0.405 against v3 0.326, `1221` 0.429); steps 1–3 regress as before (`T>S@1` cos 0.773) (`reports/warm_v3.md`) |
| `turbo_1113_1438` | warm start, fake seeded too | `turbo_1113_1221` with `network.fake_init_weights` = Hub v3 | done: not the cause. The DM gap starts lower (`dm_rel_gap` 0.214 against 0.292 in the first 25 it), but sameness is 0.483 (v3 0.326, `1221` 0.465); `S` cos 0.610, anchor MSE 0.070 (`reports/warm_v3.md`) |

## State (2026-11-13)

`v4_preview` (`1112_1512`, 1k EMA on the `1112_v1.md` recipe) is shipped. The 2-step line is closed
at the level of `1111_1115` step 250.

The open gap is composition: every merged14 student draws one frontal layout per prompt. It sits in
step 0 (`reports/handoff.md`), and the layout comes back in proportion to the step-0 anchor fit
(`reports/step0_fit.md`). The 1k students fit at 0.85 of the teacher's cond velocity; Hub v3, after
10k iterations, at 0.30, with the view from below in its renders, and its step 0 carries over to
merged14 untrained (0.063 against 0.146–0.149). Arm 1 (MFD) failed: one image per prompt, mostly
through steps 1–3 (`reports/mfd.md`).

The warm start from Hub v3 (`turbo_1113_1221`) regressed from v3 itself: Hub v3 on merged14 with no
training draws seeds as varied as the teacher's (sameness 0.326 against 0.328) and is the best
4-step result on merged14; the v1 recipe pulls any start toward the control, with `dm_first_step`
off (`turbo_1113_1309`) and with the fake seeded from v3 (`turbo_1113_1438`) alike
(`reports/warm_v3.md`).
