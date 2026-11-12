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

- [plan.md](plan.md): the open plan. It covers the fixed 4-step recipe, the A/B protocol (250 it,
  EMA 0.2, one null twin, renders on `bench/turbo/judge_hard.txt` beside the teacher), the gradient
  norm section and the arm order. The closed 2-step plan and its results are that file at commit
  `91d476e0` (then at `scripts/distill_turbo/plan.md`).
- [configs/](configs/): one TOML per run, named by its run id (`mmdd_hhmm`). Launch with
  `make turbo ARGS="--config project/anima_turbo/configs/<id>.toml --queue"`.
- [reports/](reports/): one report per run or probe.
  - [arm0.md](reports/arm0.md): arm 0 control and twin. The two runs share only 9–10 % of ΔW,
    both random-walk from step 125, and the σ_a norm profile gives arm 3 = adaptive clip alone.
    The control's raw 250 sits inside a divergence burst; rough render read of the control.
  - [mfd.md](reports/mfd.md): MFD in the DM slot (2-step 100 / 500-step arms, 2026-11-04) and the
    anchor-free MFD collapse (4-step m14, 2026-11-10).
- [vatdm.md](vatdm.md): VA-TDM (arXiv:2611.06796) against the loop and MFD; idea 1 is
  `fake_transition`.
- [proposal/](proposal/): earlier turbo proposals, moved from `docs/anima/proposal/` on 2026-11-12.
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
  `probe_grad_noise.py` (2-step only), `moment_gap.py` and `judge_hard.txt`.
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
| `1112_0833` | 0, control | the recipe in `plan.md`, seed 42 | done; renders queued |
| `1112_0834` | 0, null twin | seed 43 | done; renders queued |
| `1112_0917` | GAN-off probe (measurement only) | `gan.weight_gen = 0` | queued |

## State (2026-11-12)

The 2-step line (superturbo, `dpdmd.rollout_grad`) is closed at the level of `1111_1115` step 250.
Every noise-lowering route (`dm_draws`, `grad_accum`, checkpoint averages) landed on that level.
The 4-step line is in the same batch-1 random-walk regime (`reports/arm0.md`). Arm 0's training
is done. The control's render read is in; the twin's renders are queued.
