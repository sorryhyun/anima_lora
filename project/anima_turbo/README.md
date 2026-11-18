# anima_turbo — a few-step DP-DMD student on a soup teacher

A 4-step turbo distilled from a LoRA soup (merged14, merged15 from 2026-11-16). It ships as one
LoRA on plain anima-base-v1.0: `scripts/toolkits/merge_loras.py <soup>.safetensors
<turbo>.safetensors --normalize off --adaln sum` gives exact ΔW_soup + ΔT (the defaults, global
normalise and adaln average, break exactness). The turbo LoRA alone means nothing on base; the
merged LoRA is used at multiplier 1.0 only. The shipped base-teacher turbo
(`huggingface.co/sorryhyun/anima-turbo-4step`) is `docs/anima/methods/turbo.md`.

## State (2026-11-18)

- Shipped: `v4_preview` (`1112_1512`, the [1112_v1.md](1112_v1.md) recipe on merged14). The
  2-step line is closed.
- Students draw one layout per prompt ([reports/handoff.md](reports/handoff.md),
  [reports/step0_fit.md](reports/step0_fit.md)).
- On merged15 the shipped recipe failed at 1k ([reports/m15_1k.md](reports/m15_1k.md)). The
  official v1.1 delta as init keeps the best steps 1–3 of the line; from it the anchor alone on the
  teacher's 28-step schedule (`turbo_1116_1810`, 14 of 28) is adopted as the target to scale
  ([reports/v11_init.md](reports/v11_init.md), [anchor_only_recipe.md](anchor_only_recipe.md)).
- The full-dataset anchor cache (`turbo_1117_0823`) is the best anchor student; rank 96 ties it
  ([reports/step0_depth.md](reports/step0_depth.md)). Feature matching at block 27 and a paired GAN
  on the σ 0.75 state (FastGen's conv head on taps 13 / 20 / 27 with R1, `turbo_1118_1353`) are in
  the recipe ([reports/paired_fm_gan.md](reports/paired_fm_gan.md)).
- Step 0 is the second priority since 2026-11-18. First: a 2-step student on the fitted step 0
  (superturbo) and steps 1–3 ([plan.md](plan.md) items A and B).
- Preference RL is at its P1 gate, not passed ([pref_rl/p1.md](pref_rl/p1.md)).

Reading: the gap is composition, and the layout is set in step 0. The breaks left in the anchor
students' renders are step 0's residual to the anchor target; a third to a half of it has to go,
and `0823` falls short of that ([reports/step0_depth.md](reports/step0_depth.md)).

## Layout

- [plan.md](plan.md): the open plan.
  [question.md](question.md): turbo + SnapRef, unrun. [vatdm.md](vatdm.md): VA-TDM against the loop.
- [reports/](reports/README.md): one report per run or probe; [reports/runs.md](reports/runs.md)
  is the run table.
- [findings/](findings/README.md): settled findings. Closed proposals are in `_archive/proposals/` (`turbo_*`, `superturbo.md`).
- [configs/](configs/README.md): the live run config; finished ones in `_archive/configs/anima_turbo/`.
- [deploy/](deploy/): `deploy.sh` writes a student's release forms (LoRA for base, baked DiT, int8 GPTQ).
- [pref_rl/](pref_rl/README.md): preference RL — proposal, P1 report, the round window (`make exp-turbo-pref`) and head scripts.

## Code and data

- Loop: `scripts/distill_turbo/` (`make turbo`); structure `docs/anima/structure/turbo.md`.
- Benches and probes: `bench/turbo/` (`dw_overlap.py`, `probe_handoff.py`, `moment_gap.py`, …).
- Teacher: `models/diffusion_models/anima_merged1{4,5}.safetensors` (baked at 1.0, bf16) with their
  own `_int8_gptq` files; the base GPTQ is refused on a merged DiT.
- Checkpoints `output/ckpt/<id>/`, logs `output/logs/turbo/<stamp>/`, renders `output/tests/turbo4/`.
- Older material: `_archive/turbo_dmad/`, `_archive/proposals/turbo_*`.
