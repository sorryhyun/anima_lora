# anima_turbo — a few-step DP-DMD student on a soup teacher

A 4-step turbo distilled from a LoRA soup (merged14, merged15 from 2026-11-16). It ships as one
LoRA on plain anima-base-v1.0: `scripts/toolkits/merge_loras.py <soup>.safetensors
<turbo>.safetensors --normalize off --adaln sum` gives exact ΔW_soup + ΔT (the defaults, global
normalise and adaln average, break exactness). The turbo LoRA alone means nothing on base; the
merged LoRA is used at multiplier 1.0 only. The shipped base-teacher turbo
(`huggingface.co/sorryhyun/anima-turbo-4step`) is `docs/anima/methods/turbo.md`.

## State (2026-11-18)

- Shipped: `v4_preview` (`1112_1512`, on merged14).
- The line restarted on 2026-11-18. Its students were trained on an anchor whose chord ended at
  σ 0.75 while step 0 covered σ 1 → 0.9; the plan, reports, run configs and proposals up to then
  are in `_archive/anima_turbo/` and `_archive/configs/anima_turbo/` (local).
- The v1.1 delta finishes from the teacher's σ 0.9 state (the user's read, near tie), and
  `turbo_1118_1818` (step 0 alone on 7 of 28) lost the user's blind reads to it, and v1.1 finishes from the
  teacher's σ 0.9 state no better than from its own step 0
  ([reports/step0_k7of28.md](reports/step0_k7of28.md)). Teacher T against v1.1 S read level on
  `judge_hard` (T 5 / S 7) and behind on three random dataset captions (T 3 / S 9). Open: what the line distills toward. The 2-step student is parked ([plan.md](plan.md)).
- Preference RL is at its P1 gate, not passed ([pref_rl/p1.md](pref_rl/p1.md)). From 2026-11-18
  it is the plan's item 1, on the warm v1.1 delta on merged15 (the user's call).

## Layout

- [plan.md](plan.md): what is carried, the open items, what is measured again.
- [reports/](reports/README.md): one report per run or probe, added as runs finish.
- [findings/](findings/README.md): two findings on how to measure (an LR artifact in a gate, a
  permutation null for gradient comparisons).
- [configs/](configs/README.md): the live run config; finished ones in `_archive/configs/anima_turbo/`.
- [deploy/](deploy/): `deploy.sh` writes a student's release forms (LoRA for base, baked DiT, int8 GPTQ);
  [deploy/release_v4_longer.md](deploy/release_v4_longer.md) is the plan for the next Hub file.
- [pref_rl/](pref_rl/README.md): preference RL — proposal, P1 report, the round window (`make exp-turbo-pref`) and head scripts.

## Code and data

- Loop: `scripts/distill_turbo/` (`make turbo`); structure `docs/anima/structure/turbo.md`.
- Probes: `bench/turbo/` holds `probe_handoff.py`, `blind_sheets.py` and the judge prompts; the
  other turbo probes are in `_archive/bench/turbo/` (local) and the git history.
- Teacher: `models/diffusion_models/anima_merged1{4,5}.safetensors` (baked at 1.0, bf16) with their
  own `_int8_gptq` files; the base GPTQ is refused on a merged DiT.
- Checkpoints `output/ckpt/<id>/`, logs `output/logs/turbo/<stamp>/`, renders `output/tests/turbo4/`.
- Older material: `_archive/turbo_dmad/`, `_archive/proposals/turbo_*`.
