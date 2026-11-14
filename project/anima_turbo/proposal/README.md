# Proposals

Turbo proposals; the closed ones moved from `docs/anima/proposal/` on 2026-11-12.

- [pref_rl.md](pref_rl.md): preference RL on the student (open, 2026-11-13). The user
  picks 0–8 of 16 renders a round; the picks and gelcrawl's retrieved / selected split train a
  preference head, and the student follows its gradient on step 0. Open: the head's feature
  basis (tagger, PE / DINOv3, DiT head). The round window and the head scripts are
  [../pref_rl/](../pref_rl/); usage is in [../README.md](../README.md).
- [superturbo.md](superturbo.md): the 2-step `rollout_grad` line (closed 2026-11-12).
- [turbo_apex.md](turbo_apex.md): APEX one-LoRA student (closed 2026-11-01).
- [turbo_dmad.md](turbo_dmad.md): DMAD inside DP-DMD (no gain read, 2026-10-10).
- [turbo_data_space.md](turbo_data_space.md): ideas from data-space iteration (not run).
