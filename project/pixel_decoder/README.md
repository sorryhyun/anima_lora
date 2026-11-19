# pixel_decoder — a light pixel head on the frozen Anima DiT's last features

Opened 2026-11-19 out of `docs/anima/proposal/` (the user: its own project, not a part of
`project/anima_turbo/`). FastFrames (arXiv:2611.23226 §3.2, App. C) replaces the VAE decoder
with a small head that reads the frozen DiT's pre-head features and writes a 16×16 RGB patch per
token. Here the same on Anima: a head trained against the VAE decode, DiT untouched. What it is
for, in order: pixel-space losses that fit beside the DiT (cjk_rl arm D, a GAN or a refiner for
turbo), sharpness from a pixel loss on the head alone, and decode memory on MOSS later.

- [proposal.md](proposal.md) — the idea, what a head buys and costs, phases P1–P5.
- [plan.md](plan.md) — the order of work and the open items.
- [reports/](reports/) — one report per run or probe, indexed in its `README.md`.
- [promotion_maybe.md](promotion_maybe.md) — draft Hugging Face card material for the head as a
  turbo decoder (measured decode cost, turbo share estimated).
- Code and per-run records: `bench/pixel_readout/` (`linear_probe.py` P1, `convnext_head.py`
  head training, `e2e_speed.py`, `rollout_profile.py`, `render_sheets.py`, `ceiling_probe.py`, `overfit_probe.py`, `grad_probe.py`;
  its `README.md` maps the scripts; run records are in `reports/`). Results:
  `bench/pixel_readout/results/` (local).

State (2026-11-19): P1 and P2 done on anima-base, teacher-forced at σ 0.0577. Test PSNR against
the VAE decode, 30 held-out images / 8 text images, one seed, spread unknown:

| | test | text |
|---|---|---|
| P1 linear readout (`readout_1118_2219`) | 26.28 | 21.61 |
| P2 head with a P1 skip (`convnext_1119_0835`) | 26.50 | 21.58 |
| P2 head, no skip (`convnext_1119_0938`) | 32.93 | 27.45 |
| the same, gradient over 4 rollouts at LR 1e-3 (`convnext_1119_0950`) | 32.96 | 27.46 |
| no skip + one 2× upsample stage (`convnext_1119_1106`) | 33.64 | 28.04 |
| VAE decode of the DiT's own x̂0 (`ceiling_1119_0920`) | 37.00 | 32.02 |
| fp32 stream + head compile + replay buffer, 900 rollouts (`replay_1119_1225`) | 33.35 | 27.68 |
| + target VAE(x̂0), manga pages 0.33, 2,000 rollouts (`pagemix_1119_1242`) | 34.83 | 29.65 |

The current head is `bench/pixel_readout/results/20261119-1304-pagemix_1119_1242/head.safetensors`
(local): 24.9M parameters, no skip, two ConvNeXt blocks at token resolution and one after a 2×
upsample, 22 minutes of training. The first P2 head carried P1's linear as a skip and did not train past it
(`reports/p2_skip_2026_11_19.md`). The user's verdict on P2 (2026-11-19): "대 합격"; the
training method is settled.

P3 (`reports/p3_2026_11_19.md`): on 64 generated images (16 prompts × 2 seeds, standard sampler)
the same head reads 30.90 on colour with the base DiT and 31.24 with `anima_merged15`, 22.51 /
22.85 on mono prompts, against VAE(x_final). Claude's call, sheets sent to the user: the head
generalizes; dense fine strokes (screentone, hatching) are what is left. A 2× upsample stage
(`convnext_1119_1106`) took test / text to 33.64 / 28.04 and is in the recipe. The fp32 stream,
a compiled head and a replay buffer (`replay_1119_1225`, `reports/replay_recipe_2026_11_19.md`)
read 33.35 / 27.68 in 44 % of the time; the user read the renders as level and the recipe took
it. P3b (`pagemix_1119_1242`, `reports/pagemix_2026_11_19.md`): manga pages in the mix with
target VAE(x̂0); manga eval 28.15 against `mangaup`'s 26.64, tone level with the VAE. The user's
verdict (2026-11-19): P3 passed; speed lines slightly soft under zoom. P4
(`reports/p4_t2_2026_11_19.md`): a head warm-started from pagemix and trained 7 minutes at turbo's
σ 0.5 reads turbo 4-step rollouts at 34.58 colour / 27.85 mono (13.73 / 14.16 before); the user's
verdict: P4 passed. Not yet trained: Spectrum's σ 0.10 (T1 / T3; the `1106` head read colour
28.52 there, `e2e_1119_1209`).
