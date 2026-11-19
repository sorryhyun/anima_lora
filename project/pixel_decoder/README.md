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
- Code and per-run records: `bench/pixel_readout/` (`linear_probe.py` P1, `convnext_head.py`
  P2 training, `render_sheets.py`, `ceiling_probe.py`, `overfit_probe.py`, `grad_probe.py`;
  its `README.md` holds every run's settings and numbers). Results:
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

The current head is `bench/pixel_readout/results/20261119-0959-convnext_1119_0938/head.safetensors`
(local): 29.7M parameters, no skip, three ConvNeXt blocks at token resolution, 26 minutes of
training. The first P2 head carried P1's linear as a skip and did not train past it
(`reports/p2_skip_2026_11_19.md`). The user's verdict on P2 (2026-11-19): "대 합격"; the
training method is settled.

P3 (`reports/p3_2026_11_19.md`): on 64 generated images (16 prompts × 2 seeds, standard sampler)
the same head reads 30.90 on colour with the base DiT and 31.24 with `anima_merged15`, 22.51 /
22.85 on mono prompts, against VAE(x_final). Claude's call, sheets sent to the user: the head
generalizes; dense fine strokes (screentone, hatching) are what is left. A 2× upsample stage
(`convnext_1119_1106`) took test / text to 33.64 / 28.04 and is in the recipe. Next: the manga /
greyscale training arms in `plan.md`. Not yet
checked: any σ other than the last one.
