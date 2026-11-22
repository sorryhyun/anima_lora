# pixel_decoder — a light pixel head on the frozen Anima DiT's last features

Opened 2026-11-19 out of `docs/anima/proposal/` (the user: its own project, not a part of
`project/anima_turbo/`). FastFrames (arXiv:2611.23226 §3.2, App. C) replaces the VAE decoder
with a small head that reads the frozen DiT's pre-head features and writes a 16×16 RGB patch per
token. Here the same on Anima: a head trained against the VAE decode, DiT untouched. What it is
for, in order: pixel-space losses that fit beside the DiT (cjk_rl arm D, a GAN or a refiner for
turbo), sharpness from a pixel loss on the head alone, and decode memory on MOSS later.

- [proposal.md](proposal.md) — the idea, what a head buys and costs, phases P1–P5.
- [plan.md](plan.md) — the ×1 base recipe, the plan of its GAN finish (the release to be), the
  P5 topics and the open items. P1 – P4: `reports/phases_p1_p4_2026_11_19.md`.
- [x2_plan.md](x2_plan.md) — the ×2 output topic of P5: pointers to the reports of everything that ran, the current state, the sequence after the ×1 head is fixed (B only, then a short GAN finish), next arms, open items.
- [reports/](reports/) — one report per run or probe, indexed in its `README.md`.
- [deploy/](deploy/) — the Hub card and NOTICE of `sorryhyun/anima-pixel-decoder` (private);
  the card's sample images are local (`bench/pixel_readout/results/card_samples_1119/`) and on the Hub.
- Code and per-run records: `bench/pixel_readout/` (`linear_probe.py` P1, `convnext_head.py`
  head training, `e2e_speed.py`, `render_sheets.py`, `tone_probe.py`; ×2 training is
  `scripts/pixel_head/train_x2.py` on `library/anima/pixel_head_train.py`; the finished probes
  (`rollout_profile.py`, `ceiling_probe.py`, `overfit_probe.py`, `grad_probe.py`, `feat_probe.py`) are in
  `_archive/bench/pixel_readout/` (local);
  its `README.md` maps the scripts; run records are in `reports/`). Results:
  `bench/pixel_readout/results/` (local); on 2026-11-19 the concluded runs' dirs were deleted, their numbers stay in
  `reports/`; the heads code or docs name (readout_1118_2219, convnext_1119_0835 / 0938, pagemix, mixr) kept
  without sheets; realr / x2real and their reads kept whole.

State (2026-11-19): P1 – P4 complete, with the release, the promotion into `library/` and the
ComfyUI node (the user: "p4는 완료된걸로 갱신하고 커밋하자"). Next: the P5 topics (`plan.md`).

State (2026-11-22): the ×1 base is `x1ll_1122_1000` (12,000 rollouts at LR 5e-5; test 34.25 dB /
LPIPS 0.0725 at σ 0.0577 against the Hub head `realr_1119_2200`'s 33.75 / 0.0807;
`reports/x1_recipe_2026_11_22.md`), the init of ×2, RGBA and of a GAN finish that is to be the ×1
release (the user's proposal; `plan.md`). The base is fixed by the user:
`output/pixel_head/x1ll_1122_1000/head.safetensors` (local). The finish is not run.

P5, ×2 output (2026-11-20): a ×2 file shipped as a test file; LPIPS, a tone term and the SR
line's DMD / GAN critic did not bring the ×2 stage up to the chain; the user chose arm B (the
real 2× image as the target, base points only) as the next arm (`x2_plan.md`; its runs: `reports/x2_arm_b_2026_11_20.md`).

P1 and P2 on anima-base, teacher-forced at σ 0.0577. Test PSNR against
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

The P3 head was `bench/pixel_readout/results/20261119-1304-pagemix_1119_1242/head.safetensors`
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
verdict: P4 passed. One head over both decode points (`mix_1119_1415`,
`reports/p4_mix_2026_11_19.md`, base σ 0.10 and turbo σ 0.5 in turn): turbo rollouts 34.42 /
27.74 against the T2 head's 34.67 / 27.91, base renders 32.91 / 25.65 against the pagemix head's
29.06 / 23.13; the user settled the recipe on it. Release head `mixr_1119_1455` (base σ 0.0577 … 0.1364 and turbo σ 0.5,
3,000 rollouts): turbo 35.90 / 29.01, base 28 steps 33.83 / 26.38; the user's verdict: good
enough to release, P4 complete. Released 2026-11-19 to the private Hub repo
`sorryhyun/anima-pixel-decoder` (`reports/release_2026_11_19.md`). Promoted the same day: the model
core in `library/anima/pixel_head.py`, `inference.py --decoder pixel_head` (`make test PIXEL=1`),
`docs/anima/inference/pixel_head.md`. ComfyUI node pack `ComfyUI-Anima-PixelDecoder` built the
same day and published (GitHub, ComfyUI registry 0.1.0); turbo in ComfyUI 1.127 s with VAE
Decode vs 1.005 s with the head (`reports/comfy_node_2026_11_19.md`).
