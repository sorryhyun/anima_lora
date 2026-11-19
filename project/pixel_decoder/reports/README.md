# pixel_decoder reports

One file per run or probe; every run's settings, numbers and reads live here. Code map: `bench/pixel_readout/README.md`.

- [p1_readout_2026_11_18.md](p1_readout_2026_11_18.md) — P1 `readout_1118_2219`: a per-token ridge linear from the pre-head features reads test 26.28 against token-mean bilinear 22.07.
- [p2_skip_2026_11_19.md](p2_skip_2026_11_19.md) — the P2 head with a P1 linear skip stayed at P1's level (test 26.50); the target was reachable (VAE(x̂0) 37.00) and the head fit 16 images under every init; without the skip the same run reaches test 32.93 / text 27.45.
- [upsample_2026_11_19.md](upsample_2026_11_19.md) — `convnext_1119_1106`, one 2× upsample stage: test 33.64 / text 28.04 against `0938`'s 32.93 / 27.45, 38 / 38 up; manga +0.3 … +0.8 (`mangaup_1119_1152`); adopted.
- [replay_recipe_2026_11_19.md](replay_recipe_2026_11_19.md) — decode cost on the Spectrum sampler (head 7 ms vs VAE 141–828 ms, DiT 6.58 s), rollout profile (DiT + VAE 82 %), and `replay_1119_1225` (fp32 stream + head compile + replay buffer): test 33.35 / text 27.68 against `1106`'s 33.64 / 28.04 in 44 % of the time; adopted on the user's read.
- [p3_2026_11_19.md](p3_2026_11_19.md) — the `0938` head on 64 generated images: colour 30.90 base / 31.24 with merged15 (LoRA), mono 22.51 / 22.85; real manga pages teacher-forced: VAE vs original 21.5–39.8, VAE(x̂0) vs VAE(x0) 24.3–36.3.
- [pagemix_2026_11_19.md](pagemix_2026_11_19.md) — P3b `pagemix_1119_1242` (target VAE(x̂0), pages 0.33, 2,000 rollouts): test 34.83 / text 29.65 / manga 28.15 against replay's 33.35 / 27.68 and mangaup's manga 26.64; tone level with the VAE; P3 passed on the user's read.
- [p4_t2_2026_11_19.md](p4_t2_2026_11_19.md) — P4 T2 (turbo 4-step, last forward σ 0.5): the pagemix head reads 13.73 colour / 14.16 mono on turbo rollouts (`t2probe_1119_1328`); arm A `t2a_1119_1337` (teacher-forced at σ 0.5, warm start, 500 rollouts) reads 34.58 / 27.85 on the same rollouts, level with its teacher-forced 34.35 / 27.67; P4 passed on the user's read.
