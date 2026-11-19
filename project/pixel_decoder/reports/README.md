# pixel_decoder reports

One file per run or probe. Per-run settings and numbers: `bench/pixel_readout/README.md`.

- [p2_skip_2026_11_19.md](p2_skip_2026_11_19.md) — the P2 head with a P1 linear skip stayed at P1's level (test 26.50); the target was reachable (VAE(x̂0) 37.00) and the head fit 16 images under every init; without the skip the same run reaches test 32.93 / text 27.45.
- [p3_2026_11_19.md](p3_2026_11_19.md) — the `0938` head on 64 generated images: colour 30.90 base / 31.24 with merged15 (LoRA), mono 22.51 / 22.85; real manga pages teacher-forced: VAE vs original 21.5–39.8, VAE(x̂0) vs VAE(x0) 24.3–36.3.
