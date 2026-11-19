# Promotion notes (maybe) — the pixel head as a turbo decoder

Draft material for a Hugging Face card, should the head ship as a decoder for turbo inference.
Not a record; the records are `bench/pixel_readout/README.md` and `reports/`.

## Measured (`e2e_1119_1209`, 2026-11-19)

anima-base, `make test SPECTRUM=1` sampler (Euler 28, shift 3, CFG 4), 16 images at ~1 MP,
RTX 5070 Ti, compiled blocks, one seed. Head = `convnext_1119_1106` (24.9M params, 95 MB).

| decode, per image (median) | time | peak VRAM above the resident DiT |
|---|---|---|
| pixel head | 7 ms | 0.24 GB |
| VAE, resident | 141 ms | 1.30 GB |
| VAE, moved CPU ↔ GPU per image | 178 ms | 1.38 GB |
| VAE, loaded from disk per image | 828 ms | 1.38 GB |

DiT sampling: 6.58 s per image. On this sampler the decode is 2–11 % of an image; the head's
gain is small here.

## Why turbo is where it shows (estimate, not measured)

The Spectrum sampler above runs 16 actual forwards × 2 (CFG) ≈ 32 block passes in 6.58 s, about
0.2 s each. Turbo is 4 steps at CFG 1, about 4 passes, ~0.8 s. Against that, the VAE swap
(0.18 s) is ~20 % of an image and the head (7 ms) under 1 %; ~1.2× images per second at equal
quality would be the pitch. To measure before it is written as a fact.

## Before anything is posted

- The head is trained on anima-base at σ 0.0577. Not yet read on the turbo student's features
  or turbo's last σ; P3 showed transfer to a merged LoRA (`anima_merged15`), not to turbo.
- Quality on the shipped Spectrum sampler (σ_last 0.10): head vs VAE 28.5 dB colour, 20.9 dB
  mono, down from 30.9 / 22.5 at σ 0.0577 (P3). P4 (σ-conditioned head) is the fix to wait for.
- Dense fine strokes (screentone, hatching) stay soft against the VAE.
- A turbo e2e run (`e2e_speed.py` with the turbo LoRA, 4 steps, CFG 1) and a sheet read decide
  whether this card gets written.

## Draft blurb

> A 25M-parameter pixel head that reads Anima's last DiT features and writes RGB directly:
> decode in ~7 ms instead of ~180 ms for the VAE, with ~1.1 GB less decode VRAM, and no VAE
> load at all. Paired with the 4-step turbo LoRA, decode drops from ~20 % of each image to
> under 1 %.

(The turbo half of the blurb is the estimate above.)
