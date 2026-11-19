---
base_model:
- circlestone-labs/Anima
license: other
license_name: circlestone-labs-non-commercial-license
license_link: LICENSE.md
tags:
- anima
- decoder
---

# Anima pixel head

A 25M-parameter decoder that reads the last-layer features of the
[Anima](https://huggingface.co/circlestone-labs/Anima) DiT at the final sampling step and
writes RGB pixels directly, in place of the VAE decode. The head and its training follow
FastFrames' pixel head (a ConvNeXt readout on the diffusion transformer's features).

| | |
|---|---|
| File | `anima_pixel_head.safetensors` (fp32, 95 MB; shape in the file's metadata) |
| Input | the input of the DiT's `final_layer.linear` (2048-dim per token, after the adaLN-modulated LayerNorm), CFG-combined as f_u + g·(f_c − f_u), and x_t at that step as 2×2 latent patches (64 values per token) |
| Output | a 16×16 RGB patch per token, in [-1, 1] |
| Architecture | linear projection of the feature + linear embedding of the x_t patch, 2 ConvNeXt blocks at token resolution (width 1024), one 2× pixel-shuffle upsample, 1 ConvNeXt block at width 512, linear to 8×8×3 per upsampled position |
| DiT | `anima-base-v1.0` (28 blocks), fp32 residual stream |
| Target | VAE(x̂0) of the same forward, x̂0 = x_t − σ·v |

The file also holds an unused `linear.*` tensor pair (a skip that is off in this head,
`skip = 0` in the metadata).

## Where it was trained

Read the head only where it was trained:

- **Base DiT**: Euler, shift 3, 20 … 50 steps (last-step σ 0.058 … 0.136), read at CFG 4,
  Spectrum included (its last step is an actual forward).
- **Turbo**: the 4-step turbo LoRA of
  [anima-turbo-4step](https://huggingface.co/sorryhyun/anima-turbo-4step), Euler 4 steps,
  shift 3, CFG 1 (last-step σ 0.5). The turbo point was trained with
  `old/anima_turbo_4step.safetensors` from that repo; on v3 and v4 the head reads about
  3 dB lower against the VAE (31.5 / 31.4 dB vs 34.6 dB, 32 generated images).

Outside those settings the head still decodes; quality there is unmeasured.

## Measured

RTX 5070 Ti, 16 prompts at ~1 MP, compiled DiT blocks, one seed; medians. PSNR is head vs the
VAE decode of the same final latent. "VAE, moved" is the usual interactive path: the VAE waits on
the CPU and goes to the card for the decode.

| | turbo, 4 steps, CFG 1 | base, Euler 28 + Spectrum, CFG 4 |
|---|---|---|
| DiT sampling | 0.97 s | 6.74 s |
| decode, pixel head | 7 ms | 14 ms |
| decode, VAE resident | 141 ms | 141 ms |
| decode, VAE moved CPU↔GPU | 176 ms | 179 ms |
| decode, VAE loaded from disk | 851 ms | 796 ms |
| decode peak VRAM above the DiT, head / VAE | 0.24 / 1.30 – 1.38 GB | 0.24 / 1.30 – 1.38 GB |
| head vs VAE | 35.4 dB | 33.0 dB |
| **total per image (DiT + decode), head vs VAE moved** | **0.98 s vs 1.15 s (1.17×)** | **6.75 s vs 6.91 s (1.02×)** |

Text encoding (~0.3 s per prompt) is outside the totals.

Held-out reads during training (dB, colour / mono):

| decode point | colour | mono |
|---|---|---|
| turbo 4 steps | 35.90 | 29.01 |
| base 20 steps | 33.71 | 26.58 |
| base 28 steps | 33.83 | 26.38 |
| base 50 steps | 33.49 | 25.48 |
| base 28 + Spectrum | 34.37 | 26.00 |

## Known limits

- Screentone dots are smoothed into soft grey shading; textured regions (screentone,
  dense hatching) come out about 1 – 2.5 levels darker than the VAE in linear light.
  The lowest per-image PSNRs are manga pages and sketches (turbo 28.7 – 30.9 dB).
- Speed lines and footnote-size glyphs are slightly soft under zoom.
- One trained head, one seed; no run spread.

## Use

The loader, the feature tap and the decode call live in
[anima_lora](https://github.com/sorryhyun/anima_lora) (`bench/pixel_readout/`; moving to
`library/anima/pixel_head.py`). A ComfyUI node is planned.

## License

The weights in this repository are a **Derivative** of
[Anima](https://huggingface.co/circlestone-labs/Anima) and are distributed under the
**[CircleStone Labs Non-Commercial License](LICENSE.md)** (see [NOTICE.md](NOTICE.md)). Any
rights to use them are granted directly by CircleStone Labs LLC under that license.

- The model/weights may only be used for non-commercial purposes. Generated images (Outputs)
  may be used commercially.
- Anima is itself a Derivative Model of NVIDIA Cosmos-Predict2-2B-Text2Image and is subject
  to the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
  insofar as it applies to Derivative Models.
- This is not an official CircleStone Labs product and is not endorsed by CircleStone Labs.

Built on NVIDIA Cosmos.
