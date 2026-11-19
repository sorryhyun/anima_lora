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

Turbo 4 steps, CFG 1, RTX 5070 Ti, 16 prompts at ~1 MP, compiled DiT blocks, one seed;
medians. The VAE column is the usual interactive path: the VAE waits on the CPU and goes to the
card for the decode.

| | turbo + VAE | turbo + pixel head |
|---|---|---|
| DiT sampling | 0.97 s | 0.97 s |
| VAE move to the card and back | 35 ms | — |
| decode | 141 ms | 7 ms |
| decode peak VRAM above the DiT | 1.38 GB | 0.24 GB |
| PSNR against the VAE decode | — | 35.4 dB |
| **total per image (DiT + decode)** | **1.15 s** | **0.98 s (1.17×)** |

The move row is the moved-VAE path (176 ms) minus a decode with the VAE already on the card
(141 ms). Loading the VAE from disk per image instead costs ~710 ms (851 ms with the decode,
total 1.82 s); keeping it on the card saves the move (total 1.11 s, 1.30 GB). Text encoding
(~0.3 s per prompt) is outside the totals.

### In ComfyUI

The same turbo setting and 16 prompts in ComfyUI 0.43.0 (dynamic VRAM, PyTorch attention), same
card, medians. Workflow: Load Diffusion Model → Anima Adapter Loader (turbo LoRA, fp32 stream) →
Anima Block Compile → KSampler (euler, simple, 4 steps, CFG 1) → VAE Decode; the head column adds
Anima Pixel Head Tap before the KSampler and uses Anima Pixel Decode in place of VAE Decode.

| | KSampler + VAE Decode | KSampler + Anima Pixel Decode |
|---|---|---|
| KSampler | 0.976 s | 0.998 s (with the tap) |
| decode | 151 ms | 7 ms |
| decode peak VRAM above the loaded models | 1.22 GB | 0.21 GB |
| PSNR against VAE Decode | — | 36.9 dB |
| **total per image** | **1.13 s** | **1.00 s (1.12×)** |

ComfyUI keeps the VAE on the card after its first decode, so the VAE column has no move cost.

## Samples

Turbo 4 steps, CFG 1, seed 42, 1024×1024, in ComfyUI (the workflow above); left VAE Decode,
right Anima Pixel Decode, decoded from the same latent. Prompt: `masterpiece, best quality,
score_7, safe. An anime girl wearing a black tank-top and denim shorts is standing outdoors.
She's holding a rectangular sign out in front of her that reads "ANIMA". She's looking at the
viewer with a smile. The background features some trees and blue sky with clouds.`; the
monochrome pair adds `monochrome, comic` after `safe`. Crops: 384×384 at native resolution,
x 192–576, y 512–896.

| | VAE Decode | pixel head |
|---|---|---|
| colour, 37.3 dB | ![colour, VAE](samples/color_vae.png) | ![colour, pixel head](samples/color_head.png) |
| crop | ![colour crop, VAE](samples/color_vae_crop.png) | ![colour crop, pixel head](samples/color_head_crop.png) |
| monochrome, comic, 31.1 dB | ![mono, VAE](samples/mono_vae.png) | ![mono, pixel head](samples/mono_head.png) |
| crop | ![mono crop, VAE](samples/mono_vae_crop.png) | ![mono crop, pixel head](samples/mono_head_crop.png) |

On the monochrome crop the VAE keeps the screentone dots on the shorts; the head renders them as
flat grey (see Known limits).

## Held-out reads

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

- ComfyUI: the [ComfyUI-Anima-PixelDecoder](https://github.com/sorryhyun/ComfyUI-Anima-PixelDecoder)
  node pack (ComfyUI Manager: "Anima Pixel Decoder"). Put this file in `models/pixel_heads/`,
  add **Anima Pixel Head Tap** on the MODEL before the KSampler (stock or Spectrum) and decode
  the sampler's LATENT with **Anima Pixel Decode** instead of VAE Decode. Run the DiT on the
  fp32 stream (Anima Adapter Loader, `stream = fp32`).
- [anima_lora](https://github.com/sorryhyun/anima_lora): `inference.py --decoder pixel_head`
  (`make test PIXEL=1`, `make test-turbo PIXEL=1`); the module, loader and feature tap are in
  `library/anima/pixel_head.py`.

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
