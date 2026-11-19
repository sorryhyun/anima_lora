# ComfyUI node and the turbo end-to-end cost in ComfyUI (2026-11-19)

## Facts

Node pack `~/ComfyUI-Anima-PixelDecoder` (local git, no remote yet; symlinked as
`../comfy/custom_nodes/comfyui-anima-pixel-decoder`), version 0.1.0:

- `AnimaPixelHeadTap` (MODEL → MODEL): a `CALC_COND_BATCH` wrapper opens a record per sampler step
  (σ); a `DIFFUSION_MODEL` wrapper puts a pre-hook on the live `final_layer.linear` for the one
  call and keeps its input, x and `cond_or_uncond`; a post-CFG function records the CFG scale and
  whether a `sampler_cfg_function` ran. The last step with an actual forward is kept; a last step
  with no forward (a Spectrum cached step) is flagged.
- `AnimaPixelDecode` (MODEL, LATENT, head file → IMAGE): splits each forward by
  `cond_or_uncond`, combines f_u + g·(f_c − f_u), runs the head per image (bf16 autocast, as
  `PixelTap.decode`). Warns on a last-step σ outside the head's `sigma` stamp, a cached tail and a
  `sampler_cfg_function`; refuses more than one positive or negative at the last step and a
  latent that does not match the tapped step.
- `_vendor/pixel_head.py` = `library/anima/pixel_head.py`, written by
  `scripts/release/sync_vendor.py` (new `build_pixel_vendor` target).

Bench `bench/pixel_readout/comfy_e2e.py`, run `20261119-1730-comfy_turbo_1119_1750`
(`bench/pixel_readout/results/`, local). ComfyUI 0.43.0 started through its own `main.py`
(dynamic VRAM on, PyTorch attention), RTX 5070 Ti. Workflow: UNETLoader `anima-base-v1.0` →
AnimaAdapterLoader (`models/loras/anima_turbo_4step.safetensors` = the Hub turbo repo's `old/`
file, strength 1, `stream = fp32`) → AnimaBlockCompile → [AnimaPixelHeadTap] → `common_ksampler`
(euler, simple, 4 steps, CFG 1, seed 42) → VAE Decode (`qwen_image_vae`) and Anima Pixel Decode
on the same LATENT. P3's 16 prompts at 1024², 1216×832, 832×1216; one warmup per size dropped.
Every sample ran twice, without and with the tap. n = 16, one seed, spread unknown.

| median | VAE Decode | Anima Pixel Decode |
|---|---|---|
| KSampler | 0.976 s (no tap) | 0.998 s (tap) |
| decode | 151 ms | 7 ms |
| decode peak, torch allocator above the resident models | 1.22 GB | 0.21 GB |
| PSNR against VAE Decode | — | 36.91 dB (mean 35.83) |
| total (KSampler + decode) | 1.127 s | 1.005 s |

Per-image PSNR 28.32 (manga_page) … 40.07 (chibi); manga_page 28.32, manga_action 29.72, sketch
30.68. ComfyUI kept the VAE on the card after the warmup decode.

The CLI bench on the same prompts (`reports/release_2026_11_19.md`): DiT 0.97 s, VAE 141 ms
resident / 176 ms moved, head 7 ms, 35.40 dB; totals 1.15 s (moved VAE) vs 0.98 s.

A bf16-stream arm was dropped before it ran (the user: "bf16 스트림은 없어도돼").

Published 2026-11-19: GitHub `sorryhyun/ComfyUI-Anima-PixelDecoder` (public), ComfyUI registry
`comfyui-anima-pixel-decoder` 0.1.0. The weights repo is still private.

Card samples (the user asked for the ANIMA-sign image in colour and with `monochrome, comic`,
VAE and head, full size and crops): runs `sign_color_1119_1810` and `sign_mono_1119_1810`
(`comfy_e2e.py --only anima_sign [--tags 'monochrome, comic']`, same settings as above, seed 42).
Head vs VAE: colour 37.35 dB, mono 31.14 dB. Crops 384×384 native at x 192–576, y 512–896.
Files: `bench/pixel_readout/results/card_samples_1119/` (local), uploaded to the Hub repo's
`samples/`. Claude's read of the crops (labels visible): colour level; on mono the VAE keeps the
screentone dots on the shorts and the head renders flat grey.

## Interpretation

The node reproduces the CLI decode: the PSNR against the VAE sits at the CLI's level and the
per-image floor is again on manga pages and sketches. The VAE decode in ComfyUI costs about what
the resident VAE costs in the CLI, so the gain per turbo image is ~0.12 s (1.12×), a little under
the CLI's moved-VAE comparison. The tap adds ~20 ms to a 4-step sample; the hooks and stored
references should cost far less than that, so the cause is open (not profiled).
