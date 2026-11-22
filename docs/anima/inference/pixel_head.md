# Pixel head

A 25M-parameter decoder that replaces the VAE decode at the end of sampling. It reads the
DiT's last-step features (the input of `final_layer.linear`, 2048 values per token) and
x_t's 2×2 latent patch, and writes a 16×16 RGB patch per token. No VAE is loaded.

```bash
make test PIXEL=1                 # base DiT, 28-step Euler, CFG 4
make test-turbo PIXEL=1           # 4-step turbo, CFG 1
python inference.py … --decoder pixel_head [--pixel_head <file>]
```

The weight is the catalog row `anima_pixel_head` (`sorryhyun/anima-pixel-decoder`,
`models/vae/anima_pixel_head.safetensors`), fetched on first use;
`make download-model ARGS=anima-pixel-head` fetches it (and the ×2 file) ahead of time. `PIXEL_HEAD=<file>` or
`--pixel_head` points at another head (a `bench/pixel_readout/` run's `head.safetensors`).

## Where it reads

The head was trained at fixed decode points, stamped as `sigma` in the file's metadata:

| DiT | sampler | last-step σ |
|---|---|---|
| anima-base | shift-3 Euler, 20 … 50 steps (Spectrum included) | 0.058 … 0.136 |
| anima-base + turbo LoRA | 4 steps, CFG 1 | 0.5 |

At any other last-step σ it still decodes and logs a warning; no refusal. The current head
(`realr_1119_2200`, since 2026-11-19) was trained on the original image with LPIPS at the base
points and on VAE(x̂0) at the turbo point, with the turbo LoRA
`anima_turbo_4step_v4_longer.safetensors`; it reads 33.75 dB / LPIPS 0.081 against the original
on held-out test images at σ 0.0577 (`project/pixel_decoder/reports/real_target_2026_11_19.md`).
The previous head (`old/anima_pixel_head.safetensors` on the Hub, VAE(x̂0) target, turbo point
on the turbo repo's `old/` file) read turbo 35.4 dB and base 28 steps + Spectrum 33.0 dB against
the VAE decode; the numbers below are that head's.

Decode time ~7 ms per ~1 MP image against 141 ms (VAE resident), 176 ms (VAE moved to the card
and back) or ~850 ms (VAE loaded from disk). On turbo the VAE is 15 % of DiT + decode.

## ×2 output

The ×2 file `anima_pixel_head_x2` (`models/vae/anima_pixel_head_x2.safetensors`, same Hub repo,
fetched on first use) carries a second 2× stage after the head's own: one readout writes a 32×32
RGB patch per token, so a 1024-tier render leaves the head at 2048 (~12 ms on 85 × 48 tokens).

```bash
make test PIXEL=1 PIXEL_SCALE=2   # takes the catalog's ×2 file
python inference.py … --decoder pixel_head --pixel_scale 2
python inference.py … --decoder pixel_head --pixel_head <×2 file>   # scale from the file
```

The file decides the scale. With `--pixel_scale` unset the head decodes at its file's own: ×1
for `anima_pixel_head`, ×2 for `anima_pixel_head_x2` (stamped `scales = "2"`), ×1 for the old
two-scale preview file. An explicit scale the file does not hold refuses at load, in both
directions: the ×2 file has no ×1 output layer (it was trained with the ×1 loss off), and a ×1
file has no ×2 stage. With no `--pixel_head`, `--pixel_scale 2` takes the ×2 catalog row.

The ×2 file (`x2bgan2_1121_1340`) was trained against real 2× images at the base points only
(shift-3 Euler 20 … 50 steps, σ 0.058 … 0.136), with LPIPS, a tone term and a PE-Spatial GAN
term from rollout 1000 of 4,000; on the turbo it decodes with a warning. Against the real 2×
image on held-out images: 29.82 / 25.52 dB, LPIPS 0.158 / 0.145 (test 12 / text 5). Records:
`project/pixel_decoder/reports/x2_full_run_deploy_2026_11_21.md`.

## ComfyUI

The node pack `ComfyUI-Anima-PixelDecoder` (`comfyui-anima-pixel-decoder` under
`../comfy/custom_nodes/`): **Anima Pixel Head Tap** on the MODEL before the KSampler (stock or
Spectrum), **Anima Pixel Decode** (tapped MODEL, the sampler's LATENT, head file) in place of VAE
Decode; it decodes at the head file's own scale (no scale input since 0.3.1). The head file goes in `models/pixel_heads/`, or in a VAE folder under a name containing
`pixel_head`. Run the DiT on the fp32 stream (Anima Adapter Loader `stream = fp32`), as the head
was trained. Turbo in ComfyUI 0.43: 1.127 s per image with VAE Decode, 1.005 s with the head
(`bench/pixel_readout/comfy_e2e.py`, `project/pixel_decoder/reports/comfy_node_2026_11_19.md`).

## How it works

`library/anima/pixel_head.py` holds the modules (`PixelHead`, `PixelHeadX2`, `load_pixel_head`) and
`PixelTap`: a forward pre-hook on the DiT opens a record (σ, x_t) per forward, a pre-hook on
`final_layer.linear` fills its feature, and a new σ clears the records. After sampling the
records hold the last step: one forward at CFG 1, cond then uncond under CFG, combined as
f_u + g·(f_c − f_u) (the tap sits before a linear map, so this matches v_cfg). The hooks are
outside the compiled blocks. `library/inference/pixel_decoder.py` wires it into
`inference.py`: the head rides on the DiT as `anima._pixel_decoder` (a plain attribute, not a
submodule).

## Limits

- Screentone dots come out as soft grey shading; textured regions run 1 – 2.5 levels darker
  than the VAE in linear light. Speed lines and footnote-size glyphs are slightly soft under
  zoom.
- `--tiled_diffusion` is refused (one forward per tile). `--latent_path` decodes with the VAE
  (no features). `--cfgpp` / `--smc_cfg` warn: the feature is combined as plain CFG.
- A Spectrum cached last step warns: the head reads the last actual forward. The `make test`
  Spectrum flags end on an actual forward.
- On the int8 DiT (`--int8`) the head reads at the bf16 level: base 28 steps 32.32 dB against
  32.44, turbo 34.60 against 34.61 (32 renders each, one run). Other LoRAs are untested; P3 read
  a merged style LoRA at the base level.

Structure, training and the ×2 work: [../experimental/pixel_decoder.md](../experimental/pixel_decoder.md).
Training, evals and probes: `bench/pixel_readout/` (×2 training:
`scripts/pixel_head/train_x2.py`); the project record:
`project/pixel_decoder/`.
