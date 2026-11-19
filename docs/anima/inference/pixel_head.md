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
`make download-model ARGS=anima-pixel-head` fetches it ahead of time. `PIXEL_HEAD=<file>` or
`--pixel_head` points at another head (a `bench/pixel_readout/` run's `head.safetensors`).

## Where it reads

The head was trained at fixed decode points, stamped as `sigma` in the file's metadata:

| DiT | sampler | last-step σ |
|---|---|---|
| anima-base | shift-3 Euler, 20 … 50 steps (Spectrum included) | 0.058 … 0.136 |
| anima-base + turbo LoRA | 4 steps, CFG 1 | 0.5 |

At any other last-step σ it still decodes and logs a warning; no refusal. Measured against the
VAE decode of the same latent (one seed, 16 prompts, `project/pixel_decoder/reports/`): turbo
35.4 dB, base 28 steps + Spectrum 33.0 dB. The turbo point was trained on the Hub turbo repo's
`old/anima_turbo_4step.safetensors`; v3 and v4 read about 3 dB lower (31.5 / 31.4).

Decode time ~7 ms per ~1 MP image against 141 ms (VAE resident), 176 ms (VAE moved to the card
and back) or ~850 ms (VAE loaded from disk). On turbo the VAE is 15 % of DiT + decode.

## How it works

`library/anima/pixel_head.py` holds the module (`PixelHead`, `load_pixel_head`) and
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
- The int8 DiT (`--int8`) and other LoRAs are untested; P3 read a merged style LoRA at the
  base level.

Training, evals and probes: `bench/pixel_readout/`; the project record:
`project/pixel_decoder/`.
