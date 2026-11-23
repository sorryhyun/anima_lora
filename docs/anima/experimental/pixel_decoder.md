# Pixel decoder — a light head in place of the VAE

A 25M-parameter ConvNeXt head that reads the frozen Anima DiT's last features and writes RGB
directly, so sampling ends without a VAE decode. The form follows FastFrames (arXiv:2611.23226,
§3.2 and App. C). The DiT is never trained; the head is trained against the VAE.

This page covers the structure, the training and where the line stands. Usage (`PIXEL=1`,
`--decoder pixel_head`, the ComfyUI nodes) is in [../inference/pixel_head.md](../inference/pixel_head.md).
Records: `project/pixel_decoder/`; training: `scripts/pixel_head/` (entry points `base_x1.py`,
`train_x1.py`, `train_x2.py`, `train_rgba.py`, `cache_alpha.py`, `render_alpha.py`; their
modules in `train/` and `cache/`); evals and earlier arms: `bench/pixel_readout/`; the
module: `library/anima/pixel_head.py`.

## Where it reads

At the last sampling step the DiT runs its blocks as usual. The head reads the tensor that goes
into `final_layer.linear` (after the final LayerNorm and adaLN modulation), together with the
x_t latent of that step.

```
                         last sampling step (σ_last)
                                     │
   x_t (1,16,1,h,w) ──────────►  DiT blocks  ──► final_layer ─┬─ LayerNorm + adaLN(σ)
          │                                                   │
          │                                     ┌─────────────┴──────────┐
          │                              tap ◄──┤ f  (gh, gw, 2048)      │
          │                                     └─────────────┬──────────┘
          │                                                   ▼
          │                                          final_layer.linear ──► v ──► x̂0 = x_t − σ·v
          │                                                                          │
          │                                                       VAE path:  VAE.decode(x̂0) ──► RGB
          │                                                                  (141 ms resident, ~1 MP)
          │
          └── 2×2 latent patch (gh, gw, 64) ──┐
                                              ▼
                              head path:  PixelHead(f, x_t patch) ──► RGB
                                          (~7 ms, ~1 MP)
```

- One token is a 2×2 latent patch, which the VAE maps to 16×16 pixels; the head writes the same
  16×16 per token.
- The head needs x_t beside f: f predicts a velocity, and x̂0 = x_t − σ·v depends on x_t.
- Under CFG the tap keeps the cond and uncond features of the last step and combines them as
  f_u + g·(f_c − f_u). The tap sits before a linear map, so this combined feature maps exactly to
  v_cfg.
- `PixelTap` installs two forward pre-hooks outside the compiled blocks: one on the DiT
  (records σ and x_t per forward) and one on `final_layer.linear` (records f). A new σ clears
  the records, so after sampling they hold the last step.

## In code

The head file is the catalog row `anima_pixel_head` (`models/vae/anima_pixel_decoder_base.safetensors`,
Hub `sorryhyun/anima-pixel-decoder`, `x1/`). The module lives in `library/anima/pixel_head.py` (torch only).

### anima_lora

The head is one field on the request. `generate()` attaches the tap before sampling, and
`pixel_decode()` reads it afterwards. No VAE is loaded.

```python
import torch
from anima_lora import GenerationRequest, default_checkpoints, generate, get_generation_settings
from anima_lora.inference import pixel_decode, save_decoded

ckpt = default_checkpoints()
args = GenerationRequest(dit=ckpt.dit, vae=ckpt.vae, text_encoder=ckpt.text_encoder,
                         prompt="1girl, solo, smile, school uniform", save_path="out.png",
                         infer_steps=28, guidance_scale=4.0,   # shift-3 Euler, 20…50 steps
                         decoder="pixel_head",                 # pixel_head="<file>" for another head
                         ).to_args()
args.device = torch.device("cuda")

shared = {}
latent = generate(args, get_generation_settings(args), shared)
pixels = pixel_decode(args, shared["model"])          # (B, 3, H, W) in [-1, 1], CPU
save_decoded(args, latent, pixels[0])                 # PNG + generation metadata
```

At a lower level, `PixelTap(anima)` hooks a built DiT, and `tap.decode(head, guidance_scale)`
returns the pixels after any sampling loop. Both are exported from `anima_lora.models`, together
with `load_pixel_head`.

### diffusers (`ModularPipeline`, ≥ 0.39)

Two forward pre-hooks do what `PixelTap` does: one on the transformer (x_t and σ of each
forward) and one on `proj_out` (the feature). Pass `sigmas` explicitly. With the stock scheduler
the last forward lands at σ 0.009, far below the head's range. `np.linspace(1, 1/n, n)` under
the pipeline's shift 3 reproduces anima_lora's schedule, so the last forward lands at σ 3/(n+2).

```python
import numpy as np, torch
from PIL import Image
from diffusers.modular_pipelines.anima import AnimaAutoBlocks
from library.anima.pixel_head import lat_patches, load_pixel_head  # anima_lora repo

head = load_pixel_head("models/vae/anima_pixel_decoder_base.safetensors", "cuda")

blocks = AnimaAutoBlocks()
blocks.sub_blocks.pop("decode")                       # no VAE decode
pipe = blocks.init_pipeline("circlestone-labs/Anima-Base-v1.0-Diffusers")
pipe.load_components(torch_dtype=torch.bfloat16)
pipe.to("cuda")

calls = []                                            # forwards at the newest σ: cond, then uncond
def on_forward(_m, args, kw):
    t = float(kw["timestep"].flatten()[0])
    if calls and calls[-1]["t"] != t:
        calls.clear()
    calls.append({"t": t, "x": kw["hidden_states"]})
def on_proj_out(_m, inp):
    calls[-1]["f"] = inp[0]
pipe.transformer.register_forward_pre_hook(on_forward, with_kwargs=True)
pipe.transformer.proj_out.register_forward_pre_hook(on_proj_out)

n, cfg = 28, 4.0
pipe.guider.guidance_scale = cfg
pipe(prompt="1girl, solo, smile, school uniform", num_inference_steps=n,
     sigmas=np.linspace(1, 1 / n, n).tolist(), output="latents")

f = calls[0]["f"].float()                             # (1, h/2 · w/2, 2048)
if len(calls) == 2:                                   # CFG: combine like the velocity
    f = calls[1]["f"].float() + cfg * (f - calls[1]["f"].float())
x_t = calls[0]["x"][0, :, 0].float()                  # (16, h, w)
gh, gw = x_t.shape[1] // 2, x_t.shape[2] // 2
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    rgb = head(f[0].view(gh, gw, -1), lat_patches(x_t)).float().clamp(-1, 1)
Image.fromarray(((rgb + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save("out.png")
```

The head was trained on fp32-stream features (`--stream fp32`). The diffusers transformer runs in
bf16, so expect it to read somewhat lower than the numbers below; this path has not been
measured.

## The head

Released head `mixr_1119_1455`: width 1024, three ConvNeXt blocks (two at token resolution, one
after a 2× upsample), no skip, 24.9M parameters.

```
 f  (gh, gw, 2048)          x_t patch (gh, gw, 64)
        │                           │
   Linear 2048→1024           Linear 64→1024
        └───────────── + ───────────┘
                       │                         token grid, 16 px per position
                       ▼
            ┌─────────────────────┐
            │ ConvNeXt block ×2   │  dw 3×3 → LN → MLP 4× → γ·h + x
            │ width 1024          │
            └──────────┬──────────┘
                       │  LN → Linear 1024→4·512 → pixel shuffle 2×
                       ▼                         2× grid, 8 px per position
            ┌─────────────────────┐
            │ ConvNeXt block ×1   │
            │ width 512           │
            └──────────┬──────────┘
                       │  LN → Linear 512→8·8·3 (fp32)
                       ▼
             RGB (3, 16·gh, 16·gw) in [-1, 1]
```

Choices that held up:

| Choice | What it replaced | Record |
|---|---|---|
| No linear skip from f to the patch | A P1 linear readout as skip, zero-init output, layer scale 1e-6: stalled at 26.50 dB | `archive/p2_skip_2026_11_19.md` (local) |
| One 2× upsample stage before the last block | All blocks at token resolution: 32.93 → 33.64 dB test | `archive/upsample_2026_11_19.md` (local) |
| Output layer kept in fp32 under bf16 autocast | bf16 output weights cost dB (measured with the skip head) | `library/anima/pixel_head.py` |

## Training

Teacher-forced from cached latents: each rollout noises a training latent to the decode point,
runs one frozen DiT forward and one VAE decode, and trains the head on the result. No sampling
loop runs during training.

```
 cached latent x0 ──► x_t = (1−σ)·x0 + σ·ε     σ drawn per rollout (decode points, below)
                          │
                          ▼
              frozen DiT forward (fp32 stream, compiled blocks;
                                  turbo LoRA on at σ 0.5)
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
       f, x_t patch             x̂0 = x_t − σ·v ──► VAE.decode ──► target VAE(x̂0)
              │                                                       │
              └──────────────► replay buffer (64 rollouts) ◄──────────┘
                                         │
                                16 head updates per rollout
                                         │
                                         ▼
                   loss = 0.5·L1 + 5·MSE + 0.05·L_grad    (FastFrames' spatial terms)
                   AdamW, LR 2e-4, cosine, clip 1.0
```

- **Target VAE(x̂0), not the image.** The head can only know what the features hold. At σ_last
  that is x̂0, and the VAE caps how close anything gets to the original: on real manga pages the
  VAE alone is 21.5–39.8 dB from the original, VAE(x̂0) 24.3–36.3 dB from VAE(x0). The reachable
  target is VAE(x̂0).
- **Decode points.** One head covers the samplers it ships for:

  | DiT | sampler | σ at the last forward |
  |---|---|---|
  | anima-base | shift-3 Euler, 20…50 steps (Spectrum included) | uniform 0.0577…0.1364 |
  | anima-base + turbo LoRA (Hub `old/`) | 4 steps, CFG 1 | 0.5 |

  The two points alternate per rollout, with the LoRA toggled by `set_enabled`.
- **Data.** The LoRA training caches (`post_image_dataset/lora/`) plus 3,000 manga pages
  (Manga109-s and JA web manga, one fixed tag caption) at share 0.33. The pages took the manga eval from 26.64 to 28.15 dB
  (`archive/pagemix_2026_11_19.md` (local)).
- **Cost.** 900 rollouts in 11.3 min on a 5070 Ti (`replay_1119_1225`). The release head took
  3,000 rollouts on top of `pagemix_1119_1242` (2,000).

## Results

PSNR against the VAE decode of the same latent, one seed, held-out artists. Colour / mono are
generated renders; test / text are teacher-forced held-out images.

| Head | Read on | Numbers |
|---|---|---|
| P1 linear readout | teacher-forced test / text | 26.28 / 21.61 |
| P2 ConvNeXt, no skip | teacher-forced test / text | 32.93 / 27.45 |
| `pagemix_1119_1242` (+ upsample, pages, VAE(x̂0) target) | test / text / manga | 34.83 / 29.65 / 28.15 |
| `mixr_1119_1455` (release) | turbo rollouts colour / mono | 35.90 / 29.01 |
| | base 28 steps colour / mono | 33.83 / 26.38 |
| | base 20 / 50 steps colour | 33.71 / 33.49 |
| | Spectrum colour / mono | 34.37 / 26.00 |

What is left, from the user's reads: screentone dots come out as soft grey shading, textured
regions run 1–2.5 levels darker than the VAE in linear light, and speed lines and footnote-size
glyphs are slightly soft under zoom. The loss has no perceptual or local-tone term; that is the
"pixel losses on the head" topic below.

## Phases

```
 P1 linear readout ──► P2 ConvNeXt head ──► P3 generated images, LoRA transfer, manga pages
                                                │
                                                ▼
                       P4 decode points (base range + turbo), release, library/, ComfyUI node
                                                │
                                                ▼
                       P5 (open): ×2 output · pixel losses on the head · RGBA · a head at
                              every step · MOSS (losses through the head into the DiT: closed)
```

P1–P4 are complete (2026-11-19). Per-phase record: `project/pixel_decoder/archive/phases_p1_p4_2026_11_19.md` (local).

## ×2 output (P5)

The goal is to decode and upscale 2× in one readout, so a 1024-tier render leaves the head at
2048. A second 2× stage goes on top of the release head. In the preview file the ×1 output
stays, so one file serves both; the shipped ×2 file (`anima_pixel_head_x2`, `x2bgan2_1121_1340`,
2026-11-21: real 2× targets, ×1 loss off, PE-Spatial GAN) drops the ×1 output layer and decodes
at ×2 only ([../inference/pixel_head.md](../inference/pixel_head.md#×2-output)). The rest of
this section is the research record.

```
   release head (as above) ── … ── ConvNeXt ×1, width 512, 8 px per position
                                         │
                       ┌─────────────────┴──────────────────┐
                       ▼                                    ▼
          LN → Linear 512→8·8·3                LN → Linear 512→4·256 → pixel shuffle 2×
                       │                                    │      4 px per position
                       ▼                                    ▼
            ×1 RGB (16 px / token)             ConvNeXt ×2, width 256
                                                            │
                                               LN → Linear 256→8·8·3 (fp32)
                                                            ▼
                                               ×2 RGB (32 px / token)
```

Target: the released 1-step ResShift ×2 student (`sorryhyun/Distilled-ResShift-2x`) run on
VAE(x̂0). It sees one 30×30-token window per rollout, with the loss on the inner 24×24. The head
runs in 13 ms on 80×52 tokens (×1 and ×2, eager), against 3.2 s for VAE → ResShift.

The level to read against is the **chain**: the release head's ×1 output, then the teacher.
The target is a function of VAE(x̂0), which the head knows only to its ×1 error, so the
teacher's second draw (42–46 dB) is out of reach.

| Base σ 0.097, against the teacher on VAE(x̂0) | test | text | manga | LPIPS test | dark-pixel lift |
|---|---|---|---|---|---|
| bicubic ×2 of VAE(x̂0) | 29.82 | 28.25 | 27.00 | | |
| arm A, recipe loss only (`x2a_1119_1802`) | 30.70 | 27.25 | 26.49 | 0.205 | +10.1 |
| chain: head ×1 → teacher | 33.71 | 29.55 | 28.17 | 0.152 | +5.7 |

Dark-pixel lift is the mean luma difference, in 8-bit levels, over the pixels the teacher has
under 64.

Arm A came out soft: lines thicker than the chain's, dark pixels lifted, and the ×1 output down
about 1 dB. Fit probes showed that the stage can express the target on 2 windows but does not
learn it as a mapping on 16 under the recipe loss. The running arm adds VGG-LPIPS and the SR
line's tone term (`dc_loss`, L1 on 32-px block means) on the ×2 output. If its lines stay soft,
a GAN term is next. Algorithm and findings: `project/pixel_decoder/README.md`; runs: `project/pixel_decoder/reports/`.

## Limits

- A head is tied to the DiT, and to the adapter, whose features it reads. The turbo point was
  trained on the Hub turbo repo's `old/` LoRA; v3 and v4 read about 3 dB lower.
- An off-point σ still decodes, with a warning. Tiled diffusion is refused, since it makes one
  forward per tile.
- The int8 DiT and other LoRAs are untested; a merged style LoRA read level in P3.
- Every number is from one seed; no head has a seed twin yet.
