# Pixel decoder — a light head in place of the VAE

A ~23M-parameter ConvNeXt head that reads the frozen Anima DiT's last features and writes the
image directly, so sampling ends without a VAE decode. The form follows FastFrames
(arXiv:2611.23226, §3.2 and App. C). The DiT is never trained.

Three forms share one module (`library/anima/pixel_head.py`, torch only): ×1 RGB, ×2 RGB (decode
and 2× upscale in one readout) and ×1 RGBA (a transparent image). Usage (`PIXEL=1`,
`--decoder pixel_head`, the ComfyUI nodes): [../inference/pixel_head.md](../inference/pixel_head.md).
Training: `scripts/pixel_head/` (entry points `base_x1.py`, `train_x1.py`, `train_x2.py`,
`cache_alpha.py`, `train_rgba.py`, `render_alpha.py`; their modules in `train/` and `cache/`),
shipped configs in `configs/pixel_head/`.

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

The head file is the catalog row `anima_pixel_head` (`models/pixel_heads/anima_pixel_decoder_base.safetensors`,
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

head = load_pixel_head("models/pixel_heads/anima_pixel_decoder_base.safetensors", "cuda")

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
bf16; this path has not been measured.

## The heads

### ×1 (`PixelHead`)

Width 1024, three ConvNeXt blocks: two at token resolution, one after a 2× upsample. 23.3M
parameters.

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
                       │  LN → Linear 512→8·8·C (fp32)
                       ▼
             (C, 16·gh, 16·gw) in [-1, 1]      C = 3 (RGB) or 4 (RGB, 2α − 1)
```

The output layer runs in fp32 under bf16 autocast. `channels = 4` is the RGBA head: the same
trunk, an output layer with an α row per pixel. Its RGB is straight colour; inference saves it
with α = clamp((a + 1) / 2, 0, 1) as a transparent PNG.

### ×2 (`PixelHeadX2`)

The ×1 head plus a second 2× stage, so a 1024-tier render leaves the head at 2048:

```
   ×1 trunk ── … ── ConvNeXt at width 512, 8 px per position
                                         │
                       LN → Linear 512→4·256 → pixel shuffle 2×     4 px per position
                                         ▼
                              ConvNeXt ×2, width 256
                                         │
                              LN → Linear 256→8·8·3 (fp32)
                                         ▼
                              ×2 RGB (32 px / token)
```

A file stamped `scales = "2"` carries no ×1 output layer and decodes at ×2 only. Two optional
forms are read from the file: noise inputs (`x2_noise`, zero-init Gaussian channels before each
second-stage block) and an overlapping output (`out2` writes 16×16 patches at stride 8, blended
under a sin² window so no patch border carries a step).

### Files

| Catalog id | File (Hub `sorryhyun/anima-pixel-decoder`) | Form |
|---|---|---|
| `anima_pixel_head` | `x1/anima_pixel_decoder_base.safetensors` | ×1, the default |
| `anima_pixel_head_sharp` | `x1/anima_pixel_decoder_sharp.safetensors` | ×1 with a GAN finish at the base points |
| `anima_pixel_head_rgba` | `x1/anima_pixel_decoder_rgba_preview.safetensors` | ×1 writing RGB and alpha (preview), base points only |
| `anima_pixel_head_x2` | `x2/anima_pixel_head_x2.safetensors` | ×2 only, base points only |

`load_pixel_head` builds the form from the file's metadata (`width`, `depth`, `up`, `channels`,
`x2_depth`, `scales`, `sigma`). The `sigma` stamp is the trained decode range; off it the head
still decodes, with a warning.

## Training

Teacher-forced: each rollout noises a cached latent to a decode point, runs one frozen DiT forward
(fp32 stream, compiled blocks) and trains the head on that feature. No sampling loop runs.

```
 cached latent x0 ──► x_t = (1−σ)·x0 + σ·ε       σ drawn per rollout from the decode points
                          │
                          ▼
              frozen DiT forward  (turbo LoRA on at σ 0.5)
                          │
                    f, x_t patch ──► replay buffer (64 rollouts) ◄── target
                                         │
                                16 head updates per rollout
                                         │
                                         ▼
          loss = 0.5·L1 + 5·MSE + 0.05·L_grad  +  λ_dino · DINO  [+ λ_gan · L_G]
          AdamW, one cosine cycle, clip 1.0
```

- **Decode points.** The base DiT at shift-3 Euler 20 … 50 steps (σ uniform 0.0577 … 0.1364 at the
  last forward) and the 4-step turbo LoRA (σ 0.5), alternating per rollout.
- **Targets.** At the base points the original image. At σ 0.5, where x̂0's outline can leave the
  original's, the VAE's decode of the DiT's own x̂0.
- **DINO term.** DINOv3 ViT-S/16 (`facebook/dinov3-vits16-pretrain-lvd1689m`, gated on the Hub):
  4 tiles of 224 px per update at the same places on head output and target, patch tokens after
  blocks 3 / 6 / 9 / 12, L2-normalised per position, MSE averaged over the blocks.
- **GAN term** (optional). A frozen PE-Spatial-B16 with trainable token heads, a paired
  relativistic loss on 512-px crops of output and target, R1 / R2 by noise perturbation. Its
  weight is set so its gradient norm over the head is `pegan_ratio` × the regression terms'.
- **Data.** The LoRA training caches (`post_image_dataset/lora/`) plus manga pages
  (`post_image_dataset/pixel_pages`, share 0.33).
- **Compile.** The DiT blocks, the head and the DINO tower compile. The head's depthwise convs run
  as opaque custom ops (`library/anima/pixel_head_compile.py`), so one graph covers every image
  size.

The stages, each a daemon job (`make daemon-run ARGS="--queue --stall-timeout 900 <script>
--config <file>"`):

| Stage | Script, config | Init | What changes |
|---|---|---|---|
| ×1 base | `base_x1.py`, `configs/pixel_head/base_x1.toml` | a ×1 head | regression at both points, DINO (331.6) at the base points, 12,000 rollouts, LR 5e-5 |
| ×1 GAN finish | `train_x1.py`, `configs/pixel_head/x1.toml` | the ×1 base | + the GAN term at ratio 2, 2,000 rollouts |
| ×2 | `train_x2.py`, `configs/pixel_head/x2.toml` | a ×1 head + a fresh second stage | target the real 2× image (`x2flex` set), base points only, one 24-token window (+3 margin) per rollout; + DINO (245.2), a tone term (L1 on 32-px block means), the GAN term from rollout 1,000; 4,000 rollouts, LR 2e-4 |
| RGBA | `cache_alpha.py`, then `train_rgba.py` | the ×1 base, α rows fresh | below |

### RGBA

`cache_alpha.py` lays RGBA cutouts on light backgrounds (white, a flat light colour, a light
affine field with grain) and caches the DiT's inputs: the composite's VAE latent, the caption's
TE + `llm_adapter` embed and the cutout itself. A rerun encodes only what changed.

`train_rgba.py` runs the DiT on those inputs (base points only) and trains the whole four-channel
head against the RGBA image: its own colour where α > 0, the composite where α = 0, and 2α − 1.
Loss: `lambda_rgb` · rec(RGB) + rec(α) + DINO on the RGB + DINO on α read as three channels.
`render_alpha.py` renders prompts through a trained file and saves the transparent PNGs.

## Limits

- A head is tied to the DiT, and to the adapter, whose features it reads. The turbo point was
  trained on the Hub turbo repo's `old/` LoRA; v3 and v4 read about 3 dB lower.
- Tiled diffusion is refused, since it makes one forward per tile.
- The int8 DiT and other LoRAs are untested.
- The RGBA head is trained on light simple backgrounds only.
