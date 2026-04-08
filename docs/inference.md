# Inference Reference

## Basic usage

```bash
python inference.py \
    --dit ../models/diffusion_models/anima-preview3-base.safetensors \
    --text_encoder ../models/text_encoders/qwen_3_06b_base.safetensors \
    --vae ../models/vae/qwen_image_vae.safetensors \
    --lora_weight ../output/anima_lora.safetensors \
    --prompt "your prompt" \
    --image_size 1024 1024 \
    --infer_steps 50 \
    --guidance_scale 3.5 \
    --save_path ../output/images
```

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--lora_weight` | — | LoRA weight path(s), space-separated for multiple |
| `--lora_multiplier` | 1.0 | LoRA strength multiplier(s) |
| `--infer_steps` | 50 | Denoising steps |
| `--guidance_scale` | 3.5 | CFG scale |
| `--flow_shift` | 5.0 | Flow-matching schedule shift |
| `--sampler` | euler | euler (deterministic ODE) or er_sde (stochastic) |
| `--from_file` | — | Batch prompts from text file |
| `--interactive` | false | Interactive prompt mode |
| `--fp8` | false | FP8 quantization for DiT |
| `--compile` | false | torch.compile speedup |

## P-GRAFT inference

Loads LoRA as dynamic hooks instead of a static merge, allowing mid-denoising cutoff:

```bash
python inference.py ... \
    --pgraft \
    --lora_cutoff_step 37    # LoRA active for steps 0–36, disabled 37+
```

## Prompt file format

```
a girl standing in a field --w 1024 --h 1024 --s 50 --g 3.5
another prompt --seed 42 --flow_shift 4.0
```

## LoRA Format Conversion

Convert between anima and ComfyUI key formats:

```bash
python scripts/convert_lora_to_comfy.py input.safetensors output.safetensors          # anima → ComfyUI
python scripts/convert_lora_to_comfy.py --reverse input.safetensors output.safetensors  # ComfyUI → anima
```
