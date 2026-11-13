# Releasing a turbo student

A student of this line is trained on the merged14 DiT (`W_base + ΔW14`), so its LoRA (`ΔT`) means
nothing on plain base by itself. `deploy.sh <turbo_ema.safetensors> <name> [lora|bake|gptq|all]`
writes the three release forms:

| Form | File | Use |
|---|---|---|
| One LoRA for base | `output/ckpt/release/<name>.safetensors` | anima-base-v1.0 + this LoRA at **1.0 only**, 4 steps, cfg 1. Rank 256 + 64 concat of the soup and the turbo, adaln summed; about 1.07 GB in bf16. bf16 rounds the α/r-scaled ups (rel err 8e-4, adaln 1.2e-3); `DTYPE=fp32` is bit-exact at 2.14 GB |
| Baked DiT | `models/diffusion_models/<name>.safetensors` | full bf16 checkpoint, no LoRA. Turbo LoRA baked onto the bf16 merged14 DiT, the DiT the student trained on |
| int8 DiT | `models/diffusion_models/<name>_int8_gptq.safetensors` | ConvRot int8 GPTQ of the baked DiT, ComfyUI `int8_tensorwise` + `convrot` layout. `inference.py --int8` picks it up beside the baked DiT |

The GPTQ calibration differs from the teacher's (28 Euler steps, caption and negative pass): σ
is drawn from the 4-step shift-3 grid and only the caption pass runs (`dit_gptq.py
--no_negative`), with 4 images per artist so the sample count stays at 368. It runs as a daemon
job (`--queue`); `lora` and `bake` run on the CPU.
