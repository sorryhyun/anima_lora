# CUDA 13.2 setup

## How to install

1. Install CUDA 13.2 (Linux only, `nvidia-driver-595` **open** variant needed).
2. Install FA2 and build. Python 3.13 requires a custom wheel; [prebuilt here](https://github.com/sorryhyun/flash-attention-sm120-fix/releases/download/fa2cuda132/flash_attn-2.8.3-cp313-cp313-linux_x86_64.whl).

## How much faster?

Roughly 10%. Benchmark results on a single RTX 5060 Ti 16 GB, fixed seed, bsz=1, 2 epochs / 446 steps, fully compiled (4096-token static pad), r=16, lr=5e-5:

- `make lora`: 1.15 s/step
- `make lora-fast`: 0.97 s/step
- `make lora-gui GUI_PRESETS=tlora`: 1.30 s/step

Environment: CUDA 13.2, torch nightly (2.12), FA2 without gradient checkpointing.

### FA2, no gradient checkpointing

| Variant | Peak VRAM | Total | 2nd Epoch | sec/step | Train Loss | Val Loss |
|---|---|---|---|---|---|---|
| baseline | 14.1 GB | 9:07 | 4:26 | 1.19 s | 0.087 | 0.225 |
| + RoPE cache, fused qkv | 14.1 GB | 8:54 | 4:23 | 1.18 s | 0.086 | 0.223 |

### FA2, no checkpoint, no block swap (skipped 4 LoRA layers)

| Peak VRAM | Total | 2nd Epoch | sec/step | Train Loss | Val Loss |
|---|---|---|---|---|---|
| 14.1 GB | 7:48 | 3:52 | 1.05 s | 0.084 | 0.241 |

### FA4, no gradient checkpointing

| Peak VRAM | Total | 2nd Epoch | sec/step | Train Loss | Val Loss |
|---|---|---|---|---|---|
| 15.0 GB | 9:41 | 4:36 | 1.24 s | 0.088 | 0.216 |

### FA4, no checkpoint, no block swap (skipped 4 LoRA layers)

| Peak VRAM | Total | 2nd Epoch | sec/step | Train Loss | Val Loss |
|---|---|---|---|---|---|
| 14.9 GB | 9:07 | 4:01 | 1.12 s | 0.087 | 0.209 |

FA4 was subsequently dropped from the default training path; see [fa4.md](fa4.md) for why.
