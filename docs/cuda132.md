## How can we install cuda 13.2?

1. First install cuda 13.2 (linux only, nvidia-driver-595 **open** needed)
2. install fa2 and build. I tried build in python3.13, prebuilt [wheel](https://github.com/sorryhyun/flash-attention-sm120-fix/releases/download/fa2cuda132/flash_attn-2.8.3-cp313-cp313-linux_x86_64.whl)
3. install bitsandbytes from source and build with cmake


## How faster it is?

* In some extent, like ~10%
* Achieved 1.15s/step in `make lora`, 1.30s/step in `make tlora`, 0.97s/step in `make lora-fast`

cuda 13.2, torch nightly (2.12)
bsz=1, 2epochs, 446 steps, fixed seed
all compiled (padded images to 4096)
r=16, lr=5e-5

fa2 without checkpoint

| Peak VRAM | Total Time | 2nd Epoch | sec/step | Train Loss | Val Loss |
|---|---|---|---|---|---|---|---|
| fa2 without checkpoint | 14.1 GB | 9:07 | 4:26 | 1.19s | 0.087 | 0.225 |
| + RoPE cache, fused qkv | 14.1 GB | 8:54 | 4:23 | 1.18s | 0.086 | 0.223 |


fa2 without checkpoint, without blocks to swap (skipped 4 lora layers)

| Peak VRAM | Total Time | 2nd Epoch | sec/step | Train Loss | Val Loss |
|---|---|---|---|---|---|---|
| 14.1 GB | 7:48 | 3:52 | 1.05s | 0.084 | 0.241 |


fa4 without checkpoint

| Peak VRAM | Total Time | 2nd Epoch | sec/step | Train Loss | Val Loss |
|---|---|---|---|---|---|---|
| 15.0 GB | 9:41 | 4:36 | 1.24s | 0.088 | 0.216 |

fa4 without checkpoint, without blocks to swap (skipped 4 lora layers)

| Peak VRAM | Total Time | 2nd Epoch | sec/step | Train Loss | Val Loss |
|---|---|---|---|---|---|---|
| 14.9 GB | 9:07 | 4:01 | 1.12s | 0.087 | 0.209 |

