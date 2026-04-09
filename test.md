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
| + RoPE cache, fused qkv | 14.1 GB | 8:54 | 4:23 | 1.18s | 0.086 | 0.223 |


fa4 without checkpoint

| Peak VRAM | Total Time | 2nd Epoch | sec/step | Train Loss | Val Loss |
|---|---|---|---|---|---|---|
| 15.0 GB | 9:41 | 4:36 | 1.24s | 0.088 | 0.216 |

fa4 without checkpoint, without blocks to swap (skipped 4 lora layers)

| Peak VRAM | Total Time | 2nd Epoch | sec/step | Train Loss | Val Loss |
|---|---|---|---|---|---|---|
| 14.9 GB | 9:07 | 4:01 | 1.12s | 0.087 | 0.209 |

