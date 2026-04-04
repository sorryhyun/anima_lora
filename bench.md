| Configuration | Peak VRAM | Total Time | 2nd Epoch | Train Loss | Val Loss |
|---|---|---|---|---|---|
| FA2 + grad ckpt | 7.0 GB | 14:51 | 7:26 | 0.092 | 0.212 |
| FA2 + compile + grad ckpt | 6.2 GB | 11:07 | 5:01 | 0.086 | 0.193 |
| Flex + compile - grad ckpt | 15.2 GB | 8:19 | 4:09 | 0.090 | 0.201 |
| FA2 + compile - grad ckpt | 15.2 GB | 7:07 | 3:30 | 0.088 | 0.206 |
| FA4 + compile - grad ckpt | 15.2 GB | 7:28 | 3:44 | 0.091 | 0.199 |
| + fp32 lora + cross trim | 15.2 GB | 7:29 | 3:39 | 0.093 | 0.203 |