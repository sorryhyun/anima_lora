# qwen21_lora — Qwen-Image-2.1 on a 16 GB card

`blockswap.swap_schedule` — 2S block moves per forward (S to make room, S to restore
during the tail); forward-only, ring past S > N/2. `accel.compile_blocks` —
`torch.compile` per `block.forward`, decode shape only, hooks stay eager outside.
`accel.set_attention_backend` — per-processor, so the prefill keeps its bool mask.
1024²/20 steps s/step: swap 4 0.829, +compile 0.769; swap 0 0.784, +compile 0.708; flash ±0.
