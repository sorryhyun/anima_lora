# qwen21_lora — Qwen-Image-2.1 on a 16 GB card

`blockswap.swap_schedule` — 2S block moves per forward (S to make room, S to restore
during the tail); forward path, ring past S > N/2. `accel.compile_blocks` —
`torch.compile` per `block.forward`, decode shape, hooks stay eager outside.
Decode attention is already FlashAttention-2 (`pytorch_flash::flash_fwd_kernel`, 3.06 ms
× 32/step, 5.5 % of CUDA time, 92 TFLOPS) — SDPA picks it once the mask is None, so
`accel.set_attention_backend flash` is ±0; the masked prefill segments are the
attention left on a slow path. 1024²/20 steps s/step: swap 4 0.829, +compile 0.769;
swap 0 0.784, +compile 0.708.
