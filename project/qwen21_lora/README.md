# qwen21_lora — Qwen-Image-2.1 on a 16 GB card

`blockswap.swap_schedule` — 2S block moves per forward (S to make room, S to restore
during the tail); forward path, ring past S > N/2. `accel.compile_blocks` —
`torch.compile` per `block.forward`, decode shape, hooks stay eager outside.
Decode attention is already FlashAttention-2 (`pytorch_flash::flash_fwd_kernel`, 3.06 ms
× 32/step, 5.5 % of CUDA time, 92 TFLOPS) — SDPA picks it once the mask is None, so
`accel.set_attention_backend flash` is ±0; the masked prefill segments are the
attention left on a slow path. 1024²/20 steps s/step: swap 4 0.829, +compile 0.769;
swap 0 0.784, +compile 0.708.

## LoRA training (2026-09-21)

`lora.LoRANetwork` — rank 16 on the 7 linears per block (224, 41.9M params bf16), held
**outside** `transformer_blocks`: peft nests `lora_A`/`lora_B` inside the block and
`ModelOffloader` moves every `.weight` under it, so trainable weights would page to CPU
while their `.grad` stayed on the card. Training half-swaps — `swap_schedule(restore=False)`
queues S forward moves, the offloader's backward hooks walk the tail back. Activation
checkpointing's recompute re-fires the swap hooks; `blockswap.checkpoint_context_fn` makes
them inert. `cache_dataset.py` precaches both encoders at native aspect / ~1 MP and natural
text length — padding either would move the image block's rope position.
**Swap count is not the lever at 1024²**: 14 → 5 swapped left the step at 4.0 s, 100 % util,
compute-bound — the opposite of 512²/1088 tokens (PCIe-bound at 1.18 s, compile ±0). Size
the swap against `mem_get_info` free, not `total - max_allocated`: the allocator's reserve
(~0.7 GB) and the desktop (~0.5 GB) live in that gap, and sizing on allocated bytes
recommends a swap that OOMs. 30 images × 8 epochs at swap 5: peak 13.96 GB, 118 s/epoch.
