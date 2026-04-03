tested with 5060ti 16GB
lora r=32, lr=5e-5, bsz=2, epochs=2(step=182), seed=42
validation loss was measured with fixed seed, timestep sigma = {0.05, 0.1, 0.2, 0.35}
gradient_checkpointing=true
unsloth_offload_checkpointing=true
latent and text emb were cached.

---

fa2 (plain)

peak vram : 7.0 GB
total time : 14:51
second epoch (after compilation) : 7:26
train loss : 0.092
val loss : 0.212

---

fa2 compile (eager fallback)

peak vram : 7.7 GB
total time : 15:10
second epoch (after fallback) : 7:26
train loss : 0.089
val loss : 0.211

---

fa2 compile by static token sized latent

peak vram : 6.2 GB
total time : 11:07
second epoch (after compilation) : 5:01
train loss : 0.086
val loss : 0.193

---

fa4 compile by static token sized latent

peak vram : 6.3 GB
total time : 11:01
second epoch (after compilation) : 5:17
train loss : 0.089
val loss : 0.204

---

*everything below was run under fa4 compile by static token sized latent*

+ lora_fp32_accumulation

peak vram : 6.4 GB
total time : 10:57
second epoch (after compilation) : 5:15
train loss : 0.089
val loss : 0.196

+ dora + lora_fp32_accumulation

peak vram : 6.4 GB
total time : 12:04
second epoch (after compilation) : 5:25
train loss : 0.092
val loss : 0.204

+ tlora + lora_fp32_accumulation

peak vram : 6.9 GB
total time : 12:57
second epoch (after compilation) : 5:44
train loss : 0.093
val loss : 0.21