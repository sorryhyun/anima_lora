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

fa2 compile (dynamo fallback)

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
total time : 11:13
second epoch (after compilation) : 5:19
train loss : 0.089
val loss : 0.204

---

*everything is fa4 compile by static token sized latent below:*

+ lora_fp32_accumulation

peak vram : 6.5 GB
total time : 
second epoch (after compilation) : 
train loss : 
val loss : 

+ dora


+ tlora