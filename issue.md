# Issue: check each released FA2 sm_120 wheel on a real training step

Each sorryhyun/flash-attention-fp16 release was benched per call, and on a training step only for the
fp16acc backward. Measured 2026-11-04 on the precast release, MOSS `configs/moss/lora.toml` on
`hews_layout`, `project/finished/int8_train/src/phase_timer_moss.py`, 40 steps:

| wheel / flags | step | peak |
|---|---|---|
| mjun0812 stock | 1107 ms | 12.85 GB |
| precast, defaults (PV forward on) | 1094–1102 ms | 13.01 GB |
| precast, `FLASH_ATTN_FP16ACC_FWD=0` | 1048 ms | 12.85 GB |
| fp16acc release, 2026-11-02 | 1013–1035 ms | 12.88 GB |

With the PV forward on, the training backward is about 55 ms slower and the gain over stock is gone.
The suspect is precast's saved fp16 Q / K / V under `activation_memory_budget` 0.3. It has not been
separated from the PV forward itself. Generation is unaffected (1024² 10.2 s).

To do: run the same step on each release (fp16acc, fwdpv, precast), for MOSS and Anima, PV forward on
and off. Then decide whether training defaults to `FLASH_ATTN_FP16ACC_FWD=0`, or precast changes.
