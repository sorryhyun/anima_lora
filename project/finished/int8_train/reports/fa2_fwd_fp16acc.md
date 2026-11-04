# FA2 forward with fp16 accumulation: built and run (2026-11-03)

The FA2 port of the parked arms in [moss_fwd16_triton.md](moss_fwd16_triton.md), the same arms as
Nunchaku's `nunchaku-fp16` ([nunchaku.md](nunchaku.md)). Patch: `patches/fwd_fp16acc.patch` in
`~/anima/flash-attention-fp16`, after its `sm120` + `fp16acc` + `hdim64` patches. The earlier unbuilt version used `fp16acc_fwd_mode()` and
`run_mha_fwd_fp16acc` in `mha_fwd` before declaring them, so `flash_api.cpp` could not compile; they now
sit above `run_mha_fwd`. Build: `output/int8_train/fa2_build_fwd16/` (unpacked) and the wheel
`flash_attn-2.8.3+sm120.hdim64128.fp16acc.fwd`, about 9 min at MAX_JOBS=6.

`FLASH_ATTN_FP16ACC_FWD` is read on every call: `pv` (the default when unset) puts PV in fp16
accumulation inside each 32-key tile, `0` gives the bf16 forward, and `all` also accumulates QKᵀ in fp16,
the backward's S included. Calls with fewer than 1024 keys keep the bf16 forward (below).
The tables in this report come from the first build, where unset meant bf16 and no key gate existed.
The arm applies to bf16, head dim 128, non-causal, no dropout / window / softcap / alibi / GQA. Q, K and
V are cast to fp16 in `mha_fwd`, O comes back in bf16.

## Kernel

[`src/fa2_fwd_fp16acc_bench.py`](../src/fa2_fwd_fp16acc_bench.py), job `20261103-193705-*`, raw:
[`fa2_fwd_fp16acc.json`](../assets/fa2_fwd_fp16acc.json). The backward is the shipped fp16acc in every arm.

Error against fp32 SDPA on the Anima capture (48 calls), median, then the ratio over stock as median
[max]:

| | O | dQ | dK | dV |
|---|---|---|---|---|
| stock, self | 0.0017 | 0.0030 | 0.0020 | 0.0017 |
| `pv` / stock, self | 0.99 [1.00] | 0.99 [1.23] | 1.00 [1.04] | 1.00 [1.00] |
| `all` / stock, self | 1.10 [1.44] | 1.04 [1.43] | **1.54 [2.04]** | **1.72 [2.79]** |
| `pv` / stock, cross | 0.96 [1.01] | 1.00 [1.00] | 1.00 [1.06] | 1.00 [1.00] |
| `all` / stock, cross | 1.09 [1.22] | 1.05 [1.17] | 1.04 [1.24] | 1.04 [1.28] |

- `pv` costs nothing measurable. P goes into the MMA as fp16, not bf16, which may explain its O sitting
  slightly under stock.
- `all`'s dK and dV cost comes from the backward's fp16 S, as [attn_bwd_fp16acc.md](attn_bwd_fp16acc.md)
  found. Generation never runs that path.
- No NaN or infinity in any tensor.

ms per call, median:

| shape (q / k) | stock fwd | `pv` fwd | `all` fwd | stock fwd+bwd | `pv` | `all` |
|---|---|---|---|---|---|---|
| MOSS self `[1, 4608, 24, 128]`, random | 2.801 | 2.436 (1.15×) | 2.073 (1.35×) | 8.731 | 8.456 (1.03×) | 8.126 (1.07×) |
| Anima self `[1, 4131, 16, 128]` | 1.550 | 1.327 (1.17×) | 1.120 (1.38×) | 4.643 | 4.404 (1.05×) | 3.951 (1.18×) |
| Anima cross `4131 / 512` | 0.194 | 0.217 (0.89×) | 0.187 (1.04×) | 0.841 | 0.856 (0.98×) | 0.778 (1.08×) |

The forward gains match the Triton forecast (1.21× / 1.26× there, before the casts). On the short
cross call the three casts cost more than `pv` saves. `pv` against bf16 by key length, random input,
forward ms gain:

| keys | 256 | 512 | 768 | 1024 | 1536 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|
| q 4131, 16 heads | 0.68× | 0.89× | 0.98× | 1.03× | 1.09× | 1.12× | 1.18× |
| q 4608, 24 heads | 0.73× | 0.90× | 0.98× | 1.04× | 1.10× | 1.11× | 1.14× |

So the patch takes the bf16 forward under 1024 keys.

## MOSS generation

`generate.py`, int8 base, 1024², 50 steps, CFG 3, three prompts from `ab_prompts.txt`, seed 7, one
process per arm (`output/tests/fa2_fwd16/{stock,pv,all}`). The first image includes warmup.

| arm | 000 | 001 | 002 |
|---|---|---|---|
| stock | 15.8 s | 10.6 s | 10.9 s |
| `pv` | 11.5 s | 10.3 s | 10.3 s |
| `all` | 11.2 s | 9.9 s | 9.9 s |

That is about −4% for `pv` and −8% for `all` per image. The three arms keep the same composition, and
neither fp16 arm shows an artifact. Both drift from stock by about the same amount in details (mean
|Δ| 5–11 / 255), so the drift is trajectory sensitivity, not something S adds.

## MOSS training

Released build, `configs/moss/lora.toml` on `channel_(caststation)_layout_sure`, lr 1e-4, EMA window
0.139, 240 steps, `caststation_fwdpv` then `caststation_fwd0` (`FLASH_ATTN_FP16ACC_FWD=0`), jobs
`20261103-195938-{4a219a,72a5e2}`. Seconds per 30-step epoch, epochs 2–8:

| run | mean [range] | peak | last-epoch loss |
|---|---|---|---|
| `pv` | 32.79 [32.63–33.13] | 13.19 GB | 0.2852 |
| bf16 forward | 33.41 [32.99–34.12] | 13.19 GB | 0.2849 |
| `caststation_te8_sure`, 14:02, same cache, bf16 forward | 32.66 [32.55–32.73] | 13.19 GB | 0.2839 |

`pv` reads 1.9% faster than its twin, about 21 ms a step, in line with the call table. But the twin's
epochs spread over 1.1 s, against 0.2 s in the 14:02 run on the same cache, and a browser was open
on the card, so the pair does not establish a training gain. Loss and peak memory do not move.

## Released

`v2.8.3-sm120-hdim64-128-fp16acc-fwdpv-cu132-torch2.14` on sorryhyun/flash-attention-fp16, Linux
x86_64 only, `pv` on by default. The `fa2-sm120` group installs it on Linux; Windows stays on the
previous tag (`WINDOWS_TAG` in `library/runtime/fa2.py`).

## Not done

- A training step without outside load (the order reversed, `0` first).
- The Windows wheel.
- `all` stays opt-in: its O error is 1.10x [1.44] stock's, and its σ 0.9 prediction move in Triton
  was never separated from spread.
