# fp16 headroom on Anima (2026-11-05)

**Verdict: the block GEMMs fit fp16; the residual stream does not.** The stream peaks at
285k in blocks 12–20, 4.3x over fp16's 65504, so an all-fp16 forward is NaN on every image
from block 12. Kept in fp32, the stream runs fp16 GEMMs with 7x less prediction error than
bf16 GEMMs. Scaled by 1/8 or 1/16, it runs all-fp16 with no NaN. fp16 accumulation is 1.6x
bf16 on the RTX 5070 Ti and brings the error back to bf16's. The backward needs a loss
scale near 2^20. Nothing shipped; no fp16 plan exists.

Scripts: [`_archive/bench/anima_fp16/probe.py`](../../../bench/anima_fp16/probe.py) (ranges, errors,
backward), [`_archive/bench/anima_fp16/gemm.py`](../../../bench/anima_fp16/gemm.py) (GEMM time). Raw
numbers: `_archive/bench/anima_fp16/results/20261105-1220/` (`fp16_probe2.json`: 4 images with the
backward, `fp16_probe3.json`: 3 images with the scaled arms).

## Setup

`anima-base-v1.0`, 28 blocks, eager, flash attention, no LoRA, no vocab pack. Images of
3500–4608 tokens from `post_image_dataset/lora`, chosen as in `bench/anima_stream`. σ 0.1,
0.5 and 0.9, one forward each. The reference is a true fp32 forward (`fp32_reference`).

| arm | stream | block GEMMs |
|---|---|---|
| `bf16s` | fp32 | bf16 (the default, `--stream fp32`) |
| `f16s` | fp32 | fp16, fp32 accumulation |
| `f16acc` | fp32 | fp16, fp16 accumulation (`allow_fp16_accumulation`) |
| `f16all` | fp16 | fp16, fp32 accumulation |
| `f16all_s8`, `_s16` | fp16, × 1/8 or 1/16 | fp16, fp32 accumulation |
| `f16accall_s16` | fp16, × 1/16 | fp16, fp16 accumulation |

The scaled arms divide block 0's input and every self-attention, cross-attention and MLP
output by c. Every sublayer and the final layer read the stream through a LayerNorm, so this
is exact up to the norms' eps.

## Range

fp32 forward, maxima over images and σ:

| where | peak |
|---|---|
| stream, blocks 0–11 | 4k–10k |
| **stream, blocks 12–20** | **261k–285k** |
| stream, blocks 21–26 | 13k–92k |
| stream, block 27 (the final LayerNorm's input) | 45k |
| `adaln_up_cross_attn` output (modulation) | 4.6k |
| block GEMM inputs | ≤ 642 (`cross_attn.q_proj`) |
| block GEMM outputs | ≤ 747 (`mlp.layer2`, block 12) |
| weights | ≤ 123 |

- Block 12 writes the jump and block 21 takes most of it back. Which channel or token holds it
  was not traced.
- In fp16, `final_layer.linear` puts 9.4% of its nonzero weights below fp16's smallest normal,
  and `adaln_fused_down` 5–7.6%. The block linears are not among the worst eight. At most 5e-5
  of any matrix rounds to zero.

## Prediction error

`‖v − v_fp32‖ / ‖v_fp32‖`, %, mean over images:

| arm | σ 0.1 | σ 0.5 | σ 0.9 |
|---|---|---|---|
| `bf16s` | 0.55 | 0.23 | 0.24 |
| `f16s` | **0.085** | **0.031** | **0.030** |
| `f16acc` | 0.45 | 0.21 | 0.19 |
| `f16all` | NaN | NaN | NaN |
| `f16all_s8` | 0.27 | 0.13 | 0.14 |
| `f16all_s16` | 0.27 | 0.14 | 0.15 |
| `f16accall_s16` | 1.01 | 0.52 | 0.87 |

For scale: the bf16 stream is 1.76 / 0.85 / 1.15 and the int8 GPTQ base on an fp32 stream is
1.32 / 0.48 / 0.51 ([fp32_stream.md](../optimizations/fp32_stream.md)).

- `f16all`'s first non-finite tensor is `blocks.12`'s output in all 12 cells.
- Per block, `f16acc`'s error steps from 0.27–0.30% in blocks 0–11 to 0.9–1.0% in blocks
  12–20, and `f16accall_s16`'s from 0.5–0.7% to 3.4–4.0%. The block error is pooled over all
  channels, so the large channel may dominate it there; this was not separated.
- Which linear fp16 accumulation hurts most was not measured. `mlp.layer2` (K = 8192) is the
  first suspect.

## GEMM time

RTX 5070 Ti, `x @ Wᵀ`, M = 4096, median of 50 (`gemm.py`):

| shape (M×K×N) | bf16 | fp16 | fp16acc | int8 `_int_mm` |
|---|---|---|---|---|
| self qkv 4096×2048×6144 | 0.978 ms | 0.980 | 0.622 | 0.294 |
| mlp 1 4096×2048×8192 | 1.341 | 1.349 | 0.856 | 0.414 |
| mlp 2 4096×8192×2048 | 1.342 | 1.359 | 0.841 | 0.400 |
| self out 4096×2048×2048 | 0.351 | 0.352 | 0.216 | 0.113 |
| TFLOPS | ~102 | ~102 | ~162 | ~330 |

- fp16 with fp32 accumulation runs at bf16's rate on this card: no gain from the dtype alone.
- fp16 accumulation is 1.6x. The ConvRot int8 linear, quantization included, is about 2.6x
  ([gemm_microbench.md](../../../project/finished/int8_train/reports/gemm_microbench.md)), so
  fp16acc is slower than the int8 plans. `--int8 stream` already runs dX in int8, and Turbo
  ships it.
- `dY @ W` (the dX layout) was not timed.

## Backward

The gradient each block linear's output receives, `bf16s`, mean MSE loss, no loss scale,
2 images × 3 σ, checkpointed:

| module | max | min median | below fp16 normal | rounds to 0 |
|---|---|---|---|---|
| `self_attn.qkv_proj` | 4.8e-4 | 2.7e-11 | 100% | 99.99% |
| `cross_attn.q_proj` | 1.8e-4 | 5.7e-11 | 100% | 99.98% |
| `self_attn.output_proj` | 9.7e-4 | 3.1e-9 | 100% | 97% |
| `cross_attn.output_proj` | 2.1e-2 | 2.7e-9 | 100% | 95% |
| `mlp.layer1` | 7.8e-4 | 1.9e-9 | 100% | 96% |
| `mlp.layer2` | 8.1e-4 | 7.1e-10 | 100% | 84% |

The last two columns are the worst module of each kind. A loss scale of 2^20 puts the
largest gradient at 22k, under 65504, and the smallest medians near 3e-5, still subnormal.
The range is tight. A per-row scale inside a custom backward, as the int8 dX and FA2's
fp16acc backward already do with `dY` / `dO`, avoids the global scale. Neither was run.
Where the small medians sit by depth was not recorded. LayerNorm's backward dividing by the
285k stream in blocks 12–20 is a hypothesis.

## Where this leaves fp16

- **Turing-and-later GeForce, bf16 plans** (Anima's default `int8 = "off"`, MOSS
  `lora_bf16.toml`, MOSS i2i `train.toml`s, Turbo with DMAD): an fp16acc linear op on an fp32
  stream. Forward fp16 cast, fp16acc against an fp16 copy of the block weight; backward `dY`
  per-row scaled into fp16 range. Predicted: the block GEMMs were 280 of 594 ms of a bf16
  Anima step ([step_profile.md](../../../project/finished/int8_train/reports/step_profile.md),
  bf16-stream era), so 1.6x would take off about 105 ms, a 1.2x step. Error about bf16's;
  no VRAM saving.
- **V100 and other cards without bf16**: fp32 accumulation runs at the full fp16 rate there,
  so fp16acc buys nothing. What carries over is the range: an fp16 stream scaled by 1/8–1/16
  and a loss scale near 2^20 (or a per-row backward scale). FA2 needs sm_80, and the int8
  linears (`comfy_kitchen.int8_linear`) also require sm_80 here.

## Not done

- The fp16acc linear op itself, its time in a compiled step and dX's layout.
- The LoRA gradient under fp16acc (`project/finished/int8_train/src/grad_error.py`).
- A backward with any loss scale.
- Which channel and token carry the 285k, and the per-linear fp16acc error.
- MOSS: its base stream peaks at 61.4k at 1 MP
  ([heal_finetune.md](../../moss/findings/heal_finetune.md)), so the same scaling applies.
