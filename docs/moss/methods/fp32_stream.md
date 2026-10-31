# fp32 residual stream

`--stream fp32` runs everything around the block GEMMs in fp32: the residual stream `x`,
modulation, gates, norms, embedders and head. The GEMMs themselves, FA2 and the LoRA stay as
they were. On the int8 base it is what brings prediction error below bf16.

**Status:** on by default in training (`TrainRequest.stream`), off by default in generation
(`GenerateRequest.stream`). `configs/moss/lora_bf16.toml` and the i2i configs run bf16.
Closer to fp32 has not been shown to look better. Measurements and renders:
[`findings/bf16_residual_stream.md`](../findings/bf16_residual_stream.md).

## Why

MOSS runs bf16 end to end and nothing runs under autocast, so each block's
`x = x + gate·proj(...)` rounds to an 8-bit mantissa, twice per block over 42 blocks. That
rounding is most of bf16's error, more than the GEMMs. Turning off cuBLAS's reduced-precision
reduction does not move the prediction.

The precision goes where it is cheap. The expensive kernels are the GEMMs and attention, and
those stay narrow; the fp32 work is elementwise adds, norms and modulation.

## How it works

`loader.stream_fp32_` casts every parameter except the 252 block linears' weights to fp32,
before `compile_blocks`. Nothing else changes: the block forward casts to the linears' dtype
on the way in (`y.to(self.proj["q"].weight.dtype)`), and their bf16 output promotes back to
fp32 at the add. An int8 base keeps an empty bf16 `weight` buffer for that read.

```bash
# training: already the default
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --cache post_image_dataset/moss/<set>"

# generation: opt in
make daemon-run ARGS="scripts/moss/generate.py --lora <path> --stream fp32 …"
```

`--stream` in training is stamped in the LoRA as `stream`, as a record: `generate` does not read
it, and a LoRA loads under either stream. The ComfyUI MOSS Toolkit loader has the same knob.

## Accuracy

Prediction error, `‖v − v_fp32‖ / ‖v_fp32‖` in %, on held-out t2i images:

| σ | bf16 | bf16, fp32 stream | int8 GPTQ | int8 GPTQ, fp32 stream |
|---|---|---|---|---|
| 0.1 | 4.81 | 1.64 | 5.93 | **3.85** |
| 0.5 | 1.93 | 0.81 | 2.43 | **1.71** |
| 0.9 | 2.08 | 0.59 | 2.30 | **1.34** |

- Around bf16 GEMMs the stream cuts the error 2.4–3.5x, and the loss matches fp32 to the
  fourth decimal.
- Around int8 GPTQ it lands below today's bf16 at every σ tested, in t2i and in i2i
  ([table](../optimization/int8.md#accuracy)).
- LoRA gradients improve only in the late blocks: blocks 28–41 move 1.5–3x closer to fp32 around
  bf16 GEMMs. Blocks 0–27 barely move, since they still pass back through every later block's
  dX GEMMs and the FA2 backward.

## Cost

| | bf16 stream | fp32 stream |
|---|---|---|
| generate, bf16 GEMMs, 1024² | 21.6 s, 10.7 GiB | 21.9 s, 12.3 GiB |
| generate, int8 GPTQ, 1024² | 11.3 s, 6.3 GiB | 15.1 s, 7.8 GiB |
| generate, int8 GPTQ, i2i 1216×864 | 27.7 s, 8.6 GiB | 27.7 s, 11.2 GiB |
| train, bf16 GEMMs (`ama_mitsuki`, 848 steps) | 1.80 s/step, 13.73 GB reserved | 1.85 s/step, 13.98 GB reserved |

One run per arm. The int8 generate rows sit inside that arm's run-to-run noise on time (11–15
s). The parameters themselves take about 0.6 GB. The training run needed swap 38 / budget 0.45
where bf16 ran 36 / 0.5, after a first run ran 72 MB short on the largest sample.

**i2i is where it costs the most**, because the reference rows ride the fp32 stream too. On
`swap_text`'s largest sample (8.9k joint tokens) the bf16 GEMM plan OOMs at swap 38 with budget
0.17, 0.15 and 0.10. It fits at swap 36 / budget 0.05 with 1.34 GB free, 4.30 s/step against 4.18. Generation
with a full-size reference and bf16 GEMMs needs `--blocks_to_swap 10`; on the int8 base it fits
resident.

## What it changes

- **A trained LoRA, more than a render.** With the same request and seed, changing the training
  stream moves a render about 3x more than changing the inference stream (14.6 against 4.3/255
  on Base). Matching the two shows no extra effect.
- **Renders, in detail.** On Base accessories, eyes and small marks change and the composition
  holds; on SFT the composition can change. Across 24 cells neither stream wins consistently.
- **Loss, slightly.** 0.0002–0.0004 lower from step 1 on, before any update, so this is the
  forward's own error and not better learning.

Two leads, neither a result: an fp32-trained LoRA drew hands better in 4 of 8 cropped cells,
worse in 1 (one run per arm, not blind), and on `swap_text` it tied bf16 within checkpoint noise.
If pretraining ran a bf16 stream, the network learned with that rounding, so fp32 is not a
target in itself.

## Trying it on another model

The arm is a parameter cast, so it costs little to test on any residual network run in bf16
without autocast. Cast everything but the block GEMMs, then score `‖v − v_fp32‖` against a true
fp32 forward with TF32 off. If the error drops by about this much, the stream was the main
source. Attention and the dX GEMMs are what it leaves behind.
