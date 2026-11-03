# fp32 residual stream

`--stream fp32` runs everything around the block GEMMs in fp32: the residual stream `x`,
modulation, gates, norms, embedders and head. The GEMMs themselves, FA2 and the LoRA stay as
they were. On the int8 base it is what brings prediction error below bf16.

**Status:** on by default in training (`TrainRequest.stream`), off by default in generation
(`GenerateRequest.stream`). `configs/moss/lora_bf16.toml` and the i2i configs run bf16.
It reads better on text rendering and finger placement. Measurements and renders:
[`findings/bf16_residual_stream.md`](../findings/bf16_residual_stream.md).

## Why

MOSS runs bf16 end to end and nothing runs under autocast, so each block's
`x = x + gate·proj(...)` rounds to an 8-bit mantissa, twice per block over 42 blocks. That
rounding is most of bf16's error, more than the GEMMs. Turning off cuBLAS's reduced-precision
reduction does not move the prediction.

The expensive kernels, the GEMMs and attention, stay narrow; the fp32 work is elementwise
adds, norms and modulation.

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

Prediction error per σ, t2i and i2i: [int8.md](../optimization/int8.md#accuracy). Around bf16
GEMMs the stream cuts the error 2.4–3.5x, and around int8 GPTQ it lands below bf16 at every σ
tested. LoRA gradients improve only in blocks 28–41.

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
  stream moves a render about 3x more than changing the inference stream. Matching the two
  shows no extra effect.
- **Renders, in detail.** On Base accessories, eyes and small marks change and the composition
  holds; on SFT the composition can change. Read as whole images, neither stream wins across the
  24 cells of the findings. Read on text and fingers, fp32 wins (no blind count yet).
- **Loss, slightly.** 0.0002–0.0004 lower from step 1 on, before any update, so this is the
  forward's own error and not better learning.

The hands and `swap_text` leads, with their caveats, are in the
[finding](../findings/bf16_residual_stream.md).
