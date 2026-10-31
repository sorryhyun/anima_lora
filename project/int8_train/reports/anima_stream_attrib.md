# Anima: where the stream error and the mixed gradient come from (2026-10-31)

**Verdict: the mixed gradient of [anima_fp32_stream.md](anima_fp32_stream.md) is two effects.
Blocks 0–13 are too sensitive to compare arms on: fp32 with rounding-sized noise reads as far from
fp32 as any arm. Blocks 14–27 read cleanly, and there int8's forward quantization dominates, which
a stream cannot reach. Widening the stream (`wide`) takes bf16's prediction error down 1.8–3.3x
against 1.5–2.1x for the earlier arm, and halves a real bias in the mean gradient of blocks 21–27.
dX GEMMs and FA2 are not sources.**

Scripts: [`src/anima_stream_attrib.py`](../src/anima_stream_attrib.py),
[`src/anima_grad_bias.py`](../src/anima_grad_bias.py). Jobs `20261031-204232-0a9054` (gradient,
21.0 min), `20261031-204232-4c4103` (prediction, 1.6 min), `20261031-210752-7ae94d` (bias,
20.7 min). Raw numbers: `assets/anima_stream_attrib{,_pred}.json`, `assets/anima_grad_bias.json`.

## Setup

The six images, σ ∈ {0.1, 0.5, 0.9} and fp32 reference of `anima_fp32_stream.md`, one ε each, under
bf16 autocast, no LoRA. An arm is a base plus flags, each in fp32:

| flag | what |
|---|---|
| `s` | `x` at each block, `adaln_up_*` outputs: the earlier report's `*_s32` |
| `head` | `final_layer.linear` |
| `mod` | `t_embedder`, `adaln_*`, `final_layer.adaln_modulation` |
| `embed` | `x_embedder` |
| `rope` | RoPE and the q / k norms, rounded on entry to FA2 |
| `wide` | `s+head+mod+embed+rope` |
| `y32` | int8 kernel output and `dY`, both directions |
| `dx` | dX of the block linears as `dY.float() @ W.float()`, TF32 off |
| `sdpa` | attention in fp32 SDPA, both directions |

`fp32_jit<n>` is the fp32 reference with Gaussian noise of relative size 2⁻⁹ on `x` at each block,
seed n, and no bf16 anywhere. Its prediction error sits between `s` and bf16, so the noise is the
size of the rounding it stands in for.

## Prediction

`‖v − v_fp32‖ / ‖v_fp32‖`, %, mean over images:

| arm | σ 0.1 | σ 0.5 | σ 0.9 |
|---|---|---|---|
| `bf16` | 1.719 | 0.837 | 0.922 |
| `bf16+s` | 1.118 | 0.421 | 0.434 |
| `bf16+s+embed` | 1.101 | 0.416 | 0.431 |
| `bf16+s+rope` | 1.106 | 0.419 | 0.421 |
| `bf16+s+head` | 1.101 | 0.375 | 0.388 |
| `bf16+s+mod` | 1.023 | 0.349 | 0.362 |
| `bf16+wide` | **0.972** | **0.273** | **0.281** |
| `int8` | 2.093 | 0.933 | 1.000 |
| `int8+y32` | 1.628 | 0.612 | 0.611 |
| `int8+s` | 1.646 | 0.600 | 0.645 |
| `int8+s+y32` | 1.607 | 0.580 | 0.603 |
| `int8+s+embed` | 1.631 | 0.592 | 0.593 |
| `int8+s+rope` | 1.643 | 0.595 | 0.624 |
| `int8+s+head` | 1.634 | 0.569 | 0.615 |
| `int8+s+mod` | 1.588 | 0.548 | 0.572 |
| `int8+wide` | **1.550** | **0.506** | **0.530** |
| `int8+wide+y32` | 1.513 | 0.485 | 0.488 |
| `fp32_jit0` | 1.073 | 0.610 | 0.609 |

- `mod` and `head` are the two large parts. `embed` and `rope` barely move, but the model input
  stays bf16 in every arm, so `embed` is a lower bound.
- The parts add in quadrature. At σ 0.5, from `s`, they predict 0.285 for `wide`; it reads 0.273.
- `int8+wide` sits below `bf16` at every σ.
- `y32` buys 2–8% on `int8+wide`. What is left of int8's error is not the output's rounding.
- `dx` cannot change prediction, and `sdpa` does not (0.421 → 0.425).
- The kernel does return more digits in fp32: median `|y32 − y16| / |y|` is 1.35e-3.

## Gradient, one draw

`‖S − S_fp32‖ / ‖S_fp32‖`, %, median over images, blocks 0–6 | 7–13 | 14–20 | 21–27:

| arm | σ 0.1 | σ 0.5 | σ 0.9 |
|---|---|---|---|
| `fp32_jit0` | 85 \| 70 \| 3.8 \| 2.4 | 74 \| 46 \| 4.4 \| 2.1 | 63 \| 33 \| 3.9 \| 2.2 |
| `fp32_jit1` | 94 \| 84 \| 3.9 \| 2.6 | 94 \| 59 \| 4.3 \| 2.1 | 94 \| 46 \| 4.2 \| 2.5 |
| `bf16` | 96 \| 92 \| 7.7 \| 9.7 | 111 \| 75 \| 9.0 \| 8.4 | 109 \| 55 \| 10.1 \| 7.7 |
| `bf16+s` | 110 \| 99 \| 6.2 \| 6.8 | 92 \| 65 \| 7.2 \| 5.7 | 86 \| 44 \| 7.7 \| 5.6 |
| `bf16+s+dx` | 110 \| 100 \| 6.2 \| 6.8 | 92 \| 65 \| 7.2 \| 5.7 | 86 \| 44 \| 7.8 \| 5.6 |
| `bf16+s+sdpa` | 109 \| 113 \| 6.3 \| 6.7 | 108 \| 75 \| 7.5 \| 5.8 | 96 \| 45 \| 8.0 \| 6.1 |
| `bf16+s+sdpa+dx` | 109 \| 112 \| 6.3 \| 6.6 | 108 \| 76 \| 7.5 \| 5.7 | 95 \| 45 \| 7.9 \| 6.1 |
| `bf16+wide` | 92 \| 80 \| 5.6 \| 4.6 | 122 \| 89 \| 5.0 \| 3.6 | 66 \| 34 \| 5.0 \| 4.1 |
| `int8` | 175 \| 181 \| 11.4 \| 11.2 | 105 \| 82 \| 12.7 \| 9.8 | 100 \| 64 \| 14.0 \| 9.6 |
| `int8+s` | 94 \| 98 \| 10.1 \| 9.2 | 153 \| 118 \| 11.9 \| 8.3 | 94 \| 50 \| 14.0 \| 9.1 |
| `int8+s+dx` | 94 \| 97 \| 9.4 \| 8.7 | 154 \| 119 \| 11.4 \| 7.7 | 94 \| 50 \| 13.6 \| 8.8 |
| `int8+wide` | 80 \| 81 \| 10.6 \| 7.6 | 102 \| 77 \| 11.2 \| 7.2 | 97 \| 51 \| 12.8 \| 7.6 |
| `int8+wide+y32` | 90 \| 88 \| 9.4 \| 7.3 | 120 \| 66 \| 10.2 \| 6.5 | 100 \| 49 \| 11.9 \| 7.9 |

**Blocks 0–13 are a sensitivity, not a dtype effect.**

- Every arm reads about 100%, and two noise seeds with no bf16 read 63–94%. Six images give the
  medians no resolution: `int8+s` is 94 at σ 0.1 and 153 at σ 0.5.
- At σ 0.1 the largest spike, 434%, is `fp32_jit0`, above `int8`'s 307%. Spikes run to 224% in
  noise at σ 0.5 and 555% in `int8+wide`, so their size does not order the arms.
- About 30% of the error sits in one token and 70% in ten of ~4100. The same in fp32 noise.
- The hot token moves with the noise seed alone. At σ 0.1, image 0, block 0, fp32 holds 49% of
  `‖dY‖²` at token 1012, `fp32_jit0` 50% at 752, `fp32_jit1` 6% at 91.

The ratios of the earlier report in these blocks divide one such draw by another. Whether `int8`'s
175 at σ 0.1 is more than chance, three images of six spiking, six images cannot say.

**Blocks 14–27 read cleanly.** The noise floor is 2–4%. bf16 falls to it step by step: `s`, then
`wide`. Around int8 the arms barely move, and `bf16+wide` against `int8+wide`, which differ only in
the block linears, reads 5.0 against 11.2 at σ 0.5. With `dx` showing int8's backward kernel worth
3–7%, that gap is its forward copies. This is by elimination, not traced directly. It is
why `int8` and `int8+s` tie at 14.0 at σ 0.9, blocks 14–20.

**dX and FA2 are ruled out as sources in blocks 14–27.** MOSS's doc names both for its early blocks,
which this does not test. `dx` keeps the arm's bf16 `dY`, so `dY`'s own rounding is only tested on
int8, by `y32`, which moves blocks 14–27 by −11% to +4%.

## Mean gradient

Over D = 16 paired draws at σ 0.7, `e_d = S_arm − S_fp32`, and
`bias² = ‖ē‖² − s²/D`, unbiased for `‖E[e]‖²`. % of `‖mean S_fp32‖`, in brackets the standard
error `s/√D`:

| arm | image | 0–6 | 7–13 | 14–20 | 21–27 |
|---|---|---|---|---|---|
| `fp32_jit` | hews | 32 (60) | 0 (28) | 0.5 (1.6) | 0.0 (1.0) |
| | patricia | 64 (90) | 0 (36) | 0.8 (3.0) | 0.3 (1.3) |
| `bf16` | hews | 0 (76) | 0 (37) | 3.4 (3.0) | **7.9 (2.2)** |
| | patricia | 0 (110) | 0 (56) | 1.8 (6.8) | **5.6 (3.1)** |
| `bf16+s` | hews | 16 (56) | 0 (28) | 2.6 (2.3) | **5.5 (1.9)** |
| | patricia | 21 (107) | 18 (52) | 2.2 (5.5) | **3.8 (2.2)** |
| `bf16+wide` | hews | 24 (83) | 0 (39) | 1.9 (2.0) | **3.6 (1.4)** |
| | patricia | 28 (96) | 0 (38) | 1.4 (3.4) | **3.2 (1.8)** |
| `int8` | hews | 21 (85) | 11 (41) | 4.7 (4.3) | **8.4 (3.1)** |
| | patricia | 0 (139) | 16 (57) | 4.7 (8.0) | 6.1 (4.3) |
| `int8+wide` | hews | 43 (71) | 17 (33) | 5.1 (3.7) | **6.0 (2.6)** |
| | patricia | 0 (97) | 10 (48) | 4.3 (7.0) | 3.9 (3.6) |

- The control holds: 0.0–0.8 in blocks 14–27.
- Blocks 21–27 carry a bias that more draws will not average away, 6–8% in bf16. `wide` takes it to
  about 3.5%, in the same order on both images.
- Blocks 0–13 are not readable: the control itself reads 32–64 at this D. Whether they are biased
  stays open. Single-draw distance there is 112–554% of the mean gradient.
- Blocks 14–20: every arm is within its standard error on patricia, and within about one on hews.

## σ-demoted training

Every bench of that line ran bf16 autocast (`preserve/sigma-lowres-yarnsig`), and no fp32 DiT
forward turned up in a search of it. This run shows its mean gradients carry a real bias in the late
blocks, and says nothing about the early ones. It was not compared with the line's measured
residuals: the units differ, and native and demoted arms could share the bias. The cancellation
ρ of −0.7 to −0.9 between two large vectors is unlikely to fall to a 6–8% error; the sign and size
of the residual is where it could matter. That is a judgement, not a measurement.

## Not done

- Everything here ran under bf16 autocast with a bf16 model input. `inference.py` and ComfyUI have
  no autocast, and `inference.py` rounds its sampler state to bf16 each step
  (`generation.py:487`), a source no arm measured.
- Cost of `wide` in time or memory, and any LoRA or render.
- More than two images, or any σ but 0.7, for the bias.
- Whether blocks 0–13 are biased: the standard error has to fall about tenfold, so about 100x the
  draws. Not worth running on one σ.
- `int8` at σ 0.1 in blocks 0–13 on 30 images.

Proposal: [`docs/anima/proposal/fp32_stream.md`](../../../docs/anima/proposal/fp32_stream.md).
