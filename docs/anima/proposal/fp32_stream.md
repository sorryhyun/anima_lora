# Anima fp32 stream — the `wide` arm in the library and the adapter node

Status: **proposed, nothing implemented** (2026-10-31). Measurements:
`project/int8_train/src/anima_stream_attrib.py`, jobs `20261031-204232-0a9054` (gradient) and
`20261031-204232-4c4103` (prediction). Earlier arm:
[`anima_fp32_stream.md`](../../../project/int8_train/reports/anima_fp32_stream.md).
MOSS's version: [`docs/moss/methods/fp32_stream.md`](../../moss/methods/fp32_stream.md).

## Goal

Anima gets what MOSS has: `--stream fp32` for training and generation, and a patch in ComfyUI
applied together with the LoRA. The aim is small structure in renders. On MOSS the fp32 stream
visibly changed text rendering and finger placement, and that is what the A/B here reads.

Anima's bf16 stream error is about half MOSS's, and no Anima render has been compared yet. The
stream ships on by default for generation on MOSS's visible result, with `bf16` one switch away.

## What was measured

Prediction error, `‖v − v_fp32‖ / ‖v_fp32‖` in %, six images, one ε each, under bf16 autocast.

| σ | bf16 | `s` | `wide` | int8 | int8 `s` | int8 `wide` |
|---|---|---|---|---|---|---|
| 0.1 | 1.72 | 1.12 | **0.97** | 2.09 | 1.65 | **1.55** |
| 0.5 | 0.84 | 0.42 | **0.27** | 0.93 | 0.60 | **0.51** |
| 0.9 | 0.92 | 0.43 | **0.28** | 1.00 | 0.65 | **0.53** |

`s` casts `x` at each block and the `adaln_up_*` outputs to fp32. `wide` adds, each in fp32:

| part | what it covers | bf16 GEMMs, σ 0.5, from `s` 0.421 | int8, σ 0.5, from `s` 0.600 |
|---|---|---|---|
| `mod` | `t_embedder`, `adaln_*`, `final_layer.adaln_modulation` | 0.349 | 0.548 |
| `head` | `final_layer.linear` | 0.375 | 0.569 |
| `embed` | `x_embedder` | 0.416 | 0.592 |
| `rope` | RoPE and the q / k norms | 0.419 | 0.595 |

The parts add in quadrature: from `s`, they predict 0.285 for `wide`, against 0.273 measured.
Rounding the int8 kernel's output too (`y32`) buys only 2–8%, and it is not proposed.

LoRA gradient, `‖S − S_fp32‖ / ‖S_fp32‖` in %, median over images, blocks 14–20 | 21–27:

| σ | fp32 + noise | bf16 | `s` | `wide` | int8 | int8 `wide` |
|---|---|---|---|---|---|---|
| 0.1 | 3.8 \| 2.4 | 7.7 \| 9.7 | 6.2 \| 6.8 | 5.6 \| 4.6 | 11.4 \| 11.2 | 10.6 \| 7.6 |
| 0.5 | 4.4 \| 2.1 | 9.0 \| 8.4 | 7.2 \| 5.7 | 5.0 \| 3.6 | 12.7 \| 9.8 | 11.2 \| 7.2 |
| 0.9 | 3.9 \| 2.2 | 10.1 \| 7.7 | 7.7 \| 5.6 | 5.0 \| 4.1 | 14.0 \| 9.6 | 12.8 \| 7.6 |

- Blocks 14–27 improve in every arm that widens the stream. The int8 base keeps most of its
  distance: its forward quantization dominates, and a stream cannot reach it.
- Blocks 0–13 read about 100% in every arm, and 63–94% in fp32 with noise of 2⁻⁹ on `x` and no
  bf16 at all. Their single-draw gradient is too sensitive to compare arms on. Whether the
  *mean* gradient is biased is [below](#mean-gradient-bias).
- dX GEMMs and FA2 are not sources: an exact fp32 dX and fp32 SDPA leave blocks 14–27 unchanged.

## Mean gradient bias

`src/anima_grad_bias.py`, σ = 0.7, two images, 16 paired draws: % of the mean gradient that more
draws would not average away, blocks 21–27, with fp32 + noise as the control.

| image | fp32 + noise | bf16 | `s` | `wide` | int8 | int8 `wide` |
|---|---|---|---|---|---|---|
| hews | 0.0 | 7.9 | 5.5 | 3.6 | 8.4 | 6.0 |
| patricia | 0.3 | 5.6 | 3.8 | 3.2 | 6.1 | 3.9 |

Standard errors are 1.0–4.3, so the int8 readings on patricia are weak. The stream removes about
half of a real late-block bias, which gives Phase 4 a reason beyond renders. Blocks 0–13 cannot be
read at this D: the control itself reads 32–64 there. Full tables:
[`anima_stream_attrib.md`](../../../project/int8_train/reports/anima_stream_attrib.md).

## What the measurements do not cover

The arms ran under bf16 autocast with a bf16 model input. Neither path users generate on matches
that, so the numbers above are not a prediction for those paths. Phase 1's bench reads them.

| | trainer | `inference.py` | ComfyUI |
|---|---|---|---|
| autocast | bf16 | **none** (no autocast in `library/inference/`) | none |
| `LayerNorm` output | fp32 | **bf16** | bf16 |
| sampler state | – | **bf16**, rounded each step (`generation.py:487`) | fp32, truncated on entry to the DiT |
| block input `x` | bf16 | bf16 | bf16 |

So today's training and generation already differ in what the block linears read, and the
sampler's rounding in `inference.py` is a source no arm measured. It would also take back part of
`head`'s gain on the next step.

## Library

One mechanism for both paths, in a new `library/anima/stream.py`. `--stream bf16|fp32` in
`configs/base.toml` and the generation request: `fp32` by default for generation, `bf16` for
training until Phase 4. The LoRA records it as `ss_stream`, a record only, as on MOSS.

**`stream_fp32_(dit)`** runs at the point int8 converts: after `create_network`, before
`apply_to` and `compile_blocks`, so a LoRA's `org_forward` captures the fp32 forward.

- A `Fp32Linear` class swapped onto the `mod`, `head` and `embed` linears. Its forward leaves
  autocast and casts the weight for the call: `F.linear(x.float(), self.weight.float())`. No
  parameter changes dtype, so the fp32 copy costs no VRAM and `unet.to(dtype=weight_dtype)`
  cannot undo it. This is what the arm measured.
- `x` to fp32 once in `_run_blocks`, and `adaln_up_*` already fp32 through the swap.
- `RMSNorm` keeps fp32 on the stream (`rope`), a few lines. RoPE then rotates q and k in fp32, and
  the cast already in `Attention.forward` rounds them on entry to FA2.
- `Attention` and the block linears are otherwise unchanged. Autocast rounds their input, and the
  int8 linears keep their fp32 kernel input.

**Generation runs under bf16 autocast when `stream = fp32`.** The block forward then has one
behaviour in both paths, and needs no input casts of the kind MOSS's has. `bf16` stays as it is
today, without autocast, bit for bit.

- Latents in fp32 from `randn_tensor` on (`generation.py:304`, `:560`), so `:487` stops rounding.
- The Spectrum, SPD and foveated runners take the same dtype; each is checked, not assumed.
- `make test STREAM=1`, composing with the other `test-*` levers.

**Training** needs the same conversion and little else. `work = org_forwarded.dtype` in
`networks/lora_modules/base.py` already handles a LoRA on an fp32 output, and
`tests/test_lora_dtype_policy.py` pins it. To be established:

- Whether `torch.autocast(enabled=False)` inside a compiled block keeps `compile_dynamic_seq`'s
  one graph. If it does not, the modulation has to be computed outside the compiled block.
- The saved set grows with `x`, so the budget table in `docs/anima/optimizations/int8.md` does
  not carry over. Re-sweep at 0.99 and at 0.4.
- Turbo is out of scope: bespoke loop, no int8 step has run on it either.

## ComfyUI

`~/ComfyUI-Anima_lora-Adapter`, a new `stream.py` on the pattern of
`ComfyUI-MOSS-Toolkit/stream.py`. ComfyUI's `Block.forward` already separates `residual_dtype`
(`x`) from `compute_dtype` (`emb`), which is why the patch stays small.

`apply_model_wrapper`, an `APPLY_MODEL` wrapper, for the one call:

- hands the forward fp32 `x`, `context` and timestep embedding, so `mod` and `head` run in fp32:
  `final_layer` reads `x.to(crossattn_emb.dtype)`, so `context` has to be fp32 for `head`;
- pre-hooks on the block linears (`q_proj`, `k_proj`, `v_proj`, `output_proj`, `layer1`, `layer2`)
  cast their input back to `get_dtype_inference()`, or the emb's fp32 would pull every GEMM to fp32;
- sets `comfy_cast_weights`, restores everything in `finally`, and returns early on an fp32 model.

Node surface:

- `AnimaAdapterLoader` gets an optional `stream` (`fp32` by default, or `bf16`), applied in
  `apply()` after `apply_adapter`. Keyed wrapper, so chained loaders apply it once. It is an
  `optional` input: existing workflows keep loading, and pick up `fp32`.
- On by default means it may not break a working graph. Where the patch cannot run (the table
  below decides which cases), the node logs a warning and leaves that model on bf16.
- A small `AnimaStreamFP32` node (`MODEL → MODEL`) for a model with no adapter, sharing the core
  function.
- No `_vendor` change: the module names differ, so this is a copy of the idea, not of the code.

`rope` is not in v1 here. ComfyUI runs norm and RoPE as one fused kernel
(`comfy.quant_ops.ck.rms_rope_split_half`, `cosmos/predict2.py`), so no hook reaches between them.
Widening it means replacing `apply_norm_and_rotary_pos_emb` or a kernel with an fp32 path. What
dtype that kernel works in internally was not checked.

Compatibility to test, each with a LoRA attached:

| | risk |
|---|---|
| plain LoRA | none expected: `add_patches` changes weights, ops cast them |
| Hydra, ChimeraHydra, FeRA | a `forward_hook` on an `adaln_*` Linear reads an fp32 input against bf16 expert weights |
| ReFT | block `forward_hook` adds to an fp32 output |
| soft tokens, vocab pack | rewrite `crossattn_emb` with their own dtype |
| step-expert turbo | none expected |
| DAVE | block hooks read an fp32 `x` |
| KSampler (Spectrum) | caches block residuals, now fp32 |
| Block Compile | **hooks installed per call do not reach a compiled block.** The likeliest blocker |
| INT8 / fp8 files | `to_compute` drops the kernel's fp32 input, the 6–8% the trainer keeps. Acceptable for v1, fixed if the op allows |

## Phases

**Phase 1 — generation.** `library/anima/stream.py`, the generation path, `STREAM=bf16` for the old
one. `CONTRIBUTING.md` Tier 1.5: an invariant test that `bf16` is bit-identical to today, and a
bench that records, without gating anything:

- prediction error on `inference.py` as shipped, with and without `rope` (the gap ComfyUI v1 has);
- a full trajectory with `src/traj_eval.py`, final latent against an fp32 run;
- ms/step and peak.

**Phase 2 — ComfyUI.** The patch, the two node surfaces, the compatibility table. Numerical check:
one step through ComfyUI against `library`, with the stream on. Publish once the table passes.

**Phase 3 — render A/B, for the record.** Not a gate on Phases 1–2. Same seed, bf16 against fp32,
base and with a LoRA.
Read on crops of text (CJK included) and hands, with more than one seed per cell, blind where
possible. MOSS's hand read was neither, which is why it stayed a lead. Base model first, since no
retraining is needed.

**Phase 4 — training.** `--stream fp32`, the compile check, the budget re-sweep, one LoRA run with
its A/B, then the default. On MOSS the training stream moved a render about 3x more than the
inference stream. It comes last because it is the one path with memory and compile unknowns.

## Decisions to make

- **`rope` on ComfyUI.** Proposed: left out of v1, and only taken up if Phase 1's bench reads a gap
  against the library's `wide`. It measured about 3% at most on its own.

## Risks

- `mod` is a fixed rounding per σ, and pretraining's dtype was not checked. If the base model
  trained on a bf16 stream, closer to fp32 is not automatically closer to what it learned.
- Generation under autocast on `stream = fp32` changes what `LayerNorm` hands the block linears,
  a separate effect from the stream, and the two ship together.
- σ 0.1 stays near 1% in every arm, and fp32 with noise reads the same. Low-σ detail may be where
  a gain is least, which is also where text and fingers finish.
