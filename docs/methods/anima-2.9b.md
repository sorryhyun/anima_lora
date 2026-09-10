# Anima-2.9B (40-block base)

[`Gazingstars123/Anima-2.9B`](https://huggingface.co/Gazingstars123/Anima-2.9B) is a
community depth expansion of `circlestone-labs/Anima`: 28 → 40 transformer
blocks at the same 2048 width, ~2.9B parameters. New layers were inserted
interleaved with deep-copied neighbour weights and zeroed output projections
([LLaMA Pro](https://arxiv.org/abs/2401.02415)), so the expanded model is
functionally identical to Anima-base at initialization; only the new layers were
trained for preview-v1.

```bash
make download-anima-variant ARGS=Anima-2.9B-preview-v1
```

## What changes on our side: nothing but depth

Same Qwen3-0.6B text encoder, same Qwen-Image VAE, same 16-channel latents, same
patch size. TE / VAE / PE caches and the whole preprocess pipeline are reused
unchanged — existing caches train against 2.9B as-is.

The loader reads depth and width off the checkpoint header
(`library/anima/weights.py::probe_dit_arch`) and builds the matching module list,
so there is no flag to set: point `--pretrained_model_name_or_path` (training) or
`--dit` (inference) at the file. See the *DiT depth is read from the checkpoint*
invariant in the root `CLAUDE.md` for the contract and its consequences.

## Measured envelope (RTX 5070 Ti, 16GB)

Inference — ~31.5 s/image at 832×1216, 32 steps euler, CFG 4.0, versus ~24.5 s
for the 28-block base: about 28% slower, consistent with 40/28 on the DiT plus
fixed load/decode overhead. Fits in 16GB with no special flags.

Training — the working recipe on 16GB is `--preset low_vram` with
`--blocks_to_swap 0`:

| Configuration | Result |
|---|---|
| default preset | OOM immediately (weights alone are 5.9GB bf16) |
| `--blocks_to_swap 16` | survives exactly one step, then OOM |
| `--preset low_vram --blocks_to_swap 0` | works — 32 steps at 4.6 → 2.2 s/it |

That one surviving step matters: it says the 40-block forward/backward path is
correct and the failure is purely a VRAM ceiling. Note `blocks_to_swap` and
`unsloth_offload_checkpointing` are mutually exclusive (hard assert in
`train.py::assert_extra_args`), so the two offload levers do not stack.

`torch_compile` is already on by default in `base.toml`, so the usual
"block-compile first" response to OOM is spent before you get here — gradient
checkpointing is the genuine next lever.

### Calibration coverage

`networks/calibration/channel_stats.safetensors` is baked at 28 blocks, so on a
40-block DiT `channel_scaling` applies to blocks 0–27 and warns for the remaining
120 modules (observed: `280 DiT modules received calibration-based input scaling`
/ `120 DiT modules have no calibration stats`). Training is unaffected in
correctness but blocks 28–39 train without input rebalancing — regenerate with
`scripts/calibration/analyze_lora_input_channels.py` before serious 2.9B runs.
`dave_alpha.npz` is likewise a 28-vector and raises a clear re-derive error.
**CNS γ is not depth-baked** — its `(1, 28, 32)` middle axis is timesteps.

### Adapters are depth-specific

Module names carry the block index, so a 40-block adapter merged onto the
28-block base silently drops its tail blocks behind a `not all LoRA keys are
used` warning. `save_weights` stamps `ss_num_blocks` so the mismatch is
machine-detectable.

## Sample comparison

Same prompt, seed, sampler and resolution on both checkpoints — the only variable
is the DiT. Rendered with euler, 32 steps, CFG 4.0, 832×1216.

![portrait](img/anima29b_p1_portrait.jpg)

On character prompts 2.9B stays a clear sibling of base: same composition family,
cleaner hands and cloth folds, smoother shading.

![scene](img/anima29b_p2_scene.jpg)

Backgrounds show the largest gap, and it is seed-dependent — seed 777 produces a
near-sibling pair, while seed 1234 diverges into a much more composed illustration
with stronger perspective, where base stays flat and photographic.

![two characters](img/anima29b_p3_twochar.jpg)

Multi-character prompts are mixed. 2.9B follows the posing better (base turned both
characters away from the viewer, which the prompt never asked for) but leaked the
straw hat assigned to only one character onto both — base got that attribution
right.

Reproduce with `bench`-style ad-hoc generation or `make gen`; the source images
for these sheets were rendered per-prompt/seed against both `--dit` paths.

## Prompting

Follow the base Anima conventions (quality tags, year/period tags, `@artist` tags,
character count, Danbooru/Gelbooru character tags, series/copyright). The upstream
model card adds: the training captions carry no score tags, character names
should be paired with their series tag, and short prompts underperform — detailed
prompts matter more here than on base. Recommended sampling is euler or
res-multistep at 28–50 steps, CFG 3.5–5.
