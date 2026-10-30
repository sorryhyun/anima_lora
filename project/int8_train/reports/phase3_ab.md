# Phase 2–3 — implementation and the first A/B (2026-10-30)

**Verdict: passes both kill criteria.** The int8 `full` base trains 1.37x faster than
bf16 on the same data, seed and steps. Loss curves overlap, and samples from the two
LoRAs, both generated on the bf16 base, are of the same quality.

## Phase 2

- `src/train_int8.py` runs the stock trainer with `--int8 {off,fwd,full}` (see its
  docstring). `off` installs no patch.
- `src/int8_linear.py` gained `Int8TrainLinear._apply`. The trainer calls
  `unet.to(dtype=bf16)` after `apply_to` (`library/anima/trainer/optim.py`), which
  would have rounded the fp32 row scales to bf16.
- `src/test_int8_linear.py`: 7 tests pass. They cover forward and dX against bf16 (rel
  < 0.03), LoRA grads against an eager reference (rel < 1e-2) in both modes, scales
  staying fp32 through a dtype cast, and no graph break in a compiled converted block,
  forward and backward.
- The saved LoRA has the same 1092 keys and shapes as the bf16 arm's, plus
  `ss_int8_base=full`.

## Phase 3 setup

- `ama mitsuki`, 106 images (`--artist_filter 'ama mitsuki'`), method `lora`, preset
  `default`, 8 epochs = 848 steps, B=1, seed 1408198296, compiled
  (`seq∈(2990, 4200)`, 1 graph).
- int8 arm: job `20261030-103512-4af761`. bf16 arm (`--int8 off`): job
  `20261030-104207-7d67f9`. LoRAs: `output/ckpt-artist/int8ab_{full,bf16}.safetensors`.
- Samples: `make gen` on the bf16 base with each LoRA, prompts
  [`src/ab_prompts.txt`](../src/ab_prompts.txt): one training caption and three prompts
  outside the set, seeds 11 and 22. Output in `output/tests/int8ab/{bf16,full}/`.

## Results

| arm | 848 steps | per step | final avr_loss |
|---|---|---|---|
| bf16 | 8 min 23 s | 593 ms | 0.0737 |
| int8 `full` | 6 min 7 s | 433 ms | 0.0737 |

Speedup **1.37x**, including compile warmup in both arms. Phase 1a predicted 1.41x.

Epoch-mean `loss/current`, bf16 / int8:

| ep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| bf16 | 0.0751 | 0.0720 | 0.0777 | 0.0747 | 0.0772 | 0.0716 | 0.0741 | 0.0755 |
| int8 | 0.0754 | 0.0724 | 0.0775 | 0.0746 | 0.0769 | 0.0716 | 0.0740 | 0.0755 |

REPA `align_loss` matches the same way (epochs 3–8 within 0.0008).

The LoRA weights differ by 12.8% (relative L2 over all `lora_up` / `lora_down`). The
samples differ in composition details: hoodie colour, cup type, background framing.
Style, anatomy and rendering quality show no gap in either direction. Mean pixel
|Δ| per pair ranges from 11 to 64 (0–255 scale).

## Not measured

- The bf16 vs bf16 floor: two bf16 runs on the same seed, to tell whether the 12.8%
  weight difference and the composition changes exceed run-to-run nondeterminism.
- A longer or larger run (one artist, 848 steps here).
- Modes `fwd` and `full` against each other end to end. Phase 1b found them equal on the
  gradient.

## Speed ceiling

The converted GEMMs are 47% of the bf16 step. At 2.6x on them, the step bound is
1/(0.53 + 0.47/2.6) ≈ 1.41x, and 1.89x if they cost nothing. Beyond that, the remaining
53% has to come down: attention (comfy_kitchen has no int8 attention backward), the
rotate-and-quantize overhead, the LoRA rank GEMMs and norms.
