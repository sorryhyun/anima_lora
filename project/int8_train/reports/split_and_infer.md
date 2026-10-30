# Error split and generation on the int8 base (2026-10-30)

Two steps before an offline GPTQ base ([gptq_offline.md](../gptq_offline.md)): split the
int8 forward error into its weight and activation terms, and generate on the int8 base.

**Verdict.** Neither term dominates. The weight term is 35–50% of the extra forward
error (in variance), so a weight-only GPTQ can remove at most that share. On the
generation side, swapping the bf16 base for the int8 base moves pixels by 4/255, and
swapping the bf16-trained LoRA for the int8-trained one moves them by 26/255. The open
question is that 26: int8 training or run-to-run nondeterminism. A deterministic
bf16 × 2 floor answers it.

## Error split

`src/grad_error.py` gained two arms. Both take dX from the bf16 W, as `fwd` does:

- `w8a16`: the dequantised int8 weight, bf16 activation.
- `w16a8`: the bf16 weight, the activation through comfy_kitchen's eager reference
  (bf16 rotation, per-token int8), dequantised and rotated back.

Both run as fp32 GEMMs, so the emulation adds no rounding of its own. The CUDA kernel
and the eager reference round differently: they are 0.5% apart on random shapes, at the
same distance from bf16 (1.18%). `test_split_emulation_matches_kernel_error` pins the
emulation to the eager reference and to the kernel's error size, not to its exact
rounding.

Job `20261030-111545-c74021`, same images and σ as phase 1b. Raw:
[`grad_error_split.json`](grad_error_split.json).

Prediction error relative to the fp32 prediction (mean over 6 images):

| σ | bf16 | `fwd` (W8A8) | `w8a16` | `w16a8` |
|---|---|---|---|---|
| 0.1 | 0.0172 | 0.0244 | 0.0197 | 0.0215 |
| 0.5 | 0.0084 | 0.0104 | 0.0091 | 0.0096 |
| 0.9 | 0.0092 | 0.0111 | 0.0100 | 0.0100 |

Extra over bf16, in quadrature (`√(e² − e_bf16²)`):

| σ | `fwd` | `w8a16` | `w16a8` | weight share of `w8a16² + w16a8²` |
|---|---|---|---|---|
| 0.1 | 0.0173 | 0.0096 | 0.0129 | 36% |
| 0.5 | 0.0061 | 0.0035 | 0.0046 | 36% |
| 0.9 | 0.0062 | 0.0039 | 0.0039 | 50% |

The two terms add to 80–90% of `fwd`'s extra variance, so the split roughly closes.

**The gradient medians do not split cleanly.** At σ=0.5, `w16a8` alone sits further from
the bf16 gradient than `fwd` (0.238 against 0.179 of noise), and the terms do not add in
quadrature at any σ. One reading: on real activations the eager reference's bf16 math
quantizes worse than the kernel, which the random-shape test cannot see. The prediction
split above is the one to use.

**For GPTQ:** a weight-only GPTQ that removed the whole weight term would take `fwd`'s
prediction error at σ=0.1 from 0.0244 to about 0.0215 (−12%). The MOSS TE measured −14%
for W8A8. The activation term needs act-aware GPTQ or smoothing.

## Generation on the int8 base

`src/infer_int8.py` runs `inference.py` on the int8 base (see its docstring). LoRAs ride
the P-GRAFT hook path with no cutoff, so the base runs `int8(x, W_q) + B·A·x`, the
function `train_int8.py` trained. `int8_linear` gained mode `infer` (forward copy only,
bf16 weight dropped).

Phase 3's argv, both LoRAs, int8 base. Jobs `20261030-111545-1b071e` (int8-trained LoRA)
and `-c3fd42` (bf16-trained LoRA), 1.9 min each. Output:
`output/tests/int8ab/int8base_{full,bf16}/`, next to phase 3's `{full,bf16}/` on the
bf16 base.

The phase 3 argv passes `--lora_weight` twice, and argparse keeps only the last one
(`nargs="*"`), so all four arms ran the test LoRA alone, without
`corr_p1_cap40_sc_D`.

Mean pixel |Δ| (0–255), 8 images (4 prompts × seeds 11, 22):

| pair | changes | mean | per image |
|---|---|---|---|
| bf16 vs int8 base, bf16-trained LoRA | base | 3.9 | 2–7 |
| bf16 vs int8 base, int8-trained LoRA | base | 3.8 | 2–7 |
| bf16 vs int8-trained LoRA, bf16 base | LoRA | 25.9 | 11–64 |
| bf16 vs int8-trained LoRA, int8 base | LoRA | 25.5 | 10–62 |

- The int8 base at inference costs about one sixth of what the LoRA swap costs. The
  composition, pose and props follow the LoRA, never the base.
- The matched arm (int8 LoRA on int8 base) and the mismatched ones show no visible
  quality gap. At this size of base error, matching training and inference is not
  visible yet.

## Next

- **bf16 × 2 deterministic floor** (two `--int8 off --deterministic` runs, phase 3's data,
  seed and steps, about 8 min each). If two bf16 runs also differ by about 26/255, the
  LoRA gap in phase 3 is nondeterminism, not int8.
- GPTQ ([gptq_offline.md](../gptq_offline.md) steps 2–3) only after that. Its ceiling is
  the weight term, and the inference-side base error it would shrink is already 4/255.
