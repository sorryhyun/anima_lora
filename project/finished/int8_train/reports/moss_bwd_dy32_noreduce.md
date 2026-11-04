# MOSS: fp32 `dY` into the int8 dX kernel, and bf16 split-K off (2026-11-03)

**Verdict: neither arm moves the LoRA gradient. Both sit inside the plan's own repeat
spread, so they are closed.** The two bf16 points left on the backward of the int8 plan
(the `dY` the dX kernel rounds around its activation division, comfy-kitchen#289 mirrored,
and the LoRA-side cuBLAS GEMMs' bf16 split-K partials) are not where the gradient error
is.

Script: [`src/moss_dy32_noreduce.py`](../src/moss_dy32_noreduce.py), job
`20261103-112945-49fefe`, raw numbers: [`moss_dy32_noreduce.json`](../assets/moss_dy32_noreduce.json).
`grad_error_moss.py` on the production numerics (`gptq_fused_f32s`: GPTQ, fused qkv, fp32
stream, fp32 kernel input), six held-out t2i images, σ ∈ {0.1, 0.5, 0.9}, eager, FA2 with
the fp16acc backward.

| arm | change against the plan |
|---|---|
| `plan_rep` | none: the repeat, FA2's non-determinism floor |
| `plan_dy32` | `int8_mm(dy.float(), wq_t, s_t)` in the backward, every converted linear |
| `plan_noreduce` | `allow_bf16_reduced_precision_reduction = False`; on this plan the only bf16 cuBLAS GEMMs are the LoRA rank GEMMs and their weight gradients (K = 4.6k tokens) |

## Results

Prediction error and loss are identical to the plan in every arm, to all printed digits,
since neither touches the forward.

`‖S − S_fp32‖` as a multiple of bf16's, median over modules:

| σ | plan | `plan_rep` | `plan_dy32` | `plan_noreduce` |
|---|---|---|---|---|
| 0.1 | 0.961 | 0.950 | 0.965 | 0.960 |
| 0.5 | 2.084 | 2.148 | 2.071 | 2.138 |
| 0.9 | 1.232 | 1.234 | 1.228 | 1.249 |

Each arm over the plan, pooled per image, median [max] over the 18 (image, σ) pairs by depth:

| over plan | 0–6 | 7–13 | 14–20 | 21–27 | 28–34 | 35–41 |
|---|---|---|---|---|---|---|
| `plan_rep` | 1.005 [1.09] | 1.004 [1.13] | 1.004 [1.13] | 1.005 [1.13] | 0.999 [1.01] | 1.001 [1.01] |
| `plan_dy32` | 1.002 [1.02] | 0.999 [1.02] | 1.001 [1.02] | 1.001 [1.04] | 1.002 [1.02] | 0.999 [1.01] |
| `plan_noreduce` | 1.006 [1.03] | 1.004 [1.04] | 1.004 [1.04] | 1.002 [1.03] | 1.000 [1.01] | 1.002 [1.01] |

- The repeat alone moves a module group by up to 1.13x; both arms stay under that at every
  depth, and their medians sit at 1.00.
- `dy32` reads 0.6% below the plan at σ 0.5 overall, the repeat 3% above: the arm is inside
  the floor.
- The early-block gradient error (blocks 7–27 at 3.1–3.5x bf16 at σ 0.5) is untouched by
  the backward kernel's input rounding. It comes from the int8 forward, as
  [moss_grad_error.md](moss_grad_error.md) read it.

## Not done

- The compiled step, where Inductor may already elide `dY`'s rounding as it does the
  forward's ([moss_fwd16_triton.md](moss_fwd16_triton.md#where-fa2s-output-goes)).
- The cost of either arm was not timed: nothing here would be shipped.
