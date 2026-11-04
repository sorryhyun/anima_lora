# Reference stream under `no_grad` on MOSS i2i (2026-11-02)

**Verdict: stops at the gradient check.** Candidate 4 of
`_archive/proposals/step_headroom.md`. Cutting the reference stream
turns the mean LoRA gradient by cos 0.66 at init (norm 0.74x) and 0.65 at epoch 1. Keeping only
`ref_mod` and the per-block K / V does not save the mod adapter.

Script: [`src/ref_detach_grad_moss.py`](../src/ref_detach_grad_moss.py), job
`20261102-111903-1042ed`, about 8 min, peak 12.84 GB. Raw numbers:
[`moss_ref_detach_grad.json`](../assets/moss_ref_detach_grad.json).

## Setup

swap_text v3 en400_ko1709 recipe (`ref_timestep = 0`, `ref_isolate`, `mod_rank = 16`,
`region_weight = 2`), bf16 base, fp32 stream, eager, checkpointed per block, swap 20. Six
full-size pairs (3 ko, 3 en), σ ∈ {0.6, 0.9}, the same noise across arms, 24 passes per arm.
States: `init` (fresh adapters, `up` only has a gradient) and `e1` (the epoch checkpoint).

Every arm runs one split block, the reference rows with `kv_out`, then the main rows with
`ref_kv`. `attached` cuts nothing, `detached` is the proposal, and `mod_kv` cuts the reference
`x` at each block and keeps `ref_mod` and K / V in the graph. The forward is the same in all
three (the logged losses match), so these are paired comparisons. The floor is the cosine
between the `attached` mean gradients of two disjoint pair sets.

## Results

Mean gradient against `attached`, cosine (norm ratio):

| state | group | floor | `detached` | `mod_kv` |
|---|---|---|---|---|
| init | all | 0.82 | 0.66 (0.74) | 0.67 (0.77) |
| init | mod | 0.85 | 0.57 (0.71) | 0.58 (0.73) |
| init | k | 0.59 | 0.73 (0.89) | 0.99 (0.99) |
| init | v | 0.74 | 0.86 (0.80) | 0.90 (0.93) |
| init | o | 0.74 | 0.83 (0.82) | 0.83 (0.82) |
| init | ffn_up_gate | 0.80 | 0.80 (0.81) | 0.80 (0.81) |
| init | ffn_down | 0.89 | 0.80 (0.80) | 0.80 (0.80) |
| e1 | all | 0.47 | 0.65 (0.83) | 0.68 (0.87) |
| e1 | mod | 0.48 | 0.71 (1.11) | 0.71 (1.12) |
| e1 | k | 0.10 | 0.75 (0.85) | 0.98 (0.99) |
| e1 | v | 0.29 | 0.70 (0.61) | 0.79 (0.89) |
| e1 | o | 0.39 | 0.70 (0.81) | 0.70 (0.81) |
| e1 | ffn_up_gate | 0.63 | 0.57 (0.62) | 0.57 (0.63) |
| e1 | ffn_down | 0.68 | 0.51 (0.56) | 0.51 (0.56) |

- `q` barely moves (0.98–0.99), since a reference query only reaches the reference stream.
- The late blocks lose the most. Blocks 28–41 at e1, `detached`: `o` 0.45, `ffn_down` 0.44,
  `v` 0.41, with floors of 0.80–0.85. These are the blocks the i2i tasks share for copying
  ([moss_i2i_ambitious.md](../../../../docs/moss/proposal/moss_i2i_ambitious.md)).
- `mod_kv` restores `k` and most of `v`, not `mod`, `o` or the FFN. A reading of `_Block`, not
  measured by parts: the gates and `shift_f` / `scale_f` reach the loss through the block's `x`
  output, which `mod_kv` cuts, so only `shift_a` / `scale_a` stay in the graph.

Eager median s/pass: `attached` 1.87 + 4.65, `detached` 1.87 + 3.36 (20% off the pass),
`mod_kv` 1.87 + 3.55 (17%). The proposal estimated 30–35% on the compiled plan, which this
does not measure.

## Open

`joint` against `attached` on one sample: cos 0.99994 at init, 0.80 (norm 1.04x) at e1. Rounding
is the likely reason, since the math matches and the
[bf16 gradient error](moss_grad_error.md) is on the order of minibatch noise, and the e1
gradient is small. Not checked further: the verdict stands on init alone, and the arms are
paired on one forward.
