# Removing a "numerically inert" custom autograd Function changed the compile partitioner and OOMed: numerically-inert ≠ memory-inert

On 2026-06-10 we removed `custom_autograd.py` (the LoRA
down-projection `autograd.Function` + the fp32-bottleneck matmul policy it
serviced) after the `lora_fp32_bottleneck` bench proved it numerically dead under
the trainer's autocast(bf16). The numerics verdict was correct — and the very
next no-grad-ckpt training run OOMed at step 0 on the 16 GB card, in a way that
resisted every config-level mitigation. This doc records why, how it was
root-caused, and the generic replacement, because the trap is reusable: a
custom `autograd.Function` under `torch.compile` is a partitioner constraint,
so removing one changes activation memory even when its math traces
identically.

## The failure

After commits `87aad30` (removal) + `8c2005c` (compute-dtype fix), the default
`make lora` job (OrthoInit + T-LoRA + channel scaling, `torch_compile=true`,
`gradient_checkpointing=false`, batch 1 @ 4200 tokens) died on the first step:

    torch.OutOfMemoryError: Tried to allocate 18.00 MiB. GPU 0 has a total
    capacity of 15.47 GiB ... 14.58 GiB is allocated by PyTorch

The same recipe had been training at ~14.4–14.7 GB process peak the day before.
None of the obvious levers moved it: channel scaling off, plain LoRA instead of
OrthoInit, `compile_dynamic_seq` off — every combination OOMed at step 0 with
~14.2–14.5 GB allocated. (`blocks_to_swap=24` in base.toml was additionally a
silent no-op: `presets.toml[default]` sets `blocks_to_swap=0` and preset
overrides base in the merge chain.)

## Red herrings, in the order we cleared them

1. fp32 upcast from `x.dtype` keying (`8c2005c` + the same-day OrthoInit
   fix). Real bugs — the AdaLN `nn.LayerNorm` feeding the adapted Linears
   emits fp32 under autocast, so keying the rank GEMMs off `x.dtype` upcast
   the whole rank path — but fixing them did not cure the OOM. Necessary, not
   sufficient.
2. `compile_dynamic_seq` single-graph compile (`b0a07b8`, landed hours
   before the removal). Exonerated: a static-per-family run OOMed identically.
3. `mixed_precision` regression (`d48264c` touched precision plumbing the
   same evening). Exonerated: default is `"bf16"` end-to-end into
   `Accelerator(...)`.
4. Channel scaling's `_rebalance` (`x * inv_scale` materialization — the
   thing the scaled Function specifically folded into the weight).
   Insufficient: `channel_scaling_alpha=0` runs OOMed too.

## The bisect that settled it

Two bounded 26-step runs of the identical job (same dataset, same configs,
no grad-ckpt, dynamic-seq on, channel scaling on):

| code state | outcome | peak (nvidia-smi total) | step time |
|---|---|---|---|
| `6955a57` (pre-removal, custom Function live) | trains 26/26 | 15.2 GiB | 1.01 s/it |
| `8c2005c` + OrthoInit fix (control) | OOM at step 0 | >15.8 GiB (died) | — |

Same data, same configs, same compile mode — the only delta is the custom
Function vs plain traceable ops. The removal is the memory regression,
≈ +0.8 GB of saved-for-backward activations.

## Mechanism

Dynamo traces an `autograd.Function` as a higher-order op that honors the
user's `ctx.save_for_backward` choice. The old `LoRADownProjectFn` saved
exactly `{x, weight}` (references to tensors that were alive anyway) and
recomputed the casts/scale-fold in its hand-written backward — an explicit,
human-chosen recompute boundary embedded in the graph.

Replace it with plain ops and that boundary disappears: AOT autograd's min-cut
partitioner now decides globally, per block graph, what to save for backward —
and across 196 adapted Linears × 28 blocks its default choice kept ~0.8 GB more
intermediates than the Function's. Nothing in the numerics changed; only the
*partition* did. This is the same class of trap as the earlier residue note
("partitioner decisions differ between toy graphs and realistic blocks"), one
level up: the bench validated forward/backward *values*, not the *saved set*.

## The fix: `activation_memory_budget`

Rather than resurrect the Function, cap the partitioner generically:

```toml
# configs/base.toml
activation_memory_budget = 0.85
```

wired as a trainer arg (`library/config/cli_args.py`) and applied in `train.py`
right before `compile_blocks`, setting
`torch._functorch.config.activation_memory_budget`. Below 1.0, the min-cut
partitioner solves a knapsack to recompute the cheapest intermediates in
backward instead of saving them — the role the Function played implicitly,
now applied to the whole block graph.

Verified on the same 26-step job:

| config | outcome | peak | step time |
|---|---|---|---|
| HEAD, budget unset (control) | OOM at step 0 | — | — |
| HEAD, `budget=0.85` | trains 26/26 | 15.2 GiB | 1.02 s/it |
| pre-removal reference | trains 26/26 | 15.2 GiB | 1.01 s/it |

i.e. the pre-removal footprint and speed, restored exactly.

Two operational notes:

* Auto-skipped under `gradient_checkpointing` (logged). The budget
  repartitions the joint graph, so checkpoint's recompute pass can select a
  different compiled graph than the forward and trip
  `torch.utils.checkpoint.CheckpointError` (saved-vs-recomputed metadata
  mismatch; pytorch #166926). It's also redundant there — ckpt already
  minimizes saved activations.
* It's a plain module attr on `torch._functorch.config`, not a
  ContextVar-backed dynamo entry, so it does not suffer the
  `recompile_limit`-style revert inside the backward-compile context.

## What ports

* **Numerically-inert ≠ memory-inert.** Before deleting any
  `autograd.Function` (or `torch.utils.checkpoint` wrapper, or
  `allow_in_graph` boundary) from a compiled path, compare the saved-tensor
  footprint (`torch.cuda.max_memory_allocated` over a real step, or
  `torch._functorch.config.debug_partitioner`), not just outputs/grads.
* `activation_memory_budget` is the sanctioned knob for "the partitioner
  saves too much" — prefer it over hand-written recompute Functions, and try
  it before reaching for gradient checkpointing on OOM (it was free at 0.85 on
  the 2026-06-10 workload; grad-ckpt cost ~18% step time here).
  Update 2026-07-04 (post-free-fit): the free region has since shrunk —
  a re-sweep on the current free-fit workload put the knee exactly at the
  shipped `0.99` (1.0 OOMs, 0.99 trains at ~13.1 GB), while `0.85` bought only
  a few hundred MB more at a real step-time cost. The knapsack's cheap-recompute
  tail is workload-dependent; re-measure after any change to the compiled
  block's op mix rather than trusting this doc's 0.85 number.
* Bisect with the real job. Config-level guesses (five of them above) all
  pattern-matched and all missed; two bounded 26-step runs at adjacent commits
  settled it in minutes.
