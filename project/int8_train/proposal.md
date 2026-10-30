# Proposal — int8 GEMMs for Anima LoRA training (2026-10-30)

## Claim

LoRA training freezes the DiT, so each base linear needs only two GEMMs per step: the
forward `Y = X Wᵀ` and the input gradient `dX = dY W`. Its weight gradient is never
computed. Both GEMMs can run as int8 × int8 → int32 with ConvRot rotation, using one
existing kernel. On the RTX 5070 Ti that kernel is 2.6x faster than bf16 for both GEMMs
([reports/gemm_microbench.md](reports/gemm_microbench.md)).

The target is a **≥ 1.25x faster training step at equal LoRA quality**, judged on samples
generated from the bf16 base.

Everything else stays in float: attention, norms, softmax, adaLN modulation, the residual
stream, and the LoRA rank GEMMs. "Everything int8/int32" is not the goal.

## Why the scope is narrow

- **The kernel exists.** `torch.ops.comfy_kitchen.int8_linear` (comfy_kitchen 0.2.40,
  already a dependency) fuses the per-256-column Hadamard rotation, per-token int8
  quantization and the CUTLASS int8 GEMM. It is a registered custom op with a fake
  implementation, so `torch.compile` can trace it. It has no autograd.
- **The backward is the same op.** With a second int8 copy of `Wᵀ`, rotated along N and
  scaled per row, `dX = int8_linear(dY, Wᵀ_q, s_t, convrot=True)`. The online rotation
  then acts on dY, which is where gradient outliers sit. Two int8 copies take the same
  memory as one bf16 weight.
- **No dW.** The hard part of int8 training, the weight gradient, does not exist for a
  frozen base. SwitchBack (Wortsman et al., 2023) runs the forward and dX in int8 and
  keeps only dW in 16-bit for this reason.
- **Inputs are not saved for the base.** The base backward needs only W. LoRA still saves
  its own input, as it does today.

## Design: all code in the project

Nothing under `library/` or `networks/` changes. Two files in `src/`:

### `src/int8_linear.py`

- `Int8TrainLinear`, an `nn.Linear` subclass with the bf16 `weight` removed. It holds:
  - `wq` (N, K) int8, rotated along K, with per-row fp32 scale `s` (N, 1);
  - `wq_t` (K, N) int8, rotated along N, with per-row fp32 scale `s_t` (K, 1), only in
    mode `full`;
  - in mode `fwd`, the bf16 `weight` is kept for a bf16 backward instead of `wq_t`.
- Forward and backward go through one `torch.library.custom_op`
  (`int8_train::linear`) that calls the comfy_kitchen op, with `register_fake` and
  `register_autograd`, so compiled blocks see one opaque node per linear.
- `convert_(module, mode)` changes an existing `nn.Linear` **in place** with a `__class__`
  swap. LoRA modules keep a reference to the original module object, so replacing the
  object would orphan them.
- Quantization reuses `library.moss.int8.quantize_weight` (rotate, row scale, round to
  nearest) on the GPU at load. Anima's ~2B parameters take seconds, so v0 writes no
  quantized cache file.

### `src/train_int8.py`

The entry point. It strips its own `--int8 {off,fwd,full}` flag, installs two patches and
calls `library.anima.trainer.cli.main(argv)` with the rest.

- **`LoRANetwork.apply_to`** (`networks/lora_anima/network.py`) is wrapped to convert the
  DiT's linears first and then call the original. This point is after `create_network`,
  so `down_init=weight_svd` and the `grad_svd` sketch still read bf16 weights. It is also
  before LoRA captures `org_forward` and before `compile_blocks`, which preserves the
  compile-after-apply order.
- **`AnimaTrainer._update_metadata`** is wrapped to stamp `ss_int8_base=<mode>` into the
  saved LoRA.
- Launch:
  `make daemon-run ARGS="--stall-timeout 900 project/int8_train/src/train_int8.py --int8 full --method lora --preset default …"`.
  This is a command job, not a `make lora` train job, so `--queue` / pause-release
  semantics are the command-job ones.

### Which linears

| converted | kept bf16 |
|---|---|
| self-attn `qkv_proj`, `output_proj` | cross-attn `kv_proj` (M = 512 context tokens; int8 measured slower) |
| cross-attn `q_proj`, `output_proj` | adaLN modulation (M = batch) |
| MLP `layer1`, `layer2` | `llm_adapter`, embedders, final layer |

### Refused in v0

The entry point exits with an error on these combinations rather than letting them run wrong:

- `blocks_to_swap > 0`: the Anima offloader moves only `module.weight`
  (`library/runtime/offloading.py`), so it would leave `wq` / `wq_t` behind. The default
  16 GB preset does not swap.
- Anything that reads or writes `org_module.weight` after conversion: `backup_weights`,
  `restore_weights` and `pre_calculation` in `LoRANetwork`. The entry point fails if
  they are reached.
- `make turbo`: it is a bespoke loop that does not go through `train.py`.

`make merge` is unaffected because it reads the bf16 checkpoint from disk.

## Phases and gates

**Phase 0, done.** The GEMM microbench ([report](reports/gemm_microbench.md)).

**Phase 1, probe (about half a day, no training).**

- **1a, step split.** Run `torch.profiler` over a few compiled steps at the default
  preset (`lora`, 1024², B=1). Sum the time of the six converted GEMMs, forward and dX,
  then predict the step speedup with phase 0's 2.6x. **Gate: predicted ≥ 1.25x**, else
  stop and write the reason here.
- **1b, real-gradient error.** On a few real batches at σ ∈ {0.1, 0.5, 0.9}, compute the
  LoRA parameter gradient three ways: bf16, mode `fwd`, mode `full`. Compare the int8
  error with minibatch noise, measured as the gradient difference between two different
  batches at the same σ. **Gate: ‖g_int8 − g_bf16‖ < 0.1 · ‖g_batchA − g_batchB‖** per
  module, median over modules, at every σ. Report the worst modules by name. Mode `fwd`
  separates forward error from gradient-quantization error.
  *Revised 2026-10-30:* bf16 itself sits at 0.15–0.20 of noise from the fp32 gradient,
  so this gate fails bf16 too. The gate is now: the int8 gradient's distance from fp32
  is at most 2x bf16's, median over modules, at every σ
  ([reports/grad_error.md](reports/grad_error.md)).

**Phase 2, implement.** The two `src/` files and
`project/int8_train/src/test_int8_linear.py`:

- int8 forward and dX against bf16 on small shapes, within a tolerance;
- autograd through `register_autograd` gives the same LoRA grads as an eager reference;
- a compiled block with converted linears traces with no graph break
  (`torch._dynamo.explain`);
- the saved LoRA has the same keys and shapes as a bf16 run, plus `ss_int8_base`.

**Phase 3, end to end.**

- Step time on the default preset, bf16 against `full`, through the daemon.
- One 25-minute micro A/B on the same data, seed and steps: bf16 against `full`. Compare
  loss curves and fixed-seed samples, generated on the **bf16** base.
- **Kill criteria:** measured step speedup < 1.2x, or samples visibly worse than bf16.

**Phase 4, only if phase 3 passes.** Each one is a separate decision:

- MOSS-Image. Its 5B DiT swaps blocks, so int8 halves the swap traffic too. That needs an
  offloader that moves the int8 tensors.
- The turbo loop: the teacher and fake-score forwards are forward-only.
- An int8 forward path for `inference.py`.

**Phase 5, int8 attention in Triton (opened 2026-10-30).** After phase 3, flash
attention is about 40% of the 433 ms int8 step (51 ms forward, 127 ms backward in the
bf16 profile). No public int8 attention has a backward: comfy_kitchen and SageAttention
1–3 are forward only, and SageBwd (SageAttention3 paper, six of seven matmuls int8,
dO·Vᵀ kept fp16) is unreleased. The kernels live in `src/int8_attn.py`, in Triton, on
FA2's loop. The rule above that own kernels move to a separate repo is dropped for this
line.

- **5a, forward.** int8 QKᵀ (Q per block in the kernel, K mean-removed and per 64
  tokens), with PV in bf16, fp16, fp16-accumulate or int8. Scored on q, k, v captured
  from real training calls (4 self- and 4 cross-attention blocks, σ ∈ {0.1, 0.5, 0.9})
  against fp32 SDPA. **Gate: an arm whose output error is at most 2x FA2's, median per
  kind and σ, and that runs at least 1.3x FA2's forward on the self-attention shape,
  quantization included.** A Triton bf16 arm on the same loop separates the
  Triton-against-CUDA gap from the int8 gain. `src/attn_bench.py` also saves dO for 5b.
  *Result:* passes for self-attention only; cross-attention stays FA2
  ([reports/attn_fwd.md](reports/attn_fwd.md)).
- **5b, backward**, only if 5a passes: recompute QKᵀ with the forward's int8 Q and K
  copy (so the stored logsumexp matches), int8 dV, dQ, dK, fp16 dO·Vᵀ, with int8 dOᵀ,
  Qᵀ, Kᵀ copies made by a prep kernel, since int8 MMA needs the reduction axis
  contiguous. Same gate against FA2's backward on the captured dO.
  *Result:* fails. Recomputing P from the int8 Q and K alone puts dK and dV at
  2.5–3.3x FA2's error, and FA2's own backward on the int8 forward's O and logsumexp
  is 2.5–3.4x; the fastest arm is 1.17x. 5c is not run
  ([reports/attn_bwd.md](reports/attn_bwd.md)).
- **5c, end to end**, only if 5b passes: the op behind `dispatch_attention` in
  `train_int8.py`, phase 1b's gradient gate, then phase 3's A/B.

## Out of scope

- int8 LoRA rank GEMMs: rank 16–64 GEMMs are bandwidth-bound and gain nothing.

## Risks

- **Gradient outliers.** dY at low σ may have outliers that the 256-column rotation does
  not spread enough. Phase 1b measures this directly.
- **Train/inference mismatch.** The LoRA learns against a quantized base and is applied
  to the bf16 base. QLoRA suggests this is usually harmless, and phase 3 samples are
  generated on the bf16 base to check it.
- **Forward error compounding.** About 1.3% per layer over 28 blocks. Phase 1b's mode
  `fwd` measures what reaches the LoRA gradient.
- **Compile.** `compile_dynamic_seq` with a custom op across token bands is untested. The
  phase 2 test covers one band and phase 3 the rest.
