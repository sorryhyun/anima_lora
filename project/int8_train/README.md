# int8_train — int8 GEMMs for Anima LoRA training

Opened 2026-10-30. The frozen DiT's large linears run as ConvRot int8 × int8 → int32, in
the forward and in the backward input-gradient GEMM, through comfy_kitchen's existing
kernel. Anima first. Anima's code lives in `src/`: the line patches the trainer from its
own entry point and does not change `library/anima/` or `networks/`. MOSS training moved to
`library/moss/dit_int8/` (`TrainRequest.int8`), the linears with it; `src/int8_linear.py`
keeps Anima's half and re-exports them.

- **Completed work and its results:** [reports.md](reports.md). In short, the int8 base
  trains 1.37x faster than bf16 with matching loss and samples; no int8 attention arm in
  Triton has passed the accuracy gate.
- **Current task:** [tasks.md](tasks.md).
- **Where a step's time goes, bf16 against int8:** [status.md](status.md).

## Why the scope is narrow

LoRA training freezes the DiT, so each base linear needs two GEMMs per step: the forward
`Y = X Wᵀ` and the input gradient `dX = dY W`. The weight gradient, the hard part of int8
training, is never computed (SwitchBack, Wortsman et al., 2023, keeps only dW in 16-bit
for this reason). Both GEMMs run as int8 with ConvRot rotation through
`torch.ops.comfy_kitchen.int8_linear` (comfy_kitchen 0.2.40, already a dependency). It
fuses the per-256-column Hadamard rotation, per-token quantization and the CUTLASS int8
GEMM, and has a fake implementation, so `torch.compile` traces it.

- **The backward is the same op.** With a second int8 copy of `Wᵀ`, rotated along N and
  scaled per row, `dX = int8_linear(dY, Wᵀ_q, s_t, convrot=True)`. The online rotation
  acts on dY, where gradient outliers sit. Two int8 copies take the memory of one bf16
  weight.
- **Inputs are not saved for the base.** Its backward needs only W. LoRA still saves its
  own input.
- Attention, norms, softmax, adaLN modulation, the residual stream and the LoRA rank
  GEMMs stay in float. The rank 16–64 GEMMs are bandwidth-bound and gain nothing in int8.

## Design

### `src/int8_linear.py`

- `Int8TrainLinear`, an `nn.Linear` subclass with the bf16 `weight` removed. It holds
  `wq` (N, K) int8 rotated along K with per-row fp32 scale `s`, and, in mode `full`,
  `wq_t` (K, N) int8 rotated along N with per-row scale `s_t`. Mode `fwd` keeps the bf16
  `weight` for a bf16 backward instead of `wq_t`; mode `infer` keeps the forward copy
  only.
- Forward and backward go through one `torch.library.custom_op` (`int8_train::linear`)
  with `register_fake` and `register_autograd`, so a compiled block sees one opaque node
  per linear.
- `convert_(module, mode)` changes an `nn.Linear` **in place** with a `__class__` swap,
  because LoRA modules keep a reference to the original module object.
- Quantization reuses `library.moss.int8.quantize_weight` (rotate, row scale, round to
  nearest) on the GPU at load, or reads a GPTQ file (`--int8_file`).
- `Int8TrainLinear._apply` keeps the row scales fp32 through the trainer's
  `unet.to(dtype=bf16)`.

### `src/train_int8.py`

Strips its own `--int8 {off,fwd,full}` flag, installs two patches and calls
`library.anima.trainer.cli.main(argv)` with the rest.

- **`LoRANetwork.apply_to`** is wrapped to convert the DiT's linears first. This is after
  `create_network`, so `down_init=weight_svd` and the `grad_svd` sketch still read bf16
  weights, and before LoRA captures `org_forward` and before `compile_blocks`.
- **`AnimaTrainer._update_metadata`** is wrapped to stamp `ss_int8_base=<mode>` into the
  saved LoRA.
- It is a command job, not a `make lora` train job, so `--queue` / pause-release follow
  command-job semantics.

### Which linears

| converted | kept bf16 |
|---|---|
| self-attn `qkv_proj`, `output_proj` | cross-attn `kv_proj` (M = 512 context tokens; int8 measured slower) |
| cross-attn `q_proj`, `output_proj` | adaLN modulation (M = batch) |
| MLP `layer1`, `layer2` | `llm_adapter`, embedders, final layer |

### Refused

The entry point exits on these rather than running them wrong:

- `blocks_to_swap > 0`: the Anima offloader moves only `module.weight`
  (`library/runtime/offloading.py`) and would leave `wq` / `wq_t` behind.
- Anything that reads or writes `org_module.weight` after conversion: `backup_weights`,
  `restore_weights` and `pre_calculation` in `LoRANetwork`.
- `make turbo`: its bespoke loop does not go through `train.py`.

`make merge` is unaffected: it reads the bf16 checkpoint from disk.

## Layout

- `reports.md`: what is done, with links to the per-bench reports.
- `tasks.md`: the current task.
- `status.md`: per-kernel step breakdown, bf16 against int8.
- `gptq_offline.md`: the GPTQ plan, carried out in [reports/gptq.md](reports/gptq.md).
- `src/gemm_bench.py`: GEMM microbench, run as
  `make daemon-run ARGS="project/int8_train/src/gemm_bench.py 4096"`.
- `src/profile_step.py`: step profile of the stock trainer (argv after `--`).
- `src/profile_step_moss.py`: the same for the MOSS trainer (TrainRequest argv after `--`).
  `src/gemm_bench.py <M> moss` benches MOSS's block shapes.
- `src/int8_linear.py`: the int8 custom ops and the in-place `convert_`.
- `src/grad_error.py`: sketched LoRA-gradient error, bf16 vs `fwd` vs `full`.
- `src/grad_error_moss.py`: the same for MOSS (per-call quantization, two-sided sketch).
- `src/train_int8.py`: the trainer on an int8 base, run as
  `make daemon-run ARGS="--stall-timeout 900 project/int8_train/src/train_int8.py --int8 full --method lora --preset default"`.
- `src/infer_int8.py`: `inference.py` on the int8 base, LoRAs as hooks, run as
  `make daemon-run ARGS="project/int8_train/src/infer_int8.py --int8 on <inference args>"`.
- `src/dit_gptq.py`: offline GPTQ of the forward copies, writes a ComfyUI-layout int8 DiT
  (`models/diffusion_models/anima-base-v1.0_int8_gptq.safetensors`, on the NVMe). `train_int8.py` and
  `infer_int8.py` take it as `--int8_file`.
- `src/dit_gptq_moss.py`: the same for MOSS-Image, t2i and i2i
  (`models/moss_image/moss_image_base_int8_gptq.safetensors`, published as
  `sorryhyun/MOSS-Image-Base-int8`). `TrainRequest.int8_file`, `infer_int8_moss.py` and
  `grad_error_moss.py` take it.
- `library/moss/dit_int8/`: `scripts/moss/train.py --int8 stream` is `full` with `wq_t` on
  the host, loaded block by block during the backward (`stream.WqtStream`,
  [report](reports/moss_wqt_stream.md)).
- `src/infer_int8_moss.py`: `library.moss.generate` on the int8 base, run as
  `make daemon-run ARGS="project/int8_train/src/infer_int8_moss.py --int8 on --int8_file <file> -- <generate args>"`.
- `src/grad_spike.py`: traces one image's gradient outlier by block and token.
- `src/traj_eval.py`: RTN against a GPTQ file on bf16 sampling trajectories and held-out
  real images.
- `src/int8_attn.py`: Triton attention, int8 QKᵀ and the PV variants, and the backward
  (`attn_fwd_train`, `attn_bwd`).
- `src/attn_bench.py`: attention capture (q, k, v, dO from real calls, saved to
  `output/int8_train/attn_capture.pt`) and the forward error and speed.
- `src/attn_bwd_bench.py`: dQ, dK, dV error and backward speed on the capture.
- `src/attn_fwd_grad_bench.py`: forward arms (kernels and torch emulations of the int8
  rounding) scored on dQ, dK, dV through a 16-bit backward.
- `src/test_int8_attn.py`: attention tests (`INT8_ATTN_FIXED=1` skips the autotune sweep).
- `tests/test_moss_int8.py`: linear, stream and plan tests (`ANIMA_TEST_GPU=1
  .venv/bin/python -m pytest tests/test_moss_int8.py`); `src/test_int8_linear.py`: the
  split-emulation test that needs `grad_error.py`.
- `src/ab_prompts.txt`: A/B sample prompts.
- `reports/`: one report per bench.
- `assets/`: the raw JSON each bench writes, linked from its report.
