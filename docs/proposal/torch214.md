# torch 2.12 → 2.13 / 2.14 bump — candidates and cost

Status: **BENCHED 2026-09-09 — no step-time win on sm_120; 2.14 line closed.**
See "Bench verdict" at the bottom. Wheel availability and dependency drift
verified against the live indexes on 2026-09-09; repo-side risks verified by
grep (cited inline). Release-note claims are quoted; the bench measured them.

## TL;DR

- **CUDA 13.3 is not an option.** PyTorch ships no stable `cu133`/`cu134`
  index (both 403, zero wheels); only 2.15 nightlies carry `+cu134`. Stay on
  `cu132`. Note the cu132 wheels already bundle `nvidia-cublas 13.4.0.1` (2.12,
  2.13 and 2.14 alike) — the toolkit tag lags the bundled cuBLAS.
- **torch 2.13 is the fully covered bump** — flash-attn 2.8.3 prebuilds exist
  for cu132 on every platform we ship (Linux x86_64, Linux aarch64, Windows).
  Payload for this repo is small: `isolate_recompiles`, combo-kernel autotune
  on by default, deterministic flex backward.
- **torch 2.14 is Linux-x86_64-only for now** — no cu13x flash-attn prebuild
  for Windows (cu126 only) or for cp313 aarch64 (cu130 only). It carries the
  candidates actually worth a bench: Inductor NVGEMM epilogue fusion and
  `precompile` + `mark_unbacked` for the free-fit band cascade.
- Recommended path: bench 2.14 in a scratch venv on the training box; if
  nothing measurable, ship 2.13 everywhere (cheap, uniform) and revisit 2.14
  when the Windows cu132 wheel appears.

## Current pins

`pyproject.toml`: Linux `torch>=2.12.0,<2.13` / `torchvision>=0.27,<0.28`,
Windows `cuda-windows` group `torch==2.12.0+cu132`, `rocm-windows` group
`torch==2.13.0+rocm10.0.0` (already on 2.13). flash-attn 2.8.3 from
`mjun0812/flash-attention-prebuild-wheels` per platform. Lock: triton 3.7.0,
cuDNN 9.20.0.48, NCCL 2.29.7. Python is pinned `==3.13.*`.

Training box: RTX 5070 Ti (sm_120, consumer Blackwell), driver 610.43, nvcc
13.2.

## Wheel matrix (cp313, cu132, flash-attn 2.8.3)

| Platform | 2.12 (today) | 2.13 | 2.14 |
|---|---|---|---|
| torch on `whl/cu132` | 2.12.0 | 2.13.0 (+ torchvision 0.28.0) | 2.14.0 (+ torchvision 0.29.0) |
| flash-attn linux x86_64 | v0.9.17 | v0.9.47 `flash_attn-2.8.3+cu132torch2.13-cp313-cp313-linux_x86_64.whl` | v0.10.0 `flash_attn-2.8.3+cu132torch2.14-cp313-cp313-linux_x86_64.whl` |
| flash-attn linux aarch64 | v0.9.22 | v0.9.49 `flash_attn-2.8.3+cu132torch2.13-cp313-cp313-linux_aarch64.whl` | **none** (cu130 only; cu132 exists for cp312/cp314) |
| flash-attn win_amd64 | v0.9.25 | v0.9.52 `flash_attn-2.8.3+cu132torch2.13-cp313-cp313-win_amd64.whl` | **none** (cu126 only — useless on Blackwell) |

URL pattern:
`https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/<tag>/<wheel>`.

Resolved dependency drift (`uv pip compile` against the cu132 index):

| | 2.12 | 2.13 | 2.14 |
|---|---|---|---|
| triton | 3.7.0 | 3.7.1 | **3.8.0** |
| cuDNN | 9.20.0.48 | 9.20.0.48 | **9.24.0.43** |
| NCCL | 2.29.7 | 2.29.7 | 2.30.7 |
| cuBLAS | 13.4.0.1 | 13.4.0.1 | 13.4.0.1 |
| CUDA runtime | 13.2.75 | 13.2.75 | 13.2.75 |

2.14 also changes the `triton-windows` story (`>=3.7,<3.8` pinned for the
ROCm/CUDA Windows groups) — irrelevant while Windows stays on ≤2.13.

## Candidates — what could actually move a number here

Ordered by expected payoff for this repo. "Auto" = lands without code change.

### 2.14

1. **Inductor NVGEMM (CuTeDSL-generated CUTLASS GEMMs) with epilogue fusion**
   — "epilogue fusion, scaled and NVFP4 GEMM, and grouped-reduction epilogues
   autotuned alongside Triton and ATen"; "pointwise operations and output casts
   fuse into autotuned matrix multiplications; grouped GEMM support on
   Hopper/Blackwell". **Not auto** (see verdict): NVGEMM is absent from the
   default `max_autotune_gemm_backends` (`ATEN,TRITON,CPP`) and needs two extra
   packages (`nvidia-cutlass-operators`, `nvidia-matmul-heuristics`). The LoRA
   forward is a pile of small GEMMs followed by pointwise scale/add, so this is
   the one throughput candidate. Open question: whether sm_120 (consumer
   Blackwell) is in the tuned set — the notes say Hopper/Blackwell and add
   Rubin `sm_107`, nothing about `sm_120`. FA4's SM120 story
   (`docs/optimizations/fa4.md`) is the cautionary precedent: "Blackwell" in a
   release note has meant B200, not RTX 50.
2. **`torch.compiler.precompile` + `mark_unbacked`** — "allowing one artifact to
   serve multiple runtime sizes without guarding". This is the free-fit band
   cascade: `compile_blocks` pins `recompile_limit = 2n+8` and traces one graph
   per band (`library/anima/models.py:1595-1663`), `build_anima` sizes the cache
   to `2*n_shapes+8` (`library/runtime/harness.py:634-686`). A single unbacked
   artifact would cut compile wall (not step time) and remove the per-band
   graph budget the `bucketing` skill has to reason about. Not auto — needs a
   code path.
3. **Unified `dynamic_shapes=` / `@dynamic_spec` on `torch.compile`** — could
   replace the eager `mark_dynamic` prologue wrapper (`library/anima/models.py:
   101-137`, `networks/methods/easycontrol.py:1505-1510`, register's widened
   bound at `networks/methods/register.py:219`). Simplification only; do it
   after (2) if (2) pans out, since `precompile` may make both moot.
4. **Per-region Inductor config patches, separate fwd/bwd** — "apply per-region
   Inductor configuration patches throughout nested-region compilation and
   allow separate forward and backward patches". Lets `pin_inductor_config`
   (`library/runtime/dynamo.py:61`) stop being process-global; matters for the
   bespoke distill loops that compile with different knobs than `train.py`.
5. **`torch.cuda.use_mem_pool` inside compiled regions** — possible lever for
   block swap (`blocks_to_swap`) allocator churn. Speculative.
6. **FlexAttention flash-backend backward through LSE; memory-efficient GQA
   under `vmap`** — only if the `flex` attention mode
   (`networks/attention_dispatch.py:195`) is in use. It is not the default.
7. Not relevant: cuDNN SDPA head_dim 256 (Anima is head_dim 128,
   `library/anima/models.py:294`, and FA2 is the backend); native bf16 FFT (both
   FFT sites already upcast — `cns_core.py:153`, `spectrum_sea.py:88`);
   cuBLASLt grouped GEMM (needs explicit `_grouped_mm` calls; Hydra/Chimera
   loop over experts); Python 3.15 wheels (`torch.compile` unsupported there
   anyway); NCCL2 backend / fault tolerance (single GPU).

### 2.13

1. **`torch.compile(isolate_recompiles=True)`** — "isolated cache buckets per
   call". Directly addresses the flex pre-compile / recompile-limit exhaustion
   documented in `docs/optimizations/for_compile.md` §1.2 and the EasyControl
   `accumulated_recompile_limit` juggling (`easycontrol.py:679-687`). Not auto.
2. **Combo kernel autotuning on by default** — auto; small pointwise-fusion
   win, could also be a compile-time regression. Measure compile wall.
3. **CuTeDSL backend (prototype) — "faster compilation"** — precursor to
   2.14's NVGEMM. Prototype; not something to enable in a shipped config.
4. **Deterministic FlexAttention backward on CUDA** — only under `flex`.
5. **`nn.LinearCrossEntropyLoss`**, FSDP2 overlap, torchcomms, MPS work — not
   applicable.

## Breakage checklist (grep-verified 2026-09-09)

| Change | Where it would bite | Status |
|---|---|---|
| 2.13: named tensors removed (`Tensor.names`, `refine_names`, `align_to`…) | — | clean; the only `.names` hits are a dataclass in `library/anima/merge_analysis.py` |
| 2.13: `all_gather_into_tensor` → `all_gather_single`, `reduce_scatter_tensor` → `reduce_scatter_single` (FutureWarning) | accelerate internals, not us | clean in-tree |
| 2.13: `quint8`/`qint8`/`qint32` creation deprecated | INT8 paths in the ComfyUI nodes | clean in-tree; node `_vendor/` not checked |
| 2.13: custom ops returning input-aliasing outputs warn | none registered | clean |
| 2.14: `torch.cholesky` / `torch.qr` removed | `networks/spectrum_forecast.py:134` uses `cholesky_solve` | clean — `cholesky_solve` stays |
| 2.14: profiler `use_cuda=` removed | — | clean |
| 2.14: bf16 × complex promotes to `bcomplex32`, not `complex64` | FFT paths | clean — `cns_core.py:153` and `spectrum_sea.py:88` upcast to float32 first |
| 2.14: `clamp` subgradient at boundaries with tensor bounds now splits evenly | losses with tensor-valued `clamp` bounds | no hits in `library/training` |
| 2.14: selective activation checkpointing will honor `saved_tensors_hooks` (FutureWarning now) | gradient checkpointing + block swap ([[project_blockswap_extra_forwards_gradcache]]) | **watch for the warning; behavior flips in a later release** |
| 2.14: triton 3.7 → 3.8 | `pin_dynamo_limit` alias resolution (`library/runtime/dynamo.py:36-47`), Triton kernel cache | re-verify aliases; expect a cold cache |
| 2.14: `CUDAGraph.register_generator_state` deprecated | `reduce-overhead` mode + `_last_gate` note (`networks/lora_anima/network.py:1322`) | warning only |
| 2.13 known issue: ROCm wheel without GPU breaks CPU `torch.compile` | `rocm-windows` group is already on 2.13.0 | pre-existing, not new |

Two majors of drift (2.12 → 2.14) land at once; the 2.13 rows apply to a 2.14
bump too.

## Cost

- **2.13 everywhere**: bump three lines per platform, `uv lock`, `uv sync
  --frozen`, `make test-unit`, one short compiled `make lora` to confirm no
  recompile regression. ~4 GB download, no code change, all platforms.
- **2.14 Linux-only**: same on the Linux lines; Windows `cuda-windows` group
  stays at 2.12 (or moves to 2.13), so two torch majors in the support matrix
  until the Windows cu132 prebuild lands. Candidates (1)–(2) need a bench each
  before any of the code-side items (2)–(4) are worth writing.

## Trial recipe (2.14, scratch venv, reversible)

1. Branch; set Linux `torch>=2.14.0,<2.15`, `torchvision>=0.29.0,<0.30`, swap
   the x86_64 flash-attn URL to the v0.10.0 wheel above; leave aarch64/Windows
   lines untouched (the aarch64 line will fail to resolve — mark it
   `platform_machine == 'aarch64'`-only as it already is and accept that
   aarch64 stays on 2.12 for the trial).
2. `uv lock`, then `UV_PROJECT_ENVIRONMENT=.venv-t214 uv sync --frozen` — does
   not touch `.venv` ([[project_uv_run_resolve_broken]]: use the venv's python
   directly, never bare `uv run`).
3. `make test-unit` from `.venv-t214`.
4. Through the daemon, 200-step `make lora` with `torch_compile` on, from each
   venv, same seed/dataset; compare s/it after warm-up, first-step compile
   wall, graph count in the log line `dynamic-seq per-band mark_dynamic`, peak
   VRAM. Then the same with `--compile_mode max-autotune` to give NVGEMM a
   chance to be selected; check `TORCH_LOGS=inductor` for which GEMM backend
   won.
5. Verdict rule: ship 2.14 on Linux only if (4) shows a step-time win beyond
   in-batch noise ([[project_deterministic_flag_chaos_floor]]); otherwise ship
   2.13 everywhere and file (2)/(4) above as follow-ups gated on the Windows
   wheel.

## CUDA 13.3: cuTile C++/Python, CompileIQ, CUDA Python 1.0 — not for us yet

The 13.3 launch post is about *authoring* kernels; this repo authors none
(zero `@triton.jit`, no `cpp_extension`, grep 2026-09-09). Every kernel we run
comes from torch/Inductor, flash-attn, or a library, so a new kernel DSL only
matters through one of three doors:

- **Via torch wheels** — closed. No stable `cu133` wheel exists and the 2.15
  nightlies are `+cu134`, so 13.3's cuBLAS/cuSOLVER work reaches us only when
  torch ships a cu134 build. **CompileIQ** (evolutionary nvcc flag search,
  "up to 15%" on GEMM/attention) tunes kernels *you* compile; torch's are
  compiled by PyTorch CI and Triton's by ptxas, so it does not touch our hot
  path at all.
- **Via Inductor** — this is the real door, and it is already in the table
  above: NVIDIA's tile kernels enter torch as the **CuTeDSL backend** (2.13
  prototype → 2.14 NVGEMM). Inductor emits Triton, not cuTile; nothing
  suggests a cuTile backend is coming.
- **Hand-written cuTile kernel** — runnable on the box today (cuTile Python
  requires "compute capability 8.x, 9.x, 10.x, 11.x or 12.x", driver r580+,
  CUDA 13.1+; we have sm_120, driver 610, 13.2), but it re-creates the FA4
  postmortem: an opaque custom op under `compile_blocks` means
  `@torch.compiler.disable` and 28×2 graph breaks per step, and the only
  kernel worth the effort (4096-token self-attention) is exactly the one where
  the SM120 port lost to FA2. Not a candidate.

**CUDA Python 1.0** (`cuda.core`) has two items worth one line each for the
daemon, not the trainer: **green contexts** (SM partitioning — could let the
resident inference server and a training job share the GPU deterministically;
torch 2.14 already exposes its own and deprecates `GreenContext.set_context`
in favour of streams) and **process checkpointing** ("snapshot the full CUDA
state", Linux-only — a real pause/preempt primitive for the queue, cf.
[[project_daemon_gotchas]] "pause is untrusted"). Both are speculative and
independent of the torch bump.

## Sources

- https://github.com/pytorch/pytorch/releases/tag/v2.14.0
- https://github.com/pytorch/pytorch/releases/tag/v2.13.0
- https://github.com/mjun0812/flash-attention-prebuild-wheels (`doc/packages.md`)
- https://download.pytorch.org/whl/cu132/ (cu133 / cu134: 403, no stable wheels)

## Bench verdict (2026-09-09, `bench/torch_bump/`)

Scratch venv `.venv-t214` (torch 2.14.0+cu132, torchvision 0.29, triton 3.8.0,
cuDNN 9.24, NCCL 2.30, flash-attn 2.8.3 `torch2.14` prebuild) built from the
lock without touching `pyproject.toml`; `make test-unit` (not-slow suite) under
it: 1592 passed, the one failure is the pre-existing `CLAUDE.md` placeholder
path caught by `test_doc_refs`, unrelated to torch. Every arm is a real
`train.py --method lora --preset default` run (`path_pattern mikozin/*`, 70
images, 3 epochs = 210 steps, seed 42, `torch_compile` on, cold private
Inductor + Triton cache per arm) submitted through the daemon. Results:
`bench/torch_bump/results/20260909-1331-t214/` and `…-1357-t214-combo/`.

| arm | s/it warm (n) | first step cold / warm cache | peak VRAM used |
|---|---|---|---|
| 2.12 default | 0.588 (2, spread 0.17 %) | 17.1 s / 9.6 s | 13.9 GB |
| 2.14 default | **0.584** (2, spread 0.08 %), −0.7 % | 19.4 s / 10.0 s | 13.9 GB |
| 2.12 max-autotune | 1.032 mean, 0.556 median, p90 2.77 | 97 s | 15.1 GB |
| 2.14 max-autotune | 0.971 mean, 0.553 median, p90 2.49 | 113 s | 15.1 GB |
| 2.12 default + `combo_kernels=True` | 0.597 (+1.5 %) | 16.9 s | 13.9 GB |
| 2.14 default + `combo_kernels=True` | 0.598 (+2.4 %) | 19.2 s | 13.8 GB |

- **Candidate (1) NVGEMM is dead on consumer Blackwell.** With the backend
  enabled (`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON,NVGEMM`) the
  nvMatmulHeuristics return 10 configs for a 4096×1024×3072 bf16 GEMM, every
  one fails at launch with `cudaErrorNoKernelImageForDevice`, and the autotune
  subprocess pool then hangs (`bench/torch_bump/nvgemm_smoke.py`, log in the
  results dir). `cutlass.operators` 0.2.0 only carries sm100-family kernels
  (`FamilyPortable targets must be Blackwell (sm100) or newer`). Same story as
  FA4: "Blackwell" in the notes means B200, not RTX 50.
- **Default-mode drift is −0.7 % s/it** (0.004 s/step), outside the in-batch
  spread but not worth a two-major bump on its own; cold compile is +2.3 s.
  Loss trajectories match to the third decimal, VRAM identical.
- **Combo kernels** (`combo_kernels` is `False` by default in *both* versions —
  the 2.13 "autotune on by default" note only touched knobs that apply once
  they are on) are a 1.5–2.4 % step-time **loss** here: 28 of 51 generated
  kernels were combo-fused and the fused launches are slower than the
  separate ones on this GPU. Not a lever in either version.
- **max-autotune is not a shipping config in either version**: the median
  step is ~5 % faster but it re-autotunes mid-run (30 recompile log lines,
  p90 2.5–2.8 s), so the mean is ~1.7× worse, and cold compile is 97–113 s.
- Candidate (2) (`precompile` / `mark_unbacked`) is moot: the cold compile it
  would save is 17 s and the warm-cache load is under 10 s.

**Decision:** do not bump to 2.14 for performance. If a bump is wanted for
hygiene, 2.13 everywhere remains the uniform, fully-prebuilt option; its only
substantive item is `isolate_recompiles=True` (a code change for the
recompile-limit juggling, not a speed win). Finding:
`docs/findings/torch214_no_win_on_sm120.md`. The scratch venv was deleted after
the bench; `bench/torch_bump/README.md` has the recipe to rebuild it.
