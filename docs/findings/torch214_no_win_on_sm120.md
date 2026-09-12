# torch 2.14 buys nothing on sm_120: NVGEMM has no kernel image, the rest is ±1 % drift

Kill-gate bench for the torch 2.12 → 2.14 bump proposal
(`docs/proposal/torch214.md`), run 2026-09-09 on the training box (RTX 5070 Ti,
consumer Blackwell `sm_120`, driver 610, cu132). Bench + driver:
`bench/torch_bump/`; results `bench/torch_bump/results/20260909-1331-t214/` and
`…-1357-t214-combo/`.

Every arm is a real `train.py --method lora --preset default` run — `path_pattern
mikozin/*` (70 images), 3 epochs = 210 steps, seed 42, `torch_compile` on, a
cold private Inductor + Triton cache per arm — submitted through the daemon
under either `.venv` (2.12.0+cu132, triton 3.7) or a scratch venv (2.14.0+cu132,
triton 3.8, cuDNN 9.24, NCCL 2.30, flash-attn 2.8.3 `torch2.14` prebuild).

| arm | s/it warm (n) | first step, cold / warm cache | peak VRAM used |
|---|---|---|---|
| 2.12 default | 0.588 (2, spread 0.17 %) | 17.1 s / 9.6 s | 13.9 GB |
| 2.14 default | 0.584 (2, spread 0.08 %) — **−0.7 %** | 19.4 s / 10.0 s | 13.9 GB |
| 2.12 max-autotune | 1.032 mean, 0.556 median, p90 2.77 s | 97 s | 15.1 GB |
| 2.14 max-autotune | 0.971 mean, 0.553 median, p90 2.49 s | 113 s | 15.1 GB |
| 2.12 default + `combo_kernels=True` | 0.597 — **+1.5 %** | 16.9 s | 13.9 GB |
| 2.14 default + `combo_kernels=True` | 0.598 — **+2.4 %** | 19.2 s | 13.8 GB |

Loss trajectories agree to the third decimal across all arms; VRAM is identical
within 60 MB.

## 1. NVGEMM (the headline candidate) cannot run on consumer Blackwell

The 2.14 "NVIDIA Universal GEMM with epilogue fusion" is not what the release
note makes it sound like for this box:

- It is **not on by default**: `torch._inductor.config.max_autotune_gemm_backends`
  is `ATEN,TRITON,CPP`; NVGEMM needs
  `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON,NVGEMM` **and** two extra
  packages (`nvidia-cutlass-operators`, `nvidia-matmul-heuristics`) **and**
  `mode="max-autotune"`.
- With all of that in place (`bench/torch_bump/nvgemm_smoke.py`,
  `TORCH_LOGS=autotuning`), the heuristics return 10 configs for a
  4096×1024×3072 bf16 GEMM and **every kernel fails at launch with
  `cudaErrorNoKernelImageForDevice`**; Inductor logs "Ignoring this choice" and
  the autotune subprocess pool then hangs for the rest of the process.
- Root cause is in the package, not torch: `cutlass.operators` 0.2.0 only
  carries sm100-family kernels (`FamilyPortable targets must be Blackwell
  (sm100) or newer`; the sibling `CUTEDSL` backend is documented "SM100-SM109
  only"). Same trap as FA4 (`docs/optimizations/fa4.md`): "Blackwell" in a
  PyTorch/NVIDIA release note means B200, not RTX 50.

## 2. Default-mode drift: −0.7 % s/it, +2 s cold compile

With no flags, everything that changes between 2.12 and 2.14 (Inductor codegen,
triton 3.7 → 3.8, cuDNN 9.20 → 9.24) sums to 4 ms/step on a 0.59 s step. It is
outside the in-batch spread of the repeated arms, so it is real, but it does not
justify carrying two torch majors in the support matrix (2.14 has no cu132
flash-attn prebuild for Windows or cp313 aarch64).

## 3. Combo kernels are off in both versions, and a loss when forced on

The 2.13 note "combo kernel autotuning on by default" changed only the knobs
that apply once combo kernels are enabled — `combo_kernels` itself is `False`
in 2.12 **and** 2.14, so the default arm never exercised them. Forced on via the
repo's `pin_inductor_flag("combo_kernels", True)` (there is no env var), 28 of
the 51 generated kernels were combo-fused and the step got 1.5–2.4 % **slower**
in both versions. Not a lever here.

## 4. `max-autotune` is not a shipping config in either version

The median step is ~5 % faster than default, but the run re-autotunes mid-run
(30 recompile log lines, p90 step 2.5–2.8 s vs 0.6 s), so the mean is ~1.7×
worse than default, plus 97–113 s cold compile and +1.2 GB VRAM. This also moots
the `precompile` / `mark_unbacked` candidate: the cold compile it would save is
17 s and the warm-cache load is under 10 s.

## Verdict

**CLOSED — no performance reason to bump to 2.14.** If a bump is wanted for
hygiene, 2.13 everywhere (fully prebuilt on every platform) remains the option;
its only substantive item is `isolate_recompiles=True`, an engineering cleanup
for the recompile-limit juggling, not a speed win. `make test-unit` (not-slow
suite) under 2.14 passed 1592/1593; the one failure is the pre-existing
`CLAUDE.md` placeholder path caught by `test_doc_refs`, unrelated to torch.

Trap for next time: check `max_autotune_gemm_backends` and the backend's
arch gate (`grep sm100 site-packages/cutlass/operators`) before spending GPU
hours on an "auto" Inductor backend — a 10-minute `TORCH_LOGS=autotuning`
GEMM smoke settles it.
