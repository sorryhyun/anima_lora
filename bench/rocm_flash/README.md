# Windows ROCm FlashAttention comparison

This compares the previous SDPA path (`attn_mode="torch"`) with the rebuilt
native ROCm FlashAttention path (`attn_mode="flash"`) through Anima's actual
attention dispatcher. Both use identical seeded BF16 tensors, B=1, H=16,
D=128. It records synchronized forward+backward wall time and peak PyTorch
allocated memory after warmup, and checks outputs and all three gradients.

Run after installing the `rocm-windows` group:

```powershell
uv run --no-sync python tasks.py daemon-run -- bench/rocm_flash/run_bench.py --device cuda:0 --seq-len 1024 --label gfx1200
```

Use `--device cuda:1` (or device visibility) for another GPU. Run the GPUs
serially; do not launch overlapping benchmarks. No model or dataset download
is required. This measures attention kernels, not complete training steps or
model quality. It does not claim bitwise equality between attention backends.

Results are written to `bench/rocm_flash/results/<timestamp>-<label>/result.json`
using the shared benchmark envelope. `median_forward_backward_ms` includes
dispatch and backward, while `samples_ms` preserves every measured step.
Differences use `(rtol, atol)=(0.06, 0.015)`; a failed comparison or non-finite
gradient aborts the run. The installer smoke test separately checks against
CPU/FP64 references across all compiled head dimensions and tail lengths.

## Baseline (2026-10-16)

Python 3.13.13, torch 2.14.0+rocm10.1.0, native tail-fix build; 10 warmup
iterations followed by 30 measured iterations, SDPA then FlashAttention.
Clocks were not locked. These are attention-only milliseconds, not training
step times; the benchmark does not enable torch.compile.

| GPU | Tokens | SDPA median | Flash median |
|---|---:|---:|---:|
| RX 9060 XT | 1024 | 8.311 | 1.706 |
| RX 9060 XT | 4096 | 124.089 | 16.947 |
| RX 9070 XT | 1024 | 0.980 | 1.126 |
| RX 9070 XT | 4096 | 8.518 | 8.854 |

Flash is **not uniformly faster**: SDPA is slightly faster on RX 9070 XT at
these shapes. Outputs and gradients passed the comparison at both lengths.
Use `--seq-len 4096` for the longer case and `--label` for distinct run dirs.
Raw samples and hardware scope accompany the
[custom build release](https://github.com/qweqweewqe7-create/flash-attention/releases/tag/windows-rocm101-torch214-gfx12-20261016).
