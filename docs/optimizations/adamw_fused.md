# AdamW8bit → fused AdamW

This doc records why `optimizer_type = "AdamW8bit"` was replaced with `"AdamW"` + `optimizer_args = ["fused=True"]` as the project default, and why `bitsandbytes` was dropped from `pyproject.toml`.

## What was there

`configs/base.toml` defaulted to `optimizer_type = "AdamW8bit"`, which routes through `library/training/optimizers.py` to `bitsandbytes.optim.AdamW8bit`. State (m, v) was stored in 8-bit blockwise quantized form; on every step bnb dequantizes a block, runs the AdamW math, and re-quantizes. State VRAM cost ≈ 2 bytes/param.

## Why it was removed

A `--profile_steps` chrome trace under plain LoRA training, analyzed with `scripts/analyze_profile_gaps.py`, showed a clean 11–12 percentage-point GPU-utilization drop coinciding exactly with the optimizer phase of every step.

| Phase | Duration | GPU util |
|---|---|---|
| forward + backward + everything pre-optimizer | ~860 ms | **99.0%** |
| `Optimizer.step#AdamW8bit.step` | ~308 ms | **88.0%** |
| **Total step** | **~1166 ms** | — |

Inside the 308 ms optimizer phase, the trace shows ~50 tiny `kOptimizerStatic8bit2StateBlockwise` kernels back-to-back, each ~8–30 µs, with kernel-to-kernel gaps of ~6 ms. bnb dispatches one launch per parameter block. The CPU can't refill the launch queue fast enough between micro-kernels, so the GPU sits at ~88 % across the entire optimizer phase. This was the source of the "GPU goes 100 % → 80 % every ~8 steps" symptom — actually every step, but step time is ~1.17 s and aliases against `nvidia-smi`'s 1 Hz refresh.

The 308 ms optimizer plateau was the dominant idle source — `loss.item()` syncs and `accelerator.unwrap_model` overhead at the step boundary added another ~1–3 ms per step but were rounding error compared to the optimizer phase.

## What we switched to

```toml
# configs/base.toml
optimizer_type = "AdamW"
optimizer_args = ["fused=True"]
```

`torch.optim.AdamW(..., fused=True)` runs the entire optimizer step as a single fused CUDA kernel — no per-block dispatch, no quantize/dequantize round-trip. The 308 ms plateau collapses to a few milliseconds.

`fused=True` is parsed by `get_optimizer` via `ast.literal_eval`, which only accepts Python literals — must be capitalized (`"fused=True"`, not `"fused=true"`). Lowercase will raise.

## Trade-off

- **VRAM**: state grows from ~2 bytes/param (bnb 8-bit) to 8 bytes/param (fp32 m + v). For LoRA-only training with `network_train_unet_only=true` and dim=32 (~30 M trainable params), that's about **+180 MB** — fits comfortably on a 16 GB card. For a full-DiT train it would be a few GB extra and may not fit. None of the shipped configs hit that case.
- **Speed**: step time drops from ~1.17 s to ~0.86 s on the same RTX 5060 Ti / FA2 / static-token-pad setup that produced the [cuda132.md](cuda132.md) numbers. GPU utilization sits flat at ~99 %.
- **Numerics**: full-precision optimizer state. No quantization noise on the second moment, so anything sensitive to that (very small LR, long training runs accumulating error) will be slightly more stable. Not a measurable quality difference at our typical scales.

## bitsandbytes is gone

`bitsandbytes` was removed from `[project.dependencies]` in `pyproject.toml`. The bnb-using optimizer branches in `library/training/optimizers.py` (`AdamW8bit`, `Lion8bit`, `SGDNesterov8bit`, `PagedAdamW`, `PagedAdamW8bit`, `PagedAdamW32bit`, `PagedLion8bit`) were left in place — every one of them lazy-imports `bitsandbytes` inside the branch and raises a clear `ImportError("No bitsandbytes")` if you select that optimizer without the package installed. So:

- A leftover config with `optimizer_type = "AdamW8bit"` produces a friendly install hint, not a silent fallback.
- If you need 8-bit state back (say, to free VRAM on a smaller card), `uv pip install bitsandbytes` and the existing branches just work — no code change required.

## If you want to bring AdamW8bit back

1. `uv pip install bitsandbytes`
2. In `configs/base.toml`, swap:
   ```toml
   optimizer_type = "AdamW8bit"
   # remove optimizer_args = ["fused=True"]  — bnb doesn't accept it
   ```
3. Optionally re-add `"bitsandbytes"` to `pyproject.toml` if you want it as a hard dependency again.

Expect the GPU-utilization dip to come back. If 8-bit state is genuinely needed for VRAM (full-DiT train on 16 GB), the dip is worth the trade. For LoRA, fused AdamW is strictly better.
