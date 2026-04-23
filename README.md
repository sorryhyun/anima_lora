# anima_lora

[한국어](README.ko.md) · 📖 [가이드북 (Windows 초보자용 한국어 종합 가이드)](docs/guidelines/가이드북.md)

LoRA / T-LoRA training and inference engine for the [Anima](https://huggingface.co/circlestone-labs/Anima) diffusion model (DiT-based, flow-matching).

Three things this repo aims to do well:

1. **Fast LoRA training** on consumer GPUs — compile-friendly data pipeline tuned end to end.
2. **Verified merge-compatible variants** — LoRA, OrthoLoRA, and T-LoRA stack together and bake into a standalone DiT checkpoint.
3. **A broad experimental surface** — HydraLoRA, ReFT, APEX distillation, postfix/prefix tuning, embedding inversion, img2emb, Spectrum inference, mod guidance.

> **At-a-glance diagrams** for every method (DiT internals, LoRA, OrthoLoRA, T-LoRA, HydraLoRA, ReFT, Spectrum, modulation, compile optimizations) live in [`docs/structure_images/`](docs/structure_images/) — paired with prose walkthroughs in [`docs/structure/`](docs/structure/).

---

## 1. Fast training

**15.2 GB peak VRAM · 1.3 s/step** on a single RTX 5060 Ti — achieved by co-designing the data pipeline, attention, and compiler stack so Dynamo sees one static shape for the whole run.

| Lever | Summary |
|---|---|
| Constant-token bucketing | All buckets target `(H/16)×(W/16) ≈ 4096` patches; batches zero-pad to exactly 4096. One static shape, no compile recompilation. |
| Max-padded text encoder | Text outputs padded to 512 and zero-filled — the pretrained DiT uses zero keys as cross-attn sinks, so trimming breaks it. Also gives the compiler another fixed dim. |
| Per-block `torch.compile` | Each DiT block compiled independently with Inductor. Combined with static tokens this eliminates guard recompilation. |
| Compile-friendly hot path | Audited every forward for patterns dynamo can't trace cleanly — `einops.rearrange` replaced with explicit `.unflatten()/.permute()` chains, `torch.autocast` context managers replaced with direct `.to(dtype)` casts, dict `.items()` loops hoisted out of compiled regions, FA4 wrapped in `@torch.compiler.disable` for clean graph breaks. |
| Flash Attention 2 | `flash_attn` 2.x with SDPA fallback. FA4 evaluated and removed — see [fa4.md](docs/optimizations/fa4.md). |

**Benchmarks** — RTX 5060 Ti 16 GB, LoRA rank 32, bs 2, 182 steps, seed 42. Val loss measured at σ ∈ {0.05, 0.1, 0.2, 0.35}.

| Configuration | Peak VRAM | Total | 2nd epoch | Train | Val |
|---|---|---|---|---|---|
| FA2 (plain) | 7.0 GB | 14:51 | 7:26 | 0.092 | 0.212 |
| FA2 + compile (eager fallback) | 7.7 GB | 15:10 | 7:26 | 0.089 | 0.211 |
| FA2 + compile (static tokens) | 6.2 GB | 11:07 | 5:01 | 0.086 | 0.193 |
| FA2 + compile − grad ckpt | 15.2 GB | **7:07** | **3:30** | 0.088 | 0.206 |
| same, rank 32 fast preset | 15.6 GB | 6:20 | 2:59 | 0.090 | 0.212 |

> CUDA 13.2 hits **1.15 s/step** at 15.5 GB — landing once PyTorch 2.12 ships. See [cuda132.md](docs/optimizations/cuda132.md).

Compile pipeline details in [docs/optimizations/for_compile.md](docs/optimizations/for_compile.md).

---

## 2. Verified merge-compatible variants

The default training config stacks **LoRA + OrthoLoRA + T-LoRA** together. All three fold losslessly into a standalone DiT checkpoint via thin-SVD export at save time, so you can ship ComfyUI-compatible `*_merged.safetensors` with no adapter loader dependency.

| Variant | Pitch | Details |
|---|---|---|
| **LoRA** | Classic low-rank, rank 16–32. | — |
| **OrthoLoRA** | SVD-parameterized with orthogonality regularization; exports as plain LoRA. | [psoft-integrated-ortholora.md](docs/methods/psoft-integrated-ortholora.md) |
| **T-LoRA** | Timestep-dependent rank masking — low rank at high noise, full rank at low noise. Training-only mask, so merge is bit-equivalent. | [timestep_mask.md](docs/methods/timestep_mask.md) |

**Side-by-side** — same prompt, `er_sde` 30 steps, `cfg=4.0`, 1024². Each LoRA trained at rank 16 for 2 epochs on a 20% subset with training seed 42; inference seeds `{41, 42, 43}`. Reproduce with `python scripts/bench_methods.py`.

|  | **plain (base)** | **OrthoLoRA + T-LoRA** |
|:---:|:---:|:---:|
| seed 41 | <img src="bench/side_by_side/plain/20260423-160513-382_41_.png" width="320"> | <img src="bench/side_by_side/ortho_tlora/20260423-155545-258_41_.png" width="320"> |
| seed 42 | <img src="bench/side_by_side/plain/20260423-160556-697_42_.png" width="320"> | <img src="bench/side_by_side/ortho_tlora/20260423-155631-762_42_.png" width="320"> |
| seed 43 | <img src="bench/side_by_side/plain/20260423-160640-759_43_.png" width="320"> | <img src="bench/side_by_side/ortho_tlora/20260423-155718-280_43_.png" width="320"> |

<details>
<summary>Individual variants (LoRA, OrthoLoRA, T-LoRA)</summary>

|  | **LoRA** | **OrthoLoRA** | **T-LoRA** |
|:---:|:---:|:---:|:---:|
| seed 41 | <img src="bench/side_by_side/lora/20260423-154854-014_41_.png" width="240"> | <img src="bench/side_by_side/ortholora/20260423-155109-338_41_.png" width="240"> | <img src="bench/side_by_side/tlora/20260423-155327-834_41_.png" width="240"> |
| seed 42 | <img src="bench/side_by_side/lora/20260423-154938-584_42_.png" width="240"> | <img src="bench/side_by_side/ortholora/20260423-155155-526_42_.png" width="240"> | <img src="bench/side_by_side/tlora/20260423-155413-304_42_.png" width="240"> |
| seed 43 | <img src="bench/side_by_side/lora/20260423-155024-080_43_.png" width="240"> | <img src="bench/side_by_side/ortholora/20260423-155241-905_43_.png" width="240"> | <img src="bench/side_by_side/tlora/20260423-155458-996_43_.png" width="240"> |

</details>

**Merging**:

```bash
make merge                                  # bake latest LoRA at multiplier 1.0
make merge ADAPTER_DIR=output/ckpt MULTIPLIER=0.8
```

Refuses non-linear-delta variants (ReFT / HydraLoRA `_moe` / postfix / prefix) by default; `--allow-partial` drops those and bakes only the LoRA portion.

---

## 3. Experimental features

Each ships with a doc — see the link for usage, flags, and caveats.

| Feature | What it is | Doc |
|---|---|---|
| **HydraLoRA** | MoE-style multi-head routing: shared `lora_down`, per-expert `lora_up_i`, layer-local router. Needs the `AnimaAdapterLoader` ComfyUI node. | [hydra-lora.md](docs/methods/hydra-lora.md) |
| **ReFT** | Block-level residual-stream intervention (LoReFT, NeurIPS 2024). Composes with any LoRA variant. | [reft.md](docs/methods/reft.md) |
| **APEX** | Self-adversarial 1–4 NFE distillation via learned condition shift; no discriminator, no teacher. | [apex.md](docs/methods/apex.md) |
| **Postfix / prefix tuning** | Continuous vectors appended (postfix) or prepended (prefix) to adapter cross-attention. Five postfix variants. | [postfix-sigma.md](docs/methods/postfix-sigma.md), [prefix-tuning.md](docs/methods/prefix-tuning.md) |
| **Embedding inversion** | Optimize a text embedding to match a target image through the frozen DiT. | [invert.md](docs/methods/invert.md) |
| **img2emb resampler** | Learn a reference-image → embedding mapping via TIPSv2-L/14 features + anchor injection. | [scripts/img2emb/README.md](scripts/img2emb/README.md) |
| **Spectrum inference** | Training-free ~3.75× speedup via Chebyshev feature forecasting (Han et al., CVPR 2026). Stable ComfyUI node in a separate repo: [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler). | [spectrum.md](docs/methods/spectrum.md) |
| **Modulation guidance** | Distill a `pooled_text_proj` MLP that steers AdaLN coefficients (Starodubcev et al., ICLR 2026). | [mod-guidance.md](docs/methods/mod-guidance.md) |
| **GRAFT** | Rejection-sampling fine-tuning — train, generate, curate survivors, retrain. | [graft-guideline.md](docs/guidelines/graft-guideline.md) |

---

## Setup

```bash
uv sync                   # Python 3.13 with pre-built flash attention 2
hf auth login
make download-models      # DiT + Qwen3 text encoder + QwenImage VAE into models/
# place training images in image_dataset/ with .txt caption sidecars
make gui                  # recommended — config editor + dataset browser + training monitor
```

CLI path:

```bash
make preprocess           # VAE-compatible resize & validation
make lora                 # or: make lora-fast / lora-low-vram / make postfix / make apex
make test                 # sample generation with the latest trained LoRA
```

Config chain: `configs/base.toml → configs/presets.toml[<preset>] → configs/methods/<method>.toml → CLI args`. Override with `PRESET=low_vram make lora` or `--network_dim 32 --max_train_epochs 64`. Full flag reference in [docs/guidelines/training.md](docs/guidelines/training.md) and [docs/guidelines/inference.md](docs/guidelines/inference.md).

---

## Documentation

| Doc | Contents |
|-----|----------|
| [guidelines/training.md](docs/guidelines/training.md) | Training flags, LoRA variants, caption shuffle, masked loss, dataset config |
| [guidelines/inference.md](docs/guidelines/inference.md) | Inference flags, P-GRAFT, prompt files, LoRA format conversion |
| [guidelines/graft-guideline.md](docs/guidelines/graft-guideline.md) | GRAFT curation workflow |
| [optimizations/](docs/optimizations/) | Compile pipeline, FA4 post-mortem, CUDA 13.2 |
| [methods/](docs/methods/) | One doc per method — APEX, HydraLoRA, ReFT, Spectrum, inversion, mod guidance, postfix/prefix, T-LoRA, OrthoLoRA |

---

## License

Toolkit code: [MIT](LICENSE).

Anima / CircleStone **base model weights** ship under the **CircleStone Labs Non-Commercial License v1.0** and are not relicensed by this repo. Any LoRA, fine-tune, or merged checkpoint trained from those weights is a Derivative and inherits the non-commercial terms. See [NOTICE](NOTICE).
