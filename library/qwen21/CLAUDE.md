# CLAUDE.md — `library/qwen21/`

**This is not Anima.** The root `CLAUDE.md` describes the Anima DiT, and its
invariants are load-bearing *there* — carrying them over here has produced wrong work more
than once. Qwen-Image-2.1 is a different architecture, VAE, text encoder, rope and sampler
schedule. What the two share is `library.runtime.offloading` (the block swapper),
`library.runtime.dynamo` (torch config pins), the `library.env` path helpers, and nothing else: the rest of `library/`, `networks/`,
`configs/` and `train.py` are Anima-only, and nothing Anima-side imports `library.qwen21`.

| Module | Role |
|---|---|
| `requests.py` | **torch-free** `CacheRequest` / `TrainRequest` — every flag, its default and help; `to_argv()` / `from_argv()`; model-dir resolution |
| `loader.py` | phase-split loading under a 16 GB budget, `place()`, `load_transformer` |
| `blockswap.py` | `ModelOffloader` on stock diffusers/transformers block lists |
| `accel.py` | attention backend + per-block `torch.compile` |
| `lora.py` | adapters held **outside** the swapped blocks |
| `cache.py` / `train.py` / `generate.py` | `run_cache` / `run_train` / `run_generate(req)` — sidecars in `scripts/qwen21/` |
| `scan.py` | **torch-free** source/cache folder counts for the GUI (stale text caches) |

Running, the GUI, adding a flag, model-dir resolution and gotchas: **load the `qwen21`
skill**.

The research line — smokes, benches, reports, measured numbers — is
`project/qwen21_lora/`.

## What does NOT carry over

| Root `CLAUDE.md` says | Here |
|---|---|
| Latents are 5D `(B,C,T=1,H,W)`, singleton at **dim 2**; always `unsqueeze(2)`/`squeeze(2)` | The transformer takes **packed 3D** `(B, T, 64)` — `_pack_latents` is a plain spatial flatten. 5D is only the VAE's own boundary: `(B,4,1,H,W)` in, `(B,64,1,h,w)` out |
| TE outputs **must** be max-padded; trimming gives black images; never mask padding | Padding the text is **wrong** here. Rope gives the image block a frame position equal to the text length, so a padded caption trains an offset inference never reproduces. `encoder_hidden_states_mask` excludes padded keys properly |
| Free-fit native bucketing via `EDGE_TOKEN_BANDS`, edges 512…1536 | `calculate_dimensions(res², aspect)` — area ~res², both edges a multiple of **32**, which is what keeps the token count divisible by 4 for the target's `img_mask` slots. Still native aspect: do not crop to square |
| VAE is 3-channel | **4-channel RGBA** in, `z_dim` 64, /16 downscale (`dim_mult` has 4 spatial stages) |
| Block-compile FIRST on OOM, not grad checkpointing | **Activation checkpointing is the lever** — 32 blocks at ~4096 tokens is what does not fit, and compile is a speed lever, not a memory one: ±0 at 512² with 12 swaps (PCIe-bound), −13 % at 1024² with 7 swaps (compute-bound; swap 14→5 changed nothing). `--compile_seq bounded` = automatic dynamic + `mark_dynamic` over the cache's joint-token range |
| DiT depth probed from the checkpoint (28 vs 40 blocks); a LoRA is depth-specific | Fixed 32 `transformer_blocks`, `num_layers` in the config. No probing, no depth-baked calibration artifacts |
| CFG is standard | `true_cfg_scale` defaults to **1.0** and the transformer has no guidance embedding. `>1` costs a second forward per step |
| Adapters live in `networks/`, selected by `network_module` | `networks/` is bound to Anima module names. `lora.py` keeps its pairs **outside** `transformer_blocks` — the swapper moves every `.weight` under a block, so a nested adapter (peft) pages trainable weights to CPU while their `.grad` stays on the card |
| `make lora`, config chain `base → preset → method → CLI`, `masked_loss` | None of it. Flags only: `requests.py` (`CacheRequest` / `TrainRequest`) defines them once for the sidecar CLIs `scripts/qwen21/{cache,train}.py` and the GUI |

## Standing rules

- **Read the pipeline before setting a sampler knob.** Defaults for CFG, shift (`mu` from
  `calculate_shift` off the *scheduler config*, not the function's own fallbacks) and
  preprocessing all live in `diffusers/pipelines/qwenimage21/` — take them from there
  rather than from Anima habit or from the function signature's defaults.
- **Size the swap from a measurement, not a reserve guess.** `train.py::report_fit`
  prints the fit after step 1. Compare against `mem_get_info` free — not
  `total - max_allocated`, which ignores the allocator's reserve (~0.7 GB) and the desktop
  (~0.5 GB) and will recommend a swap that OOMs.
- Both encoders are precached (`cache.py`); training never loads them.
- GPU work goes through the daemon, as everywhere in this repo.
