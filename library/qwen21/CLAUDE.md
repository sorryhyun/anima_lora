# CLAUDE.md — `library/qwen21/`

**This is not Anima.** The root `CLAUDE.md`'s Anima section describes the Anima DiT, and its
invariants are load-bearing *there* — carrying them over here has produced wrong work more
than once. Qwen-Image-2.1 is a different architecture, VAE, text encoder, rope and sampler
schedule. What the two share is `library.runtime.offloading` (the block swapper) with
`library.runtime.blockswap` (its hook attachment for block lists we don't own; shared with `library/moss/`),
`library.runtime.dynamo` (torch config pins), `library.runtime.device`, the `library.env` path helpers, and `library.lines` (the request base,
adapters and scan counts shared with `library/moss/`), and nothing else: the rest of `library/`, `networks/`,
`configs/` and `train.py` are Anima-only, and nothing Anima-side imports `library.qwen21`.
`tests/test_qwen21_boundary.py` enforces this and keeps `requests` and `gui.qwen21.app`
torch-free.

| Module | Role |
|---|---|
| `requests.py` | **torch-free** `CacheRequest` / `TrainRequest` / `GenerateRequest`: every flag, its default and help, defined once for the sidecar CLIs and the GUI (adding a field: `qwen21` skill); argv round-trip from `library.lines.requests.Request`; model-dir resolution |
| `loader.py` | phase-split loading under a 16 GB budget, `load_transformer`; re-exports `place()` (`library.runtime.blockswap`) |
| `accel.py` | attention backend + per-block `torch.compile` |
| `lora.py` | `library.lines.adapters` bound to `transformer_blocks` (LoRA / LoKr / LoHa); `--algo` / `--down_init` default to LoRA + Kaiming |
| `cache.py` / `train.py` / `generate.py` | `run_cache` / `run_train` / `run_generate(req)` — sidecars in `scripts/qwen21/` |
| `scan.py` | **torch-free** source/cache folder counts for the GUI (stale text caches) |

Operating the line (commands, flags, cache behaviour, swap sizing, compile): the `qwen21` skill.
Research line (smokes, benches, reports, measured numbers): `project/finished/qwen21_lora/`.

## What does NOT carry over

| Anima habit | Here |
|---|---|
| Latents are 5D `(B,C,T=1,H,W)`, singleton at **dim 2**; always `unsqueeze(2)`/`squeeze(2)` | The transformer takes **packed 3D** `(B, T, 64)` — `_pack_latents` is a plain spatial flatten. 5D is only the VAE's own boundary: `(B,4,1,H,W)` in, `(B,64,1,h,w)` out |
| TE outputs **must** be max-padded; trimming gives black images; never mask padding | Padding the text is **wrong** here. Rope gives the image block a frame position equal to the text length, so a padded caption trains an offset inference never reproduces. `encoder_hidden_states_mask` excludes padded keys properly |
| Free-fit native bucketing via `EDGE_TOKEN_BANDS`, edges 512…1536 | `calculate_dimensions(res², aspect)` — area ~res², both edges a multiple of **32**, which is what keeps the token count divisible by 4 for the target's `img_mask` slots. Still native aspect: the image is resized, never cropped (to square or otherwise), so its aspect drifts up to ~2 % |
| VAE is 3-channel | **4-channel RGBA** in, `z_dim` 64, /16 downscale (`dim_mult` has 4 spatial stages) |
| Block-compile FIRST on OOM, not grad checkpointing | **Activation checkpointing is the lever**: 32 blocks at ~4096 tokens is what does not fit. Compile (on by default) is a speed lever, not a memory one |
| DiT depth probed from the checkpoint (28 vs 40 blocks); a LoRA is depth-specific | Fixed 32 `transformer_blocks`, `num_layers` in the config. No probing, no depth-baked calibration artifacts |
| CFG is standard | `true_cfg_scale` defaults to **1.0** and the transformer has no guidance embedding. `>1` costs a second forward per step |
| Adapters live in `networks/`, selected by `network_module` | `networks/` is bound to Anima module names. `lora.py` keeps its pairs **outside** `transformer_blocks` — the swapper moves every `.weight` under a block, so a nested adapter (peft) pages trainable weights to CPU while their `.grad` stays on the card |
| `make lora`, config chain `base → preset → method → CLI`, `masked_loss` | None of it: flags only (`requests.py`) |

## Standing rules

- **Read the pipeline before setting a sampler knob.** Defaults for CFG, shift (`mu` from
  `calculate_shift` off the *scheduler config*, not the function's own fallbacks) and
  preprocessing all live in `diffusers/pipelines/qwenimage21/` — take them from there
  rather than from Anima habit or from the function signature's defaults.
- Both encoders are precached (`cache.py`); training never loads them.
