# CLAUDE.md — `library/moss/`

**This is not Anima.** MOSS-Image (OpenMOSS, Apache-2.0) is a 5B single-stream DiT on a
frozen Qwen3.5-9B text encoder and the frozen FLUX.2 VAE; the Anima
invariants in the root `CLAUDE.md` are wrong here. Shared with Anima: `library.runtime.{offloading,blockswap,dynamo}`,
`library.env`, the daemon, the ER-SDE sampler + CNS core (`library.inference.sampling`,
`library.inference.corrections.cns_core`), `library.training.repa`. Shared with
`library/qwen21/`: `library.lines` (`requests.py` request base, `adapters.py` LoRA / LoKr /
LoHa). `spectrum.py` re-implements Anima's forecaster rather than importing `networks/`
(`tests/test_moss_spectrum.py` checks it against the core). Nothing Anima-side imports
`library.moss`.

| Module | Role |
|---|---|
| `model.py` | vendored `MOSSImageTransformer` / `MOSSImageScheduler` (attribution kept, state-dict names unchanged); per-sample `forward` must stay op-for-op with the pipeline, `forward_batched` masks text. i2i: `ref=` latents on RoPE's `r = 1` axis, `ref_timestep` / `ref_isolate` are transformer attributes; isolated at a fixed `ref_timestep`, `generate.denoise` sets `ref_cache` (`RefCache`) so the reference stream runs once per loop |
| `requests.py` | **torch-free** `CacheRequest` / `ColorizeCacheRequest` / `GenerateRequest` / `TrainRequest` / `TuneSwapRequest`; every flag and default is defined here once. `I2I_CACHE` maps each i2i task to its cache request; its recipes are `configs/moss/i2i/<task>/{cache,train,generate}.toml` (`i2i_config`) |
| `loader.py` | phase split TE → encode → free → DiT; TE paged by decoder layer (`offload_by_layer`, `te_reserve_gb`) |
| `fp8.py` | fp8 TE export (`quantize_te`, ComfyUI `comfy_quant` format) and `load_fp8_text_encoder`; `text_encoder=` picks catalog row `moss_te_{precision}`, and TE caches stamp `encoder_id(precision)` so a switch re-encodes |
| `dit_int8/` | the DiT's 252 block linears as int8 GEMMs, the training default (`TrainRequest.int8`): `linear.py` custom ops + `Int8TrainLinear` (shared with `project/int8_train`'s Anima half), `stream.py` `WqtStream`, `base.py` targets, GPTQ forward copies (catalog `moss_dit_int8`), `convert_for_training`. Refuses loha; block swap moves `wq` / `wq_t` as `HeldCopy` children's `.weight`. Generation on it is still `project/int8_train/src/infer_int8_moss.py`. `docs/moss/optimization/int8.md` |
| `int8.py` | INT8 ConvRot TE (`Int8ConvRotLinear`, W8A8 or weight-only with `int8_act=False`). **The default TE** (`DEFAULT_TEXT_ENCODER = "int8"`, row `moss_te_int8`); SmoothQuant scales are folded into its RMSNorm weights, so its norms differ from Base's. Regenerate with `project/finished/moss_lora/src/te_gptq.py --grid int8_convrot --smooth 0.5 --act_aware` |
| `cache.py` | TE cache at natural length (never padded) + packed, BN-normalised latent cache; caption variants `embeds_v{k}` survive a re-encode while the caption is unchanged |
| `generate.py` | `run_generate(req)`: ER-SDE or Euler, CFG + uncond skip-22, Spectrum. Benches reuse `denoise`, `cached_embeds`, `build_spectrum`. `ref_image` makes a job i2i: `ref_latents` (fitted as `cache_refs` does) into every forward, and with `ref_vision` the page through the vision tower ahead of the prompt (`vision_embeds`) |
| `vision_te.py` | the TE call with an image in the user turn (Qwen3.5's own processor settings); `load_te` keeps the vision tower resident, any `text_encoder` precision; `encode_variants` prefills the page once for several captions |
| `colorize.py` | `run_colorize_cache(req)`: staged B&W page → vision-tower TE (OCR layout over full / sparse / bare colour captions; page alone with vision on) + `*.moss_ref`, colour target latent; OCR floors shared with the ComfyUI MOSS Toolkit's caption helper |
| `cns.py` | CNS recolor on the un-patchified 32 ch grid; bench-only |
| `spectrum.py` | forecasts cond and uncond separately; `MossSpectrum()` defaults are `GenerateRequest`'s |
| `dave.py` | DAVE on image tokens via `model.layer_hook` (sees only rows that ran the block, so skip-22 is handled); τ is a step count; massive channels (`calibration/massive_channels.npz`) excluded. `auto` = `calibration/dave_alpha.npz`; the ComfyUI node `~/ComfyUI-Anima-DAVE` bundles the same pool + channel list. User doc `docs/moss/dave.md` |
| `lora.py` | adapters bound to `layers`, held **outside** the blocks (the swapper moves every `.weight` under a block). `down_init weight_svd_nodc` / `svd_drop_loud` read `calibration/` stats from `scripts/calibration/moss_channel_stats.py`; `load_for_inference` applies the LoRA's stamped `ref_timestep` / `ref_isolate` (`ref_settings`) |
| `training/` | `train.py`'s parts: `data.py` samples, `choose_embeds`, σ; `losses.py` the terms; `ema.py`; `muon.py` `BatchedMuon`, torch's Muon stacked by matrix shape; `memory.py` placement, budget, fit report; `setup.py` what `train` builds before its loop. `train.py` re-exports them: it stays their import path and the seam finished scripts patch (`T.training_loss`, `T.place`, `T.LoRANetwork`, `T.load_transformer`) |
| `train.py` | `run_train(req)`, the loop; `choose_embeds` picks plain / variant / empty caption per step, one RNG each (the empty prompt is the shipped `asset/empty_prompt.safetensors`). Reference latents (`*.moss_ref`) train with `ref_timestep` / `ref_isolate`, stamped in the LoRA. `tone_weight` (off) adds the x̂₀ tone term on the unedited tokens, projected to Lab by `calibration/tone_dirs.safetensors` (`docs/moss/tone_loss.md`). Streams `progress.jsonl` (the daemon job dir, else `output/logs/<stem>.progress.jsonl`) so `make run-status` follows it. `--fit_report` measures peak VRAM on the largest sample; above `PRELOAD_LIMIT_GB` samples stream from disk |
| `repa.py` | relational REPA on one block via `layer_hook`; parked (`repa_weight` 0 = off) |
| `tune.py` | `run_tune(req)`: recommends `--blocks_to_swap` / `--activation_memory_budget` on the bf16 plan |
| `scan.py` | **torch-free** cache layout + GUI counts; its stale check (`source_sha256` vs `.txt`) can disagree with `cache.py`'s (`_te_current`) |

Sidecars `scripts/moss/{cache,colorize_cache,generate,train,tune_swap}.py` are thin argparse over the
requests, submitted through the daemon; `make gui-moss` submits the same jobs (its i2i tabs:
`gui/moss/i2i.py`). User doc
`docs/moss/README.md`. Weights: catalog packs `moss-base` / `moss-sft` under
`models/moss_image/MOSS-Image-{Base,SFT}`; SFT ships its DiT alone, TE / VAE / tokenizer
always load from Base. Research line (finished): `project/finished/moss_lora/`; A/B and i2i eval scripts: `bench/moss_ab/`, `bench/moss_i2i_eval/`.

## What does NOT carry over

| Root `CLAUDE.md` (Anima) says | Here |
|---|---|
| Latents are 5D `(B,C,T=1,H,W)`, singleton at **dim 2** | **4D packed `(B, 128, H/16, W/16)`** (FLUX.2's 32 ch 2×2-patchified), no frame axis; normalised by the VAE's BatchNorm buffers (`model.pack_latents` / `unpack_latents`) |
| TE outputs **must** be max-padded; never mask padding | **Never pad.** Each prompt is encoded alone at its true length (Qwen3.5's GDN layers misbehave on padding); `forward_batched` masks padded keys. RoPE places the image at frame position = text length, so padding shifts the image |
| Free-fit bucketing via `EDGE_TOKEN_BANDS`, edges 512…1536 | Native aspect, both sides **multiples of 16** |
| DiT depth probed from the checkpoint (28 vs 40) | Fixed **42** blocks (`num_layers` in `transformer/config.json`) |
| Standard CFG | **CFG 3**, the **uncond branch skips block 22** (still 2 forwards/step); optional SLG third forward. Timestep input is `1000·σ`. Defaults come from `pipeline_moss_image.py` |
| Anima's sampler shift | **Shift 5 snapped to the nearest training-grid σ** (`model.snapped_sigmas`); ER-SDE by default, Euler is the pipeline's ODE |
| Adapters live in `networks/`, selected by `network_module` | `networks/` is bound to Anima module names; `lora.py` targets the six bias-free linears per block (`proj.{q,k,v,o,ffn_up_gate,ffn_down}`) |
| `make lora`, config chain `base → preset → method → CLI`, `masked_loss` | None of it. `--config configs/moss/lora.toml` (flat field-name TOML) layers field default → TOML → CLI |

## Standing rules

- **Read the pipeline before setting a sampler knob.** `pipeline_moss_image.py` (in the
  model folder) is the reference; `model.py` stays op-for-op with it on the unbatched path.
- RoPE frequencies stay fp32 (`_RoPE` rebuilds them on every `.to()`).
- Decode with the DiT off the card: the fp32 VAE decode at ~2 MP does not fit beside it.
- GPU work goes through the daemon.
