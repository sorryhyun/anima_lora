# CLAUDE.md — `library/moss/`

MOSS-Image (OpenMOSS, Apache-2.0) is a 5B single-stream DiT on a frozen Qwen3.5-9B text
encoder and the frozen FLUX.2 VAE. **The Anima invariants in the root `CLAUDE.md` are wrong
here.** Shared with Anima: `library.runtime.{offloading,blockswap,dynamo,progress}`,
`library.env`, the daemon, the ER-SDE sampler + CNS core (`library.inference.sampling`,
`library.inference.corrections.cns_core`), `library.vision.repa`. Shared with
`library/qwen21/`: `library.lines` (`requests.py` request base, `adapters.py` LoRA / LoKr /
LoHa, `scan.py` the GUI's counts). `spectrum.py` re-implements Anima's forecaster without
importing `networks/` (`tests/test_moss_spectrum.py` checks it against the core). Nothing
Anima-side imports `library.moss`, and it imports the rest of `library` only through the
list in `tests/test_moss_boundary.py`.

| Module | Role |
|---|---|
| `model.py` | vendored `MOSSImageTransformer` / `MOSSImageScheduler` (attribution kept, state-dict names unchanged); `forward_batched` is the batched path. i2i: `ref=` latents on RoPE's `r = 1` axis, `ref_timestep` / `ref_isolate` are transformer attributes; isolated at a fixed `ref_timestep`, `generate.denoise` sets `ref_cache` (`RefCache`) so the reference stream runs once per loop, and `text_cache` (`TextCache`) so the text embedder runs once per prompt |
| `requests.py` | **torch-free** `CacheRequest` / `ColorizeCacheRequest` / `CaptionsRequest` / `GenerateRequest` / `TrainRequest` / `TuneSwapRequest` / `DistillCacheRequest`; every flag and default is defined here once. `I2I_CACHE` maps each i2i task to its cache request; its recipes are `configs/moss/i2i/<task>/{cache,train,generate}.toml` (`i2i_config`) |
| `loader.py` | phase split TE → encode → free → DiT; TE paged by decoder layer (`offload_by_layer`, `te_reserve_gb`) |
| `fp8.py` | fp8 TE export (`quantize_te`, ComfyUI `comfy_quant` format) and `load_fp8_text_encoder`; `text_encoder=` picks catalog row `moss_te_{precision}`, and TE caches stamp `encoder_id(precision)` so a switch re-encodes |
| `dit_int8/` | the DiT's 252 block linears as int8 GEMMs, the training default (`TrainRequest.int8`): `library/int8/` holds the custom ops, `Int8TrainLinear` and `WqtStream`, shared with Anima; `base.py` here: targets, GPTQ forward copies (catalog `moss_dit_int8`), `convert_for_training`; `proj.q/k/v` run as one GEMM, `block.qkv`, and stay as `QKVPart` for the adapters. Refuses loha (`configs/moss/lora_bf16.toml` is the bf16 plan). Block swap runs on it, trading VRAM, not speed; it moves `wq` / `wq_t` as `HeldCopy` children's `.weight`. i2i trains on it (`int8 stream`) except `sanitize`, whose bf16 `train.toml` is the old v3 recipe (superseded i2i configs: `_archive/configs/moss_i2i/`). Generation runs on it by default (`GenerateRequest.int8 = on`, `stream fp32`, t2i and i2i, `generate_base`): 11–15 s per 1024² render against 22 s on bf16; it refuses LoHa and `blocks_to_swap` > 0. `docs/moss/optimization/int8.md`, `docs/moss/optimization/training_step.md` |
| `int8.py` | INT8 ConvRot TE (`Int8ConvRotLinear`, W8A8 or weight-only with `int8_act=False`). **The default TE** (`DEFAULT_TEXT_ENCODER = "int8"`, row `moss_te_int8`); SmoothQuant scales are folded into its RMSNorm weights, so its norms differ from Base's. Regenerate with `project/finished/moss_lora/src/te_gptq.py --grid int8_convrot --smooth 0.5 --act_aware` |
| `token_split.py` | Hangul token splits: `with_token_split(tok, mode)` wraps the tokenizer for `prompt_ids`. `CacheRequest.token_split` stamps TE files whose caption has Hangul and re-encodes only those; the LoRA stamps it; `GenerateRequest.token_split` `auto` follows the LoRA. Not on the colorize cache or `ref_vision`. The MOSS Toolkit node keeps a hand copy (`custom-nodes` skill) |
| `prompt_tail.py` | assistant-turn tail after the template (`sure`: `\n</think>\n\nSure,`, 4 rows; `docs/moss/findings/chat_template.md`): `with_prompt_tail(tok, mode)` wraps the tokenizer's template, under `with_token_split`. `sure` is the cache default; the i2i configs pin `off`. Stamped like `token_split`: `CacheRequest.prompt_tail` in every TE file, the LoRA from the cache, `GenerateRequest.prompt_tail` `auto` follows the LoRA (unstamped: `off`; no LoRA: `sure`), on cond and uncond. Training refuses `delta_rows` / `preserve_weight` with it |
| `caching/` | `cache.py`'s parts: `items.py` reads a cache folder (`list_items`, `load_*`, `read_te`) and rewrites caption variants; `images.py` `target_size`, `fit`, `ref_size`, `prefetch`; `sources.py` `find_pairs`, `manifest_pairs`, `caption_converter`, and the OCR boxes on text lines (`text_boxes`: `ocr_boxes`; a `layout_json` text item's `bbox`, after the quote under `tags`, `box_text_clauses`). `cache.py` re-exports them: it stays their import path, so a monkeypatch of what they call goes on `caching.<module>` |
| `cache.py` | the passes (`cache_text`, `cache_latents`, `cache_refs`, `run_cache`): TE cache at natural length + packed, BN-normalised latent cache; caption variants `embeds_v{k}` survive a re-encode while the caption is unchanged. Cache into a fresh folder, not a symlink farm of another cache: every write (latent re-encode when the image is newer than the latent, `token_split` TE re-encode, `add_variants`) goes through the symlink into the source cache |
| `generation/` | `generate.py`'s parts: `jobs.py` request → jobs + resume `fingerprint`; `conditioning.py` `cached_embeds`, `vision_embeds`, `ref_latents`; `records.py` the latent / PNG paths and the manifest. `generate.py` re-exports them as `train.py` does |
| `generate.py` | `denoise` and `run_generate(req)`: ER-SDE or Euler, CFG + uncond skip-22, Spectrum, CNS. Benches reuse `denoise`, `cached_embeds`, `build_spectrum`. `ref_image` makes a job i2i: `ref_latents` (fitted as `cache_refs` does) into every forward, and with `ref_vision` the page through the vision tower ahead of the prompt (`vision_embeds`) |
| `vision_te.py` | the TE call with an image in the user turn (Qwen3.5's own processor settings); `load_te` keeps the vision tower resident, any `text_encoder` precision; `encode_variants` prefills the page once for several captions |
| `colorize.py` | `run_colorize_cache(req)`: staged B&W page → TE captions (`control`, the v5 default: colour + series tags, or `comic` + box-only text items + per-character colour boxes; `ocr`: OCR layout over full / sparse / bare colour captions; the page through the vision tower with vision on) + `*.moss_ref`, colour target latent; OCR floors shared with the ComfyUI MOSS Toolkit's caption helper |
| `captions.py` | `run_captions(req)`: MOSS-format captions for one set, into a tree of its own (`post_image_dataset/moss_captions/<set>`, images symlinked): flatten the Anima clauses, anime_tools' position stage with `POSITION_FORMAT` (every clause-carried tag leaves the bag), then `drop_implied_tags`. The cache reads the tree as `src`; the Anima captions are never written |
| `cns.py` | CNS recolor on the un-patchified 32 ch grid, `GenerateRequest.cns` (`auto` = the shipped γ per variant × CFG, `networks/calibration/moss_cns_gamma*.npz`) |
| `spectrum.py` | forecasts cond and uncond separately; `MossSpectrum()` defaults are `GenerateRequest`'s |
| `dave.py` | DAVE on image tokens via `model.layer_hook` (sees only rows that ran the block, so skip-22 is handled); τ is a step count; massive channels (`calibration/massive_channels.npz`) excluded. `auto` = `calibration/dave_alpha.npz`; the ComfyUI node `~/ComfyUI-Anima-DAVE` bundles the same pool + channel list. User doc `docs/inference/dave.md` |
| `lora.py` | adapters bound to `layers`, held **outside** the blocks (the swapper moves every `.weight` under a block). `down_init weight_svd_nodc` / `svd_drop_loud` read `calibration/` stats from `scripts/calibration/moss_channel_stats.py`; `load_for_inference` applies the LoRA's stamped `ref_timestep` / `ref_isolate` (`ref_settings`). `ref_rank` adds the plan A export adapters or the plan B bank (`_archive/proposals/reference_adapters.md`); neither beat the shared bank |
| `training/` | `train.py`'s parts: `data.py` samples, `choose_embeds`, σ; `losses.py` the terms; `ema.py`; `muon.py` `BatchedMuon`, torch's Muon stacked by matrix shape; `memory.py` placement, budget, fit report; `setup.py` what `train` builds before its loop; `sampling.py` the `sample_prompts` renders (TE before the DiT loads, no-grad `denoise` at epoch ends (the compile range takes in the sample lengths), CPU VAE decode in a thread; under block swap `attached.prepare` after each forward and `transformer.skip_hook = attached.skip` for the skipped block 22). `train.py` re-exports them: it stays their import path and the seam finished scripts patch (`T.training_loss`, `T.place`, `T.LoRANetwork`, `T.load_transformer`) |
| `train.py` | `run_train(req)`, the loop; `choose_embeds` picks plain / variant / empty caption per step, one RNG each (the empty prompt is the shipped `asset/empty_prompt.safetensors`). Reference latents (`*.moss_ref`) train with `ref_timestep` / `ref_isolate`, stamped in the LoRA; `ref_gate cut` cuts a random hole's image queries from the reference keys in the hole (`training/holes.py`, `model._attention`'s `ref_gate`; `project/lora_blocks`). `tone_weight` (off): `docs/moss/methods/tone_loss.md`. Streams `progress.jsonl` (the daemon job dir, else `output/logs/<stem>.progress.jsonl`) so `make run-status` follows it. Above `PRELOAD_LIMIT_GB` samples stream from disk |
| `distill.py` | self-distillation teacher cache (`cache_teachers`, `{stem}.moss_vis.safetensors` beside each TE file) for `TrainRequest.distill`; `docs/moss/proposal/self_distill_lora.md` |
| `repa.py` | relational REPA on one block via `layer_hook`; parked (`repa_weight` 0 = off) |
| `tune.py` | `run_tune(req)`: recommends `--blocks_to_swap` / `--activation_memory_budget` on the bf16 plan |
| `scan.py` | **torch-free** cache layout + GUI counts; its stale check (`source_sha256` vs `.txt`) can disagree with `cache.py`'s (`_te_current`) |

Sidecars `scripts/moss/{cache,captions,colorize_cache,distill_cache,generate,quantize_te_fp8,train,tune_swap}.py`
are thin argparse over the requests, submitted as `make daemon-run ARGS="--stall-timeout 900
scripts/moss/<sidecar>.py …"`; `make gui` (Model: MOSS-Image) or `make gui-moss` (its own window)
submits the same jobs (i2i tabs: `gui/moss/i2i.py`). `--config <file.toml>` is a flat TOML of field
names, layered field default → TOML → CLI; shipped defaults `configs/moss/{cache,lora}.toml`.
Caches land in `post_image_dataset/moss[/<set>]` (flat, two files per image), LoRAs in
`output/ckpt_moss/` (i2i: `output/ckpt_moss_i2i/`). i2i: `make i2i-stage TASK=<task>` →
`make i2i-preprocess TASK=<task> MODEL=moss` → train / generate with
`configs/moss/i2i/<task>/{cache,train,generate}.toml`; tasks and status in
`docs/moss/methods/i2i.md`.

User doc: `docs/moss/README.md` (GUI walkthrough `docs/guidelines/moss-guidebook.md`). Weights:
catalog packs `moss-base` / `moss-sft` under `models/moss_image/MOSS-Image-{Base,SFT}`; SFT ships
its DiT alone, TE / VAE / tokenizer always load from Base. Research line (finished):
`project/finished/moss_lora/`; A/B and i2i eval scripts: `bench/moss_ab/`,
`bench/moss_i2i_eval/`; findings: `docs/moss/findings/`. ComfyUI: the MOSS Toolkit and the
Spectrum pack's KSampler (Spectrum for MOSS) (`custom-nodes` skill).

## What does NOT carry over

| Root `CLAUDE.md` (Anima) says | Here |
|---|---|
| Latents are 5D `(B,C,T=1,H,W)`, singleton at **dim 2** | **4D packed `(B, 128, H/16, W/16)`** (FLUX.2's 32 ch 2×2-patchified), no frame axis; normalised by the VAE's BatchNorm buffers (`model.pack_latents` / `unpack_latents`) |
| TE outputs **must** be max-padded; never mask padding | **Never pad.** Each prompt is encoded alone at its true length (Qwen3.5's GDN layers misbehave on padding); `forward_batched` masks padded keys. RoPE places the image at frame position = text length, so padding shifts the image |
| Free-fit bucketing via `EDGE_TOKEN_BANDS`, edges 512…1536 | Native aspect, both sides **multiples of 16** |
| DiT depth probed from the checkpoint (28 vs 40) | Fixed **42** blocks (`num_layers` in `transformer/config.json`) |
| Standard CFG | **CFG 3**, the **uncond branch skips block 22** (still 2 forwards/step); optional SLG third forward. Timestep input is `1000·σ` |
| Anima's sampler shift | **Shift 5 snapped to the nearest training-grid σ** (`model.snapped_sigmas`); ER-SDE by default, Euler is the pipeline's ODE |
| Adapters live in `networks/`, selected by `network_module` | `networks/` is bound to Anima module names; `lora.py` targets the six bias-free linears per block (`proj.{q,k,v,o,ffn_up_gate,ffn_down}`) |
| `make lora`, config chain `base → preset → method → CLI`, `masked_loss` | None of it: a flat `--config` TOML of field names |

## Standing rules

- **Read the pipeline before setting a sampler knob.** `pipeline_moss_image.py` (in the
  model folder) is the reference; `model.py` stays op-for-op with it on the unbatched path.
- RoPE frequencies stay fp32 (`_RoPE` rebuilds them on every `.to()`).
- **Cache, train and generate with the same `text_encoder`, `caption_format` and
  `prompt_tail`.** TE files stamp the encoder, the encoded caption (so its format) and the tail,
  and a cache pass re-encodes when one changes; generate's `prompt_tail auto` follows the LoRA's
  stamp.
- Decode with the DiT off the card: the fp32 VAE decode at ~2 MP does not fit beside it.
- **A probe that patches, swaps or masks activations carries three controls from its first run**:
  an `own` arm (must read 0; store the fed rows at the stream's precision, since bf16 storage on
  an fp32 stream reads the floor), a `jitter` arm and an equal-norm isotropic arm. Arms are
  compared against those, not against 0. Any nonzero change re-draws the rounding inside the
  blocks: a 1e-3 relative jitter of the text rows moves the velocity by 1e-4…2e-4 (σ 0.95) to
  5e-4…1e-3 (σ 0.3), as much as a change 10–30 times larger, on bf16 and on int8 + fp32 stream
  alike (`docs/moss/findings/text_stream_page_state.md`). Probes run on generation's DiT (int8 +
  fp32 stream).
