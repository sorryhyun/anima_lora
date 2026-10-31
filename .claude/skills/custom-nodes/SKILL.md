---
name: custom-nodes
description: The ComfyUI node map — which node lives in which standalone repo vs in-tree under custom_nodes/, where each is symlinked, and the vendor-sync rule for the _vendor/ subsets. Load before editing, publishing, or hunting for a ComfyUI node, or before touching anything under custom_nodes/.
---

# ComfyUI custom nodes

Most nodes live in standalone repos, **symlinked** into `../comfy/custom_nodes/`. Edit
the source repo, never the symlink.

## Out of tree

| Node | Repo / path | Notes |
|---|---|---|
| Spectrum KSampler + mod-guidance | https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler | ships DCW scalar default `+0.01` + `auto` mode |
| PiD decode | https://github.com/sorryhyun/ComfyUI-Anima-PiD | symlinked as `comfyui-anima-pid` |
| EasyControl KSampler | `~/ComfyUI-EasyControl-KSamplerCompat` | |
| Block Compile | https://github.com/sorryhyun/ComfyUI-Anima-BlockCompile | standalone at `~/ComfyUI-Anima-BlockCompile`, symlinked as `comfyui-anima-blockcompile` |
| DAVE | https://github.com/sorryhyun/ComfyUI-Anima-DAVE | standalone at `~/ComfyUI-Anima-DAVE`, symlinked as `comfyui-anima-dave`. One node for Anima and the `moss-image` transformer; `mask=auto` picks the bundled pool per model. Its npz masks are copies of `networks/calibration/dave_alpha.npz` and `library/moss/calibration/{dave_alpha,massive_channels}.npz`, not vendor-synced |
| MOSS Toolkit | https://github.com/sorryhyun/ComfyUI-MOSS-Toolkit (renamed from ComfyUI-MOSS-i2i; registry id `comfyui-moss-toolkit`, the old `comfyui-moss-i2i` id is retired), standalone at `~/ComfyUI-MOSS-Toolkit`, symlinked as `comfyui-moss-toolkit` | `MOSSImageLoader` (`loader.py`): DiT + TE (bf16 dir, or an INT8 / fp8 file under `moss_image/{int8,fp8}/`) + VAE in one node, all through ComfyUI's own loaders, `MOSSImage` model, MOSS-Image TE and bundled tokenizer; the `moss-image` node is not required (since 0.10.0, which needs Spectrum ≥ 2.13.1 for **KSampler (Spectrum for MOSS)** on that model). fp32 stream (`stream.py`, mirrors `library/moss/loader.py::stream_fp32_`): the `stream` knob and the `MOSSStreamFP32` node add an `APPLY_MODEL` wrapper (fp32 `x`, pre-hooks cast the blocks' GEMM inputs back to the compute dtype, ops cast their weights for the call, no weight changes, dynamic VRAM and INT8 work); the node refuses a `moss-image` DiT. i2i: page-through-vision-tower text encode + VAE-reference MODEL patch for the `moss-image` node; the Reference node loads the i2i LoRA itself (`lora_name` / `strength`) and reads `reference_latents` / `ref_scale` / `ref_timestep` / `ref_isolate` from its metadata (an ai-toolkit edit LoRA, no such metadata: full size, t = 0, isolated); `lora_compat.py` splits fused `proj.qkv` LoRA / LoKr / LoHa groups for the split `moss-image` model and logs the patched non-block modules; on ComfyUI's core `MOSSImage` (fused or split) `reference.py` hands the ref to the model's own `ref_kv` / `ref_latents` path instead of its forward copy (full-size refs only); `task = auto` picks the ref size from the recipe (`ref_isolate`: full size, text-only prompt through a plain CLIP Text Encode; joint: half size, vision-tower Text Encode); saved `colorize` runs as `auto`, saved `custom` still loads (via `VALIDATE_INPUTS`) and uses the `ref_scale` widget. The ref forward (`reference.py`) mirrors `library/moss/model.py` (incl. `_ref_only`: under `ref_isolate` a `patches_replace`-skipped block still runs the ref rows); an isolated ref with a `ref_timestep` is run once per page into `RefKV` (per-block K before RoPE + V, `ref_cache` gpu / cpu / off) and each forward runs the image + text rows only (`_main_block`), the blocks' forward hooks (DAVE) called by hand and runs as an `APPLY_MODEL` wrapper: the OpenMOSS forward never calls `DIFFUSION_MODEL` wrappers. Text: `MOSSCaptionHelper` (task `default` — the caption as written; `layout` — Anima caption → layout JSON / `colorize` / `sanitize` / `decensor`; `layout` passes a caption that already is layout JSON through, a copy of `library/moss/cache.py::is_layout_json`; colorize appends the page's OCR lines, a copy of `library/moss/colorize.py`'s `readable` / `dedupe` / layout — change both together; sanitize / decensor return their `FIXED_CAPTIONS` row, a copy of that task's `configs/moss/i2i/<task>/cache.toml` `fixed_caption`), registered only when `anime_tools` imports; `AnimaCaptionToMossLayout` is its `layout` alone, `DEPRECATED` (hidden from the menu, registered so saved workflows load) |
| Anima Adapter Loader | https://github.com/sorryhyun/ComfyUI-Anima_lora-Adapter | Adapter / FeRA / Soft Tokens loaders; standalone at `~/ComfyUI-Anima_lora-Adapter`. **Read its `CLAUDE.md` for the `forward_hook`-not-override invariant** |
| Anima Tagger | inside the `anime_tools` repo (`comfyui/anima_tagger/` there) | symlinked as `comfyui-anima-tagger`; imports `anime_tools.tagger` and vendors nothing. The standalone `ComfyUI-Anima-Tagger` repo is **retired but still on disk** — don't edit it |

## In tree

Under `custom_nodes/`: `comfyui-anima-directedit/`, `comfyui-anima-register/`,
`comfyui-anima-trainer/` (daemon-backed one-shot trainer).

## Vendor trees

Several nodes carry a `_vendor/` subset of the live tree. **Regenerate with `make
vendor-sync` (`scripts/release/sync_vendor.py`), never `cp` by hand** — and re-run it
before every node publish.
