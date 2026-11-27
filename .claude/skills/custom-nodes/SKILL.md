---
name: custom-nodes
description: The ComfyUI node map — which node lives in which standalone repo vs in-tree under custom_nodes/, where each is symlinked, which files are hand copies of this repo's code, and the publish flow. Load before editing, publishing, or hunting for a ComfyUI node, or before touching anything under custom_nodes/.
---

# ComfyUI custom nodes

Most nodes live in standalone repos, **symlinked** into `../comfy/custom_nodes/`. Edit
the source repo, never the symlink. A node's behaviour and internals are in its repo's
`README.md` (and `CLAUDE.md` where one exists).

## Out of tree

| Node | Repo / path | Notes |
|---|---|---|
| Spectrum KSampler + mod-guidance | https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler | ships DCW scalar default `+0.01` + `auto` mode |
| PiD decode | https://github.com/sorryhyun/ComfyUI-Anima-PiD | symlinked as `comfyui-anima-pid` |
| EasyControl KSampler | https://github.com/sorryhyun/ComfyUI-EasyControl-KSamplerCompat, standalone at `~/ComfyUI-EasyControl-KSamplerCompat` | symlinked as `comfyui-easycontrol` |
| Distilled ResShift | https://github.com/sorryhyun/ComfyUI-Distilled-ResShift, standalone at `~/ComfyUI-Distilled-ResShift` | symlinked as `comfyui-distilled-resshift` |
| Block Compile | https://github.com/sorryhyun/ComfyUI-Anima-BlockCompile | standalone at `~/ComfyUI-Anima-BlockCompile`, symlinked as `comfyui-anima-blockcompile` |
| DAVE | https://github.com/sorryhyun/ComfyUI-Anima-DAVE | standalone at `~/ComfyUI-Anima-DAVE`, symlinked as `comfyui-anima-dave`. One node for Anima and the `moss-image` transformer; `mask=auto` picks the bundled pool per model |
| MOSS Toolkit | https://github.com/sorryhyun/ComfyUI-MOSS-Toolkit (renamed from ComfyUI-MOSS-i2i; registry id `comfyui-moss-toolkit`, the old `comfyui-moss-i2i` id is retired), standalone at `~/ComfyUI-MOSS-Toolkit`, symlinked as `comfyui-moss-toolkit` | `MOSSImageLoader`, fp32 stream (`MOSSStreamFP32`), `MOSSLoRA` bypass (`lora_compat.py`, `DownRows`), i2i Reference, `MOSSCaptionHelper`. Its model patches are `APPLY_MODEL` wrappers: the OpenMOSS forward never calls `DIFFUSION_MODEL` wrappers |
| Anima Adapter Loader | https://github.com/sorryhyun/ComfyUI-Anima_lora-Adapter | Adapter / FeRA / Soft Tokens loaders, vocab pack, fp32 stream and INT8 (`stream.py`); standalone at `~/ComfyUI-Anima_lora-Adapter`. **Read its `CLAUDE.md` for the `forward_hook`-not-override invariant** |
| Pixel Decoder | https://github.com/sorryhyun/ComfyUI-Anima-PixelDecoder (registry id `comfyui-anima-pixel-decoder`, `node.zip` gitignored), standalone at `~/ComfyUI-Anima-PixelDecoder`, symlinked as `comfyui-anima-pixel-decoder` | `AnimaPixelHeadTap` (MODEL → MODEL: `CALC_COND_BATCH` step records, `DIFFUSION_MODEL` wrapper with a call-scoped pre-hook on the live `final_layer.linear`, post-CFG scale) + `AnimaPixelDecode` (MODEL, LATENT, head → IMAGE). `_vendor/pixel_head.py` is `library/anima/pixel_head.py` (vendor-sync). Timed in ComfyUI by `bench/pixel_readout/comfy_e2e.py` |
| Anima Tagger | inside the `anime_tools` repo (`comfyui/anima_tagger/` there) | symlinked as `comfyui-anima-tagger`; imports `anime_tools.tagger` and vendors nothing. The standalone `ComfyUI-Anima-Tagger` repo is **retired but still on disk**; don't edit it |

## In tree

Under `custom_nodes/`: `comfyui-anima-directedit/`, `comfyui-anima-register/`,
`comfyui-anima-trainer/` (daemon-backed one-shot trainer).

## Vendor trees and hand copies

`_vendor/` subsets and their targets are listed in the `scripts/release/sync_vendor.py`
docstring; the regeneration rule is in the root `CLAUDE.md` (GUI and ComfyUI nodes).

These files are hand copies of this repo's code or artifacts, not vendor-synced: change
or regenerate the source, then update the copy.

- Spectrum repo root: `cns_gamma.npz` and `moss_cns_gamma*.npz` are copies of
  `networks/calibration/` (Anima CNS γ ships there, no release download); copy after
  regenerating, then publish.
- DAVE: `dave_alpha.npz` is `networks/calibration/dave_alpha.npz`; `moss_dave_alpha.npz`
  carries `library/moss/calibration/{dave_alpha,massive_channels}.npz`.
- MOSS Toolkit:
  - `stream.py` mirrors `library/moss/loader.py::stream_fp32_`.
  - The ref forward (`reference.py`) mirrors `library/moss/model.py`, incl. `_ref_only`
    (under `ref_isolate` a `patches_replace`-skipped block still runs the ref rows).
  - Caption Helper: `is_layout_json` is a copy of
    `library/moss/caching/sources.py::is_layout_json`; `colorize`'s OCR lines copy
    `library/moss/colorize.py`'s `readable` / `dedupe` / layout; `FIXED_CAPTIONS` rows
    are each task's `configs/moss/i2i/<task>/cache.toml` `fixed_caption`; `token_split.py`
    is a copy of `library/moss/token_split.py`.

## Publishing

- MOSS Toolkit and the Anima Tagger node install `anime-tools` through their
  `requirements.txt`, one extra each (`[ocr]`, `[tagger]`) pinned to a release tag. When a
  release changes what a node imports, move its pin and republish.
- Run `make vendor-sync` first for a node with a `_vendor/` tree (root `CLAUDE.md`).
- Spectrum (`~/ComfyUI-Spectrum-KSampler`): check `git status` and ask before
  vendor-sync or publish; always bump the version. Its `pyproject.toml` is CRLF, so an
  anchored `sed` bump matches nothing; check `git show --stat`.
- Spectrum, ResShift and PiD track `node.zip`, and `comfy node publish` zips tracked
  files only and rewrites `node.zip`. Order: bump → `git add` new files → publish →
  check `unzip -l node.zip` → commit (zip included) → push.
- `comfy node publish --token $COMFY_REG` (from `.env`) echoes the PAT. The registry
  check flags `exec` / `eval`, tests included.
