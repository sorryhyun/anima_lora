---
name: custom-nodes
description: The ComfyUI node map — which node lives in which standalone repo vs in-tree under custom_nodes/, where each is symlinked, and the vendor-sync rule for the _vendor/ subsets. Load before editing, publishing, or hunting for a ComfyUI node, or before touching anything under custom_nodes/.
---

# ComfyUI custom nodes

Most nodes were extracted to standalone repos and are **symlinked** into
`../comfy/custom_nodes/`. Edit the source repo, never the symlink.

## Out of tree

| Node | Repo / path | Notes |
|---|---|---|
| Spectrum KSampler + mod-guidance | https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler | ships DCW scalar default `+0.01` + `auto` mode |
| PiD decode | https://github.com/sorryhyun/ComfyUI-Anima-PiD | full handoff 2026-06-04; symlinked as `comfyui-anima-pid` |
| EasyControl KSampler | `~/ComfyUI-EasyControl-KSamplerCompat` | |
| Block Compile | https://github.com/sorryhyun/ComfyUI-Anima-BlockCompile | moved out 2026-06-30; standalone at `~/ComfyUI-Anima-BlockCompile`, symlinked as `comfyui-anima-blockcompile` |
| Anima Adapter Loader | https://github.com/sorryhyun/ComfyUI-Anima_lora-Adapter | Adapter / FeRA / Soft Tokens loaders; standalone at `~/ComfyUI-Anima_lora-Adapter`. **Read its `CLAUDE.md` for the `forward_hook`-not-override invariant** |
| Anima Tagger | inside the `anime_tools` repo (`comfyui/anima_tagger/` there) | moved 2026-08-30, the standalone `ComfyUI-Anima-Tagger` repo is retired; symlinked as `comfyui-anima-tagger`; imports `anime_tools.tagger` and vendors nothing |

## In tree

Under `custom_nodes/`: `comfyui-anima-directedit/`, `comfyui-anima-register/`,
`comfyui-anima-trainer/` (daemon-backed one-shot trainer).

## Vendor trees

Several nodes carry a `_vendor/` subset of the live tree. **Regenerate with `make
vendor-sync` (`scripts/release/sync_vendor.py`), never `cp` by hand** — and re-run it
before every node publish.
