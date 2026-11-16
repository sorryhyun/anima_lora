# Anima DirectEdit (ComfyUI)

A ComfyUI node that edits an image by changing its caption. Type the source caption into `source_tag`, the edited caption into `target_tag`, and out comes the source image with that change applied — backgrounds, composition, and unchanged subject details preserved.

Built on **DirectEdit** (Yang & Ye, [arXiv:2605.02417](https://arxiv.org/abs/2605.02417v1)) — a training-free flow-inversion editor. Reference implementation: [Tr1stesse/DirectEdit](https://github.com/Tr1stesse/DirectEdit).

This node ports that idea to the Anima (DiT, flow-matching) model.

## What the node does

```
                source_tag ─► ψ_src ──┐
                                       │
                target_tag ─► ψ_tar ──┤   (ψ_tar empty → ψ_tar = ψ_src,
                                       │    reconstruction sanity check)
                       ┌───────────────┴─────────────────────┐
        IMAGE ───────► │ DirectEdit                          │
                       │   1. invert(image, ψ_src)           │
                       │      ─► z_inv, Δz                   │
                       │   2. edit_forward(z_inv[0], Δz, ψ_tar) │
                       │      ─► z_edit                      │
                       └─────────────────────────────────────┘
                                              │
                                              ▼
                                       edited LATENT
                                       (wire into VAEDecode)
```

Two passes through the DiT:

1. **Inversion** (clean → noise): step backward through the same Euler ODE the generator runs forward, querying v_θ at each step's input. Record per-step residuals `Δz_i = z_inv[i+1] − z_inv[i]` — these are the "anchor" the paper uses to reconstruct the source exactly.
2. **Editing** (noise → clean): standard generation loop, but every model call is queried at `z[i] + Δz[i]` instead of `z[i]`. The cross-attn prompt is the edit target ψ_tar; the residual Δz pins the trajectory to the source.

Result: regions the prompt *doesn't change* stay locked to the source; regions it *does* change get re-rendered under the target prompt.

## Breaking change in 0.2.0

The internal tagger / dispatcher was removed. ψ_src is now a plain STRING input (`source_tag`); ψ_tar is a plain STRING input (`target_tag`). The previous `tagger` / `prompt_src_override` / `edit_text` / `use_dispatcher` inputs are gone, and the debug `prompt_src` / `prompt_tar` STRING outputs are gone too (your upstream STRING nodes already carry the same values). Any node that emits a STRING can drive the captions — paste them, hand-type them, or wire in your captioner of choice. Older workflows that used the embedded tagger must be re-wired.

## Install

Drop `custom_nodes/comfyui-anima-directedit/` into your ComfyUI `custom_nodes/`. Restart ComfyUI; the node appears as **Anima DirectEdit** in the `anima` category.

The node works in two install shapes:

1. **Inside the anima_lora repo** (dev / monorepo). It imports the live `library.inference.directedit` etc., so edits in the parent repo are picked up immediately.
2. **Standalone** (just this directory dropped into a vanilla ComfyUI `custom_nodes/`). It falls back to a bundled inference subset under `_vendor/` — no need to clone the parent repo or run `uv sync`. Pip deps are minimal (`torch`, `numpy`, `pillow`, `tqdm`), all already present in any ComfyUI install.

### For maintainers — keeping the vendor copy fresh

The `_vendor/` tree is generated from the live anima_lora source. Regenerate it before bumping the node version:

```bash
python scripts/release/sync_vendor.py     # from the anima_lora repo root
```

## Inputs

| Input | Type | Notes |
|-------|------|-------|
| `model` | MODEL | Anima DiT (`UNETLoader` / `CheckpointLoaderSimple`). `LoraLoader` / `comfyui-hydralora` adapter loaders compose naturally upstream. |
| `clip` | CLIP | Anima text encoder (Qwen3 06B + T5xxl tokenizer) via `CLIPLoader`. |
| `vae` | VAE | Qwen Image VAE via `VAELoader`. |
| `image` | IMAGE | Source image. Auto-snapped to the closest `CONSTANT_TOKEN_BUCKETS` aspect ratio. |
| `source_tag` | STRING (multiline) | ψ_src — caption describing the source image. **Required (empty raises).** |
| `target_tag` | STRING (multiline) | ψ_tar — caption for the edited image. Empty → falls back to `source_tag` (reconstruction sanity check). Typical usage: copy `source_tag` and add / replace / remove tags. |
| `negative_prompt` | STRING | CFG negative for the edit pass. Default `"worst quality"`. |
| `infer_steps` | INT | Both inversion and edit step count. Default 20. |
| `flow_shift` | FLOAT | Sigma-shift schedule. Default 1.0 (Anima base-v1.0 standard). |
| `guidance_scale` | FLOAT | CFG for the edit (target) pass. Default 2.0. |
| `invert_guidance` | FLOAT | CFG during inversion. Default 1.0 (no CFG). |
| `t_inj` | INT | Number of early steps to inject src self-attn V into the tar pass. 0 = pure ΔZ-anchored edit; higher = stronger source-feature preservation, weaker edit leverage. Default 6. |
| `use_slot_surgery` | BOOLEAN (optional) | Transplant only the T5-diff-span slots of ψ_tar's crossattn_emb into ψ_src's encoding. Off by default. |

## Outputs

| Output | Type | Notes |
|--------|------|-------|
| `latent` | LATENT | Edited latent in raw VAE space (comfy convention). Wire into `VAEDecode` to render. |

## Usage

```
[UNETLoader] ──► model ──┐
[CLIPLoader] ──► clip ──┤
[VAELoader]  ──► vae ───┤
[Load Image] ──► image ─┤
                         ├─► [Anima DirectEdit] ──► [VAEDecode] ──► [Save Image]
                       source_tag: "1girl, school uniform, classroom"
                       target_tag: "1girl, school uniform, classroom, double peace"
```

For reconstruction sanity-checks, leave `target_tag` empty. The node logs `"reconstruction pass"` and runs the same caption through both branches; output should reproduce the source.

CLI equivalent (for reference):

```bash
make exp-test-directedit PROMPT='double peace'
```

## Caveats (v0)

- **No mask blending.** Background-lock via spatial mask is deferred — DirectEdit's `Δz` anchoring already preserves untouched regions in practice.
- **Inversion runs at `invert_guidance=1.0`** by default (no CFG). Raise only if you need the inverted noise to match a high-CFG generation seed.
- **Single-frame only.** Anima's qwen-image VAE is a video VAE that this pipeline drives at `T=1`. If you wire in a video-shaped IMAGE, only the first frame is processed.

## Files

| File | Role |
|------|------|
| `nodes.py` | The `AnimaDirectEdit` node — encode prompts → invert → edit_forward → decode. |
| `__init__.py` | Re-exports `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`. |
| `pyproject.toml` | ComfyUI Registry metadata. |
| `_vendor/` | Bundled inference subset (used when this node isn't sitting inside the anima_lora repo). Regenerated by `scripts/release/sync_vendor.py`. |

## References

- **DirectEdit paper.** Yang & Ye, "Direct flow-inversion image editing for rectified flow models." [arXiv:2605.02417](https://arxiv.org/abs/2605.02417v1).
- **Reference implementation.** [Tr1stesse/DirectEdit](https://github.com/Tr1stesse/DirectEdit) — original PyTorch reference, source for the inversion/edit-forward step rules ported here.
- **Anima editing pipeline.** `scripts/edit.py` (CLI) and `library/inference/editing/directedit.py` (primitives).
