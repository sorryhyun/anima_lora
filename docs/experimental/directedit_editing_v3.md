# DirectEdit (v3) — flow-inversion image editing on Anima

Successor to [`_archive/proposals/directedit_editing_v2.md`](../proposal/directedit_editing_v2.md).
v2 was the proposal; this doc covers what's actually wired and runnable
in the tree. The Anima Tagger arm of v2 ("phase v3.0") is documented
separately in [`anime_tools/docs/anima_tagger.md`](https://github.com/sorryhyun/anime_tools/blob/main/docs/anima_tagger.md)
(sibling checkout: `../anime_tools/docs/anima_tagger.md`).

## Status

| Component | State |
|---|---|
| `library/inference/editing/directedit.py` — invert + edit_forward primitive | **wired** |
| V-injection (paper Eq. 13) | **wired** in both CLI and ComfyUI node |
| ψ_src tagger (Anima Tagger), CLI side | **wired** |
| `scripts/edit.py` — standalone CLI | **wired** |
| `make exp-test-directedit` / `exp-test-directedit-dry` driver | **wired** |
| `comfyui-anima-directedit` ComfyUI node (stock MODEL/CLIP/VAE sockets) | **wired**, 0.2.0 took caption inputs as plain STRINGs (no embedded tagger / dispatcher) |
| Mask blending (paper Eq. 12) | **stub** — `--mask` accepted but ignored |
| Embedding inversion fallback (v2.1) | **deferred** — `archive/inversion/` not yet promoted |

## Method recap

Two passes through the frozen DiT (Yang & Ye, [arXiv:2605.02417v1](https://arxiv.org/abs/2605.02417v1)):

1. **Inversion** (clean → noise): step backward through the same Euler
   ODE the generator runs forward, querying `v_θ` at each step's input.
   Record per-step residuals `Δz_i = z_inv[i+1] − z_inv[i]` — these
   are the "anchor" the paper uses to make reconstruction bit-exact
   instead of trying to rectify the inversion path itself.
2. **Editing** (noise → clean): standard generation loop, but every
   model call is queried at `z[i] + Δz[i]` instead of `z[i]`. The
   cross-attn prompt is the edit target ψ_tar; the residual Δz pins the
   trajectory to the source. For `t_inj > 0` (paper Eq. 13), the first
   `t_inj` steps additionally evolve a parallel ψ_src branch and inject
   its self-attn V into the tar self-attn at all blocks except the last
   (SD3.5-style default).

Anima conventions used:

* `sigmas[0] = 1` (pure noise), `sigmas[T] = 0` (clean), per
  `library/inference/sampling.py::get_timesteps_sigmas`.
* Latents are 5D `[B, C, 1, H/8, W/8]` (frame dim of 1 — image, not
  video).
* DiT call signature mirrors what `generate_body` uses:
  `anima(latents, t_expand, embed, padding_mask=...)` where `embed` is
  already-preprocessed crossattn (post-T5, 512-padded).

## The primitive — `library/inference/editing/directedit.py`

Self-contained module (~414 LoC). Two public entry points consumed by
all three call sites (CLI, make-target driver, ComfyUI node):

| Function | Signature highlights |
|---|---|
| `invert(anima, z_clean, embed_src, embed_neg, sigmas, guidance_scale=1.0)` | Returns `(z_inv: List[Tensor], delta_z: List[Tensor])`. Iterates `i = T-1 .. 0`: `z_inv[i] = z_inv[i+1] + (sigmas[i] − sigmas[i+1]) · v_θ(z_inv[i+1], σ=sigmas[i+1], embed_src)`. CFG defaults to 1.0 — the source has no negative concept to push away from. |
| `edit_forward(anima, z_init, delta_z, embed_tar, embed_neg, sigmas, guidance_scale=4.0, embed_src=None, t_inj=0, t_inj_blocks=None, mask=None)` | Standard Euler from `z_init = z_inv[0]`, but `v_i = v_θ(z[i] + Δz[i], σ=sigmas[i], embed_tar)`. When `t_inj > 0`, first `t_inj` steps also evolve a parallel src branch (CFG=1, no neg) and inject its self-attn V into tar via `_v_injection_scope`. `mask=` reserved for paper Eq. 12 (currently warns + ignores). |

### V-injection plumbing (`_v_injection_scope`)

`_VInjectionState` carries a per-block V cache plus a mode flag
(`CAPTURE` / `INJECT`). For each editing step with `i < t_inj`:

1. `mode = CAPTURE`, run src forward → V tensors stashed per block.
2. `mode = INJECT`, run tar forward → cached V replaces the freshly
   computed V inside the patched `Attention.forward`.

The cache is overwritten each step (28 entries, no growth).

`_make_patched_self_attn_forward` builds a replacement
`Attention.forward` that routes V through `state.hook` before the
attention dispatcher. **Two backends are supported** so the same
primitive runs in both call shapes:

* `library/anima/models.py::Attention` (standalone CLI) — signature
  `(x, attn_params, context, rope_cos_sin=None)`, dispatches via
  `attention_dispatch.dispatch_attention`.
* `comfy/comfy/ldm/cosmos/predict2.py::Attention` (inside ComfyUI) —
  signature `(x, context=None, rope_emb=None, transformer_options={})`,
  dispatches via `self.compute_attention`.

The patcher detects via `hasattr(attn, "compute_attention")` and emits a
function whose signature matches the actual call site. Patching with
the wrong signature would raise `TypeError: ... unexpected keyword
argument 'rope_emb'` on the first edit step.

The scope context manager carefully tracks whether each `Attention` had
a pre-existing instance-level `forward` (vs the class method) and
restores by either reassigning or `del`-ing — a plain reassign would
leak instance state (and a method ref-cycle) across scopes.

When tar runs CFG (batch dim 2), the cached src V (batch dim 1) is
broadcast via `repeat_interleave` so the injection still aligns.

### The src CFG=1 invariant

The src capture pass is always run at CFG=1 (no `embed_neg`). Paper
Algorithm 1 doesn't apply CFG to the src branch, and capturing V from a
CFG-mixed branch would conflate ψ_src and the negative concept.

## Call sites

### 1. Standalone CLI — `scripts/edit.py`

```bash
python scripts/edit.py \
    --image path/to/source.png \
    --prompt_src "1girl, smile, school_uniform" \
    --prompt_tar "1girl, smile, school_uniform, double peace" \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --save_path output/tests/directedit/
```

Notable flags:

| Flag | Default | Notes |
|---|---|---|
| `--infer_steps` | 28 | Both inversion and edit step count. |
| `--flow_shift` | 1.0 | Anima base-v1.0 standard. |
| `--guidance_scale` | 2.0 | CFG on the edit (target) pass. |
| `--invert_guidance` | 1.0 | CFG during inversion. Raise only if you need the inverted noise to match a high-CFG generation seed. |
| `--t_inj` | 12 | First N steps inject src self-attn V into tar (paper Eq. 13). 0 = pure ΔZ-anchored edit. Typical paper setting `T/10..T/3`. Higher = stronger source-feature preservation, weaker edit leverage. |
| `--t_inj_blocks` | `all_but_last` | `all`, `all_but_last`, or comma/range string (`8-22`, `8,9,12,14-18`). |
| `--image_size` | bucket-snap | Defaults to closest `CONSTANT_TOKEN_BUCKETS` entry by aspect ratio. |
| `--cached_embed` | unset | Sanity-check mode: load preprocessed `_anima_te.safetensors` cache and run one invert + edit pass per stored variant with ψ_tar == ψ_src. Skips the text encoder entirely. |
| `--cached_embed_variants` | `all` | Selector for `--cached_embed`. `all` sweeps every stored variant; otherwise comma-separated indices (`0`, `0,2`). |
| `--mask` | unset | Reserved — paper Eq. 12 background-lock blend. Currently warns + ignored. |
| `--compile_blocks` | on | Per-block torch.compile of `_forward`. Auto-disabled when `--t_inj > 0` because the V-injection scope monkey-patches `Attention.forward` and would invalidate the dynamo graph cache on every block on the first call. |

The script orchestrates the full pipeline:

1. Load source image, snap to bucket if `--image_size` unset.
2. Tokenize/encoding strategies (matches `inference.py`).
3. Load DiT (needed by `prepare_text_inputs`'s `_preprocess_text_embeds`).
4. Encode `prompt_src` / `prompt_tar` / `negative_prompt` (or load from
   `--cached_embed` cache). Logs `|src-tar|` / `|src-neg|` / `|tar-neg|`
   abs-mean diffs as a sanity check that the encoder path is doing its
   job; near-zero `|src-tar|` flags an empty/identical caption.
5. VAE-encode source image to clean 5D latent (`[1, C, 1, H/8, W/8]`).
6. Sigma schedule via `get_timesteps_sigmas`.
7. Per-variant: `directedit.invert` → `directedit.edit_forward`. Variants
   collapse to a single pass when `--cached_embed` is unset.
8. Re-mount VAE, decode each `z_edit`, save with `save_images`.

### 2. Make-target driver — `make exp-test-directedit{,-dry}`

Lives in `scripts/experimental_tasks/inference.py` (`cmd_test_directedit`
and `cmd_test_directedit_dry`).

**`make exp-test-directedit`** picks a source image, runs the Anima
Tagger to seed `--prompt_src`, builds `--prompt_tar = src + ", " + edit`,
and invokes `scripts/edit.py`:

```bash
make exp-test-directedit PROMPT='double peace'                # random source from post_image_dataset/resized/
REF_IMAGE=foo.png make exp-test-directedit PROMPT='glasses'   # explicit source
python tasks.py exp-test-directedit foo.png --prompt 'smile'  # positional source
```

Requires the Anima Tagger checkpoint at
`models/captioners/anima-tagger-v5/model.safetensors`; otherwise the
task exits with an instruction to train it via `python -m
scripts.anima_tagger.cli`. After the edit completes, the source is
copied next to the output as `<name>_src.png` for side-by-side review.

**`make exp-test-directedit-dry`** is the reconstruction sanity check —
auto-resolves the source's `_anima_te.safetensors` cache (the file
`cache_text_embeddings.py` writes), passes it via `--cached_embed`, and
runs invert + edit with ψ_tar == ψ_src. Output should reconstruct the
source. With `--caption_shuffle_variants N` caches, this sweeps v0
(pristine) + v1..v{N-1} (tag-shuffled re-encodings) — divergence across
variants flags numeric drift in `invert`/`edit_forward`. Bypasses both
the tagger and the text encoder.

`_filter_inference_base_for_edit` strips generation-only flags from
`INFERENCE_BASE` (`--prompt`, `--seed`, `--image_size`, `--infer_steps`,
…) so `scripts/edit.py` keeps its own defaults; only model/path flags
(`--dit`, `--text_encoder`, `--vae`, `--vae_chunk_size`, `--attn_mode`,
`--vae_disable_cache`) are forwarded.

### 3. ComfyUI node — `custom_nodes/comfyui-anima-directedit/`

`AnimaDirectEdit` consumes stock ComfyUI sockets: `MODEL` (DiT via
`UNETLoader`/`CheckpointLoaderSimple`), `CLIP` (Anima Qwen3 + T5 via
`CLIPLoader`), `VAE` (Qwen Image VAE via `VAELoader`), `IMAGE`. As of
0.2.0 (the "remove tagger implant" pass), ψ_src and ψ_tar are plain
STRING inputs (`source_tag`, `target_tag`) — any node that emits STRING
can drive them. Empty `target_tag` falls back to `source_tag` for the
reconstruction sanity check. Empty `source_tag` raises.

Returns `(LATENT,)` — wire it into `VAEDecode` to render. Returning
latent (rather than IMAGE) keeps the node composable with downstream
KSampler-style refiners. The pre-0.2.0 debug `prompt_src` / `prompt_tar`
STRING outputs were dropped; the same values are already on canvas
upstream of the node's STRING inputs.

#### What 0.2.0 removed

The 0.1.x node embedded captioning + edit-target derivation inside
`AnimaDirectEdit` itself:

* `tagger` (`ANIMA_TAGGER`) socket — ran `AnimaTagger.predict_caption`
  on the image to derive ψ_src.
* `prompt_src_override` STRING — escape hatch when the user wanted to
  paste ψ_src directly.
* `edit_text` STRING — short edit instruction; ψ_tar built as
  `psi_src + ", " + edit_text` (or via the dispatcher below).
* `use_dispatcher` / `replace_threshold` / `replace_gap` — ran
  `library.inference.edit_dispatcher.derive_target_caption`, which used
  Qwen3 last-pool cosine similarity to choose between REPLACE / REMOVE
  / APPEND from `edit_text` against an existing tag in ψ_src.

All of the above were removed in 0.2.0. The node is now agnostic about
where its caption strings come from — pipe in `AnimaTaggerCaption` from
the sibling `comfyui-anima-tagger` package if you want image-driven
captioning, paste the original generation prompt, hand-type the
captions, or run any other STRING-producing node. The dispatcher's
intent (RANK / REPLACE / REMOVE / APPEND from a single edit instruction)
becomes the caller's responsibility — typically expressed as a hand-
edited `target_tag` that mirrors `source_tag` with the relevant tag(s)
added, swapped, or removed.

#### Pipeline

Mirrors `scripts/edit.py` but routes through ComfyUI's CLIP socket:

* `_encode_prompt_comfy` — `clip.tokenize` →
  `clip.encode_from_tokens(return_dict=True)` → mirror
  `model_base.Anima.extra_conds` shape conventions →
  `unet.preprocess_text_embeds(cond, t5xxl_ids, t5xxl_weights=...)` for
  the LLMAdapter + 512 pad. Verified to land within numerical noise of
  the library's `prepare_text_inputs` for the prompts DirectEdit feeds.
* `vae.encode` → `model.model.process_latent_in` (Anima DiT was trained
  on standardized latents — comfy's KSampler applies this via
  `samplers.py::process_latent_in`, but since we call the diffusion
  model directly we must do it ourselves; without this the model sees
  OOD inputs and emits ~zero velocity, so the Δz-anchored Euler step
  reconstructs the source regardless of `t_inj` / `guidance_scale`).
* `directedit.invert` → `directedit.edit_forward` (with the same
  `t_inj` / `t_inj_blocks=None` defaults as the CLI).
* `model.model.process_latent_out` to return raw VAE-space LATENT in
  the comfy convention.

Two-phase progress bar: invert occupies `[0, T)`, edit occupies
`[T, 2T)`.

#### Install shapes + vendor

The package supports two install shapes:

1. **Inside the anima_lora repo** (dev / monorepo). Imports the live
   `library.inference.directedit` etc.
2. **Standalone** (just dropped into vanilla ComfyUI `custom_nodes/`).
   Falls back to a bundled inference subset under `_vendor/`,
   regenerated by `python scripts/release/sync_vendor.py` from the live tree
   before bumping the node version. The vendor resolution drops any
   partially-imported `library.*` modules from `sys.modules` before
   re-importing so vendor copies actually take effect.

After 0.2.0, the directedit vendor only carries `library/inference/
directedit*.py`, the trimmed `sampling.py` / `buckets.py`, and a stub
`library/anima/models.py` — no `captioning/`, no `vision/`, no
`edit_dispatcher.py`. Pip deps shrank to `torch / numpy / pillow / tqdm`
(all ComfyUI-bundled). `scripts/release/sync_vendor.py` reflects this split:
the previously-shared `SHARED_*` lists were renamed `TAGGER_*` since
only the tagger node's vendor needs them.

## What hasn't shipped (deferred from v2)

### v2.1 — embedding inversion as a premium fallback

`archive/inversion/invert_embedding.py` already does per-image gradient
descent on ψ_src to minimize FM loss through the frozen DiT — ground-truth
quality at the cost of minutes per image. Not yet wired into the edit
pipeline. Intended use: a (not-yet-built) `psi_src_mode=invert` option in
`scripts/edit.py` for users willing to wait for max fidelity.

Move-out-of-archive needed; otherwise no new code.

### v2.2 — img2emb

Defers per the v2 plan to "only if the tagger arm + v2.1 don't cover the
use cases." Has its own design doc in
[`_archive/proposals/img2emb_plan.md`](../proposal/img2emb_plan.md). The
tagger arm shipped first to avoid the failure mode the archived img2emb
hit — solving the cheap problem (Anima-distribution vocabulary) before
attempting the hard one (manifold-correct continuous embeddings).

### Mask blending (paper Eq. 12)

`scripts/edit.py` and `library.inference.directedit.edit_forward` both
accept a `mask=` argument that currently warns and is ignored. The
intent is per-step background-lock blending: `z_new = mask * z_anchor +
(1-mask) * z_predicted` so regions outside the edit area stay pinned to
the source-anchor latent. Adding it is purely a per-step linear
combination on top of the existing Δz step rule; not blocked by anything
upstream.

### Other gaps

* **No bench harness.** A directedit bench per the standard envelope
  (cf. `bench/_common.py::write_result`) is the next add. Should
  compare edit-success rate vs. ψ_src reconstruction fidelity across
  tagger arms (Anima vs wd) and `t_inj` settings.

## Validation

Two-tier:

* **Reconstruction** (`make exp-test-directedit-dry`): ψ_tar == ψ_src
  must reconstruct the source. With multi-variant caches, all variants
  must agree pixel-wise within numeric noise. This isolates
  `invert`/`edit_forward` correctness from edit semantics. Failure here
  points at: VAE standardization (the `process_latent_in` issue the
  ComfyUI node hit), text-encoder padding (must be max-padded, not
  trimmed — see CLAUDE.md "Text encoder padding"), or sigma-schedule
  drift.
* **Edit leverage** (`make exp-test-directedit PROMPT='...'`): with a
  good ψ_src and a clear edit instruction, the edit must apply locally
  while preserving non-edited regions. Tagger choice matters here —
  swap `TAGGER=anima` ↔ `TAGGER=wd` to attribute leverage failures to
  ψ_src manifold-fit vs. the editing primitive.

The CLI logs `|src-tar|` abs-mean as a quick sanity signal: near-zero
means the encoder path collapsed (likely empty/identical caption); far
from zero means ψ_src and ψ_tar are doing different things and the edit
result is on the editor, not the encoder.

## References

* **Paper.** Yang & Ye, "Direct flow-inversion image editing for rectified
  flow models." [arXiv:2605.02417v1](https://arxiv.org/abs/2605.02417v1).
* **Reference implementation.** [Tr1stesse/DirectEdit](https://github.com/Tr1stesse/DirectEdit) — original PyTorch
  reference; source for the inversion / edit-forward step rules and the
  V-injection scheme.
* **Tagger arm.** [`anime_tools/docs/anima_tagger.md`](https://github.com/sorryhyun/anime_tools/blob/main/docs/anima_tagger.md).
* **v2 design doc.** [`_archive/proposals/directedit_editing_v2.md`](../proposal/directedit_editing_v2.md).
