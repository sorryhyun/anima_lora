# AdaLN LoRA — training → ComfyUI-compatible shipping

LoRA on the per-block modulation MLPs. Two paths produce one: extraction from a
full-model delta (`extract_delta_lora.py --include_adaln`), and training
(`train_adaln = true` in `configs/base.toml`, on for every LoRA-family method).
Both save in the ComfyUI key layout.

## What adaln is in this architecture

Anima inherits Cosmos-Predict2's per-block modulation: each `Block` has three
branch-modulation MLPs (`adaln_modulation_{self_attn,cross_attn,mlp}`) producing
shift/scale/gate from the timestep embedding only — no text, no spatial
conditioning. Every Anima checkpoint uses the AdaLN-LoRA bottleneck form
(`use_adaln_lora=True, adaln_lora_dim=256`):

```
SiLU → Linear 2048→256 (".1", down) → Linear 256→6144 (".2", up)
```

The name "AdaLN-LoRA" is NVIDIA's — it's a pretraining architecture choice
(low-rank bottleneck on the modulation MLPs), not an adapter. Full-rank adaln
would be 2048×6144 ×3 branches ×28 blocks ≈ 1.06B params; the 256-bottleneck is
≈176M. The weights (`blocks.{b}.adaln_modulation_{br}.1/.2.weight`) exist in
base and turbo alike.

### `use_adaln_lora=False` in diffusion-pipe

`models/cosmos_predict2_modeling.py` is vendored NVIDIA reference code (SPDX
NVIDIA header) and `False` is the upstream class-signature default for the
vanilla-DiT form. diffusion-pipe hardcodes `use_adaln_lora=True,
adaln_lora_dim=256` for every checkpoint it loads
(`models/cosmos_predict2.py::get_dit_config`, ~line 125). ComfyUI's detection
does the same (`comfy/model_detection.py:801`).

## Which trainers target adaln

| Trainer | adaln in LoRA target set? | Evidence |
|---|---|---|
| diffusion-pipe (Anima author's trainer) | YES — targets every `nn.Linear` under `Block`/`TransformerBlock`, no exclusions and no config filter | `models/base.py::configure_adapter` (~line 219) |
| sd-scripts (official kohya, upstream Anima support) | NO by default — appends `.*(_modulation\|_norm\|_embedder\|final_layer).*`; opt-in via `include_patterns` (include beats exclude, same mechanism as ours) | `sd-scripts/networks/lora_anima.py:254` |
| this repo | YES — `_DEFAULT_EXCLUDE` blocks `adaln_up_`, and `train_adaln = true` in `configs/base.toml` re-includes it via `include_patterns` on every LoRA-family method | `networks/lora_anima/config.py` (`_DEFAULT_EXCLUDE`, `from_kwargs`) |

Our exclusion is inherited from kohya's `lora_anima.py` (extended with the
runtime rename names). kohya's `sd-scripts/networks/convert_anima_lora_to_comfy.py`
handles adaln module names, so an sd-scripts user who opts in gets a
ComfyUI-shippable adaln LoRA.

diffusion-pipe saves in "ComfyUI format" (`diffusion_model.`-prefixed PEFT
keys, `models/cosmos_predict2.py::save_adapter`), so diffusion-pipe-trained
Anima LoRAs carry adaln and ComfyUI applies it; sd-scripts-trained ones default
to no adaln. On the diffusion-pipe side:

- Ecosystem LoRAs carry both `.1` and `.2` adaln LoRAs (both are Linears
  inside `Block`), plus the LLM adapter (`TransformerBlock` is in the target
  list) — more surface than our target set at equal rank.
- Loading an ecosystem LoRA in-repo via `--lora_weight` drops its adaln +
  LLM-adapter keys (different naming, excluded targets), so the same file
  renders differently here and in ComfyUI.

## ComfyUI support

ComfyUI needs no special-casing: `comfy/lora.py::model_lora_keys_unet` (generic
branch, ~line 191) maps every weight in the model's state dict to a
`lora_unet_<path-with-underscores>` key. Base Anima and turbo are the same model
class, so the same keys apply to both. An adaln-inclusive extraction fed through
ComfyUI's own `comfy.lora.load_lora` against a key map built from the real
checkpoint consumes every key: 364 patch targets = 280 attn/MLP + 84 adaln,
aimed at `diffusion_model.blocks.{b}.adaln_modulation_{br}.2.weight`.

## Key-naming contract

The same Linear has two names, and ComfyUI only knows one of them:

| Surface | Module path | LoRA key |
|---|---|---|
| Checkpoint / ComfyUI | `blocks.{b}.adaln_modulation_{br}.2` | `lora_unet_blocks_{b}_adaln_modulation_{br}_2` |
| This repo, runtime (post `_dit_rename_hook`, `library/anima/weights.py`) | `blocks.{b}.adaln_up_{br}` | `lora_unet_blocks_{b}_adaln_up_{br}` |

Runtime-named adaln keys are **silently dropped** by ComfyUI. Anything shipped
must use the comfy layout; anything consumed in-repo (warm-start, `--lora_weight`)
wants the runtime layout.

## Path 1 — extraction

`scripts/toolkits/extract_delta_lora.py` extracts adaln up-projections from a full-model
delta with `--include_adaln`, and `--adaln_layout comfy` renames to the ComfyUI
layout at save time (default `runtime` keeps warm-start compat; layout recorded
in `ss_delta_extract` metadata).

```bash
python scripts/toolkits/extract_delta_lora.py \
    --tuned models/diffusion_models/anima_turboV10.net.safetensors \
    --rank 96 --act_scales models/extracted/act_scales_base_4step.safetensors \
    --include_adaln --adaln_layout comfy \
    --out models/extracted/anima_turboV10_delta_r96_asvd_adaln.safetensors
```

Artifact: `models/extracted/anima_turboV10_delta_r96_asvd_adaln.safetensors`.

Delta facts (official turbo vs base):

- `.2` up-projs are the largest movers in the whole delta (rel-Δ 0.008–0.03);
  `.1` down-projs barely moved (0.0002–0.006) — the extractor takes `.2` only.
- `.2` in-dim is 256, so r=96 captures ~99% energy (recon rel-err ≈ 0.10);
  the residual infidelity lives in attn/MLP.
- Adding adaln lifts mean captured energy 0.803 → 0.860 vs the adaln-less
  r96 ASVD file (min stays 0.423, a non-adaln module).

## Path 2 — training

`include_patterns` beats the default exclude (`networks/lora_anima/network.py`
~line 245: `if excluded and not included: continue`), and runtime
`adaln_up_{br}` is a plain `nn.Linear`:

```toml
# method TOML — the raw primitive; `train_adaln = true` is the sugar for it
include_patterns = [".*adaln_up_.*"]
```

### The `train_adaln` surface (LoRA family)

`train_adaln` / `adaln_rank` / `adaln_alpha` are read by
`LoRANetworkCfg.from_kwargs` (`networks/lora_anima/config.py`) and desugar to
exactly the primitives above: `include_patterns += [".*adaln_up_.*"]`, plus
`network_reg_dims` / `network_reg_alphas` entries on the same pattern. `adaln_rank`
defaults to the network's rank (`configs/base.toml` sets 16); `adaln_alpha` is
derived from the network's own rank/alpha by the √r law below
(`network_alpha·√(adaln_rank/network_dim)`) whenever it is 0/unset — set it only
to override. `adaln_rank` / `adaln_alpha` without `train_adaln` raise.

They reach the network as top-level TOML keys, not `network_args` — the
allowlist is AST-derived from the `kwargs.get()` reads (`networks/__init__.py::
_derive_network_kwargs`) and `train.py::resolve_network_kwargs` forwards them. So
`configs/base.toml`'s `train_adaln = true` applies to every LoRA-family method
(lora / soup / byg) with no per-method opt-in. Verify a merge with
`make print-config METHOD=<m> PRESET=<p>`.

`soft_tokens` takes `**kwargs` and ignores the adaln keys, so the base default is
a no-op there.

### EasyControl

EasyControl reads the same three knobs. `configs/easycontrol/easycontrol.toml`
sets `train_adaln = true` (`adaln_rank = 16`); `configs/gui-methods/easycontrol.toml`
pins `train_adaln = false` — the pin is needed because `resolve_network_kwargs`
forwards base.toml's `train_adaln = true` to every network module. Unlike the
LoRA family, `adaln_rank`/`adaln_alpha` are inert when `train_adaln` is off (never
a raise — base always supplies them).

Shape: per-block `_LoRAProj(256 → 3·2048)` deltas on the target stream's
three `adaln_up_{br}` up-projections, applied inside the two-stream inner and
the cached-cond-KV inference path only — the no-cond fallback runs the
original `Block.forward`, so the delta is cond-gated by construction and
"no reference = exact baseline DiT" holds. The cond stream's own
modulation (`t_embedder(zeros)`) stays stock; zero-init up preserves step-0
equivalence; `multiplier` scales the delta (a strength knob at inference).
Alpha follows the √r law from the cond LoRA. adaln here is a t-only global prior
(e.g. colorize chroma/tone commitment); it cannot carry reference content.
Guards in `tests/test_easycontrol_adaln.py`. EC checkpoints are consumed by the
EC KSampler node (which reads the `adaln_lora_*` keys), not ComfyUI's LoRA
loader, so the comfy relayout does not apply.

### Turbo

`network.train_adaln` wires the include on student+fake;
`network.fake_adaln = false` builds an adaln-less critic (VRAM lever);
`network.adaln_rank = N` builds the adaln modules at their own rank via the
factory's `network_reg_dims` regex→dim override — on the r96 extract the adaln
ΔW keeps 0.980/0.991/0.995 of its energy at r32/48/64 (in-dim is 256). The
reg_dims path keeps the NETWORK alpha (alpha/rank scale on adaln is hotter by
rank/adaln_rank; warm start folds scale so init is exact). The extractor mirrors
it as `--adaln_rank`. The turbo loop pins its own `adaln_alpha` rather than
deriving it. Guards in `tests/test_turbo_adaln.py`.

### Sizing `adaln_rank` / `adaln_alpha`

Trained adaln deltas are strongly low-effective-rank. SVD spectrum of the
learned `.2` ΔW (mean over the 84 modules; energy fraction captured at rank r):

| Source | eff. rank | top-1 | r8 | r16 | r32 |
|---|:-:|:-:|:-:|:-:|:-:|
| Official turboV10, exact full-rank delta (full fine-tune, all 256 dims free) | 10.8 | 51% | 86% | 93% | 96% |
| r96 ASVD extract of it | 7.9 | 52% | 88% | 95% | 98% |
| superturbo_E student (alloc r16) | 8.6 | 43% | 87% | 96% | — |
| soup s1001 style LoRA (alloc r32) | 13.1 | 37% | 74% | 89% | 100% |
| same ckpt, attn/MLP for contrast (alloc r32) | 25.3 | 13% | 49% | 73% | 100% |

Half the energy sits in one direction regardless of objective (DP-DMD distill
vs style FM) or parametrization — a property of the pathway (global
tone/gate/σ-remap), not the task. attn/MLP nearly saturates whatever rank it is
given; adaln fills ~40–55% of it with a fast-decaying tail. `adaln_rank = 16`
keeps ~93% of even the official full-rank delta; matching a rank-64 network adds
~30M params of low-energy tail (4.3M at r8). The r32/48/64 extract-energy numbers
above are the extraction bar, not the training-allocation bar.

`adaln_alpha` follows the √r law, not linear α/r. Optimal LoRA α scales
sublinearly as α\*(r) ≈ C·√r ([LoRA-α, arXiv 2606.12883](https://arxiv.org/abs/2606.12883)),
so cross-rank scale preservation means keeping α/√r constant:

```
adaln_alpha = network_alpha · sqrt(adaln_rank / network_rank)
```

`from_kwargs` computes it whenever `adaln_alpha` is 0/unset, so the branch stays
scale-matched under any `network_dim`/`network_alpha` (including a CLI
`--network_alpha` override). E.g. student 180@r64 → adaln 90@r16 (both
α/√r = 22.5); the linear rule (45) would under-scale the smaller-rank module.
Only the relative √r consistency is used; the paper's absolute α_base ≈ 256√r
calibration is for fresh-init LLM SFT.

Compile interaction: under `compile_dynamic_seq`, an adaln LoRA gives the backward a
seq-axis reduction that trips inductor's mix-order-reduction 4096 guard, so
`compile_blocks` pins `triton.mix_order_reduction` off — this applies to any LoRA on a
broadcast-consumed Linear, not just adaln. Mechanism:
[`../optimizations/for_compile.md`](../optimizations/for_compile.md) §2.6.

Shipping trained adaln: the save pipeline relays runtime-layout keys to the
comfy layout — `lora_save.py::_relayout_adaln_to_comfy` runs on the
standard write path (after the qkv defuse, before hashing), stamps
`ss_adaln_layout = "comfy"`, and is presence-gated so adaln-less checkpoints are
untouched. The in-repo loader renames back on load
(`create_network_from_weights` → `relayout_adaln_comfy_to_runtime`), so one file
round-trips both ecosystems. Every method on the standard path is covered,
turbo included; the Hydra MoE `_moe` siblings are not ComfyUI-loadable
regardless.

Checkpoints trained with `train_adaln = false` (including every LoRA trained
here before it became the default) carry no adaln keys.

### What adaln can learn

t-conditioned global modulation only: per-σ tone/contrast/gating shifts.
No text or spatial pathway. Few-step distillation changes exactly this σ→behavior
map, and the official turbo moved adaln more than any other module.

### Interaction with modulation guidance (MOD=1)

The adaln modulation MLPs are a shared pipe with three tenants
(`../inference/mod-guidance.md`):

1. Stock: t-embedding → per-σ shift/scale/gate.
2. Mod-guidance: adds `schedule[ℓ]·δ` (pooled-text steering delta) to the
   t-embedding *upstream of each block's modulation MLPs* — the same MLPs
   transduce the delta into coefficient steering. It also installs
   `pooled_text_proj`, making the t-embedding itself pooled-text-aware.
3. AdaLN LoRA (this doc): changes the MLP weights themselves.

Consequences:

- An adaln LoRA reshapes mod-guidance's steering. Guided coefficients become
  `(W+ΔW)(t_emb + w·δ) = stock + ΔW·t_emb + w·ΔW·δ` — the `ΔW·δ` term re-rotates
  the calibrated steering direction. Mod-guidance schedules and the external
  Spectrum node's scalar defaults were tuned with stock `W`. The distortion is
  `‖ΔW‖/‖W‖` per block: ≤1–3% for the turbo extract, 3.6% for a hot-alpha style
  LoRA (soup s1001, alpha 128 @ r32). At 3.6% the adaln LoRA still transduces
  the steering delta unchanged (weight-space probe: norm ratio 1.000, cosine
  0.9999, all σ). Turbo inference runs CFG 1.0 / 4-step without mod-guidance.
- On a stock model adaln is text-blind. On a mod-distilled model
  (`pooled_text_proj` installed and loaded — it loads only in `load_dit_model`,
  gated on `--pooled_text_proj`) the t-embedding carries max-pooled text. A soup
  ingredient trained with the projection loaded frozen (`train.py
  --pooled_text_proj`) learned no extra text coupling (ΔU text-coupling index
  ≈0.003, same as the stock-trained arm) and became proj-coupled (~1.7× higher
  injection dependence at render, i.e. off-distribution in a stock ComfyUI flow
  without the proj).
