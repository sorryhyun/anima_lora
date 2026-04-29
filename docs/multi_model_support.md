# Multi-Model Support

Sketch of what it would take to add a second image-generation model (e.g. Z-Image-Base) alongside Anima in this repo. This is a repo-wide architectural note, not a method deep-dive — it lives at the top of `docs/` rather than under `docs/methods/`.

Status: exploratory. Nothing here has been implemented; this is a terrain map and a recommended boundary so the conversation about "should we do it" can be concrete.

## Current coupling

The repo has surprisingly clean bones in some places and surprisingly tight Anima coupling in others. Map by area:

| Area | File(s) | Anima-specific? | Effort to decouple |
|------|---------|-----------------|--------------------|
| Attention dispatch | `networks/attention_dispatch.py` | No — generic `(B, H, L, D)` | none |
| VAE loader | `library/models/qwen_vae.py` | No — generic loader | none |
| Strategy base classes | `library/anima/text_strategies.py` | No — abstract base | small (move up) |
| Anima strategy subclasses | `library/anima/strategy.py` | Thin wrapper around base | small |
| Configs | `configs/base.toml` | Mostly generic; just paths | trivial |
| LoRA target regex | `networks/lora_anima/network.py` | Hardcodes `blocks\.(\d+)\.` | small |
| Cache filename suffixes | `library/io/cache.py` | `_anima.npz`, `_anima_te.safetensors` | small |
| Weights loader | `library/anima/weights.py` | Fixed config dict + Anima-specific rename hooks | medium |
| DiT class | `library/anima/models.py` | Monolithic Anima architecture (~2700 LOC) | medium-large |
| Trainer forward path | `train.py::get_noise_pred_and_target` | Reaches into `unet.llm_adapter`, `_mod_guidance_*`, fused-projection assumptions, Anima's cross-attention LSE invariant | **large — biggest blocker** |
| Adapter monkey-patches | `networks/methods/{ip_adapter,easycontrol,postfix}.py` | Target Anima's exact module names + cross-attention shape | large (per adapter) |

## Proposed boundary

A `library/models/<family>/` namespace with a small protocol the trainer talks to.

```
library/models/
├── base.py                 # ModelFamily protocol
├── factory.py              # load_family(name) -> ModelFamily
├── anima/
│   ├── dit.py              # was library/anima/models.py
│   ├── weights.py          # load_dit / load_text_encoder / load_vae
│   ├── strategy.py         # tokenize + encode strategies
│   └── lora_targets.py     # block regex, exclude list, scale rules
└── zimage/
    └── ... mirrors anima/
```

Protocol roughly:

```python
class ModelFamily(Protocol):
    name: str
    cache_suffix: str                # "_anima" vs "_zimage"
    latent_channels: int

    def load_dit(self, args) -> nn.Module: ...
    def load_text_encoder(self, args) -> nn.Module: ...
    def load_vae(self, args) -> nn.Module: ...

    def tokenize_strategy(self) -> TokenizeStrategy: ...
    def text_encoding_strategy(self) -> TextEncodingStrategy: ...

    def lora_target_spec(self) -> LoRATargetSpec: ...   # block regex + excludes
    def forward_for_loss(self, dit, latents, text_emb, t, **kw) -> Tensor: ...
```

`train.py` selects via a new `model_family = "anima"` key in `configs/base.toml` and never imports `library.anima.*` directly. `cache_latents.py` / `cache_text_embeddings.py` read `family.cache_suffix` so Anima and Z-Image caches coexist in `post_image_dataset/` without colliding. The LoRA factory in `networks/lora_anima/` reads `lora_target_spec()` instead of hard-coding `blocks\.(\d+)\.`.

Most layers are already close to this shape: `attention_dispatch.py` is fully generic, the strategy base classes are clean, the VAE loader is generic, `configs/base.toml` is mostly model-agnostic. The work is renames and parameterization, not deep surgery — *for the parts above the trainer line*.

## The main blocker

**`train.py::get_noise_pred_and_target` and the wider trainer forward path** — not the DiT class itself.

The DiT can be solved with a factory in an afternoon. The real pain is that `train.py` reaches *into* the Anima DiT for adapter-specific knobs:

- `unet.llm_adapter` direct access
- `unet._mod_guidance_*` distillation hooks
- Fused-projection assumptions (`qkv_proj`, `kv_proj`)
- The cross-attention LSE-correction path that depends on Anima's `crossattn_full_len` invariant (see `attention_dispatch.py`)
- The postfix / IP-Adapter / EasyControl monkey-patches that target Anima's exact module names
- The 4096-patch constant-token bucketing built into `library/datasets/`

Plus every `configs/methods/*.toml` LoRA target list was hand-tuned against Anima block names.

So the abstraction boundary has to extend past "load the model" into "how does an adapter attach + how does a training step run." Z-Image will likely have a different cross-attention shape (probably no LLMAdapter, different text-encoder fan-in), which means the `forward_for_loss` slot on the protocol is the load-bearing one — and getting it right means picking which adapters you want to support on Z-Image day one.

## Adapter portability

Not all adapter families port equally. Rough triage:

| Adapter | Portability to a new DiT | Why |
|---------|--------------------------|-----|
| LoRA / OrthoLoRA / DoRA | High | Operates on any `nn.Linear`; only target list needs to change |
| ReFT | High | Wraps block forwards; needs a block-naming pattern only |
| HydraLoRA | High-medium | Same target story as LoRA + a router on the Linear's input |
| T-LoRA | High | Timestep mask is model-agnostic |
| APEX | Medium | Needs the trainer to do 3 forwards + warm-start LoRA target compatibility |
| Modulation guidance | Medium-low | Assumes AdaLN coefficients of a specific shape; needs `pooled_text_proj` slot |
| Postfix / Prefix | Low | Hardcoded to Anima cross-attention shape and module names |
| IP-Adapter | Low | Per-block `to_k_ip` / `to_v_ip` parallel projections + Anima cross-attn patch |
| EasyControl | Low | Two-stream block forward + per-block cond LoRA + scalar gate, all bound to Anima block internals |

Day-one Z-Image with **LoRA + ReFT + HydraLoRA** is realistic. Porting IP-Adapter / EasyControl / postfix is a per-adapter project against the new model's attention layout.

## Effort estimate

Realistic scope:

- **3–5 days** for LoRA-only Z-Image (factory + config + cache suffixes + LoRA target spec + Z-Image DiT class + strategy subclass).
- **~2 weeks** if all adapter families need to port across.

The factory itself is half a day; the rest is the long tail of "where else did Anima leak."

## Recommended next step before committing

Read Z-Image-Base's actual block structure and decide:

1. Does its self-attention / cross-attention layout look enough like Anima's that `attention_dispatch.py` runs unchanged?
2. Does it use a separate text encoder + VAE pipeline we can cache the same way (different suffix, same flow)?
3. Are its block names regular enough to fit a `lora_target_spec()`?

If all three are yes, this is a ~week of work. If any are no, scope grows accordingly — most likely (3) is fine, (2) is fine, and (1) is the swing factor.

## Out of scope for this doc

- The actual `ModelFamily` interface design (this is a sketch; real signatures will fall out of the first port).
- Backward compatibility for existing `_anima.npz` caches (a config-default keeps them working).
- Whether to share or fork preprocessing scripts (`preprocess/cache_*.py`) — likely share, parameterized by family.
- ComfyUI custom-node story for a second model — separate concern.
