# Anima Register Adapter — ComfyUI node

Run a **register-token adapter** (DSR-style non-decoded self-attention tokens +
trained self-attn QKV surface) on a stock Anima MODEL. The training method
(`train.py --method register`) was removed from the trainer; this node still loads
adapters trained before that. It was the inference/eyeball vehicle for the
`_archive/proposals/headroom_register_tokens.md` line — there is no automated Anima
quality reward, so a human-in-ComfyUI A/B is the quality gate.

## Node

| Node | In → Out | Purpose |
|------|----------|---------|
| **Anima Register Adapter** | `MODEL`, `adapter_name` (+ `path_override`, `strength`) → `MODEL` | Loads a register adapter and patches the model so a downstream KSampler runs with registers live. |

`adapter_name` is a dropdown of register adapters discovered under `output/ckpt/`,
`output/temp/`, or `bench/headroom/results/` (filtered by the `ss_num_registers`
header, so unrelated safetensors don't show). `path_override` (STRING) loads one
from anywhere. `strength` scales the register tokens + QKV ΔW (1.0 = as trained).

## Wiring

```
UNETLoader ─► Anima Register Adapter ─► KSampler ─► VAEDecode
```

## LoRA-family checkpoints (LoRA + registers)

A LoRA trained with `num_registers > 0` in a lora-family TOML (see
`configs/methods/lora.toml` §Register tokens) saves standard `lora_unet_*`
keys plus one top-level `register_tokens` tensor and
`ss_num_registers` / `ss_register_insert_block` stamps. The node detects that
layout and applies **both halves**: the LoRA ΔW via ComfyUI's standard weight
patcher (incl. `inv_scale` folding for channel-scaled checkpoints) and the
registers via the eager register forward — no separate LoRA loader needed:

```
UNETLoader ─► Anima Register Adapter ─► KSampler ─► VAEDecode
                (ΔW + registers, one .safetensors)
```

**Do NOT also load the same file through a LoRA loader** — the ΔW would apply
twice. One `strength` scales both halves, matching in-repo inference's single
`--lora_multiplier`. Registers at strength 0 still occupy K sequence slots
(zero-token attention-sink lesion), matching training semantics.

## Comfy-native (the q/k/v fix)

The training-side adapter patches the DiT's **fused** `self_attn.qkv_proj` and
wraps `_run_blocks`. ComfyUI's Anima runs `comfy.ldm.cosmos.predict2.MiniTrainDIT`,
which has **split** `q_proj`/`k_proj`/`v_proj` and an inline block loop (no
`_run_blocks`). So `register_apply.py` is a from-scratch reimplementation against
the comfy backbone (the pattern the EasyControl node used — no vendored training
code):

* The checkpoint stores the QKV surface **already split** into q/k/v (a `lora`
  mode is folded to a single `(inner, D)` ΔW per component on load), so each
  delta adds directly onto its own projection — zero fusion logic.
* Register tokens are concatenated on the self-attn sequence, carried through the
  blocks, and stripped before the final layer. Comfy keeps the `(B,T,H,W,D)`
  patch grid, so the forward is reimplemented to native-flatten to `(B,1,L+K,1,D)`
  around the block loop (valid because Anima is image-only, `T==1`). Registers
  are rope-exempt: the cosmos rope is a per-position `(head_dim//2, 2, 2)` stack
  of 2×2 rotation matrices, so register rows get the **2×2 identity**.

## Constraints

* **Do NOT stack with the Block Compile node** — the register mechanism runs
  eager; a compiled block graph is not supported.
* Supports both trained arms: `qkv_mode="lora"` (low-rank) and `qkv_mode="unfrozen"`
  (full-rank ΔW) — read from the checkpoint `ss_qkv_mode` metadata.
* The adapter is (re)built and applied at the first sampling step against the
  resident `diffusion_model`, so it survives ComfyUI's clone/reload/recompile.

## Status

The mechanism is verified headless — with K=0 and a zero QKV delta the
reimplemented forward is **bit-exact** to comfy's original `MiniTrainDIT._forward`
(the native-flatten reshape + rope handling round-trip losslessly), K>0 strips
registers and changes the output, and the split q/k/v ΔW patch is live. Still
smoke-test end-to-end against a live ComfyUI instance (real weights, one sampled
image that decodes) before relying on it.
