---
name: embedder-api
description: The anima_lora programmatic façade for embedders — namespaces and lazy re-exports, the request-driven GenerationRequest inference path, adapter family from checkpoint metadata, and ANIMA_HOME path anchoring. Load before writing or editing an examples/ script, embedding the trainer in another program, or changing what anima_lora re-exports.
---

# Programmatic API (`anima_lora`)

`uv sync` installs the repo editable, so `anima_lora` is importable anywhere. Runnable
scripts live in `examples/` (`01`–`04` high-level flows, `05`–`06` raw primitives);
`examples/README.md` is the embedder guide.

## It's a thin façade

Canonical homes are unchanged — `library.inference`, `library.config.io`,
`library.anima.weights`, `library.models.qwen_vae`, `library.runtime.device`.
`anima_lora/__init__.py` is a lazy (PEP 562) re-export of the curated entry points,
grouped into namespaces `anima_lora.{models, inference, config, training, captioning}`.
Pre-namespace flat names (`anima_lora.generate`) stay as aliases. `ROOT` is the repo root.
`anima_lora.training` loads repo-root `train.py` **by path**, so `AnimaTrainer` etc. work
from any CWD.

Import `anima_lora` instead of reverse-engineering a `main()`.

## Inference is request-driven

Build a typed `GenerationRequest` (`library/inference/request.py`) and call `.to_args()`,
which routes through `inference.parse_args` so every `getattr()`-read knob is populated.
Long-tail method flags ride `extra_argv`.

**Adapter family lives in the checkpoint metadata, not the call** — the DiT loader
merges-or-keeps-live accordingly. Prompt encoding installs two process-global strategy
singletons lazily (`ensure_text_strategies`).

## Path anchoring — `ANIMA_HOME`

Repo-relative model/config paths resolve against the **repo home**
(`library.env.anima_home()` / `resolve_under_home()`), **not the CWD** — that's what makes
`import anima_lora` work from any directory. Set `ANIMA_HOME` for a relocated checkout, or
override individual model paths with `ANIMA_DIT` / `ANIMA_VAE` / `ANIMA_TEXT_ENCODER`.

The anchor is wired at the config-loader chokepoint (`library/config/io.py`) and at the
model-loader leaves (`load_anima_model` / `load_vae` / `load_qwen3_text_encoder`).
**New code opening a repo-relative path should call `resolve_under_home()`** rather than
assuming CWD.
