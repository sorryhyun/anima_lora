# Anima LoRA — DX Refactor Plan

Staged milestones aimed at making new experiments and new adapter methods cheap to add. Each milestone is independently landable; existing commands (`make lora`, `make tlora`, `make hydralora`, `make postfix`, `make test`) keep passing after every stage. Numerical behavior and checkpoint compatibility are preserved throughout.

> **Status:** M0 (test harness + importability), M1 (loss / sampler / metric registries), M2 (NETWORK_REGISTRY + `lora_save.py`), and M3 (`--print-config`, snapshot, schema validation) landed. `make test-unit` runs 69 tests green. Remaining: M4–M7.

---

## M4 — Sweep runner & unique output naming (~1 day)

`scripts/sweep.py` reads `configs/sweeps/<name>.toml` (parameter grids: `network_dim = [8, 16, 32]`), generates merged configs through the M3 machinery, derives deterministic `output_name` via a short hash of the diff against base (`anima_lora__dim16__alpha8__a3f21c.safetensors`), writes snapshots alongside checkpoints, exposes `make sweep NAME=rank_study`. Output-collision guard in `read_config_from_file` refuses to overwrite an existing `output/<name>.safetensors` unless `--overwrite`. No DB, just a `sweeps/<name>/runs.jsonl` ledger.

## M5 — `InferenceEngine` class extraction (~1 day)

Lift `inference.py` body into `library/inference/engine.py::InferenceEngine` with `load(weights_paths)`, `generate(prompts, opts)`, `attach_lora(path, multiplier)`. CLI in `inference.py` becomes a thin wrapper. Reuses M2 `NETWORK_REGISTRY` for LoRA/postfix/prefix detection instead of the current filename-based dispatch in Makefile (`LATEST_LORA`, `LATEST_HYDRA`, `LATEST_POSTFIX_*`). Enables: validation loop inside `train.py` using a tiny `InferenceEngine`, `custom_nodes/` integration, unit tests that generate a 64x64 dummy image.

## M6 — Attention backend registry

`networks/attention.py:216-477` → `ATTN_BACKENDS: dict[str, AttnBackendFn]` with entries for `torch`, `flex`, `flash`, `sageattn`, `xformers`. Each backend is a pure function `(q, k, v, attn_params) -> x`. Selection via `ATTN_BACKENDS[attn_params.attn_mode]`. Straight refactor, no new capabilities — numerical parity required. Enables adding a new backend (e.g. FlashAttention-3) by one-line registry insertion.

## M7 — Polish pass

- Delete `networks/lora_deprecated.py` import from `networks/lora_anima.py:17` (no configs in `configs/methods/` use `use_dora=true`).
- Deduplicate Makefile vs `tasks.py`: pick Makefile as source of truth, regenerate `tasks.py` from it or delete it.
- Add `docs/adding_a_method.md` — "copy a TOML, register a `NetworkSpec`, optionally register a loss."

---

## Sequencing

**Recommended order.** M6 → M5 → M4 → M7.

- **M5 before M4** — sweep's per-run validation wants a cheap `InferenceEngine`.
- **M6, M7 parallelizable** any time.

**Hard deps.** None.

---

## Out of scope (deliberate)

- **No DiT / `anima_models.py` surgery.** The user's request is DX around training/adapters, not model surgery; that file is the load-bearing numerical contract.
- **No Pydantic / attrs / Hydra / OmegaConf.** Stdlib `dataclass` + `dict` is enough; adding a config framework on a one-author research repo is net-negative DX.
- **No plugin discovery via `entry_points`.** `REGISTRY[name] = spec` module-local dicts meet every use case listed; entrypoints add packaging overhead for zero benefit.
- **No checkpoint format migration.** All registries preserve byte-identical `state_dict` keys; existing `models/` continues to load. A future format change (if desired) is a separate, clearly-flagged milestone.
- **No test coverage target beyond smoke-level.** Tests gate milestones, not a coverage number. Numerical-parity checks on one training step are the strongest signal we need.
- **No GRAFT branching or preprocessing incrementality.** Tier 3 audit items — parked explicitly.

---

## Critical files

- `/home/sorryhyun/anima_lora/train.py`
- `/home/sorryhyun/anima_lora/networks/lora_anima.py`
- `/home/sorryhyun/anima_lora/networks/postfix_anima.py`
- `/home/sorryhyun/anima_lora/networks/attention.py`
- `/home/sorryhyun/anima_lora/library/train_util.py`
- `/home/sorryhyun/anima_lora/library/training/__init__.py`
- `/home/sorryhyun/anima_lora/inference.py`
