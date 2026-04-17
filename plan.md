# Anima LoRA — DX Refactor Plan

Staged milestones aimed at making new experiments and new adapter methods cheap to add. Each milestone is independently landable; existing commands (`make lora`, `make tlora`, `make hydralora`, `make postfix`, `make test`) keep passing after every stage. Numerical behavior and checkpoint compatibility are preserved throughout.

> **Status:** M0 (test harness + importability) and M1 (loss / sampler / metric registries) landed. `make test-unit` runs 20 smoke tests green. Remaining: M2–M7.

---

## M2 — `NETWORK_REGISTRY` for adapter-method dispatch (priority 2)

**Goal.** Replace flag-dispatch soup at `networks/lora_anima.py:492-500` and mode-by-filename inference at `networks/postfix_anima.py:84-112` with a registry.

**Target shape.**

```
networks/__init__.py
  @dataclass NetworkSpec:
      name: str
      module_class: type              # LoRAModule / OrthoLoRAModule / HydraLoRAModule / ...
      kwarg_flags: tuple[str, ...]    # flags this entry consumes
      post_init: Callable[[LoRANetwork, dict], None] | None = None
      save_variant: str = "standard"  # keys into SAVE_HANDLERS
      serialization_meta: dict = ...  # goes into safetensors metadata header

  NETWORK_REGISTRY: dict[str, NetworkSpec] = {
      "lora":            NetworkSpec("lora",            LoRAModule,                (),                                      None,          "standard"),
      "ortho":           NetworkSpec("ortho",           OrthoLoRAModule,           ("sig_type","ortho_reg_weight"),         _attach_ortho, "ortho_to_lora"),
      "ortho_exp":       NetworkSpec("ortho_exp",       OrthoLoRAExpModule,        (),                                      None,          "ortho_exp_to_lora"),
      "hydra":           NetworkSpec("hydra",           HydraLoRAModule,           ("num_experts","balance_loss_weight"),   _attach_hydra, "hydra_moe"),
      "ortho_hydra_exp": NetworkSpec("ortho_hydra_exp", OrthoHydraLoRAExpModule,   ("num_experts","balance_loss_weight"),   _attach_hydra, "ortho_hydra_to_hydra"),
      "dora":            NetworkSpec("dora",            DoRAModule,                (),                                      None,          "standard"),
  }

  resolve_network_spec(kwargs) -> NetworkSpec
      # fixed precedence order (documented in module docstring);
      # raises on ambiguous combos (e.g. use_ortho + use_ortho_exp).

networks/lora_save.py  (extracted from ~400-line save_weights @ lora_anima.py:1620+)
  SaveHandler = Callable[[dict[str, Tensor], dict], dict[str, Tensor]]
  SAVE_HANDLERS: dict[str, SaveHandler] = {
      "standard":             _save_standard,
      "ortho_to_lora":        _save_ortho_via_svd,
      "ortho_exp_to_lora":    _save_ortho_exp_cayley,
      "hydra_moe":            _save_hydra_moe,
      "ortho_hydra_to_hydra": _save_ortho_hydra,
  }
```

`create_network` (`lora_anima.py:505`) reduces to: resolve spec → build `LoRANetwork(module_class=spec.module_class)` → `spec.post_init(network, kwargs)` → stamp `network._network_type = spec.name`.

`save_weights` becomes: `SAVE_HANDLERS[self._network_spec.save_variant](state_dict, ctx)` then the shared qkv/kv defuse.

`postfix_anima.py`: stamp `ss_network_spec = "postfix"` (plus `ss_mode`, `ss_splice_position`, `ss_cond_hidden_dim`) into metadata on save. Keep the key-sniffing block as legacy-checkpoint fallback.

**Checkpoint compatibility.** On-disk format unchanged — all registered `save_variant` handlers produce the same keys/shapes as today. Existing `models/` checkpoints continue to load because `create_network_from_weights` already reconstructs from key shapes; we just route it through the registry.

**Verify.**
- `tests/test_network_registry.py`: every key in `configs/methods/*.toml` resolves to a `NetworkSpec` without warnings; ambiguous combos raise.
- Round-trip: build each adapter type, `save_weights` to tmp, `create_network_from_weights`, assert bit-identical state_dict keys and shapes.
- Load two known-good checkpoints from `models/`: one standard LoRA, one hydra `_moe.safetensors`. Compare loaded-weights hash to pre-refactor.
- `make hydralora && make test-hydra`, `make postfix && make test-postfix`.

**Risks.**
- `save_weights` has subtle ordering: OrthoHydra must run before OrthoExp (`lora_anima.py:1638`). Preserve as a fixed pipeline — `[ortho_hydra_to_hydra, ortho_exp_to_lora, ortho_to_lora]` chain, then variant-agnostic qkv defuse. Document in `lora_save.py` header.
- `OrthoLoRAExpModule._cayley` and `OrthoHydraLoRAExpModule._cayley` are referenced from inside save code — keep imports; don't pure-functionalize Cayley here.

---

## M3 — `--print-config`, auto-snapshot, schema validation (priority 3)

**Goal.** Kill silent typos in `configs/methods/*.toml`; let the user preview merged config before a 2-hour run.

**Target shape.**

- `library/config_schema.py` (new). Module-level `CONFIG_SCHEMA: dict[str, ConfigKey]` where `ConfigKey` is `@dataclass(frozen=True)` with `type, default, choices, doc, since_version, aliases`. Seeded by walking `setup_parser()._actions` at import — a one-shot build step. Known extras not in argparse (`network_args` list shape, `base_config` sentinel) registered manually.
- `library/train_util.py:1492 (_flatten_toml)` → `_flatten_toml(d, path="", schema=CONFIG_SCHEMA) -> dict` that:
  - warns with `file:line` when a key isn't in schema or aliases,
  - warns when a choice-valued key gets an off-list value,
  - coerces types (TOML allows `1` where `float` is wanted).
  - `strict` flag (default warn; `--config-strict` turns warnings into errors).
- New CLI flags handled in `setup_parser`:
  - `--print-config` — dumps fully merged namespace as TOML to stdout and `sys.exit(0)`. Includes provenance comments (`# from base.toml / from presets.toml[default] / from methods/lora.toml`).
  - `--config-snapshot` (default `true`). On every real run, writes `output/<output_name>/config.snapshot.toml` with provenance + `ss_git_sha` header. Disable via `--no-config-snapshot`.
- `make print-config METHOD=lora PRESET=default`.

**Files touched.** New `library/config_schema.py`, modifications to `library/train_util.py` (`_flatten_toml`, `_load_toml_with_base`, `load_method_preset`, `read_config_from_file`), `train.py` `setup_parser`, `Makefile`.

**Verify.**
- `tests/test_config.py`: typo detection (`network_ditm = 64` warns), alias mapping, `--print-config` output re-parses as valid TOML whose keys are a subset of schema.
- `make print-config METHOD=postfix_func PRESET=default` dumps a clean merged config — manual eyeball check.
- All 9 method configs × 5 presets round-trip without warnings (baseline — if a real typo surfaces, fix the config, don't widen the schema).

**Risks.**
- kohya inheritance: `train_util.py` has a lot of legacy keys. Seed schema from `setup_parser()._actions` to stay in sync automatically.
- `network_args` is a list of `key=value` strings whose valid keys depend on `network_module`. Validate in a second pass by looking up the resolved `NetworkSpec.kwarg_flags` from M2. Soft dep: M3 is nicer with M2 done, but can land standalone with a pending-keys allowlist.

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

**Recommended order.** M3 → M2 → M6 → M5 → M4 → M7.

- **M3 before M2** — schema validation catches typos that M2's resolver would silently absorb as "unknown flag → use LoRA".
- **M5 before M4** — sweep's per-run validation wants a cheap `InferenceEngine`.
- **M6, M7 parallelizable** any time.

**Hard deps.** None. M4 prefers M3 for schema validation but can run without it. M5 prefers M2's registry but can short-term keep filename sniffing.

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
