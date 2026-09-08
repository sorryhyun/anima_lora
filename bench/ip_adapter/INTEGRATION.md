# IP-Adapter — re-integration map

IP-Adapter was **removed from the live codebase on 2026-06-10** and downgraded to
this bench probe (see `plan.md`). It was unproven on Anima (never benched against
a baseline, data-appetite mismatched to the personalization use case) and had
accreted cross-cutting surface (configs, GUI, i18n, distinct-pair orchestration)
it hadn't earned. This file is the obituary-and-restore map: if the bench in
`plan.md` returns `IP-ADAPTER WINS A NICHE`, re-applying the points below restores
the shipped wiring. If it returns `IP-ADAPTER IS REDUNDANT`, this directory is the
only record kept.

The removal commit's diff is the authoritative source; this file is the index so
re-integration doesn't require commit archaeology.

## What was deliberately LEFT in the live tree (do NOT restore these — already present)

- **PE feature cache.** `scripts/preprocess/cache_pe_encoder.py`, `make preprocess-pe`,
  `library/preprocess/pe.py`, and the `{stem}_anima_pe.safetensors` /
  `anima_pe_centroid_{encoder}.safetensors` sidecars. Shared with **CMMD validation**
  (`library/training/cmmd.py`) — not IP-Adapter-private.
- **`networks/methods/ip_adapter_pe_lora.py`** (`inject_pe_lora` / `PELoRALayer`).
  Vendored into the live **Anima-Tagger** ComfyUI node (`scripts/release/sync_vendor.py`
  `TAGGER_VERBATIM`, `custom_nodes/comfyui-anima-tagger/nodes.py` `pe_lora` UI). The
  live trainer no longer imports it once IP-Adapter is gone, but the published node
  does. Left in place.
- **`networks/methods/base.py`** (`MethodAdapter` lifecycle base) — shared by
  easycontrol / soft_tokens.
- **`bench/ip_adapter/pair_audit.py` + `pair_audit.md`** — the Phase-0 dataset audit.

## Preserved source (`impl/`)

| File | Was at | Notes |
|---|---|---|
| `impl/ip_adapter.py` | `networks/methods/ip_adapter.py` | The module — `forward` / `set_ip_tokens` / `IPAdapterMethodAdapter` / `create_network_from_weights`. Restore verbatim. Imports `inject_pe_lora` from the still-live `ip_adapter_pe_lora.py`. |
| `impl/methods_ip_adapter.toml` | `configs/methods/ip_adapter.toml` | Shipped training config (rank/encoder/resampler/gate_lr/distinct-pair/PE-LoRA knobs). |
| `impl/gui-methods_ip_adapter.toml` | `configs/gui-methods/ip_adapter.toml` | Clean GUI variant (`[variant]` family="ip_adapter" order=10). |
| `impl/datasets_ip_adapter.toml` | `configs/datasets/ip_adapter.toml` | Self/distinct-pair dataset blueprint. |
| `impl/docs_experimental_ip-adapter.md` | `docs/experimental/ip-adapter.md` | Method deep-dive. |

## Live-code hook points removed (re-apply in this order)

The module is an additive side-channel (frozen DiT + Perceiver resampler + per-block
`to_k_ip`/`to_v_ip` + init-0 `ip_gate`). Every hook is a small gated block, not a
rewrite. Restoring is mechanical.

### 1. Module + dispatch registration
- `networks/methods/ip_adapter.py` — restore from `impl/ip_adapter.py`.
- `networks/methods/__init__.py` — re-add the `ip_adapter` bullet to the module docstring.
- `library/training/method_adapter.py` `resolve_adapters` — re-add the
  `if getattr(args, "use_ip_adapter", False): from networks.methods.ip_adapter import
  IPAdapterMethodAdapter; adapters.append(IPAdapterMethodAdapter())` block.

### 2. Training args (`library/anima/training.py`)
- Re-add the `--use_ip_adapter`, `--ip_features_cache_to_disk`, `--ip_image_drop_p`,
  `--ip_encoder`, `--ip_pair_mode`, `--ip_pair_prob`, `--ip_pair_min_level`,
  `--ip_pair_caption_strip_p` argparse block (was the contiguous `--ip_*` group).
  All the `getattr(args, "ip_*", …)` reads in train.py / datasets depend on these.

### 3. Training loop (`train.py`)
- Re-add the IP-Adapter dataset-wiring block (was ~L417–490): the
  `ip_features_cache_to_disk` propagation onto each dataset, the live-PE-fallback
  `force_load_images_for_ip` branch (`use_ip_adapter` and not cache-to-disk), and the
  distinct-pair setup (`ip_pair_mode != "self"` → load `ip_pair_index` from the caption
  index, call `dataset.configure_ip_pairs(...)`, with the `pe_lora_enabled` /
  cache-to-disk compatibility guards).

### 4. Dataset (`library/datasets/base.py`) — the bulk
- `__init__` fields: `ip_features_cache_to_disk`, `ip_features_encoder`,
  `force_load_images_for_ip`, `ip_pair_*` (prob / caption_strip_p / is_validation /
  `_ip_pair_strip_warned`).
- `_try_load_ip_features(image_abs_path)` — per-image PE sidecar loader (self-pair).
- `_load_ip_features_for_stem(...)` + `configure_ip_pairs(...)` — distinct-pair stem-swap
  loader and its setup method.
- `__getitem__` assembly: the `ip_features_list` / `ip_features_shuffled_list` build
  (self vs distinct-pair vs shuffled-baseline branches, `force_load_images_for_ip`
  image-load forcing, caption-strip handling) and the
  `example["ip_features"] = torch.stack(...)` / shuffled tail.

### 5. Inference (`library/inference/`)
- `generation.py`: restore `_setup_ip_adapter(args, anima, device)` and its call site in
  the generate path; it lazily imports `create_network_from_weights`, encodes the ref
  image to PE features (or via `pe_lora_enabled` live encode), `network.set_ip_tokens(...)`,
  stashes `anima._ip_adapter_network`.
- `request.py`: re-add the `ip_adapter_weight: Optional[str]` field + the `to_args`
  `--ip_adapter_weight` emission.
- `args.py`: re-add `--ip_adapter_weight` and `--ip_scale` argparse entries.

### 6. Task entry points
- `scripts/experimental_tasks/training.py`: `cmd_ip_adapter` + `cmd_ip_adapter_preprocess`.
- `scripts/experimental_tasks/inference.py`: `cmd_test_ip` (+ the shared ref-image resolver).
- `tasks.py`: the `exp-ip-adapter`, `exp-ip-adapter-preprocess`, `exp-test-ip` registry
  entries + the `exp-test-ip` line in the usage banner.

### 7. Configs
- `configs/methods/ip_adapter.toml`, `configs/gui-methods/ip_adapter.toml`,
  `configs/datasets/ip_adapter.toml` — restore from `impl/`.

### 8. GUI surface (`gui/`)
- `gui/app.py`: `ip_adapter` in the methods list.
- `gui/tabs/adapter_tab.py`, `gui/tabs/config_tab.py`, `gui/_paths.py`: the IP-Adapter
  form fields / path handling.
- `gui/i18n/{en,ja,ko,cn}.py`: the IP-Adapter UI strings (4 languages).

### 9. Tests
- `tests/test_method_network_lifecycle.py`: the `IPAdapterNetwork` import + lifecycle case.
- `tests/test_smoke.py`: the IP-Adapter smoke reference.

### 10. Docs
- Restore `docs/experimental/ip-adapter.md` from `impl/`.
- Re-thread the mentions in `CLAUDE.md` (methods table row + §Methods), `networks/CLAUDE.md`
  (layout table), `docs/guidelines/{inference,training}.md`, `docs/multi_model_support.md`,
  `../anime_tools/docs/anima_tagger.md` cross-ref.

## Key facts the bench must honor (from removal-time state)

- **Frozen DiT.** IP-Adapter trains only the resampler + per-block `to_k_ip`/`to_v_ip` +
  `ip_gate` (and optionally PE-LoRA). It forces `blocks_to_swap=2` for PE-forward headroom.
- **Step-0 ≡ baseline DiT** is guaranteed by the per-block `ip_gate` scalar (init 0), **not**
  by zero-init projections. Any restore must preserve the gate-init-0 invariant.
- **Decoupled cross-attention.** `set_ip_tokens` injects image K/V into a second attention
  path; `ip_scale` (`ss_ip_scale`, default 1.0) trades reference strength vs prompt adherence.
- **Default reference is pre-cached PE** (`ip_features_cache_to_disk=true`, `ip_encoder="pe"`).
  Live-encode is the fallback (`force_load_images_for_ip`).
- **Distinct-pair / identity training** needs the caption index (`make caption-index`) and is
  the data-hungry path the removal was premised on — see `pair_audit.py`.
- **Not foldable.** Frozen-DiT method; `merge_to_dit.py` never supported it. ComfyUI needs a
  custom node to run it.
- **PE cache + `ip_adapter_pe_lora.py` were NOT removed** (see top of this file) — do not
  re-create them on restore.
