"""Config discovery, load/save, merge, and lint for the GUI.

Qt-free: reads/writes TOML, resolves the base → preset → variant merge chain
that drives the Config tab, and lints the dataset blueprint.
"""

from __future__ import annotations

import re
from pathlib import Path

import toml

from gui.core.paths import (
    CONFIGS_DIR,
    CUSTOM_DIR,
    CUSTOM_VARIANTS_DIR,
    GUI_METHODS_DIR,
    PRESETS_FILE,
    ROOT,
    _METHOD_ORDER,
)
from gui.core.validation import (
    _base_folder_repeats,
    _base_validation_enabled,
    _base_validation_split_num,
    _variant_folder_repeats_override,
    _variant_validation_override,
    _variant_validation_split_num,
)

# Built-in variants are discovered from each gui-methods/*.toml ``[variant]`` table
# (``family`` / ``label`` / optional ``order``). Family order in the combo is
# ``_METHOD_ORDER`` (a family left out is hidden). Customs under
# ``configs/gui-methods/custom/`` need no ``[variant]`` block and show under every family.


def _read_variant_metadata(path) -> dict:
    """The ``[variant]`` table of a gui-methods TOML, or ``{}`` on any failure."""
    if not path.is_file():
        return {}
    try:
        data = toml.loads(path.read_text(encoding="utf-8"))
    except (toml.TomlDecodeError, OSError):
        return {}
    meta = data.get("variant")
    return meta if isinstance(meta, dict) else {}


def _builtin_variants_by_family() -> dict[str, list[tuple[int, str, str]]]:
    """family → sorted (order, stem, label) for files directly under
    ``configs/gui-methods/``; files without ``[variant].family`` are skipped."""
    by_family: dict[str, list[tuple[int, str, str]]] = {}
    if not GUI_METHODS_DIR.is_dir():
        return by_family
    for path in GUI_METHODS_DIR.glob("*.toml"):
        meta = _read_variant_metadata(path)
        family = meta.get("family")
        if not isinstance(family, str) or not family:
            continue
        order = meta.get("order")
        order_int = order if isinstance(order, int) else 100
        label = meta.get("label") if isinstance(meta.get("label"), str) else path.stem
        by_family.setdefault(family, []).append((order_int, path.stem, label))
    for entries in by_family.values():
        entries.sort(key=lambda e: (e[0], e[1]))
    return by_family


def variant_metadata(variant: str) -> dict:
    """The ``[variant]`` metadata of a built-in or ``custom/<name>`` variant
    (``{}`` when absent, as customs may omit it)."""
    return _read_variant_metadata(variant_path(variant))


def list_methods() -> list[str]:
    """Families in ``_METHOD_ORDER`` that have built-in variant files on disk
    (so no family shows an empty variant combo)."""
    available = set(_builtin_variants_by_family())
    return [m for m in _METHOD_ORDER if m in available]


def list_gui_variants(method: str) -> list[str]:
    """The family's built-in variants (by ``[variant].order``, then stem),
    then every ``custom/*.toml``."""
    by_family = _builtin_variants_by_family()
    ordered = [stem for _, stem, _ in by_family.get(method, [])]
    if CUSTOM_VARIANTS_DIR.exists():
        for p in sorted(CUSTOM_VARIANTS_DIR.glob("*.toml")):
            ordered.append(f"custom/{p.stem}")
    return ordered


def is_custom_variant(name: str) -> bool:
    return name.startswith("custom/")


def custom_variant_path(name: str):
    """Resolve 'custom/<name>' (or bare '<name>') to the on-disk file path."""
    stem = name[len("custom/") :] if name.startswith("custom/") else name
    return CUSTOM_VARIANTS_DIR / f"{stem}.toml"


def variant_path(variant: str):
    """Resolve a variant identifier (built-in or 'custom/<name>') to its file."""
    return GUI_METHODS_DIR / f"{variant}.toml"


def _load_all_presets() -> dict:
    """Built-in sections in ``configs/presets.toml`` plus user-created flat
    files under ``configs/custom/<name>.toml`` (one preset per file)."""
    presets: dict = {}
    if PRESETS_FILE.exists():
        data = toml.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        presets.update({k: v for k, v in data.items() if isinstance(v, dict)})
    if CUSTOM_DIR.exists():
        for p in sorted(CUSTOM_DIR.glob("*.toml")):
            try:
                presets[p.stem] = toml.loads(p.read_text(encoding="utf-8"))
            except (toml.TomlDecodeError, OSError):
                continue
    return presets


def list_presets() -> list[str]:
    """All preset names, from the library so the GUI and trainer agree."""
    from library.config.io import list_presets as _lib_list_presets

    return _lib_list_presets()


def list_hardware_presets() -> list[tuple[str, dict]]:
    """``(name, meta)`` for the presets the Hardware dropdown offers.

    A presets.toml section opts in with a ``[<name>.gui]`` sub-table carrying
    ``group = "hardware"`` (optional ``label`` / ``description`` / ``order``);
    the trainer's merge strips it (``_METADATA_CONFIG_SECTIONS`` in
    ``library/config/io.py``). Sorted by ``order`` then name; never empty."""
    out: list[tuple[str, dict]] = []
    if PRESETS_FILE.exists():
        try:
            data = toml.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        except (toml.TomlDecodeError, OSError):
            data = {}
        for name, section in data.items():
            if not isinstance(section, dict):
                continue
            meta = section.get("gui")
            if isinstance(meta, dict) and meta.get("group") == "hardware":
                out.append((name, meta))
    out.sort(key=lambda item: (item[1].get("order", 100), item[0]))
    return out or [("default", {})]


def is_custom_preset(name: str) -> bool:
    return (CUSTOM_DIR / f"{name}.toml").exists()


def custom_preset_path(name: str):
    return CUSTOM_DIR / f"{name}.toml"


_GROUPS = {
    "Architecture": {
        "network_dim",
        "network_alpha",
        "network_module",
        "network_args",
        "use_timestep_mask",
        "use_moe_style",
        "route_per_layer",
        "router_source",
        "min_rank",
        "alpha_rank_scale",
        "num_experts",
        "balance_loss_weight",
        "balance_loss_warmup_ratio",
        "sigma_feature_dim",
        "router_targets",
        "per_bucket_balance_weight",
        "num_sigma_buckets",
        "specialize_experts_by_sigma_buckets",
        "sigma_bucket_boundaries",
        "use_repa",
        "repa_target_dog",
        # σ-demoted training; every key below is inert while sigma_lowres is off.
        "sigma_lowres",
        "sigma_lowres_route",
        "sigma_lowres_threshold",
        "sigma_lowres_threshold_max",
        "sigma_lowres_yarnsig",
        "sigma_lowres_span",
        "sigma_lowres_route2",
        "sigma_lowres_threshold2",
        "sigma_lowres_threshold2_max",
        "sigma_lowres_span2",
        "train_adaln",
        "adaln_rank",
        "adaln_alpha",
        "network_train_unet_only",
    },
    "Paths": {
        "pretrained_model_name_or_path",
        "qwen3",
        "vae",
        "vocab_pack",
        "path_scope",
        "output_dir",
        "output_name",
        "save_model_as",
        "source_image_dir",
        "resized_image_dir",
        "lora_cache_dir",
        "path_pattern",
    },
    "Training": {
        "learning_rate",
        "max_train_epochs",
        "save_every_n_epochs",
        "checkpointing_epochs",
        "gradient_accumulation_steps",
        "use_shuffled_caption_variants",
        "caption_dropout_rate",
        "optimizer_type",
        "lr_scheduler",
        "lr_warmup_steps",
        "max_grad_norm",
        "clip_mode",
        "clip_k",
        "clip_window",
        "clip_per_micro",
        "clip_sigma_bands",
        "ema_window",
        "ema_save_raw",
        "timestep_sampling",
        "discrete_flow_shift",
        "masked_loss",
        "use_valid",
        "validation_split_num",
        "repeat_by_folder_name",
        "sample_ratio",
        "artists_shard",
        "training_comment",
    },
    "Samples": {
        "sample_prompts",
        "sample_every_n_epochs",
        "sample_at_first",
        "sample_decode_inline",
    },
    "Performance": {
        "int8",
        "stream",
        "attn_mode",
        "gradient_checkpointing",
        "unsloth_offload_checkpointing",
        "activation_memory_budget",
        "blocks_to_swap",
        "torch_compile",
        "cache_llm_adapter_outputs",
        "mixed_precision",
        "vae_chunk_size",
        "vae_disable_cache",
        "use_vae_cache",
        "use_text_cache",
        "skip_cache_check",
        "layer_start",
        "use_cmmd",
    },
}
_K2G = {k: g for g, ks in _GROUPS.items() for k in ks}
# target_res is owned by the Preprocess tab. ``drop_lowres_images`` / ``min_pixels`` are retired
# but a user-owned preprocess.toml may still carry them; skipping them keeps them off the form.
_SKIP = {
    "base_config",
    "dataset_config",
    "general",
    "datasets",
    "variant",
    "target_res",
    "drop_lowres_images",
    "min_pixels",
}

# Virtual keys are form fields that live in the ``[[datasets]]`` override, not as flat TOML keys;
# Save skips them and the ``gui.core.validation`` helpers write them.
_VIRTUAL_KEYS = {"use_valid", "validation_split_num", "repeat_by_folder_name"}

# Fields shown under "Basic"; everything else goes in the collapsible "Advanced" section.
_BASIC = {
    "learning_rate",
    "max_train_epochs",
    "save_every_n_epochs",
    "checkpointing_epochs",
    "network_dim",
    "network_alpha",
    "network_weights",
    "num_experts",
    "use_shuffled_caption_variants",
    "caption_dropout_rate",
    "masked_loss",
    "sample_ratio",
    "artists_shard",
    "gradient_checkpointing",
    "blocks_to_swap",
    "int8",
    "path_scope",
    "output_name",
    "path_pattern",
    "use_valid",
    "validation_split_num",
    "sample_prompts",
    "sample_every_n_epochs",
    "sample_at_first",
    "sample_decode_inline",
}


def is_basic_field(key: str) -> bool:
    return key in _BASIC


def _load(p) -> dict:
    return toml.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _load_base() -> dict:
    """``base.toml`` overlaid on ``configs/preprocess.toml``.

    Same layering as ``load_path_overrides``, so the form baseline matches what
    preprocess/training read: a key still in base.toml wins. Use this anywhere
    the GUI needs base.toml as a flat-key baseline."""
    merged = _load(CONFIGS_DIR / "preprocess.toml")
    merged.update(_load(CONFIGS_DIR / "base.toml"))
    return merged


def _abs_under_root(rel) -> Path:
    """Resolve a (possibly relative) config path string against the repo root."""
    p = Path(str(rel))
    return p if p.is_absolute() else ROOT / p


def default_lora_cache_dir() -> Path:
    """Absolute VAE/TE/PE cache dir: ``lora_cache_dir`` from base.toml, as the
    trainer reads it."""
    return _abs_under_root(
        _load_base().get("lora_cache_dir") or "post_image_dataset/lora"
    )


def default_resized_dir() -> Path:
    """Absolute resized-image dir — ``resized_image_dir`` from base.toml."""
    return _abs_under_root(
        _load_base().get("resized_image_dir") or "post_image_dataset/resized"
    )


def dataset_cache_root() -> Path:
    """The ``post_image_dataset/`` root that masks + per-method caches hang off
    (parent of the lora cache dir)."""
    return default_lora_cache_dir().parent


def default_mask_dir() -> Path:
    """Absolute ``mask_dir`` (the key ``make mask`` and training read), else
    ``masks/`` under the cache root."""
    configured = _load_base().get("mask_dir")
    if configured:
        return _abs_under_root(configured)
    return dataset_cache_root() / "masks"


def _save(p, d: dict):
    p.write_text(toml.dumps(d), encoding="utf-8")


def _dataset_lint_sources(variant: str):
    """The (path, label) pairs the dataset-blueprint linter scans: shared
    ``base.toml`` plus the active variant file. ``label`` is what shows up in
    the banner and must match the ``source=`` passed to ``lint_dataset_sections``."""
    return (
        (CONFIGS_DIR / "base.toml", "base.toml"),
        (variant_path(variant), f"gui-methods/{variant}.toml"),
    )


def lint_variant_configs(variant: str) -> list:
    """``DatasetKeyIssue``s for keys in the dataset-blueprint sections of
    ``base.toml`` and the variant file that the trainer's validator will reject
    (e.g. a stale ``resolution`` in ``[[datasets]]``). Torch-free."""
    from library.config.dataset_keys import lint_dataset_sections

    issues: list = []
    for path, label in _dataset_lint_sources(variant):
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
            raw = toml.loads(text)
        except (OSError, toml.TomlDecodeError):
            continue
        issues.extend(lint_dataset_sections(raw, source=label, text=text))
    return issues


def remove_unknown_dataset_keys(variant: str) -> list[str]:
    """Delete the lines :func:`lint_variant_configs` flags; returns
    ``"key (label)"`` for each removal.

    Edits raw text because ``_save`` (``toml.dumps``) drops comments and
    base.toml is heavily commented. Deletes bottom-up so line numbers stay
    valid, and re-checks each line starts with ``<key> =``.
    """
    from library.config.dataset_keys import lint_dataset_sections

    removed: list[str] = []
    for path, label in _dataset_lint_sources(variant):
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
            raw = toml.loads(text)
        except (OSError, toml.TomlDecodeError):
            continue
        issues = lint_dataset_sections(raw, source=label, text=text)
        targets = sorted(
            (i for i in issues if i.line is not None),
            key=lambda i: i.line,
            reverse=True,
        )
        if not targets:
            continue
        lines = text.splitlines(keepends=True)
        changed = False
        for issue in targets:
            idx = issue.line - 1
            if 0 <= idx < len(lines) and re.match(
                rf"^\s*{re.escape(issue.key)}\s*=", lines[idx]
            ):
                del lines[idx]
                removed.append(f"{issue.key} ({label})")
                changed = True
        if changed:
            path.write_text("".join(lines), encoding="utf-8")
    return removed


def _origin_from_tag(tag: str) -> str:
    """Map a ``library.config.io`` provenance tag to the GUI's coarse
    base/preset/method bucket (the keys ``ConfigTab`` styles fields by)."""
    if "presets.toml" in tag or "/custom/" in tag:
        return "preset"
    if tag.endswith("base.toml") or tag.endswith("preprocess.toml"):
        return "base"
    return "method"


def _merged_via_library(
    method: str, preset: str, methods_subdir: str
) -> tuple[dict, dict[str, str]]:
    """``(merged, origin)`` from the trainer's own base→preset→method merge
    (:func:`library.config.io.load_method_preset`).

    Seeded with ``_load_base()`` because the preprocess-only scalars (e.g.
    ``source_image_dir``) live in preprocess.toml, which the training merge
    does not fold in.
    """
    from library.config.io import load_method_preset

    merged: dict = dict(_load_base())
    origin: dict[str, str] = {k: "base" for k in merged}
    try:
        lib_merged, provenance = load_method_preset(
            method, preset, methods_subdir=methods_subdir, return_provenance=True
        )
    except (FileNotFoundError, KeyError, ValueError):
        # Missing method/preset file: fall back to the base baseline.
        return merged, origin
    for k, v in lib_merged.items():
        merged[k] = v
        origin[k] = _origin_from_tag(provenance.get(k, ""))
    return merged, origin


def merged_method_preset(method: str, preset: str) -> tuple[dict, dict[str, str]]:
    """Return (merged_dict, origin_map). origin_map[key] is 'base' | 'preset' | 'method'."""
    return _merged_via_library(method, preset, methods_subdir="methods")


def merged_gui_variant_preset(variant: str, preset: str) -> tuple[dict, dict[str, str]]:
    """Merge base + preset + gui-methods/<variant>.toml, plus the GUI's
    virtual keys and ``path_scope``."""
    base = _load_base()
    meth = _load(GUI_METHODS_DIR / f"{variant}.toml")
    merged, origin = _merged_via_library(variant, preset, methods_subdir="gui-methods")

    # path_scope lives under [variant] so CLI loading strips it as metadata; it is
    # expanded into concrete paths at submit.
    meta = meth.get("variant")
    if isinstance(meta, dict) and isinstance(meta.get("path_scope"), str):
        merged["path_scope"] = meta["path_scope"]
        origin["path_scope"] = "method"
    elif "path_scope" not in merged:
        merged["path_scope"] = ""
        origin["path_scope"] = "base"

    variant_override = _variant_validation_override(meth)
    if variant_override is not None:
        merged["use_valid"] = variant_override
        origin["use_valid"] = "method"
    else:
        merged["use_valid"] = _base_validation_enabled(base)
        origin["use_valid"] = "base"

    variant_vsn = _variant_validation_split_num(meth)
    if variant_vsn is not None:
        merged["validation_split_num"] = variant_vsn
        origin["validation_split_num"] = "method"
    else:
        merged["validation_split_num"] = _base_validation_split_num(base)
        origin["validation_split_num"] = "base"

    # repeat_by_folder_name: Kohya-style {n}_folder repeats.
    variant_rbf = _variant_folder_repeats_override(meth)
    if variant_rbf is not None:
        merged["repeat_by_folder_name"] = variant_rbf
        origin["repeat_by_folder_name"] = "method"
    else:
        merged["repeat_by_folder_name"] = _base_folder_repeats(base)
        origin["repeat_by_folder_name"] = "base"

    # Show the sample-image knobs even when no TOML in the chain sets them. Cadence 0 = disabled
    # (train.py maps non-positive to None).
    for _k, _default in (
        ("sample_prompts", []),
        ("sample_every_n_epochs", 0),
        ("sample_at_first", False),
        ("sample_decode_inline", "false"),
    ):
        if _k not in merged:
            merged[_k] = _default
            origin[_k] = "base"
    return merged, origin
