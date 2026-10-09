"""Submit plan — what the GUI hands the daemon for a train / preprocess job.

Pure dict-in / dict-out over the merged config chain: the GUI-only
``path_scope`` layering, the training and preprocess config snapshots, the
preprocess env, and the ``chain_train`` spec the daemon enqueues after a
successful auto-chain preprocess. ConfigTab / EasyControlTab / PreprocessingTab
call these with the widget state they own.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from gui.core.config_io import _VIRTUAL_KEYS, default_lora_cache_dir
from gui.core.paths import ROOT

PATH_SCOPE_KEY = "path_scope"
METHODS_SUBDIR = "gui-methods"

# Run paths a ``path_scope`` is appended onto, with the default each falls back
# to when the merged chain leaves it unset.
_SCOPED_PATH_DEFAULTS = {
    "source_image_dir": "image_dataset",
    "resized_image_dir": "post_image_dataset/resized",
    "lora_cache_dir": "post_image_dataset/lora",
    "output_dir": "output/ckpt",
}

# Merge-chain bookkeeping and GUI-only keys no submitted snapshot carries.
_SNAPSHOT_META_KEYS = (
    "base_config",
    "dataset_config",
    "variant",
    "method",
    "preset",
    "methods_subdir",
    PATH_SCOPE_KEY,
    "preprocess_path_pattern",
)


def normalize_path_scope(scope: Any) -> str | None:
    """A safe relative GUI path scope like ``data_group1``, else ``None``."""
    if not isinstance(scope, str):
        return None
    value = scope.strip().replace("\\", "/").strip("/")
    if not value:
        return None
    if value.endswith("/*"):
        value = value[:-2].strip("/")
    if not value or "|" in value or any(ch in value for ch in "*?[]:"):
        return None
    parts = value.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        return None
    return "/".join(parts)


def append_scope(path_value: Any, scope: str) -> str:
    base = str(path_value).strip() if path_value is not None else ""
    if not base:
        return scope
    norm = base.replace("\\", "/").rstrip("/")
    if norm == scope or norm.endswith("/" + scope):
        return base
    return f"{norm}/{scope}"


def scoped_paths(merged: dict[str, Any]) -> dict[str, Any]:
    """Apply the GUI-only ``path_scope`` to the concrete run paths.
    ``path_pattern`` keeps its training-filter meaning, evaluated relative to
    the scoped image/cache directories. Returns ``merged`` itself when unscoped."""
    scope = normalize_path_scope(merged.get(PATH_SCOPE_KEY))
    if not scope:
        return merged
    out = copy.deepcopy(merged)
    for key, default in _SCOPED_PATH_DEFAULTS.items():
        out[key] = append_scope(out.get(key) or default, scope)
    out.pop(PATH_SCOPE_KEY, None)
    out.pop("variant", None)
    return out


def clean_snapshot(value: Any) -> Any:
    """Drop ``None`` values recursively and stringify paths (TOML/JSON-safe)."""
    if isinstance(value, dict):
        return {k: clean_snapshot(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [clean_snapshot(v) for v in value if v is not None]
    if isinstance(value, Path):
        return str(value)
    return value


def training_snapshot(
    variant: str, merged: dict[str, Any], preprocess_overrides: dict[str, Any]
) -> dict[str, Any]:
    """Full training config captured at submit time, with the dataset blueprint
    resolved into ``general`` / ``datasets``.

    Preprocess-only knobs leak into ``merged`` / ``preprocess_overrides`` but
    must not ride into the training config: ``caption_tag_dropout_rate``
    collides with a real train arg meaning *live* dataloader tag dropout, and tag
    dropout is already baked into the cached caption variants — running it live
    too trips the TE-cache assertion."""
    from gui.tabs.preprocess.knobs import PREPROCESS_ONLY_KEYS
    from library.config.io import load_dataset_config_from_base

    snapshot = scoped_paths(copy.deepcopy(merged))
    snapshot.update(preprocess_overrides)
    for key in (
        *_SNAPSHOT_META_KEYS,
        "caption_tag_randomize_rate",
        *PREPROCESS_ONLY_KEYS,
        *_VIRTUAL_KEYS,
    ):
        snapshot.pop(key, None)

    dataset_cfg = load_dataset_config_from_base(
        overrides=snapshot, method=variant, methods_subdir=METHODS_SUBDIR
    )
    if dataset_cfg:
        snapshot["general"] = dataset_cfg.get("general", {})
        snapshot["datasets"] = dataset_cfg.get("datasets", [])
    return clean_snapshot(snapshot)


def preprocess_snapshot(
    merged: dict[str, Any], preprocess_overrides: dict[str, Any]
) -> dict[str, Any]:
    """Preprocess command config: the scoped merged chain plus the Preprocess
    tab's overrides, minus merge bookkeeping. ``preprocess_path_pattern`` is
    forwarded as env (``PREPROCESS_PATH_PATTERN``), never as a flat key."""
    snapshot = scoped_paths(copy.deepcopy(merged))
    snapshot.update(preprocess_overrides)
    for key in _SNAPSHOT_META_KEYS:
        snapshot.pop(key, None)
    return clean_snapshot(snapshot)


def preprocess_env(
    variant: str, preset: str, tab_env: dict[str, str] | None = None
) -> dict[str, str]:
    env = {"METHOD": variant, "METHODS_SUBDIR": METHODS_SUBDIR, "PRESET": preset}
    if tab_env:
        env.update(tab_env)
    return env


def chain_train_spec(
    variant: str, preset: str, *, config_snapshot: dict[str, Any] | None = None
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "method": variant,
        "preset": preset,
        "methods_subdir": METHODS_SUBDIR,
    }
    if config_snapshot is not None:
        spec["config_snapshot"] = config_snapshot
    return spec


def cache_dir(scoped: dict[str, Any]) -> Path:
    """Absolute ``lora_cache_dir`` of an already-scoped merged config."""
    cache_rel = scoped.get("lora_cache_dir")
    if not cache_rel:
        return default_lora_cache_dir()
    path = Path(cache_rel)
    return path if path.is_absolute() else ROOT / path


def repa_requirements(scoped: dict[str, Any]) -> tuple[bool, str | None]:
    """``(require_pe, pe_encoder)`` for the cache probe. PE sidecar names are
    encoder-specific (``{stem}_anima_{encoder}.…``), so the probe must look for
    the encoder REPA will actually read."""
    use_repa = scoped.get("use_repa")
    require_pe = use_repa is True or str(use_repa).strip().lower() in (
        "1",
        "true",
        "yes",
    )
    encoder = str(scoped.get("repa_encoder") or "pe_spatial").strip() or None
    return require_pe, encoder
