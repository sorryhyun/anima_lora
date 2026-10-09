"""ConfigTab's form ↔ variant-file logic, Qt-free.

ConfigTab lays its widgets out from :func:`group_fields` and writes a Save
back through :func:`variant_from_form`. Widget access is a ``read(key,
baseline)`` callback (the baseline types the value, as ``_read`` does), so
tests drive the writeback with plain dicts.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

import toml

from gui.core.config_io import _GROUPS, _K2G, _VIRTUAL_KEYS, is_basic_field
from gui.core.submit import PATH_SCOPE_KEY
from gui.core.validation import (
    _base_folder_repeats,
    apply_folder_repeats_choice,
    apply_validation_choice,
)

# Position of a field inside its group box; unlisted keys sort at 100, then
# alphabetically.
FIELD_ORDER = {
    PATH_SCOPE_KEY: 10,
    "source_image_dir": 11,
    "resized_image_dir": 12,
    "lora_cache_dir": 13,
    "output_dir": 14,
    "output_name": 15,
    "save_model_as": 16,
    "path_pattern": 20,
    "pretrained_model_name_or_path": 30,
    "qwen3": 31,
    "vae": 32,
    # Pins must stay BELOW the unpinned default (100) or alphabetical sort
    # interleaves the block with the rest of its group box.
    "use_repa": 80,
    "repa_target_dog": 81,
    "train_adaln": 82,
    "adaln_rank": 83,
    "adaln_alpha": 84,
    "sigma_lowres": 85,
    "sigma_lowres_route": 86,
    "sigma_lowres_threshold": 87,
    "sigma_lowres_threshold_max": 88,
    "sigma_lowres_yarnsig": 89,
    "sigma_lowres_span": 90,
    "sigma_lowres_route2": 91,
    "sigma_lowres_threshold2": 92,
    "sigma_lowres_threshold2_max": 93,
    "sigma_lowres_span2": 94,
    "sample_prompts": 10,
    "sample_every_n_epochs": 11,
    "sample_at_first": 12,
    "sample_decode_inline": 13,
}


def field_sort_key(key: str) -> tuple[int, str]:
    return (FIELD_ORDER.get(key, 100), key)


def group_fields(cfg: dict[str, Any]) -> tuple[dict[str, dict], dict[str, dict]]:
    """Split ``cfg`` into ``(basic, advanced)``, each ``{group: {key: value}}``
    in ``_GROUPS`` order with a trailing ``"Other"``; empty groups stay."""
    basic: dict[str, dict] = {g: {} for g in _GROUPS}
    basic["Other"] = {}
    advanced: dict[str, dict] = {g: {} for g in _GROUPS}
    advanced["Other"] = {}
    for k, v in cfg.items():
        (basic if is_basic_field(k) else advanced)[_K2G.get(k, "Other")][k] = v
    return basic, advanced


class ExtraArgsError(ValueError):
    """The extra-args box does not parse as TOML."""


def parse_extra_args(text: str) -> dict[str, Any]:
    """The extra-args box as top-level TOML scalars (tables are dropped).

    Bare backslashes (a pasted Windows path) break TOML escapes, so a failed
    parse is retried once with ``\\`` → ``/``; the first error is raised."""
    text = text.strip()
    if not text:
        return {}
    try:
        parsed = toml.loads(text)
    except toml.TomlDecodeError as e:
        if "\\" not in text:
            raise ExtraArgsError(str(e)) from e
        try:
            parsed = toml.loads(text.replace("\\", "/"))
        except toml.TomlDecodeError:
            raise ExtraArgsError(str(e)) from e
    return {k: v for k, v in parsed.items() if not isinstance(v, dict)}


def _set_path_scope(out: dict, scope: str) -> None:
    """``path_scope`` lives in the ``[variant]`` table, never as a flat key."""
    meta = out.get("variant")
    if not isinstance(meta, dict):
        meta = {}
    if scope:
        meta[PATH_SCOPE_KEY] = scope
        out["variant"] = meta
    else:
        meta.pop(PATH_SCOPE_KEY, None)
        if meta:
            out["variant"] = meta
        else:
            out.pop("variant", None)
    out.pop(PATH_SCOPE_KEY, None)


def _base_validation_split_num(base: dict) -> int | None:
    datasets = base.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return None
    first = datasets[0]
    if not isinstance(first, dict) or first.get("validation_split_num") is None:
        return None
    try:
        return int(first["validation_split_num"])
    except (TypeError, ValueError):
        return None


def variant_from_form(
    method_orig: dict[str, Any],
    keys: Collection[str],
    read: Callable[[str, Any], Any],
    *,
    base: dict[str, Any],
    preset_overlay: dict[str, Any],
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The variant file a Save writes.

    Starts from the file as it is (``method_orig``). A form key is written when
    the file already has it or its value differs from what the chain would
    give anyway (preset, then base) — so a value the hardware preset provides
    is never baked in, which would pin it against later preset switches
    (method beats preset). The virtual keys go into the ``[[datasets]]``
    override, ``path_scope`` into ``[variant]``; ``extras`` win last."""
    out: dict[str, Any] = dict(method_orig)
    for k in keys:
        if k in _VIRTUAL_KEYS:
            continue
        if k == PATH_SCOPE_KEY:
            _set_path_scope(out, str(read(k, "") or "").strip())
            continue
        baseline = method_orig.get(k, preset_overlay.get(k, base.get(k)))
        v = read(k, baseline)
        if k in method_orig or v != baseline:
            out[k] = v

    if "use_valid" in keys:
        split_num: int | None = None
        if "validation_split_num" in keys:
            try:
                split_num = int(read("validation_split_num", None))
            except (TypeError, ValueError):
                split_num = None
        apply_validation_choice(
            out,
            bool(read("use_valid", None)),
            split_num=split_num,
            base_split_num=_base_validation_split_num(base),
        )
    if "repeat_by_folder_name" in keys:
        apply_folder_repeats_choice(
            out,
            bool(read("repeat_by_folder_name", None)),
            base_enabled=_base_folder_repeats(base),
        )
    if extras:
        out.update(extras)
    return out
