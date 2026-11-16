"""Dataset-blueprint override encoding for the GUI config form.

Reads the virtual keys (``use_valid`` / ``validation_split_num`` /
``repeat_by_folder_name``) out of a TOML ``[[datasets]]`` block and writes the
user's choice back into a variant dict. Pure dicts, no Qt.
"""

from __future__ import annotations

from typing import Any, Optional

# Held-out count written when validation is enabled but neither the form nor
# base.toml has a positive validation_split_num.
_DEFAULT_VALIDATION_SPLIT_NUM = 16


def _validation_enabled_from_datasets(datasets: Any) -> Optional[bool]:
    """Whether the first ``[[datasets]]`` entry enables validation; ``None``
    when it sets neither validation key (the parent layer decides)."""
    if not isinstance(datasets, list) or not datasets:
        return None
    first = datasets[0]
    if not isinstance(first, dict):
        return None
    vsn = first.get("validation_split_num")
    vs = first.get("validation_split")
    if vsn is None and vs is None:
        return None
    return (vsn or 0) > 0 or (vs or 0.0) > 0.0


def _variant_validation_override(variant_data: dict) -> Optional[bool]:
    """The variant's explicit use_valid override, or None."""
    return _validation_enabled_from_datasets(variant_data.get("datasets"))


def _base_validation_enabled(base_data: dict) -> bool:
    """Default use_valid from configs/base.toml's [[datasets]] block (False
    when missing, as the loader treats it)."""
    return bool(_validation_enabled_from_datasets(base_data.get("datasets")))


def _validation_split_num_from_datasets(datasets: Any) -> Optional[int]:
    """``validation_split_num`` of the first [[datasets]] entry as an int, or None."""
    if not isinstance(datasets, list) or not datasets:
        return None
    first = datasets[0]
    if not isinstance(first, dict):
        return None
    vsn = first.get("validation_split_num")
    if vsn is None:
        return None
    try:
        return int(vsn)
    except (TypeError, ValueError):
        return None


def _variant_validation_split_num(variant_data: dict) -> Optional[int]:
    """The variant's explicit validation_split_num override, or None."""
    return _validation_split_num_from_datasets(variant_data.get("datasets"))


def _base_validation_split_num(base_data: dict) -> int:
    """Default validation_split_num from configs/base.toml (0 when missing)."""
    return _validation_split_num_from_datasets(base_data.get("datasets")) or 0


def _folder_repeats_from_datasets(datasets: Any) -> Optional[bool]:
    """``repeat_by_folder_name`` of the first ``[[datasets]]`` entry (else its
    first subset, where hand-edited configs may carry it), or ``None``."""
    if not isinstance(datasets, list) or not datasets:
        return None
    first = datasets[0]
    if not isinstance(first, dict):
        return None
    val = first.get("repeat_by_folder_name")
    if val is None:
        subsets = first.get("subsets")
        if isinstance(subsets, list) and subsets and isinstance(subsets[0], dict):
            val = subsets[0].get("repeat_by_folder_name")
    return None if val is None else bool(val)


def _variant_folder_repeats_override(variant_data: dict) -> Optional[bool]:
    """The variant's explicit repeat_by_folder_name override, or None."""
    return _folder_repeats_from_datasets(variant_data.get("datasets"))


def _base_folder_repeats(base_data: dict) -> bool:
    """Default repeat_by_folder_name from configs/base.toml's ``[[datasets]]``
    block, else ``[general]``, else False (the BaseSubsetParams default)."""
    val = _folder_repeats_from_datasets(base_data.get("datasets"))
    if val is None:
        general = base_data.get("general")
        if (
            isinstance(general, dict)
            and general.get("repeat_by_folder_name") is not None
        ):
            val = bool(general["repeat_by_folder_name"])
    return bool(val)


def apply_folder_repeats_choice(out: dict, enabled: bool, base_enabled: bool) -> None:
    """Encode the repeat_by_folder_name checkbox into the variant dict ``out``.

    Matches base → strip the override. Differs → write it on the first
    ``[[datasets]]`` entry at dataset level: ``_apply_dataset_overrides``
    (library/config/io.py) merges only top-level dataset scalars, and the key
    ascends to every subset from there."""
    existing = out.get("datasets")
    if enabled == base_enabled:
        if not isinstance(existing, list) or not existing:
            return
        first = existing[0]
        if not isinstance(first, dict):
            return
        first.pop("repeat_by_folder_name", None)
        if not first and len(existing) == 1:
            del out["datasets"]
        return
    if not isinstance(existing, list):
        existing = []
        out["datasets"] = existing
    if not existing:
        existing.append({})
    first = existing[0]
    if not isinstance(first, dict):
        first = {}
        existing[0] = first
    first["repeat_by_folder_name"] = enabled


def apply_validation_choice(
    out: dict,
    enabled: bool,
    split_num: Optional[int] = None,
    base_split_num: Optional[int] = None,
) -> None:
    """Encode the use_valid checkbox (+ optional validation_split_num) into
    the variant dict ``out``; other ``[[datasets]]`` keys are kept.

    Enabled  → the count is the form's ``split_num``, else ``base_split_num``,
               else ``_DEFAULT_VALIDATION_SPLIT_NUM`` (first positive). Equal to
               an already-enabled base → strip both keys; otherwise write
               ``validation_split_num`` on the first [[datasets]] entry and drop
               a fractional ``validation_split``.
    Disabled → write ``validation_split_num = 0, validation_split = 0.0``,
               creating the block if absent (``split_num`` is ignored)."""
    existing = out.get("datasets")
    if enabled:
        base_vsn = base_split_num or 0
        # Fall back to a positive default (not strip): with base's validation_split_num=0,
        # stripping would let base's 0 win and silently turn validation back off.
        if split_num and split_num > 0:
            effective = int(split_num)
        elif base_vsn > 0:
            effective = base_vsn
        else:
            effective = _DEFAULT_VALIDATION_SPLIT_NUM
        if effective == base_vsn and base_vsn > 0:
            # Base already enables this exact count — strip any override so base.toml wins.
            if not isinstance(existing, list) or not existing:
                return
            first = existing[0]
            if not isinstance(first, dict):
                return
            first.pop("validation_split_num", None)
            first.pop("validation_split", None)
            if not first and len(existing) == 1:
                del out["datasets"]
            return
        if not isinstance(existing, list):
            existing = []
            out["datasets"] = existing
        if not existing:
            existing.append({})
        first = existing[0]
        if not isinstance(first, dict):
            first = {}
            existing[0] = first
        first["validation_split_num"] = effective
        first.pop("validation_split", None)
        return

    if not isinstance(existing, list):
        existing = []
        out["datasets"] = existing
    if not existing:
        existing.append({})
    first = existing[0]
    if not isinstance(first, dict):
        first = {}
        existing[0] = first
    first["validation_split_num"] = 0
    first["validation_split"] = 0.0
