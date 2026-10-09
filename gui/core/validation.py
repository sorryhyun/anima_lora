"""Dataset-blueprint override encoding for the GUI config form.

Pure-dict logic (no Qt, no other ``gui`` imports): reads virtual-key state
(``use_valid`` / ``validation_split_num`` / ``repeat_by_folder_name``) out of
a TOML ``[[datasets]]`` block and writes the user's choice back into a variant
dict. Consumed by ``gui.core.config_io`` (merge-time injection of the virtual keys)
and the Config tab (apply on save).
"""

from __future__ import annotations

from typing import Any, Optional

# Held-out slice written when validation is enabled but no positive validation_split_num
# exists (form or base.toml). Matches the historical base.toml default (library/config/loader.py).
_DEFAULT_VALIDATION_SPLIT_NUM = 16


def _validation_enabled_from_datasets(datasets: Any) -> Optional[bool]:
    """Inspect a TOML ``[[datasets]]`` list and decide whether validation is
    enabled. Returns ``True`` / ``False`` when either validation key is
    explicitly set on the first dataset entry, or ``None`` when no override
    is present (caller falls back to the parent layer in the merge chain)."""
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
    """Return the variant TOML's explicit use_valid override, or None when
    the variant doesn't touch validation_split[_num]."""
    return _validation_enabled_from_datasets(variant_data.get("datasets"))


def _base_validation_enabled(base_data: dict) -> bool:
    """Default use_valid pulled from configs/base.toml's [[datasets]] block.
    Falls back to False when the block is missing — matches the
    `validation_split == 0 and validation_split_num <= 0` short-circuit in
    library/config/loader.py:generate_dataset_group_by_blueprint."""
    return bool(_validation_enabled_from_datasets(base_data.get("datasets")))


def _validation_split_num_from_datasets(datasets: Any) -> Optional[int]:
    """Pull ``validation_split_num`` off the first [[datasets]] entry as an
    int. Returns None when the block is missing or the key isn't set."""
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
    """Return the variant TOML's explicit validation_split_num override, or
    None when the variant doesn't touch it."""
    return _validation_split_num_from_datasets(variant_data.get("datasets"))


def _base_validation_split_num(base_data: dict) -> int:
    """Default validation_split_num pulled from configs/base.toml. Falls back
    to 0 when the block / key is missing."""
    return _validation_split_num_from_datasets(base_data.get("datasets")) or 0


def _folder_repeats_from_datasets(datasets: Any) -> Optional[bool]:
    """Pull ``repeat_by_folder_name`` off the first ``[[datasets]]`` entry
    (falling back to its first subset, where older hand-edited configs may
    carry it). Returns ``None`` when no override is present."""
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
    """Return the variant TOML's explicit repeat_by_folder_name override, or
    None when the variant doesn't touch it."""
    return _folder_repeats_from_datasets(variant_data.get("datasets"))


def _base_folder_repeats(base_data: dict) -> bool:
    """Default repeat_by_folder_name pulled from configs/base.toml's
    ``[[datasets]]`` block (or ``[general]``). Falls back to False — matching
    the BaseSubsetParams dataclass default."""
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
    """Encode the repeat_by_folder_name checkbox into the variant TOML dict
    ``out``.

    Matches base → strip any override so base.toml stays the single source of
    truth. Differs → write ``repeat_by_folder_name = <enabled>`` on the first
    ``[[datasets]]`` entry (dataset level, NOT subsets: _apply_dataset_overrides
    in library/config/io.py only merges top-level dataset scalars, and the key
    is subset-ascendable so it reaches every subset from there). Other keys in
    the variant's [[datasets]] block are preserved."""
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
    """Encode the use_valid checkbox (+ optional validation_split_num int)
    into the variant TOML dict ``out``.

    Enabled  → resolve the held-out count that should take effect (the form
               ``split_num`` if positive, else ``base_split_num`` if positive,
               else ``_DEFAULT_VALIDATION_SPLIT_NUM``). When that count equals
               an already-enabled base, strip both keys so base.toml stays the
               single source of truth; otherwise write {validation_split_num =
               <count>} on the first [[datasets]] entry (dropping any fractional
               validation_split). The fallback to a positive default is what
               makes ticking the checkbox stick when base.toml ships
               validation_split_num=0 — without it "enabled" would strip to
               base's disabled 0 and silently turn validation back off.
    Disabled → write {validation_split_num = 0, validation_split = 0.0} on the
               first [[datasets]] entry, creating the block if absent. This is
               applied by _apply_dataset_overrides in library/config/io.py and
               causes generate_dataset_group_by_blueprint to skip the val set.
               (The ``split_num`` int is ignored when disabled.)

    Other keys in the variant's [[datasets]] block (e.g. a custom batch_size)
    are preserved; we only touch the two validation keys."""
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
