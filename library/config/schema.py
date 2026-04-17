"""Config key schema and TOML validation.

The schema is seeded at runtime by walking ``train.setup_parser()._actions``
(via :func:`populate_schema`). TOML-only or non-argparse keys are registered
manually via ``extras``.

Once populated, :func:`validate_entry` is the single place that decides:

* whether a key is known (or an alias of one),
* whether a value is within declared ``choices``,
* type-coercion of soft mismatches (e.g. TOML ``1`` where ``float`` is wanted).

Warnings get a ``file:line`` locator when possible so a typo in
``configs/methods/lora.toml`` surfaces with the exact offending line before
a two-hour run starts.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Any, Optional


class ConfigSchemaError(ValueError):
    """Raised when strict-mode schema validation rejects a key/value."""


@dataclass(frozen=True)
class ConfigKey:
    name: str
    type: str = "str"  # "int" | "float" | "str" | "bool" | "path" | "list[str]" | ...
    default: Any = None
    choices: tuple = ()
    nargs: Any = None
    action: str = "store"
    help: str = ""
    aliases: tuple = ()
    source: str = "argparse"


CONFIG_SCHEMA: dict[str, ConfigKey] = {}
ALIAS_MAP: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

_BOOL_ACTIONS = {"_StoreTrueAction", "_StoreFalseAction"}


def _action_type_str(action: argparse.Action) -> str:
    cls = action.__class__.__name__
    if cls in _BOOL_ACTIONS or cls == "BooleanOptionalAction":
        return "bool"
    t = action.type
    if t is None:
        return "str"
    if t is int:
        return "int"
    if t is float:
        return "float"
    if t is str:
        return "str"
    name = getattr(t, "__name__", str(t))
    if name in ("Path", "PosixPath", "WindowsPath"):
        return "path"
    return name


def _nargs_wrap(base: str, nargs: Any) -> str:
    if nargs in ("*", "+") or isinstance(nargs, int):
        return f"list[{base}]"
    return base


def _key_from_action(action: argparse.Action) -> Optional[ConfigKey]:
    name: Optional[str] = None
    for opt in action.option_strings:
        if opt.startswith("--"):
            name = opt[2:]
            break
    if name is None:
        return None
    name = name.replace("-", "_")
    base = _action_type_str(action)
    tstr = base if base == "bool" else _nargs_wrap(base, action.nargs)
    choices = tuple(action.choices) if action.choices else ()
    return ConfigKey(
        name=name,
        type=tstr,
        default=action.default,
        choices=choices,
        nargs=action.nargs,
        action=action.__class__.__name__,
        help=(action.help or "").strip(),
    )


def populate_schema(
    parser: argparse.ArgumentParser,
    extras: Optional[dict[str, ConfigKey]] = None,
) -> None:
    """(Re)populate ``CONFIG_SCHEMA`` from a parser's actions + manual extras.

    Idempotent: replaces the previous contents. Call this once after
    ``setup_parser()`` in ``train.py``; tests may call it directly.
    """
    CONFIG_SCHEMA.clear()
    ALIAS_MAP.clear()
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction,)):
            continue
        if action.__class__.__name__ == "_VersionAction":
            continue
        key = _key_from_action(action)
        if key is None:
            continue
        CONFIG_SCHEMA[key.name] = key

    # Manual TOML-only / non-argparse extras. `base_config` is the only one
    # essential today; future methods can extend via ``extras``.
    CONFIG_SCHEMA.setdefault(
        "base_config",
        ConfigKey(
            name="base_config",
            type="str",
            default=None,
            help=(
                "Relative path to a parent TOML whose keys are merged before "
                "this file. Resolved recursively."
            ),
            source="manual",
        ),
    )

    if extras:
        for k, v in extras.items():
            CONFIG_SCHEMA[k] = v

    for k, v in CONFIG_SCHEMA.items():
        for a in v.aliases:
            ALIAS_MAP[a] = k


def get_schema() -> dict[str, ConfigKey]:
    return CONFIG_SCHEMA


def resolve_alias(key: str) -> str:
    return ALIAS_MAP.get(key, key)


def is_known_key(key: str) -> bool:
    return resolve_alias(key) in CONFIG_SCHEMA


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _coerce_value(spec: ConfigKey, value: Any) -> Any:
    t = spec.type
    if t.startswith("list["):
        return value
    if value is None:
        return value
    if t == "bool" and not isinstance(value, bool):
        return bool(value)
    if t == "float" and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if t == "int" and isinstance(value, float) and value.is_integer():
        return int(value)
    if t == "path" and not isinstance(value, str):
        return str(value)
    return value


def find_line(text: Optional[str], key: str) -> Optional[int]:
    """Best-effort line locator for a flat TOML assignment."""
    if not text:
        return None
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for i, line in enumerate(text.splitlines(), 1):
        if pattern.match(line):
            return i
    return None


def validate_entry(
    key: str,
    value: Any,
    *,
    source: Optional[str] = None,
    line: Optional[int] = None,
    strict: bool = False,
    logger: Any = None,
) -> tuple[str, Any]:
    """Validate & coerce a single (key, value).

    Returns ``(resolved_key, coerced_value)``. If the schema is empty (not yet
    populated), returns the input unchanged — this keeps ``load_method_preset``
    usable in contexts that never call ``populate_schema``.
    """
    if not CONFIG_SCHEMA:
        return key, value

    resolved = resolve_alias(key)
    loc = source or "<config>"
    if line is not None:
        loc = f"{loc}:{line}"

    if resolved not in CONFIG_SCHEMA:
        msg = f"[config] {loc}: unknown key {key!r}"
        if strict:
            raise ConfigSchemaError(msg)
        if logger is not None:
            logger.warning(msg)
        return key, value

    spec = CONFIG_SCHEMA[resolved]
    coerced = _coerce_value(spec, value)

    if spec.choices and coerced not in spec.choices:
        # argparse tolerates ``None`` when it's the declared default; argparse
        # puts ``None`` into ``choices`` explicitly in a few places.
        if not (coerced is None and None in spec.choices):
            msg = (
                f"[config] {loc}: {resolved!r} = {coerced!r} "
                f"not in choices {list(spec.choices)}"
            )
            if strict:
                raise ConfigSchemaError(msg)
            if logger is not None:
                logger.warning(msg)

    return resolved, coerced
