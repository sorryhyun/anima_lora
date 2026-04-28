"""TOML config loading, merging, and snapshot rendering.

Chain: ``base.toml`` → ``presets.toml[<preset>]`` → ``methods/<method>.toml`` → CLI.
Method settings win over preset settings on overlap (so a method can force its
own hardware constraints, e.g. ``blocks_to_swap=0``).

``_DATASET_CONFIG_SECTIONS`` names top-level TOML keys that belong to the
dataset blueprint (``[general]`` / ``[[datasets]]``). They're consumed by the
dataset blueprint generator and skipped by the flat argparse merge.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import subprocess
from typing import Any, Optional

import toml

from library.config import schema as _config_schema

logger = logging.getLogger(__name__)

_DATASET_CONFIG_SECTIONS = {"general", "datasets"}
_SNAPSHOT_SUFFIX = ".snapshot.toml"
_DUMP_SKIP_KEYS = {
    "print_config",
    "config_snapshot",
    "config_strict",
    "config_file",
    "output_config",
    "wandb_api_key",
    "huggingface_token",
}


def _read_text_silent(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _flatten_toml(
    d: dict,
    *,
    source: Optional[str] = None,
    strict: bool = False,
) -> dict:
    """Flatten top-level sections into a single namespace (ignores nesting).

    When ``source`` is given and the schema has been populated via
    ``config_schema.populate_schema``, each leaf is validated: unknown keys
    warn (or raise in strict mode), off-choice values warn, and soft type
    mismatches (TOML ``1`` where a ``float`` is wanted) are coerced.

    ``[general]`` and ``[[datasets]]`` sections are skipped — they're consumed
    by the dataset blueprint generator, not the argparse namespace.
    """
    out: dict = {}
    src_text = _read_text_silent(source)

    def _visit(key: str, value: Any) -> None:
        line = _config_schema.find_line(src_text, key)
        resolved, coerced = _config_schema.validate_entry(
            key,
            value,
            source=source,
            line=line,
            strict=strict,
            logger=logger,
        )
        out[resolved] = coerced

    for k, v in d.items():
        if k in _DATASET_CONFIG_SECTIONS:
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                _visit(kk, vv)
        else:
            _visit(k, v)
    return out


class _SafeFormatDict(dict):
    """`str.format_map` helper: leaves unknown ``{key}`` placeholders intact
    rather than raising ``KeyError``. Lets captions / paths with literal
    braces pass through untouched."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _substitute_templates(value: Any, ctx: dict) -> Any:
    """Recursively format-substitute string values in nested dict/list trees.

    Used by ``load_dataset_config_from_base`` so the dataset blueprint can
    reference top-level path keys (e.g. ``image_dir = '{resized_image_dir}'``)
    and stay in sync with ``tasks.py`` preprocess commands when users override
    those paths via preset / method.
    """
    if isinstance(value, str):
        if "{" not in value:
            return value
        try:
            return value.format_map(_SafeFormatDict(ctx))
        except (ValueError, IndexError):
            return value
    if isinstance(value, dict):
        return {k: _substitute_templates(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_templates(v, ctx) for v in value]
    return value


def load_dataset_config_from_base(
    configs_dir: str = "configs",
    overrides: Optional[dict] = None,
) -> Optional[dict]:
    """Extract the dataset blueprint (``[general]`` + ``[[datasets]]``) from
    ``configs/base.toml``. Returns ``None`` if no dataset sections are present,
    so callers can fall back to the DreamBooth/in_json code paths.

    String values in the blueprint may reference top-level scalar keys via
    ``{key}`` placeholders; these are substituted at load time. ``overrides``
    (typically the merged preset/method args namespace) wins over the raw
    base.toml top-level — that's how preset / CLI overrides of
    ``resized_image_dir`` etc. propagate into the dataset subset paths.
    """
    base_path = os.path.join(configs_dir, "base.toml")
    if not os.path.exists(base_path):
        return None
    with open(base_path, "r", encoding="utf-8") as f:
        raw = toml.load(f)
    sections = {k: v for k, v in raw.items() if k in _DATASET_CONFIG_SECTIONS}
    if not sections.get("datasets"):
        return None
    ctx = {
        k: v
        for k, v in raw.items()
        if k not in _DATASET_CONFIG_SECTIONS and isinstance(v, str)
    }
    if overrides:
        ctx.update({k: v for k, v in overrides.items() if isinstance(v, str)})
    return _substitute_templates(sections, ctx)


def load_path_overrides(
    preset: str = "default",
    configs_dir: str = "configs",
    method: Optional[str] = None,
    methods_subdir: str = "methods",
) -> dict:
    """Top-level scalar keys from base.toml → ``presets.toml[<preset>]`` →
    ``<methods_subdir>/<method>.toml`` (when given).

    Lightweight — used by ``tasks.py`` preprocess commands so they pick up
    ``source_image_dir`` / ``resized_image_dir`` / ``lora_cache_dir`` overrides
    without launching accelerate. The method layer is the same one training
    uses (so a value set in ``configs/gui-methods/lora.toml`` is honored by
    preprocess too). Missing files / unknown presets are silently ignored —
    callers fall back to whatever earlier layer provided a value, then to
    hard-coded defaults.
    """
    out: dict = {}

    def _flat_scalars(d: dict) -> dict:
        """Pluck top-level non-container values, skipping dataset blueprint
        sections (``[general]`` / ``[[datasets]]``)."""
        return {
            k: v
            for k, v in d.items()
            if k not in _DATASET_CONFIG_SECTIONS
            and not isinstance(v, (dict, list))
        }

    base_path = os.path.join(configs_dir, "base.toml")
    if os.path.exists(base_path):
        with open(base_path, "r", encoding="utf-8") as f:
            out.update(_flat_scalars(toml.load(f)))

    try:
        section, _path, _tag = _resolve_preset(preset, configs_dir)
        out.update(_flat_scalars(section))
    except (KeyError, FileNotFoundError, ValueError):
        pass

    if method:
        method_path = os.path.join(configs_dir, methods_subdir, f"{method}.toml")
        if os.path.exists(method_path):
            with open(method_path, "r", encoding="utf-8") as f:
                out.update(_flat_scalars(toml.load(f)))

    return out


def _load_toml_with_base(path: str, *, strict: bool = False) -> dict:
    """Load a TOML file and recursively resolve its 'base_config' reference."""
    with open(path, "r", encoding="utf-8") as f:
        config_dict = toml.load(f)
    base_ref = config_dict.pop("base_config", None)
    if base_ref is None:
        return _flatten_toml(config_dict, source=path, strict=strict)
    if not os.path.isabs(base_ref):
        base_ref = os.path.join(os.path.dirname(path), base_ref)
    logger.info(f"Loading base config from {base_ref}...")
    base_dict = _load_toml_with_base(base_ref, strict=strict)
    merged = dict(base_dict)
    merged.update(_flatten_toml(config_dict, source=path, strict=strict))
    return merged


def _resolve_preset(
    preset: str, configs_dir: str = "configs"
) -> tuple[dict, str, str]:
    """Resolve a preset name to ``(section, source_path, source_tag)``.

    Looks in ``configs/presets.toml`` first (built-in sections); falls back to
    ``configs/custom/<preset>.toml`` (one file per user-created preset, flat
    key=value with no section header — the filename is the preset name).
    """
    presets_path = os.path.join(configs_dir, "presets.toml")
    if os.path.exists(presets_path):
        with open(presets_path, "r", encoding="utf-8") as f:
            presets = toml.load(f)
        if preset in presets:
            section = presets[preset]
            if not isinstance(section, dict):
                raise ValueError(
                    f"Preset '{preset}' in {presets_path} is not a table"
                )
            return dict(section), presets_path, f"{presets_path}[{preset}]"
    custom_path = os.path.join(configs_dir, "custom", f"{preset}.toml")
    if os.path.exists(custom_path):
        with open(custom_path, "r", encoding="utf-8") as f:
            data = toml.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Custom preset {custom_path} is not a TOML table")
        return data, custom_path, custom_path
    available: list[str] = []
    if os.path.exists(presets_path):
        with open(presets_path, "r", encoding="utf-8") as f:
            available.extend(sorted(toml.load(f)))
    custom_dir = os.path.join(configs_dir, "custom")
    if os.path.isdir(custom_dir):
        available.extend(
            sorted(n[:-5] for n in os.listdir(custom_dir) if n.endswith(".toml"))
        )
    raise KeyError(
        f"Preset '{preset}' not found in {presets_path} or {custom_path}. "
        f"Available: {sorted(set(available))}"
    )


def load_preset_section(preset: str, configs_dir: str = "configs") -> dict:
    """Load a named preset section from configs/presets.toml or configs/custom/."""
    section, _path, _tag = _resolve_preset(preset, configs_dir)
    return section


def load_method_preset(
    method: str,
    preset: str = "default",
    configs_dir: str = "configs",
    methods_subdir: str = "methods",
    *,
    strict: bool = False,
    return_provenance: bool = False,
):
    """Merge base.toml → presets.toml[<preset>] → <methods_subdir>/<method>.toml into a flat dict.

    Method settings win over preset settings on overlap (e.g. postfix can force
    blocks_to_swap=0 regardless of the hardware preset).

    `methods_subdir` selects which folder under `configs_dir` holds the method
    files. Defaults to ``"methods"``; pass ``"gui-methods"`` to pick up the
    clean, self-contained per-variant files used by the GUI / `make lora-gui`
    path instead of the toggle-block method files.

    When ``return_provenance=True`` returns ``(merged, provenance)`` where
    ``provenance[key]`` is a short human-readable source tag (e.g.
    ``"configs/presets.toml[default]"``).
    """
    base_path = os.path.join(configs_dir, "base.toml")
    method_path = os.path.join(configs_dir, methods_subdir, f"{method}.toml")
    for p in (base_path, method_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Config file not found: {p}")

    merged: dict = {}
    provenance: dict[str, str] = {}

    with open(base_path, "r", encoding="utf-8") as f:
        base_raw = toml.load(f)
    base_flat = _flatten_toml(base_raw, source=base_path, strict=strict)
    for k, v in base_flat.items():
        merged[k] = v
        provenance[k] = base_path

    preset_section, preset_path, preset_tag = _resolve_preset(preset, configs_dir)
    preset_flat = _flatten_toml(
        {preset: preset_section}, source=preset_path, strict=strict
    )
    for k, v in preset_flat.items():
        merged[k] = v
        provenance[k] = preset_tag

    with open(method_path, "r", encoding="utf-8") as f:
        method_raw = toml.load(f)
    method_flat = _flatten_toml(method_raw, source=method_path, strict=strict)
    for k, v in method_flat.items():
        merged[k] = v
        provenance[k] = method_path

    if return_provenance:
        return merged, provenance
    return merged


def _git_sha() -> Optional[str]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _format_toml_line(key: str, value: Any) -> Optional[str]:
    """Render one TOML assignment. Returns None if ``value`` can't be encoded
    (e.g. ``None``) — such keys are skipped from the dump."""
    if value is None:
        return None
    try:
        dumped = toml.dumps({key: value}).strip()
    except (TypeError, ValueError):
        return None
    return dumped


def _collect_dump_entries(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    provenance: dict[str, str],
) -> list[tuple[str, Any, str]]:
    """Return (key, value, source) triples worth dumping.

    Includes every key from ``provenance`` (i.e. came from a TOML layer) plus
    any CLI override — detected as ``args[k] != defaults[k]`` for keys not in
    provenance.
    """
    defaults = vars(parser.parse_args([]))
    args_dict = vars(args)
    entries: list[tuple[str, Any, str]] = []
    for key in sorted(args_dict):
        if key in _DUMP_SKIP_KEYS:
            continue
        value = args_dict[key]
        if key in provenance:
            entries.append((key, value, provenance[key]))
        elif key in defaults and value != defaults[key]:
            entries.append((key, value, "CLI"))
    return entries


def _render_merged_toml(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    provenance: dict[str, str],
) -> str:
    """Produce a provenance-annotated TOML dump of the merged config."""
    entries = _collect_dump_entries(args, parser, provenance)
    lines: list[str] = []
    method = getattr(args, "method", None) or "<unset>"
    preset = getattr(args, "preset", None) or "<unset>"
    lines.append("# Merged config — generated by train.py --print-config")
    lines.append(f"# Method: {method}")
    lines.append(f"# Preset: {preset}")
    sha = _git_sha()
    if sha:
        lines.append(f"# Git: {sha}")
    lines.append("")

    # Group by source so the dump reads top-down: base → preset → method → CLI.
    by_source: dict[str, list[tuple[str, Any]]] = {}
    for key, value, source in entries:
        by_source.setdefault(source, []).append((key, value))

    def _rank(src: str) -> int:
        if src == "configs/base.toml":
            return 0
        if src.startswith("configs/presets.toml") or src.startswith(
            "configs/custom/"
        ):
            return 1
        # Method file — lives under configs/methods/ by default, or under
        # configs/gui-methods/ when --methods_subdir=gui-methods is used.
        if src.startswith("configs/methods/") or src.startswith("configs/gui-methods/"):
            return 2
        if src == "CLI":
            return 4
        return 3

    order = sorted(by_source, key=_rank)

    for source in order:
        lines.append(f"# --- from {source} ---")
        for key, value in by_source[source]:
            rendered = _format_toml_line(key, value)
            if rendered is None:
                lines.append(f"# {key} = <null>")
            else:
                lines.append(rendered)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _write_config_snapshot(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    provenance: dict[str, str],
) -> Optional[str]:
    output_dir = getattr(args, "output_dir", None)
    output_name = getattr(args, "output_name", None)
    if not output_dir or not output_name:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{output_name}{_SNAPSHOT_SUFFIX}")
    rendered = _render_merged_toml(args, parser, provenance)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(rendered)
    except OSError as e:
        logger.warning(f"Could not write config snapshot to {path}: {e}")
        return None
    logger.info(f"Config snapshot written: {path}")
    return path


def read_config_from_file(args: argparse.Namespace, parser: argparse.ArgumentParser):
    strict = bool(getattr(args, "config_strict", False))
    print_config = bool(getattr(args, "print_config", False))
    write_snapshot = bool(getattr(args, "config_snapshot", False))

    # New-style chain: --method / --preset
    method = getattr(args, "method", None)
    preset = getattr(args, "preset", None) or "default"
    methods_subdir = getattr(args, "methods_subdir", None) or "methods"
    if method is not None and not args.config_file:
        logger.info(
            f"Loading chain: base → presets/{preset} → {methods_subdir}/{method}"
        )
        try:
            merged, provenance = load_method_preset(
                method,
                preset,
                methods_subdir=methods_subdir,
                strict=strict,
                return_provenance=True,
            )
        except FileNotFoundError as e:
            logger.error(str(e))
            exit(1)

        config_args = argparse.Namespace(**merged)
        args = parser.parse_args(namespace=config_args)
        args.config_file = os.path.join("configs", methods_subdir, f"{method}.toml")

        if print_config:
            import sys as _sys

            _sys.stdout.write(_render_merged_toml(args, parser, provenance))
            _sys.stdout.flush()
            exit(0)

        if write_snapshot:
            _write_config_snapshot(args, parser, provenance)

        return args

    if not args.config_file:
        if print_config:
            import sys as _sys

            _sys.stdout.write(_render_merged_toml(args, parser, {}))
            _sys.stdout.flush()
            exit(0)
        return args

    config_path = (
        args.config_file + ".toml"
        if not args.config_file.endswith(".toml")
        else args.config_file
    )

    if args.output_config:
        if os.path.exists(config_path):
            logger.error("Config file already exists. Aborting...")
            exit(1)

        args_dict = vars(args)

        for key in ["config_file", "output_config", "wandb_api_key"]:
            if key in args_dict:
                del args_dict[key]

        default_args = vars(parser.parse_args([]))

        for key, value in list(args_dict.items()):
            if key in default_args and value == default_args[key]:
                del args_dict[key]

        for key, value in args_dict.items():
            if isinstance(value, pathlib.Path):
                args_dict[key] = str(value)

        with open(config_path, "w") as f:
            toml.dump(args_dict, f)

        logger.info("Saved config file")
        exit(0)

    if not os.path.exists(config_path):
        logger.info(f"{config_path} not found.")
        exit(1)

    logger.info(f"Loading settings from {config_path}...")
    merged = _load_toml_with_base(config_path, strict=strict)

    config_args = argparse.Namespace(**merged)
    args = parser.parse_args(namespace=config_args)
    args.config_file = os.path.splitext(args.config_file)[0]

    if print_config:
        import sys as _sys

        provenance = {k: config_path for k in merged}
        _sys.stdout.write(_render_merged_toml(args, parser, provenance))
        _sys.stdout.flush()
        exit(0)

    if write_snapshot:
        provenance = {k: config_path for k in merged}
        _write_config_snapshot(args, parser, provenance)

    return args
