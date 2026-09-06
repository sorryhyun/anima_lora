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
from typing import Any, Optional, Sequence

import toml

from library.config import schema as _config_schema
from library.env import anima_home, resolve_under_home
from library.runtime.proc import no_window_kwargs

logger = logging.getLogger(__name__)

_DATASET_CONFIG_SECTIONS = {"general", "datasets"}
# tooling metadata (GUI variant registry, [<preset>.gui] display info, [soup]
# pipeline knobs), not argparse values — stripped before flattening
_METADATA_CONFIG_SECTIONS = {"variant", "gui", "soup"}
_NON_FLAT_SECTIONS = _DATASET_CONFIG_SECTIONS | _METADATA_CONFIG_SECTIONS
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


def toml_get(cfg: dict, key_path: str, default: Any = None) -> Any:
    """Look up a dotted ``a.b.c`` path in a nested (TOML) dict.

    Returns ``default`` if any segment is missing. Used by the bespoke-schema
    distill scripts (``spd.toml`` / ``turbo.toml``, read raw rather than
    through :func:`load_method_preset`). Unlike :func:`_flatten_toml` this
    does NOT collapse sections — it walks the exact path.
    """
    node: Any = cfg
    for part in key_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


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

    ``[general]`` / ``[[datasets]]`` are consumed by the dataset blueprint
    generator; ``[variant]`` carries GUI registry metadata. Both kinds of
    section are stripped before the argparse-flat merge.
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
        if k in _NON_FLAT_SECTIONS:
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


def _resolve_method_path(
    configs_dir: str, methods_subdir: str, method: Optional[str]
) -> Optional[str]:
    """Resolve the on-disk path of a method TOML.

    A method may live in the flat folder (``configs/<methods_subdir>/
    <method>.toml``) or in its own consolidated directory
    (``configs/<method>/<method>.toml`` — method config + inline dataset
    blueprint in one file, the EasyControl pilot). The per-method dir wins
    when present, but only for the default ``methods`` subdir — ``gui-methods``
    stays flat. Returns the flat-folder path when no per-method file exists,
    or ``None`` when ``method`` is falsy.
    """
    if not method:
        return None
    if methods_subdir == "methods":
        per_method = os.path.join(configs_dir, method, f"{method}.toml")
        if os.path.exists(per_method):
            return per_method
    return os.path.join(configs_dir, methods_subdir, f"{method}.toml")


def _normalize_config_path(config_file: str) -> str:
    return config_file if config_file.endswith(".toml") else config_file + ".toml"


def _is_scalar_list(v) -> bool:
    """A list of plain scalars (e.g. ``target_res = [768, 1024]``)."""
    return isinstance(v, list) and all(not isinstance(e, (dict, list)) for e in v)


def _flat_scalars(d: dict) -> dict:
    """Top-level flat values usable by preprocess path overrides."""
    return {
        k: v
        for k, v in d.items()
        if k not in _NON_FLAT_SECTIONS
        and (not isinstance(v, (dict, list)) or _is_scalar_list(v))
    }


def load_path_overrides_from_config(config_file: str) -> dict:
    """Top-level scalar keys from one immutable config file."""
    path = _normalize_config_path(str(config_file))
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return _flat_scalars(toml.load(f))


def _apply_dataset_overrides(blueprint: dict, override: dict) -> None:
    """Shallow-merge override sections into ``blueprint`` in place.

    ``[general]``: per-key overwrite. ``[[datasets]]``: matched by index
    against the base blueprint, top-level scalars only — ``subsets`` arrays
    in the override are ignored with a warning (subset-level overrides are
    out of scope, to keep the merge predictable).
    """
    g_override = override.get("general")
    if isinstance(g_override, dict):
        base_general = blueprint.setdefault("general", {})
        for k, v in g_override.items():
            base_general[k] = v

    override_datasets = override.get("datasets") or []
    base_datasets = blueprint.get("datasets") or []
    for i, override_ds in enumerate(override_datasets):
        if not isinstance(override_ds, dict):
            continue
        if i >= len(base_datasets):
            logger.warning(
                "Dataset override index %d has no matching base dataset; ignoring.",
                i,
            )
            continue
        if "subsets" in override_ds:
            logger.warning(
                "Dataset override index %d declares [[datasets.subsets]]; "
                "subset-level overrides are not supported. Ignoring `subsets`.",
                i,
            )
        for k, v in override_ds.items():
            if k == "subsets":
                continue
            base_datasets[i][k] = v


def load_dataset_config_from_base(
    configs_dir: str = "configs",
    overrides: Optional[dict] = None,
    *,
    method: Optional[str] = None,
    methods_subdir: str = "methods",
    config_file: Optional[str] = None,
) -> Optional[dict]:
    """Extract the dataset blueprint (``[general]`` + ``[[datasets]]``) from
    ``configs/base.toml``. Returns ``None`` if no dataset sections are present,
    so callers can fall back to the DreamBooth/in_json code paths.

    String values may reference top-level scalar keys via ``{key}``
    placeholders, substituted at load time; ``overrides`` (the merged
    preset/method namespace) wins over the raw base.toml top-level, which is
    how preset/CLI overrides of ``resized_image_dir`` etc. propagate into
    subset paths.

    When ``method`` is given, its ``[general]``/``[[datasets]]`` sections
    shallow-override the base blueprint (see ``_apply_dataset_overrides``) —
    unless it carries a *full* inline blueprint, which replaces it wholesale.
    """
    configs_dir = str(resolve_under_home(configs_dir))
    base_path = os.path.join(configs_dir, "base.toml")
    if not os.path.exists(base_path):
        return None
    with open(base_path, "r", encoding="utf-8") as f:
        raw = toml.load(f)
    source_raw = raw
    sections = {k: v for k, v in raw.items() if k in _DATASET_CONFIG_SECTIONS}

    if config_file:
        cfg_path = _normalize_config_path(str(config_file))
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_raw = toml.load(f)
            cfg_sections = {
                k: v for k, v in cfg_raw.items() if k in _DATASET_CONFIG_SECTIONS
            }
            if cfg_sections.get("datasets"):
                sections = cfg_sections
                source_raw = cfg_raw

    if method is not None and not config_file:
        method_path = _resolve_method_path(configs_dir, methods_subdir, method)
        if method_path and os.path.exists(method_path):
            with open(method_path, "r", encoding="utf-8") as f:
                method_raw = toml.load(f)
            method_sections = {
                k: v for k, v in method_raw.items() if k in _DATASET_CONFIG_SECTIONS
            }
            # full blueprint == [[datasets]] with a subset -> replace wholesale
            # (mirrors config_file); scalar-only sections shallow-override so a
            # method bumping batch_size doesn't drop base's subsets
            # (tests/test_folder_repeats.py)
            method_has_full_blueprint = any(
                isinstance(d, dict) and d.get("subsets")
                for d in (method_sections.get("datasets") or [])
            )
            if method_has_full_blueprint:
                sections = method_sections
                source_raw = method_raw
            elif method_sections:
                _apply_dataset_overrides(sections, method_sections)

    if not sections.get("datasets"):
        return None

    ctx = {
        k: v
        for k, v in source_raw.items()
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
    """Top-level scalar keys from ``preprocess.toml`` → base.toml →
    ``presets.toml[<preset>]`` → ``<methods_subdir>/<method>.toml`` (when given).

    Lightweight — used by ``tasks.py`` preprocess commands so they pick up
    ``source_image_dir`` / ``resized_image_dir`` / ``lora_cache_dir`` overrides
    without launching accelerate. Missing files / unknown presets are silently
    ignored — callers fall back to an earlier layer, then hard-coded defaults.
    """
    configs_dir = str(resolve_under_home(configs_dir))
    out: dict = {}

    # read preprocess.toml FIRST, before base.toml, so a legacy copy of its
    # keys left in a user's customized base.toml still wins (never regress an
    # existing customization); preset/method layers below still override
    preprocess_path = os.path.join(configs_dir, "preprocess.toml")
    if os.path.exists(preprocess_path):
        with open(preprocess_path, "r", encoding="utf-8") as f:
            out.update(_flat_scalars(toml.load(f)))

    # exception: target_res is owned by preprocess.toml (preserved across
    # `make update`), so a stray copy in base.toml must NOT clobber it
    pp_has_target_res = "target_res" in out

    # base → preset → method, each projected to its flat scalars (later wins).
    for _kind, _path, _tag, raw in _iter_method_preset_layers(
        preset, configs_dir, methods_subdir, method, require_files=False
    ):
        flat = _flat_scalars(raw)
        if _kind == "base" and pp_has_target_res:
            flat.pop("target_res", None)
        out.update(flat)

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


def _posix(path: str) -> str:
    """Normalize a path to forward slashes for stable, cross-platform
    human-readable provenance tags (e.g. ``configs/base.toml`` on Windows too).
    """
    return path.replace(os.sep, "/")


def _display_path(path: str) -> str:
    """Human-readable provenance tag for a config file.

    File access uses home-anchored absolute paths (so loading works from any
    CWD), but the tags that land in ``.snapshot.toml`` / ``print-config`` stay
    repo-relative (``configs/base.toml``) instead of leaking the user's home
    directory. Paths outside the repo home are shown absolute.
    """
    try:
        rel = pathlib.Path(path).resolve().relative_to(anima_home())
        return _posix(str(rel))
    except ValueError:
        return _posix(path)


def _resolve_preset(preset: str, configs_dir: str = "configs") -> tuple[dict, str, str]:
    """Resolve a preset name to ``(section, source_path, source_tag)``.

    Looks in ``configs/presets.toml`` first (built-in sections); falls back to
    ``configs/custom/<preset>.toml`` (one file per user-created preset, flat
    key=value with no section header — the filename is the preset name).

    Metadata sub-tables (``[<preset>.gui]`` — display label/group for the GUI
    preset picker) are stripped so they never reach the flat argparse merge.
    """
    configs_dir = str(resolve_under_home(configs_dir))

    def _strip_meta(section: dict) -> dict:
        return {k: v for k, v in section.items() if k not in _METADATA_CONFIG_SECTIONS}

    presets_path = os.path.join(configs_dir, "presets.toml")
    if os.path.exists(presets_path):
        with open(presets_path, "r", encoding="utf-8") as f:
            presets = toml.load(f)
        if preset in presets:
            section = presets[preset]
            if not isinstance(section, dict):
                raise ValueError(f"Preset '{preset}' in {presets_path} is not a table")
            return (
                _strip_meta(section),
                presets_path,
                f"{_display_path(presets_path)}[{preset}]",
            )
    custom_path = os.path.join(configs_dir, "custom", f"{preset}.toml")
    if os.path.exists(custom_path):
        with open(custom_path, "r", encoding="utf-8") as f:
            data = toml.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Custom preset {custom_path} is not a TOML table")
        return _strip_meta(data), custom_path, _display_path(custom_path)
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


def list_presets(configs_dir: str = "configs") -> list[str]:
    """Every resolvable preset name, sorted.

    The union of ``[<name>]`` table sections in ``configs/presets.toml`` and the
    one-file-per-preset stems under ``configs/custom/*.toml`` — i.e. exactly the
    set :func:`_resolve_preset` can resolve. Shared with the GUI so it doesn't
    re-derive the preset discovery rule.
    """
    configs_dir = str(resolve_under_home(configs_dir))
    names: set[str] = set()
    presets_path = os.path.join(configs_dir, "presets.toml")
    if os.path.exists(presets_path):
        with open(presets_path, "r", encoding="utf-8") as f:
            names.update(k for k, v in toml.load(f).items() if isinstance(v, dict))
    custom_dir = os.path.join(configs_dir, "custom")
    if os.path.isdir(custom_dir):
        names.update(n[:-5] for n in os.listdir(custom_dir) if n.endswith(".toml"))
    return sorted(names)


def _iter_method_preset_layers(
    preset: str,
    configs_dir: str,
    methods_subdir: str,
    method: Optional[str],
    *,
    require_files: bool,
):
    """Yield ``(kind, path, tag, raw_dict)`` for the base → preset → method
    merge spine, in lowest→highest priority order.

    ``kind`` ∈ {``"base"``, ``"preset"``, ``"method"``}; ``raw_dict`` is
    un-flattened so each caller applies its own projection. ``preprocess.toml``
    is NOT yielded here — callers layer it themselves with different policies.

    Error policy follows ``require_files``: True (training path) raises
    ``FileNotFoundError`` on a missing base/method TOML; False (lightweight
    preprocess path) skips missing files and unknown presets silently.
    """
    base_path = os.path.join(configs_dir, "base.toml")
    method_path = _resolve_method_path(configs_dir, methods_subdir, method)
    if require_files:
        for p in (base_path, method_path):
            if p and not os.path.exists(p):
                raise FileNotFoundError(f"Config file not found: {p}")

    if os.path.exists(base_path):
        with open(base_path, "r", encoding="utf-8") as f:
            yield "base", base_path, _display_path(base_path), toml.load(f)

    if require_files:
        section, preset_path, tag = _resolve_preset(preset, configs_dir)
        yield "preset", preset_path, tag, section
    else:
        try:
            section, preset_path, tag = _resolve_preset(preset, configs_dir)
        except (KeyError, FileNotFoundError, ValueError):
            pass
        else:
            yield "preset", preset_path, tag, section

    if method_path and os.path.exists(method_path):
        with open(method_path, "r", encoding="utf-8") as f:
            yield "method", method_path, _display_path(method_path), toml.load(f)


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
    blocks_to_swap=0 regardless of the hardware preset). ``methods_subdir``
    defaults to ``"methods"``; pass ``"gui-methods"`` for the clean per-variant
    files used by the GUI / `make lora-gui` path.

    When ``return_provenance=True`` returns ``(merged, provenance)`` where
    ``provenance[key]`` is a short human-readable source tag (e.g.
    ``"configs/presets.toml[default]"``).
    """
    configs_dir = str(resolve_under_home(configs_dir))

    merged: dict = {}
    provenance: dict[str, str] = {}

    # preprocess.toml owns two keys the training side also reads. `target_res`
    # is inert at train time (training is self-describing from the cached
    # latents) and seeded only so it shows up in the snapshot/provenance for
    # the record; `mask_dir` is load-bearing — it is the lowest-priority
    # fallback under a subset's own `mask_dir`. Both are seeded here below
    # base/preset/method, which still override.
    preprocess_path = os.path.join(configs_dir, "preprocess.toml")
    if os.path.exists(preprocess_path):
        with open(preprocess_path, "r", encoding="utf-8") as f:
            pp_raw = toml.load(f)
        for _pp_key in ("target_res", "mask_dir"):
            if _pp_key in pp_raw:
                merged[_pp_key] = pp_raw[_pp_key]
                provenance[_pp_key] = _display_path(preprocess_path)

    # a stale target_res copy in base.toml must not clobber the seed above
    pp_has_target_res = "target_res" in merged

    for kind, path, tag, raw in _iter_method_preset_layers(
        preset, configs_dir, methods_subdir, method, require_files=True
    ):
        # wrap preset's flat scalar table as {preset: section} to mirror
        # base/method's top-level section tables for _flatten_toml
        to_flatten = {preset: raw} if kind == "preset" else raw
        for k, v in _flatten_toml(to_flatten, source=path, strict=strict).items():
            if k == "target_res" and kind == "base" and pp_has_target_res:
                continue
            merged[k] = v
            provenance[k] = tag

    if return_provenance:
        return merged, provenance
    return merged


def _git_sha() -> Optional[str]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            **no_window_kwargs(),
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

    # group by source so the dump reads top-down: base -> preset -> method -> CLI
    by_source: dict[str, list[tuple[str, Any]]] = {}
    for key, value, source in entries:
        by_source.setdefault(source, []).append((key, value))

    def _rank(src: str) -> int:
        if src == "configs/base.toml":
            return 0
        if src.startswith("configs/presets.toml") or src.startswith("configs/custom/"):
            return 1
        if src == "CLI":
            return 4
        return 2  # method file (any of the possible method dirs)

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

    # mirror into the run's TB log dir too, so it's a self-contained record;
    # failure here is logged but not fatal
    if getattr(args, "logging_dir", None):
        try:
            from library.runtime.accelerator import resolve_run_log_dir

            run_log_dir = resolve_run_log_dir(args)
        except ImportError as e:
            logger.warning(f"Could not resolve run log dir for snapshot mirror: {e}")
            run_log_dir = None
        if run_log_dir:
            mirror_path = os.path.join(run_log_dir, f"{output_name}{_SNAPSHOT_SUFFIX}")
            try:
                os.makedirs(run_log_dir, exist_ok=True)
                with open(mirror_path, "w", encoding="utf-8") as f:
                    f.write(rendered)
                logger.info(f"Config snapshot mirrored: {mirror_path}")
            except OSError as e:
                logger.warning(
                    f"Could not mirror config snapshot to {mirror_path}: {e}"
                )

    return path


def read_config_from_file(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    argv: Optional[Sequence[str]] = None,
):
    """Apply the base→preset→method (or single-file) config merge to ``args``.

    CLI overrides are layered on top by re-parsing through ``parser``. By
    default that re-parse reads the process ``sys.argv`` (the CLI path). Pass an
    explicit ``argv`` list to drive the override layer programmatically — e.g.
    from an embedder that builds its argv in code rather than from the shell.
    """
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
        args = parser.parse_args(args=argv, namespace=config_args)
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
    args = parser.parse_args(args=argv, namespace=config_args)
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
