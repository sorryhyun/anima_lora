"""CLI/TOML/default resolution helpers shared by the bespoke distill schemas.

The bespoke distillation loops (``scripts/distill_turbo`` / ``project/finished/mod_guidance``)
read sectioned TOML files raw (not through ``load_method_preset``), resolve each
knob CLI → TOML → default, and freeze the result into a dataclass.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Optional, Sequence

from library.config.io import toml_get

# Python 3.11+; fall back to ``tomli`` if needed.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


# argparse defaults that mean "unset → use the TOML/default value". ``None`` is
# the natural store_const/optional sentinel; ``-1`` / ``-1.0`` are the int/float
# defaults the distill argparsers use so a real ``0`` still overrides.
_DEFAULT_SENTINELS: tuple = (None, -1, -1.0)


def _is_sentinel(value: Any, sentinels: Sequence[Any]) -> bool:
    """True if ``value`` equals one of the sentinels (identity OR ``==``).

    Booleans never match the numeric ``-1`` sentinels (``True == -1`` is False),
    so ``store_true``/``store_false`` flags pass through untouched.
    """
    for s in sentinels:
        if value is s:
            return True
        if value == s:
            return True
    return False


def pick(
    cli_value: Any,
    cfg: dict,
    key: str,
    default: Any,
    *,
    sentinels: Sequence[Any] = _DEFAULT_SENTINELS,
) -> Any:
    """CLI override (non-sentinel) wins, else dotted TOML ``key``, else ``default``.

    ``cfg`` is a raw (nested) TOML dict; ``key`` is a dotted ``a.b.c`` path read
    via :func:`library.config.io.toml_get`.
    """
    if not _is_sentinel(cli_value, sentinels):
        return cli_value
    return toml_get(cfg, key, default)


def load_toml(path: str) -> dict:
    """Read a raw TOML file (the sectioned bespoke-schema distill configs)."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def dataclass_snapshot_toml(
    config: Any,
    *,
    title: str,
    source_config: Optional[str] = None,
) -> str:
    """Render a resolved frozen config as a provenance TOML snapshot.

    Dumps every resolved field (CLI overrides folded in), the bespoke-loop
    analogue of ``train.py``'s ``<output_name>.snapshot.toml``. ``None`` fields
    are written as ``# k = (unset)`` comments (tomlkit has no null).
    """
    import tomlkit

    doc = tomlkit.document()
    doc.add(tomlkit.comment(title))
    if source_config:
        doc.add(tomlkit.comment(f"source config: {source_config}"))
    doc.add(tomlkit.nl())
    for k, v in dataclasses.asdict(config).items():
        if v is None:
            doc.add(tomlkit.comment(f"{k} = (unset)"))
        else:
            doc[k] = v
    return tomlkit.dumps(doc)


def dataclass_tb_text(
    config: Any,
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> str:
    """Render a dataclass config as the ``key: value`` block TensorBoard wants.

    ``include`` selects an explicit ordered subset (turbo's hand-picked summary);
    otherwise every field is emitted in declaration order, minus ``exclude``.
    """
    d = dataclasses.asdict(config)
    if include is not None:
        items = [(k, d[k]) for k in include]
    else:
        skip = set(exclude or ())
        items = [(k, v) for k, v in d.items() if k not in skip]
    return "  \n".join(f"{k}: {v}" for k, v in items)
