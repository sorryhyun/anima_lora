"""Trainer-side model catalog — the Anima half of the download surface.

The curation half already exists: :mod:`anime_tools.downloads` is one
:class:`~anime_tools.downloads.Asset` per checkpoint (tagger, SAM3, PE-Spatial,
the tag KB, the OCR stack…) carrying repo, files, destination and an **offline**
installed probe. This module adds the rows only the trainer needs — the Anima
base weights, PE-Core, the CJK vocab pack — as the same ``Asset`` kind, so
``make download-*`` and the GUI read one list instead of two hand-kept copies.

Reusing the package's row type is deliberate: destinations already coincide
because ``anime_tools._env.curation_home()`` falls back to ``ANIMA_HOME``, which
``scripts/tasks/_common.py`` pins to this checkout. So ``models_dir()`` is
``<repo>/models`` and a catalog row writes exactly where a loader here looks.

Rows group into **packs** (:class:`~anime_tools.downloads.Pack`): the
trainer's ``anima`` / ``pe`` / ``cjk`` plus the package's, minus
:data:`HIDDEN_PACKS`. ``make download-model`` and the GUI's pack buttons
address packs, legacy make-target aliases or row ids through :func:`resolve`.

Same rule as the package's: **this is the single source of truth for weight
locations.** ``library/vision/encoders.py`` and ``library/anima/vocab_pack.py``
import their defaults from here; a path spelled in a loader is a Download button
that writes where the loader will not look.
"""

from __future__ import annotations

import os
from pathlib import Path

from library.env import anima_home

# Pin the package's home to the trainer's *before* anything reads a catalog:
# ``anime_tools._env.curation_home()`` falls back to ``ANIMA_HOME`` and then to
# the CWD, while ``anima_home()`` falls back to the checkout. Without this an
# ``import anima_lora`` from another directory would probe the curation rows at
# ``./models`` while the trainer rows resolve under the repo — the two halves
# of one catalog disagreeing about where a file is.
# (an explicit ANIME_TOOLS_HOME still wins — curation_home() checks it first)
if not os.environ.get("ANIMA_HOME"):
    os.environ["ANIMA_HOME"] = str(anima_home())

from anime_tools.downloads import (  # noqa: E402
    PACKS as _CURATION_PACKS,
    Asset,
    Pack,
    catalog as _package_catalog,
)


def models_dir() -> Path:
    """``<repo home>/models`` — where every row below lands."""
    return anima_home() / "models"


# Anima base: DiT + text encoder + VAE, published under one repo's
# ``split_files/`` tree. Three rows rather than one because the three land in
# three different directories (``Asset`` flattens into a single ``dest``), which
# also lets the GUI report them separately — a partial download is the common
# failure.
ANIMA_REPO = "circlestone-labs/Anima"
ANIMA_DIT_FILE = "anima-base-v1.0.safetensors"
ANIMA_TE_FILE = "qwen_3_06b_base.safetensors"
ANIMA_VAE_FILE = "qwen_image_vae.safetensors"

# PE-Core-L14-336 — global/CLIP-aligned vision features (CMMD validation,
# IP-Adapter). PE-Spatial, the REPA/grouping tower, is the package's row.
PE_CORE_REPO = "facebook/PE-Core-L14-336"
PE_CORE_FILENAME = "PE-Core-L14-336.pt"
PE_DIR = "pe"

# CJK vocab pack — a text-encoder asset, not an adapter. The repo's
# ``tokenizer_qwen3/`` is for pipelines without a Qwen3 tokenizer of their own;
# this one reuses the text encoder's, so it is not fetched.
VOCAB_PACK_REPO = "sorryhyun/anima-vocab-pack-cjk"
VOCAB_PACK_STEM = "anima_cjk_vocab_pack"
VOCAB_PACK_DIR = "vocab_packs"


def default_pe_core_path() -> Path:
    """``<models_dir>/pe/PE-Core-L14-336.pt`` — what ``library/vision`` loads."""
    return models_dir() / PE_DIR / PE_CORE_FILENAME


def default_vocab_pack_dir() -> Path:
    """``<models_dir>/vocab_packs`` — the dir ``vocab_pack`` in
    ``configs/base.toml`` points a stem into."""
    return models_dir() / VOCAB_PACK_DIR


def default_vocab_pack_prefix() -> Path:
    """``<models_dir>/vocab_packs/anima_cjk_vocab_pack`` — the path prefix
    (no suffix) ``configs/base.toml`` ships as ``vocab_pack`` and the loader's
    auto-fetch recognises as the shipped pack."""
    return default_vocab_pack_dir() / VOCAB_PACK_STEM


# The trainer's packs — what a "Download pack" button on the Anima tab is a
# button *for*. The package's own (tagger / tags / masking / ocr / grouping)
# follow them in ``PACKS``; a row's ``pack`` is one of these ids.
TRAINER_PACKS: tuple[Pack, ...] = (
    Pack(
        "anima",
        "Anima base",
        "The DiT, the Qwen3-0.6B text encoder and the Qwen-Image VAE — every "
        "training and inference run needs all three.",
    ),
    Pack(
        "pe",
        "PE-Core",
        "PE-Core-L14-336: CMMD validation and IP-Adapter conditioning. "
        "PE-Spatial, the grouping tower, is the Grouping pack.",
    ),
    Pack(
        "cjk",
        "CJK vocab pack",
        "Extra text-encoder rows for Japanese / Korean / Chinese caption and "
        "prompt spans. On by default since v2; English text is bit-exact "
        "either way.",
    ),
)

# Package packs the trainer does not offer. ``text_mask`` (the MIT UNet++ text
# segmenter + its ComicTextDetector gate) went with v2's in-image text masking:
# the rows stay in the package catalog for its own users, but they are not
# listed, resolved or downloaded from here.
HIDDEN_PACKS: tuple[str, ...] = ("text_mask",)

PACKS: tuple[Pack, ...] = (
    *TRAINER_PACKS,
    *(p for p in _CURATION_PACKS if p.id not in HIDDEN_PACKS),
)
"""Display order: the trainer's packs, then the package's that survive
:data:`HIDDEN_PACKS`. ``make download-model`` and the GUI accept a pack id
wherever they accept a row id."""

PACK_BY_ID: dict[str, Pack] = {p.id: p for p in PACKS}


def _anima_row(kind: str, filename: str, title: str, used_by: str) -> Asset:
    """One of the three Anima base weights.

    The repo path rides in ``files`` rather than ``subfolder`` so the flatten in
    ``Asset.fetch`` lands it directly on the config's default path.
    """
    return Asset(
        id=f"anima_{kind}",
        pack="anima",
        title=title,
        repo=ANIMA_REPO,
        files=(f"split_files/{kind_dir(kind)}/{filename}",),
        dest=models_dir() / kind_dir(kind),
        used_by=used_by,
        notes="Path is a `configs/base.toml` default; the three together are "
        "~5 GB and every training and inference run needs all of them.",
    )


def kind_dir(kind: str) -> str:
    """``models/`` subdirectory for an Anima base component."""
    return {
        "dit": "diffusion_models",
        "te": "text_encoders",
        "vae": "vae",
    }[kind]


def catalog() -> tuple[Asset, ...]:
    """The trainer's own rows, rebuilt per call so they follow the home."""
    return (
        _anima_row(
            "dit",
            ANIMA_DIT_FILE,
            "Anima base DiT",
            "every training and inference run",
        ),
        _anima_row(
            "te",
            ANIMA_TE_FILE,
            "Qwen3-0.6B text encoder",
            "text-embedding caching · every run",
        ),
        _anima_row(
            "vae",
            ANIMA_VAE_FILE,
            "Qwen-Image VAE",
            "latent caching · decoding samples",
        ),
        Asset(
            id="pe_core",
            pack="pe",
            title="PE-Core-L14-336",
            repo=PE_CORE_REPO,
            files=(PE_CORE_FILENAME,),
            dest=models_dir() / PE_DIR,
            used_by="CMMD validation metric · IP-Adapter conditioning",
            notes="Global/CLIP-aligned features. The REPA and grouping tower is "
            "PE-Spatial, a separate row.",
        ),
        Asset(
            id="vocab_pack",
            pack="cjk",
            title="CJK vocab pack",
            repo=VOCAB_PACK_REPO,
            files=(f"{VOCAB_PACK_STEM}.safetensors", f"{VOCAB_PACK_STEM}.json"),
            dest=default_vocab_pack_dir(),
            used_by="Japanese / Korean / Chinese caption and prompt spans "
            "(on by default since v2)",
            notes="~285 MB. `vocab_pack` in configs/base.toml points here by "
            'default (`""` turns it off); the loader fetches this row itself '
            "when the default is missing. Changing the pack needs "
            "`make preprocess-te ARGS=--overwrite` for CJK captions; EN-only "
            "captions are bit-exact either way.",
        ),
    )


def curation_catalog() -> tuple[Asset, ...]:
    """The ``anime_tools`` rows — tagger, SAM3, OCR, the tag KB — minus the
    packs in :data:`HIDDEN_PACKS`."""
    return tuple(a for a in _package_catalog() if a.pack not in HIDDEN_PACKS)


def full_catalog() -> tuple[Asset, ...]:
    """Both halves, trainer rows first. Ids are unique across the two."""
    return (*catalog(), *curation_catalog())


def by_id() -> dict[str, Asset]:
    return {a.id: a for a in full_catalog()}


def by_pack(rows: tuple[Asset, ...] | None = None) -> dict[str, tuple[Asset, ...]]:
    """Rows bucketed by pack — :data:`PACKS` order, catalog order inside each,
    packs with no rows left out. ``rows`` defaults to :func:`full_catalog`."""
    rows = full_catalog() if rows is None else rows
    out: dict[str, tuple[Asset, ...]] = {}
    for pack in PACKS:
        picked = tuple(a for a in rows if a.pack == pack.id)
        if picked:
            out[pack.id] = picked
    return out


# Legacy ``make download-<target>`` names that are not pack ids, or that mean
# something narrower than the pack of the same name. Kept because they are in
# `make help`, the guidebook and its three translations:
#
# * ``pe`` — the make target has always fetched PE-Core *and* PE-Spatial,
#   while the ``pe`` pack is PE-Core alone (PE-Spatial belongs to ``grouping``).
# * ``tagger`` — ``make download-tagger`` (and the ``make preprocess``
#   auto-fetch) means the small checkpoint row only, never the gated backbone;
#   the ``tagger`` *pack* (checkpoint + backbone + ONNX trace) is reached by
#   spelling its rows, which is what the GUI's pack button does.
GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "anima": ("anima_dit", "anima_te", "anima_vae"),
    "pe": ("pe_core", "pe_spatial"),
    "pe-spatial": ("pe_spatial",),
    "sam3": ("sam3",),
    "tagger": ("tagger",),
    "tagger-model": ("tagger", "tagger_backbone"),
    "danbooru-tags": ("danbooru_tags", "danbooru_tags_en"),
    "vocab-pack": ("vocab_pack",),
}

# Every multi-row token ``resolve`` accepts: the packs (pack id → its rows),
# overlaid with the aliases above where the names collide. Ids only — the
# rows themselves are rebuilt per call so they follow the home.
GROUPS: dict[str, tuple[str, ...]] = {
    **{pid: tuple(a.id for a in rows) for pid, rows in by_pack().items()},
    **GROUP_ALIASES,
}

# What a first-run `make download-models` fetches — the mandatory set: the
# Anima base, both PE towers, the tagger checkpoint, the tag KB, and (since v2)
# the CJK vocab pack, because ``configs/base.toml`` now enables it. Not
# "everything missing": SAM3 is gated and masking is opt-in (``make
# download-sam3``), and the OCR stack is opt-in (``make download-model ocr``).
DEFAULT_SET: tuple[str, ...] = (
    "anima_dit",
    "anima_te",
    "anima_vae",
    "pe_core",
    "pe_spatial",
    "vocab_pack",
    "tagger",
    "danbooru_tags",
    "danbooru_tags_en",
)


def resolve(names: list[str] | tuple[str, ...]) -> list[Asset]:
    """Tokens → rows, in catalog order, deduped.

    A token is looked up as a legacy alias (:data:`GROUP_ALIASES`) first, then
    as a pack id (:data:`PACKS`), then as a row id — so ``pe`` keeps its
    two-tower make-target meaning and ``ocr`` expands to the pack. Raises
    ``KeyError`` naming the unknown token, so a typo in a make target fails
    loudly rather than downloading nothing.
    """
    assets = by_id()
    packed = by_pack()
    picked: list[str] = []
    for name in names:
        if name in GROUP_ALIASES:
            ids = GROUP_ALIASES[name]
        elif name in packed:
            ids = tuple(a.id for a in packed[name])
        elif name in assets:
            ids = (name,)
        else:
            raise KeyError(
                f"unknown model id {name!r} — known ids: {', '.join(assets)}; "
                f"packs: {', '.join(packed)}; aliases: {', '.join(GROUP_ALIASES)}"
            )
        unknown = [i for i in ids if i not in assets]
        if unknown:
            raise KeyError(
                f"{name!r} names rows the catalog does not have: {', '.join(unknown)}"
            )
        picked.extend(i for i in ids if i not in picked)
    order = list(assets)
    return [assets[i] for i in sorted(picked, key=order.index)]


def _prune_empty(root: Path | None) -> None:
    """Drop the empty ``split_files/…`` scaffolding ``hf`` leaves behind after
    :meth:`Asset.fetch` flattens a repo-path row into ``dest``."""
    if root is None or not root.is_dir():
        return
    for path in sorted(root.rglob("*"), key=lambda p: -len(p.parts)):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def fetch(asset: Asset, log=print, force: bool = False) -> bool:
    """Download one row unless it is already installed. True if it fetched.

    Skip-if-present is the idempotency contract (GH #21): several rows move
    files out of ``hf``'s ``--local-dir`` layout, so the hub would re-pull
    gigabytes on a re-run that only meant to verify.
    """
    if asset.installed and not force:
        log(f"  ✓ {asset.title} already present (pass --force to re-download)")
        return False
    log(f"\n{asset.title}  [{asset.repo}] → {asset.location}")
    asset.fetch(log)
    _prune_empty(asset.dest)
    return True


def fetch_all(assets: list[Asset], log=print, force: bool = False) -> list[str]:
    """Fetch every row, continuing past failures. Returns the titles that failed.

    Continue-on-failure matters because one gated repo without granted access
    (SAM3, the tagger backbone) must not abort the Anima download beside it.
    """
    failed: list[str] = []
    for asset in assets:
        try:
            fetch(asset, log=log, force=force)
        except Exception as exc:  # noqa: BLE001 — reported per row at the end
            failed.append(asset.title)
            log(f"  ✗ {asset.title} failed: {type(exc).__name__}: {exc}")
    return failed
