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

from anime_tools.downloads import Asset, catalog as _curation_catalog  # noqa: E402


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


def _anima_row(kind: str, filename: str, title: str, used_by: str) -> Asset:
    """One of the three Anima base weights.

    The repo path rides in ``files`` rather than ``subfolder`` so the flatten in
    ``Asset.fetch`` lands it directly on the config's default path.
    """
    return Asset(
        id=f"anima_{kind}",
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
            title="CJK vocab pack",
            repo=VOCAB_PACK_REPO,
            files=(f"{VOCAB_PACK_STEM}.safetensors", f"{VOCAB_PACK_STEM}.json"),
            dest=default_vocab_pack_dir(),
            used_by="Japanese / Korean / Chinese caption spans (opt-in)",
            notes="~285 MB. Inert until `vocab_pack = "
            f'"models/{VOCAB_PACK_DIR}/{VOCAB_PACK_STEM}"` is set in '
            "configs/base.toml; changing it needs "
            "`make preprocess-te ARGS=--overwrite` for CJK captions.",
        ),
    )


def curation_catalog() -> tuple[Asset, ...]:
    """The ``anime_tools`` rows, verbatim — tagger, SAM3, OCR, the tag KB."""
    return _curation_catalog()


def full_catalog() -> tuple[Asset, ...]:
    """Both halves, trainer rows first. Ids are unique across the two."""
    return (*catalog(), *curation_catalog())


def by_id() -> dict[str, Asset]:
    return {a.id: a for a in full_catalog()}


# ``make download-<target>`` → the rows it fetches. Every legacy target name is
# kept: they are in `make help`, in the guidebook and in three translations.
GROUPS: dict[str, tuple[str, ...]] = {
    "anima": ("anima_dit", "anima_te", "anima_vae"),
    "pe": ("pe_core", "pe_spatial"),
    "pe-spatial": ("pe_spatial",),
    "sam3": ("sam3",),
    "mit": ("mit_text", "ctd_onnx"),
    "tagger": ("tagger",),
    "tagger-model": ("tagger", "tagger_backbone"),
    "danbooru-tags": ("danbooru_tags", "danbooru_tags_en"),
    "vocab-pack": ("vocab_pack",),
}

# What a first-run `make download-models` fetches. Deliberately not "everything
# missing": SAM3 is gated (masking is opt-in), the vocab pack is opt-in, and the
# OCR stack is a curation concern the trainer does not need to install.
DEFAULT_SET: tuple[str, ...] = (
    "anima_dit",
    "anima_te",
    "anima_vae",
    "pe_core",
    "pe_spatial",
    "tagger",
    "danbooru_tags",
    "danbooru_tags_en",
)


def resolve(names: list[str] | tuple[str, ...]) -> list[Asset]:
    """Asset ids and/or ``GROUPS`` keys → rows, in catalog order, deduped.

    Raises ``KeyError`` naming the unknown token, so a typo in a make target
    fails loudly rather than downloading nothing.
    """
    assets = by_id()
    picked: list[str] = []
    for name in names:
        ids = GROUPS.get(name, (name,))
        unknown = [i for i in ids if i not in assets]
        if unknown:
            raise KeyError(
                f"unknown model id {', '.join(unknown)!r} — known ids: "
                f"{', '.join(assets)}; groups: {', '.join(GROUPS)}"
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
