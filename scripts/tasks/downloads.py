"""Model download entry-points — thin wrappers over the model catalog.

Every row (repo, files, destination, installed-probe) lives in
``library/downloads.py`` for the Anima weights and in ``anime_tools.downloads``
for the curation ones; this module only maps ``make download-<target>`` onto
ids. Adding a weight means adding an ``Asset`` there, not a command here — the
GUI's Models panels read the same two catalogs, so a row can't drift out of
sync with its button.

Idempotency contract (see GH #21): every target skips when its destination
files already exist, so a re-run *verifies* rather than re-fetching gigabytes.
This matters because several rows move files out of ``hf``'s ``--local-dir``
layout after download — once moved, the hub no longer sees them at the path it
checks and would otherwise re-pull the whole repo. Pass ``--force`` (e.g.
``make download-anima ARGS=--force``) to re-fetch regardless.
``download-models`` continues past a failed component (a gated SAM3 without
granted access shouldn't abort the Anima download) and reports the failures at
the end.

The one target that is not a catalog row is ``download-anima-variant``: it is a
picker over alternate base DiTs, not a checklist of things a run needs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from library import downloads as DL

from ._common import ROOT, run


# Re-exported for the GUI and the tests: the vocab pack's enable-string.
VOCAB_PACK_REL = f"models/{DL.VOCAB_PACK_DIR}"
VOCAB_PACK_STEM = DL.VOCAB_PACK_STEM


def _force(extra) -> bool:
    return "--force" in (extra or [])


def _skip(name: str, paths: list[Path], extra) -> bool:
    """Skip-if-present for the variant picker, which has no catalog row."""
    if _force(extra):
        return False
    if all(p.exists() for p in paths):
        print(f"  ✓ {name} already present (pass --force to re-download)")
        return True
    return False


def _fetch(*names: str, extra=None) -> None:
    """Fetch the rows named by asset ids and/or ``DL.GROUPS`` keys.

    Continue-on-failure, then a single ``SystemExit`` naming what did not land —
    ``run`` already exits on a non-zero child, so callers that aggregate
    (``download-models``) catch ``SystemExit`` per component.
    """
    try:
        assets = DL.resolve(list(names))
    except KeyError as e:
        raise SystemExit(str(e)) from e
    failed = DL.fetch_all(assets, force=_force(extra))
    if failed:
        raise SystemExit(
            "failed to download: "
            + ", ".join(failed)
            + "\n  - not authenticated? run `hf auth login` and re-run"
            + "\n  - a gated repo needs its terms accepted on the model page"
        )


def cmd_download_list(_extra):
    """Print every catalog row with its install state — the offline probe."""
    for label, rows in (
        ("anima", DL.catalog()),
        ("curation (anime_tools)", DL.curation_catalog()),
    ):
        print(f"\n{label}")
        for a in rows:
            mark = "✓ installed" if a.installed else "✗ MISSING  "
            print(f"  {mark}  {a.id:<17} {a.repo:<46} → {a.location}")
    print(f"\nGroups: {', '.join(DL.GROUPS)}")
    print("Fetch by id or group: make download-model ARGS='sam3 pe'")


def cmd_download_model(extra):
    """``make download-model ARGS='<id|group>...'`` — the generic front door.

    This is what the GUI's per-row Download buttons run, for both panels.
    """
    names = [a for a in (extra or []) if not a.startswith("-")]
    if not names:
        cmd_download_list(extra)
        return
    _fetch(*names, extra=extra)


def cmd_download_sam3(extra):
    _fetch("sam3", extra=extra)


def cmd_download_pe(extra):
    """PE-Core plus PE-Spatial — the REPA/grouping tower ships beside it."""
    _fetch("pe", extra=extra)


def cmd_download_pe_spatial(extra):
    _fetch("pe_spatial", extra=extra)


def cmd_download_tagger(extra):
    """The Anima Tagger checkpoint — vocab, rules, thresholds, sidecar head.

    Small (the weights are the gated backbone, which ``download-tagger-model``
    adds). ``preprocess`` auto-calls this when the caption-index vocab is
    missing.
    """
    _fetch("tagger", extra=extra)


def cmd_download_tagger_model(extra):
    """The whole tagger — the checkpoint above plus the gated backbone.

    The backbone (``animetimm/caformer_b36.dbv4-full`` by default) is GPL-3.0
    and gated, so it is never vendored: the user's own token downloads it, which
    is also the record of them accepting the repo terms. Auto-approve gate —
    ``hf auth login`` (or the GUI's token field), then click through once on the
    repo page. It lands in the **hub cache**, not under ``models/``, because
    that is where the loader looks.
    """
    _fetch("tagger-model", extra=extra)


def _download_danbooru_base(extra):
    """Base CSV only — ``preprocess`` calls this when the tag KB is missing."""
    _fetch("danbooru_tags", extra=extra)


def cmd_download_danbooru_tags(extra):
    """The Danbooru tag table for caption correction, both languages.

    The Korean-description base CSV from Localsmile, then the English sibling
    built by joining tag names against the ``isek-ai/danbooru-wiki-2024`` mirror
    so the GUI tag-explanation tooltip works for non-Korean UIs.
    """
    _fetch("danbooru-tags", extra=extra)


def cmd_download_vocab_pack(extra):
    """The shipped CJK vocab pack (.safetensors + .json pair)."""
    _fetch("vocab-pack", extra=extra)
    print(
        f'  → enable with vocab_pack = "{VOCAB_PACK_REL}/{VOCAB_PACK_STEM}" '
        "in configs/base.toml"
    )


def cmd_download_mit(extra):
    """MIT text-segmentation net + the ComicTextDetector gate it reads."""
    _fetch("mit", extra=extra)


def cmd_download_anima(extra):
    """Anima base — DiT + text encoder + VAE (~5 GB)."""
    _fetch("anima", extra=extra)


# Alternate base DiTs, as name -> (repo_id, path within the repo).
#
# The official circlestone-labs variants are the same 28-block DiT as
# anima-base-v1.0 and differ only in weights and in the state-dict prefix
# ("model.diffusion_model." vs base's "net."), which the loader strips either
# way (library/anima/weights.py::_DIT_PREFIXES).
#
# Anima-2.9B is a community depth-expansion of the same architecture: 40 blocks
# instead of 28, same width, same Qwen3-0.6B text encoder and Qwen-Image VAE.
# The loader counts depth off the checkpoint (``probe_dit_arch``), so it needs
# no flag. Its LoRAs are NOT interchangeable with 28-block ones.
ANIMA_VARIANTS = {
    "anima-aesthetic-v1.0": (
        "circlestone-labs/Anima",
        "split_files/diffusion_models/anima-aesthetic-v1.0.safetensors",
    ),
    "anima-aesthetic-v1.0b": (
        "circlestone-labs/Anima",
        "split_files/diffusion_models/anima-aesthetic-v1.0b.safetensors",
    ),
    "anima-aesthetic-v1.1": (
        "circlestone-labs/Anima",
        "split_files/diffusion_models/anima-aesthetic-v1.1.safetensors",
    ),
    "anima-turbo-v1.0": (
        "circlestone-labs/Anima",
        "split_files/diffusion_models/anima-turbo-v1.0.safetensors",
    ),
    "anima-preview3-base": (
        "circlestone-labs/Anima",
        "split_files/diffusion_models/anima-preview3-base.safetensors",
    ),
    "Anima-2.9B-preview-v1": (
        "Gazingstars123/Anima-2.9B",
        "Anima-2.9B-preview-v1.safetensors",
    ),
}


def cmd_download_anima_variant(_extra):
    """Download an alternate Anima base DiT (aesthetic / turbo / preview)."""
    names = [a for a in (_extra or []) if not a.startswith("-")]
    if not names:
        print("Usage: make download-anima-variant ARGS=<name> [<name>...]")
        print("Available: " + ", ".join(ANIMA_VARIANTS))
        return
    unknown = [n for n in names if n not in ANIMA_VARIANTS]
    if unknown:
        raise SystemExit(
            f"Unknown Anima variant(s): {', '.join(unknown)}\n"
            f"Available: {', '.join(ANIMA_VARIANTS)}"
        )
    models = ROOT / "models"
    dst = models / "diffusion_models"
    finals = [dst / f"{n}.safetensors" for n in names]
    if _skip(f"Anima variant(s) {', '.join(names)}", finals, _extra):
        return
    dst.mkdir(parents=True, exist_ok=True)
    # Group by repo so one `hf download` covers every variant from the same repo.
    by_repo: dict[str, list[str]] = {}
    for n in names:
        repo, path = ANIMA_VARIANTS[n]
        by_repo.setdefault(repo, []).append(path)
    for repo, paths in by_repo.items():
        run(["hf", "download", repo, *paths, "--local-dir", "models"])
    # Repos nest the file differently (circlestone under split_files/, others at
    # the root), so normalize by moving whatever landed under models/ into
    # diffusion_models/.
    split = models / "split_files"
    src = split / "diffusion_models"
    if src.exists():
        for f in src.iterdir():
            shutil.move(str(f), str(dst / f.name))
    if split.exists():
        shutil.rmtree(split)
    for n in names:
        stray = models / Path(ANIMA_VARIANTS[n][1]).name
        if stray.exists() and stray.parent != dst:
            shutil.move(str(stray), str(dst / stray.name))
    print(
        "\nTrain against one with:\n"
        f"  make lora ARGS='--pretrained_model_name_or_path "
        f"models/diffusion_models/{names[0]}.safetensors'"
    )


def cmd_download_models(extra):
    """The first-run set: Anima base, PE, the tagger checkpoint, the tag KB.

    Deliberately not "everything in the catalog". SAM3 is gated and masking is
    opt-in since v2, the vocab pack is opt-in, and the OCR stack is a curation
    concern — each has its own target. See ``DEFAULT_SET`` in
    ``library/downloads.py``.
    """
    failed = DL.fetch_all(DL.resolve(DL.DEFAULT_SET), force=_force(extra))
    if not failed:
        return
    print()
    print("The following downloads did not complete:")
    for title in failed:
        print(f"  - {title}")
    print()
    print("Common causes:")
    print("  - not authenticated: run `hf auth login` and re-run")
    print("  - a gated repo needs its terms accepted on its model page")
    print("Successful components are cached; re-running only retries the failures.")
    raise SystemExit(1)
