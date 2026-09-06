"""Model download entry-points (Anima base, SAM3, MIT, PE-Core, Tagger vocab).

All targets shell out to ``hf download`` (rather than the SDK) so the user's
``hf auth login`` cache is honored.

Idempotency contract (see GH #21): every target skips when its final
destination files already exist, so a re-run *verifies* rather than re-fetching
gigabytes. This matters because several targets ``shutil.move`` files out of
``hf``'s ``--local-dir`` layout after download — once moved, ``hf download``
no longer sees them at the path it checks and would otherwise re-pull the whole
repo. Pass ``--force`` (e.g. ``make download-anima ARGS=--force``) to re-fetch
regardless. ``download-models`` continues past a failed component (a gated SAM3
without granted access shouldn't abort the Anima download) and reports the
failures at the end.
"""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from pathlib import Path

from anime_tools.contract import DBV4_REQUIRED_FILES
from anime_tools.tagger.dbv4_meta import (
    DBV4_BACKBONE_FILES,
    DEFAULT_DBV4_REPO,
    backbone_cached,
    backbone_repo_for,
)

from ._common import PY, ROOT, run


DANBOORU_TAGS_PATH = ROOT / "models" / "danbooru_tags_classified.csv"
DANBOORU_TAGS_EN_PATH = ROOT / "models" / "danbooru_tags_classified.en.csv"
DANBOORU_TAGS_URLS = (
    "https://raw.githubusercontent.com/Localsmile/danbooru_KR_wiki_tag_search/main/danbooru_tags_classified.csv",
)

# Anima Tagger, which is a thin head over an off-the-shelf danbooru tagger:
# our part (vocab / rules / thresholds / groups / sidecar head) ships from
# our own repo, the backbone comes from its gated upstream one.
TAGGER_CKPT_REPO = "sorryhyun/anima-tagger"
TAGGER_CKPT_SUBFOLDER = "dbv4"
TAGGER_CKPT_REL = "models/captioners/anima-tagger-dbv4"
# Required-file set of a dbv4 checkpoint dir — the package's own answer, from
# its stdlib-only contract module (the task runner stays import-light).
TAGGER_CKPT_REQUIRED = DBV4_REQUIRED_FILES
# Backbone facts come from the torch-free anime_tools.tagger.dbv4_meta so the
# loader, this task and the GUI can never disagree on repo / file set.
TAGGER_BACKBONE_REPO = DEFAULT_DBV4_REPO
TAGGER_BACKBONE_FILES = DBV4_BACKBONE_FILES


def _present(paths: list[Path]) -> bool:
    """True when every expected destination path already exists."""
    return all(p.exists() for p in paths)


def _skip(name: str, paths: list[Path], extra) -> bool:
    """Return True (caller should skip) when files exist and ``--force`` absent."""
    if "--force" in (extra or []):
        return False
    if _present(paths):
        print(f"  ✓ {name} already present (pass --force to re-download)")
        return True
    return False


def cmd_download_sam3(_extra):
    dst = ROOT / "models" / "sam3"
    # SAM3 is a gated repo; the full snapshot lands a config.json + weights.
    if _skip("SAM3", [dst / "config.json"], _extra):
        return
    dst.mkdir(parents=True, exist_ok=True)
    run(["hf", "download", "facebook/sam3", "--local-dir", "models/sam3"])


def cmd_download_pe(_extra):
    # Only the .pt is needed; vision tower is vendored at anime_tools/vision/pe.py (library.models.pe).
    dst = ROOT / "models" / "pe"
    # Skip only the PE-Core fetch — still fall through to PE-Spatial below, which
    # may be missing even when PE-Core is on disk.
    if not _skip("PE-Core", [dst / "PE-Core-L14-336.pt"], _extra):
        dst.mkdir(parents=True, exist_ok=True)
        run(
            [
                "hf",
                "download",
                "facebook/PE-Core-L14-336",
                "PE-Core-L14-336.pt",
                "--local-dir",
                "models/pe",
            ]
        )
    # PE-Spatial is the default REPA alignment encoder — fetch it alongside PE-Core.
    cmd_download_pe_spatial(_extra)


def cmd_download_pe_spatial(_extra):
    # Auxiliary encoder for the Anima Tagger's dual-encoder config; only the .pt.
    dst = ROOT / "models" / "pe"
    if _skip("PE-Spatial", [dst / "PE-Spatial-B16-512.pt"], _extra):
        return
    dst.mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "facebook/PE-Spatial-B16-512",
            "PE-Spatial-B16-512.pt",
            "--local-dir",
            "models/pe",
        ]
    )


def _flatten_subfolder(dst: Path, sub: str) -> None:
    """Move ``dst/<sub>/*`` up into ``dst`` — ``hf`` mirrors the repo layout, the
    loader wants a flat checkpoint dir.

    ``Path.replace`` rather than ``shutil.move`` so a ``--force`` re-download
    overwrites the existing file instead of raising on Windows (``os.rename``
    onto an existing path fails there; POSIX would have silently overwritten).
    """
    nested = dst / sub
    if not nested.is_dir():
        return
    for f in nested.iterdir():
        f.replace(dst / f.name)
    shutil.rmtree(nested, ignore_errors=True)


def cmd_download_tagger(_extra):
    # Just the Tagger ``vocab.json`` (~0.7 MB) that caption-index/preprocess need.
    # The full model is not fetched here, so this won't clobber a local model.safetensors.
    # Tracks the live checkpoint (``TAGGER_HF_SUBFOLDER`` / ``DEFAULT_TAGGER_DIR``
    # in anime_tools.tagger.tagger) so the vocab matches the model that
    # actually runs. For the whole tagger (head + backbone) see
    # ``cmd_download_tagger_model``.
    sub = TAGGER_CKPT_SUBFOLDER
    dst = ROOT / TAGGER_CKPT_REL
    if _skip("Anima Tagger vocab", [dst / "vocab.json"], _extra):
        return
    dst.mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            TAGGER_CKPT_REPO,
            f"{sub}/vocab.json",
            "--local-dir",
            TAGGER_CKPT_REL,
        ]
    )
    _flatten_subfolder(dst, sub)


def _tagger_backbone_repo() -> str:
    """Backbone repo of the installed checkpoint (``config.json`` → default)."""
    return backbone_repo_for(ROOT / TAGGER_CKPT_REL)


def cmd_download_tagger_model(_extra):
    """Download the *whole* Anima Tagger — our head plus the gated backbone.

    Two halves, because the tagger is a thin head over an off-the-shelf model:

    * ``sorryhyun/anima-tagger`` ``dbv4/`` → ``models/captioners/anima-tagger-dbv4/``
      (vocab, rules, thresholds, groups, and the sidecar head that supplies what
      the backbone cannot say — copyright / OC characters / people count);
    * the backbone itself (``animetimm/caformer_b36.dbv4-full`` by default),
      which is **GPL-3.0 and gated**. It is never vendored: the user's own
      token downloads it, which is also the record of them accepting the repo
      terms. Auto-approve gate — ``hf auth login`` (or the GUI's token field),
      then click through once on the repo page.

    The backbone lands in the **HuggingFace hub cache**, not under ``models/``,
    because that is where the loader looks (``Dbv4Backend._load_model`` →
    ``hf_hub_download`` with no ``local_dir``). Idempotent like every other
    target; ``--force`` re-fetches both halves.
    """
    force = "--force" in (_extra or [])
    sub = TAGGER_CKPT_SUBFOLDER
    dst = ROOT / TAGGER_CKPT_REL
    if not _skip(
        "Anima Tagger checkpoint",
        [dst / f for f in TAGGER_CKPT_REQUIRED],
        _extra,
    ):
        dst.mkdir(parents=True, exist_ok=True)
        run(
            [
                "hf",
                "download",
                TAGGER_CKPT_REPO,
                "--include",
                f"{sub}/*",
                "--local-dir",
                TAGGER_CKPT_REL,
            ]
        )
        _flatten_subfolder(dst, sub)

    repo = _tagger_backbone_repo()
    if not force and backbone_cached(repo):
        print(
            f"  ✓ tagger backbone {repo} already cached (pass --force to re-download)"
        )
        return
    print(f"  tagger backbone: {repo} (gated, GPL-3.0 — downloaded under your token)")
    run(["hf", "download", repo, *TAGGER_BACKBONE_FILES])


def _download_danbooru_base(_extra):
    """Fetch the Korean-description base CSV from Localsmile (idempotent)."""
    if _skip("Danbooru classified tags", [DANBOORU_TAGS_PATH], _extra):
        return
    DANBOORU_TAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DANBOORU_TAGS_PATH.with_suffix(".csv.tmp")
    last_error = ""
    for url in DANBOORU_TAGS_URLS:
        print(f"  download {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "anima-lora"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                tmp.write_bytes(resp.read())
            if tmp.stat().st_size <= 0:
                raise OSError("downloaded file is empty")
            tmp.replace(DANBOORU_TAGS_PATH)
            print(f"  ✓ wrote {DANBOORU_TAGS_PATH}")
            return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
            if tmp.exists():
                tmp.unlink()
            print(f"  ✗ failed: {last_error}")
    raise SystemExit(
        "failed to download danbooru_tags_classified.csv from "
        "Localsmile/danbooru_KR_wiki_tag_search"
    )


def cmd_download_danbooru_tags(_extra):
    """Fetch the Danbooru tag table for caption correction — both languages.

    Downloads the Korean-description base CSV (``danbooru_tags_classified.csv``)
    from Localsmile, then builds the English sibling
    (``danbooru_tags_classified.en.csv``) by joining tag names against the
    ``isek-ai/danbooru-wiki-2024`` wiki mirror so the GUI tag-explanation tooltip
    works for non-Korean UIs. Both steps are idempotent (``--force`` re-fetches).
    """
    _download_danbooru_base(_extra)
    if _skip("Danbooru English tags", [DANBOORU_TAGS_EN_PATH], _extra):
        return
    # Pass through only the builder's own flags (e.g. --revision); --force is a
    # task-runner concept the build script doesn't accept.
    build_args = [a for a in (_extra or []) if a != "--force"]
    run(
        [PY, "-m", "anime_tools.tagger.cli.build_english_tag_csv", *build_args],
        cwd=ROOT,
    )


# CJK vocab pack (library.anima.vocab_pack): a ~285 MB text-encoder asset, not
# a release-tag asset. Opt-in — `make download-models` does not fetch it.
VOCAB_PACK_REPO = "sorryhyun/anima-vocab-pack-cjk"
VOCAB_PACK_STEM = "anima_cjk_vocab_pack"
VOCAB_PACK_REL = "models/vocab_packs"


def cmd_download_vocab_pack(_extra):
    """Fetch the shipped CJK vocab pack (.safetensors + .json pair).

    Lands at ``models/vocab_packs/anima_cjk_vocab_pack.{safetensors,json}``;
    point ``vocab_pack`` in ``configs/base.toml`` at the prefix
    ``models/vocab_packs/anima_cjk_vocab_pack`` to enable it, then re-run
    ``make preprocess-te ARGS=--overwrite`` for any CJK captions. The repo's
    ``tokenizer_qwen3/`` is for pipelines without a Qwen3 tokenizer of their
    own — this repo reuses the text encoder's, so it is not fetched.
    """
    dst = ROOT / VOCAB_PACK_REL
    files = [f"{VOCAB_PACK_STEM}.safetensors", f"{VOCAB_PACK_STEM}.json"]
    if _skip("CJK vocab pack", [dst / f for f in files], _extra):
        return
    dst.mkdir(parents=True, exist_ok=True)
    run(["hf", "download", VOCAB_PACK_REPO, *files, "--local-dir", VOCAB_PACK_REL])
    print(
        f'  → enable with vocab_pack = "{VOCAB_PACK_REL}/{VOCAB_PACK_STEM}" in configs/base.toml'
    )


def cmd_download_mit(_extra):
    dst = ROOT / "models" / "mit"
    if _skip("MIT", [dst / "model.pth"], _extra):
        return
    dst.mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "a-b-c-x-y-z/Manga-Text-Segmentation-2025",
            "model.pth",
            "--local-dir",
            "models/mit",
        ]
    )


def cmd_download_anima(_extra):
    models = ROOT / "models"
    # Final (post-move) destinations — this is what we verify against, NOT the
    # transient split_files/ layout hf downloads into (see module docstring).
    finals = [
        models / "diffusion_models" / "anima-base-v1.0.safetensors",
        models / "text_encoders" / "qwen_3_06b_base.safetensors",
        models / "vae" / "qwen_image_vae.safetensors",
    ]
    if _skip("Anima base (DiT + TE + VAE, ~5GB)", finals, _extra):
        return
    for d in ["diffusion_models", "text_encoders", "vae"]:
        (models / d).mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "circlestone-labs/Anima",
            "split_files/diffusion_models/anima-base-v1.0.safetensors",
            "split_files/text_encoders/qwen_3_06b_base.safetensors",
            "split_files/vae/qwen_image_vae.safetensors",
            "--local-dir",
            "models",
            "--include",
            "split_files/*",
        ]
    )
    split = models / "split_files"
    for subdir in ["diffusion_models", "text_encoders", "vae"]:
        src = split / subdir
        dst = models / subdir
        if src.exists():
            for f in src.iterdir():
                shutil.move(str(f), str(dst / f.name))
    if split.exists():
        shutil.rmtree(split)


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


def cmd_download_models(_extra):
    # Continue-on-failure: a gated/un-authed component (SAM3) must not abort the
    # rest. ``run`` sys.exits on a non-zero subprocess, so catch SystemExit per
    # component; each is skip-if-present so the retry doesn't re-download successes.
    components = [
        ("Anima base", cmd_download_anima),
        ("SAM3 (gated)", cmd_download_sam3),
        ("MIT", cmd_download_mit),
        ("PE-Core", cmd_download_pe),
        ("PE-Spatial", cmd_download_pe_spatial),
        ("Anima Tagger vocab", cmd_download_tagger),
        ("Danbooru classified tags", cmd_download_danbooru_tags),
    ]
    failed: list[str] = []
    for name, fn in components:
        try:
            fn(_extra)
        except SystemExit as e:
            if e.code:
                failed.append(name)
                print(f"  ✗ {name} failed (exit {e.code}); continuing")
    if failed:
        print()
        print("The following downloads did not complete:")
        for name in failed:
            print(f"  - {name}")
        print()
        print("Common causes:")
        print("  - not authenticated: run `hf auth login` and re-run")
        print(
            "  - SAM3 is gated: request access at https://huggingface.co/facebook/sam3"
        )
        print("Successful components are cached; re-running only retries the failures.")
        raise SystemExit(1)
