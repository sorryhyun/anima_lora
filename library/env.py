"""Minimal ``.env`` loader — no external dependency.

Used by scripts that need user-specific paths and credentials (HF token,
ComfyUI registry token, external corpus directories) without hardcoding
them in the repo.

Format: standard ``KEY=VALUE`` lines, ``#`` for comments, optional surrounding
single or double quotes around the value. No shell interpolation; values are
taken literally. Existing process env wins over file values (so a CLI
``CAPTION_CORPUS_DIR=… make foo`` overrides the file).

Looks for ``.env`` at the project root by default — the directory two levels
up from this file (``anima_lora/``).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def anima_home() -> Path:
    """Repo home used to anchor every repo-relative path.

    Defaults to :func:`project_root` (the ``anima_lora/`` checkout that holds
    ``configs/``, ``models/``, ``output/`` …). Set ``ANIMA_HOME`` to override —
    this is what lets ``import anima_lora`` and the CLI run from *any* working
    directory instead of requiring a ``cd`` into the repo first.
    """
    override = os.environ.get("ANIMA_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return project_root()


def resolve_under_home(path) -> Path:
    """Resolve a possibly-relative path against :func:`anima_home`.

    Absolute and ``~``-prefixed paths pass through untouched; bare relative
    paths are interpreted relative to the repo home rather than the current
    working directory. Idempotent (absolute in → same path out), so it is safe
    to call at every layer of a call chain without double-anchoring.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return anima_home() / p


def load_dotenv(path: Optional[Path] = None) -> dict[str, str]:
    """Read a ``.env`` file into ``os.environ`` (without overriding existing keys).

    Returns the dict of values that were *added* (useful for logging /
    test introspection). A missing file is a no-op — callers shouldn't
    depend on .env being present.
    """
    if path is None:
        path = anima_home() / ".env"
    added: dict[str, str] = {}
    if not path.exists():
        return added
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val
            added[key] = val
    return added


# Default checkpoint paths — single source of truth for the DiT / VAE / text
# encoder a bench script or example needs. Resolution order, highest first:
#   1. env vars ANIMA_DIT / ANIMA_VAE / ANIMA_TEXT_ENCODER (incl. anything
#      promoted from .env by load_dotenv)
#   2. configs/base.toml (the real source of truth shared with training)
#   3. the literals below (only if base.toml is missing/unreadable)
_CKPT_ENV = {
    "dit": "ANIMA_DIT",
    "vae": "ANIMA_VAE",
    "text_encoder": "ANIMA_TEXT_ENCODER",
    "vocab_pack": "ANIMA_VOCAB_PACK",
}
_CKPT_BASE_TOML_KEY = {
    "dit": "pretrained_model_name_or_path",
    "vae": "vae",
    "text_encoder": "qwen3",
    "vocab_pack": "vocab_pack",
}
_CKPT_FALLBACK = {
    "dit": "models/diffusion_models/anima-base-v1.0.safetensors",
    "vae": "models/vae/qwen_image_vae.safetensors",
    "text_encoder": "models/text_encoders/qwen_3_06b_base.safetensors",
    # The CJK vocab pack is opt-in: "" = off (stock T5 tokenizer, bit-exact).
    "vocab_pack": "",
}


@dataclass(frozen=True)
class DefaultCheckpoints:
    """The repo's default DiT / VAE / text-encoder paths (repo-relative).

    Paths are returned as-is (relative or absolute); the model loaders anchor
    relative paths under :func:`anima_home` via :func:`resolve_under_home`.
    """

    dit: str
    vae: str
    text_encoder: str
    # CJK vocab pack path prefix (``library.anima.vocab_pack``); ``""`` = off.
    vocab_pack: str = ""


def default_checkpoints() -> DefaultCheckpoints:
    """Resolve the default DiT / VAE / text-encoder paths.

    Env (``ANIMA_DIT`` / ``ANIMA_VAE`` / ``ANIMA_TEXT_ENCODER``) wins over
    ``configs/base.toml``, which wins over hardcoded fallbacks. ``.env`` is
    consulted (via :func:`load_dotenv`, which never clobbers real env vars), so
    callers need not load it themselves. This is the one place bench scripts and
    examples should reach for these paths instead of re-deriving the
    ``os.environ.get("ANIMA_DIT", "models/…")`` pattern.
    """
    load_dotenv()

    base: dict[str, str] = {}
    base_path = anima_home() / "configs" / "base.toml"
    if base_path.exists():
        try:
            with base_path.open("rb") as fh:
                data = tomllib.load(fh)
            base = {
                k: data[toml_key]
                for k, toml_key in _CKPT_BASE_TOML_KEY.items()
                if isinstance(data.get(toml_key), str)
            }
        except (OSError, tomllib.TOMLDecodeError):
            base = {}

    def pick(kind: str) -> str:
        return os.environ.get(_CKPT_ENV[kind]) or base.get(kind) or _CKPT_FALLBACK[kind]

    return DefaultCheckpoints(
        dit=pick("dit"),
        vae=pick("vae"),
        text_encoder=pick("text_encoder"),
        vocab_pack=pick("vocab_pack"),
    )
