#!/usr/bin/env python3
"""Perceptual-hash edit-pair miner for the EasyControl *phash_edit* descriptor.

Mines **aligned in-place edit pairs** out of the raw crawl pool by perceptual
hash, and captions each pair with the **tag delta** between its members — so the
prompt is an edit instruction ("given this image, apply these changes") rather
than a description.

Why phash (measured 2026-08-20)
------------------------------
The curated pool (``post_image_dataset/``, 3,007 images) yields only **72**
pairs at tag-Δ≤16 — curation stripped the variant uploads. The raw crawl pool
(``$CAPTION_CORPUS_DIR/retrieved``, 16k images) has them, and ``gelcrawl``
already caches a 256-bit ``imagehash.phash`` per file:

* random pairs sit at phash 128; tag-Δ≤16 candidates at median 60 — tag
  similarity predicts image similarity too weakly to gate on.
* ``phash <= 40`` over all 128.9M pairs → ~2.4k aligned pairs in ~7 s of CPU,
  still genuine in-place variants when spot-checked through 36.
* A tag-Δ prefilter would discard ~800 of them, so phash finds pairs and the
  tag delta only captions them.

Output shape
------------
``pool/<artist>/<id>.<ext>`` — one symlink per **distinct** participating image
— plus ``pairs.json``. The pool is what gets resized and VAE-encoded; every pair
view is then materialized as symlinks over it by
``make easycontrol-preprocess EASYADAPTER=phash_edit``:

* ``resized/<artist>/{pair}_no_tags.<ext>`` → the target's resized image, with a
  real ``.txt`` holding the **delta caption** (the only pair-specific artifact).
* ``cache/…/{pair}_no_tags_{WxH}_anima.npz`` → the target's latent.
* ``cond/…/{pair}_no_tags_{W'xH'}_anima.npz`` → the cond's latent, keyed by the
  target stem (the EasyControl loader's convention).

Each distinct image is resized and VAE-encoded once (per-pair staging would
encode it once per pair and direction); only the delta captions are TE-encoded.

Contract (mirrors ``near_twins`` / ``subject_edit_pairs``):
  * reads ``[staging]`` + ``name`` from ``--config`` (default
    ``configs/easycontrol/phash_edit.toml``); explicit CLI flags win.
  * ``{CAPTION_CORPUS_DIR}`` / ``$CAPTION_CORPUS_DIR`` expand inside path knobs.
  * emits ``pool/<artist>/`` + ``pairs.json`` and rewrites the blueprint tail
    of the config in place.

Instruction format (``--instruction_format``)
---------------------------------------------
``prefix`` (default, subject_edit-compatible) ``glasses, smile, -hat``
``word``                                     ``add glasses, add smile, remove hat``

Both are flat comma bags, so the TE caching pass can shuffle them safely. The
literal ``Add: a, b. Remove: c.`` phrasing is **not** offered: the period is the
position-clause delimiter, and ``parse_caption`` glues it into a single tag
(``'smile. Remove: hat'``) — see ``anime_tools.captions.position_clauses``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from easycontrol_adapters.tools.near_twins.outputs import (  # noqa: E402
    _BLUEPRINT_SENTINEL,
    _strip_blueprint,
)

try:
    from dotenv import load_dotenv  # picks up CAPTION_CORPUS_DIR from anima_lora/.env
except ImportError:  # soft dependency — plain env vars still work

    def load_dotenv(*_a, **_k):  # type: ignore
        return False


DEFAULT_CONFIG = "configs/easycontrol/phash_edit.toml"
IMAGE_EXTS = (".png", ".webp", ".jpg", ".jpeg", ".gif")
HASH_CACHE = ".hash_cache.json"

# Count tags: a pair that flips these is a different *scene*, not an edit — the
# gate that actually removes junk here (aspect ratio does not: same artist means
# same canvas, so AR divergence is 0.000 for over half the candidates).
COUNT_TAGS = {f"{n}{g}" for n in "123456" for g in ("girl", "girls", "boy", "boys")} | {
    "multiple girls",
    "multiple boys",
    "6+girls",
    "6+boys",
    "solo",
    "solo focus",
}


# ---------------------------------------------------------------------------- config


def expand_path(s: str) -> str:
    """Expand ``{VAR}`` / ``${VAR}`` / ``$VAR`` / ``~`` in a path string.

    ``{CAPTION_CORPUS_DIR}`` resolves from the environment (loaded from
    ``anima_lora/.env``), so the toml never hardcodes an absolute corpus path.
    """
    s = re.sub(r"\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), s)
    return os.path.expanduser(os.path.expandvars(s))


def _explicit_dests(argv: list[str]) -> set[str]:
    return {
        tok[2:].split("=", 1)[0].replace("-", "_")
        for tok in argv
        if tok.startswith("--")
    }


def apply_staging_config(args: argparse.Namespace, argv: list[str]) -> None:
    """Layer ``[staging]`` from ``--config`` under the CLI (explicit flag wins)."""
    if not args.config:
        return
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        return
    doc = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    explicit = _explicit_dests(argv)
    if doc.get("name") is not None and "name" not in explicit:
        args.name = doc["name"]
    table = doc.get("staging") or {}
    for key, val in table.items():
        dest = key.replace("-", "_")
        if dest in explicit:
            continue
        if not hasattr(args, dest):
            print(
                f"  [warn] unknown [staging] key {key!r} in {cfg_path}", file=sys.stderr
            )
            continue
        if dest == "image_dirs":
            items = val if isinstance(val, (list, tuple)) else [val]
            setattr(args, dest, ",".join(expand_path(str(x)) for x in items))
        else:
            setattr(args, dest, val)


def _default_image_dirs() -> str:
    """``$CAPTION_CORPUS_DIR/retrieved`` — the RAW crawl pool.

    Deliberately not ``selected/``: that tree is the deduplicated curation and
    has essentially no variant pairs left (72 at tag-Δ≤16 out of 4.5M).
    """
    root = os.environ.get("CAPTION_CORPUS_DIR")
    base = Path(root) if root else Path.home() / "gelcrawl"
    return str(base / "retrieved")


# ---------------------------------------------------------------------------- gather


class Member:
    __slots__ = ("key", "image", "tags", "phash", "root")

    def __init__(self, key: str, image: Path, tags: list[str], phash: str, root: Path):
        self.key, self.image, self.tags, self.phash, self.root = (
            key,
            image,
            tags,
            phash,
            root,
        )

    @property
    def artist(self) -> str:
        return self.key.split("/")[0]

    @property
    def stem(self) -> str:
        return self.key.split("/")[-1]


def load_hash_cache(root: Path) -> dict[str, str]:
    """``{artist}/{id}`` → hex phash, from gelcrawl's ``.hash_cache.json``.

    Cache keys are written with Windows separators (``artist\\id.webp``) and
    carry the extension; both are normalized away here.
    """
    path = root / HASH_CACHE
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for k, v in raw.items():
        out[k.replace("\\", "/").rsplit(".", 1)[0]] = v["hash"]
    return out


def gather(image_dirs: list[Path], verbose: bool = True) -> list[Member]:
    """Discover ``<root>/<artist>/<id>`` members carrying a caption AND a phash."""
    members: list[Member] = []
    for root in image_dirs:
        if not root.is_dir():
            raise SystemExit(f"--image-dirs entry not found: {root}")
        hashes = load_hash_cache(root)
        if not hashes:
            raise SystemExit(
                f"{root / HASH_CACHE} missing or empty — run `make post` in the "
                "gelcrawl repo first (it computes imagehash.phash(hash_size=16) "
                "and persists this cache)."
            )
        n_before = len(members)
        no_hash = no_image = 0
        for txt in sorted(root.rglob("*.txt")):
            if txt.name.endswith(".variants.txt"):
                continue
            key = str(txt.relative_to(root).with_suffix(""))
            h = hashes.get(key)
            if h is None:
                no_hash += 1
                continue
            image = next(
                (p for e in IMAGE_EXTS if (p := txt.with_suffix(e)).exists()), None
            )
            if image is None:
                no_image += 1
                continue
            tags = [
                t.strip()
                for t in txt.read_text(encoding="utf-8", errors="replace").split(",")
                if t.strip()
            ]
            if not tags:
                continue
            members.append(Member(key, image, tags, h, root))
        if verbose:
            note = []
            if no_hash:
                note.append(f"{no_hash} without a cached phash")
            if no_image:
                note.append(f"{no_image} without an image")
            print(
                f"[phash_edit] {root}: {len(members) - n_before} members"
                + (f" ({', '.join(note)} skipped)" if note else "")
            )
    return members


# ---------------------------------------------------------------------------- pairing


def phash_pairs(
    members: list[Member], phash_max: int, chunk: int = 1024
) -> list[tuple[int, int, int]]:
    """All ``(phash_distance, i, j)`` with ``i < j`` and distance ``<= phash_max``.

    Hamming over 256-bit hashes via the ``|a| + |b| - 2·(a·b)`` matmul identity —
    one BLAS call per chunk, so the full 128.9M-pair sweep is seconds, not hours.
    """
    n = len(members)
    bits = np.zeros((n, 256), dtype=np.float32)
    for i, m in enumerate(members):
        bits[i] = np.unpackbits(np.frombuffer(bytes.fromhex(m.phash), dtype=np.uint8))
    ones = bits.sum(1).astype(np.int32)

    out: list[tuple[int, int, int]] = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        dist = (ones[s:e, None] + ones[None, :] - 2 * (bits[s:e] @ bits.T)).astype(
            np.int32
        )
        dist[np.arange(e - s), np.arange(s, e)] = 1 << 20  # no self-pairs
        ii, jj = np.where(dist <= phash_max)
        for a, b in zip(ii, jj):
            if s + a < b:
                out.append((int(dist[a, b]), s + int(a), int(b)))
    out.sort()
    return out


def delta_caption(
    src_tags: list[str], dst_tags: list[str], fmt: str, removal_prefix: str
) -> tuple[str, int, int]:
    """(caption, n_additions, n_removals) — the edit instruction A → B.

    Additions in B's caption order, then removals in A's. Shared tags cancel, so
    the character/artist tags drop out whenever both members carry them and the
    condition stream is the only identity source.
    """
    src, dst = set(src_tags), set(dst_tags)
    additions = [t for t in dst_tags if t not in src]
    removals = [t for t in src_tags if t not in dst]
    if fmt == "word":
        tags = [f"add {t}" for t in additions] + [f"remove {t}" for t in removals]
    else:  # "prefix"
        tags = list(additions) + [f"{removal_prefix}{t}" for t in removals]
    return ", ".join(tags), len(additions), len(removals)


def _aspect(path: Path) -> float | None:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as im:
            w, h = im.size
        return w / h if h else None
    except Exception:
        return None


def build_pairs(members: list[Member], args) -> tuple[list[dict], Counter]:
    """Apply every gate to the phash candidates and emit directed pair records."""
    cand = phash_pairs(members, args.phash_max)
    print(f"[phash_edit] {len(cand)} candidate pairs at phash <= {args.phash_max}")

    drop: Counter = Counter()
    md5: dict[str, str] = {}
    degree: Counter = Counter()
    per_artist: Counter = Counter()
    accepted: list[dict] = []

    for dist, i, j in cand:  # sorted by distance → tightest pairs claim quota first
        a, b = members[i], members[j]
        _, n_add, n_rem = delta_caption(a.tags, b.tags, "prefix", args.removal_prefix)
        tag_delta = n_add + n_rem

        if tag_delta < args.delta_min:
            drop["tag delta below --delta_min (no instruction)"] += 1
            continue
        if args.delta_max and tag_delta > args.delta_max:
            drop["tag delta above --delta_max"] += 1
            continue
        if args.same_artist_only and a.artist != b.artist:
            drop["cross-artist (--same_artist_only)"] += 1
            continue
        if a.stem == b.stem:
            # Gelbooru mirrors Danbooru and the crawler files a post under every
            # artist tag, so the same booru id can appear in two artist dirs.
            drop["same booru id under two artist dirs"] += 1
            continue
        for m in (a, b):
            if m.key not in md5:
                md5[m.key] = hashlib.md5(m.image.read_bytes()).hexdigest()
        if md5[a.key] == md5[b.key]:
            drop["byte-identical file"] += 1
            continue
        if args.drop_count_flip:
            if {t for t in a.tags if t in COUNT_TAGS} != {
                t for t in b.tags if t in COUNT_TAGS
            }:
                drop["count-tag flip (1girl <-> 4girls)"] += 1
                continue
        if args.ar_max_log2 > 0:
            ar_a, ar_b = _aspect(a.image), _aspect(b.image)
            if ar_a and ar_b and abs(math.log2(ar_a / ar_b)) > args.ar_max_log2:
                drop["aspect-ratio divergence"] += 1
                continue
        if args.max_pairs_per_image and (
            degree[a.key] >= args.max_pairs_per_image
            or degree[b.key] >= args.max_pairs_per_image
        ):
            drop["--max_pairs_per_image quota"] += 1
            continue
        if (
            args.max_pairs_per_artist
            and per_artist[b.artist] >= args.max_pairs_per_artist
        ):
            drop["--max_pairs_per_artist quota"] += 1
            continue

        degree[a.key] += 1
        degree[b.key] += 1
        per_artist[b.artist] += 1

        directions = [(a, b)] if not args.both_directions else [(a, b), (b, a)]
        for cond, target in directions:
            caption, n_add_d, n_rem_d = delta_caption(
                cond.tags, target.tags, args.instruction_format, args.removal_prefix
            )
            accepted.append(
                {
                    "pair_id": f"{cond.stem}__{target.stem}",
                    "artist": target.artist,
                    "kind": "edit",
                    "cond": cond.key,
                    "target": target.key,
                    "cond_image": str(cond.image),
                    "target_image": str(target.image),
                    "phash": dist,
                    "tag_delta": tag_delta,
                    "n_additions": n_add_d,
                    "n_removals": n_rem_d,
                    "same_artist": cond.artist == target.artist,
                    "delta_caption": caption,
                    "cond_caption": ", ".join(cond.tags),
                }
            )
        if args.limit and len(accepted) >= args.limit:
            print(f"[phash_edit] --limit {args.limit} reached — stopping early.")
            break
    return accepted, drop


# ---------------------------------------------------------------- synthetic arms


MONO_TAGS = ("monochrome", "greyscale")


def _rng_for(key: str) -> "random.Random":
    """Deterministic per-record RNG — the mine is reproducible across re-runs."""
    return random.Random(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))


def _spread(keys: list[str], n: int) -> list[str]:
    """``n`` keys spread round-robin over their ``<artist>/`` prefix.

    A flat sample would inherit the edit arm's artist skew (one artist owns 322
    of the 3,712 edit pairs); round-robin keeps the synthetic arms broad.
    """
    by_artist: dict[str, list[str]] = {}
    for k in sorted(keys):
        by_artist.setdefault(k.split("/")[0], []).append(k)
    out: list[str] = []
    depth = 0
    while len(out) < n:
        row = [v[depth] for v in by_artist.values() if len(v) > depth]
        if not row:
            break
        out.extend(row[: n - len(out)])
        depth += 1
    return out


def build_identity_pairs(
    edit_pairs: list[dict], by_key: dict[str, "Member"], n: int
) -> list[dict]:
    """No-op records: cond IS the target (same latent), caption empty.

    The condition stream and the denoising target are literally the same cached
    latent, so the sample teaches "empty instruction → reproduce the cond".
    Deliberately a *small* share: the twin_edit arm showed that empty-instruction
    identity pairs, given enough mass, harden into a copy-lock that swallows
    small edits (see the note in configs/easycontrol/phash_edit.toml).
    """
    if n <= 0:
        return []
    keys = _spread(sorted({p["target"] for p in edit_pairs}), n)
    out = []
    for key in keys:
        m = by_key[key]
        out.append(
            {
                "pair_id": f"{m.stem}__self",
                "artist": m.artist,
                "kind": "identity",
                "cond": key,
                "target": key,
                "cond_image": str(m.image),
                "target_image": str(m.image),
                "phash": 0,
                "tag_delta": 0,
                "n_additions": 0,
                "n_removals": 0,
                "same_artist": True,
                "delta_caption": "",
                "cond_caption": ", ".join(m.tags),
                "variants": [""],
            }
        )
    return out


def build_colorize_pairs(
    edit_pairs: list[dict],
    by_key: dict[str, "Member"],
    n: int,
    args,
    exclude: frozenset[str] = frozenset(),
) -> list[dict]:
    """Synthetic colorize records: cond = mangafied B&W of the target.

    The condition image does not exist yet — ``make easycontrol-preprocess``
    mangafies (XDoG + screentone, ``easycontrol_adapters/colorization``) each
    selected *resized* target into ``mono/`` and VAE-encodes it into
    ``mono_cache/``; the link step files that latent as the pair's cond. The
    target latent is the pool latent the edit arm already encoded, so the arm
    costs one extra cond encode per record and nothing on the target side.

    Captions are written as an explicit ``{stem}.variants.txt`` sidecar (the TE
    step treats it as the source of truth) with one of two forms drawn per
    variant:

    * ``-monochrome, -greyscale`` — the task marker, expressed in the same delta
      grammar as the edit arm and in real corpus vocabulary (636/570 occurrences
      in the crawl pool), so it composes with ordinary edit instructions.
    * the target's **color tags only** (``filter_to_colors``) at
      ``--colorize_dropout`` tag dropout, floored at one surviving tag — the
      steering form. Images with no color tag at all fall back to the marker.

    Copyright / character / comic tags are deliberately absent: they are shared
    between cond and target, so the delta grammar cancels them by construction
    and identity comes from the cond stream (unlike the standalone colorize
    task, whose lineart cond cannot carry series identity).

    One direction only — the reverse (decolorize) would need the mangafied view
    resized as a denoising *target*, which the pool layout does not stage.
    """
    if n <= 0:
        return []
    from easycontrol_adapters.colorization.color_caption import filter_to_colors

    mono = set(MONO_TAGS)
    # already B&W: nothing to colorize, and mangafying it is close to a no-op.
    skip = {k for k, m in by_key.items() if mono & set(m.tags)} | exclude
    pool = sorted(
        ({p["target"] for p in edit_pairs} | {p["cond"] for p in edit_pairs}) - skip
    )
    keys = _spread([k for k in pool if k in by_key], n)
    out = []
    for key in keys:
        m = by_key[key]
        rng = _rng_for(f"colorize/{key}")
        colors = [
            t for t in filter_to_colors(", ".join(m.tags)).split(",") if t.strip()
        ]
        variants = []
        for _ in range(max(1, args.colorize_variants)):
            if not colors or rng.random() < 0.5:
                tags = [f"{args.removal_prefix}{t}" for t in MONO_TAGS]
            else:
                tags = [
                    t.strip() for t in colors if rng.random() >= args.colorize_dropout
                ]
                # A 0.8 draw over 2-4 color tags empties out ~half the time; keep
                # one so the steering form always carries a hue to steer with.
                if not tags:
                    tags = [rng.choice(colors).strip()]
            rng.shuffle(tags)
            variants.append(", ".join(tags))
        out.append(
            {
                "pair_id": f"{m.stem}__colorize",
                "artist": m.artist,
                "kind": "colorize",
                "cond": key,  # same image; the cond *latent* comes from mono_cache/
                "target": key,
                "cond_image": str(m.image),
                "target_image": str(m.image),
                "phash": 0,
                "tag_delta": len(MONO_TAGS),
                "n_additions": 0,
                "n_removals": len(MONO_TAGS),
                "same_artist": True,
                "delta_caption": variants[0],
                "cond_caption": ", ".join(m.tags),
                "variants": variants,
            }
        )
    return out


# ---------------------------------------------------------------------------- export


def export_pool(pairs: list[dict], base: Path) -> int:
    """``pool/<artist>/<id>.<ext>`` — one symlink per **distinct** participating image.

    Deliberately NOT the ``_tags``/``_no_tags`` pair tree the other descriptors
    stage. A member joins several pairs and both directions of each, so staging
    per pair would resize and VAE-encode the same image up to ~3× (measured:
    7,424 members over 2,722 distinct images). A VAE latent depends only on the
    image, so the pool is encoded **once** and the preprocess step materializes
    every pair view — image, target latent, cond latent — as symlinks over it.
    The pair-specific artifact is only the delta caption, which is text.

    No ``.txt`` is written here: the pool images are never denoising targets
    (``path_pattern = '*_no_tags.*'``), and the TE pass is filtered to the same
    pattern, so a caption on them would be dead weight in both caches.
    """
    pool = base / "pool"
    if pool.exists():
        shutil.rmtree(pool)
    seen: set[str] = set()
    for p in pairs:
        for key, image in (
            (p["cond"], p["cond_image"]),
            (p["target"], p["target_image"]),
        ):
            if key in seen:
                continue
            seen.add(key)
            src = Path(image)
            dst = pool / f"{key}{src.suffix}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src.resolve())
    return len(seen)


def blueprint_text() -> str:
    return f"""{_BLUEPRINT_SENTINEL}
# phash-mined aligned edit pairs wired as an EasyControl *instruction-edit*
# control task. The miner stages a DEDUPLICATED `pool/` of the participating
# images; the preprocess step resizes and VAE-encodes that pool ONCE, then
# materializes every pair as symlinks over it — a `{{pair}}_no_tags` image in
# `resized/`, its latent in `cache/`, and the partner's latent under the same
# stem in `cond/`. Only the delta captions are TE-encoded.
#
# Roles (EasyControl: cache_dir = denoising target, cond_cache_dir = condition):
#   target = the `_no_tags` view; its caption is the TAG DELTA vs the cond,
#            i.e. an edit instruction, not a description.
#   cond   = the partner's latent (the source image fed via set_cond), filed
#            under the target stem at its own bucket (cond != target shapes are
#            supported; cond_diff_loss self-skips on a mismatch).
# So the adapter learns: given THIS image and THIS edit instruction, produce the
# edited result. Shared tags cancel out of the delta, so the character name is
# absent from the prompt and identity must come from the cond stream.
#
# `path_pattern` keeps only the `_no_tags` views as targets — the resized pool
# members are not denoising targets and carry no caption.
# `{{name}}` below interpolates from the top-level `name` key at train time.

[general]
caption_extension = '.txt'

[[datasets]]
batch_size = 1

  [[datasets.subsets]]
  image_dir = 'post_image_dataset/easycontrol/{{name}}/resized'
  cache_dir = 'post_image_dataset/easycontrol/{{name}}/cache'        # denoising target: the _no_tags members
  cond_cache_dir = 'post_image_dataset/easycontrol/{{name}}/cond'    # condition: the paired _tags latent (keyed by the _no_tags stem)
  path_pattern = '*_no_tags.*'     # targets = the edited members only
  recursive = true                 # tree is nested <artist>/<pair_id>_no_tags; caches mirror it
  flip_aug = false                 # latents can't be flipped post-hoc; the cond cache has no flipped variant
  num_repeats = 1
"""


def write_dataset_config(config_path: Path) -> None:
    """Rewrite the blueprint tail, preserving the user-owned head verbatim."""
    if not config_path.is_file():
        raise SystemExit(
            f"{config_path} not found — the phash_edit descriptor ships with the "
            "repo (configs/easycontrol/phash_edit.toml); restore it first."
        )
    head = _strip_blueprint(config_path.read_text(encoding="utf-8"))
    config_path.write_text(f"{head}\n\n{blueprint_text()}", encoding="utf-8")


# ---------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m easycontrol_adapters.tools.phash_edit_pairs",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="toml with a [staging] table of run knobs (CLI wins; '' disables)",
    )
    p.add_argument(
        "--config-out",
        dest="config_out",
        default=None,
        help="write the regenerated blueprint here instead of --config",
    )
    p.add_argument(
        "--name",
        default=None,
        help="output slug — routes post_image_dataset/easycontrol/<name>/",
    )
    p.add_argument(
        "--image-dirs",
        dest="image_dirs",
        default=_default_image_dirs(),
        help="comma-separated <root>/<artist>/<id> trees; each needs gelcrawl's "
        ".hash_cache.json. Defaults to $CAPTION_CORPUS_DIR/retrieved (the RAW "
        "pool — selected/ is deduplicated and has no variant pairs left)",
    )
    p.add_argument(
        "--export-dir",
        dest="export_dir",
        default=None,
        help="override the staging base (default post_image_dataset/easycontrol/<name>)",
    )

    p.add_argument(
        "--phash_max",
        type=int,
        default=40,
        help="max Hamming distance between the two 256-bit phashes. Spot-checked "
        "aligned through 36; random pairs sit at 128. NB gelcrawl's own dedup "
        "already deleted within-artist pairs at distance <= 2",
    )
    p.add_argument(
        "--delta_min",
        type=int,
        default=1,
        help="drop pairs whose tag delta is below this — 0 means the two captions "
        "are identical, i.e. an empty instruction",
    )
    p.add_argument(
        "--delta_max",
        type=int,
        default=0,
        help="drop pairs whose tag delta exceeds this (0 = no cap). A tag-delta "
        "prefilter discards aligned pairs whose caption moved a lot; leave off "
        "unless you specifically want terse instructions",
    )
    p.add_argument(
        "--drop_count_flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop pairs that flip a count tag (1girl <-> 4girls) — a different "
        "scene, not an edit",
    )
    p.add_argument(
        "--ar_max_log2",
        type=float,
        default=0.0,
        help="max |log2(AR_a/AR_b)| (0 = off). Near-useless in practice: same artist "
        "means same canvas, so AR divergence is 0.000 for most candidates",
    )
    p.add_argument(
        "--same_artist_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="restrict to within-artist pairs (measured: no cross-artist pair "
        "reaches phash <= 40 anyway)",
    )
    p.add_argument(
        "--max_pairs_per_image",
        type=int,
        default=4,
        help="cap how many pairs one image may join (0 = unlimited). Tightest pairs "
        "claim the quota first; stops one variant series dominating",
    )
    p.add_argument(
        "--max_pairs_per_artist",
        type=int,
        default=0,
        help="cap accepted pairs per artist (0 = unlimited)",
    )
    p.add_argument(
        "--both_directions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="emit A->B and B->A, so every attribute is taught as both an addition "
        "and a removal. Doubles the images the preprocess pass must encode",
    )
    p.add_argument(
        "--instruction_format",
        choices=("prefix", "word"),
        default="prefix",
        help="'prefix': `glasses, smile, -hat` (subject_edit-compatible). "
        "'word': `add glasses, remove hat` (lexical, so the text encoder reads "
        "real negation words). Both stay flat comma bags — a literal "
        "'Add: a. Remove: b.' would be corrupted by the position-clause parser",
    )
    p.add_argument(
        "--removal_prefix",
        default="-",
        help="prefix marking a removal tag in the 'prefix' format",
    )
    p.add_argument(
        "--identity_frac",
        type=float,
        default=0.0,
        help="share of the final set to emit as no-op identity records (cond IS "
        "the target latent, caption empty). Keep small: enough mass here hardens "
        "into a copy-lock that swallows small edits (the twin_edit failure)",
    )
    p.add_argument(
        "--colorize_frac",
        type=float,
        default=0.0,
        help="share of the final set to emit as synthetic colorize records "
        "(cond = mangafied B&W of the target, produced by the preprocess step)",
    )
    p.add_argument(
        "--colorize_dropout",
        type=float,
        default=0.8,
        help="tag-dropout applied to the color-tag caption form of a colorize "
        "record (the other form is the bare `-monochrome, -greyscale` marker)",
    )
    p.add_argument(
        "--colorize_variants",
        type=int,
        default=4,
        help="caption variants written into each colorize record's "
        "{stem}.variants.txt; each independently draws marker-vs-color-tags",
    )
    p.add_argument(
        "--limit", type=int, default=0, help="stop after N directed pairs (smoke runs)"
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="report the yield without writing pool/, pairs.json or the blueprint",
    )

    argv = list(sys.argv[1:] if argv is None else argv)
    args = p.parse_args(argv)
    load_dotenv(REPO_ROOT / ".env")
    # image_dirs default was computed before .env loaded; recompute if untouched.
    if "image_dirs" not in _explicit_dests(argv):
        args.image_dirs = _default_image_dirs()
    apply_staging_config(args, argv)
    args.name = args.name or "phash_edit"
    return args


def main() -> None:
    args = parse_args()
    image_dirs = [
        Path(expand_path(d.strip())) for d in args.image_dirs.split(",") if d.strip()
    ]
    members = gather(image_dirs)
    if len(members) < 2:
        raise SystemExit("[phash_edit] fewer than 2 usable members — nothing to pair.")
    print(f"[phash_edit] {len(members)} members with caption + phash")

    pairs, drop = build_pairs(members, args)
    if drop:
        print("[phash_edit] dropped:")
        for reason, n in drop.most_common():
            print(f"    {n:>7,}  {reason}")
    if not pairs:
        raise SystemExit(
            "[phash_edit] no pair survived the gates — loosen --phash_max."
        )

    # Synthetic arms sized as a share of the FINAL set, so the requested
    # fractions hold after they are appended (edit arm = the remainder).
    frac = args.identity_frac + args.colorize_frac
    if frac >= 1.0:
        raise SystemExit(
            f"--identity_frac + --colorize_frac = {frac} — must be < 1 "
            "(the edit arm is the remainder)."
        )
    if frac > 0:
        by_key = {m.key: m for m in members}
        total = len(pairs) / (1.0 - frac)
        ident = build_identity_pairs(pairs, by_key, round(total * args.identity_frac))
        # Disjoint from the identity arm: an image serving as both would be
        # over-represented relative to the round-robin spread.
        color = build_colorize_pairs(
            pairs,
            by_key,
            round(total * args.colorize_frac),
            args,
            frozenset(p["target"] for p in ident),
        )
        print(
            f"[phash_edit] synthetic arms: {len(ident)} identity + {len(color)} "
            f"colorize on top of {len(pairs)} edit records"
        )
        pairs = pairs + ident + color

    edits = [p for p in pairs if p.get("kind", "edit") == "edit"]
    n_unordered = len({tuple(sorted((p["cond"], p["target"]))) for p in edits})
    images = {p["cond"] for p in pairs} | {p["target"] for p in pairs}
    deltas = sorted(p["tag_delta"] for p in edits)
    phashes = sorted(p["phash"] for p in edits)
    print(
        f"[phash_edit] accepted {n_unordered:,} pairs → {len(pairs):,} directed examples\n"
        f"             {len(images):,} distinct images to resize + VAE-encode; "
        f"{len(pairs):,} delta captions to TE-encode "
        f"(the {len(pairs) * 2:,} pair views are symlinks over that pool)\n"
        f"             phash median {phashes[len(phashes) // 2]} | "
        f"tag delta median {deltas[len(deltas) // 2]} (p10 {deltas[len(deltas) // 10]}, "
        f"p90 {deltas[len(deltas) * 9 // 10]})"
    )
    top = Counter(p["artist"] for p in pairs).most_common(5)
    print(f"             top artists: {top}")

    if args.dry_run:
        print("[phash_edit] --dry-run: nothing written.")
        return

    base = (
        Path(expand_path(args.export_dir))
        if args.export_dir
        else (REPO_ROOT / "post_image_dataset" / "easycontrol" / args.name)
    )
    base.mkdir(parents=True, exist_ok=True)
    n_pool = export_pool(pairs, base)
    manifest = {
        "meta": {
            "image_dirs": [str(d) for d in image_dirs],
            "phash_max": args.phash_max,
            "delta_min": args.delta_min,
            "delta_max": args.delta_max,
            "drop_count_flip": args.drop_count_flip,
            "same_artist_only": args.same_artist_only,
            "max_pairs_per_image": args.max_pairs_per_image,
            "max_pairs_per_artist": args.max_pairs_per_artist,
            "both_directions": args.both_directions,
            "instruction_format": args.instruction_format,
            "identity_frac": args.identity_frac,
            "colorize_frac": args.colorize_frac,
            "colorize_dropout": args.colorize_dropout,
            "n_by_kind": dict(Counter(p.get("kind", "edit") for p in pairs)),
            "n_members": len(members),
            "n_pairs_unordered": n_unordered,
            "n_directed": len(pairs),
            "n_images": len(images),
        },
        "pairs": pairs,
    }
    (base / "pairs.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    cfg_out = Path(args.config_out or args.config or DEFAULT_CONFIG)
    write_dataset_config(cfg_out if cfg_out.is_absolute() else REPO_ROOT / cfg_out)
    print(
        f"[phash_edit] pooled {n_pool:,} distinct images → {base / 'pool'}  "
        f"(manifest: {base / 'pairs.json'})"
    )
    print(f"[phash_edit] blueprint rewritten in {cfg_out}")
    print(
        "[phash_edit] next: make easycontrol-preprocess EASYADAPTER="
        f"{args.name}  →  make easycontrol EASYADAPTER={args.name}"
    )


if __name__ == "__main__":
    main()
