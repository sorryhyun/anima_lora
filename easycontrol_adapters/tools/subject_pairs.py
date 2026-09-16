"""Cross-image subject-pair miner for the EasyControl *subject* descriptor.

Mines (cond = image A of a character, target = image B of the SAME character)
pairs from the typed caption index, so the adapter must learn content-based
retrieval rather than positional copying. Same ``EasyControlNetwork``; only the
data pairing changes. Background: ``_archive/directedit_ec/initial_proposal.md``.

No PE matching and no encode pass: both members are corpus images whose
latents/TE already live in the shared LoRA cache (``post_image_dataset/lora/``),
so staging is pure symlinks.

Contract (mirrors ``easycontrol_adapters.tools.near_twins``):
  * reads its ``[staging]`` table + top-level ``name`` slug from
    ``--config`` (default ``configs/easycontrol/subject.toml``); CLI wins.
  * emits ``post_image_dataset/easycontrol/<name>/staging/<artist>/<stem>.png``
    (+ ``.txt``) symlinks for every *target*, a ``pairs.json`` manifest beside
    it, and rewrites the descriptor's ``[[datasets]]`` blueprint tail in place.
  * ``--cond-only`` (the ``easycontrol-preprocess`` step) skips mining and
    (re)builds ``cond/`` from the manifest: for each pair, the cond member's
    shared-cache latent is symlinked under the TARGET's stem at the COND's
    bucket, matching ``_load_cond_latent``'s stem-glob fallback. Re-run it
    after any shared-corpus re-preprocess (buckets may have moved).

Pairing policy: solo single-character images grouped by character tag;
each target draws a seeded random cond partner from its group, preferring a
DIFFERENT artist dir; groups are capped so no single character dominates.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from easycontrol_adapters.tools.near_twins.outputs import (  # noqa: E402
    _BLUEPRINT_SENTINEL,
    _strip_blueprint,
)

DEFAULT_CONFIG = "configs/easycontrol/subject.toml"


def _blueprint_text() -> str:
    # `{name}` placeholders interpolate from the descriptor's top-level `name`
    # at train time (see _resolve_blueprint_path). cache_dir points at the
    # SHARED corpus cache on purpose — both pair members are ordinary corpus
    # images, so target latents + TE/PE variants are reused, not re-encoded
    # (it is outside post_image_dataset/easycontrol/ so slug rerouting skips it).
    return f"""{_BLUEPRINT_SENTINEL}
# Cross-image subject pairs wired as an EasyControl control task. staging/ holds
# symlinks to the resized corpus for every mined TARGET; pairs.json (sibling of
# staging/) records its cond partner (another image of the same character).
#
# Roles (EasyControl: cache_dir = denoising target, cond_cache_dir = condition):
#   target = image B (+ its full caption — the prompt keeps owning layout/pose)
#   cond   = image A of the same character, filed in cond/ under B's stem at
#            A's bucket (cond≠target shapes are supported downstream).
# So the adapter learns: given ANOTHER image of the subject as reference,
# generate this one — retrieval that positional copying cannot satisfy.

[general]
caption_extension = '.txt'

[[datasets]]
batch_size = 1

  [[datasets.subsets]]
  image_dir = 'post_image_dataset/easycontrol/{{name}}/staging'
  cache_dir = 'post_image_dataset/lora'                             # shared corpus cache: target latents + TE/PE reused as-is
  cond_cache_dir = 'post_image_dataset/easycontrol/{{name}}/cond'   # condition: the partner's latent, keyed by the target stem
  recursive = true                 # tree is nested <artist>/<stem>; the shared cache mirrors it
  flip_aug = false                 # cond latents have no flipped variant
  num_repeats = 1
"""


def write_dataset_config(config_path: Path) -> None:
    """Rewrite the blueprint tail of ``config_path``, preserving the head tables."""
    if not config_path.is_file():
        raise SystemExit(
            f"{config_path} not found — the subject descriptor ships with the "
            "repo (configs/easycontrol/subject.toml); restore it before mining."
        )
    head = _strip_blueprint(config_path.read_text(encoding="utf-8"))
    config_path.write_text(f"{head}\n\n{_blueprint_text()}", encoding="utf-8")


def load_staging_config(path: Path) -> tuple[str, dict]:
    """Read the descriptor → (name slug, [staging] table)."""
    cfg = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    name = str(cfg.get("name") or path.stem).strip()
    return name, dict(cfg.get("staging") or {})


def latent_buckets(cache_dir: Path, key: str) -> list[str]:
    """The ``WxH`` bucket strings this corpus image is cached at (usually one)."""
    d = cache_dir / Path(key).parent
    stem = Path(key).name
    return sorted(
        p.name[len(stem) + 1 : -len("_anima.npz")]
        for p in d.glob(f"{stem}_*_anima.npz")
    )


def mine_pairs(args) -> list[dict]:
    """Group solo single-character corpus images and assign cond partners."""
    index_path = REPO_ROOT / args.caption_index
    if not index_path.is_file():
        raise SystemExit(f"{index_path} not found — run `make caption-index` first.")
    image_meta = json.loads(index_path.read_text(encoding="utf-8"))["image_meta"]
    cache_dir = REPO_ROOT / args.lora_cache_dir

    groups: dict[str, list[str]] = defaultdict(list)
    for key, meta in sorted(image_meta.items()):
        chars = meta.get("character", [])
        if len(chars) != 1:
            continue
        if args.solo_only and not any(
            c in ("1girl", "1boy") for c in meta.get("count", [])
        ):
            continue
        groups[chars[0]].append(key)

    rng = random.Random(args.seed)
    pairs: list[dict] = []
    n_uncached = 0
    for character in sorted(groups):
        members = [k for k in groups[character] if latent_buckets(cache_dir, k)]
        n_uncached += len(groups[character]) - len(members)
        if len(members) < args.min_group:
            continue
        targets = members
        if len(targets) > args.max_per_character:
            targets = rng.sample(targets, args.max_per_character)
        for target in targets:
            candidates = [k for k in members if k != target]
            cross = [k for k in candidates if k.split("/")[0] != target.split("/")[0]]
            cond = rng.choice(cross or candidates)
            pairs.append(
                {
                    "target": target,
                    "cond": cond,
                    "character": character,
                    "cond_bucket": latent_buckets(cache_dir, cond)[0],
                }
            )
    if n_uncached:
        print(
            f"[subject_pairs] {n_uncached} group members skipped — no latent in "
            f"{args.lora_cache_dir} (run `make preprocess` to cache them).",
            file=sys.stderr,
        )
    return pairs


def export_staging(pairs: list[dict], base: Path, resized_dir: Path) -> None:
    """Symlink each target's resized image + caption into ``staging/``."""
    staging = base / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    for pair in pairs:
        src_png = resized_dir / f"{pair['target']}.png"
        dst = staging / f"{pair['target']}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src_png.resolve())
        src_txt = src_png.with_suffix(".txt")
        if src_txt.is_file():
            dst.with_suffix(".txt").symlink_to(src_txt.resolve())


def build_cond_tree(base: Path, lora_cache_dir: Path) -> None:
    """(Re)build ``cond/`` from ``pairs.json`` — pure symlinks, idempotent.

    For target ``<artist>/<stem>`` with cond partner ``<a2>/<s2>`` cached at
    bucket ``WxH``, links ``cond/<artist>/<stem>_<WxH>_anima.npz`` →
    ``<lora_cache>/<a2>/<s2>_<WxH>_anima.npz``. ``_load_cond_latent`` resolves
    it via the stem-glob fallback and reads the cond's reso off the filename.
    """
    manifest = base / "pairs.json"
    if not manifest.is_file():
        raise SystemExit(f"{manifest} not found — run the mining step first.")
    pairs = json.loads(manifest.read_text(encoding="utf-8"))["pairs"]
    cond_dir = base / "cond"
    if cond_dir.exists():
        shutil.rmtree(cond_dir)
    linked = skipped = 0
    for pair in pairs:
        buckets = latent_buckets(lora_cache_dir, pair["cond"])
        if not buckets:
            print(
                f"[subject_pairs] cond latent gone for {pair['cond']} — skipping "
                "(re-run staging after a corpus re-preprocess).",
                file=sys.stderr,
            )
            skipped += 1
            continue
        bucket = buckets[0]
        src = lora_cache_dir / f"{pair['cond']}_{bucket}_anima.npz"
        link = cond_dir / f"{pair['target']}_{bucket}_anima.npz"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(src.resolve())
        linked += 1
    print(
        f"[subject_pairs] linked {linked} cond latents into {cond_dir}"
        + (f" ({skipped} skipped)" if skipped else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--config-out", dest="config_out", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--caption_index", default="post_image_dataset/captions/caption_index.json"
    )
    parser.add_argument("--resized_dir", default="post_image_dataset/resized")
    parser.add_argument("--lora_cache_dir", default="post_image_dataset/lora")
    parser.add_argument("--min_group", type=int, default=2)
    parser.add_argument(
        "--max_per_character",
        type=int,
        default=16,
        help="cap on targets per character (keeps miku from dominating the set)",
    )
    parser.add_argument(
        "--solo_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep only 1girl/1boy images (unambiguous character↔pixels binding)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cond-only",
        dest="cond_only",
        action="store_true",
        help="skip mining; rebuild cond/ from the existing pairs.json",
    )

    # Precedence CLI > [staging] toml > argparse default: seed the toml values
    # in as defaults, then parse (explicit CLI flags override them naturally).
    config_arg = None
    argv = sys.argv[1:]
    if "--config" in argv:
        config_arg = argv[argv.index("--config") + 1]
    config_path = REPO_ROOT / (config_arg or DEFAULT_CONFIG)
    name, staging = load_staging_config(config_path)
    known = {a.dest for a in parser._actions}
    parser.set_defaults(name=name, **{k: v for k, v in staging.items() if k in known})
    args = parser.parse_args()

    base = REPO_ROOT / "post_image_dataset" / "easycontrol" / args.name
    lora_cache_dir = REPO_ROOT / args.lora_cache_dir

    if args.cond_only:
        build_cond_tree(base, lora_cache_dir)
        return

    pairs = mine_pairs(args)
    resized_dir = REPO_ROOT / args.resized_dir
    stale = [p for p in pairs if not (resized_dir / f"{p['target']}.png").is_file()]
    if stale:
        print(
            f"[subject_pairs] dropping {len(stale)} pairs — target has a cached "
            f"latent but no resized PNG under {args.resized_dir} (stale cache?).",
            file=sys.stderr,
        )
        pairs = [p for p in pairs if p not in stale]
    if not pairs:
        raise SystemExit("[subject_pairs] mined 0 pairs — nothing to stage.")
    characters = {p["character"] for p in pairs}
    cross = sum(
        1 for p in pairs if p["target"].split("/")[0] != p["cond"].split("/")[0]
    )
    export_staging(pairs, base, resized_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / "pairs.json").write_text(
        json.dumps(
            {
                "meta": {
                    "miner": "subject_pairs",
                    "caption_index": args.caption_index,
                    "seed": args.seed,
                    "solo_only": args.solo_only,
                    "min_group": args.min_group,
                    "max_per_character": args.max_per_character,
                    "n_pairs": len(pairs),
                    "n_characters": len(characters),
                    "n_cross_artist": cross,
                },
                "pairs": pairs,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    build_cond_tree(base, lora_cache_dir)

    out = args.config_out
    write_dataset_config(REPO_ROOT / out if out else config_path)
    print(
        f"[subject_pairs] {len(pairs)} pairs over {len(characters)} characters "
        f"({cross} cross-artist) → {base}"
    )


if __name__ == "__main__":
    main()
