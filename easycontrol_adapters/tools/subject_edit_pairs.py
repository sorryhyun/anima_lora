"""Delta-caption edit-pair miner for the EasyControl *subject_edit* descriptor.

Same mined cross-image pairs as the subject descriptor (cond = image A of a
character, target = image B of the SAME character), but the prompt is the
**tag delta** between the two captions instead of B's full caption
(background: ``_archive/directedit_ec/roadmap.md``):

    additions = B's tags not in A   (in B's caption order)
    removals  = A's tags not in B   (prefixed, e.g. ``-hat``)

The prompt is an *edit instruction*. Shared tags cancel, so the character
NAME tag drops out whenever both captions carry it and identity must come from
the cond stream.

Pair quality is delta-driven: the default partner policy picks the group
member with the SMALLEST symmetric tag difference (same character + small
delta ≈ a near-edit pair), and pairs outside ``[min_delta, max_delta]`` are
dropped (tiny = nothing to learn, huge = unrelated images + caption noise).

Contract (mirrors ``subject_pairs``):
  * reads ``[staging]`` + ``name`` from ``--config``
    (default ``configs/easycontrol/subject_edit.toml``); CLI wins.
  * emits ``staging/<artist>/<stem>.png`` symlinks + **real** ``.txt`` files
    holding the delta caption (NOT symlinks — the prompt is new text), a
    ``pairs.json`` manifest, the ``cond/`` latent tree, and rewrites the
    blueprint tail in place.
  * ``--cond-only`` rebuilds ``cond/`` from the manifest (idempotent).

The delta captions are new text, so the shared LoRA TE cache does NOT apply:
the blueprint routes ``text_cache_dir`` to ``{name}/text`` (inpaint-style) and
``make easycontrol-preprocess EASYADAPTER=subject_edit`` runs the TE encode
pass over the staged tree. VAE latents + PE still ride the shared cache.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from easycontrol_adapters.tools.near_twins.outputs import (  # noqa: E402
    _BLUEPRINT_SENTINEL,
    _strip_blueprint,
)
from easycontrol_adapters.tools.subject_pairs import (  # noqa: E402
    build_cond_tree,
    latent_buckets,
    load_staging_config,
)

DEFAULT_CONFIG = "configs/easycontrol/subject_edit.toml"


def _blueprint_text() -> str:
    # Differs from subject's tail in exactly one line: text_cache_dir routes
    # the TE lookup at the descriptor-owned delta-caption cache (the shared
    # LoRA TE cache holds B's FULL caption — wrong text for this task).
    return f"""{_BLUEPRINT_SENTINEL}
# Delta-caption edit pairs wired as an EasyControl control task. staging/ holds
# PNG symlinks for every mined TARGET plus real .txt files with the TAG DELTA
# (additions + prefixed removals) vs the cond partner; pairs.json records the
# partner + delta stats.
#
# Roles (EasyControl: cache_dir = denoising target, cond_cache_dir = condition):
#   target = image B; prompt = tags(B) - tags(A) + removal-prefixed tags(A)-(B)
#   cond   = image A of the same character, filed under B's stem at A's bucket.
# So the adapter learns: given THIS image and THIS edit instruction, produce
# the edited result — the prompt no longer describes, it instructs (and the
# character name cancels out of it, so identity must come from the cond).

[general]
caption_extension = '.txt'

[[datasets]]
batch_size = 1

  [[datasets.subsets]]
  image_dir = 'post_image_dataset/easycontrol/{{name}}/staging'
  cache_dir = 'post_image_dataset/lora'                             # shared corpus cache: target latents + PE reused as-is
  cond_cache_dir = 'post_image_dataset/easycontrol/{{name}}/cond'   # condition: the partner's latent, keyed by the target stem
  text_cache_dir = 'post_image_dataset/easycontrol/{{name}}/text'   # delta-caption TE cache (the preprocess step encodes it)
  recursive = true                 # tree is nested <artist>/<stem>; the shared cache mirrors it
  flip_aug = false                 # cond latents have no flipped variant
  num_repeats = 1
"""


def write_dataset_config(config_path: Path) -> None:
    if not config_path.is_file():
        raise SystemExit(
            f"{config_path} not found — the subject_edit descriptor ships with "
            "the repo (configs/easycontrol/subject_edit.toml); restore it first."
        )
    head = _strip_blueprint(config_path.read_text(encoding="utf-8"))
    config_path.write_text(f"{head}\n\n{_blueprint_text()}", encoding="utf-8")


def read_tags(resized_dir: Path, key: str) -> list[str] | None:
    """The caption of corpus image ``key`` as an ordered tag list (None = no .txt)."""
    txt = resized_dir / f"{key}.txt"
    if not txt.is_file():
        return None
    return [t.strip() for t in txt.read_text(encoding="utf-8").split(",") if t.strip()]


def delta_caption(
    src_tags: list[str], dst_tags: list[str], removal_prefix: str, include_removals: bool
) -> tuple[list[str], int, int]:
    """(delta tag list, n_additions, n_removals). Order: additions in the
    target caption's order, then prefixed removals in the source's order."""
    src, dst = set(src_tags), set(dst_tags)
    additions = [t for t in dst_tags if t not in src]
    removals = [t for t in src_tags if t not in dst]
    tags = list(additions)
    if include_removals:
        tags += [f"{removal_prefix}{t}" for t in removals]
    return tags, len(additions), len(removals)


def mine_pairs(args) -> list[dict]:
    """Group solo single-character images; pick the minimal-delta cond partner."""
    index_path = REPO_ROOT / args.caption_index
    if not index_path.is_file():
        raise SystemExit(f"{index_path} not found — run `make caption-index` first.")
    image_meta = json.loads(index_path.read_text(encoding="utf-8"))["image_meta"]
    cache_dir = REPO_ROOT / args.lora_cache_dir
    resized_dir = REPO_ROOT / args.resized_dir

    groups: dict[str, list[str]] = {}
    for key, meta in sorted(image_meta.items()):
        chars = meta.get("character", [])
        if len(chars) != 1:
            continue
        if args.solo_only and not any(
            c in ("1girl", "1boy") for c in meta.get("count", [])
        ):
            continue
        groups.setdefault(chars[0], []).append(key)

    rng = random.Random(args.seed)
    pairs: list[dict] = []
    n_uncached = n_filtered = 0
    for character in sorted(groups):
        members = [k for k in groups[character] if latent_buckets(cache_dir, k)]
        n_uncached += len(groups[character]) - len(members)
        tags = {k: read_tags(resized_dir, k) for k in members}
        members = [k for k in members if tags[k]]
        if len(members) < args.min_group:
            continue
        targets = members
        if len(targets) > args.max_per_character:
            targets = rng.sample(targets, args.max_per_character)
        for target in targets:
            t_artist = target.split("/")[0]
            candidates = [k for k in members if k != target]
            if args.partner_scope == "same_artist":
                candidates = [k for k in candidates if k.split("/")[0] == t_artist]
            elif args.partner_scope == "cross_artist":
                candidates = [k for k in candidates if k.split("/")[0] != t_artist]
            scored = []  # (sym_delta, key) within the [min_delta, max_delta] band
            for cand in candidates:
                sym = len(set(tags[cand]) ^ set(tags[target]))
                if args.min_delta <= sym <= args.max_delta:
                    scored.append((sym, cand))
            if not scored:
                n_filtered += 1
                continue
            if args.partner == "min_delta":
                best = min(s for s, _ in scored)
                cond = rng.choice([k for s, k in scored if s == best])
            else:
                cond = rng.choice([k for _, k in scored])
            caption, n_add, n_rem = delta_caption(
                tags[cond], tags[target], args.removal_prefix, args.include_removals
            )
            pairs.append(
                {
                    "target": target,
                    "cond": cond,
                    "character": character,
                    "cond_bucket": latent_buckets(cache_dir, cond)[0],
                    "same_artist": cond.split("/")[0] == t_artist,
                    "n_additions": n_add,
                    "n_removals": n_rem,
                    "delta_caption": ", ".join(caption),
                }
            )
    if n_uncached:
        print(
            f"[subject_edit] {n_uncached} group members skipped — no latent in "
            f"{args.lora_cache_dir} (run `make preprocess` to cache them).",
            file=sys.stderr,
        )
    if n_filtered:
        print(
            f"[subject_edit] {n_filtered} targets dropped — no partner inside "
            f"delta band [{args.min_delta}, {args.max_delta}] "
            f"(scope={args.partner_scope}).",
            file=sys.stderr,
        )
    return pairs


def export_staging(pairs: list[dict], base: Path, resized_dir: Path) -> None:
    """PNG symlinks + REAL .txt delta captions (the one departure from subject)."""
    staging = base / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    for pair in pairs:
        src_png = resized_dir / f"{pair['target']}.png"
        dst = staging / f"{pair['target']}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src_png.resolve())
        dst.with_suffix(".txt").write_text(pair["delta_caption"], encoding="utf-8")


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
    parser.add_argument("--max_per_character", type=int, default=16)
    parser.add_argument(
        "--solo_only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--partner",
        choices=("min_delta", "random"),
        default="min_delta",
        help="cond-partner policy: smallest symmetric tag delta, or random "
        "within the delta band",
    )
    parser.add_argument(
        "--partner_scope",
        choices=("any", "same_artist", "cross_artist"),
        default="any",
        help="restrict partners to the target's artist dir (style held constant "
        "→ purest edit pairs) or away from it (@artist lands in the delta → "
        "style transfer becomes an edit instruction)",
    )
    parser.add_argument(
        "--min_delta",
        type=int,
        default=2,
        help="drop pairs whose symmetric tag difference is below this "
        "(near-duplicates teach nothing)",
    )
    parser.add_argument(
        "--max_delta",
        type=int,
        default=24,
        help="drop pairs whose symmetric tag difference exceeds this "
        "(unrelated images + caption noise dominate large deltas)",
    )
    parser.add_argument(
        "--include_removals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="encode source-only tags as removal instructions (see "
        "--removal_prefix); without them, absence is ambiguous between "
        "'keep' and 'remove'",
    )
    parser.add_argument(
        "--removal_prefix",
        default="-",
        help="prefix marking a removal tag in the delta caption",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cond-only", dest="cond_only", action="store_true")

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
            f"[subject_edit] dropping {len(stale)} pairs — no resized PNG "
            f"under {args.resized_dir} (stale cache?).",
            file=sys.stderr,
        )
        pairs = [p for p in pairs if p not in stale]
    if not pairs:
        raise SystemExit("[subject_edit] mined 0 pairs — nothing to stage.")
    characters = {p["character"] for p in pairs}
    same = sum(1 for p in pairs if p["same_artist"])
    med_delta = sorted(p["n_additions"] + p["n_removals"] for p in pairs)[
        len(pairs) // 2
    ]
    export_staging(pairs, base, resized_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / "pairs.json").write_text(
        json.dumps(
            {
                "meta": {
                    "miner": "subject_edit_pairs",
                    "caption_index": args.caption_index,
                    "seed": args.seed,
                    "solo_only": args.solo_only,
                    "min_group": args.min_group,
                    "max_per_character": args.max_per_character,
                    "partner": args.partner,
                    "partner_scope": args.partner_scope,
                    "min_delta": args.min_delta,
                    "max_delta": args.max_delta,
                    "include_removals": args.include_removals,
                    "removal_prefix": args.removal_prefix,
                    "n_pairs": len(pairs),
                    "n_characters": len(characters),
                    "n_same_artist": same,
                    "median_delta_tags": med_delta,
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
        f"[subject_edit] {len(pairs)} pairs over {len(characters)} characters "
        f"({same} same-artist, median delta {med_delta} tags) → {base}"
    )


if __name__ == "__main__":
    main()
