#!/usr/bin/env python3
"""near_twins_tag_gap_miner — mine in-artist variant pairs by attribute gap.

An **exploration / curation tool** (not a training step) that surfaces
near-duplicate *variant* pairs within a single artist where the two members
differ by a **specified attribute** — e.g. one has a speech bubble and the
other doesn't. It feeds EasyControl builders: eval sets, seed data for unpaired
editing, and a difference-region mask localizing *where* the two members differ.

Pipeline (see ``docs/proposal/near_twins_tag_gap_miner.md`` for the full design):

1. **Gather members** per artist from ``--image-dirs`` (default the raw crawl
   pool ``~/gelcrawl/{retrieved,selected}``), scoped ``union`` so a twin can
   straddle the curated cut. Each member's native pixel ``(W, H)`` is read from
   the image header here (no decode).
1b. **Same-size gate (+ tag pivot)**: a true variant pair (a redraw that adds
   one attribute) shares the **exact** canvas, so only members that share their
   ``(W, H)`` with ≥1 sibling *in the same artist* survive — the rest can never
   pair and are dropped before embedding. The pair loop then only ever compares
   equal-size members, which also makes the dense grid match pixel-aligned by
   construction (the original cross-crop case the PE machinery was hedging
   against is gone). In **tag mode** the gate sharpens further: an accepted pair
   has the target tag in *exactly one* member, so a same-size group is only kept
   when it holds BOTH a tagged and an untagged member (an all-tagged or
   all-untagged group can never produce a gap). The whole prune runs *before* the
   PE-Spatial load, so an empty candidate set skips the GPU entirely.
2. **Embed** each *surviving* image with **PE-Spatial-B16-512** (``library.vision``)
   at a fixed 512x512 native bucket → CLS descriptor + 32x32 patch grid (pooled to
   16x16, L2-normed). Cached per-image under ``~/.cache/near_twin/``.
3. **Stage A — global prefilter**: within-artist all-pairs cosine on the CLS
   descriptor; keep ``>= --sim-min``.
4. **Stage B — dense grid match**: pool each survivor's grid to ``G x G``, run a
   mutual-NN + ratio test, count inliers ``>= --cell-match-min``; a pair is a
   near-twin when the inlier fraction ``>= --match-frac-min``. Unmatched cells
   are the **difference region**. Optional ``--geom-check`` RANSAC-rejects pose
   twins and estimates the crop offset.
5. **Discriminator** (``--tag`` / ``--tag-any`` / ``--region`` / ``--signal``):
   keep pairs where the attribute is present in **exactly one** member.
6. **Rank by edit-cleanliness**: fewest *other* differences first.
7. **Output**: a materialized ``_tags`` / ``_no_tags`` pair tree (the
   training-shaped output), plus a ready-to-use EasyControl dataset config
   under ``configs/easycontrol/``.

Run from the repo root::

    python -m easycontrol_adapters.tools.near_twins \
        --tag-any "speech bubble,thought bubble,blank speech bubble" \
        --artists ama_mitsuki

    # tagless visual attribute (recommended for bubbles on an untagged tree):
    python -m easycontrol_adapters.tools.near_twin --region \
        --artists ama_mitsuki

Features are cached, so the intended loop is: run → inspect the exported pair
tree → adjust ``--sim-min`` / ``--match-frac-min`` / ``--cell-match-min`` /
``--max-extra-diff`` → re-run (seconds).

Algorithm core lives in ``near_twin.engine``; rendering/export in
``near_twin.outputs``. This module holds the ``[staging]`` config layering, the
argparse surface, and the run orchestration.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
from pathlib import Path

import torch

try:
    from dotenv import load_dotenv  # picks up CAPTION_CORPUS_DIR from anima_lora/.env
except ImportError:  # dotenv is a soft dependency — env vars still work without it

    def load_dotenv(*_a, **_k):  # type: ignore
        return False


# Run from the repo root; `library` is installed editable (`uv sync`).
from library.vision import load_pe_encoder

from .engine import (
    PairRecord,
    embed_members,
    gather_members,
    normalize_tag,
    prune_for_pairing,
    run_artist,
    select_identity_members,
)
from .outputs import export_identity_members, export_pairs, write_dataset_config


# ---------------------------------------------------------------------------- config ([staging])


def expand_path(s: str) -> str:
    """Expand ``{VAR}`` / ``${VAR}`` / ``$VAR`` / ``~`` in a path string.

    ``{CAPTION_CORPUS_DIR}`` and ``${CAPTION_CORPUS_DIR}`` both resolve from the
    environment (loaded from ``anima_lora/.env``), so the toml can reference the
    corpus root without hardcoding an absolute path.
    """
    s = re.sub(r"\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), s)
    return os.path.expanduser(os.path.expandvars(s))


# toml [staging] keys that name a comma-joined list flag (accept a TOML array too).
_LIST_FLAGS = {"image_dirs": True, "tag_any": False}  # value→expand-as-path?


def _explicit_dests(argv: list[str]) -> set[str]:
    """dest names the user passed explicitly on the CLI (so they override toml)."""
    return {
        tok[2:].split("=", 1)[0].replace("-", "_")
        for tok in argv
        if tok.startswith("--")
    }


def apply_staging_config(args: argparse.Namespace, argv: list[str]) -> None:
    """Layer a ``[staging]`` table from ``--config`` under the CLI.

    Precedence: explicit CLI flag > ``[staging]`` toml > argparse default. Keys
    mirror the flag dest names (e.g. ``tag_any``, ``sim_min``, ``match_frac_min``,
    ``image_dirs``). List-valued keys (``image_dirs``, ``tag_any``) accept a TOML
    array and are stored as the comma-joined string the flags already parse. The
    legacy table name ``[miner]`` is still honored as a fallback.
    """
    if not args.config:
        return
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        return
    doc = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    explicit = _explicit_dests(argv)
    # Top-level `name` slug (outside any table) layered under --name.
    if doc.get("name") is not None and "name" not in explicit:
        args.name = doc["name"]
    table = doc.get("staging") or doc.get("miner")
    if not table:
        return
    for key, val in table.items():
        dest = key.replace("-", "_")
        if dest in explicit:
            continue  # CLI wins
        if not hasattr(args, dest):
            print(
                f"  [warn] unknown [staging] key {key!r} in {cfg_path}", file=sys.stderr
            )
            continue
        if dest in _LIST_FLAGS:
            items = val if isinstance(val, (list, tuple)) else [val]
            if _LIST_FLAGS[dest]:  # path list → expand each
                items = [expand_path(str(x)) for x in items]
            setattr(args, dest, ",".join(str(x) for x in items))
        else:
            setattr(args, dest, val)


# ---------------------------------------------------------------------------- CLI


def _default_image_dirs() -> str:
    """Default source trees: ``$CAPTION_CORPUS_DIR/{retrieved,selected}`` when the
    corpus root is set (the Anima-Tagger convention), else ``~/gelcrawl/…``."""
    root = os.environ.get("CAPTION_CORPUS_DIR")
    base = Path(root) if root else Path.home() / "gelcrawl"
    return f"{base / 'retrieved'},{base / 'selected'}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m easycontrol_adapters.tools.near_twins",
        description="Mine in-artist near-twin variant pairs that differ by one attribute.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="configs/easycontrol/near_twins.toml",
        help="toml with a [staging] table of run knobs (CLI flags override it; '' disables)",
    )
    p.add_argument(
        "--name",
        default=None,
        help="output slug — reroutes the staging/resized/cache/cond tree (and the "
        "generated blueprint paths) under post_image_dataset/easycontrol/<name>/. "
        "Defaults to the config's top-level `name` key, else 'near_twins'. "
        "An explicit --export-dir overrides this.",
    )
    disc = p.add_mutually_exclusive_group()
    disc.add_argument(
        "--tag", help="discriminator: this tag present in exactly one member"
    )
    disc.add_argument(
        "--tag-any", help="discriminator: any of these comma-separated synonyms"
    )
    disc.add_argument(
        "--region",
        action="store_true",
        help="discriminator: Stage-B diff region (tagless)",
    )
    disc.add_argument(
        "--signal", choices=["mit_text"], help="discriminator: per-image scalar gap"
    )

    p.add_argument(
        "--image-dirs",
        default=_default_image_dirs(),
        help="comma-separated source trees (<dir>/<artist>/<id>.<ext>); "
        "supports {CAPTION_CORPUS_DIR}/$VARS/~ expansion",
    )
    p.add_argument("--artists", help="comma-separated artist allowlist (default: all)")
    p.add_argument(
        "--per-artist-topk",
        type=int,
        default=0,
        help="keep only the top-K pairs per artist (0=all)",
    )

    p.add_argument(
        "--sim-min",
        type=float,
        default=0.85,
        help="Stage-A CLS-cosine prefilter threshold",
    )
    p.add_argument(
        "--grid", type=int, default=7, help="Stage-B pooled grid edge (G×G cells)"
    )
    p.add_argument(
        "--cell-match-min",
        type=float,
        default=0.9,
        help="per-cell cosine for an inlier match",
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="ratio-test distinctiveness (lower = stricter)",
    )
    p.add_argument(
        "--match-frac-min",
        type=float,
        default=0.5,
        help="inlier fraction to call a near-twin",
    )
    p.add_argument(
        "--geom-check",
        action="store_true",
        help="RANSAC translation consistency (reject pose twins)",
    )

    p.add_argument(
        "--rest-jaccard-min",
        type=float,
        default=0.0,
        help="'same scene' rest-tag overlap floor (0=off)",
    )
    p.add_argument(
        "--max-extra-diff",
        type=int,
        default=6,
        help="cap on non-target differing tags (-1=off)",
    )
    p.add_argument(
        "--signal-delta",
        type=float,
        default=0.04,
        help="signal-mode gap / low-side threshold",
    )
    p.add_argument(
        "--region-min-frac",
        type=float,
        default=0.02,
        help="region mode: min diff-blob cell fraction",
    )
    p.add_argument(
        "--region-max-frac",
        type=float,
        default=0.4,
        help="region mode: max diff-blob cell fraction",
    )
    p.add_argument(
        "--region-scatter-max",
        type=int,
        default=4,
        help="region mode: max diff cells outside main blob",
    )
    p.add_argument(
        "--id-window", type=int, default=0, help="only pair posts within N ids (0=off)"
    )

    p.add_argument(
        "--add-identity-pairs",
        type=float,
        default=0.0,
        help="fraction of the final tree to fill with identity (cond==target) pairs "
        "— clean singles carrying no target tag, staged so the adapter learns the "
        "no-op and stops globally transforming clean inputs (0=off, e.g. 0.25). "
        "Sized as a fraction of the total, so n_identity = round(n_pairs·f/(1−f)).",
    )
    p.add_argument(
        "--identity-saturation-min",
        type=float,
        default=0.0,
        help="only draw identity images with mean HSV saturation >= this (0..1, "
        "0=off); biases the identity set vivid to counter the mined-twin "
        "desaturation bias",
    )
    p.add_argument(
        "--identity-seed",
        type=int,
        default=0,
        help="RNG seed for identity-image sampling (run-to-run idempotency)",
    )
    p.add_argument(
        "--export-dir",
        default="post_image_dataset/easycontrol/near_twins/staging",
        help="materialized _tags/_no_tags pair tree (empty string disables)",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="copy images into the export tree (default: symlink)",
    )
    p.add_argument(
        "--emit-mask",
        action="store_true",
        help="also write the Stage-B diff-region mask per pair",
    )
    p.add_argument(
        "--config-out",
        default="configs/easycontrol/near_twins.toml",
        help="dataset blueprint written for the export tree (empty string disables)",
    )
    p.add_argument(
        "--batch-size", type=int, default=16, help="PE-Spatial embed batch size"
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for image decode/resize",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # CAPTION_CORPUS_DIR etc. from anima_lora/.env, before defaults resolve
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    apply_staging_config(args, argv)  # name + [staging] toml under the CLI
    # The `name` slug reroutes the export tree (and thus the blueprint's resized/
    # cache/cond paths, derived from export_dir.parent) under easycontrol/<name>/.
    # Skip when the user pinned --export-dir explicitly, or disabled it ('').
    if "export_dir" not in _explicit_dests(argv) and args.export_dir:
        name = str(args.name or "near_twins").strip()
        args.export_dir = f"post_image_dataset/easycontrol/{name}/staging"
    args.mode = "region" if args.region else "signal" if args.signal else "tag"

    target_tags: set[str] = set()
    if args.mode == "tag":
        raw = args.tag_any or args.tag
        if not raw:
            print(
                "error: tag mode needs --tag or --tag-any (or use --region / --signal)",
                file=sys.stderr,
            )
            return 2
        target_tags = {normalize_tag(t) for t in raw.split(",") if t.strip()}

    image_dirs = [Path(expand_path(d)) for d in args.image_dirs.split(",") if d.strip()]
    artists_filter = (
        {a.strip() for a in args.artists.split(",")} if args.artists else None
    )

    print(f"Gathering members from {len(image_dirs)} dir(s)…", file=sys.stderr)
    by_artist = gather_members(image_dirs, artists_filter)
    total = sum(len(v) for v in by_artist.values())
    print(f"  {len(by_artist)} artist(s), {total} unique images", file=sys.stderr)
    if not total:
        print("No images found — check --image-dirs / --artists.", file=sys.stderr)
        return 1

    # Prune to embeddable candidates BEFORE loading the encoder: same-size gate
    # for region/signal, same-size + tag pivot for tag mode. If nothing survives
    # (e.g. no size group holds both a tagged and an untagged member), we skip the
    # GPU load entirely.
    gate = (
        f"same-size + '{', '.join(sorted(target_tags))}' tag pivot"
        if args.mode == "tag"
        else "exact-(W×H) same-size"
    )
    pairable: dict[str, list] = {}
    for artist, members in by_artist.items():
        kept = prune_for_pairing(members, args.mode, target_tags)
        if len(kept) >= 2:
            pairable[artist] = kept
    n_pairable = sum(len(v) for v in pairable.values())
    print(f"Gate ({gate}): {n_pairable}/{total} image(s) embeddable.", file=sys.stderr)
    if not pairable:
        print("No candidate pairs after the gate — nothing to embed.", file=sys.stderr)
        return 0

    device = torch.device(args.device)
    print(f"Loading PE-Spatial-B16-512 on {device}…", file=sys.stderr)
    bundle = load_pe_encoder(device, name="pe_spatial")

    all_pairs: list[PairRecord] = []
    for artist, members in pairable.items():
        feats = embed_members(bundle, members, args.batch_size, args.num_workers)
        pairs = run_artist(artist, members, feats, args, target_tags)
        if pairs:
            print(f"  {artist}: {len(pairs)} pair(s)", file=sys.stderr)
        all_pairs.extend(pairs)

    all_pairs.sort(key=lambda p: (p.verdict.n_extra_diff, -p.cosine))
    print(f"\n{len(all_pairs)} accepted pair(s) total", file=sys.stderr)

    if args.export_dir and all_pairs:
        export_dir = Path(args.export_dir)
        n = export_pairs(
            all_pairs, export_dir, copy=args.copy, emit_mask=args.emit_mask
        )
        print(
            f"  pairs → {export_dir}/  ({n} pair tree(s), {'copied' if args.copy else 'symlinked'})",
            file=sys.stderr,
        )
        _export_identity(args, export_dir, by_artist, all_pairs, target_tags)
        if args.config_out:
            write_dataset_config(export_dir, Path(args.config_out))
            print(f"  cfg   → {args.config_out}", file=sys.stderr)

    return 0


def _export_identity(
    args: argparse.Namespace,
    export_dir: Path,
    by_artist: dict[str, list],
    all_pairs: list[PairRecord],
    target_tags: set[str],
) -> None:
    """Stage the identity (cond==target) no-op regularizer alongside the mined pairs.

    Count is a fraction of the final tree (``n = round(n_pairs·f/(1−f))``), drawn
    from clean singles with no target tag (optionally saturation-floored). No-op
    when ``--add-identity-pairs`` is 0; needs ≥1 mined pair to size against.
    """
    f = float(args.add_identity_pairs)
    if f <= 0.0:
        return
    if not 0.0 < f < 1.0:
        print(
            f"  [warn] --add-identity-pairs must be in (0, 1); got {f} — skipping",
            file=sys.stderr,
        )
        return
    n_id = round(len(all_pairs) * f / (1.0 - f))
    if n_id <= 0:
        return
    exclude = {m.stem for p in all_pairs for m in (p.holder_member(), p.clean_member())}
    members = select_identity_members(
        by_artist,
        target_tags,
        n_id,
        saturation_min=float(args.identity_saturation_min),
        seed=int(args.identity_seed),
        exclude_stems=exclude,
    )
    written = export_identity_members(members, export_dir, copy=args.copy)
    sat = float(args.identity_saturation_min)
    print(
        f"  identity → {export_dir}/  ({written}/{n_id} cond==target pair(s)"
        + (f", sat≥{sat:g}" if sat > 0 else "")
        + ")"
        + ("  [pool exhausted before target]" if written < n_id else ""),
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
