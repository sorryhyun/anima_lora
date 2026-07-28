"""Aligned-pair instruction-edit staging for the EasyControl *twin_edit* arm.

Consumes the Phase-0 pair census manifest
(``project/directedit_ec/bench/run_pair_census.py`` →
``pairs_manifest.json``) and materializes the training tree the request doc
(`project/directedit_ec/easycontrol_request.md`) specifies:

  * **slice-balanced selection** — pairs are drawn per discriminator slice in
    the doc's priority order (text_bubbles → clothing_state → expression →
    object), ranked by edit-cleanliness (fewest extra tag diffs, then CLS
    cosine), deduped across slices, capped per slice, and filtered to the
    ``[min_delta, max_delta]`` instruction-length band;
  * **direction doubling** — every mined twin yields TWO examples (A→B and
    B→A) with mirrored delta captions;
  * **delta captions** — the census already computed them per the
    ``subject_edit_pairs`` contract (additions in target order +
    `-`-prefixed removals); staged as REAL ``.txt`` files;
  * **identity no-op pairs** — cond == target with an EMPTY instruction,
    sized as a fraction of the final tree, saturation-floored (the mined
    twins skew 96.5% pale — census, 2026-07-26).

Output is a near_twins-shaped ``_tags`` / ``_no_tags`` pair tree (cond =
``_tags``, target = ``_no_tags``), so ``make easycontrol-preprocess
EASYADAPTER=twin_edit`` reuses the near_twins resize → VAE/TE → cond-pairing
pass unchanged. Unlike subject_edit there is no separate text cache: the
delta caption IS the staged target caption, so the TE pass encodes the right
text into ``cache/`` directly.

Pipeline:
    make easycontrol-staging    EASYADAPTER=twin_edit   # this tool (CPU, symlinks)
    make easycontrol-preprocess EASYADAPTER=twin_edit   # resize + VAE/TE + cond/
    make easycontrol            EASYADAPTER=twin_edit   # train
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from easycontrol_adapters.tools.near_twins.engine import (  # noqa: E402
    select_identity_members,
)
from easycontrol_adapters.tools.near_twins.__main__ import expand_path  # noqa: E402
from easycontrol_adapters.tools.near_twins.outputs import (  # noqa: E402
    _BLUEPRINT_SENTINEL,
    _strip_blueprint,
)
from easycontrol_adapters.tools.subject_pairs import load_staging_config  # noqa: E402
from library.vision.pe_features import Member, gather_members  # noqa: E402

DEFAULT_CONFIG = "configs/easycontrol/twin_edit.toml"

# Request-doc priority order; earlier slices claim shared pairs first.
SLICE_PRIORITY = ("text_bubbles", "clothing_state", "expression", "object")


def _blueprint_text() -> str:
    return f"""{_BLUEPRINT_SENTINEL}
# Aligned near-twin pairs wired as an EasyControl *instruction-edit* control
# task. staging/ holds direction-doubled `{{pair}}-f|r_tags` / `_no_tags`
# couples: `_tags` = the edit SOURCE (cond), `_no_tags` = the edit TARGET,
# whose .txt is the TAG-DELTA instruction (additions + `-`-prefixed removals;
# identity pairs carry an EMPTY instruction). The preprocess pass resizes into
# `resized/`, VAE/TE-encodes into `cache/` (the delta caption is the staged
# caption, so no separate text cache), and symlinks each `_tags` latent into
# `cond/` under its `_no_tags` stem.
#
# Roles (EasyControl: cache_dir = denoising target, cond_cache_dir = condition):
#   target = the `_no_tags` member; prompt = the delta instruction
#   cond   = the paired `_tags` latent (the pre-edit image)
# So the adapter learns: given THIS image and THIS instruction, apply exactly
# the instructed changes in place. `{{name}}` interpolates at train time.

[general]
caption_extension = '.txt'

[[datasets]]
batch_size = 1

  [[datasets.subsets]]
  image_dir = 'post_image_dataset/easycontrol/{{name}}/resized'
  cache_dir = 'post_image_dataset/easycontrol/{{name}}/cache'        # denoising target: the _no_tags members (delta captions)
  cond_cache_dir = 'post_image_dataset/easycontrol/{{name}}/cond'    # condition: the paired _tags latent (keyed by the _no_tags stem)
  path_pattern = '*_no_tags.*'     # targets only; the _tags twin rides in as the cond
  recursive = true                 # tree is nested <artist>/<pair_id>_{{tags,no_tags}}
  flip_aug = false                 # cond latents have no flipped variant
  num_repeats = 1
"""


def write_dataset_config(config_path: Path) -> None:
    if not config_path.is_file():
        raise SystemExit(
            f"{config_path} not found — the twin_edit descriptor ships with the "
            "repo (configs/easycontrol/twin_edit.toml); restore it first."
        )
    head = _strip_blueprint(config_path.read_text(encoding="utf-8"))
    config_path.write_text(f"{head}\n\n{_blueprint_text()}", encoding="utf-8")


def select_pairs(pairs: list[dict], args) -> tuple[list[dict], dict[str, int]]:
    """Slice-balanced, deduped, delta-band-filtered selection."""
    taken: dict[tuple[str, str], dict] = {}
    per_slice: dict[str, int] = {}
    for name in SLICE_PRIORITY:
        cands = [
            p
            for p in pairs
            if name in p["slices"]
            and (p["a"], p["b"]) not in taken
            and args.min_delta <= p["sym_delta"] <= args.max_delta
        ]
        cands.sort(key=lambda p: (p["slices"][name]["extra"], -p["cosine"]))
        picked = cands[: args.slice_cap] if args.slice_cap else cands
        for p in picked:
            taken[(p["a"], p["b"])] = {**p, "slice": name}
        per_slice[name] = len(picked)
    return list(taken.values()), per_slice


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def export_pairs(
    selected: list[dict], members: dict[tuple[str, str], Member], staging: Path
) -> tuple[int, list[dict]]:
    """Direction-doubled `_tags`/`_no_tags` export. Returns (n_staged, dropped)."""
    staged = 0
    dropped: list[dict] = []
    for p in selected:
        ma = members.get((p["artist"], p["a"]))
        mb = members.get((p["artist"], p["b"]))
        if ma is None or mb is None:
            dropped.append(p)
            continue
        adir = staging / p["artist"]
        pid = f"{p['a']}-{p['b']}"
        for d, cond, target, delta in (
            ("f", ma, mb, p["delta_a_to_b"]),
            ("r", mb, ma, p["delta_b_to_a"]),
        ):
            _symlink(cond.image_path, adir / f"{pid}-{d}_tags{cond.image_path.suffix}")
            # cond captions are never read as targets (path_pattern *_no_tags.*);
            # blank keeps the TE pass cheap and unambiguous.
            (adir / f"{pid}-{d}_tags.txt").write_text("", encoding="utf-8")
            _symlink(
                target.image_path, adir / f"{pid}-{d}_no_tags{target.image_path.suffix}"
            )
            (adir / f"{pid}-{d}_no_tags.txt").write_text(delta, encoding="utf-8")
            staged += 1
    return staged, dropped


def export_identity(members: list[Member], staging: Path) -> int:
    """cond == target, EMPTY instruction (the no-op anchor; request doc §3)."""
    for m in members:
        adir = staging / m.artist
        pid = f"id_{m.stem}"
        for side in ("_tags", "_no_tags"):
            _symlink(m.image_path, adir / f"{pid}{side}{m.image_path.suffix}")
            (adir / f"{pid}{side}.txt").write_text("", encoding="utf-8")
    return len(members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--config-out", dest="config_out", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--manifest",
        default=None,
        help="census pairs_manifest.json (from run_pair_census.py)",
    )
    parser.add_argument(
        "--image_dirs",
        default="{CAPTION_CORPUS_DIR}/retrieved",
        help="comma-separated source trees (must match the census run)",
    )
    parser.add_argument(
        "--slice_cap",
        type=int,
        default=250,
        help="max pairs per slice, priority order claims shared pairs (0=all)",
    )
    parser.add_argument(
        "--min_delta", type=int, default=1, help="drop pairs below this tag delta"
    )
    parser.add_argument(
        "--max_delta",
        type=int,
        default=24,
        help="drop pairs above this tag delta (instruction-length cap)",
    )
    parser.add_argument(
        "--identity_fraction",
        type=float,
        default=0.25,
        help="identity (no-op) share of the FINAL tree",
    )
    parser.add_argument(
        "--identity_saturation_min",
        type=float,
        default=0.4,
        help="identity draws need mean HSV saturation >= this (census: mined "
        "members are 96.5%% below 0.4)",
    )
    parser.add_argument("--seed", type=int, default=42)

    config_arg = None
    argv = sys.argv[1:]
    if "--config" in argv:
        config_arg = argv[argv.index("--config") + 1]
    config_path = REPO_ROOT / (config_arg or DEFAULT_CONFIG)
    name, staging_cfg = load_staging_config(config_path)
    known = {a.dest for a in parser._actions}
    parser.set_defaults(
        name=name, **{k: v for k, v in staging_cfg.items() if k in known}
    )
    args = parser.parse_args()

    if not args.manifest:
        raise SystemExit(
            "[twin_edit] no census manifest — set [staging].manifest in the "
            "descriptor (run project/directedit_ec/bench/run_pair_census.py first)."
        )
    manifest_path = REPO_ROOT / args.manifest
    if not manifest_path.is_file():
        raise SystemExit(f"[twin_edit] manifest not found: {manifest_path}")
    pairs = json.loads(manifest_path.read_text(encoding="utf-8"))["pairs"]

    selected, per_slice = select_pairs(pairs, args)
    if not selected:
        raise SystemExit("[twin_edit] selection is empty — check the delta band.")

    image_dirs = [
        Path(expand_path(d))
        for d in (
            args.image_dirs
            if isinstance(args.image_dirs, str)
            else ",".join(args.image_dirs)
        ).split(",")
        if d.strip()
    ]
    by_artist = gather_members(image_dirs, None)
    members = {(a, m.stem): m for a, ms in by_artist.items() for m in ms}

    base = REPO_ROOT / "post_image_dataset" / "easycontrol" / args.name
    staging = base / "staging"
    if staging.exists():
        shutil.rmtree(staging)

    staged, dropped = export_pairs(selected, members, staging)
    if dropped:
        print(
            f"[twin_edit] {len(dropped)} selected pairs dropped — source image "
            "no longer under --image_dirs.",
            file=sys.stderr,
        )

    f = args.identity_fraction
    n_id = round(staged * f / (1.0 - f)) if 0.0 < f < 1.0 else 0
    exclude = {
        s for p in selected for s in (p["a"], p["b"]) if (p["artist"], s) in members
    }
    identity = select_identity_members(
        by_artist,
        set(),  # no attribute constraint — any image is a valid no-op case
        n_id,
        saturation_min=args.identity_saturation_min,
        seed=args.seed,
        exclude_stems=exclude,
    )
    n_written = export_identity(identity, staging)

    (base / "twin_edit_pairs.json").write_text(
        json.dumps(
            {
                "meta": {
                    "miner": "twin_edit_pairs",
                    "manifest": args.manifest,
                    "slice_cap": args.slice_cap,
                    "min_delta": args.min_delta,
                    "max_delta": args.max_delta,
                    "identity_fraction": args.identity_fraction,
                    "identity_saturation_min": args.identity_saturation_min,
                    "seed": args.seed,
                    "per_slice": per_slice,
                    "n_twins": len(selected) - len(dropped),
                    "n_examples": staged,
                    "n_identity": n_written,
                },
                "pairs": selected,
                "identity": [f"{m.artist}/{m.stem}" for m in identity],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    out = args.config_out
    write_dataset_config(REPO_ROOT / out if out else config_path)
    print(
        f"[twin_edit] {len(selected) - len(dropped)} twins "
        f"({', '.join(f'{k}={v}' for k, v in per_slice.items())}) → "
        f"{staged} direction-doubled examples + {n_written}/{n_id} identity "
        f"(sat≥{args.identity_saturation_min:g}) → {staging}"
    )


if __name__ == "__main__":
    main()
