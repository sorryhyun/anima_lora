#!/usr/bin/env python3
"""`make soup` orchestrator: uncond inter-train → seeded fine-tunes → SVD soup.

Ships the recipe validated in ``bench/memorization/report.md`` (the uncond-init
ladder) as one pipeline:

1. **Uncond inter-train** (Self-Soupervision, arXiv:2602.02890): a short
   ``caption_dropout_rate 1.0`` run on a *diluted* pool. The pool is selected by
   ``--pool_path_pattern`` (default ``*`` = the whole dataset) and diluted by
   ``--uncond_ratio`` (train.py's ``--sample_ratio``); the fine-tune selection
   (``--path_pattern``) is always unioned into it. Dose rule from the report:
   uncond member exposure = epochs × target share — 2 epochs at a minority share
   is memorization-neutral while buying the quality win; same-frame full-dose
   uncond is destructive, which is why the pool must be *broader* than the
   fine-tune set. The checkpoint name is deterministic in (pool, ratio, epochs),
   so it is trained **once** and reused across any soup drawing the same pool.
2. **Seeded fine-tunes**: N normal captioned runs on the ``--path_pattern``
   images, ``--network_weights``-initialized from the uncond checkpoint (or,
   with ``--no_uncond``, straight from the method config's ``down_init`` — Phase
   1 is skipped entirely; see :func:`init_tag`). Souping
   absorbs the training-seed lottery (a catastrophic draw is invisible to
   loss/AUC — report Act 5). Seed is the only axis that varies by default;
   ``--lr_pool`` / ``--lr_interval`` add per-ingredient learning-rate diversity
   (the model-soup recipe's other axis — see ``resolve_lrs`` for the caveat that
   we have no greedy-selection gate to protect a uniform average from a bad draw).
3. **ΔW soup, SVD-truncated to the ingredient rank** (``scripts/soup/build.py``)
   so the artifact stays single-adapter-sized (~99.9% retained energy on
   shared-init ingredients). The first ingredient's ``.snapshot.toml`` is
   copied next to the soup — the memorization probes replay membership from it.

Selection is by fnmatch **path_pattern** (``|``-separated alternatives, matched
against each image's path relative to its subset ``image_dir``) rather than a
single artist directory — so a soup can span any glob-expressible slice of the
dataset, not just one artist. ``--artists_shard k_N`` is the alternative
selector: it expands to one round-robin shard of the artist subdirectories
(``<artist>/*|…``) before anything else runs, so the pool union / slug / Phase-2
flags all see a plain glob. The output slug derives from the selection (or an
explicit ``--name``).

Runs ``train.py`` as direct subprocesses (single daemon command job — never
submit nested daemon jobs from here; that deadlocks the serial queue).
Unknown CLI args are forwarded to the fine-tune runs (that's how
``make soup ARGS="--network_dim 32"`` reaches Phase 2). The ``--sigma_lowres*``
family is the one exception: it is a whole-pipeline data-routing knob, so it is
*also* replayed onto Phase 1 and folded into the uncond checkpoint name.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `python scripts/soup/pipeline.py`

from scripts.tasks._common import build_launch_cmd, build_method_args, run  # noqa: E402

UNCOND_SEED = 1000  # fixed: the uncond init is a shared, reusable artifact


_DEFAULT_CONFIG = ROOT / "configs" / "soup" / "soup.toml"

# The σ-demote routing surface (library/config/cli_args.py). These change what
# the uncond weights become, so they are both forwarded to Phase 1 and folded
# into its checkpoint name — see sigma_settings() / sigma_argv() / uncond_name().
_SIGMA_FLOAT_KEYS = (
    "sigma_lowres_threshold",
    "sigma_lowres_threshold_max",
    "sigma_lowres_threshold2",
    "sigma_lowres_threshold2_max",
)
_SIGMA_STR_KEYS = (
    "sigma_lowres_route",
    "sigma_lowres_route2",
    "sigma_lowres_span",
    "sigma_lowres_span2",
)
_YARNSIG_DEFAULT = "1,4,0.35,2"  # train.py's --sigma_lowres_yarnsig const


def _resolve_config(argv: list[str]) -> Path:
    """Pre-scan argv for ``--config <file>`` (needed before argparse, since the
    parser's defaults are seeded from the config). Returns the resolved path,
    falling back to ``configs/soup/soup.toml``. The value is resolved relative
    to the repo root (so ``configs/soup/custom/soup.toml`` works) then as an
    absolute path; a bare stem falls back to ``configs/soup/custom/<stem>.toml``.
    """
    val = None
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            val = argv[i + 1]
        elif a.startswith("--config="):
            val = a.split("=", 1)[1]
    if not val:
        return _DEFAULT_CONFIG
    stem = val if val.endswith(".toml") else f"{val}.toml"
    for cand in (
        ROOT / stem,
        Path(stem),
        ROOT / "configs" / "gui-methods" / "custom" / stem,
    ):
        if cand.exists():
            return cand
    raise SystemExit(f"--config: no such file {val!r} (looked under {ROOT}).")


def _method_for_config(config: Path) -> tuple[str, str | None]:
    """``(method, methods_subdir)`` for a soup config, so the fine-tune train.py
    runs merge the SAME toml that seeds the pipeline knobs. The default
    ``configs/soup/soup.toml`` → ``("soup", None)`` (per-method-dir resolution);
    a custom ``configs/<a>/<b>/<stem>.toml`` → ``(stem, "<a>/<b>")``."""
    if config.resolve() == _DEFAULT_CONFIG.resolve():
        return "soup", None
    try:
        rel = config.resolve().relative_to((ROOT / "configs").resolve())
    except ValueError:
        raise SystemExit(f"--config must live under configs/ (got {config}).")
    return config.stem, str(rel.parent).replace("\\", "/")


def _soup_toml(path: Path | None = None) -> dict:
    """The whole soup config toml as a dict (``{}`` if missing). Both the
    ``[soup]`` pipeline-knob table and the top-level flat scalars (incl. the
    default ``path_pattern`` fine-tune glob) are seeded from here. ``path``
    defaults to ``configs/soup/soup.toml`` — a custom ``--config`` overrides it."""
    import toml

    path = path or _DEFAULT_CONFIG
    if not path.exists():
        return {}
    return toml.load(path)


def _soup_defaults() -> dict:
    """The ``[soup]`` table from configs/soup/soup.toml — pipeline knob defaults
    (pool / dose / seeds / rank). Stripped from the train.py merge (registered as
    a metadata section), so it lives here purely to seed this argparser. Missing
    file / table → ``{}`` (argparse hard-coded fallbacks below still apply)."""
    return _soup_toml().get("soup", {})


def pool_glob(pool_pattern: str, ft_pattern: str) -> str:
    """The Phase-1 uncond pool glob: ``pool_pattern`` with the fine-tune
    selection unioned in (duplicate ``|`` alternatives dropped so an image
    matched by both patterns is still kept once). ``*`` (match everything)
    already covers the fine-tune set, so it is returned as-is."""
    pool_pattern = (pool_pattern or "*").strip()
    if pool_pattern in ("", "*"):
        return "*"
    alts = [a for a in pool_pattern.split("|") if a]
    for a in ft_pattern.split("|"):
        if a and a not in alts:
            alts.append(a)
    return "|".join(alts)


def expand_artists_shard(
    spec: str, method: str, preset: str, methods_subdir: str | None
) -> str:
    """Expand ``--artists_shard k_N`` into an explicit ``<artist>/*|…`` glob.

    train.py expands the shard *inside* its own blueprint build, which is too
    late for us — the pipeline needs the concrete selection up front to union it
    into the Phase-1 uncond pool, derive the output slug, and hand Phase 2 a
    plain ``--path_pattern``. So we replay the same expansion here against the
    same blueprint (``library.datasets.artist_shard.apply_artist_shard``) and
    then hand train.py the resulting glob with ``--artists_shard ""``, keeping
    exactly one selection mechanism live per run.
    """
    from library.config.io import load_dataset_config_from_base, load_method_preset
    from library.datasets.artist_shard import apply_artist_shard, parse_shard
    from library.env import resolve_under_home

    parse_shard(spec)  # fail fast on a malformed 'k_N' before any I/O
    subdir = methods_subdir or "methods"
    merged = load_method_preset(method, preset, methods_subdir=subdir)
    user_config = load_dataset_config_from_base(
        method=method, methods_subdir=subdir, overrides=merged
    )
    if not user_config:
        raise SystemExit(
            f"--artists_shard {spec!r}: no dataset blueprint found for method "
            f"{method!r} — cannot enumerate artist directories."
        )
    apply_artist_shard(user_config, spec, resolve=resolve_under_home)
    alts: list[str] = []
    for ds in user_config.get("datasets", []):
        for sub in ds.get("subsets", []):
            for a in (sub.get("path_pattern") or "").split("|"):
                if a and a not in alts:
                    alts.append(a)
    if not alts:
        raise SystemExit(
            f"--artists_shard {spec!r}: no artist subdirectories found under the "
            f"blueprint's image_dir(s) — nothing to select."
        )
    return "|".join(alts)


def slug_for_shard(spec: str) -> str:
    """Output slug for a shard selection: ``1_6`` → ``shard1of6`` (the artist
    glob it expands to is far too long to name a checkpoint after)."""
    k, n = spec.strip().split("_")
    return f"shard{int(k)}of{int(n)}"


def slug_for_pattern(pattern: str) -> str:
    """A filename-safe slug for a path_pattern. A plain ``<dir>/*`` (single
    alternative, no other glob metachars) → the dir's basename, so the common
    per-artist case stays ``anima_soup_<artist>``; anything richer → a sanitized
    prefix plus a short hash of the full pattern for uniqueness."""
    pattern = pattern.strip()
    m = re.fullmatch(r"([^|*?\[\]]+)/\*", pattern)
    if m:
        base = m.group(1).rstrip("/").split("/")[-1]
        slug = re.sub(r"[^0-9A-Za-z._-]+", "_", base).strip("_")
        if slug:
            return slug
    slug = re.sub(r"[^0-9A-Za-z]+", "_", pattern).strip("_")[:32]
    digest = hashlib.sha1(pattern.encode()).hexdigest()[:6]
    return f"{slug}_{digest}" if slug else f"pat_{digest}"


def resolve_lrs(num: int, pool: str | None, interval: str | None) -> list[float] | None:
    """The per-ingredient learning rates, or ``None`` to leave every fine-tune on
    the method config's ``learning_rate`` (the default: a seed-only soup).

    ``pool`` is an explicit comma-separated list, cycled if shorter than ``num``;
    ``interval`` is ``"<lo>:<hi>"``, spread **geometrically** over ``num`` points
    (LR is a scale knob, so log-spacing is the natural sweep). The two are
    mutually exclusive.

    Hyperparameter-diverse ingredients are the model-soup recipe (Wortsman et
    al.), but there the pool is protected by *greedy* selection on a held-out
    metric. This soup is a uniform ΔW average with no such gate, so a bad LR draw
    is averaged in rather than dropped — keep the spread narrow.
    """
    if pool and interval:
        raise SystemExit("--lr_pool and --lr_interval are mutually exclusive.")
    if pool:
        try:
            lrs = [float(x) for x in pool.replace(" ", "").split(",") if x]
        except ValueError:
            raise SystemExit(f"--lr_pool: not a comma-separated float list: {pool!r}")
        if not lrs:
            raise SystemExit("--lr_pool: empty list.")
        return [lrs[i % len(lrs)] for i in range(num)]
    if interval:
        parts = [p for p in interval.replace(" ", "").split(":") if p]
        if len(parts) != 2:
            raise SystemExit(f"--lr_interval: expected '<lo>:<hi>', got {interval!r}")
        try:
            lo, hi = (float(p) for p in parts)
        except ValueError:
            raise SystemExit(f"--lr_interval: not two floats: {interval!r}")
        if lo <= 0 or hi <= 0:
            raise SystemExit("--lr_interval: both bounds must be > 0 (geometric).")
        if num == 1:
            return [lo]
        return [lo * (hi / lo) ** (i / (num - 1)) for i in range(num)]
    return None


def sigma_overrides(extra: list[str]) -> dict:
    """The ``--sigma_lowres*`` flags present in ARGS, as a dict.

    Absent flags are *omitted* rather than defaulted (``argparse.SUPPRESS``), so
    the result overlays the method config without clobbering it with argparse
    defaults. Needed because ARGS reaches only the fine-tunes — Phase 1 has to
    be handed the same routing explicitly (:func:`sigma_argv`)."""
    # allow_abbrev=False: ARGS is full of flags we don't model, and prefix
    # matching would let one of them be claimed as a sigma flag.
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    sup = argparse.SUPPRESS
    ap.add_argument("--sigma_lowres", action="store_true", default=sup)
    ap.add_argument(
        "--sigma_lowres_yarnsig", nargs="?", const=_YARNSIG_DEFAULT, default=sup
    )
    for k in _SIGMA_FLOAT_KEYS:
        ap.add_argument(f"--{k}", type=float, default=sup)
    for k in _SIGMA_STR_KEYS:
        ap.add_argument(f"--{k}", default=sup)
    known, _ = ap.parse_known_args(extra)
    return vars(known)


def sigma_settings(merged: dict, overrides: dict | None = None) -> dict:
    """The σ-demote routing the uncond run will actually train under: the merged
    method config overlaid with :func:`sigma_overrides`. ``{}`` when
    ``sigma_lowres`` is off — so a soup without it keeps the pre-existing uncond
    names untouched.

    Values that are ``None``/``""`` are dropped from both sources: an empty
    string is how a span is switched off, and train.py reads that identically to
    an absent key, so the two must not fork the name digest."""
    overrides = overrides or {}
    cfg: dict = {}
    for k in ("sigma_lowres", "sigma_lowres_yarnsig", *_SIGMA_STR_KEYS):
        v = overrides[k] if k in overrides else merged.get(k)
        if v is None or v == "":
            continue
        cfg[k] = v
    for k in _SIGMA_FLOAT_KEYS:
        v = overrides[k] if k in overrides else merged.get(k)
        if v is None or v == "":
            continue
        cfg[k] = float(v)  # a TOML int and a CLI float must digest the same
    if not cfg.get("sigma_lowres"):
        return {}
    # train.py turns yarnsig on at its operating point whenever --sigma_lowres is
    # set, so normalize that implicit default in — an explicit config copy of the
    # same value must not fork the name.
    cfg.setdefault("sigma_lowres_yarnsig", _YARNSIG_DEFAULT)
    return cfg


def sigma_argv(cfg: dict, merged: dict | None = None) -> list[str]:
    """``cfg`` rendered as train.py flags for the Phase-1 uncond run. Passing
    them explicitly (even when they only restate the method config) also makes
    the uncond snapshot self-describing.

    Phase 1 merges the same method config, so a key ARGS switched **off** has to
    be cleared explicitly — dropping the flag would leave the config's value
    live (the ``--artists_shard ""`` trick, same reason)."""
    if not cfg:
        return []
    argv = ["--sigma_lowres"]
    for k in sorted(cfg):
        if k != "sigma_lowres":
            argv += [f"--{k}", f"{cfg[k]}"]
    for k in sorted((*_SIGMA_STR_KEYS, "sigma_lowres_yarnsig")):
        if k not in cfg and (merged or {}).get(k):
            argv += [f"--{k}", ""]
    return argv


def uncond_name(pool: str, ratio: float, epochs: int, sigma: dict | None = None) -> str:
    """Deterministic uncond checkpoint name — same pool+dose → same artifact.

    ``sigma`` is the σ-demote routing the uncond run trains under (from
    :func:`sigma_settings`). It changes what the uncond weights *become*, so a
    non-empty one is folded in as an ``_sl<digest>`` tag: a sigma_lowres soup
    then can't silently reuse an init trained without it, nor vice versa. Empty
    / ``None`` (the default, sigma_lowres off) → the name is unchanged."""
    digest = hashlib.sha1(pool.encode()).hexdigest()[:8]
    ratio_tag = f"{ratio:g}".replace(".", "p")
    name = f"anima_uncond_{digest}_r{ratio_tag}_e{epochs}"
    if sigma:
        payload = ";".join(f"{k}={sigma[k]}" for k in sorted(sigma))
        name += "_sl" + hashlib.sha1(payload.encode()).hexdigest()[:6]
    return name


def resolve_uncond_init(ref: str) -> Path:
    """Resolve a pinned uncond-init reference to a checkpoint path. Accepts a
    bare name (``anima_uncond_...`` → ``output/ckpt/<name>.safetensors``), a
    filename, or an explicit/relative path. ``.safetensors`` is optional."""
    ref = ref.strip()
    p = Path(ref)
    if p.suffix != ".safetensors":
        p = p.parent / f"{p.name}.safetensors"
    for cand in (p, ROOT / p, ROOT / "output" / "ckpt" / p.name):
        if cand.is_file():
            return cand
    # Not found — return the canonical location for a clear error message.
    return ROOT / "output" / "ckpt" / p.name


def configured_svd_slice(merged: dict, ft_extra: list[str]) -> int:
    """The ``svd_slice`` the Phase-2 fine-tunes would build their ``A`` from:
    the merged method config's top-level key, overridden by a
    ``--network_args svd_slice=k`` in the forwarded ARGS. 0 = plain top-r."""
    slice_ = int(merged.get("svd_slice", 0) or 0)
    for i, a in enumerate(ft_extra):
        vals: list[str] = []
        if a == "--network_args":
            j = i + 1
            while j < len(ft_extra) and not ft_extra[j].startswith("-"):
                vals.append(ft_extra[j])
                j += 1
        elif a.startswith("--network_args="):
            vals = [a.split("=", 1)[1]]
        for v in vals:
            if v.startswith("svd_slice="):
                slice_ = int(v.split("=", 1)[1] or 0)
    return slice_


def init_tag(no_uncond: bool, svd_slice: int = 0) -> str:
    """Slug suffix that keeps a no-uncond soup (and each ``svd_slice`` window of
    it) from colliding with the uncond-init soup of the same selection:
    ``""`` for the shipped uncond path, ``_nouncond`` / ``_nouncond_k<slice>``
    otherwise. Applied to the derived slug only — an explicit ``--name`` is
    taken verbatim."""
    if not no_uncond:
        return ""
    return f"_nouncond_k{svd_slice}" if svd_slice else "_nouncond"


def check_init_mode(
    no_uncond: bool, uncond_init: str | None, svd_slice: int, ft_extra: list[str]
) -> None:
    """Refuse the init combinations that would silently do something else:
    ``--no_uncond`` with a pinned ``--uncond_init`` (contradiction), with a
    ``--network_weights`` in ARGS (the flag exists to *not* warm-start), and an
    ``svd_slice`` window under the uncond
    path, where ``--network_weights`` overwrites ``A`` with the uncond
    checkpoint's (slice-0) rows and the slice is silently lost."""
    if no_uncond and uncond_init:
        raise SystemExit("--no_uncond and --uncond_init are mutually exclusive.")
    if no_uncond and any(a.split("=", 1)[0] == "--network_weights" for a in ft_extra):
        raise SystemExit(
            "--no_uncond with --network_weights in ARGS: the flag means the "
            "fine-tunes start from down_init, not from a checkpoint. Drop one."
        )
    if svd_slice and not no_uncond:
        raise SystemExit(
            f"svd_slice={svd_slice} under the uncond-init path: the fine-tunes load "
            "the uncond checkpoint via --network_weights, which overwrites A with "
            "its slice-0 rows — the slice would be silently lost. Pass "
            "--no_uncond (NO_UNCOND=1 / [soup] no_uncond = true) to seed the "
            "ingredients from weight_svd + svd_slice directly."
        )


def _train(args_list: list[str], dry_run: bool) -> None:
    cmd = build_launch_cmd(*args_list)
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return
    run(cmd)


def main() -> None:
    config = _resolve_config(sys.argv[1:])
    method_default, methods_subdir = _method_for_config(config)
    cfg = _soup_toml(config)
    d = cfg.get("soup", {})
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        default=str(config),
        help="soup config toml (default configs/soup/soup.toml). A custom file "
        "under configs/<dir>/<stem>.toml seeds BOTH the [soup] pipeline knobs "
        "and the fine-tune method config (--method <stem> --methods_subdir <dir>). "
        "make soup CUSTOM=<file> resolves configs/gui-methods/custom/<file>.",
    )
    ap.add_argument(
        "--path_pattern",
        default=None,
        help="fnmatch glob (| = alternatives) selecting the fine-tune images, "
        "matched against each image's path relative to its subset image_dir. "
        "Replaces the old --target artist dir. Defaults to the top-level "
        "path_pattern in configs/soup/soup.toml when omitted.",
    )
    ap.add_argument(
        "--artists_shard",
        default=None,
        help="select the fine-tune images as one round-robin shard of the "
        "artist subdirectories instead of a glob, format 'k_N' (e.g. '1_6'). "
        "Expanded here into an explicit <artist>/*|… path_pattern, so it also "
        "drives the Phase-1 pool union and the output slug (anima_soup_"
        "shard1of6). Mutually exclusive with --path_pattern; a top-level "
        "artists_shard in the soup config is picked up as the default.",
    )
    ap.add_argument(
        "--pool_path_pattern",
        default=d.get("pool_path_pattern", "*"),
        help="Phase-1 uncond pool selection glob; default '*' (whole dataset), "
        "diluted by --uncond_ratio. The fine-tune set is always unioned in.",
    )
    ap.add_argument(
        "--name",
        default=None,
        help="output slug → output/ckpt/anima_soup_<name>.safetensors "
        "(default: derived from --path_pattern).",
    )
    ap.add_argument("--uncond_ratio", type=float, default=d.get("uncond_ratio", 0.5))
    ap.add_argument("--uncond_epochs", type=int, default=d.get("uncond_epochs", 2))
    ap.add_argument(
        "--uncond_init",
        default=d.get("uncond_init"),
        help="pin an explicit pre-trained uncond adapter to start the soup from "
        "(a bare name resolved under output/ckpt/, a filename, or a path). When "
        "set, Phase 1 is SKIPPED — pool / uncond_ratio / uncond_epochs no longer "
        "affect the init. Set uncond_init in the [soup] table to make it a default.",
    )
    ap.add_argument(
        "--no_uncond",
        action="store_true",
        default=bool(d.get("no_uncond", False)),
        help="skip Phase 1 entirely: the fine-tunes start from the method "
        "config's down_init (weight_svd [+ svd_slice]) instead of an uncond "
        "checkpoint. Required for svd_slice != 0 (a --network_weights init "
        "would overwrite the slice). The derived slug gets a _nouncond[_k<slice>] "
        "suffix. Set no_uncond = true in the [soup] table to make it a default.",
    )
    ap.add_argument(
        "--num_soup",
        type=int,
        default=d.get("num_soup", 3),
        help="number of seeded fine-tunes to soup (seeds are UNCOND_SEED+1..+N, "
        "i.e. 1001..1000+N). Must be >= 2.",
    )
    ap.add_argument(
        "--lr_pool",
        default=d.get("lr_pool"),
        help="per-ingredient learning rates, comma-separated (e.g. "
        "'1e-5,2e-5,4e-5'); cycled if shorter than --num_soup. Default: unset = "
        "every fine-tune uses the method config's learning_rate (seed-only soup).",
    )
    ap.add_argument(
        "--lr_interval",
        default=d.get("lr_interval"),
        help="per-ingredient learning rates as '<lo>:<hi>', spread geometrically "
        "over --num_soup points (e.g. '1e-5:4e-5'). Mutually exclusive with "
        "--lr_pool.",
    )
    ap.add_argument(
        "--rank",
        type=int,
        default=d.get("rank"),
        help="SVD truncation rank (default: the method config's network_dim)",
    )
    ap.add_argument("--method", default=method_default)
    ap.add_argument("--preset", default="default")
    ap.add_argument("--dry_run", action="store_true")
    args, ft_extra = ap.parse_known_args()

    # Fine-tune selection: an explicit CLI flag wins, then the config's
    # artists_shard / path_pattern. The two are mutually exclusive at every
    # level (a shard can't compose with an arbitrary OR-glob in one string) —
    # the same rule train.py enforces.
    if args.path_pattern and args.artists_shard:
        raise SystemExit(
            "--path_pattern and --artists_shard are mutually exclusive "
            "(the shard expands to a path_pattern)."
        )
    shard = args.artists_shard
    if not args.path_pattern and not shard:
        shard = cfg.get("artists_shard") or None
        if shard and cfg.get("path_pattern"):
            raise SystemExit(
                f"{config}: sets BOTH path_pattern and artists_shard — remove "
                f"one (or override on the CLI)."
            )
    if shard:
        args.path_pattern = expand_artists_shard(
            shard, args.method, args.preset, methods_subdir
        )
        n_alts = len(args.path_pattern.split("|"))
        print(f"[soup] artists_shard {shard} -> {n_alts} artists")
    elif not args.path_pattern:
        args.path_pattern = cfg.get("path_pattern")
    if not args.path_pattern:
        raise SystemExit(
            f"no fine-tune selection: pass --path_pattern or --artists_shard "
            f"(or set a top-level path_pattern / artists_shard in {config})."
        )
    if args.num_soup < 2:
        raise SystemExit(
            f"--num_soup must be >= 2 (a soup needs at least 2 ingredients); "
            f"got {args.num_soup}."
        )
    # Seeds are derived from the count: UNCOND_SEED+1 .. UNCOND_SEED+num_soup
    # (1001..1000+N), deterministic so re-runs reproduce the same ingredients.
    seeds = [UNCOND_SEED + i for i in range(1, args.num_soup + 1)]
    lrs = resolve_lrs(args.num_soup, args.lr_pool, args.lr_interval)
    if lrs and any(a.split("=", 1)[0] == "--learning_rate" for a in ft_extra):
        raise SystemExit(
            "--learning_rate in ARGS conflicts with --lr_pool/--lr_interval "
            "(the pipeline sets a per-ingredient LR). Drop one."
        )

    # The merged method config, resolved once: the σ-demote routing Phase 1 must
    # replay (ARGS only reaches Phase 2) and the Phase-3 SVD rank both read it.
    from library.config.io import load_method_preset  # torch-free, ~0.1s

    merged = load_method_preset(
        args.method, args.preset, methods_subdir=methods_subdir or "methods"
    )
    sigma = sigma_settings(merged, sigma_overrides(ft_extra))
    sigma_flags = sigma_argv(sigma, merged)

    svd_slice = configured_svd_slice(merged, ft_extra)
    check_init_mode(args.no_uncond, args.uncond_init, svd_slice, ft_extra)

    ckpt_dir = ROOT / "output" / "ckpt"
    name = args.name or (
        (slug_for_shard(shard) if shard else slug_for_pattern(args.path_pattern))
        + init_tag(args.no_uncond, svd_slice)
    )
    pool = pool_glob(args.pool_path_pattern, args.path_pattern)
    print(f"[soup] fine-tune pattern {args.path_pattern!r} -> slug {name!r}")
    if sigma:
        print(f"[soup] sigma_lowres ON for both phases: {' '.join(sigma_flags)}")
    # Both phases get the expanded glob, so any artists_shard living in the
    # method config must be switched OFF — train.py refuses to see both.
    no_shard = ["--artists_shard", ""]
    if not args.no_uncond:
        print(f"[soup] uncond pool pattern {pool!r}")

    # Phase 1 — uncond inter-train (skipped when the init already exists,
    # bypassed when a checkpoint is pinned via --uncond_init, or dropped
    # altogether with --no_uncond: the ingredients then share the deterministic
    # weight_svd[+svd_slice] init from W0 — the same shared-subspace property
    # the rank-r truncation relies on, minus the uncond ΔW).
    uncond_path: Path | None
    if args.no_uncond:
        uncond_path = None
        slice_note = f" (svd_slice={svd_slice})" if svd_slice else ""
        print(
            f"[soup] phase 1: SKIPPED (--no_uncond) — ingredients init from "
            f"down_init={merged.get('down_init', 'kaiming')!r}{slice_note}"
        )
    elif args.uncond_init:
        uncond_path = resolve_uncond_init(args.uncond_init)
        if not uncond_path.is_file():
            raise SystemExit(
                f"--uncond_init {args.uncond_init!r}: checkpoint not found "
                f"(looked for {uncond_path})."
            )
        print(f"[soup] phase 1: using pinned uncond init {uncond_path}")
    else:
        uncond = uncond_name(pool, args.uncond_ratio, args.uncond_epochs, sigma)
        uncond_path = ckpt_dir / f"{uncond}.safetensors"
        if uncond_path.exists():
            print(f"[soup] phase 1: reusing existing uncond init {uncond_path}")
        else:
            print(f"[soup] phase 1: training uncond init {uncond}")
            _train(
                build_method_args(
                    args.method,
                    preset=args.preset,
                    methods_subdir=methods_subdir,
                    extra=[
                        *no_shard,
                        *sigma_flags,
                        "--path_pattern",
                        pool,
                        "--caption_dropout_rate",
                        "1.0",
                        "--sample_ratio",
                        str(args.uncond_ratio),
                        "--seed",
                        str(UNCOND_SEED),
                        "--max_train_epochs",
                        str(args.uncond_epochs),
                        "--save_every_n_epochs",
                        str(args.uncond_epochs),
                        "--checkpointing_epochs",
                        str(args.uncond_epochs),
                        "--output_name",
                        uncond,
                    ],
                ),
                args.dry_run,
            )

    # Phase 2 — captioned fine-tunes from the shared init, one per seed (and,
    # when --lr_pool/--lr_interval is set, one LR per ingredient).
    ingredients: list[Path] = []
    for i, seed in enumerate(seeds):
        iname = f"anima_soup_{name}_s{seed}"
        ingredients.append(ckpt_dir / f"{iname}.safetensors")
        lr_extra = ["--learning_rate", f"{lrs[i]:g}"] if lrs else []
        lr_note = f" (lr {lrs[i]:g})" if lrs else ""
        print(f"[soup] phase 2: fine-tune seed {seed}{lr_note} -> {iname}")
        _train(
            build_method_args(
                args.method,
                preset=args.preset,
                methods_subdir=methods_subdir,
                extra=[
                    *no_shard,
                    "--path_pattern",
                    args.path_pattern,
                    *(["--network_weights", str(uncond_path)] if uncond_path else []),
                    "--seed",
                    str(seed),
                    "--output_name",
                    iname,
                    *lr_extra,
                    *ft_extra,
                ],
            ),
            args.dry_run,
        )

    # Phase 3 — ΔW soup, SVD-truncated back to the single-adapter rank.
    rank = args.rank if args.rank is not None else int(merged.get("network_dim", 16))
    soup_path = ckpt_dir / f"anima_soup_{name}.safetensors"
    print(f"[soup] phase 3: rank-{rank} SVD soup of {len(ingredients)} -> {soup_path}")
    if args.dry_run:
        print(
            f"  [dry-run] soup {', '.join(p.name for p in ingredients)} "
            f"--rank {rank} --out {soup_path}"
        )
        return
    from scripts.soup.build import build_soup_file

    build_soup_file([str(p) for p in ingredients], str(soup_path), rank=rank)
    # Probes (bench/memorization/*) replay training membership from the
    # snapshot; the ingredients share data/config, so the first one stands in.
    snapshot = ingredients[0].with_suffix("").with_suffix(".snapshot.toml")
    if snapshot.exists():
        shutil.copy(snapshot, soup_path.with_suffix("").with_suffix(".snapshot.toml"))
    print(f"[soup] done: {soup_path}")


if __name__ == "__main__":
    main()
