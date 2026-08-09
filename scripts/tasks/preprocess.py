"""Default-dataset preprocessing: resize → VAE latents → text-embedding caches."""

from __future__ import annotations

import os
from pathlib import Path

from ._common import PY, ROOT, _path, run


# Subfolders are walked by default (matches base.toml's `recursive = true`).
# Stems must stay unique across the tree — cache filenames are stem-keyed and flat.
def _min_pixels_args() -> list[str]:
    """``--min_pixels <N>`` derived from the variant TOML's
    ``drop_lowres_images`` + ``min_pixels`` keys (resolved through the same
    base → preset → method merge chain training uses, via ``_path_overrides``
    in scripts/tasks/_common.py).

    Returns ``[]`` when both keys are absent so plain CLI use keeps each
    script's own argparse default (500_000 = 0.5MP). ``drop_lowres_images
    = false`` forces ``--min_pixels 0`` even when ``min_pixels`` is set, so
    the user can flip a single boolean to disable the filter.

    The GUI Train auto-chain forwards the filter via the ``DROP_LOWRES_IMAGES`` /
    ``MIN_PIXELS`` env vars (mirrors ``PREPROCESS_PATH_PATTERN``): its CONFIG_FILE
    snapshot has the preprocess-only keys stripped, so the merged-config read
    below would miss them. Env wins over the merged config; absent env falls back
    to the merged chain (``preprocess.toml`` → base → preset → method)."""
    from ._common import _path_overrides  # local import: avoids unused circular

    env_drop = os.environ.get("DROP_LOWRES_IMAGES")
    env_min = os.environ.get("MIN_PIXELS")
    if env_drop is not None or env_min is not None:
        if env_drop is not None and not _boolish(env_drop, True):
            return ["--min_pixels", "0"]
        if env_min is None:
            return []
        try:
            return ["--min_pixels", str(max(0, int(env_min)))]
        except (TypeError, ValueError):
            return []

    overrides = _path_overrides()
    if "drop_lowres_images" not in overrides and "min_pixels" not in overrides:
        return []
    if overrides.get("drop_lowres_images") is False:
        return ["--min_pixels", "0"]
    raw = overrides.get("min_pixels", 500_000)
    try:
        n = max(0, int(raw))
    except (TypeError, ValueError):
        return []
    return ["--min_pixels", str(n)]


def _config_min_pixels() -> int:
    """The configured ``min_pixels`` threshold (merged chain), default 0.5MP."""
    from ._common import _path_overrides

    raw = _path_overrides().get("min_pixels", 500_000)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 500_000


def _target_res_args(extra) -> list[str]:
    """``--target_res E1 E2 …`` derived from the merged TOML's ``target_res`` key.

    Returns ``[]`` when an explicit ``--target_res`` is already in ``extra`` (CLI
    ARGS wins, no duplicate) or when the config value is absent / a bare
    ``[1024]`` (the legacy single-tier default — leave it off so the resize
    script's own default path runs). Invalid / unknown edges are dropped here so
    a typo in the TOML doesn't abort preprocessing.
    """
    if "--target_res" in extra:
        return []

    from library.datasets.buckets import ALLOWED_TARGET_RES

    # GUI Train auto-chain forwards tiers via env (its CONFIG_FILE snapshot strips
    # target_res); env wins over the merged config. Space/comma separated edges.
    env_tr = os.environ.get("TARGET_RES")
    if env_tr is not None:
        raw = env_tr.replace(",", " ").split()
    else:
        from ._common import _path_overrides

        raw = _path_overrides().get("target_res")
    if not raw:
        return []
    edges = raw if isinstance(raw, (list, tuple)) else [raw]
    try:
        edges = [int(e) for e in edges]
    except (TypeError, ValueError):
        return []
    edges = [e for e in edges if e in ALLOWED_TARGET_RES]
    if not edges or edges == [1024]:
        return []
    return ["--target_res", *(str(e) for e in edges)]


def _preprocess_path_pattern_args(extra) -> list[str]:
    """``--path_pattern <glob>`` for GUI preprocess subset filtering.

    CLI ARGS wins when it already carries a path-pattern flag. GUI submits pass
    ``PREPROCESS_PATH_PATTERN`` so training can keep using the method's regular
    ``path_pattern`` independently.
    """
    if "--path_pattern" in extra or "--path-pattern" in extra:
        return []

    from ._common import _path_overrides

    raw = os.environ.get("PREPROCESS_PATH_PATTERN")
    if raw is None:
        raw = _path_overrides().get("preprocess_path_pattern")
    pattern = str(raw or "").strip()
    if not pattern or pattern == "*":
        return []
    return ["--path_pattern", pattern]


def _resolved_path_pattern_args(extra) -> list[str]:
    for i, tok in enumerate(extra):
        if tok in {"--path_pattern", "--path-pattern"}:
            if i + 1 >= len(extra):
                raise SystemExit(f"{tok} requires a value")
            return ["--path_pattern", str(extra[i + 1])]
    return _preprocess_path_pattern_args(extra)


def _boolish(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _sigma_demote_routes(extra) -> list[str]:
    """The σ-demote routes (``["N:D", …]``) to chain, or ``[]`` when off.

    Enable with ``sigma_demote = true`` in ``configs/preprocess.toml`` (the
    measured-safe ``1024:896`` route), a ``"N:D"`` string to pick another route
    (probe it first), or a **comma list** of routes to emit several siblings in
    one pass — the stacked router (``--sigma_lowres_route2``) needs BOTH its
    routes' keys present, e.g. ``sigma_demote = "1024:896,1024:768"``. Each
    route lands its own ``demoted_{H}x{W}`` key inside the same native npz, so
    the passes are independent and idempotent. Env ``SIGMA_DEMOTE`` wins over
    the merged config (GUI auto-chain parity — the CONFIG_FILE snapshot strips
    preprocess-only keys). An explicit ``--sigma_demote`` in ``ARGS`` means this
    invocation IS a demote run already — never chain a second one.
    """
    if "--sigma_demote" in extra:
        return []
    raw = os.environ.get("SIGMA_DEMOTE")
    if raw is None:
        from ._common import _path_overrides

        raw = _path_overrides().get("sigma_demote")
    if raw is None or raw is False:
        return []
    if raw is True:
        return ["1024:896"]
    text = str(raw).strip()
    if not text or text.lower() in {"0", "false", "no", "off"}:
        return []
    if text.lower() in {"1", "true", "yes", "on"}:
        return ["1024:896"]
    routes = []
    for part in text.split(","):
        route = part.strip()
        if not route:
            continue
        if ":" not in route:
            print(
                f"  [preprocess] ignoring sigma_demote entry {route!r} — expected "
                'true/false or "NATIVE:DEMOTE" (e.g. "1024:896", or a comma '
                'list "1024:896,1024:768" for the stacked router)'
            )
            continue
        if route not in routes:  # a repeated route would just re-scan the corpus
            routes.append(route)
    return routes


def _pop_explicit_demote_routes(extra) -> tuple[list[str], list[str]]:
    """Pull an explicit ``--sigma_demote`` out of ``extra``, splitting a comma list.

    ``cache_latents.py`` parses a single ``NATIVE:DEMOTE`` (``int()`` on the
    halves), so a comma list has to be expanded into one pass per route before
    it reaches the script. Returns ``(routes, cleaned_extra)``.
    """
    routes: list[str] = []
    cleaned: list[str] = []
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok in {"--sigma_demote", "--sigma-demote"}:
            if i + 1 >= len(extra):
                raise SystemExit(f"{tok} requires a value (e.g. 1024:896)")
            for part in str(extra[i + 1]).split(","):
                route = part.strip()
                if route and route not in routes:
                    routes.append(route)
            i += 2
            continue
        cleaned.append(tok)
        i += 1
    return routes, cleaned


def _caption_correction_config(extra) -> tuple[dict[str, object], list[str]]:
    """Caption correction flags/config for preprocess-time TE caching.

    CLI ARGS wins over env/config. Returned ``extra`` has these flags removed so
    resize/cache scripts that do not know them never see unknown arguments.
    """

    from ._common import _path_overrides

    overrides = _path_overrides()
    env_trigger = os.environ.get("CAPTION_TRIGGER_WORD")
    config: dict[str, object] = {
        "correct_order": _boolish(
            os.environ.get("CAPTION_CORRECT_ORDER"),
            _boolish(overrides.get("caption_correct_order"), False),
        ),
        "insert_no_artist": _boolish(
            os.environ.get("CAPTION_INSERT_NO_ARTIST"),
            _boolish(overrides.get("caption_insert_no_artist"), False),
        ),
        "trigger_word": str(
            env_trigger
            if env_trigger is not None
            else overrides.get("caption_trigger_word", "")
        ).strip(),
        "trigger_at_front": _boolish(
            os.environ.get("CAPTION_TRIGGER_AT_FRONT"),
            _boolish(overrides.get("caption_trigger_at_front"), False),
        ),
    }

    cleaned: list[str] = []
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok in {"--caption_correct_order", "--caption-correct-order"}:
            config["correct_order"] = True
            i += 1
        elif tok in {"--no_caption_correct_order", "--no-caption-correct-order"}:
            config["correct_order"] = False
            i += 1
        elif tok in {"--caption_insert_no_artist", "--caption-insert-no-artist"}:
            config["insert_no_artist"] = True
            i += 1
        elif tok in {
            "--no_caption_insert_no_artist",
            "--no-caption-insert-no-artist",
        }:
            config["insert_no_artist"] = False
            i += 1
        elif tok in {"--caption_trigger_at_front", "--caption-trigger-at-front"}:
            config["trigger_at_front"] = True
            i += 1
        elif tok in {
            "--no_caption_trigger_at_front",
            "--no-caption-trigger-at-front",
        }:
            config["trigger_at_front"] = False
            i += 1
        elif tok in {"--caption_trigger_word", "--caption-trigger-word"}:
            if i + 1 >= len(extra):
                raise SystemExit(f"{tok} requires a value")
            config["trigger_word"] = str(extra[i + 1]).strip()
            i += 2
        else:
            cleaned.append(tok)
            i += 1
    return config, cleaned


def _caption_correction_enabled(config: dict[str, object]) -> bool:
    """Run the caption-rewrite pass when ANY caption-rewriting knob is set.

    ``correct_captions.py`` (→ ``correct_caption``) is the only path that
    injects the trigger word / ``@no-artist``, and it slots them by reordering
    into category buckets. So a trigger word or insert-no-artist with order
    correction *off* still has to run it — otherwise the GUI's trigger-word
    field is silently ignored at TE-cache time. Reordering is inherent to
    placing the trigger at the artist slot, so it rides along.
    """
    return bool(
        config.get("correct_order")
        or str(config.get("trigger_word") or "").strip()
        or config.get("insert_no_artist")
    )


def _caption_correction_args(config: dict[str, object]) -> list[str]:
    args: list[str] = []
    if config.get("insert_no_artist"):
        args.append("--caption_insert_no_artist")
    trigger = str(config.get("trigger_word") or "").strip()
    if trigger:
        args += ["--caption_trigger_word", trigger]
    if config.get("trigger_at_front"):
        args.append("--caption_trigger_at_front")
    return args


def _resize_crop_args(extra) -> list[str]:
    """Preprocess-only resize crop controls from the merged config chain."""
    if "--resize_crop_anchor" in extra or "--resize-crop-anchor" in extra:
        anchor_args: list[str] = []
    else:
        from library.preprocess.resize_preview import (
            DEFAULT_RESIZE_CROP_ANCHOR,
            RESIZE_CROP_ANCHORS,
        )

        from ._common import _path_overrides

        anchor = str(
            _path_overrides().get("resize_crop_anchor") or DEFAULT_RESIZE_CROP_ANCHOR
        ).strip()
        anchor_args = (
            ["--resize_crop_anchor", anchor]
            if anchor in RESIZE_CROP_ANCHORS and anchor != DEFAULT_RESIZE_CROP_ANCHOR
            else []
        )

    if "--resize_bucket_resos" in extra or "--resize-bucket-resos" in extra:
        bucket_args: list[str] = []
    else:
        from ._common import _path_overrides

        raw = _path_overrides().get("resize_bucket_resos")
        if isinstance(raw, str):
            buckets = [part.strip() for part in raw.split(",") if part.strip()]
        elif isinstance(raw, (list, tuple)):
            buckets = [str(item).strip() for item in raw if str(item).strip()]
        else:
            buckets = []
        bucket_args = ["--resize_bucket_resos", *buckets] if buckets else []

    if "--resize_crop_margins" in extra or "--resize-crop-margins" in extra:
        margin_args: list[str] = []
    else:
        from library.preprocess.resize_preview import normalize_crop_margins

        from ._common import _path_overrides

        margins = normalize_crop_margins(_path_overrides().get("resize_crop_margins"))
        values = [margins[key] for key in ("top", "right", "bottom", "left")]
        margin_args = (
            ["--resize_crop_margins", *(f"{value:g}" for value in values)]
            if any(value > 0 for value in values)
            else []
        )
    return [*anchor_args, *bucket_args, *margin_args]


def _freefit_args(extra) -> list[str]:
    """``--freefit_max_ratio R`` from the merged config chain.

    Free-fit is the only resize mode now (the ``--freefit`` toggle was removed),
    so we only forward the preprocess-only ``freefit_max_ratio`` knob
    (preprocess.toml → base → preset → method). CLI ``ARGS`` wins: if the flag is
    already in ``extra`` we emit nothing for it (no duplicate). A stale
    ``--freefit`` in ``ARGS`` is silently dropped by ``_strip_resize_only_args``.
    """
    from ._common import _path_overrides

    out: list[str] = []
    if "--freefit_max_ratio" not in extra and "--freefit-max-ratio" not in extra:
        # Env (GUI auto-chain) wins over the merged config, which the snapshot strips.
        raw = os.environ.get("FREEFIT_MAX_RATIO")
        if raw is None:
            raw = _path_overrides().get("freefit_max_ratio")
        if raw is not None:
            try:
                out += ["--freefit_max_ratio", f"{float(raw):g}"]
            except (TypeError, ValueError):
                pass
    return out


def _curation_decisions_args() -> list[str]:
    """Optional GUI curation decisions consumed by resize only."""

    path = Path(
        _path("curation_decisions", "post_image_dataset/curation_decisions.json")
    )
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return []
    return ["--curation_decisions", str(path)]


def _repa_pe_encoder() -> str | None:
    """The REPA vision encoder to cache, or ``None`` when REPA is off.

    Reads ``use_repa`` / ``repa_encoder`` from the merged config chain (the same
    ``_path_overrides`` the path knobs use — populated from ``METHOD`` /
    ``METHODS_SUBDIR`` or a GUI ``CONFIG_FILE`` snapshot). This lets the ConfigTab
    Train auto-chain — and any ``make preprocess METHOD=<repa-variant>`` — cache
    the ``{stem}_anima_pe_spatial.safetensors`` (or ``_anima_pe``) sidecars in the
    same pass, so a ``use_repa=true`` run doesn't bounce off train.py's
    "PE features absent" error. Plain ``make preprocess`` (no variant config in
    scope) sees no ``use_repa`` and returns ``None`` — the default stays fast.
    """
    from ._common import _path_overrides

    overrides = _path_overrides()
    raw = overrides.get("use_repa")
    # TOML/snapshot bools arrive as real bools; tolerate a stringified value too.
    enabled = raw is True or str(raw).strip().lower() in ("1", "true", "yes")
    if not enabled:
        return None
    encoder = str(overrides.get("repa_encoder") or "pe_spatial").strip()
    return encoder or "pe_spatial"


# REPA encoder name → the `make` target that fetches its vision checkpoint, for
# the fail-fast hint below.
_REPA_ENCODER_DOWNLOAD_TARGET = {
    "pe": "download-pe",
    "pe_spatial": "download-pe-spatial",
}


def _require_repa_encoder_model(encoder: str) -> None:
    """Fail fast (clear error, nonzero exit) if the REPA vision checkpoint is
    absent — never silently auto-download it from inside the daemon.

    The PE caching step the auto-chain calls would otherwise fall into
    ``hf_hub_download`` with no timeout (``library/vision/encoders.py``). In the
    daemon's detached, console-less child that fetch surfaces no progress, and a
    stalled/gated download hangs indefinitely; because the daemon queue is
    *serial*, that one hung preprocess wedges every job queued behind it
    (training included). So when a ``use_repa=true`` Train auto-chain reaches
    this step we require the checkpoint up front and bail with an actionable
    message instead. Users who want the one-time download just run the named
    target manually first (it shows real progress in a foreground terminal)."""
    import sys
    from pathlib import Path

    try:
        from library.vision.encoders import get_encoder_info

        model_path = Path(get_encoder_info(encoder).default_model_id())
    except (KeyError, ImportError):
        return  # unknown encoder / import issue — let the downstream step report it
    if model_path.is_file():
        return
    target = _REPA_ENCODER_DOWNLOAD_TARGET.get(encoder, "download-models")
    sys.exit(
        f"  [preprocess] use_repa=true needs the REPA vision checkpoint, but "
        f"it's missing:\n      {model_path}\n"
        f"  Fetch it once with `make {target}` (or `make download-models`), "
        f"then start training again.\n"
        f"  (Not auto-downloading here on purpose: in the background daemon the "
        f"fetch shows no progress and a stalled download would hang the queue.)"
    )


def _pop_resize_only_args(extra) -> list[str]:
    """Strip resize-only flags from ``extra`` before cache stages run.

    The VAE/TE/PE stages read whatever latent shapes are already on disk, so they
    must never see resize-only argparse flags.
    """
    cleaned: list[str] = []
    it = iter(extra)
    for tok in it:
        if tok in {
            "--target_res",
            "--resize_bucket_resos",
            "--resize-bucket-resos",
            "--resize_crop_margins",
            "--resize-crop-margins",
        }:
            for nxt in it:
                if nxt.startswith("--"):
                    cleaned.append(nxt)
                    break
            continue
        if tok in {"--resize_crop_anchor", "--resize-crop-anchor"}:
            next(it, None)
            continue
        if tok in {"--freefit_max_ratio", "--freefit-max-ratio"}:
            next(it, None)
            continue
        if tok == "--freefit":  # store_true — no value to consume
            continue
        cleaned.append(tok)
    return cleaned


def _resolve_lowres_filter(extra) -> tuple[list[str], list[str]]:
    """Reconcile the low-res input filter against CLI ``ARGS``.

    Returns ``(min_pixels_args, cleaned_extra)`` where ``cleaned_extra`` has
    our two convenience flags popped so the underlying scripts never see an
    arg their argparse doesn't define. Precedence (highest first):

      1. An explicit ``--min_pixels N`` in ``ARGS`` — left in ``extra`` and
         wins outright; we inject nothing (no duplicate ``--min_pixels``).
      2. ``--no_drop_lowres`` in ``ARGS`` → ``--min_pixels 0`` (keep every
         image), overriding ``drop_lowres_images = true`` in the TOML.
      3. ``--drop_lowres`` in ``ARGS`` → force the configured ``min_pixels``
         threshold, overriding ``drop_lowres_images = false`` in the TOML.
      4. Neither flag → fall back to the merged-config behavior
         (``_min_pixels_args``)."""
    cleaned = list(extra)
    no_drop = "--no_drop_lowres" in cleaned
    drop = "--drop_lowres" in cleaned
    cleaned = [a for a in cleaned if a not in ("--no_drop_lowres", "--drop_lowres")]

    if "--min_pixels" in cleaned:
        return [], cleaned
    if no_drop:  # disable wins over enable when both are passed
        return ["--min_pixels", "0"], cleaned
    if drop:
        return ["--min_pixels", str(_config_min_pixels())], cleaned
    return _min_pixels_args(), cleaned


def _drop_option_with_value(extra, names: set[str]) -> list[str]:
    cleaned: list[str] = []
    i = 0
    while i < len(extra):
        if extra[i] in names:
            i += 2
            continue
        cleaned.append(extra[i])
        i += 1
    return cleaned


def cmd_preprocess_resize(extra):
    mp_args, extra = _resolve_lowres_filter(extra)
    tr_args = _target_res_args(extra)
    pp_args = _preprocess_path_pattern_args(extra)
    cd_args = _curation_decisions_args()
    rc_args = _resize_crop_args(extra)
    ff_args = _freefit_args(extra)
    run(
        [
            PY,
            "scripts/preprocess/resize_images.py",
            "--src",
            _path("source_image_dir", "image_dataset"),
            "--dst",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--no_copy_captions",
            "--recursive",
            *mp_args,
            *tr_args,
            *rc_args,
            *ff_args,
            *pp_args,
            *cd_args,
            *extra,
        ]
    )


def cmd_preprocess_reconcile(extra):
    """Remove caches stale for the configured ``target_res`` (dry-run by default).

    Pass ``ARGS="--delete"`` to actually remove. ``target_res`` comes from the
    merged config (same as resize); an explicit ``--target_res`` in ``ARGS``
    wins. Useful after adding/dropping a tier so re-running preprocess + mask
    regenerates only the images whose bucket moved.
    """
    # _target_res_args returns [] for a bare [1024]/absent config AND when ARGS
    # already carries --target_res. Inject the 1024 default only in the former case.
    tr_args = _target_res_args(extra)
    if not tr_args and "--target_res" not in extra:
        tr_args = ["--target_res", "1024"]
    run(
        [
            PY,
            "scripts/preprocess/reconcile_caches.py",
            "--image-dir",
            _path("source_image_dir", "image_dataset"),
            "--resized-dir",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--cache-dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--mask-dir",
            _path("mask_dir", "post_image_dataset/masks"),
            *tr_args,
            *extra,
        ]
    )


def cmd_preprocess_vae(extra):
    pp_args = _preprocess_path_pattern_args(extra)
    run(
        [
            PY,
            "scripts/preprocess/cache_latents.py",
            "--dir",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--batch_size",
            "1",
            "--chunk_size",
            "0",
            "--recursive",
            "--no_half_vae",
            *pp_args,
            *extra,
        ]
    )
    # sigma_demote in preprocess.toml chains the demote emit(s) here, so
    # `make preprocess` / `preprocess-vae` keep the sibling keys current and a
    # --sigma_lowres run never trains against a stale/missing demoted cache.
    # One pass per configured route — the stacked router needs both present.
    for route in _sigma_demote_routes(extra):
        print(f"  [preprocess] sigma_demote={route} → emitting demoted sibling latents")
        _run_demote_pass(route, extra)


def _run_demote_pass(route: str, extra) -> None:
    """One ``cache_latents.py`` pass emitting a single route's demoted siblings."""
    pp_args = _preprocess_path_pattern_args(extra)
    run(
        [
            PY,
            "scripts/preprocess/cache_latents.py",
            "--dir",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--batch_size",
            "1",
            "--chunk_size",
            "0",
            "--recursive",
            "--no_half_vae",
            "--sigma_demote",
            route,
            *pp_args,
            *extra,
        ]
    )


def cmd_preprocess_demote(extra):
    """Emit σ-demote sibling latents (sigma_lowres Phase 1b, e.g. 1024→896).

    Same VAE-load path as ``preprocess-vae``; appends a ``demoted_{H}x{W}``
    key inside each native-tier image's existing npz. Idempotent. Requires
    ``preprocess-vae`` to have run first.

    Routes come from ``sigma_demote`` in ``configs/preprocess.toml`` (or the
    ``SIGMA_DEMOTE`` env var) — the SAME source the automatic chain off
    ``preprocess-vae`` uses, so a comma list like ``"1024:896,1024:768"``
    emits BOTH siblings the stacked router (``--sigma_lowres_route2``) needs
    from this target too. ``ARGS="--sigma_demote N:D[,N:D…]"`` overrides
    (probe a new route before shipping it); with neither set we fall back to
    the certified ``1024:896``. ``cache_latents.py`` takes one route per
    invocation, so each route is its own pass.
    """
    routes, extra = _pop_explicit_demote_routes(extra)
    if not routes:
        # `extra` no longer carries --sigma_demote, so this reads the config/env.
        routes = _sigma_demote_routes(extra) or ["1024:896"]
    for route in routes:
        if len(routes) > 1:
            print(f"  [preprocess] sigma_demote={route} → emitting demoted siblings")
        _run_demote_pass(route, extra)


_QWEN3_TOKENIZER = "models/text_encoders/qwen_3_06b_base.safetensors"


def _variant_settings() -> tuple[str, str, str]:
    """Caption-variant knobs: env override → preprocess.toml → historical default.

    Returns ``(shuffle_variants, tag_dropout_rate, tag_randomize_rate)`` as raw
    strings (forwarded straight to the script). CAPTION_SHUFFLE_VARIANTS /
    CAPTION_TAG_DROPOUT_RATE / CAPTION_TAG_RANDOMIZE_RATE let the GUI tune these
    without editing config.
    """
    shuffle = os.environ.get("CAPTION_SHUFFLE_VARIANTS") or _path(
        "caption_shuffle_variants", "4"
    )
    dropout = os.environ.get("CAPTION_TAG_DROPOUT_RATE") or _path(
        "caption_tag_dropout_rate", "0.1"
    )
    # Lexinvariant tag regularization: identity-randomized r-family. 0.0 = off
    # (no r-family written, fully backward compatible).
    randomize = os.environ.get("CAPTION_TAG_RANDOMIZE_RATE") or _path(
        "caption_tag_randomize_rate", "0.0"
    )
    return str(shuffle), str(dropout), str(randomize)


def _float_or_zero(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _ensure_danbooru_tags() -> None:
    """Fetch the Danbooru tag KB on demand so caption correction never aborts.

    ``correct_captions.py`` loads ``danbooru_tags_classified.csv`` (the bucket
    taxonomy) and ``SystemExit``s if it's missing — GUI users reach preprocess
    without ``make download-danbooru-tags``. Mirror the tagger-vocab auto-fetch
    (best-effort): catch ``SystemExit``/``OSError`` so a failed download skips
    rather than aborts; ``correct_captions.py`` still surfaces its own clear
    error if the file is genuinely unavailable.
    """
    from library.captioning.correction import find_tag_csv

    if find_tag_csv(ROOT) is not None:
        return
    print("  [preprocess] danbooru tag KB missing; fetching it for caption correction")
    try:
        # Base CSV only — that's the file `find_tag_csv` / `correct_captions.py`
        # need. The English sibling (`download-danbooru-tags`) is a heavier
        # wiki-join step only the GUI tooltip uses; skip it on the preprocess path.
        from .downloads import _download_danbooru_base

        _download_danbooru_base([])
    except (SystemExit, OSError) as e:
        print(f"  [preprocess] danbooru tag KB auto-download failed: {e}")


def cmd_preprocess_captions(extra, caption_config: dict[str, object] | None = None):
    """Write corrected/variant caption sidecars into ``resized/``.

    Runs whenever caption order-correction is enabled **or** variants are
    requested (the default ``caption_shuffle_variants=4``). Order-correction off
    + variants on runs in passthrough (``--no_correct``): v0 mirrors the raw
    caption and the shuffle/dropout/randomize sidecars ride alongside, so the
    user can see the train-time variants directly in ``resized/``.
    """
    if caption_config is None:
        caption_config, extra = _caption_correction_config(extra)
    correct = _caption_correction_enabled(caption_config)
    shuffle, dropout, randomize = _variant_settings()
    n_variants = int(_float_or_zero(shuffle))
    if not correct and n_variants <= 0:
        print("  [preprocess] caption correction disabled")
        return
    # correct_captions.py loads the Danbooru tag KB unconditionally (bucket
    # taxonomy for both correct + variant-only paths) — fetch it on demand so a
    # GUI preprocess that skipped `make download-danbooru-tags` doesn't abort.
    _ensure_danbooru_tags()
    pp_args = _resolved_path_pattern_args(extra)
    cmd = [
        PY,
        "scripts/preprocess/correct_captions.py",
        "--src",
        _path("source_image_dir", "image_dataset"),
        "--dst",
        _path("resized_image_dir", "post_image_dataset/resized"),
        "--recursive",
        *pp_args,
    ]
    if correct:
        cmd += _caption_correction_args(caption_config)
    else:
        cmd.append("--no_correct")
    if n_variants > 0:
        cmd += [
            "--caption_shuffle_variants",
            shuffle,
            "--caption_tag_dropout_rate",
            dropout,
            "--caption_tag_randomize_rate",
            randomize,
        ]
        # Identity-randomize needs the two tokenizers to build the erasure pool.
        if _float_or_zero(randomize) > 0.0 and n_variants >= 2:
            cmd += ["--qwen3", _QWEN3_TOKENIZER]
    run(cmd)


def cmd_preprocess_te(extra, caption_config: dict[str, object] | None = None):
    if caption_config is None:
        caption_config, extra = _caption_correction_config(extra)
    shuffle, dropout, randomize = _variant_settings()
    n_variants = int(_float_or_zero(shuffle))
    # The caption step writes the variant sidecars (the encode source of truth);
    # it runs whenever correction is on OR variants are requested. In that case
    # the TE step reads ``resized/`` (already the curated set, so min_pixels=0)
    # and encodes the sidecars verbatim. Only the pure no-correction +
    # no-variants case still reads the source captions with a match filter.
    needs_caption_step = _caption_correction_enabled(caption_config) or n_variants > 0
    if needs_caption_step:
        _, extra = _resolve_lowres_filter(extra)
        extra = _drop_option_with_value(extra, {"--min_pixels"})
        pp_args = _preprocess_path_pattern_args(extra)
        cmd_preprocess_captions(extra, caption_config=caption_config)
        text_dir = _path("resized_image_dir", "post_image_dataset/resized")
        match_args: list[str] = []
        mp_args: list[str] = ["--min_pixels", "0"]
    else:
        pp_args = _preprocess_path_pattern_args(extra)
        text_dir = _path("source_image_dir", "image_dataset")
        match_args = [
            "--match_images_from",
            _path("resized_image_dir", "post_image_dataset/resized"),
        ]
        mp_args, extra = _resolve_lowres_filter(extra)
    run(
        [
            PY,
            "scripts/preprocess/cache_text_embeddings.py",
            "--dir",
            text_dir,
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            *match_args,
            "--qwen3",
            _QWEN3_TOKENIZER,
            "--dit",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            # Fallback only — when a {stem}.variants.txt sidecar is present (the
            # caption step wrote it) the encoder uses it verbatim and these are
            # ignored; they still drive in-process generation for any image that
            # reaches TE without a sidecar.
            "--caption_shuffle_variants",
            shuffle,
            "--caption_tag_dropout_rate",
            dropout,
            "--caption_tag_randomize_rate",
            randomize,
            "--recursive",
            *mp_args,
            *pp_args,
            *extra,
        ]
    )


def cmd_preprocess_pe(extra):
    """Cache PE-Core-L14-336 vision-encoder features.

    Reads pre-resized images from ``post_image_dataset/resized/`` (the
    standard LoRA pipeline source) and writes
    ``{stem}_anima_pe.safetensors`` sidecars into the LoRA cache dir so the
    dataset's existing ``cache_dir`` lookup finds them.

    Consumed by IP-Adapter when reading PE features off disk.

    Also emits the dataset-mean PE centroid sidecar
    (``post_image_dataset/ip_adapter/anima_pe_centroid_pe.safetensors``) via
    ``--centroid`` so IP-Adapter mean-centering works without a separate pass.
    """
    run(
        [
            PY,
            "scripts/preprocess/cache_pe_encoder.py",
            "--dir",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--encoder",
            "pe",
            "--recursive",
            "--centroid",
            *extra,
        ]
    )


def cmd_preprocess_pe_spatial(extra):
    """Cache PE-Spatial-B16-512 dense patch-token features for REPA v2.

    Reads pre-resized images from ``post_image_dataset/resized/`` and writes
    ``{stem}_anima_pe_spatial.safetensors`` sidecars into the LoRA cache dir
    (disjoint from the PE-Core ``_anima_pe`` caches CMMD reads). No centroid —
    REPA aligns per-patch, not against a dataset mean. Run before a
    ``use_repa=true`` training arm.
    """
    run(
        [
            PY,
            "scripts/preprocess/cache_pe_encoder.py",
            "--dir",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--encoder",
            "pe_spatial",
            "--recursive",
            *extra,
        ]
    )


def cmd_caption_index(extra):
    """Build the method-agnostic typed-tag caption index.

    Walks caption sidecars under the source dir, classifies tags into
    character / copyright / artist / count via the Anima Tagger vocab, and
    writes ``post_image_dataset/captions/caption_index.json`` (per-image typed
    tags + group inversions). Pure data, no GPU. Consumed by the IP-Adapter
    distinct-pair sampler, artist balancing, and dataset analytics. Regenerate
    when the dataset or vocab changes.
    """
    pp_args = _preprocess_path_pattern_args(extra)
    run(
        [
            PY,
            "scripts/preprocess/build_caption_index.py",
            "--src",
            _path("source_image_dir", "image_dataset"),
            *pp_args,
            *extra,
        ]
    )


# `cmd_preprocess` auto-fetches this (~0.7 MB) vocab on demand: the caption index
# it gates is a hard requirement for soft-tokens contrastive training (train.py
# raises FileNotFoundError without it). Fetch is best-effort.
_CAPTION_INDEX_VOCAB = "models/captioners/anima-tagger-v2/vocab.json"


def cmd_preprocess(extra):
    caption_config, extra = _caption_correction_config(extra)
    # PE features are NOT cached here by default (CMMD chains `preprocess-pe`
    # explicitly) — keeps the default LoRA preprocess fast. Exception: a
    # `use_repa=true` variant aligns against PE every step, so they're chained at
    # the end (see the `_repa_pe_encoder()` block below).
    #
    # Fail fast BEFORE any GPU work: a use_repa=true auto-chain with a missing REPA
    # checkpoint would stall the PE step on a silent daemon download and wedge the
    # serial queue. Surface an actionable error instead of after the full pass.
    encoder = _repa_pe_encoder()
    if encoder is not None:
        _require_repa_encoder_model(encoder)
    cmd_preprocess_resize(extra)
    # VAE/TE steps read on-disk shapes — strip the low-res convenience flags AND
    # the resize-only --target_res so their argparse never sees an undefined arg.
    downstream = _pop_resize_only_args(extra)
    _, vae_extra = _resolve_lowres_filter(downstream)
    cmd_preprocess_vae(vae_extra)
    cmd_preprocess_te(downstream, caption_config=caption_config)
    # Caption index as a free by-product — consumed by the IP-Adapter pair sampler,
    # artist balancing, analytics, AND soft-tokens (which hard-errors without it).
    vocab = _path("caption_index_vocab", _CAPTION_INDEX_VOCAB)
    if not os.path.exists(vocab):
        # GUI users reach preprocess without `make download-models`, so fetch the
        # tiny tagger vocab on demand. Catch broadly (SystemExit from run(), OSError
        # from a missing `hf`) so we skip rather than abort the already-done GPU work.
        print("  [preprocess] tagger vocab missing; fetching it for caption-index")
        try:
            from .downloads import cmd_download_tagger

            cmd_download_tagger([])
        except (SystemExit, OSError) as e:
            print(f"  [preprocess] tagger vocab auto-download failed: {e}")
    if os.path.exists(vocab):
        # Caption correction writes TE-only sidecars under resized_image_dir. The
        # caption index intentionally stays on source captions because its
        # consumers care about tag presence/relations, not corrected order.
        cmd_caption_index([])
    else:
        print(
            f"  [preprocess] skipping caption-index: tagger vocab not found at "
            f"{_CAPTION_INDEX_VOCAB} and auto-download failed. Run "
            f"`make download-tagger`, then `make caption-index` "
            f"(soft-tokens contrastive training needs it)."
        )

    # REPA arm: a `use_repa=true` variant needs the PE sidecars REPA aligns against
    # (train.py errors without them); chaining here builds them in one pass. `encoder`
    # was resolved (and its checkpoint required) at the top.
    if encoder is not None:
        print(f"  [preprocess] use_repa=true → caching REPA PE features ({encoder})")
        if encoder == "pe_spatial":
            cmd_preprocess_pe_spatial([])
        else:
            cmd_preprocess_pe([])


def cmd_preprocess_config(extra):
    """Preprocess the exact directories named in a ``--dataset_config`` TOML.

    Unlike ``cmd_preprocess`` (which resolves the repo's standard
    ``image_dataset/`` → ``post_image_dataset/`` layout from the merged
    config), this drives off the same dataset config the *training* job will
    consume, so one file fully describes an ad-hoc job — no reliance on the
    default layout. For each ``[[datasets.subsets]]`` it:

      1. bucket-resizes ``--src`` (the originals, with caption sidecars) into
         that subset's ``image_dir`` — the source dir is never modified;
      2. caches VAE latents from ``image_dir`` into the subset's ``cache_dir``;
      3. caches text embeddings (captions read from ``--src``) into ``cache_dir``.

    A config can't encode where the *un-resized* originals live (its
    ``image_dir`` is the post-resize dir training reads), so the source is the
    one explicit flag: ``--src <dir>``. The ComfyUI trainer node uses this to
    cache a single-image temp dir before its chained training job runs.

    The VAE / text-encoder / DiT used for caching default to the config-resolved
    ``models/`` paths (base → preset → method merge), but can be overridden with
    ``--vae`` / ``--qwen3`` / ``--dit`` so a caller can point the cache at models
    living elsewhere — e.g. the ComfyUI trainer node passes the paths ComfyUI's
    own ``folder_paths`` registers, so it never assumes a copy under
    ``anima_lora/models/``.

    Usage: ``preprocess-config --dataset_config <path> --src <dir>
    [--vae <path>] [--qwen3 <path>] [--dit <path>] [extra…]``
    (any remaining args are forwarded to the resize step).
    """
    import toml

    args = list(extra)
    cfg_path: str | None = None
    src_dir: str | None = None
    vae_path = _path("vae", "models/vae/qwen_image_vae.safetensors")
    qwen3_path = _path("qwen3", "models/text_encoders/qwen_3_06b_base.safetensors")
    dit_path = _path(
        "pretrained_model_name_or_path",
        "models/diffusion_models/anima-base-v1.0.safetensors",
    )
    rest: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--dataset_config" and i + 1 < len(args):
            cfg_path = args[i + 1]
            i += 2
        elif args[i] == "--src" and i + 1 < len(args):
            src_dir = args[i + 1]
            i += 2
        elif args[i] == "--vae" and i + 1 < len(args):
            vae_path = args[i + 1]
            i += 2
        elif args[i] == "--qwen3" and i + 1 < len(args):
            qwen3_path = args[i + 1]
            i += 2
        elif args[i] == "--dit" and i + 1 < len(args):
            dit_path = args[i + 1]
            i += 2
        else:
            rest.append(args[i])
            i += 1
    if not cfg_path or not src_dir:
        raise SystemExit(
            "preprocess-config requires --dataset_config <path> and --src <dir>"
        )

    # Retry through a transient PermissionError: a real-time scanner (Windows
    # Defender) briefly locks the just-created config the ComfyUI trainer node
    # writes milliseconds before the daemon's preprocess job opens it.
    import time

    last_err: OSError | None = None
    for attempt in range(10):
        try:
            cfg = toml.load(cfg_path)
            break
        except PermissionError as e:
            last_err = e
            time.sleep(0.2 * (attempt + 1))
    else:
        raise SystemExit(
            f"could not read {cfg_path} after retrying (last error: {last_err}). "
            "If this persists, exclude the dataset/temp dir from your antivirus."
        )
    subsets = [
        sub
        for ds in (cfg.get("datasets") or [])
        for sub in (ds.get("subsets") or [])
        if sub.get("image_dir")
    ]
    if not subsets:
        raise SystemExit(f"no [[datasets.subsets]] with image_dir in {cfg_path}")

    for sub in subsets:
        image_dir = sub["image_dir"]
        cache_dir = sub.get("cache_dir") or image_dir
        # 1) bucket-resize originals → image_dir. cache_latents.py keys caches by
        #    the on-disk size, so the resized size must already be the constant-token
        #    bucket the trainer selects. Captions stay in --src (TE reads them there).
        run(
            [
                PY,
                "scripts/preprocess/resize_images.py",
                "--src",
                src_dir,
                "--dst",
                image_dir,
                "--no_copy_captions",
                "--min_pixels",
                "0",
                "--bucket_reso_steps",
                "64",
                "--recursive",
                *rest,
            ]
        )
        run(
            [
                PY,
                "scripts/preprocess/cache_latents.py",
                "--dir",
                image_dir,
                "--cache_dir",
                cache_dir,
                "--vae",
                vae_path,
                "--batch_size",
                "2",
                "--chunk_size",
                "64",
                "--recursive",
            ]
        )
        # 3) text embeddings — captions read from --src
        run(
            [
                PY,
                "scripts/preprocess/cache_text_embeddings.py",
                "--dir",
                src_dir,
                "--cache_dir",
                cache_dir,
                "--qwen3",
                qwen3_path,
                "--dit",
                dit_path,
                "--recursive",
            ]
        )
