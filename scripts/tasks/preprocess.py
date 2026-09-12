"""Default-dataset preprocessing: resize → VAE latents → text-embedding caches.

The curation stages in this chain — resize, autotag, position clauses, the
caption correction/mirror — are ``anime_tools`` **request objects**
(``ResizeRequest`` / ``AutotagRequest`` / ``PositionRequest`` /
``CorrectRequest``); the trainer never spells a flag. Each runs through
``_common.execute_stage``: in-process under a daemon job (every GUI run), as a
``python -m`` child from a shell (see ``masking.py`` for the rationale). A
user's ``ARGS`` reach a stage through the request's own generated parser
(``request_with_args``), so every flag the stage has still works from ``make``.
The VAE / TE / PE caches stay trainer-side scripts.
"""

from __future__ import annotations

import os
from pathlib import Path

from anime_tools.contract import AUTOTAG_MODES

from ._common import (
    PY,
    ROOT,
    _path,
    execute_stage,
    gui_stage_values,
    in_daemon_job,
    request_from_form,
    request_with_args,
    run,
    stage_by_id,
)


# Subfolders are walked by default. Stems must stay unique across the tree —
# cache filenames are stem-keyed and flat.
def _min_pixels_args() -> list[str]:
    """``--min_pixels <N>`` derived from the merged config's ``drop_lowres_images``
    / ``min_pixels`` keys. Returns ``[]`` when both are absent (each script's own
    argparse default applies). ``drop_lowres_images = false`` forces
    ``--min_pixels 0`` even when ``min_pixels`` is set. GUI auto-chain env
    (``DROP_LOWRES_IMAGES``, the threshold from the GUI's resize form or the
    ``MIN_PIXELS`` env) wins over the merged config.
    """
    from ._common import _path_overrides  # local import: avoids unused circular

    env_drop = os.environ.get("DROP_LOWRES_IMAGES")
    env_min = os.environ.get("MIN_PIXELS")
    form_min = (_resize_form() or {}).get("min_pixels")
    if form_min in ("", None):
        form_min = None
    if env_drop is not None or env_min is not None or form_min is not None:
        if env_drop is not None and not _boolish(env_drop, True):
            return ["--min_pixels", "0"]
        raw = env_min if env_min is not None else form_min
        if raw is None:
            return []
        try:
            return ["--min_pixels", str(max(0, int(raw)))]
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


def _resize_form() -> dict | None:
    """The GUI's resize stage form (``PREPROCESS_STAGES_JSON``), or ``None``
    from a plain shell."""
    form = gui_stage_values().get("resize")
    return form if isinstance(form, dict) else None


def _config_target_res() -> tuple[int, ...] | None:
    """The configured free-fit tiers, or ``None`` for the package default
    (a single 1024 tier — a bare ``[1024]`` collapses to it too).

    GUI auto-chain env (``TARGET_RES``, space/comma separated) wins over the
    GUI's resize form, which wins over the merged config. Unknown edges are
    dropped rather than aborting on a config typo.
    """
    from library.datasets.buckets import ALLOWED_TARGET_RES

    env_tr = os.environ.get("TARGET_RES")
    form = _resize_form()
    if env_tr is not None:
        raw = env_tr.replace(",", " ").split()
    elif form is not None and form.get("target_res"):
        raw = form["target_res"]
    else:
        from ._common import _path_overrides

        raw = _path_overrides().get("target_res")
    if not raw:
        return None
    edges = raw if isinstance(raw, (list, tuple)) else [raw]
    try:
        edges = [int(e) for e in edges]
    except (TypeError, ValueError):
        return None
    edges = [e for e in edges if e in ALLOWED_TARGET_RES]
    if not edges or edges == [1024]:
        return None
    return tuple(edges)


def _target_res_args(extra) -> list[str]:
    """``--target_res E1 E2 …`` for the trainer-side scripts that take it
    (``reconcile_caches.py``). ``[]`` when ``--target_res`` is already in
    ``extra`` (CLI wins) or the config is the default single tier."""
    if "--target_res" in extra:
        return []
    edges = _config_target_res()
    return ["--target_res", *(str(e) for e in edges)] if edges else []


def _preprocess_path_pattern_args(extra) -> list[str]:
    """``--path_pattern <glob>`` for GUI preprocess subset filtering.

    CLI ARGS wins when it already carries a path-pattern flag. GUI submits
    ``PREPROCESS_PATH_PATTERN`` separately so training keeps its own
    ``path_pattern`` independent.
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


def _preprocess_path_pattern() -> str:
    """The GUI/config subset scope as a request field (``"*"`` = everything)."""
    args = _preprocess_path_pattern_args([])
    return args[1] if args else "*"


def _resolved_path_pattern(extra) -> str:
    """The subset scope a curation stage runs under: an explicit
    ``--path_pattern`` in ``extra`` wins, else the env/config one, else ``*``."""
    for i, tok in enumerate(extra):
        if tok in {"--path_pattern", "--path-pattern"}:
            if i + 1 >= len(extra):
                raise SystemExit(f"{tok} requires a value")
            return str(extra[i + 1])
    return _preprocess_path_pattern()


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


def _floatish(*values, default: float = 0.0) -> float:
    """First value that parses as a float, else ``default``. A blank env var
    (GUI writes ``""`` for an empty field) falls through rather than raising."""
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            return float(text)
        except ValueError:
            continue
    return float(default)


def _sigma_demote_routes(extra) -> list[str]:
    """The σ-demote routes (``["N:D", …]``) to chain, or ``[]`` when off.

    Enable with ``sigma_demote = true`` (the certified ``1024:896`` route), a
    ``"N:D"`` string for another route, or a comma list to emit several
    siblings in one pass (the stacked router needs both routes' keys present).
    Env ``SIGMA_DEMOTE`` wins over the merged config. An explicit
    ``--sigma_demote`` in ``ARGS`` means this invocation IS a demote run
    already — never chain a second one.
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
    """Pull an explicit ``--sigma_demote`` out of ``extra``, splitting a comma
    list into one route per pass (``cache_latents.py`` only parses a single
    ``NATIVE:DEMOTE``). Returns ``(routes, cleaned_extra)``."""
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


CAPTION_INDEX_PATH = "post_image_dataset/captions/caption_index.json"
DEFAULT_OCR_DIR = "post_image_dataset/ocr"
"""The OCR sidecar tree (``{stem}.ocr.txt``), mirroring the resized layout.
Overridable as ``ocr_dir`` in the config chain."""


def _caption_correction_config(extra) -> tuple[dict[str, object], list[str]]:
    """Caption correction flags/config for preprocess-time TE caching.

    CLI ARGS wins over env/config. Returned ``extra`` has these flags removed
    so resize/cache scripts that don't know them never see unknown arguments.
    """

    from ._common import _path_overrides

    overrides = _path_overrides()
    env_drop_groups = os.environ.get("CAPTION_DROP_GROUPS")
    config: dict[str, object] = {
        "correct_order": _boolish(overrides.get("caption_correct_order"), False),
        "insert_no_artist": _boolish(overrides.get("caption_insert_no_artist"), False),
        "trigger_word": str(overrides.get("caption_trigger_word", "")).strip(),
        "trigger_at_front": _boolish(overrides.get("caption_trigger_at_front"), False),
        # Tag groups stripped at mirror time (GH #95) — comma-separated slugs
        # / taxonomy-path prefixes, see anime_tools.captions.tag_drop_groups.
        "drop_groups": str(
            env_drop_groups
            if env_drop_groups is not None
            else _drop_groups_override(overrides.get("caption_drop_groups"))
        ).strip(),
        # Not a `correct_captions.py` flag — gates a separate stage that runs
        # BEFORE the caption/TE steps (see `cmd_preprocess`). Writes the
        # DERIVED caption (`resized/`), not the master.
        "position_clauses": _boolish(
            os.environ.get("CAPTION_POSITION_CLAUSES"),
            _boolish(overrides.get("caption_position_clauses"), False),
        ),
        # Same deal as `position_clauses`, but runs even earlier: it *creates*
        # the captions the rest of the chain reads — see `cmd_preprocess`.
        "autotag": _boolish(
            os.environ.get("CAPTION_AUTOTAG"),
            _boolish(overrides.get("caption_autotag"), False),
        ),
        "autotag_mode": str(overrides.get("caption_autotag_mode") or "missing").strip(),
        "autotag_min_confidence": _floatish(
            overrides.get("caption_autotag_min_confidence"), default=0.0
        ),
        # The caption-MASTER stages are driven from this dict alone (the
        # caller's `extra` never reaches them), so the subset scope must ride
        # along or a --path_pattern-scoped preprocess would rewrite captions
        # across the WHOLE master (destructive with autotag merge/overwrite).
        "path_pattern": _resolved_path_pattern(extra),
    }
    # The GUI's stage forms (``PREPROCESS_STAGES_JSON``): the `autotag` /
    # `correct` requests are built from them (`_autotag_request`,
    # `cmd_preprocess_captions`); the knobs the chain reasons about are
    # mirrored here so the run-or-skip logic below sees one dict.
    forms = gui_stage_values()
    autotag_form = forms.get("autotag")
    if isinstance(autotag_form, dict):
        config["autotag_form"] = autotag_form
        if autotag_form.get("mode"):
            config["autotag_mode"] = str(autotag_form["mode"]).strip()
        config["autotag_min_confidence"] = _floatish(
            autotag_form.get("min_confidence"), default=0.0
        )
    correct_form = forms.get("correct")
    if isinstance(correct_form, dict):
        config["correct_form"] = correct_form
        config["correct_order"] = not _boolish(correct_form.get("no_correct"), False)
        config["insert_no_artist"] = _boolish(
            correct_form.get("caption_insert_no_artist"), False
        )
        config["trigger_word"] = str(
            correct_form.get("caption_trigger_word") or ""
        ).strip()
        config["trigger_at_front"] = _boolish(
            correct_form.get("caption_trigger_at_front"), False
        )
        config["drop_groups"] = str(
            correct_form.get("caption_drop_groups") or ""
        ).strip()

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
        elif tok in {"--caption_drop_groups", "--caption-drop-groups"}:
            if i + 1 >= len(extra):
                raise SystemExit(f"{tok} requires a comma-separated group list")
            config["drop_groups"] = str(extra[i + 1]).strip()
            i += 2
        elif tok in {"--caption_position_clauses", "--caption-position-clauses"}:
            config["position_clauses"] = True
            i += 1
        elif tok in {
            "--no_caption_position_clauses",
            "--no-caption-position-clauses",
        }:
            config["position_clauses"] = False
            i += 1
        elif tok in {"--caption_autotag", "--caption-autotag"}:
            config["autotag"] = True
            i += 1
        elif tok in {"--no_caption_autotag", "--no-caption-autotag"}:
            config["autotag"] = False
            i += 1
        elif tok in {"--caption_autotag_mode", "--caption-autotag-mode"}:
            if i + 1 >= len(extra):
                raise SystemExit(f"{tok} requires a value ({'|'.join(AUTOTAG_MODES)})")
            config["autotag_mode"] = str(extra[i + 1]).strip()
            i += 2
        elif tok in {
            "--caption_autotag_min_confidence",
            "--caption-autotag-min-confidence",
        }:
            if i + 1 >= len(extra):
                raise SystemExit(f"{tok} requires a value")
            config["autotag_min_confidence"] = _floatish(extra[i + 1], default=0.0)
            i += 2
        elif tok in {"--caption_trigger_word", "--caption-trigger-word"}:
            if i + 1 >= len(extra):
                raise SystemExit(f"{tok} requires a value")
            config["trigger_word"] = str(extra[i + 1]).strip()
            i += 2
        else:
            cleaned.append(tok)
            i += 1

    # Fail fast: stage runs after resize, so a typo would otherwise surface
    # minutes into a GPU job.
    mode = str(config.get("autotag_mode") or "missing")
    if mode not in AUTOTAG_MODES:
        raise SystemExit(
            f"caption autotag mode {mode!r} is not one of {'|'.join(AUTOTAG_MODES)}"
        )
    config["autotag_mode"] = mode
    return config, cleaned


def _autotag_request(config: dict[str, object]):
    """The ``AutotagRequest`` the in-pipeline autotag stage runs as. Always
    ``apply`` — the user already opted in via the checkbox / env; a dry run
    here would produce a report nobody reads while TE encodes the un-tagged
    captions."""
    mode = str(config.get("autotag_mode") or "missing")
    min_confidence = float(config.get("autotag_min_confidence") or 0.0)
    form = config.get("autotag_form")
    if isinstance(form, dict):
        # The GUI's form: every other knob the stage has rides along, the
        # roots and the scope are the trainer's.
        return request_from_form(
            "autotag",
            form,
            roots=_stage_roots(),
            settings={"path_pattern": _stage_path_pattern(config)},
            apply=True,
            mode=mode,
            min_confidence=min_confidence,
        )
    from anime_tools.stages.requests import AutotagRequest

    return AutotagRequest(
        **_stage_roots(),
        path_pattern=_stage_path_pattern(config),
        mode=mode,
        min_confidence=min_confidence,
        apply=True,
    )


def _stage_roots() -> dict[str, str]:
    """The ``src`` / ``dst`` roots every caption stage binds."""
    return {
        "src": _path("source_image_dir", "image_dataset"),
        "dst": _path("resized_image_dir", "post_image_dataset/resized"),
    }


def _position_request(config: dict[str, object]):
    """The ``PositionRequest`` the in-pipeline position-clause stage runs as
    (``apply``, every detection / clause knob at the package default)."""
    from anime_tools.stages.requests import PositionRequest

    return PositionRequest(
        src=_path("source_image_dir", "image_dataset"),
        dst=_path("resized_image_dir", "post_image_dataset/resized"),
        path_pattern=_stage_path_pattern(config),
        apply=True,
    )


def _caption_correction_enabled(config: dict[str, object]) -> bool:
    """Run the caption-rewrite pass when ANY caption-rewriting knob is set.

    ``correct_captions.py`` is the only path that injects the trigger word /
    ``@no-artist``, so a trigger word or insert-no-artist with order
    correction *off* still has to run it — otherwise the GUI's trigger-word
    field is silently ignored at TE-cache time.
    """
    return bool(
        config.get("correct_order")
        or str(config.get("trigger_word") or "").strip()
        or config.get("insert_no_artist")
        or str(config.get("drop_groups") or "").strip()
    )


def _drop_groups_override(value) -> str:
    """``caption_drop_groups`` from the TOML chain: a string or a list."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _caption_correction_fields(config: dict[str, object]) -> dict[str, object]:
    """The ``CorrectRequest`` fields the correction knobs set — only the ones
    that differ from the request default, so ``{}`` means "nothing to
    correct" (the passthrough mirror)."""
    fields: dict[str, object] = {}
    if config.get("insert_no_artist"):
        fields["caption_insert_no_artist"] = True
    trigger = str(config.get("trigger_word") or "").strip()
    if trigger:
        fields["caption_trigger_word"] = trigger
    if config.get("trigger_at_front"):
        fields["caption_trigger_at_front"] = True
    drop = str(config.get("drop_groups") or "").strip()
    if drop:
        fields["caption_drop_groups"] = drop
    return fields


def _config_freefit_max_ratio() -> float | None:
    """``freefit_max_ratio`` from env (GUI auto-chain) or the merged config;
    ``None`` leaves the package default."""
    from ._common import _path_overrides

    raw = os.environ.get("FREEFIT_MAX_RATIO")
    if raw is None:
        raw = _path_overrides().get("freefit_max_ratio")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _resize_crop_fields() -> dict[str, object]:
    """``ResizeRequest`` crop fields from the merged config chain, only when
    they differ from the package default."""
    from library.preprocess.resize_preview import (
        DEFAULT_RESIZE_CROP_ANCHOR,
        RESIZE_CROP_ANCHORS,
        normalize_crop_margins,
    )

    from ._common import _path_overrides

    overrides = _path_overrides()
    fields: dict[str, object] = {}
    anchor = str(overrides.get("resize_crop_anchor") or DEFAULT_RESIZE_CROP_ANCHOR)
    anchor = anchor.strip()
    if anchor in RESIZE_CROP_ANCHORS and anchor != DEFAULT_RESIZE_CROP_ANCHOR:
        fields["resize_crop_anchor"] = anchor
    margins = normalize_crop_margins(overrides.get("resize_crop_margins"))
    values = tuple(margins[key] for key in ("top", "right", "bottom", "left"))
    if any(value > 0 for value in values):
        fields["resize_crop_margins"] = values
    return fields


# ``--curation_decisions <path>`` is the one resize flag the package's
# ``ResizeRequest`` has no field for that the trainer still honours (it becomes
# the request's ``skip`` set). The snap-era flags this used to swallow with a
# note (``--resize_bucket_resos`` / ``--bucket_reso_steps`` / ``--min_bucket_reso``
# / ``--max_bucket_reso`` / ``--resolution`` / ``--freefit``) were dropped in v2:
# they now fail the stage's own parse, which names the flags that do exist.
_CURATION_DECISIONS_FLAGS = ("--curation_decisions", "--curation-decisions")


def _pop_curation_decisions_arg(extra) -> tuple[list[str], str | None]:
    """Split ``--curation_decisions <path>`` out of ``ARGS``.

    Returns ``(cleaned, curation_decisions_path)``; everything else is left
    for the request's own parser.
    """
    cleaned: list[str] = []
    decisions: str | None = None
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok in _CURATION_DECISIONS_FLAGS:
            if i + 1 >= len(extra):
                raise SystemExit(f"{tok} requires a path")
            decisions = str(extra[i + 1])
            i += 2
            continue
        cleaned.append(tok)
        i += 1
    return cleaned, decisions


def _resize_request(
    src: str,
    dst: str,
    extra,
    *,
    min_pixels: int | None,
    path_pattern: str | None = None,
    target_res: tuple[int, ...] | None = None,
    prog: str = "make preprocess-resize ARGS=",
):
    """The ``ResizeRequest`` for ``src`` → ``dst``: trainer paths/config as the
    base, the user's ``ARGS`` applied through the request's own parser, the
    GUI's curation decisions as ``skip``."""
    from anime_tools.stages.requests import ResizeRequest

    cleaned, decisions = _pop_curation_decisions_arg(extra)
    skips = _curation_skips(src, Path(decisions) if decisions else None)
    form = _resize_form()
    if form is not None:
        # The GUI's resize form carries the geometry (tiers, crop, clamp,
        # overwrite, workers); the trainer fills the roots, the scope, the
        # walk and the curation skips, and the low-res sugar's answer.
        overrides: dict[str, object] = {
            "recursive": True,
            "excluded_dir": EXCLUDED_DIR,
        }
        if min_pixels is not None:
            overrides["min_pixels"] = int(min_pixels)
        if skips:
            overrides["skip"] = skips
        req = request_from_form(
            "resize",
            form,
            roots={"src": src, "dst": dst},
            settings={"path_pattern": path_pattern or "*"},
            **overrides,
        )
        return request_with_args(req, cleaned, prog=prog)
    fields: dict[str, object] = {
        "src": src,
        "dst": dst,
        "recursive": True,
        "excluded_dir": EXCLUDED_DIR,
        "path_pattern": path_pattern or "*",
        **_resize_crop_fields(),
    }
    if min_pixels is not None:
        fields["min_pixels"] = int(min_pixels)
    if target_res:
        fields["target_res"] = tuple(target_res)
    ratio = _config_freefit_max_ratio()
    if ratio is not None:
        fields["freefit_max_ratio"] = ratio
    if skips:
        fields["skip"] = skips
    try:
        req = ResizeRequest(**fields)
    except ValueError as exc:
        raise SystemExit(f"resize config: {exc}") from exc
    return request_with_args(req, cleaned, prog=prog)


def _min_pixels_value(mp_args: list[str]) -> int | None:
    """The ``--min_pixels N`` an argv helper produced, as the request field
    (``None`` = package default)."""
    return int(mp_args[1]) if mp_args else None


from library.datasets.curation_actions import EXCLUDED_DIR  # noqa: E402


def _curation_decisions_path() -> Path:
    path = Path(
        _path("curation_decisions", "post_image_dataset/curation_decisions.json")
    )
    return path if path.is_absolute() else ROOT / path


def _curation_skips(src: str, decisions_path: Path | None = None) -> tuple[str, ...]:
    """The images curation leaves out of preprocessing, as ``ResizeRequest.skip``
    entries (paths relative to ``src``): the GUI's decision file (``skip`` /
    ``move``) plus whatever the anime_tools GUI excluded in its own workspace
    ledger. The trainer's own exclusion ledger is not listed here — the stage
    reads it itself through ``excluded_dir``. Empty when neither exists, so a
    plain CLI preprocess is unchanged."""
    from library.datasets.curation_actions import (
        load_curation_decisions,
        workspace_excluded_rels,
    )

    skips: set[str] = set(workspace_excluded_rels())
    path = decisions_path or _curation_decisions_path()
    if path.is_file():
        decisions = load_curation_decisions(path, source_dir=src)
        skips.update(
            rel
            for rel, decision in decisions.items()
            if decision.get("action") in {"skip", "move"}
        )
    return tuple(sorted(skips))


def _repa_pe_encoder() -> str | None:
    """The REPA vision encoder to cache, or ``None`` when REPA is off.

    Reads ``use_repa`` / ``repa_encoder`` from the merged config chain, so a
    ``use_repa=true`` run auto-caches its PE sidecars in the preprocess pass
    instead of bouncing off train.py's "PE features absent" error. Plain
    ``make preprocess`` sees no ``use_repa`` and returns ``None``.
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

    A missing checkpoint would otherwise fall into a no-timeout
    ``hf_hub_download``; in the daemon's detached child that shows no progress,
    and because the queue is serial, a stalled/gated download wedges every job
    queued behind it (training included). Bail with an actionable message
    instead — run the named target manually for the one-time download."""
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
    """Strip resize-only flags from ``extra`` before cache stages run — the
    VAE/TE/PE stages read whatever latent shapes are already on disk and must
    never see resize-only argparse flags."""
    cleaned: list[str] = []
    it = iter(extra)
    for tok in it:
        if tok in {
            "--target_res",
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
        cleaned.append(tok)
    return cleaned


def _resolve_lowres_filter(extra) -> tuple[list[str], list[str]]:
    """Reconcile the low-res input filter against CLI ``ARGS``.

    Returns ``(min_pixels_args, cleaned_extra)`` with our two convenience
    flags popped so underlying scripts never see an arg their argparse
    doesn't define. Precedence (highest first): explicit ``--min_pixels N``
    in ``ARGS`` wins outright; ``--no_drop_lowres`` → ``--min_pixels 0``
    (keep every image); ``--drop_lowres`` → force the configured threshold;
    neither → fall back to the merged-config behavior (``_min_pixels_args``).
    """
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


def cmd_preprocess_resize(extra, *, chained: bool = False):
    """Resize the caption master into the bucket tree — the ``anime_tools``
    resize stage as a ``ResizeRequest`` (config chain + GUI env as the base,
    ``ARGS`` on top, the GUI's curation decisions as ``skip``).

    ``chained`` marks the call from :func:`cmd_preprocess`, where the cache
    stages that follow already receive the same ``ARGS`` — only a standalone
    ``make preprocess-resize`` needs the re-crop warning below.
    """
    mp_args, extra = _resolve_lowres_filter(extra)
    req = _resize_request(
        _path("source_image_dir", "image_dataset"),
        _path("resized_image_dir", "post_image_dataset/resized"),
        extra,
        min_pixels=_min_pixels_value(mp_args),
        path_pattern=_preprocess_path_pattern(),
        target_res=_config_target_res(),
    )
    _execute("resize", req)
    if req.overwrite and not chained:
        # A re-crop (crop anchor / margins / freefit_max_ratio) rewrites the PNG
        # at the SAME bucket, and the downstream skips are keyed on the bucket
        # (latents: `latents_{H}x{W}` present) / on sidecar existence (PE), not
        # on the pixels — so they would keep caches of the old crop.
        # `preprocess-reconcile` doesn't catch it either: nothing moved bucket.
        print(
            "  [preprocess] --overwrite re-wrote resized images: re-run "
            "`make preprocess-vae ARGS=--overwrite` (and preprocess-pe, if you "
            "cache PE features) or the latent/PE caches keep the old crop. "
            "`make preprocess ARGS=--overwrite` forwards it to every stage."
        )


def cmd_preprocess_reconcile(extra):
    """Remove caches stale for the configured ``target_res`` (dry-run by default).

    Pass ``ARGS="--delete"`` to actually remove. ``target_res`` comes from the
    merged config (same as resize); an explicit ``--target_res`` in ``ARGS``
    wins. Run after adding/dropping a tier so preprocess + mask regenerate
    only the images whose bucket moved.
    """
    # _target_res_args returns [] both for a bare [1024]/absent config and when
    # ARGS already carries --target_res; inject the 1024 default only for the former.
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
    # sigma_demote in preprocess.toml chains the demote emit(s) here so a
    # --sigma_lowres run never trains against a stale/missing demoted cache.
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
    """Emit σ-demote sibling latents (e.g. 1024→896) for ``--sigma_lowres``.

    Same VAE-load path as ``preprocess-vae``; appends a ``demoted_{H}x{W}``
    key inside each native-tier image's existing npz. Idempotent. Requires
    ``preprocess-vae`` to have run first.

    Routes come from ``sigma_demote`` in ``configs/preprocess.toml`` (or the
    ``SIGMA_DEMOTE`` env var); a comma list like ``"1024:896,1024:768"`` emits
    both siblings the stacked router (``--sigma_lowres_route2``) needs.
    ``ARGS="--sigma_demote N:D[,N:D…]"`` overrides (probe a new route before
    shipping it); default is the certified ``1024:896``.
    """
    routes, extra = _pop_explicit_demote_routes(extra)
    if not routes:
        routes = _sigma_demote_routes(extra) or ["1024:896"]
    for route in routes:
        if len(routes) > 1:
            print(f"  [preprocess] sigma_demote={route} → emitting demoted siblings")
        _run_demote_pass(route, extra)


_QWEN3_TOKENIZER = "models/text_encoders/qwen_3_06b_base.safetensors"


def _variant_settings() -> tuple[str, str, str]:
    """Caption-variant knobs: env override → preprocess.toml → default.

    Returns ``(shuffle_variants, tag_dropout_rate, tag_randomize_rate)`` as raw
    strings (forwarded straight to the script). CAPTION_SHUFFLE_VARIANTS /
    CAPTION_TAG_DROPOUT_RATE / CAPTION_TAG_RANDOMIZE_RATE let the GUI tune
    these without editing config.
    """
    shuffle = os.environ.get("CAPTION_SHUFFLE_VARIANTS") or _path(
        "caption_shuffle_variants", "4"
    )
    dropout = os.environ.get("CAPTION_TAG_DROPOUT_RATE") or _path(
        "caption_tag_dropout_rate", "0.1"
    )
    # Identity-randomized r-family tag regularization; 0.0 = off (no r-family
    # written, backward compatible).
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

    ``correct_captions.py`` loads ``danbooru_tags_classified.csv`` and
    ``SystemExit``s if it's missing — GUI users reach preprocess without
    ``make download-danbooru-tags``. Best-effort: catch ``SystemExit``/
    ``OSError`` so a failed download skips rather than aborts.
    """
    from anime_tools.captions.correction import find_tag_csv

    if find_tag_csv(ROOT) is not None:
        return
    print("  [preprocess] danbooru tag KB missing; fetching it for caption correction")
    try:
        # Base CSV only; the English sibling (download-danbooru-tags) is a
        # heavier wiki-join step only the GUI tooltip uses.
        from .downloads import _download_danbooru_base

        _download_danbooru_base([])
    except (SystemExit, OSError) as e:
        print(f"  [preprocess] danbooru tag KB auto-download failed: {e}")


def cmd_preprocess_captions(extra, caption_config: dict[str, object] | None = None):
    """Write corrected/variant caption sidecars into ``resized/``.

    Runs whenever caption order-correction is enabled **or** variants are
    requested (default ``caption_shuffle_variants=4``). Order-correction off +
    variants on runs in passthrough (``--no_correct``): v0 mirrors the raw
    caption with the shuffle/dropout/randomize sidecars alongside.

    The caption-rewrite stages (autotag, position clauses) run first when
    their config knob is on, so a tag/clause they add is visible to the
    sidecars this step writes and to the TE caches encoded from them. In the
    full ``preprocess`` chain both already ran earlier and the guard makes
    these calls no-ops.
    """
    if caption_config is None:
        caption_config, extra = _caption_correction_config(extra)
    _run_caption_autotag_stage(caption_config)
    _run_caption_position_stage(caption_config)
    correct = _caption_correction_enabled(caption_config)
    shuffle, dropout, randomize = _variant_settings()
    n_variants = int(_float_or_zero(shuffle))
    # Position clauses keep the mirror alive even with correction and variants
    # both off: they only rewrite resized/, so every OTHER image still needs
    # its master caption mirrored there or it would encode as empty.
    if not correct and n_variants <= 0 and not caption_config.get("position_clauses"):
        print("  [preprocess] caption correction disabled")
        return
    # correct_captions loads the Danbooru tag KB unconditionally — fetch it
    # on demand so a GUI preprocess that skipped the download doesn't abort.
    _ensure_danbooru_tags()
    from anime_tools.stages.requests import CorrectRequest

    # The trainer's own fields: the walk and the variant sidecars (the
    # TextCachingSection's knobs), plus the tokenizers identity-randomize needs.
    trainer_fields: dict[str, object] = {"recursive": True}
    if n_variants > 0:
        trainer_fields["caption_shuffle_variants"] = n_variants
        trainer_fields["caption_tag_dropout_rate"] = _float_or_zero(dropout)
        trainer_fields["caption_tag_randomize_rate"] = _float_or_zero(randomize)
        # Identity-randomize needs the tokenizers to build the erasure pool.
        # The curation-side stage loads tokenizers from *directories* only
        # (it must not know the safetensors→bundled-config mapping), so
        # resolve them here on the trainer side.
        if _float_or_zero(randomize) > 0.0 and n_variants >= 2:
            from library.anima.weights import qwen3_tokenizer_dir, t5_tokenizer_dir

            trainer_fields["qwen3"] = qwen3_tokenizer_dir(_QWEN3_TOKENIZER)
            trainer_fields["t5_tokenizer_path"] = t5_tokenizer_dir()
    path_pattern = _resolved_path_pattern(extra)
    form = caption_config.get("correct_form")
    if isinstance(form, dict):
        # The GUI's form supplies the stage's long tail (tag_csv, …); the
        # five rewrite knobs are set from the config dict, which the form was
        # folded into (so an explicit CLI flag still wins), and the trainer's
        # own fields on top.
        req = request_from_form(
            "correct",
            form,
            roots=_stage_roots(),
            settings={"path_pattern": path_pattern},
            # Always ``apply``, for the same reason autotag is: the correction
            # is the caption TE is about to encode, so a dry run would leave a
            # report nobody reads and cache the un-corrected text.
            apply=True,
            no_correct=not correct,
            caption_insert_no_artist=bool(caption_config.get("insert_no_artist")),
            caption_trigger_word=str(caption_config.get("trigger_word") or ""),
            caption_trigger_at_front=bool(caption_config.get("trigger_at_front")),
            caption_drop_groups=str(caption_config.get("drop_groups") or ""),
            **trainer_fields,
        )
    else:
        fields: dict[str, object] = {
            **_stage_roots(),
            "path_pattern": path_pattern,
            "apply": True,
            **trainer_fields,
        }
        if correct:
            fields.update(_caption_correction_fields(caption_config))
        else:
            fields["no_correct"] = True
        try:
            req = CorrectRequest(**fields)
        except ValueError as exc:
            raise SystemExit(f"caption correction: {exc}") from exc
    _execute("correct", req)


def cmd_preprocess_te(extra, caption_config: dict[str, object] | None = None):
    if caption_config is None:
        caption_config, extra = _caption_correction_config(extra)
    # Caption rewrites before anything reads the captions. `cmd_preprocess_captions`
    # runs them too, but the no-correction + no-variants path below skips that
    # step entirely and encodes the source captions directly.
    _run_caption_autotag_stage(caption_config)
    _run_caption_position_stage(caption_config)
    shuffle, dropout, randomize = _variant_settings()
    n_variants = int(_float_or_zero(shuffle))
    # The caption step writes the variant sidecars whenever correction is on OR
    # variants are requested; TE then reads resized/ (min_pixels=0) and encodes
    # the sidecars verbatim. Only pure no-correction + no-variants reads the
    # source captions directly. Position clauses force it too: they're written
    # into resized/ and never into the master, so encoding the master directly
    # would silently train the pre-clause caption.
    needs_caption_step = (
        _caption_correction_enabled(caption_config)
        or n_variants > 0
        or bool(caption_config.get("position_clauses"))
    )
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
    _release_stage_models()
    # CJK vocab pack from the config chain ("" = off): the caches must be
    # encoded through the same pack train.py / inference.py will route with.
    vocab_pack = _path("vocab_pack", "")
    pack_args = ["--vocab_pack", vocab_pack] if vocab_pack else []
    run(
        [
            PY,
            "scripts/preprocess/cache_text_embeddings.py",
            *pack_args,
            "--dir",
            text_dir,
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            *match_args,
            "--qwen3",
            _QWEN3_TOKENIZER,
            "--dit",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            # Fallback only — ignored when a {stem}.variants.txt sidecar is
            # present; drives in-process generation otherwise.
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

    Reads pre-resized images from ``post_image_dataset/resized/`` and writes
    ``{stem}_anima_pe.safetensors`` sidecars into the LoRA cache dir, consumed
    by IP-Adapter. Also emits the dataset-mean PE centroid sidecar
    (``post_image_dataset/ip_adapter/anima_pe_centroid_pe.safetensors``) via
    ``--centroid`` so IP-Adapter mean-centering needs no separate pass.
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
    """Cache PE-Spatial-B16-512 dense patch-token features for REPA.

    Reads pre-resized images from ``post_image_dataset/resized/`` and writes
    ``{stem}_anima_pe_spatial.safetensors`` sidecars (disjoint from the
    PE-Core ``_anima_pe`` caches CMMD reads). No centroid — REPA aligns
    per-patch, not against a dataset mean. Run before a ``use_repa=true``
    training arm.
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
    writes ``post_image_dataset/captions/caption_index.json``. Pure data, no
    GPU. Consumed by the IP-Adapter distinct-pair sampler, artist balancing,
    and dataset analytics. Regenerate when the dataset or vocab changes.
    """
    pp_args = _preprocess_path_pattern_args(extra)
    run(
        [
            PY,
            "-m",
            "anime_tools.captions.index",
            "--src",
            _path("source_image_dir", "image_dataset"),
            # The package's default moved to its own workspace/ tree (0.4.0);
            # the trainer's readers (train.py, configs/easycontrol/*.toml) keep
            # the post_image_dataset/ home.
            "--out",
            CAPTION_INDEX_PATH,
            *pp_args,
            *extra,
        ]
    )


def _stage(stage_id: str):
    return stage_by_id(stage_id)


def _execute(stage_id: str, req) -> None:
    """Run one curation stage: in-process under a daemon job, else a child.
    A GPU stage run in-process leaves its model cached in this interpreter
    (that is the point — autotag → position share one tagger), so the chain
    releases it before handing the GPU to a trainer-side child
    (``_release_stage_models``)."""
    execute_stage(_stage(stage_id), req)
    if stage_id in _GPU_STAGES and in_daemon_job():
        _MODELS_RESIDENT.add(stage_id)


_GPU_STAGES = {"autotag", "position", "ocr"}
_MODELS_RESIDENT: set[str] = set()


def _release_stage_models() -> None:
    """Drop the tagger / SAM3 an in-process caption stage left resident so
    the VAE / TE child that follows gets the VRAM. No-op from a shell (each
    stage was its own child) and when nothing ran."""
    if not _MODELS_RESIDENT:
        return
    from anime_tools.stages import release_models

    release_models()
    _MODELS_RESIDENT.clear()


def _caption_request(cls, extra, *, prog: str):
    """A standalone caption target's request: trainer paths + the env/config
    subset scope as the base, the user's ``ARGS`` on top (an explicit
    ``--path_pattern`` there overrides the scope, once)."""
    req = cls(
        src=_path("source_image_dir", "image_dataset"),
        dst=_path("resized_image_dir", "post_image_dataset/resized"),
        path_pattern=_preprocess_path_pattern(),
    )
    return request_with_args(req, extra, prog=prog)


def _caption_autotag_request(extra):
    """The ``AutotagRequest`` for ``make caption-autotag ARGS=…`` — shared
    with the in-pipeline stage's paths/scoping so the two can't drift."""
    from anime_tools.stages.requests import AutotagRequest

    return _caption_request(AutotagRequest, extra, prog="make caption-autotag ARGS=")


def _caption_position_request(extra):
    """The ``PositionRequest`` for ``make caption-position ARGS=…``."""
    from anime_tools.stages.requests import PositionRequest

    return _caption_request(PositionRequest, extra, prog="make caption-position ARGS=")


def _ocr_dir() -> str:
    """The OCR sidecar tree — ``ocr_dir`` from the merged config chain."""
    return _path("ocr_dir", DEFAULT_OCR_DIR)


def _caption_ocr_request(*, path_pattern: str, apply: bool, device: str | None):
    """The ``OcrRequest`` the ``caption-full`` chain reads text with.

    Reads the resized tree and writes ``{stem}.ocr.txt`` under ``ocr_dir``;
    no caption is read or written, so this step alone invalidates no TE cache.
    Every detector/reader knob stays at the package default — the two floors
    that decide what reaches a caption live on the combine step, not here, so
    the sidecar keeps every line for a person to look at.
    """
    from anime_tools.stages.requests import OcrRequest

    return OcrRequest(
        dst=_path("resized_image_dir", "post_image_dataset/resized"),
        ocr_dir=_ocr_dir(),
        path_pattern=path_pattern,
        apply=apply,
        device=device,
    )


def _caption_combine_request(
    *, apply: bool, min_det: float | None = None, min_glyph: float | None = None
):
    """The ``ExportRequest`` that attaches the OCR clause to the trainer's captions.

    ``with_ocr_clause`` — the one place an OCR sidecar meets a caption — is
    reachable only through the export stage's ``--combine_ocr``, which is
    written as a workspace→trainer publish. The trainer *is* its own workspace
    here (the caption stages write ``post_image_dataset/resized`` directly), so
    the export runs **in place**: ``out`` is the resized tree's parent, and
    every row but ``caption``/``variants`` compares identical and is skipped
    (verified on the live tree: 3,008 image + 873 mask + 1 index rows
    identical, 700 captions and 700 variant sidecars combined).

    ``master`` and ``excluded_dir`` keep the package's workspace defaults —
    absent trees contribute no rows, so nothing is ever written back over the
    hand-written masters under ``image_dataset/``.

    The combine is idempotent: a text clause the caption already carries is
    replaced, and a re-run whose sidecar lost its lines *removes* the clause.
    """
    from anime_tools.captions.ocr_sidecar import DEFAULT_MIN_DET, DEFAULT_MIN_GLYPH
    from anime_tools.stages.requests import ExportRequest

    resized = Path(_path("resized_image_dir", "post_image_dataset/resized"))
    return ExportRequest(
        src=_path("source_image_dir", "image_dataset"),
        dst=str(resized),
        masks=_path("mask_dir", "post_image_dataset/masks"),
        index=CAPTION_INDEX_PATH,
        # `out/resized/<rel>` is where a caption row lands — the tree it was
        # read from, which is what makes this an in-place combine.
        out=str(resized.parent),
        combine_ocr=True,
        ocr_dir=_ocr_dir(),
        ocr_min_det=DEFAULT_MIN_DET if min_det is None else min_det,
        ocr_min_glyph=DEFAULT_MIN_GLYPH if min_glyph is None else min_glyph,
        apply=apply,
    )


# Caption-rewrite stages (autotag -> image_dataset/*.txt, position clauses ->
# resized/*.txt) must run before anything reads/mirrors a caption; every entry
# point that needs them calls the stage. This key on the shared caption-config
# dict records that a stage already ran in this chain — `cmd_preprocess`
# threads ONE dict through so later calls no-op, while a standalone target
# gets its own dict and runs it. Both passes are idempotent but not free
# (each pays a tagger/SAM3 load over the whole tree), hence the guard.
_STAGE_RAN_KEY = "_master_stages_ran"


def _stage_already_ran(config: dict[str, object], stage: str) -> bool:
    """Has ``stage`` run for this caption-config dict? Marks it if not."""
    ran = config.setdefault(_STAGE_RAN_KEY, set())
    if not isinstance(ran, set):  # a caller hand-rolled the dict — treat as fresh
        ran = set()
        config[_STAGE_RAN_KEY] = ran
    if stage in ran:
        return True
    ran.add(stage)
    return False


def _stage_path_pattern(config: dict[str, object]) -> str:
    """Subset scope stashed by :func:`_caption_correction_config`. A
    hand-rolled dict without one falls back to the env/config pattern exactly
    as the standalone targets do."""
    pattern = config.get("path_pattern")
    return str(pattern) if pattern else _preprocess_path_pattern()


def _run_caption_autotag_stage(config: dict[str, object]) -> None:
    """Run the in-pipeline autotag pass if enabled and not yet run."""
    if not config.get("autotag") or _stage_already_ran(config, "autotag"):
        return
    mode = str(config.get("autotag_mode") or "missing")
    print(f"  [preprocess] autotag ({mode}): Anima Tagger → revised captions")
    _execute("autotag", _autotag_request(config))


def _run_caption_position_stage(config: dict[str, object]) -> None:
    """Run the in-pipeline position-clause pass if enabled and not yet run.

    Inline rather than through ``cmd_caption_position``: the caller is itself
    a daemon job on a serial queue, and a nested job would wait on a queue
    that can't advance.
    """
    if not config.get("position_clauses") or _stage_already_ran(config, "position"):
        return
    print("  [preprocess] position clauses: SAM3 + tagger → resized captions")
    _execute("position", _position_request(config))


def cmd_caption_autotag(extra):
    """Auto-tag the dataset with the Anima Tagger (GPU, daemon-routed).

    Writes ``.txt`` sidecars into the caption master. ``--mode missing``
    (default) only fills in uncaptioned images; ``merge`` appends novel tags to
    every caption while keeping its position clauses; ``overwrite`` replaces
    the caption outright. Dry-run by default; ``ARGS="--apply"`` writes, and
    must be followed by ``make preprocess-te`` — caption edits do NOT
    invalidate the TE caches.
    """
    from ._common import _resolve_run_mode, run_command

    mode, extra = _resolve_run_mode(extra)
    req = _caption_autotag_request(extra)
    run_command(
        "caption-autotag", ["-m", _stage("autotag").module, *req.to_argv()], mode=mode
    )


def cmd_caption_position(extra):
    """Append position-aware clauses to multi-subject captions (GPU, daemon-routed).

    SAM3 ``girl`` instances -> reading order -> mask-blanked crops -> Anima
    Tagger -> ``... On the left, <tags>. On the right, <tags>.`` written into
    the **derived** caption (``post_image_dataset/resized/<rel>.txt``); the
    hand-written master under ``image_dataset/`` is never touched. Dry-run by
    default; ``ARGS="--apply"`` writes, and must be followed by
    ``make preprocess-te`` to re-encode.
    """
    from ._common import _resolve_run_mode, run_command

    mode, extra = _resolve_run_mode(extra)
    req = _caption_position_request(extra)
    run_command(
        "caption-position",
        ["-m", _stage("position").module, *req.to_argv()],
        mode=mode,
    )


def _warn_if_te_would_read_the_master() -> None:
    """Shout when ``make preprocess-te`` would encode ``image_dataset/`` instead.

    Everything ``caption-full`` writes lives in the **derived** tree, and
    ``cmd_preprocess_te`` only reads that tree when something forces the caption
    step — correction, shuffle variants, or the ``caption_position_clauses``
    config flag. With all three off it encodes the masters directly and
    ``--match_images_from`` hides the difference: the run is silently discarded,
    clauses and all. The flag (not the clauses on disk) is what the chain keys
    on, so a hand-run ``caption-full`` on an otherwise-bare config is the one
    way to land here.
    """
    config, _ = _caption_correction_config([])
    n_variants = int(_float_or_zero(_variant_settings()[0]))
    if _caption_correction_enabled(config) or n_variants or config["position_clauses"]:
        return
    print(
        "  [caption-full] WARNING: with caption correction off, "
        "caption_shuffle_variants = 0 and caption_position_clauses unset, "
        "`make preprocess-te` encodes the image_dataset/ MASTERS — not the "
        "captions just written. Set `caption_position_clauses = true` in the "
        "config chain (or CAPTION_POSITION_CLAUSES=1) so TE reads "
        "post_image_dataset/resized/."
    )


def _caption_full_args(extra):
    """``make caption-full ARGS=…`` — the chain's own small flag set.

    Deliberately not ``request_with_args``: three stages run here, so an
    unqualified package flag would be ambiguous. Only the knobs that mean
    something for the *chain* are exposed; per-stage tuning still goes through
    ``make caption-position`` / the OCR stage's own ``-m`` invocation.
    """
    import argparse

    ap = argparse.ArgumentParser(prog="make caption-full ARGS=", add_help=False)
    ap.add_argument("--dry_run", action="store_true", help="plan only (default: write)")
    # Accepted and ignored: the other caption targets need it, and typing it
    # here should not be an error.
    ap.add_argument("--apply", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--path_pattern", default=_preprocess_path_pattern())
    ap.add_argument("--skip_position", action="store_true")
    ap.add_argument("--skip_ocr", action="store_true")
    ap.add_argument("--ocr_min_det", type=float, default=None)
    ap.add_argument("--ocr_min_glyph", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("-h", "--help", action="help")
    return ap.parse_args(extra)


def cmd_caption_full(extra):
    """Position clauses -> OCR read -> OCR clause, over the resized captions.

    The whole caption chain that runs on the **derived** tree, in the one order
    that composes (GPU, daemon-routed as a single job):

    1. **position** — SAM3 + Anima Tagger write ``On the left, <tags>.`` into
       ``post_image_dataset/resized/<rel>.txt`` and drop stale variant sidecars.
    2. **ocr** — AnimeText + the manga VL reader write ``{stem}.ocr.txt`` under
       ``post_image_dataset/ocr/``. Touches no caption.
    3. **combine** — the OCR'd lines are attached to the caption and to every
       variant line as trailing text clauses (``Japanese text reads as "…"``,
       ``Japanese SFX reads as "…"``), held to the det/glyph floors.

    Position first because the combine parses the caption it lands on and keeps
    its position clauses; OCR before the combine because the combine reads the
    sidecars the OCR pass writes.

    **Writes by default** — unlike ``caption-autotag`` / ``caption-position``,
    whose dry run guards the hand-written master under ``image_dataset/``.
    Nothing here can reach that tree: every step writes the derived,
    regenerable one (``post_image_dataset/resized`` and ``ocr/``), so a plan
    nobody reads is just a second GPU pass. ``ARGS="--dry_run"`` plans instead.

    Follow it with ``make preprocess-te``, which needs no ``--overwrite``: TE
    caches are mtime-aware, so only the stems whose caption changed re-encode.

    A **dry run of the whole chain reports the combine against the sidecars
    already on disk**, not against the ones step 2 would have written — the
    first ever dry run over a tree with no OCR tree shows no combined rows.

    ``ARGS="--skip_position --skip_ocr --ocr_min_det 0.6"`` re-combines from the
    sidecars already read, which is how the two floors get retuned without
    paying for either GPU pass again.
    """
    from ._common import _resolve_run_mode, run_command

    mode, extra = _resolve_run_mode(extra)
    if mode != "inline":
        # One daemon job for the whole chain: re-enter this target inside it
        # (`in_daemon_job()` then routes every stage in-process, so the tagger
        # SAM3 load is paid once) instead of submitting three jobs to a serial
        # queue. The stall watchdog is off — the VL reader's first-use fetch
        # (~2.8 GB) and a quiet model load both print nothing for minutes.
        run_command(
            "caption-full",
            ["tasks.py", "caption-full", "--inline", *extra],
            mode=mode,
            stall_timeout=0,
        )
        return

    args = _caption_full_args(extra)
    args.apply = not args.dry_run
    if not args.skip_position:
        print("  [caption-full] position clauses: SAM3 + tagger → resized captions")
        req = _caption_position_request(
            ["--path_pattern", args.path_pattern, *(["--apply"] if args.apply else [])]
            + (["--device", args.device] if args.device else [])
        )
        _execute("position", req)
    if not args.skip_ocr:
        print("  [caption-full] ocr: AnimeText + VL reader → post_image_dataset/ocr")
        # SAM3 + the tagger are done with; the VL reader wants the VRAM.
        _release_stage_models()
        _execute(
            "ocr",
            _caption_ocr_request(
                path_pattern=args.path_pattern,
                apply=args.apply,
                device=args.device,
            ),
        )
    print("  [caption-full] combine: OCR lines → text clauses on the resized captions")
    _release_stage_models()
    _execute(
        "export",
        _caption_combine_request(
            apply=args.apply,
            min_det=args.ocr_min_det,
            min_glyph=args.ocr_min_glyph,
        ),
    )
    if args.apply:
        print(
            "  [caption-full] captions rewritten — run `make preprocess-te` to "
            "re-encode them (TE is mtime-aware; no --overwrite needed)."
        )
        _warn_if_te_would_read_the_master()
    else:
        print("  [caption-full] dry run — nothing written (--dry_run).")


# `cmd_preprocess` auto-fetches this (~0.7 MB) vocab on demand: the caption index
# it gates is a hard requirement for soft-tokens contrastive training (train.py
# raises FileNotFoundError without it). Fetch is best-effort.
_CAPTION_INDEX_VOCAB = "models/captioners/anima-tagger-dbv4/vocab.json"


def cmd_preprocess(extra):
    """Full pipeline: resize -> VAE latents -> (caption stages) -> text
    embeddings -> caption index (-> REPA PE features if ``use_repa=true``).

    Chain order is pinned (and covered by a test): caption_autotag runs right
    after resize because it *creates* the caption master every later caption
    stage reads; position_clauses runs after the VAE pass, before TE, because
    it writes the derived caption in ``resized/`` that TE encodes.
    """
    caption_config, extra = _caption_correction_config(extra)
    # PE features are NOT cached here by default (CMMD chains `preprocess-pe`
    # explicitly) — keeps the default LoRA preprocess fast. Exception:
    # `use_repa=true` chains them at the end (see `_repa_pe_encoder()` below).
    #
    # Fail fast BEFORE any GPU work: a use_repa=true auto-chain with a missing
    # REPA checkpoint would stall the PE step on a silent daemon download and
    # wedge the serial queue.
    encoder = _repa_pe_encoder()
    if encoder is not None:
        _require_repa_encoder_model(encoder)
    cmd_preprocess_resize(extra, chained=True)
    _run_caption_autotag_stage(caption_config)
    # VAE/TE steps read on-disk shapes — strip the low-res convenience flags AND
    # the resize-only --target_res so their argparse never sees an undefined arg.
    downstream = _pop_resize_only_args(extra)
    _, vae_extra = _resolve_lowres_filter(downstream)
    _release_stage_models()
    cmd_preprocess_vae(vae_extra)
    _run_caption_position_stage(caption_config)
    cmd_preprocess_te(downstream, caption_config=caption_config)
    # Caption index as a free by-product — consumed by the IP-Adapter pair sampler,
    # artist balancing, analytics, AND soft-tokens (which hard-errors without it).
    vocab = _path("caption_index_vocab", _CAPTION_INDEX_VOCAB)
    if not os.path.exists(vocab):
        # GUI users reach preprocess without `make download-models`; fetch the
        # tiny tagger vocab on demand, skip rather than abort on failure.
        print("  [preprocess] tagger vocab missing; fetching it for caption-index")
        try:
            from .downloads import cmd_download_tagger

            cmd_download_tagger([])
        except (SystemExit, OSError) as e:
            print(f"  [preprocess] tagger vocab auto-download failed: {e}")
    if os.path.exists(vocab):
        # Caption index intentionally stays on source captions — its consumers
        # care about tag presence/relations, not corrected order.
        cmd_caption_index([])
    else:
        print(
            f"  [preprocess] skipping caption-index: tagger vocab not found at "
            f"{_CAPTION_INDEX_VOCAB} and auto-download failed. Run "
            f"`make download-tagger`, then `make caption-index` "
            f"(soft-tokens contrastive training needs it)."
        )

    # REPA arm: chain the PE sidecars REPA aligns against (train.py errors
    # without them). `encoder` was resolved (and its checkpoint required) above.
    if encoder is not None:
        print(f"  [preprocess] use_repa=true → caching REPA PE features ({encoder})")
        if encoder == "pe_spatial":
            cmd_preprocess_pe_spatial([])
        else:
            cmd_preprocess_pe([])


def cmd_preprocess_config(extra):
    """Preprocess the exact directories named in a ``--dataset_config`` TOML.

    Unlike ``cmd_preprocess`` (repo's standard ``image_dataset/`` ->
    ``post_image_dataset/`` layout), this drives off the same dataset config
    the *training* job will consume, so one file fully describes an ad-hoc
    job. For each ``[[datasets.subsets]]`` it bucket-resizes ``--src`` into
    that subset's ``image_dir`` (source never modified), caches VAE latents,
    then caches text embeddings (captions read from ``--src``).

    ``--src`` is required because a config's ``image_dir`` is the post-resize
    dir training reads, not the originals. ``--vae``/``--qwen3``/``--dit``
    override the config-resolved model paths (e.g. the ComfyUI trainer node
    points these at ComfyUI's own ``folder_paths``).

    Usage: ``preprocess-config --dataset_config <path> --src <dir>
    [--vae <path>] [--qwen3 <path>] [--dit <path>] [extra…]``
    (remaining args forward to the resize step).
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
    # Defender) can briefly lock a just-created config file.
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
        # bucket-resize originals -> image_dir; cache_latents.py keys caches by
        # on-disk size, so the resized size must match what the trainer expects.
        # min_pixels=0: an ad-hoc job keeps every image the config names.
        _execute(
            "resize",
            _resize_request(
                src_dir,
                image_dir,
                rest,
                min_pixels=0,
                prog="make preprocess-config ARGS=",
            ),
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
        # text embeddings — captions read from --src
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
