"""Mask generation: SAM3 rule passes → merged.

``make mask`` is a one-shot orchestrator: it runs every SAM rule into a
``tempfile.TemporaryDirectory()`` (cross-platform — honors ``TMPDIR`` /
``TEMP``) and writes only the merged result to
``<mask_dir>/<rel>/{stem}_mask.png``, where ``mask_dir`` comes from the
merged config chain (``configs/preprocess.toml``, default
``post_image_dataset/masks``). Per-rule intermediates are never persisted
under the project root. The MIT / ComicTextDetector text masker was removed
in v2 (text is no longer masked automatically).

Every stage runs as an ``anime_tools`` **request object**
(``anime_tools.masking.requests.{SamMaskRequest,MergeMasksRequest}``) — the
trainer never spells a flag. How a request is executed depends on where
``make mask`` runs (``_execute``):

- **Under a daemon job** (``ANIMA_DAEMON_JOB_DIR`` set — every GUI run, and
  ``make daemon-run ARGS="tasks.py mask"``) the stage runners are called
  **in-process** through the package registry (``Stage.runner()``): one
  interpreter for the whole chain, one SAM3 load shared by every rule pass
  (``load_sam3`` is cached per process), and the package's ``_progress``
  heartbeat keeps a quiet model load from reading as a stall to the daemon's
  watchdog. The job process exits at the end, so VRAM is released as before.
- **From a plain shell** each stage is a ``python -m <stage.module>`` child
  with ``req.to_argv()``, so ``make mask`` still releases the model between
  stages and on exit.

The switch is ``_common.execute_stage`` — the same one the caption stages
(``preprocess.py``) and grouping (``curate.py``) run through.

Where the rules come from:

- **The GUI** sends its SAM rule cards as ``masks_sam`` stage forms
  (``PREPROCESS_STAGES_JSON``, one ``{dest: value}`` per card, each carrying
  its own ``path_pattern`` scope); ``_common.request_from_form`` builds each
  ``SamMaskRequest`` through the package's ``build_argv`` with the resized
  tree and a per-card tempdir as the roots.
- **The CLI** reads ``configs/sam_mask.yaml``: a flat ``prompts`` /
  ``focus_prompts`` pair or a ``rules:`` list routed by ``path_pattern``,
  plus the optional ``run_sam`` switch. Every knob absent from the config
  falls back to the package's request default; the trainer carries no
  literal of its own.

Either way a rule becomes one SAM pass into its own temp dir; the merge
step's pixel-min union then composes them exactly as the old single-pass
``rules`` did (ignore regions unioned).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ._common import (
    ROOT,
    _path,
    execute_stage,
    gui_stage_values,
    request_from_form,
    stage_by_id,
)

DEFAULT_MASK_DIR = "post_image_dataset/masks"
RESIZED_IMAGE_DIR = ROOT / "post_image_dataset" / "resized"
SAM_CONFIG = ROOT / "configs" / "sam_mask.yaml"


def _mask_output_dir() -> Path:
    """Unscoped mask root — ``mask_dir`` from the merged config chain.

    Owned by ``configs/preprocess.toml`` (preserved across ``make update``);
    a preset/method/GUI snapshot may override it. Kept a function rather than
    a module constant so a ``CONFIG_FILE`` snapshot is read at call time.
    """
    return ROOT / _path("mask_dir", DEFAULT_MASK_DIR)


def _resized_image_dir() -> Path:
    """Scoped resized dir to mask, honoring GUI ``path_scope``.

    Reads ``resized_image_dir`` from the merged config chain (the GUI passes a
    config snapshot via ``CONFIG_FILE`` whose ``resized_image_dir`` is already
    scoped to ``post_image_dataset/resized/<path_scope>``). Scoping the input
    is what stops a scoped run from re-masking every other folder. Without a
    snapshot (direct ``make mask``) this falls back to the unscoped default, so
    CLI behavior is unchanged.
    """
    return ROOT / _path("resized_image_dir", "post_image_dataset/resized")


def _scoped_mask_output_dir(resized_dir: Path) -> Path:
    """Re-apply the ``path_scope`` offset onto the mask output root.

    SAM emits masks with rel paths taken **relative to the scoped resized
    dir** (``resized/<scope>``), so a scoped run drops the ``<scope>`` prefix.
    But training resolves masks relative to the **unscoped** cache root
    (``lora/<scope>/<rel>`` → ``masks/<scope>/<rel>``, see
    ``CachedDataset._resolve_mask_path``), so masking must land them under
    ``masks/<scope>`` — not flat in ``masks/`` — or the trainer won't find
    them. Mirror whatever scope ``resized_dir`` carries over the unscoped
    ``post_image_dataset/resized`` default. Unscoped (direct ``make mask``)
    returns the bare output dir, so CLI behavior is unchanged.
    """
    mask_root = _mask_output_dir()
    try:
        scope = resized_dir.resolve().relative_to(RESIZED_IMAGE_DIR.resolve())
    except ValueError:
        return mask_root
    if str(scope) == ".":
        return mask_root
    return mask_root / scope


# ----- config (CLI fallback) ---------------------------------------------------


def _load_mask_config() -> dict:
    """``configs/sam_mask.yaml`` as a dict (``{}`` when absent / unparseable)."""
    try:
        import yaml

        with open(SAM_CONFIG, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, ImportError):
        return {}


def _config_path_pattern(cfg: dict) -> str | None:
    """Read the config's global ``path_pattern`` scope. Missing key / ``"*"``
    means mask everything."""
    pattern = cfg.get("path_pattern")
    return pattern if pattern and pattern != "*" else None


def _config_flag(cfg: dict, key: str, default: bool = True) -> bool:
    """A ``run_sam`` switch: absent → on; a string spelled the env-var way
    (``"0"`` / ``"false"`` / ``"no"`` / ``"off"``) → off."""
    raw = cfg.get(key)
    if raw is None:
        return default
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(raw)


# ----- requests ------------------------------------------------------------------


def _sam_rules(cfg: dict) -> list[dict]:
    """Normalize the SAM config into an ordered list of mask rules.

    Two schemas, as documented in ``configs/sam_mask.yaml``: a flat
    ``prompts`` / ``focus_prompts`` set (wrapped as one rule with no pattern
    of its own) or a ``rules:`` list, each entry routing its own prompt set
    by ``path_pattern``. Per-rule ``threshold`` / ``dilate`` fall back to the
    top-level values, and those to the package's request defaults.
    """
    default_threshold = cfg.get("threshold")
    default_dilate = cfg.get("dilate")
    raw_rules = cfg.get("rules")
    if raw_rules is None:
        raw_rules = [
            {
                "prompts": cfg.get("prompts") or [],
                "focus_prompts": cfg.get("focus_prompts") or [],
            }
        ]
    rules: list[dict] = []
    for raw in raw_rules:
        pattern = raw.get("path_pattern")
        rule = {
            "prompts": tuple(str(p) for p in (raw.get("prompts") or ())),
            "focus_prompts": tuple(str(p) for p in (raw.get("focus_prompts") or ())),
            "path_pattern": pattern if pattern and pattern != "*" else None,
        }
        threshold = raw.get("threshold", default_threshold)
        if threshold is not None:
            rule["threshold"] = float(threshold)
        dilate = raw.get("dilate", default_dilate)
        if dilate is not None:
            rule["dilate"] = int(dilate)
        rules.append(rule)
    return rules


def _sam_request(image_dir: Path, out_dir: Path, rule: dict, path_pattern: str | None):
    """The ``SamMaskRequest`` one yaml rule runs as.

    A rule's own ``path_pattern`` routes *within* the global scope in the old
    single-pass CLI; the package takes one glob per run, so a rule that names
    a pattern runs on that pattern alone (the global scope still applies to
    every rule without one). The SAM3 checkpoint and batch size are the
    request defaults — the package's download catalog is where the weights
    land. Validation fires here: a rule with neither prompt list would
    otherwise fail minutes in, after the SAM3 load.
    """
    from anime_tools.masking.requests import SamMaskRequest

    kwargs = {key: rule[key] for key in ("threshold", "dilate") if key in rule}
    try:
        return SamMaskRequest(
            image_dir=str(image_dir),
            mask_dir=str(out_dir),
            prompts=rule["prompts"],
            focus_prompts=rule["focus_prompts"],
            recursive=True,
            path_pattern=rule["path_pattern"] or path_pattern,
            **kwargs,
        )
    except ValueError as exc:
        raise SystemExit(f"SAM mask rule {rule!r}: {exc}") from exc


def _sam_request_from_form(image_dir: Path, tmp_root: Path, form: dict):
    """The ``SamMaskRequest`` one GUI rule card runs as: the card's values
    through the package's ``build_argv`` (its ``__post_init__`` — "nothing to
    mask" — fires here), the resized tree as the ``dst`` root, this card's
    own tempdir as the mask root (``<tmp>/masks_sam``), the card's
    ``path_pattern`` as the run's scope (blank / ``*`` = everything)."""
    pattern = str(form.get("path_pattern") or "").strip()
    return request_from_form(
        "masks_sam",
        form,
        roots={"dst": str(image_dir)},
        settings={"path_pattern": pattern if pattern and pattern != "*" else None},
        mask_root=str(tmp_root),
        recursive=True,
    )


def _merge_request(sources: list[str], output_dir: Path):
    from anime_tools.masking.requests import MergeMasksRequest

    return MergeMasksRequest(mask_dirs=tuple(sources), output_dir=str(output_dir))


# ----- execution -----------------------------------------------------------------


def _stage(stage_id: str):
    return stage_by_id(stage_id)


def _execute(stage_id: str, req) -> None:
    """Run one mask stage: in-process under a daemon job, else as a child."""
    execute_stage(_stage(stage_id), req)


def _sam_requests(resized_dir: Path, tmp_root: Path) -> list:
    """Every SAM pass this run makes: one per GUI rule card when the job
    carries the forms, else one per ``sam_mask.yaml`` rule (``run_sam: false``
    → none). Built up front so validation fires before the first model load."""
    forms = gui_stage_values().get("masks_sam")
    if isinstance(forms, list):
        return [
            _sam_request_from_form(resized_dir, tmp_root / f"sam{i}", form)
            for i, form in enumerate(forms)
            if isinstance(form, dict)
        ]
    cfg = _load_mask_config()
    if not _config_flag(cfg, "run_sam"):
        return []
    pattern = _config_path_pattern(cfg)
    return [
        _sam_request(resized_dir, tmp_root / f"sam{i}", rule, pattern)
        for i, rule in enumerate(_sam_rules(cfg))
    ]


def cmd_mask(extra):
    """Run every SAM rule into a tempdir, merge, write to post_image_dataset/masks/."""
    if extra:
        raise SystemExit(
            f"make mask takes no ARGS ({' '.join(extra)!r}); the knobs live in "
            f"{SAM_CONFIG.relative_to(ROOT)} (or the GUI's Preprocessing tab)."
        )
    resized_dir = _resized_image_dir()
    mask_output_dir = _scoped_mask_output_dir(resized_dir)
    with tempfile.TemporaryDirectory(prefix="anima-masks-") as tmp_root:
        requests = _sam_requests(resized_dir, Path(tmp_root))
        if not requests:
            print("SAM masking is disabled — nothing to do.")
            return
        # One SAM pass per rule, each into its own dir; the merge below unions
        # them (pixel-min), which is the old rules compose.
        for req in requests:
            _execute("masks_sam", req)
        mask_output_dir.mkdir(parents=True, exist_ok=True)
        _execute(
            "masks_merge",
            _merge_request([req.mask_dir for req in requests], mask_output_dir),
        )


def cmd_mask_clean(_extra):
    mask_dir = _mask_output_dir()
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
        try:
            shown = mask_dir.relative_to(ROOT)
        except ValueError:  # mask_dir configured outside the repo
            shown = mask_dir
        print(f"  Removed {shown}/")
