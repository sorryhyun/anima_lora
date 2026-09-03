"""Mask generation: SAM3 + MIT/ComicTextDetector → merged.

``make mask`` is a one-shot orchestrator: it runs SAM and MIT into a
``tempfile.TemporaryDirectory()`` (cross-platform — honors ``TMPDIR`` /
``TEMP``) and writes only the merged result to
``post_image_dataset/masks/<rel>/{stem}_mask.png``. Per-tool intermediates
are never persisted under the project root.

Every stage runs as an ``anime_tools`` **request object**
(``anime_tools.masking.requests.{SamMaskRequest,MitMaskRequest,MergeMasksRequest}``)
— the trainer never spells a flag. How a request is executed depends on where
``make mask`` runs (``_execute``):

- **Under a daemon job** (``ANIMA_DAEMON_JOB_DIR`` set — every GUI run, and
  ``make daemon-run ARGS="tasks.py mask"``) the stage runners are called
  **in-process** through the package registry (``Stage.runner()``): one
  interpreter for the whole chain, one SAM3 load shared by every ``rules:``
  pass (``load_sam3`` is cached per process), and the package's
  ``_progress`` heartbeat keeps a quiet model load from reading as a stall to
  the daemon's watchdog. The job process exits at the end, so VRAM is
  released as before.
- **From a plain shell** each stage is a ``python -m <stage.module>`` child
  with ``req.to_argv()``, so ``make mask`` still releases the model between
  stages and on exit.

The switch is ``_common.execute_stage`` — the same one the caption stages
(``preprocess.py``) and grouping (``curate.py``) run through.

The mask config (``configs/sam_mask.yaml`` or the GUI's ``MASK_CONFIG_JSON``
env snapshot) carries the SAM prompt set(s) — a flat ``prompts`` /
``focus_prompts`` pair or a ``rules:`` list routed by ``path_pattern`` — plus
the optional ``run_sam`` / ``run_mit`` switches and a ``mit:`` block
(``text_threshold`` / ``dilate`` / ``ctd_gate``) for the text masker. Every
knob absent from the config falls back to the package's request default; the
trainer carries no literal of its own. A ``rules:`` form becomes one SAM pass
per rule into its own temp dir; the merge step's pixel-min union then
composes them exactly as the old single-pass ``rules`` did (ignore regions
unioned).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from ._common import ROOT, _path, execute_stage, stage_by_id

MASK_OUTPUT_DIR = ROOT / "post_image_dataset" / "masks"
RESIZED_IMAGE_DIR = ROOT / "post_image_dataset" / "resized"
SAM_CONFIG = ROOT / "configs" / "sam_mask.yaml"
# Where ``make download-mit`` lands the UNet++ weights. The package's own
# default (``model_path=None``) reads the same file out of the HF hub cache, so
# this is passed only when the trainer's copy exists — no second download for
# a checkout that fetched it the trainer's way, no stale literal otherwise.
MIT_MODEL_PATH = ROOT / "models" / "mit" / "model.pth"
MASK_CONFIG_ENV = "MASK_CONFIG_JSON"
_UNSET = object()


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

    SAM/MIT emit masks with rel paths taken **relative to the scoped resized
    dir** (``resized/<scope>``), so a scoped run drops the ``<scope>`` prefix.
    But training resolves masks relative to the **unscoped** cache root
    (``lora/<scope>/<rel>`` → ``masks/<scope>/<rel>``, see
    ``CachedDataset._resolve_mask_path``), so masking must land them under
    ``masks/<scope>`` — not flat in ``masks/`` — or the trainer won't find
    them. Mirror whatever scope ``resized_dir`` carries over the unscoped
    ``post_image_dataset/resized`` default. Unscoped (direct ``make mask``)
    returns the bare output dir, so CLI behavior is unchanged.
    """
    try:
        scope = resized_dir.resolve().relative_to(RESIZED_IMAGE_DIR.resolve())
    except ValueError:
        return MASK_OUTPUT_DIR
    if str(scope) == ".":
        return MASK_OUTPUT_DIR
    return MASK_OUTPUT_DIR / scope


# ----- config ------------------------------------------------------------------


def _runtime_mask_config() -> dict | None:
    """GUI queue jobs pass an immutable mask config snapshot via env.

    Same keys as ``configs/sam_mask.yaml`` (the GUI's rule cards → ``rules``,
    its two "run" checkboxes → ``run_sam`` / ``run_mit``, its MIT knobs → the
    ``mit:`` block). Direct CLI usage leaves this unset and reads the yaml.
    """
    raw = os.environ.get(MASK_CONFIG_ENV)
    if not raw:
        return None
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid {MASK_CONFIG_ENV}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise SystemExit(f"Invalid {MASK_CONFIG_ENV}: expected an object")
    return cfg


def _load_mask_config(runtime: dict | None | object = _UNSET) -> dict:
    if runtime is _UNSET:
        runtime = _runtime_mask_config()
    if runtime is not None:
        return runtime
    try:
        import yaml

        with open(SAM_CONFIG, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, ImportError):
        return {}


def _config_path_pattern(cfg: dict) -> str | None:
    """Read ``path_pattern`` so both backends filter alike.

    The key lives with the SAM config but is a dataset-level filter, so ``make
    mask`` forwards it to the MIT backend too (both run on the same resized
    dir). Missing key / ``"*"`` means mask everything.
    """
    pattern = cfg.get("path_pattern")
    return pattern if pattern and pattern != "*" else None


def _config_flag(cfg: dict, key: str, default: bool = True) -> bool:
    """A ``run_sam`` / ``run_mit`` switch: absent → on; a string spelled the
    env-var way (``"0"`` / ``"false"`` / ``"no"`` / ``"off"``) → off."""
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
    """The ``SamMaskRequest`` one rule runs as.

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


def _mit_model_path() -> str | None:
    return str(MIT_MODEL_PATH) if MIT_MODEL_PATH.exists() else None


def _mit_request(image_dir: Path, out_dir: Path, cfg: dict, path_pattern: str | None):
    """The ``MitMaskRequest`` the text masker runs as.

    Knobs come from the config's ``mit:`` block (``text_threshold`` /
    ``dilate`` / ``ctd_gate``); an absent one is the package default. MIT
    always stays ignore-mode (there is no focus form), and ``use_sam`` is left
    at the package default — the balloon pass is the SAM stage's job here.
    """
    from anime_tools.masking.requests import MitMaskRequest

    block = cfg.get("mit") or {}
    if not isinstance(block, dict):
        raise SystemExit(f"mask config: `mit` must be a mapping, got {block!r}")
    kwargs: dict = {}
    for key, cast in (("text_threshold", float), ("dilate", int), ("ctd_gate", bool)):
        value = block.get(key)
        if value is None:
            continue
        if key == "ctd_gate":
            kwargs[key] = _config_flag(block, key)
        else:
            kwargs[key] = cast(value)
    try:
        return MitMaskRequest(
            image_dir=str(image_dir),
            mask_dir=str(out_dir),
            recursive=True,
            path_pattern=path_pattern,
            model_path=_mit_model_path(),
            **kwargs,
        )
    except ValueError as exc:
        raise SystemExit(f"MIT mask config {block!r}: {exc}") from exc


def _merge_request(sources: list[str], output_dir: Path):
    from anime_tools.masking.requests import MergeMasksRequest

    return MergeMasksRequest(mask_dirs=tuple(sources), output_dir=str(output_dir))


# ----- execution -----------------------------------------------------------------


def _stage(stage_id: str):
    return stage_by_id(stage_id)


def _execute(stage_id: str, req) -> None:
    """Run one mask stage: in-process under a daemon job, else as a child."""
    execute_stage(_stage(stage_id), req)


def cmd_mask(extra):
    """Run SAM + MIT into a tempdir, merge, write to post_image_dataset/masks/.

    ``run_sam`` / ``run_mit`` in the mask config gate each backend
    independently (default on). If both are off the command is a no-op.
    """
    if extra:
        raise SystemExit(
            f"make mask takes no ARGS ({' '.join(extra)!r}); the knobs live in "
            f"{SAM_CONFIG.relative_to(ROOT)} (or the GUI's Preprocessing tab)."
        )
    cfg = _load_mask_config()
    run_sam = _config_flag(cfg, "run_sam")
    run_mit = _config_flag(cfg, "run_mit")
    if not (run_sam or run_mit):
        print("Both SAM and MIT masking are disabled — nothing to do.")
        return
    pattern = _config_path_pattern(cfg)
    resized_dir = _resized_image_dir()
    mask_output_dir = _scoped_mask_output_dir(resized_dir)
    with tempfile.TemporaryDirectory(prefix="anima-masks-") as tmp_root:
        merge_sources: list[str] = []
        # Build every request up front: validation (a rule with no prompts, a
        # bad MIT threshold) fires before the first model load.
        sam_requests = []
        if run_sam:
            for i, rule in enumerate(_sam_rules(cfg)):
                tmp_sam = Path(tmp_root) / f"sam{i}"
                sam_requests.append(_sam_request(resized_dir, tmp_sam, rule, pattern))
        mit_request = None
        if run_mit:
            mit_request = _mit_request(
                resized_dir, Path(tmp_root) / "mit", cfg, pattern
            )
        # One SAM pass per rule, each into its own dir; the merge below unions
        # them (pixel-min), which is the old rules compose.
        for req in sam_requests:
            _execute("masks_sam", req)
            merge_sources.append(req.mask_dir)
        if mit_request is not None:
            _execute("masks_mit", mit_request)
            merge_sources.append(mit_request.mask_dir)
        mask_output_dir.mkdir(parents=True, exist_ok=True)
        _execute("masks_merge", _merge_request(merge_sources, mask_output_dir))


def cmd_mask_clean(_extra):
    if MASK_OUTPUT_DIR.exists():
        shutil.rmtree(MASK_OUTPUT_DIR)
        print(f"  Removed {MASK_OUTPUT_DIR.relative_to(ROOT)}/")
