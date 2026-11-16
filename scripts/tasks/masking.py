"""Mask generation: SAM3 rule passes → merged.

``make mask`` runs every SAM rule into a ``tempfile.TemporaryDirectory()``
(honors ``TMPDIR`` / ``TEMP``) and writes only the merged result to
``<mask_dir>/<rel>/{stem}_mask.png``, where ``mask_dir`` comes from the merged
config chain (``configs/preprocess.toml``, default ``post_image_dataset/masks``).

Every stage runs as an ``anime_tools`` request object through
``_common.execute_stage``; under a daemon job (``make daemon-run ARGS="tasks.py
mask"``) one SAM3 load serves every rule pass.

Rules come from ``configs/sam_mask.yaml`` (schema documented there); a knob
absent from it falls back to the request default. Each rule is one SAM pass
into its own temp dir, and the merge step unions them (pixel-min).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ._common import (
    ROOT,
    _path,
    execute_stage,
    stage_by_id,
)

DEFAULT_MASK_DIR = "post_image_dataset/masks"
RESIZED_IMAGE_DIR = ROOT / "post_image_dataset" / "resized"
SAM_CONFIG = ROOT / "configs" / "sam_mask.yaml"


def _mask_output_dir() -> Path:
    """Unscoped mask root — ``mask_dir`` from the merged config chain.

    A function so a ``CONFIG_FILE`` snapshot is read at call time.
    """
    return ROOT / _path("mask_dir", DEFAULT_MASK_DIR)


def _resized_image_dir() -> Path:
    """Scoped resized dir to mask, honoring GUI ``path_scope``.

    The GUI's ``CONFIG_FILE`` snapshot already scopes ``resized_image_dir`` to
    ``post_image_dataset/resized/<path_scope>``; a direct ``make mask`` gets
    the unscoped default.
    """
    return ROOT / _path("resized_image_dir", "post_image_dataset/resized")


def _scoped_mask_output_dir(resized_dir: Path) -> Path:
    """Re-apply the ``path_scope`` offset onto the mask output root.

    SAM writes rel paths relative to the scoped dir (``resized/<scope>``), but
    training resolves masks under the unscoped root (``masks/<scope>/<rel>``,
    ``CachedDataset._resolve_mask_path``), so a scoped run must write under
    ``masks/<scope>`` or the trainer will not find them.
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
    ``masks`` list (wrapped as one rule with no pattern of its own) or a
    ``rules:`` list, each entry routing its own regions by ``path_pattern``.
    Per-rule ``threshold`` / ``dilate`` fall back to the top-level values,
    and those to the package's request defaults.
    """
    from library.config.sam_masks import LEGACY_KEYS, rule_masks

    default_threshold = cfg.get("threshold")
    default_dilate = cfg.get("dilate")
    raw_rules = cfg.get("rules")
    if raw_rules is None:
        raw_rules = [{k: cfg[k] for k in ("masks", *LEGACY_KEYS) if k in cfg}]
    rules: list[dict] = []
    for raw in raw_rules:
        pattern = raw.get("path_pattern")
        rule = {
            "masks": tuple(rule_masks(raw)),
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

    A rule's own ``path_pattern`` replaces the global scope (the package takes
    one glob per run). Validation fires here, before the SAM3 load.
    """
    from anime_tools.masking.requests import MaskPrompt, SamMaskRequest

    kwargs = {key: rule[key] for key in ("threshold", "dilate") if key in rule}
    try:
        return SamMaskRequest(
            image_dir=str(image_dir),
            mask_dir=str(out_dir),
            masks=tuple(MaskPrompt.parse(m) for m in rule["masks"]),
            recursive=True,
            path_pattern=rule["path_pattern"] or path_pattern,
            **kwargs,
        )
    except ValueError as exc:
        raise SystemExit(f"SAM mask rule {rule!r}: {exc}") from exc


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
    """Every SAM pass this run makes: one per ``sam_mask.yaml`` rule
    (``run_sam: false`` → none). Built up front so validation fires before the
    first model load."""
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
            f"{SAM_CONFIG.relative_to(ROOT)}."
        )
    resized_dir = _resized_image_dir()
    mask_output_dir = _scoped_mask_output_dir(resized_dir)
    with tempfile.TemporaryDirectory(prefix="anima-masks-") as tmp_root:
        requests = _sam_requests(resized_dir, Path(tmp_root))
        if not requests:
            print("SAM masking is disabled — nothing to do.")
            return
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
