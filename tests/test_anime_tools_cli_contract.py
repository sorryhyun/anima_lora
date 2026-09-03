"""The trainer → ``anime_tools`` CLI seam, checked against the pinned package.

Every argv a task wrapper hands to ``python -m anime_tools.<stage>`` must parse
and build that stage's request object at the pinned rev, and every
``from anime_tools … import`` in the trainer must resolve. The registry and the
request modules are torch-free (the package pins that), so the argv is checked
in-process: no child interpreter, no model load.

The argv comes from the wrapper functions themselves with ``run()`` stubbed,
so a flag the trainer stops or starts emitting is caught here, not minutes
into a GPU job. ``captions.index`` is a plain CLI, not a registered stage; it
is checked by module presence only.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
from pathlib import Path


from anime_tools.stages.registry import STAGES

ROOT = Path(__file__).resolve().parents[1]
BY_MODULE = {stage.module: stage for stage in STAGES}

_TRAINER_PATHS = {
    "source_image_dir": "image_dataset",
    "resized_image_dir": "post_image_dataset/resized",
    "lora_cache_dir": "post_image_dataset/lora",
}


def _fake_path(key: str, default: str) -> str:
    return _TRAINER_PATHS.get(key, default)


def _capture(monkeypatch, module) -> list[list[str]]:
    calls: list[list[str]] = []
    monkeypatch.setattr(module, "run", lambda cmd, **kw: calls.append(list(cmd)))
    monkeypatch.setattr(module, "_path", _fake_path)
    return calls


def _build(cmd: list[str]):
    """The request a ``[PY, "-m", module, *flags]`` argv builds, through the
    stage's generated parser — the same path the child's ``main()`` takes."""
    assert cmd[1] == "-m", cmd
    stage = BY_MODULE[cmd[2]]
    cls = stage.request_class()
    return cls.from_namespace(cls.parser().parse_args(cmd[3:]))


# ----- masking -----------------------------------------------------------------


def _mask_env(monkeypatch, cfg: dict) -> None:
    monkeypatch.setenv("MASK_CONFIG_JSON", json.dumps(cfg))
    # A test suite itself running as a daemon job would otherwise take the
    # in-process path and load SAM3.
    monkeypatch.delenv("ANIMA_DAEMON_JOB_DIR", raising=False)


def test_mask_rules_form_builds_sam_mit_and_merge_requests(monkeypatch):
    from scripts.tasks import masking

    calls = _capture(monkeypatch, masking)
    _mask_env(
        monkeypatch,
        {
            "path_pattern": "manga/*",
            "rules": [
                {"prompts": ["bubble"], "threshold": 0.7},
                {
                    "path_pattern": "character_a/*",
                    "focus_prompts": ["girl"],
                    "threshold": 0.5,
                    "dilate": 8,
                },
            ],
            "run_sam": True,
            "run_mit": True,
            "mit": {"text_threshold": 0.9, "dilate": 2, "ctd_gate": False},
        },
    )

    masking.cmd_mask([])

    assert [c[2] for c in calls] == [
        "anime_tools.masking.cli.generate_masks",
        "anime_tools.masking.cli.generate_masks",
        "anime_tools.masking.cli.generate_masks_mit",
        "anime_tools.masking.cli.merge_masks",
    ]
    sam_a, sam_b, mit, merge = (_build(c) for c in calls)

    assert sam_a.prompts == ("bubble",)
    assert sam_a.focus_prompts == ()
    assert sam_a.threshold == 0.7
    assert sam_a.path_pattern == "manga/*"
    assert Path(sam_a.image_dir) == masking.RESIZED_IMAGE_DIR
    assert sam_a.recursive
    # No trainer literals: the checkpoint and batch size are the package's.
    from anime_tools.masking.requests import SamMaskRequest

    assert sam_a.checkpoint == SamMaskRequest.checkpoint
    assert sam_a.batch_size == SamMaskRequest.batch_size

    assert sam_b.focus_prompts == ("girl",)
    assert sam_b.prompts == ()
    assert sam_b.dilate == 8
    assert sam_b.path_pattern == "character_a/*"

    assert mit.text_threshold == 0.9
    assert mit.dilate == 2
    assert mit.ctd_gate is False
    assert mit.use_mit
    assert not mit.use_sam
    assert mit.path_pattern == "manga/*"

    assert merge.mask_dirs == (sam_a.mask_dir, sam_b.mask_dir, mit.mask_dir)
    assert Path(merge.output_dir) == masking.MASK_OUTPUT_DIR


def test_mask_flat_yaml_config_builds_one_sam_request(monkeypatch):
    """Direct ``make mask`` reads ``configs/sam_mask.yaml``: the shipped flat form."""
    from scripts.tasks import masking

    calls = _capture(monkeypatch, masking)
    monkeypatch.delenv("MASK_CONFIG_JSON", raising=False)
    monkeypatch.delenv("ANIMA_DAEMON_JOB_DIR", raising=False)
    yaml_cfg = masking._load_mask_config()
    monkeypatch.setattr(
        masking, "_load_mask_config", lambda: {**yaml_cfg, "run_mit": False}
    )

    masking.cmd_mask([])

    assert [c[2] for c in calls] == [
        "anime_tools.masking.cli.generate_masks",
        "anime_tools.masking.cli.merge_masks",
    ]
    sam = _build(calls[0])
    assert sam.prompts == ("speech bubble", "text bubble")
    assert sam.focus_prompts == ()
    assert sam.path_pattern is None


def test_mask_under_a_daemon_job_runs_the_stages_in_process(monkeypatch, tmp_path):
    """With ``ANIMA_DAEMON_JOB_DIR`` set every stage goes through the registry's
    ``Stage.runner()`` in this interpreter — no child, so one SAM3 load is
    shared across the rules — and the requests are the same objects the argv
    path would have spelled."""
    from scripts.tasks import masking

    calls = _capture(monkeypatch, masking)
    monkeypatch.setenv(
        "MASK_CONFIG_JSON",
        json.dumps({"rules": [{"prompts": ["a"]}, {"prompts": ["b"]}]}),
    )
    monkeypatch.setenv("ANIMA_DAEMON_JOB_DIR", str(tmp_path))
    monkeypatch.delenv("ANIMA_HOME", raising=False)

    ran: list[tuple[str, object]] = []

    class _Stage:
        def __init__(self, stage_id):
            self.module = f"stub.{stage_id}"
            self._id = stage_id

        def runner(self):
            return lambda req: ran.append((self._id, req))

    monkeypatch.setattr(masking, "_stage", _Stage)

    masking.cmd_mask([])

    assert calls == []
    assert [stage_id for stage_id, _ in ran] == [
        "masks_sam",
        "masks_sam",
        "masks_mit",
        "masks_merge",
    ]
    from anime_tools.masking.requests import (
        MergeMasksRequest,
        MitMaskRequest,
        SamMaskRequest,
    )

    sam_a, sam_b, mit, merge = (req for _, req in ran)
    assert isinstance(sam_a, SamMaskRequest) and sam_b.prompts == ("b",)
    assert isinstance(mit, MitMaskRequest)
    assert isinstance(merge, MergeMasksRequest)
    assert merge.mask_dirs == (sam_a.mask_dir, sam_b.mask_dir, mit.mask_dir)
    # The package anchors its bare defaults on ANIMA_HOME, which run() exports
    # for a child; the in-process path must pin it the same way.
    assert Path(os.environ["ANIMA_HOME"]) == ROOT


def test_mask_stage_ids_the_trainer_names_are_registered():
    from anime_tools.stages.registry import BY_ID

    for stage_id in ("masks_sam", "masks_mit", "masks_merge"):
        stage = BY_ID[stage_id]
        assert callable(stage.runner())


# ----- caption stages ------------------------------------------------------------


def test_caption_autotag_argv_builds_autotag_request(monkeypatch):
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", _fake_path)
    argv = preprocess._caption_autotag_argv(
        [
            "--mode",
            "merge",
            "--min_confidence",
            "0.3",
            "--apply",
            "--path_pattern",
            "g/*",
        ]
    )

    req = _build([preprocess.PY, *argv])
    assert req.src == "image_dataset"
    assert req.dst == "post_image_dataset/resized"
    assert req.path_pattern == "g/*"
    assert req.mode == "merge"
    assert req.min_confidence == 0.3
    assert req.apply


def test_caption_position_argv_builds_position_request(monkeypatch):
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", _fake_path)
    argv = preprocess._caption_position_argv(["--path_pattern", "g/*", "--apply"])

    req = _build([preprocess.PY, *argv])
    assert req.src == "image_dataset"
    assert req.path_pattern == "g/*"
    assert req.apply


def test_preprocess_captions_argv_builds_correct_request(monkeypatch):
    from scripts.tasks import preprocess

    calls = _capture(monkeypatch, preprocess)
    monkeypatch.setattr(preprocess, "_ensure_danbooru_tags", lambda: None)
    for name in (
        "CAPTION_SHUFFLE_VARIANTS",
        "CAPTION_TAG_DROPOUT_RATE",
        "CAPTION_TAG_RANDOMIZE_RATE",
    ):
        monkeypatch.delenv(name, raising=False)

    preprocess.cmd_preprocess_captions(
        ["--path_pattern", "g/*"],
        caption_config={
            "correct_order": True,
            "insert_no_artist": True,
            "trigger_word": "@dataset-trigger",
            "trigger_at_front": True,
            "drop_groups": "meta,artist",
            "position_clauses": False,
            "autotag": False,
        },
    )

    assert len(calls) == 1
    req = _build(calls[0])
    assert req.src == "image_dataset"
    assert req.dst == "post_image_dataset/resized"
    assert req.recursive
    assert req.path_pattern == "g/*"
    assert req.caption_insert_no_artist
    assert req.caption_trigger_word == "@dataset-trigger"
    assert req.caption_trigger_at_front
    assert req.caption_drop_groups == "meta,artist"
    assert not req.no_correct
    assert req.caption_shuffle_variants == 4
    assert req.caption_tag_dropout_rate == 0.1


def test_preprocess_captions_passthrough_builds_no_correct_request(monkeypatch):
    from scripts.tasks import preprocess

    calls = _capture(monkeypatch, preprocess)
    monkeypatch.setattr(preprocess, "_ensure_danbooru_tags", lambda: None)
    monkeypatch.setenv("CAPTION_SHUFFLE_VARIANTS", "2")
    monkeypatch.delenv("CAPTION_TAG_RANDOMIZE_RATE", raising=False)

    preprocess.cmd_preprocess_captions(
        [],
        caption_config={
            "correct_order": False,
            "insert_no_artist": False,
            "trigger_word": "",
            "trigger_at_front": False,
            "drop_groups": "",
            "position_clauses": False,
            "autotag": False,
        },
    )

    assert len(calls) == 1
    req = _build(calls[0])
    assert req.no_correct
    assert req.caption_shuffle_variants == 2


# ----- grouping ------------------------------------------------------------------


def test_curate_group_argv_builds_group_request(monkeypatch):
    from scripts.tasks import curate

    calls = _capture(monkeypatch, curate)
    curate.cmd_curate_group(["--match-frac-min", "0.4", "--cell-match-min", "0.9"])

    assert len(calls) == 1
    req = _build(calls[0])
    assert req.source_dir == "image_dataset"
    assert req.match_frac_min == 0.4
    assert req.cell_match_min == 0.9


# ----- every module string and import name the trainer spells -------------------

_SCAN_ROOTS = ("scripts", "gui", "library", "bench", "anima_lora", "networks")
_SCAN_FILES = ("tasks.py", "train.py", "inference.py")


def _trainer_sources():
    for root in _SCAN_ROOTS:
        for path in (ROOT / root).rglob("*.py"):
            if "_vendor" in path.parts:
                continue
            yield path
    for name in _SCAN_FILES:
        yield ROOT / name


def _trainer_anime_tools_refs() -> tuple[
    dict[tuple[str, str | None], list[str]], set[str]
]:
    """``(module, name) -> where`` for every import, plus every string literal
    naming an ``anime_tools`` module (the ``-m`` targets)."""
    imports: dict[tuple[str, str | None], list[str]] = {}
    modules: set[str] = set()
    for path in _trainer_sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        where = str(path.relative_to(ROOT))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "anime_tools"
            ):
                for alias in node.names:
                    imports.setdefault((node.module, alias.name), []).append(where)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("anime_tools"):
                        imports.setdefault((alias.name, None), []).append(where)
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("anime_tools.")
                and " " not in node.value
                and node.value.replace(".", "").replace("_", "").isalnum()
            ):
                modules.add(node.value)
    return imports, modules


def test_every_trainer_import_from_anime_tools_resolves():
    imports, _ = _trainer_anime_tools_refs()
    assert imports, "no anime_tools imports found — scan roots moved?"
    broken = []
    for (module, name), where in sorted(imports.items()):
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            broken.append(f"{module} ({exc}) <- {where}")
            continue
        if name in (None, "*") or hasattr(mod, name):
            continue
        if importlib.util.find_spec(f"{module}.{name}") is None:
            broken.append(f"{module}.{name} <- {where}")
    assert not broken, "\n".join(broken)


def test_every_module_string_the_trainer_spells_exists():
    """The ``-m`` targets (stages, tagger CLIs, ``captions.index``) and any
    other dotted ``anime_tools.…`` literal must be importable modules."""
    _, modules = _trainer_anime_tools_refs()
    assert "anime_tools.masking.cli.generate_masks" in modules
    missing = sorted(m for m in modules if importlib.util.find_spec(m) is None)
    assert not missing, missing


def test_stage_modules_the_trainer_uses_are_registered():
    used = {
        "anime_tools.masking.cli.generate_masks",
        "anime_tools.masking.cli.generate_masks_mit",
        "anime_tools.masking.cli.merge_masks",
        "anime_tools.stages.cli.autotag_captions",
        "anime_tools.stages.cli.position_captions",
        "anime_tools.stages.cli.correct_captions",
        "anime_tools.grouping.cli.build_groups",
    }
    assert used <= set(BY_MODULE), used - set(BY_MODULE)


def test_caption_index_argv_keeps_the_trainer_output_path(monkeypatch):
    """``captions.index`` is a plain CLI (not a registered stage); its default
    ``--out`` moved into the package's workspace tree, so the trainer must spell
    the ``post_image_dataset/captions/`` home its readers expect."""
    from scripts.tasks import preprocess

    calls = _capture(monkeypatch, preprocess)
    monkeypatch.delenv("PREPROCESS_PATH_PATTERN", raising=False)
    preprocess.cmd_caption_index([])

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[2] == "anime_tools.captions.index"
    assert cmd[cmd.index("--out") + 1] == preprocess.CAPTION_INDEX_PATH
    assert preprocess.CAPTION_INDEX_PATH.startswith("post_image_dataset/captions/")


# ----- the contract version the task runner was written for ---------------------


def test_task_runner_pins_the_installed_contract_version():
    from anime_tools.contract import CONTRACT_VERSION

    from scripts.tasks import _common

    assert _common.ANIME_TOOLS_CONTRACT_VERSION == CONTRACT_VERSION


def test_no_hand_copied_contract_constants():
    """The three copies the API-first audit found are now the package's own."""
    from anime_tools import contract

    from gui.tabs import _autotag
    from scripts.tasks import downloads, preprocess

    assert _autotag._AUTOTAG_READY is contract.AUTOTAG_READY
    assert _autotag._AUTOTAG_RESULT_PREFIX is contract.AUTOTAG_RESULT_PREFIX
    assert _autotag._AUTOTAG_ERROR_PREFIX is contract.AUTOTAG_ERROR_PREFIX
    assert preprocess.AUTOTAG_MODES is contract.AUTOTAG_MODES
    assert downloads.TAGGER_CKPT_REQUIRED is contract.DBV4_REQUIRED_FILES
