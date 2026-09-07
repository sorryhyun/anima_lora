"""``make mask``'s rules → ``SamMaskRequest`` translation (scripts/tasks/masking.py).

Two sources: the CLI's ``sam_mask.yaml`` (the package's CLI stopped reading
it at anime_tools 0.4.0; the trainer normalizes the flat / ``rules:`` schemas
itself and builds one request per rule) and the GUI's ``masks_sam`` stage
forms (``PREPROCESS_STAGES_JSON``, one card per request through the package's
``build_argv``). The argv those requests produce is round-tripped through the
package parser in ``test_anime_tools_cli_contract.py``; this file pins the
normalization. MIT (the text masker) was removed in v2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.tasks import masking


def test_flat_config_is_one_rule_with_top_level_thresholds():
    rules = masking._sam_rules(
        {
            "prompts": ["speech bubble"],
            "focus_prompts": [],
            "threshold": 0.7,
            "dilate": 3,
        }
    )
    assert rules == [
        {
            "prompts": ("speech bubble",),
            "focus_prompts": (),
            "path_pattern": None,
            "threshold": 0.7,
            "dilate": 3,
        }
    ]


def test_rules_fall_back_to_top_level_then_package_defaults():
    rules = masking._sam_rules(
        {
            "threshold": 0.6,
            "rules": [
                {"prompts": ["bubble"]},
                {"path_pattern": "a/*", "focus_prompts": ["girl"], "dilate": 8},
                {"path_pattern": "*", "prompts": ["text"], "threshold": 0.9},
            ],
        }
    )
    assert [r["threshold"] for r in rules] == [0.6, 0.6, 0.9]
    assert "dilate" not in rules[0] and rules[1]["dilate"] == 8
    assert [r["path_pattern"] for r in rules] == [None, "a/*", None]
    req = masking._sam_request(Path("r"), Path("o"), rules[0], None)
    assert req.dilate == 5  # the package default, not a trainer literal


def test_rule_pattern_wins_over_the_global_scope():
    rules = masking._sam_rules(
        {"rules": [{"path_pattern": "a/*", "prompts": ["x"]}, {"prompts": ["y"]}]}
    )
    own = masking._sam_request(Path("r"), Path("o"), rules[0], "manga/*")
    scoped = masking._sam_request(Path("r"), Path("o"), rules[1], "manga/*")
    assert own.path_pattern == "a/*"
    assert scoped.path_pattern == "manga/*"


def test_empty_focus_list_is_spelled_explicitly():
    """The request defaults ``focus_prompts`` to the subject prompt, so a config
    that clears it must emit ``--focus-prompts none`` or the child would isolate
    the subject on top of the ignore prompts."""
    rule = masking._sam_rules({"prompts": ["bubble"], "focus_prompts": []})[0]
    argv = masking._sam_request(Path("r"), Path("o"), rule, None).to_argv()
    assert argv[argv.index("--focus-prompts") + 1] == "none"


def test_gui_rule_cards_build_one_request_each(monkeypatch, tmp_path):
    """The GUI's ``masks_sam`` forms: each card is its own pass with its own
    scope and tempdir; the trainer fills the resized tree and the walk."""
    import json

    monkeypatch.setenv(
        "PREPROCESS_STAGES_JSON",
        json.dumps(
            {
                "masks_sam": [
                    {
                        "path_pattern": "",
                        "prompts": "bubble, sfx",
                        "focus_prompts": "none",
                        "threshold": 0.35,
                        "dilate": 7,
                        "force": True,
                    },
                    {
                        "path_pattern": "character_a/*",
                        "prompts": "none",
                        "focus_prompts": "girl",
                        "threshold": 0.6,
                        "dilate": 2,
                    },
                ]
            }
        ),
    )
    a, b = masking._sam_requests(Path("resized"), tmp_path)
    assert a.prompts == ("bubble", "sfx") and a.focus_prompts == ()
    assert a.threshold == 0.35 and a.dilate == 7 and a.force
    assert a.path_pattern is None and a.recursive
    assert a.image_dir == "resized"
    assert Path(a.mask_dir) == tmp_path / "sam0" / "masks_sam"
    assert b.prompts == () and b.focus_prompts == ("girl",)
    assert b.path_pattern == "character_a/*"
    assert Path(b.mask_dir) == tmp_path / "sam1" / "masks_sam"
    # No trainer literals: the checkpoint and batch size are the package's.
    from anime_tools.masking.requests import SamMaskRequest

    assert a.checkpoint == SamMaskRequest.checkpoint
    assert a.batch_size == SamMaskRequest.batch_size


def test_gui_rule_card_with_nothing_to_mask_fails_before_the_sam3_load(
    monkeypatch, tmp_path
):
    import json

    monkeypatch.setenv(
        "PREPROCESS_STAGES_JSON",
        json.dumps({"masks_sam": [{"prompts": "none", "focus_prompts": "none"}]}),
    )
    with pytest.raises(SystemExit, match="nothing to mask"):
        masking._sam_requests(Path("resized"), tmp_path)


def test_yaml_run_sam_off_means_no_requests(monkeypatch, tmp_path):
    monkeypatch.delenv("PREPROCESS_STAGES_JSON", raising=False)
    monkeypatch.setattr(masking, "_load_mask_config", lambda: {"run_sam": False})
    assert masking._sam_requests(Path("resized"), tmp_path) == []


def test_run_switches_accept_bools_and_env_style_strings():
    assert masking._config_flag({}, "run_sam") is True
    assert masking._config_flag({"run_sam": False}, "run_sam") is False
    assert masking._config_flag({"run_sam": "0"}, "run_sam") is False
    assert masking._config_flag({"run_sam": "no"}, "run_sam") is False
    assert masking._config_flag({"run_sam": "1"}, "run_sam") is True


def test_make_mask_refuses_stray_args():
    with pytest.raises(SystemExit, match="takes no ARGS"):
        masking.cmd_mask(["--force"])


def test_rule_without_prompts_fails_before_the_sam3_load():
    rule = masking._sam_rules({"rules": [{"path_pattern": "a/*"}]})[0]
    with pytest.raises(SystemExit, match="nothing to mask"):
        masking._sam_request(Path("r"), Path("o"), rule, None)
