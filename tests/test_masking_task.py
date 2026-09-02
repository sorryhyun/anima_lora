"""``make mask``'s SAM config → ``SamMaskRequest`` translation (scripts/tasks/masking.py).

The package's CLI stopped reading ``sam_mask.yaml`` at anime_tools 0.4.0; the
trainer now normalizes the flat / ``rules:`` schemas itself and builds one
request per rule. The argv those requests produce is round-tripped through the
package parser in ``test_anime_tools_cli_contract.py``; this file pins the
normalization.
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


def test_rule_without_prompts_fails_before_the_sam3_load():
    rule = masking._sam_rules({"rules": [{"path_pattern": "a/*"}]})[0]
    with pytest.raises(SystemExit, match="nothing to mask"):
        masking._sam_request(Path("r"), Path("o"), rule, None)
