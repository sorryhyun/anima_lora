"""``make mask``'s rules → ``SamMaskRequest`` translation (scripts/tasks/masking.py).

Two sources: the CLI's ``sam_mask.yaml`` (the package's CLI stopped reading
it at anime_tools 0.4.0; the trainer normalizes the flat / ``rules:`` schemas
itself and builds one request per rule) and the GUI's ``masks_sam`` stage
forms (``PREPROCESS_STAGES_JSON``, one card per request through the package's
``build_argv``). The argv those requests produce is round-tripped through the
package parser in ``test_anime_tools_cli_contract.py``; this file pins the
normalization, including the pre-0.6.4 ``prompts`` / ``focus_prompts`` pair
(``library.config.sam_masks``). MIT (the text masker) was removed in v2.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from anime_tools.downloads import DEFAULT_SUBJECT_PROMPT_EMBED

from scripts.tasks import masking

SOFT_GIRL = f"keep:soft:{DEFAULT_SUBJECT_PROMPT_EMBED}"


def _specs(req) -> list[str]:
    return [m.spec() for m in req.masks]


def test_flat_config_is_one_rule_with_top_level_thresholds():
    rules = masking._sam_rules(
        {"masks": ["ignore:text:speech bubble"], "threshold": 0.7, "dilate": 3}
    )
    assert rules == [
        {
            "masks": ("ignore:text:speech bubble",),
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
                {"masks": ["ignore:text:bubble"]},
                {"path_pattern": "a/*", "masks": ["keep:text:girl"], "dilate": 8},
                {"path_pattern": "*", "masks": ["ignore:text:text"], "threshold": 0.9},
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
        {
            "rules": [
                {"path_pattern": "a/*", "masks": ["ignore:text:x"]},
                {"masks": ["ignore:text:y"]},
            ]
        }
    )
    own = masking._sam_request(Path("r"), Path("o"), rules[0], "manga/*")
    scoped = masking._sam_request(Path("r"), Path("o"), rules[1], "manga/*")
    assert own.path_pattern == "a/*"
    assert scoped.path_pattern == "manga/*"


def test_a_pre_064_yaml_pair_reads_as_masks():
    """``prompts`` → ignore:text, ``focus_prompts`` → keep:text, and ``girl`` —
    which the old stage served through ``--prompt_embed`` — the soft entry. An
    empty focus list is no keep region (the argv spells the ignore list alone,
    so the child does not add the default subject on top)."""
    rule = masking._sam_rules({"prompts": ["bubble"], "focus_prompts": []})[0]
    req = masking._sam_request(Path("r"), Path("o"), rule, None)
    assert _specs(req) == ["ignore:text:bubble"]
    argv = req.to_argv()
    assert argv[argv.index("--masks") + 1 :] == ["ignore:text:bubble"]
    rule = masking._sam_rules(
        {"rules": [{"prompts": ["text"], "focus_prompts": ["girl", "face"]}]}
    )[0]
    assert rule["masks"] == (SOFT_GIRL, "keep:text:face", "ignore:text:text")


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
                        "masks": ["ignore:text:bubble", "ignore:text:sfx"],
                        "threshold": 0.35,
                        "dilate": 7,
                        "force": True,
                    },
                    {
                        "path_pattern": "character_a/*",
                        "masks": ["keep:text:girl"],
                        "threshold": 0.6,
                        "dilate": 2,
                    },
                ]
            }
        ),
    )
    a, b = masking._sam_requests(Path("resized"), tmp_path)
    assert _specs(a) == ["ignore:text:bubble", "ignore:text:sfx"]
    assert a.threshold == 0.35 and a.dilate == 7 and a.force
    assert a.path_pattern is None and a.recursive
    assert a.image_dir == "resized"
    assert Path(a.mask_dir) == tmp_path / "sam0" / "masks_sam"
    assert _specs(b) == ["keep:text:girl"]
    assert b.path_pattern == "character_a/*"
    assert Path(b.mask_dir) == tmp_path / "sam1" / "masks_sam"
    # No trainer literals: the checkpoint and batch size are the package's.
    from anime_tools.masking.requests import SamMaskRequest

    assert a.checkpoint == SamMaskRequest.checkpoint
    assert a.batch_size == SamMaskRequest.batch_size


def test_a_pre_064_gui_card_is_migrated(monkeypatch, tmp_path):
    """A job queued before the upgrade carries the old card shape; its elided
    ``focus_prompts`` was the old default ``girl``, now the soft entry."""
    import json

    monkeypatch.setenv(
        "PREPROCESS_STAGES_JSON",
        json.dumps(
            {
                "masks_sam": [
                    {"prompts": "bubble, sfx"},
                    {"prompts": "watermark", "focus_prompts": "none"},
                ]
            }
        ),
    )
    a, b = masking._sam_requests(Path("resized"), tmp_path)
    assert _specs(a) == [SOFT_GIRL, "ignore:text:bubble", "ignore:text:sfx"]
    assert _specs(b) == ["ignore:text:watermark"]


def test_gui_rule_card_with_a_malformed_mask_fails_before_the_sam3_load(
    monkeypatch, tmp_path
):
    import json

    monkeypatch.setenv(
        "PREPROCESS_STAGES_JSON", json.dumps({"masks_sam": [{"masks": ["girl"]}]})
    )
    with pytest.raises(SystemExit, match="ROLE:KIND:VALUE"):
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


def test_rule_without_masks_fails_before_the_sam3_load():
    rule = masking._sam_rules({"rules": [{"path_pattern": "a/*"}]})[0]
    with pytest.raises(SystemExit, match="nothing to mask"):
        masking._sam_request(Path("r"), Path("o"), rule, None)
