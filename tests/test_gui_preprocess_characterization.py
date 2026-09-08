"""Characterization fixture for the Preprocessing tab's knob contract.

Pins, for every knob and stage form, what ``preprocess_env()``,
``preprocess_overrides()`` and the persisted ``[variant]`` meta (flat trainer
knobs + ``[variant.stages.*]``) look like at (a) all-defaults and (b) every
knob flipped — under two deterministic default sources (a bare checkout and a
populated ``preprocess.toml`` / ``gui_settings.json`` / ``sam_mask.yaml``).

This is the byte-for-byte contract the ``knobs.py`` extraction
(``docs/proposal/gui_preprocess_tab_refactor.md`` Phase 1) and the stage-form
migration (``gui_preprocess_from_anime_tools.md`` P1–P3, regenerated
2026-09-07 with the migrated keys — resize geometry, caption rewriting,
autotag mode, SAM rules — moved out of the flat keys / env) must
reproduce. Regenerate deliberately, never to make a red run green::

    uv run python tests/test_gui_preprocess_characterization.py --write
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "gui_preprocess_knobs.json"

# Default sources the tab consults, per scenario. "populated" deliberately
# diverges from every hardcoded default so the three policies (const /
# preprocess.toml / gui_settings) are distinguishable in the fixture.
SCENARIOS: dict[str, dict[str, dict]] = {
    "bare": {"preprocess_toml": {}, "gui_settings": {}, "sam_yaml": {}},
    "populated": {
        "preprocess_toml": {
            "source_image_dir": "my_images",
            "target_res": [1024, 896],
            "resize_crop_anchor": "top",
            "resize_crop_margins": {
                "top": 5.0,
                "right": 0.0,
                "bottom": 0.0,
                "left": 0.0,
            },
            "freefit_max_ratio": 3.0,
            "drop_lowres_images": False,
            "min_pixels": 250000,
            "caption_shuffle_variants": 9,
            "caption_tag_dropout_rate": 0.9,
            "caption_position_clauses": True,
            "caption_autotag": True,
            "caption_autotag_mode": "merge",
            "caption_autotag_min_confidence": 0.35,
        },
        "gui_settings": {
            "caption_shuffle_variants": 7,
            "caption_tag_dropout_rate": 0.3,
            "run_sam_mask": False,
        },
        "sam_yaml": {
            "path_pattern": "artist_x/*",
            "threshold": 0.4,
            "dilate": 3,
            "rules": [{"prompts": ["sign"], "focus_prompts": ["face"]}],
        },
    },
}


def _flip_every_knob(tab) -> None:
    """Set every widget to a value that is neither hardcoded default nor any
    scenario's populated default."""
    tab.source_dir_edit.setText("flipped_images")
    tab.path_scope_edit.setText("artist_a")
    tab.preprocess_path_pattern_edit.setText("artist_a/**")
    tab.drop_lowres_chk.setChecked(not tab.drop_lowres_chk.isChecked())
    tab.min_pixels_spin.setValue(123456)
    tab._set_target_res_widget([768, 1280])
    tab._set_resize_crop_anchor("bottom_right")
    tab._set_resize_crop_margins({"top": 1.5, "right": 2.5, "bottom": 3.5, "left": 4.5})
    tab.freefit_max_ratio_spin.setValue(2.25)
    tab.image_section.widgets["overwrite"].setChecked(True)
    tab.image_section.widgets["workers"].setValue(2)
    tab.shuffle_spin.setValue(11)
    tab.dropout_edit.setText("0.45")
    tab.caption_no_correct_chk.setChecked(not tab.caption_no_correct_chk.isChecked())
    tab.caption_insert_no_artist_chk.setChecked(True)
    tab.caption_trigger_word_edit.setText("@flipped")
    tab.caption_trigger_at_front_chk.setChecked(True)
    tab.caption_drop_groups_edit.setText("artist,lighting")
    tab.caption_position_clauses_chk.setChecked(
        not tab.caption_position_clauses_chk.isChecked()
    )
    tab.caption_autotag_chk.setChecked(not tab.caption_autotag_chk.isChecked())
    tab._set_autotag_mode("overwrite")
    tab.caption_autotag_confidence_spin.setValue(0.55)
    tab.run_sam_mask_chk.setChecked(not tab.run_sam_mask_chk.isChecked())
    tab._set_rule_cards(
        [
            {
                "path_pattern": "artist_b/*",
                "prompts": "bubble, sfx",
                "focus_prompts": "face",
                "threshold": 0.35,
                "dilate": 7,
            },
            {
                "prompts": "watermark",
                "focus_prompts": "none",
                "threshold": 0.6,
                "dilate": 2,
            },
        ]
    )


def _make_tab(monkeypatch_targets, scenario: dict[str, dict]):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from gui.tabs.preprocess import tab as preprocess_tab

    QApplication.instance() or QApplication([])
    monkeypatch_targets(
        preprocess_tab,
        {
            "_load_preprocess_toml": lambda: dict(scenario["preprocess_toml"]),
            "read_gui_settings": lambda: dict(scenario["gui_settings"]),
            "_load_sam_yaml": lambda: dict(scenario["sam_yaml"]),
        },
    )
    return preprocess_tab.PreprocessingTab()


def _observe(tab, variant: str, path: Path) -> dict:
    """env / overrides / persisted meta (with and without the mask section)."""
    from gui import _load

    assert tab.persist_preprocess_inputs()
    meta_inputs = _load(path).get("variant", {})
    assert tab._save_all()
    meta_all = _load(path).get("variant", {})
    env = tab.preprocess_env()
    # The stage forms ride as one JSON env value; pin them decoded so the
    # fixture stays readable and a changed stage default shows as a diff.
    env["PREPROCESS_STAGES_JSON"] = json.loads(env["PREPROCESS_STAGES_JSON"])
    return {
        "env": env,
        "overrides": tab.preprocess_overrides(),
        "meta_inputs_only": meta_inputs,
        "meta_full": meta_all,
    }


def capture(monkeypatch_targets) -> dict:
    from tests.test_gui_preprocess_tab import _temporary_custom_variant

    out: dict = {}
    for name, scenario in SCENARIOS.items():
        with _temporary_custom_variant(f"__pytest_char_{name}__") as (variant, path):
            tab = _make_tab(monkeypatch_targets, scenario)
            tab.set_variant(variant, method="lora")
            defaults = _observe(tab, variant, path)
            # Reload from the just-saved variant: the round trip must be a fixed point.
            tab.set_variant(variant, method="lora")
            defaults_reloaded = _observe(tab, variant, path)
            _flip_every_knob(tab)
            flipped = _observe(tab, variant, path)
            tab.set_variant(variant, method="lora")
            flipped_reloaded = _observe(tab, variant, path)
            tab.deleteLater()
        out[name] = {
            "defaults": defaults,
            "defaults_reloaded": defaults_reloaded,
            "flipped": flipped,
            "flipped_reloaded": flipped_reloaded,
        }
    return out


def _dump(data: dict) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def test_preprocess_knob_contract_matches_fixture(monkeypatch):
    def _patch(module, attrs):
        for name, value in attrs.items():
            monkeypatch.setattr(module, name, value)

    observed = capture(_patch)
    assert FIXTURE.exists(), f"missing fixture — run {__file__} --write"
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    # Compare through JSON so tuple/list and int/float formatting differences
    # surface exactly as they would on disk.
    assert json.loads(_dump(observed)) == expected


if __name__ == "__main__":
    if "--write" not in sys.argv:
        sys.exit("usage: --write (regenerates the fixture from the live tab)")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    def _patch(module, attrs):
        for name, value in attrs.items():
            setattr(module, name, value)

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(_dump(capture(_patch)), encoding="utf-8")
    print(f"wrote {FIXTURE}")
