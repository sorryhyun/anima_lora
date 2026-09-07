"""P0 pilot for drawing the Preprocess panel off ``anime_tools`` stage schemas
(``docs/proposal/gui_preprocess_from_anime_tools.md``).

Qt-free half: the schemas load in the trainer venv, bound roots are hidden,
and ``argv_for`` round-trips through each stage's own generated parser. Qt
half (offscreen): ``StageFormSection`` renders a schema, values written are
values read, and the read-back builds a parseable argv.

Dests asserted here are ones the requests have carried since the API-first
migration (``min_chars`` / ``det_conf`` / ``skip_en`` on OCR, ``threshold`` /
``dilate`` / ``prompts`` on SAM) — the OCR detector/reader switches (and
``min_score``, which went with PP-OCR) are in flux and deliberately not named.
"""

from __future__ import annotations

import os

import pytest

from gui.tabs.preprocess import stage_form as SF

STAGES = ("autotag", "correct", "masks_sam", "ocr")
ROOTS = {
    "src": "image_dataset",
    "dst": "post_image_dataset/resized",
    "masks": "post_image_dataset/masks",
}


@pytest.fixture(scope="module")
def schemas():
    return SF.load_stage_schemas()


def _stage(sid):
    from anime_tools.gui.stages import BY_ID

    return BY_ID[sid]


def _parse(sid, argv):
    from anime_tools.gui.stages import load_parser

    return load_parser(_stage(sid)).parse_args(argv)


# -- Qt-free ---------------------------------------------------------------


def test_schemas_load_for_the_pilot_stages(schemas):
    for sid in STAGES:
        sc = schemas[sid]
        assert sc["available"], sc.get("error")
        assert sc["fields"], sid
        # The field shape the renderer relies on.
        for f in sc["fields"]:
            assert {"dest", "kind", "default", "help", "advanced", "gate"} <= set(f)
            assert f["kind"] in {"bool", "int", "float", "str", "enum", "list"}


def test_visible_fields_hide_bound_and_auto(schemas):
    sam = {f["dest"]: f for f in SF.visible_fields(schemas["masks_sam"])}
    # root / mask-tail / setting / auto are all filled by build_argv, not typed.
    assert "image_dir" not in sam  # ROOT_FIELDS → dst
    assert "mask_dir" not in sam  # MASK_FIELDS → mask_root/<tail>
    assert "path_pattern" not in sam  # SETTING_FIELDS
    assert "device" not in sam  # AUTO_FIELDS
    assert {"threshold", "dilate", "prompts", "focus_prompts", "force"} <= set(sam)
    assert all("advanced" in f and "gate" in f for f in sam.values())

    correct = {f["dest"] for f in SF.visible_fields(schemas["correct"])}
    assert not {"src", "dst", "path_pattern"} & correct
    assert {"caption_trigger_word", "caption_drop_groups", "no_correct"} <= correct

    # --apply is the run bar's, never a form row.
    assert "apply" not in {f["dest"] for f in SF.visible_fields(schemas["autotag"])}


def test_knob_for_maps_schema_kinds_onto_the_knob_table(schemas):
    sam = {f["dest"]: f for f in schemas["masks_sam"]["fields"]}
    assert SF.knob_for(sam["threshold"]).kind == "float"
    assert SF.knob_for(sam["dilate"]).kind == "int"
    assert SF.knob_for(sam["force"]).kind == "bool"
    mode = next(f for f in schemas["autotag"]["fields"] if f["dest"] == "mode")
    knob = SF.knob_for(mode)
    assert knob.kind == "choice" and knob.choices and knob.default == "missing"
    # A None-default numeric is free text; a gate becomes enabled_by.
    export = {f["dest"]: f for f in schemas["export"]["fields"]}
    assert SF.knob_for(export["ocr_dir"]).enabled_by == "combine_ocr"
    assert SF.knob_for(export["combine_ocr"]).enabled_by is None


def test_argv_round_trips_masks_sam(schemas):
    sc = schemas["masks_sam"]
    argv = SF.argv_for(
        sc,
        {"threshold": 0.7, "dilate": 8, "prompts": "speech bubble,text", "force": True},
        roots=ROOTS,
        settings={"path_pattern": "artist_a/*"},
        report_root=None,
        mask_root="post_image_dataset/_masks",
    )
    ns = _parse("masks_sam", argv)
    assert ns.image_dir == ROOTS["dst"]
    assert ns.mask_dir == "post_image_dataset/_masks/masks_sam"
    assert ns.path_pattern == "artist_a/*"
    assert ns.threshold == 0.7 and ns.dilate == 8 and ns.force is True
    req = _stage("masks_sam").request_class().from_namespace(ns)
    assert req.prompts == ("speech bubble", "text")
    # A value left at the request default is not spelled.
    assert "--batch-size" not in argv and "--batch_size" not in argv


def test_argv_round_trips_ocr(schemas):
    sc = schemas["ocr"]
    argv = SF.argv_for(
        sc,
        {"min_chars": 5, "det_conf": 0.4, "skip_en": False},
        roots=ROOTS,
        settings={"path_pattern": "*"},
        report_root="post_image_dataset/reports",
        mask_root=None,
    )
    ns = _parse("ocr", argv)
    assert ns.dst == ROOTS["dst"]
    assert ns.min_chars == 5 and ns.det_conf == 0.4
    # A store_false bool is spelled by its off switch, not --no-…
    assert ns.skip_en is False and "--keep_en" in argv
    assert ns.report_dir == "post_image_dataset/reports/captions/ocr"
    assert "--min_box_px" not in argv


def test_argv_runs_the_requests_own_validation(schemas):
    # SamMaskRequest.__post_init__: nothing to mask is refused before any job.
    with pytest.raises(ValueError):
        SF.argv_for(
            schemas["masks_sam"],
            {"prompts": "none", "focus_prompts": "none"},
            roots=ROOTS,
            settings=None,
            report_root=None,
            mask_root=None,
        )


# -- Qt (offscreen) ---------------------------------------------------------


def _app():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_masks_sam_form_renders_reads_back_and_builds_argv(schemas):
    _app()
    from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QLineEdit, QSpinBox

    from gui.tabs.preprocess.stage_form import StageFormSection

    sc = schemas["masks_sam"]
    sec = StageFormSection(sc, lambda *_: None)
    try:
        assert set(sec.keys()) == {f["dest"] for f in SF.visible_fields(sc)}
        assert isinstance(sec.widgets["threshold"], QDoubleSpinBox)
        assert isinstance(sec.widgets["dilate"], QSpinBox)
        assert isinstance(sec.widgets["force"], QCheckBox)
        assert isinstance(sec.widgets["prompts"], QLineEdit)
        # Advanced fields fold away until the toggle is on.
        assert sec.advanced_box is not None and sec.advanced_box.isHidden()
        assert sec.widgets["batch_size"].parentWidget() is sec.advanced_box
        sec.advanced_toggle.setChecked(True)
        assert not sec.advanced_box.isHidden()

        seen = []
        sec.changed.connect(lambda: seen.append(1))
        sec.set_values({"threshold": 0.7, "dilate": 8, "prompts": "speech bubble,text"})
        v = sec.values()
        assert v["threshold"] == 0.7 and v["dilate"] == 8
        assert v["prompts"] == "speech bubble,text"
        assert seen  # editing marks the section changed (dirty wiring)

        argv = sec.argv(
            roots=ROOTS, settings={"path_pattern": "*"}, mask_root="post_image_dataset"
        )
        ns = _parse("masks_sam", argv)
        assert ns.threshold == 0.7 and ns.dilate == 8
        assert ns.image_dir == ROOTS["dst"]
        assert _stage("masks_sam").request_class().from_namespace(ns).prompts == (
            "speech bubble",
            "text",
        )
    finally:
        sec.deleteLater()


def test_ocr_form_path_browse_and_enum(schemas):
    _app()
    from PySide6.QtWidgets import QComboBox, QLineEdit

    from gui.tabs.preprocess.stage_form import StageFormSection

    sc = schemas["ocr"]
    sec = StageFormSection(sc, lambda *_: None)
    try:
        # A path field is a line edit inside a composite row with a … button.
        ocr_dir = sec.widgets["ocr_dir"]
        assert isinstance(ocr_dir, QLineEdit)
        assert "ocr_dir" in sec.browse_buttons
        assert ocr_dir.parentWidget() is not sec
        # Any enum the stage still carries renders as a combo over its choices.
        for f in SF.visible_fields(sc):
            if f["kind"] == "enum":
                combo = sec.widgets[f["dest"]]
                assert isinstance(combo, QComboBox)
                assert combo.count() == len(f["choices"])
        sec.set_values({"det_conf": 0.4, "min_chars": 5})
        v = sec.values()
        assert v["det_conf"] == 0.4 and v["min_chars"] == 5
        ns = _parse("ocr", sec.argv(roots=ROOTS))
        assert ns.det_conf == 0.4 and ns.min_chars == 5
    finally:
        sec.deleteLater()


def test_gate_disables_its_drawer(schemas):
    _app()
    from gui.tabs.preprocess.stage_form import StageFormSection

    sec = StageFormSection(schemas["export"], lambda *_: None)
    try:
        gate, drawer = sec.widgets["combine_ocr"], sec.widgets["ocr_dir"]
        assert not gate.isChecked() and not drawer.isEnabled()
        gate.setChecked(True)
        assert drawer.isEnabled()
        sec.set_values({"combine_ocr": False})
        assert not drawer.isEnabled()
    finally:
        sec.deleteLater()
