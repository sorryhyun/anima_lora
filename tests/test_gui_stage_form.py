"""The Preprocess panel drawn off ``anime_tools`` stage schemas
(``docs/proposal/gui_preprocess_from_anime_tools.md``, P0–P3).

Qt-free half: the schemas load in the trainer venv, bound / trainer-owned
dests are hidden, ``argv_for`` round-trips through each stage's own generated
parser, the ``preprocess.toml`` seeds land, and the ``[variant.stages.*]``
persistence elides against them. Qt half (offscreen): ``StageFormSection``
renders a schema, values written are values read, the read-back builds a
parseable argv, and the chain gate disables the stage rows.

Dests asserted here are ones the requests have carried since the API-first
migration (``min_chars`` / ``det_conf`` / ``skip_en`` on OCR, ``threshold`` /
``dilate`` / ``masks`` on SAM) — the OCR detector/reader switches (and
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
            assert f["kind"] in {"bool", "int", "float", "str", "enum", "list", "masks"}


def test_visible_fields_hide_bound_and_auto(schemas):
    sam = {f["dest"]: f for f in SF.visible_fields(schemas["masks_sam"])}
    # root / mask-tail / setting / auto are all filled by build_argv, not typed.
    assert "image_dir" not in sam  # ROOT_FIELDS → dst
    assert "mask_dir" not in sam  # MASK_FIELDS → mask_root/<tail>
    assert "device" not in sam  # AUTO_FIELDS
    assert "recursive" not in sam  # TRAINER_FIELDS — the chain walks
    # A SAM rule card shows its own scope (SHOWN_BOUND) even though the
    # package binds it as a setting; ``make mask`` threads it per card.
    assert "path_pattern" in sam
    assert {"threshold", "dilate", "masks", "force"} <= set(sam)
    assert all("advanced" in f and "gate" in f for f in sam.values())
    # FIELD_ORDER puts the card's rows first.
    assert [f["dest"] for f in SF.visible_fields(schemas["masks_sam"])][:4] == [
        "path_pattern",
        "masks",
        "threshold",
        "dilate",
    ]

    correct = {f["dest"] for f in SF.visible_fields(schemas["correct"])}
    assert not {"src", "dst", "path_pattern"} & correct
    assert {"caption_trigger_word", "caption_drop_groups", "no_correct"} <= correct
    # The variant sidecar knobs are the TextCachingSection's; the tokenizers
    # are resolved trainer-side.
    assert (
        not {
            "recursive",
            "caption_shuffle_variants",
            "caption_tag_dropout_rate",
            "caption_tag_randomize_rate",
            "qwen3",
            "t5_tokenizer_path",
        }
        & correct
    )

    resize = {f["dest"] for f in SF.visible_fields(schemas["resize"])}
    assert (
        not {"src", "dst", "path_pattern", "recursive", "skip", "excluded_dir"} & resize
    )
    assert {"target_res", "min_pixels", "overwrite", "workers"} <= resize

    # --apply is the run bar's, never a form row.
    assert "apply" not in {f["dest"] for f in SF.visible_fields(schemas["autotag"])}


def test_knob_for_maps_schema_kinds_onto_the_knob_table(schemas):
    sam = {f["dest"]: f for f in schemas["masks_sam"]["fields"]}
    assert SF.knob_for(sam["threshold"]).kind == "float"
    assert SF.knob_for(sam["dilate"]).kind == "int"
    assert SF.knob_for(sam["force"]).kind == "bool"
    # The region list is csv text; blank falls back to the request default.
    assert (SF.knob_for(sam["masks"]).kind, SF.knob_for(sam["masks"]).default) == (
        "str",
        "",
    )
    mode = next(f for f in schemas["autotag"]["fields"] if f["dest"] == "mode")
    knob = SF.knob_for(mode)
    assert knob.kind == "str" and knob.default == "missing"
    # A None-default numeric is free text; a gate becomes enabled_by.
    export = {f["dest"]: f for f in schemas["export"]["fields"]}
    assert SF.knob_for(export["ocr_dir"]).enabled_by == "combine_ocr"
    assert SF.knob_for(export["combine_ocr"]).enabled_by is None


def test_argv_round_trips_masks_sam(schemas):
    sc = schemas["masks_sam"]
    argv = SF.argv_for(
        sc,
        {
            "threshold": 0.7,
            "dilate": 8,
            "masks": ["ignore:text:speech bubble", "ignore:text:text"],
            "force": True,
        },
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
    assert [m.spec() for m in req.masks] == [
        "ignore:text:speech bubble",
        "ignore:text:text",
    ]
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
    # A malformed region (MaskPrompt.parse) is refused before any job.
    with pytest.raises(ValueError, match="ROLE:KIND:VALUE"):
        SF.argv_for(
            schemas["masks_sam"],
            {"masks": ["girl"]},
            roots=ROOTS,
            settings=None,
            report_root=None,
            mask_root=None,
        )


def test_seeded_defaults_layer_preprocess_toml_over_the_schema(schemas):
    """The user-owned TOML is the lowest-priority default for the dests it
    names — the form opens on what the CLI would run — and ``no_correct``
    is ``caption_correct_order`` inverted (default: no reordering)."""
    bare = SF.seeded_defaults(schemas["resize"], {})
    assert bare["target_res"] is None and bare["min_pixels"] == 500000
    pp = {
        "target_res": [1024, 896],
        "min_pixels": 250000,
        "resize_crop_margins": {"top": 5.0, "right": 0, "bottom": 0, "left": 0},
        "caption_autotag_mode": "merge",
        "caption_autotag_min_confidence": 0.35,
        "caption_trigger_word": "@t",
    }
    resize = SF.seeded_defaults(schemas["resize"], pp)
    assert resize["target_res"] == [1024, 896]
    assert resize["min_pixels"] == 250000
    assert resize["resize_crop_margins"] == [5.0, 0.0, 0.0, 0.0]
    autotag = SF.seeded_defaults(schemas["autotag"], pp)
    assert (autotag["mode"], autotag["min_confidence"]) == ("merge", 0.35)
    correct = SF.seeded_defaults(schemas["correct"], pp)
    assert correct["caption_trigger_word"] == "@t"
    assert correct["no_correct"] is True
    assert (
        SF.seeded_defaults(schemas["correct"], {"caption_correct_order": True})[
            "no_correct"
        ]
        is False
    )


def test_persistable_values_elide_against_the_seeded_defaults(schemas):
    defaults = SF.seeded_defaults(schemas["correct"], {})
    values = {**defaults, "caption_trigger_word": "@x", "tag_csv": ""}
    assert SF.persistable_values("correct", values, defaults) == {
        "caption_trigger_word": "@x"
    }
    # target_res is always written (the tiers are what a profile shows).
    rd = SF.seeded_defaults(schemas["resize"], {"target_res": [1024]})
    assert SF.persistable_values("resize", dict(rd), rd) == {"target_res": [1024]}


def test_stage_meta_round_trips(schemas):
    defaults = {sid: SF.seeded_defaults(schemas[sid], {}) for sid in SF.STAGE_IDS}
    forms = {
        "resize": {**defaults["resize"], "target_res": [768], "overwrite": True},
        "autotag": {**defaults["autotag"], "mode": "merge"},
        "correct": {**defaults["correct"], "caption_drop_groups": "artist"},
        "masks_sam": [{**defaults["masks_sam"], "masks": ["ignore:text:bubble"]}],
    }
    meta = SF.merge_stages_into_meta(
        {"family": "lora"}, forms, defaults, include_mask=True
    )
    assert meta["stages"]["resize"] == {"target_res": [768], "overwrite": True}
    assert meta["stages"]["autotag"] == {"mode": "merge"}
    assert meta["stages"]["correct"] == {"caption_drop_groups": "artist"}
    assert meta["stages"]["masks_sam"] == [{"masks": ["ignore:text:bubble"]}]
    for sid in ("resize", "autotag", "correct"):
        assert SF.load_stage_values(meta, sid, defaults[sid]) == forms[sid]
    cards = SF.load_stage_values(meta, "masks_sam", defaults["masks_sam"])
    assert cards == forms["masks_sam"]
    # A pre-0.6.4 card loads as its masks list; its elided `focus_prompts` was
    # the old default `girl`, which is the default soft keep entry now.
    legacy = {"stages": {"masks_sam": [{"prompts": "bubble", "dilate": 2}]}}
    (card,) = SF.load_stage_values(legacy, "masks_sam", defaults["masks_sam"])
    assert card["masks"] == [*defaults["masks_sam"]["masks"], "ignore:text:bubble"]
    assert card["dilate"] == 2 and "prompts" not in card
    # A variant without cards reports None so the tab seeds from sam_mask.yaml.
    assert SF.load_stage_values({}, "masks_sam", defaults["masks_sam"]) is None
    # Mask cards move only with the mask section; an all-default stage vanishes.
    meta2 = SF.merge_stages_into_meta(
        {},
        {"correct": dict(defaults["correct"]), "masks_sam": forms["masks_sam"]},
        defaults,
        include_mask=False,
    )
    assert meta2 == {}


def test_saved_correct_form_builds_the_env_ladders_request(schemas, monkeypatch):
    """P1 gate: a saved ``[variant.stages.correct]`` reaches ``tasks.py`` as
    the same ``CorrectRequest`` the retired ``CAPTION_*`` env ladder built
    for the default variant (trigger word + @no-artist → correction on)."""
    import json

    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_ensure_danbooru_tags", lambda: None)
    monkeypatch.setattr(preprocess, "_variant_settings", lambda: ("4", "0.1", "0.0"))
    monkeypatch.delenv("ANIMA_DAEMON_JOB_DIR", raising=False)
    for name in ("PREPROCESS_PATH_PATTERN", "CAPTION_DROP_GROUPS"):
        monkeypatch.delenv(name, raising=False)
    defaults = SF.seeded_defaults(schemas["correct"], {})
    form = {**defaults, "caption_trigger_word": "@t", "caption_insert_no_artist": True}
    monkeypatch.setenv(SF.STAGE_VALUES_ENV, json.dumps({"correct": form}))
    built = []
    monkeypatch.setattr(
        preprocess, "_execute", lambda sid, req: built.append((sid, req))
    )

    preprocess.cmd_preprocess_captions([])

    ((sid, req),) = built
    assert sid == "correct"
    assert req.caption_trigger_word == "@t" and req.caption_insert_no_artist
    assert not req.no_correct  # a trigger word needs the correction pass
    assert req.recursive and req.caption_shuffle_variants == 4
    assert req.src == "image_dataset" and req.path_pattern == "*"
    # And the argv the daemon would receive reads back as the same request.
    parser = _stage("correct").request_class().parser()
    assert _stage("correct").request_class().from_argv(parser, req.to_argv()) == req


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
        assert isinstance(sec.widgets["masks"], QLineEdit)
        # Advanced fields fold away until the toggle is on.
        assert sec.advanced_box is not None and sec.advanced_box.isHidden()
        assert sec.widgets["batch_size"].parentWidget() is sec.advanced_box
        sec.advanced_toggle.setChecked(True)
        assert not sec.advanced_box.isHidden()

        seen = []
        sec.changed.connect(lambda: seen.append(1))
        sec.set_values(
            {
                "threshold": 0.7,
                "dilate": 8,
                "masks": "ignore:text:speech bubble, ignore:text:text",
            }
        )
        v = sec.values()
        assert v["threshold"] == 0.7 and v["dilate"] == 8
        assert v["masks"] == ["ignore:text:speech bubble", "ignore:text:text"]
        assert seen  # editing marks the section changed (dirty wiring)
        # The card's own scope is a plain glob editor — no file chooser.
        assert (
            "path_pattern" in sec.widgets and "path_pattern" not in sec.browse_buttons
        )

        argv = sec.argv(
            roots=ROOTS, settings={"path_pattern": "*"}, mask_root="post_image_dataset"
        )
        ns = _parse("masks_sam", argv)
        assert ns.threshold == 0.7 and ns.dilate == 8
        assert ns.image_dir == ROOTS["dst"]
        req = _stage("masks_sam").request_class().from_namespace(ns)
        assert [m.spec() for m in req.masks] == [
            "ignore:text:speech bubble",
            "ignore:text:text",
        ]
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


def test_chain_gate_disables_the_stage_rows(schemas):
    """A trainer knob passed as ``gate`` renders first and switches every
    stage row (and the Advanced fold) off with it; ``knob_values`` carries it."""
    _app()
    from gui.tabs.preprocess.stage_form import StageFormSection

    sec = StageFormSection(schemas["autotag"], lambda *_: None, gate="caption_autotag")
    try:
        gate = sec.knob_widgets["caption_autotag"]
        assert set(sec.keys()) == {"mode", "min_confidence"}
        assert sec.knob_values() == {"caption_autotag": False}
        assert not sec.widgets["mode"].isEnabled()
        gate.setChecked(True)
        assert (
            sec.widgets["mode"].isEnabled()
            and sec.widgets["min_confidence"].isEnabled()
        )
        sec.set_knob_values({"caption_autotag": False})
        assert not sec.widgets["mode"].isEnabled()
    finally:
        sec.deleteLater()
