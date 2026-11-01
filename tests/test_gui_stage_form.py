"""The Preprocess panel drawn off ``anime_tools`` stage schemas (P0–P3).

The schemas load in the trainer venv, bound / trainer-owned dests are hidden,
``argv_for`` round-trips through each stage's own generated parser, the
``preprocess.toml`` seeds land, and a saved form reaches the job as its request.
The rendered forms and their ``[variant.stages.*]`` persistence are pinned by
``test_gui_preprocess_characterization.py``.

Dests asserted here are ones the requests have carried since the API-first
migration (``min_chars`` / ``det_conf`` / ``skip_en`` on OCR, ``threshold`` /
``dilate`` / ``masks`` on SAM) — the OCR detector/reader switches (and
``min_score``, which went with PP-OCR) are in flux and deliberately not named.
"""

from __future__ import annotations

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
    assert {"target_res", "overwrite", "workers"} <= resize
    assert "min_pixels" not in resize  # no pixel floor since anime_tools 0.7.5

    # --apply is the run bar's, never a form row.
    assert "apply" not in {f["dest"] for f in SF.visible_fields(schemas["autotag"])}


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


def test_seeded_defaults_layer_preprocess_toml_over_the_schema(schemas):
    """The user-owned TOML is the lowest-priority default for the dests it
    names — the form opens on what the CLI would run — and ``no_correct``
    is ``caption_correct_order`` inverted (default: no reordering)."""
    bare = SF.seeded_defaults(schemas["resize"], {})
    assert bare["target_res"] is None and "min_pixels" not in bare
    pp = {
        "target_res": [1024, 896],
        "min_pixels": 250000,  # retired key in a user-owned TOML: ignored
        "resize_crop_margins": {"top": 5.0, "right": 0, "bottom": 0, "left": 0},
        "caption_autotag_mode": "merge",
        "caption_autotag_min_confidence": 0.35,
        "caption_trigger_word": "@t",
    }
    resize = SF.seeded_defaults(schemas["resize"], pp)
    assert resize["target_res"] == [1024, 896]
    assert "min_pixels" not in resize
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


@pytest.mark.core
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
