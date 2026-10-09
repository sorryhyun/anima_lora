"""``gui.core.anime_tools_panel``: the panel's settings seed, the stale-Export check
and the launch argv. Qt-free."""

from __future__ import annotations

import json
import os
import time

from gui.core import anime_tools_panel as P


def _settings(home):
    return json.loads(P.settings_path(home).read_text(encoding="utf-8"))


def test_seed_defaults_export_to_sidecars_only_and_keeps_other_settings(tmp_path):
    P.settings_path(tmp_path).write_text(
        json.dumps(
            {
                "stage_defaults": {"tagger_dir": "models/tagger"},
                "values": {
                    "autotag": {"mode": "merge"},
                    "export": {"resize_cap": True, "webp": True, "combine_ocr": True},
                },
            }
        ),
        encoding="utf-8",
    )
    seeded = P.seed_settings("image_dataset", home=tmp_path)

    data = _settings(tmp_path)
    assert data["stage_defaults"] == {"tagger_dir": "models/tagger"}
    assert data["values"]["autotag"] == {"mode": "merge"}
    # The form passes ExportRequest's validation: sidecars_only refuses both.
    assert data["values"]["export"] == {"combine_ocr": True, "sidecars_only": True}
    assert "dataset" not in data
    assert seeded.warnings == []


def test_seed_points_src_at_a_non_default_source_and_blanks_it_back(tmp_path):
    P.seed_settings("my_images/set_a", home=tmp_path)
    assert _settings(tmp_path)["dataset"]["src"] == "my_images/set_a"

    outside = tmp_path.parent / "elsewhere"
    P.seed_settings(outside, home=tmp_path)
    assert _settings(tmp_path)["dataset"]["src"] == str(outside.resolve())

    P.seed_settings("image_dataset", home=tmp_path)
    assert _settings(tmp_path)["dataset"]["src"] == ""


def test_seed_warns_when_a_workspace_root_points_into_the_trainer_tree(tmp_path):
    P.settings_path(tmp_path).write_text(
        json.dumps({"dataset": {"dst": "post_image_dataset/resized", "masks": ""}}),
        encoding="utf-8",
    )
    seeded = P.seed_settings("image_dataset", home=tmp_path)
    assert seeded.warnings == ["dst = post_image_dataset/resized"]
    # Warned about, not rewritten: that is the user's call in the panel.
    assert _settings(tmp_path)["dataset"]["dst"] == "post_image_dataset/resized"


def _touch(path, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_export_is_stale_tracks_workspace_edits_against_an_applied_export(tmp_path):
    assert not P.export_is_stale(tmp_path)  # no workspace yet

    now = time.time()
    caption = tmp_path / "workspace" / "resized" / "a" / "x.txt"
    _touch(caption, now - 100)
    assert P.export_is_stale(tmp_path)  # never exported

    report = P.export_report_path(tmp_path)
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"apply": False}), encoding="utf-8")
    assert P.export_is_stale(tmp_path)  # a dry run published nothing

    report.write_text(json.dumps({"apply": True}), encoding="utf-8")
    os.utime(report, (now - 50, now - 50))
    assert not P.export_is_stale(tmp_path)

    _touch(tmp_path / "workspace" / "masks" / "a" / "x_mask.png", now)
    assert P.export_is_stale(tmp_path)


def test_launch_argv_serves_this_home_and_exits_with_its_page(tmp_path):
    argv = P.launch_argv(tmp_path)
    assert argv[1:] == [
        "-m",
        "anime_tools.gui",
        "--home",
        str(tmp_path.resolve()),
        "--exit-with-window",
    ]


def test_url_in_log_reads_only_past_the_offset(tmp_path):
    log = tmp_path / "anime_tools_gui.log"
    old = "anime_tools GUI → http://127.0.0.1:8790   (home: x)\n"
    log.write_text(old, encoding="utf-8")
    offset = log.stat().st_size
    assert P.url_in_log(log, offset) is None
    with open(log, "a", encoding="utf-8") as f:
        f.write("port 8790 is in use; using 8791\n")
        f.write("anime_tools GUI → http://127.0.0.1:8791   (home: x)\n")
    assert P.url_in_log(log, offset) == "http://127.0.0.1:8791"
    # A detached child on Windows writes the arrow in the console code page.
    log.write_bytes("anime_tools GUI → http://127.0.0.1:8792\n".encode("cp949"))
    assert P.url_in_log(log) == "http://127.0.0.1:8792"
    assert P.url_in_log(tmp_path / "missing.log") is None
