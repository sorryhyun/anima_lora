from __future__ import annotations


def test_preprocess_te_uses_corrected_resized_captions(monkeypatch):
    from scripts.tasks import preprocess

    calls: list[list[str]] = []

    def fake_path(key: str, default: str) -> str:
        values = {
            "source_image_dir": "image_dataset",
            "resized_image_dir": "post_image_dataset/resized",
            "lora_cache_dir": "post_image_dataset/lora",
        }
        return values.get(key, default)

    monkeypatch.setattr(preprocess, "run", lambda cmd: calls.append(cmd))
    monkeypatch.setattr(preprocess, "_path", fake_path)

    preprocess.cmd_preprocess_te(
        ["--min_pixels", "500000", "--path_pattern", "group/*"],
        caption_config={
            "correct_order": True,
            "insert_no_artist": True,
            "trigger_word": "@dataset-trigger",
            "trigger_at_front": False,
        },
    )

    assert len(calls) == 2
    caption_cmd, te_cmd = calls

    assert caption_cmd[:2] == [
        preprocess.PY,
        "scripts/preprocess/correct_captions.py",
    ]
    assert caption_cmd[caption_cmd.index("--src") + 1] == "image_dataset"
    assert caption_cmd[caption_cmd.index("--dst") + 1] == "post_image_dataset/resized"
    assert caption_cmd[caption_cmd.index("--path_pattern") + 1] == "group/*"
    assert "--caption_insert_no_artist" in caption_cmd
    assert caption_cmd[caption_cmd.index("--caption_trigger_word") + 1] == (
        "@dataset-trigger"
    )

    assert te_cmd[:2] == [
        preprocess.PY,
        "scripts/preprocess/cache_text_embeddings.py",
    ]
    assert te_cmd[te_cmd.index("--dir") + 1] == "post_image_dataset/resized"
    assert "--match_images_from" not in te_cmd
    assert te_cmd[te_cmd.index("--cache_dir") + 1] == "post_image_dataset/lora"
    assert te_cmd[te_cmd.index("--path_pattern") + 1] == "group/*"
    assert [i for i, arg in enumerate(te_cmd) if arg == "--min_pixels"] == [
        te_cmd.index("--min_pixels")
    ]
    assert te_cmd[te_cmd.index("--min_pixels") + 1] == "0"


def test_caption_correction_enabled_when_only_trigger_or_no_artist_set():
    from scripts.tasks.preprocess import _caption_correction_enabled

    base = {
        "correct_order": False,
        "insert_no_artist": False,
        "trigger_word": "",
        "trigger_at_front": False,
    }
    # Nothing set → no rewrite pass.
    assert not _caption_correction_enabled(base)
    # Trigger word alone must still run the pass (the bug: it was being dropped).
    assert _caption_correction_enabled({**base, "trigger_word": "@trig"})
    # Whitespace-only trigger does not count.
    assert not _caption_correction_enabled({**base, "trigger_word": "   "})
    # Insert-no-artist alone also requires the pass.
    assert _caption_correction_enabled({**base, "insert_no_artist": True})


def test_preprocess_te_runs_correction_for_trigger_word_without_correct_order(
    monkeypatch,
):
    from scripts.tasks import preprocess

    calls: list[list[str]] = []

    def fake_path(key: str, default: str) -> str:
        values = {
            "source_image_dir": "image_dataset",
            "resized_image_dir": "post_image_dataset/resized",
            "lora_cache_dir": "post_image_dataset/lora",
        }
        return values.get(key, default)

    monkeypatch.setattr(preprocess, "run", lambda cmd: calls.append(cmd))
    monkeypatch.setattr(preprocess, "_path", fake_path)

    preprocess.cmd_preprocess_te(
        [],
        caption_config={
            "correct_order": False,
            "insert_no_artist": False,
            "trigger_word": "@dataset-trigger",
            "trigger_at_front": False,
        },
    )

    # Correction pass runs (trigger injected) even though correct_order is off,
    # and the TE cache then reads the corrected captions from the resized dir.
    assert len(calls) == 2
    caption_cmd, te_cmd = calls
    assert caption_cmd[1] == "scripts/preprocess/correct_captions.py"
    assert caption_cmd[caption_cmd.index("--caption_trigger_word") + 1] == (
        "@dataset-trigger"
    )
    assert te_cmd[te_cmd.index("--dir") + 1] == "post_image_dataset/resized"


def _stub_overrides(monkeypatch, overrides: dict) -> None:
    """Pin the merged-config read both builders fall back to when env is unset."""
    from scripts.tasks import _common

    monkeypatch.setattr(_common, "_path_overrides", lambda: dict(overrides))


def test_min_pixels_args_env_drop_false_keeps_every_image(monkeypatch):
    """GUI auto-chain unchecks low-res → DROP_LOWRES_IMAGES=0 forces --min_pixels 0,
    overriding a merged config that still says drop=true (the snapshot strips it)."""
    from scripts.tasks.preprocess import _min_pixels_args

    _stub_overrides(monkeypatch, {"drop_lowres_images": True, "min_pixels": 250_000})
    monkeypatch.setenv("DROP_LOWRES_IMAGES", "0")
    monkeypatch.setenv("MIN_PIXELS", "250000")

    assert _min_pixels_args() == ["--min_pixels", "0"]


def test_min_pixels_args_env_drop_true_uses_env_threshold(monkeypatch):
    from scripts.tasks.preprocess import _min_pixels_args

    _stub_overrides(monkeypatch, {})
    monkeypatch.setenv("DROP_LOWRES_IMAGES", "1")
    monkeypatch.setenv("MIN_PIXELS", "250000")

    assert _min_pixels_args() == ["--min_pixels", "250000"]


def test_min_pixels_args_no_env_falls_back_to_config(monkeypatch):
    from scripts.tasks.preprocess import _min_pixels_args

    _stub_overrides(monkeypatch, {"drop_lowres_images": False, "min_pixels": 250_000})
    monkeypatch.delenv("DROP_LOWRES_IMAGES", raising=False)
    monkeypatch.delenv("MIN_PIXELS", raising=False)

    assert _min_pixels_args() == ["--min_pixels", "0"]


def test_target_res_args_env_wins_over_config(monkeypatch):
    from scripts.tasks.preprocess import _target_res_args

    _stub_overrides(monkeypatch, {"target_res": [1024]})
    monkeypatch.setenv("TARGET_RES", "1024 896")

    assert _target_res_args([]) == ["--target_res", "1024", "896"]
    # An explicit CLI --target_res still wins over both env and config.
    assert _target_res_args(["--target_res", "768"]) == []


def test_caption_correction_config_parses_trigger_cli_args():
    from scripts.tasks.preprocess import _caption_correction_config

    config, cleaned = _caption_correction_config(
        [
            "--caption_trigger_word",
            "@foo",
            "--caption_trigger_at_front",
            "--other",
        ]
    )

    assert config["trigger_word"] == "@foo"
    assert config["trigger_at_front"] is True
    assert cleaned == ["--other"]


def test_sigma_demote_routes_true_is_the_certified_route(monkeypatch):
    from scripts.tasks.preprocess import _sigma_demote_routes

    _stub_overrides(monkeypatch, {"sigma_demote": True})
    monkeypatch.delenv("SIGMA_DEMOTE", raising=False)
    assert _sigma_demote_routes([]) == ["1024:896"]


def test_sigma_demote_routes_off_by_default(monkeypatch):
    from scripts.tasks.preprocess import _sigma_demote_routes

    monkeypatch.delenv("SIGMA_DEMOTE", raising=False)
    for value in ({}, {"sigma_demote": False}, {"sigma_demote": "off"}):
        _stub_overrides(monkeypatch, value)
        assert _sigma_demote_routes([]) == []


def test_sigma_demote_routes_comma_list_feeds_the_stacked_router(monkeypatch):
    """The stacked router (--sigma_lowres_route2) needs BOTH routes' sibling
    keys; one comma-listed config value must emit one pass per route, in
    order, deduplicated."""
    from scripts.tasks.preprocess import _sigma_demote_routes

    _stub_overrides(monkeypatch, {})
    monkeypatch.setenv("SIGMA_DEMOTE", "1024:896, 1024:768 ,1024:896")
    assert _sigma_demote_routes([]) == ["1024:896", "1024:768"]


def test_sigma_demote_routes_skips_malformed_entries(monkeypatch):
    from scripts.tasks.preprocess import _sigma_demote_routes

    _stub_overrides(monkeypatch, {})
    monkeypatch.setenv("SIGMA_DEMOTE", "1024:896,nonsense,")
    assert _sigma_demote_routes([]) == ["1024:896"]


def test_sigma_demote_routes_never_chains_from_a_demote_run(monkeypatch):
    """An explicit --sigma_demote means this invocation IS the demote pass."""
    from scripts.tasks.preprocess import _sigma_demote_routes

    _stub_overrides(monkeypatch, {"sigma_demote": "1024:896,1024:768"})
    monkeypatch.delenv("SIGMA_DEMOTE", raising=False)
    assert _sigma_demote_routes(["--sigma_demote", "1024:768"]) == []


def _demote_passes(monkeypatch, extra) -> list[str]:
    """Run cmd_preprocess_demote with the subprocess stubbed; return the routes
    each `cache_latents.py` invocation actually received."""
    from scripts.tasks import preprocess as pp

    seen: list[str] = []

    def fake_run(cmd):
        assert "--sigma_demote" in cmd
        seen.append(cmd[cmd.index("--sigma_demote") + 1])

    monkeypatch.setattr(pp, "run", fake_run)
    monkeypatch.setattr(pp, "_path", lambda key, default="": default)
    pp.cmd_preprocess_demote(list(extra))
    return seen


def test_preprocess_demote_target_emits_every_configured_route(monkeypatch):
    """`make preprocess-demote` must honor the SAME comma list as the automatic
    chain — otherwise the stacked router's deep route silently degrades because
    only 1024:896 ever landed on disk."""
    _stub_overrides(monkeypatch, {"sigma_demote": "1024:896,1024:768"})
    monkeypatch.delenv("SIGMA_DEMOTE", raising=False)
    monkeypatch.delenv("PREPROCESS_PATH_PATTERN", raising=False)

    assert _demote_passes(monkeypatch, []) == ["1024:896", "1024:768"]


def test_preprocess_demote_target_falls_back_to_the_certified_route(monkeypatch):
    """Nothing configured → the target still does its historical single pass."""
    _stub_overrides(monkeypatch, {})
    monkeypatch.delenv("SIGMA_DEMOTE", raising=False)
    monkeypatch.delenv("PREPROCESS_PATH_PATTERN", raising=False)

    assert _demote_passes(monkeypatch, []) == ["1024:896"]


def test_preprocess_demote_args_override_and_split(monkeypatch):
    """ARGS wins over the config, and a comma list there is expanded too —
    cache_latents.py parses one route per invocation."""
    _stub_overrides(monkeypatch, {"sigma_demote": "1024:896"})
    monkeypatch.delenv("SIGMA_DEMOTE", raising=False)
    monkeypatch.delenv("PREPROCESS_PATH_PATTERN", raising=False)

    assert _demote_passes(monkeypatch, ["--sigma_demote", "1280:1024, 1024:768"]) == [
        "1280:1024",
        "1024:768",
    ]
