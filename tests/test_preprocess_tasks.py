from __future__ import annotations


def _entry(cmd: list[str]) -> str:
    """The child's entry: the module of a ``-m`` invocation (the ``anime_tools``
    caption stages) or the script path (trainer-side cache scripts)."""
    return cmd[2] if cmd[1] == "-m" else cmd[1]


def _patch_run(monkeypatch, fn) -> None:
    """Stub both child launches: ``preprocess.run`` (the trainer-side cache
    scripts) and ``_common.run`` (what ``execute_stage`` uses for a curation
    stage from a shell). Also pin the shell path — a suite running as a daemon
    job would otherwise run the stages in-process."""
    from scripts.tasks import _common, preprocess

    monkeypatch.setattr(preprocess, "run", fn)
    monkeypatch.setattr(_common, "run", fn)
    monkeypatch.delenv("ANIMA_DAEMON_JOB_DIR", raising=False)


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

    _patch_run(monkeypatch, lambda cmd: calls.append(cmd))
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

    assert caption_cmd[:3] == [
        preprocess.PY,
        "-m",
        "anime_tools.stages.cli.correct_captions",
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

    _patch_run(monkeypatch, lambda cmd: calls.append(cmd))
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
    assert _entry(caption_cmd) == "anime_tools.stages.cli.correct_captions"
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


def test_caption_position_clauses_is_not_a_correction_flag():
    """It gates its own stage — it must never reach ``correct_captions.py``.

    ``position_clauses`` rides in the caption-correction config dict (same
    family of caption-master rewrites, one GUI box), but enabling it alone must
    not turn the correction pass on or add an argv flag it doesn't know.
    """
    from scripts.tasks.preprocess import (
        _caption_correction_fields,
        _caption_correction_config,
        _caption_correction_enabled,
    )

    config, cleaned = _caption_correction_config(
        ["--caption_position_clauses", "--other"]
    )

    assert config["position_clauses"] is True
    assert cleaned == ["--other"]
    assert _caption_correction_enabled(config) is False
    assert _caption_correction_fields(config) == {}

    off, _ = _caption_correction_config(["--no_caption_position_clauses"])
    assert off["position_clauses"] is False


def test_preprocess_chains_position_clauses_before_the_caption_step(monkeypatch):
    """The stage rewrites the derived caption, so it must land before the mirror.

    Ordering is the whole contract: the caption step re-corrects that same file
    and writes the variant sidecars around it, so a position pass that ran after
    it would be encoded only on the *next* preprocess. It also has to run inline
    (plain ``run``) — this process is itself a daemon job on a serial queue.
    """
    from scripts.tasks import preprocess

    order: list[str] = []

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_repa_pe_encoder", lambda: None)
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_resize", lambda *_a, **_k: order.append("resize")
    )
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_vae", lambda *_a, **_k: order.append("vae")
    )
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_te", lambda *_a, **_k: order.append("te")
    )
    monkeypatch.setattr(preprocess, "cmd_caption_index", lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess.os.path, "exists", lambda _p: True)

    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        order.append("position")

    _patch_run(monkeypatch, fake_run)
    monkeypatch.setenv("CAPTION_POSITION_CLAUSES", "1")

    preprocess.cmd_preprocess([])

    assert order == ["resize", "vae", "position", "te"]
    (cmd,) = calls
    assert cmd[:3] == [preprocess.PY, "-m", "anime_tools.stages.cli.position_captions"]
    assert "--apply" in cmd


def test_preprocess_skips_position_clauses_when_unset(monkeypatch):
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_repa_pe_encoder", lambda: None)
    for name in ("cmd_preprocess_resize", "cmd_preprocess_vae", "cmd_preprocess_te"):
        monkeypatch.setattr(preprocess, name, lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess, "cmd_caption_index", lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess.os.path, "exists", lambda _p: True)

    calls: list[list[str]] = []
    _patch_run(monkeypatch, lambda cmd, **_k: calls.append(cmd))
    monkeypatch.delenv("CAPTION_POSITION_CLAUSES", raising=False)
    monkeypatch.delenv("CAPTION_AUTOTAG", raising=False)

    preprocess.cmd_preprocess([])

    assert calls == []


def test_preprocess_captions_runs_the_master_stages_when_configured(monkeypatch):
    """`preprocess-captions` mirrors the master, so it owns the stages too.

    Run on its own (not through the full chain) it must still honour
    ``caption_position_clauses`` / ``caption_autotag`` — otherwise the mirror,
    and every TE cache encoded from it, silently predates the rewrite.
    """
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_ensure_danbooru_tags", lambda: None)
    monkeypatch.setattr(preprocess, "_variant_settings", lambda: ("4", "0.1", "0.0"))

    calls: list[list[str]] = []
    _patch_run(monkeypatch, lambda cmd, **_k: calls.append(cmd))
    monkeypatch.setenv("CAPTION_AUTOTAG", "1")
    monkeypatch.setenv("CAPTION_POSITION_CLAUSES", "1")

    preprocess.cmd_preprocess_captions([])

    scripts = [_entry(cmd) for cmd in calls]
    assert scripts == [
        "anime_tools.stages.cli.autotag_captions",
        "anime_tools.stages.cli.position_captions",
        "anime_tools.stages.cli.correct_captions",
    ]
    # In-pipeline stages write for real — a dry run here would leave the mirror
    # encoding the un-rewritten caption.
    assert "--apply" in calls[0]
    assert "--apply" in calls[1]


def test_master_stages_inherit_an_explicit_path_pattern(monkeypatch):
    """A scoped preprocess must not rewrite captions dataset-wide.

    The caption-MASTER stages are driven from the caption-config dict alone —
    the caller's ``extra`` never reaches them — so the resolved subset scope
    has to ride in that dict. Without it, ``make preprocess-captions
    ARGS="--path_pattern artistA/*"`` would cache the requested slice while
    autotag ``merge``/``overwrite`` rewrote every caption in the master.
    """
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_ensure_danbooru_tags", lambda: None)
    monkeypatch.setattr(preprocess, "_variant_settings", lambda: ("0", "0.0", "0.0"))

    calls: list[list[str]] = []
    _patch_run(monkeypatch, lambda cmd, **_k: calls.append(cmd))
    monkeypatch.setenv("CAPTION_AUTOTAG", "1")
    monkeypatch.setenv("CAPTION_AUTOTAG_MODE", "merge")
    monkeypatch.setenv("CAPTION_POSITION_CLAUSES", "1")
    monkeypatch.delenv("PREPROCESS_PATH_PATTERN", raising=False)

    preprocess.cmd_preprocess_captions(["--path_pattern", "artistA/*"])

    assert [_entry(cmd) for cmd in calls] == [
        "anime_tools.stages.cli.autotag_captions",
        "anime_tools.stages.cli.position_captions",
        # The mirror rides along whenever position clauses are on — they land in
        # `resized/`, so every other caption has to be mirrored there too.
        "anime_tools.stages.cli.correct_captions",
    ]
    for cmd in calls:
        # Present, and emitted exactly once — the argv builder resolves the
        # scope and drops it from the tail rather than splatting it twice.
        assert cmd.count("--path_pattern") == 1
        assert cmd[cmd.index("--path_pattern") + 1] == "artistA/*"


def test_standalone_caption_target_does_not_duplicate_the_path_pattern(monkeypatch):
    """`make caption-position ARGS="--path_pattern x"` passes it through once,
    overriding the GUI/config scope rather than splatting both."""
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setenv("PREPROCESS_PATH_PATTERN", "gui/*")
    req = preprocess._caption_position_request(
        ["--path_pattern", "artistA/*", "--apply"]
    )
    argv = req.to_argv()
    assert argv.count("--path_pattern") == 1
    assert argv[argv.index("--path_pattern") + 1] == "artistA/*"
    assert "--apply" in argv


def test_preprocess_captions_runs_the_stages_even_with_correction_off(monkeypatch):
    """The early "correction disabled" return must not swallow the stages.

    With position clauses on, the mirror has to run as well even though nothing
    is being corrected or shuffled: the clause rewrite lands in ``resized/`` and
    the TE step therefore reads that tree, so every image the rewrite did *not*
    touch still needs its master caption mirrored there.
    """
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_ensure_danbooru_tags", lambda: None)
    monkeypatch.setattr(preprocess, "_variant_settings", lambda: ("0", "0.0", "0.0"))

    calls: list[list[str]] = []
    _patch_run(monkeypatch, lambda cmd, **_k: calls.append(cmd))
    monkeypatch.delenv("CAPTION_AUTOTAG", raising=False)
    monkeypatch.setenv("CAPTION_POSITION_CLAUSES", "1")

    preprocess.cmd_preprocess_captions([])

    assert [_entry(cmd) for cmd in calls] == [
        "anime_tools.stages.cli.position_captions",
        "anime_tools.stages.cli.correct_captions",
    ]
    # Nothing to correct and no variants — the mirror runs in passthrough.
    assert "--no_correct" in calls[1]


def test_preprocess_te_reads_resized_when_position_clauses_are_on(monkeypatch):
    """Clauses live in ``resized/``, so TE must encode that tree, not the master.

    No correction + no variants would normally encode the caption master
    directly with a match filter — which would silently train the pre-clause
    caption, since the position pass never writes the master.
    """
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_ensure_danbooru_tags", lambda: None)
    monkeypatch.setattr(preprocess, "_variant_settings", lambda: ("0", "0.0", "0.0"))

    calls: list[list[str]] = []
    _patch_run(monkeypatch, lambda cmd, **_k: calls.append(cmd))
    monkeypatch.delenv("CAPTION_AUTOTAG", raising=False)
    monkeypatch.setenv("CAPTION_POSITION_CLAUSES", "1")

    preprocess.cmd_preprocess_te([])

    assert [_entry(cmd) for cmd in calls] == [
        "anime_tools.stages.cli.position_captions",
        "anime_tools.stages.cli.correct_captions",
        "scripts/preprocess/cache_text_embeddings.py",
    ]
    te_cmd = calls[-1]
    assert te_cmd[te_cmd.index("--dir") + 1] == "post_image_dataset/resized"
    assert "--match_images_from" not in te_cmd


def test_preprocess_chain_runs_each_master_stage_once(monkeypatch):
    """`preprocess` runs the stages early; the caption/TE steps must not repeat.

    Both passes are idempotent on a rewritten caption, but each re-pays a SAM3 /
    tagger load over the whole tree. The shared caption-config dict is what
    carries "already ran" down the chain.
    """
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_repa_pe_encoder", lambda: None)
    monkeypatch.setattr(preprocess, "_ensure_danbooru_tags", lambda: None)
    monkeypatch.setattr(preprocess, "_variant_settings", lambda: ("4", "0.1", "0.0"))
    monkeypatch.setattr(preprocess, "cmd_preprocess_resize", lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess, "cmd_preprocess_vae", lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess, "cmd_caption_index", lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess.os.path, "exists", lambda _p: True)

    calls: list[list[str]] = []
    _patch_run(monkeypatch, lambda cmd, **_k: calls.append(cmd))
    monkeypatch.setenv("CAPTION_AUTOTAG", "1")
    monkeypatch.setenv("CAPTION_POSITION_CLAUSES", "1")

    preprocess.cmd_preprocess([])

    scripts = [_entry(cmd) for cmd in calls]
    assert scripts.count("anime_tools.stages.cli.autotag_captions") == 1
    assert scripts.count("anime_tools.stages.cli.position_captions") == 1
    # Order is unchanged: the caption rewrites, then the mirror, then the encode.
    assert scripts == [
        "anime_tools.stages.cli.autotag_captions",
        "anime_tools.stages.cli.position_captions",
        "anime_tools.stages.cli.correct_captions",
        "scripts/preprocess/cache_text_embeddings.py",
    ]


def test_caption_autotag_is_not_a_correction_flag():
    """Like ``position_clauses``: own stage, must never reach correct_captions.py."""
    from scripts.tasks.preprocess import (
        _caption_correction_fields,
        _caption_correction_config,
        _caption_correction_enabled,
    )

    config, cleaned = _caption_correction_config(
        ["--caption_autotag", "--caption_autotag_mode", "merge", "--other"]
    )

    assert config["autotag"] is True
    assert config["autotag_mode"] == "merge"
    assert cleaned == ["--other"]
    assert _caption_correction_enabled(config) is False
    assert _caption_correction_fields(config) == {}

    off, _ = _caption_correction_config(["--no_caption_autotag"])
    assert off["autotag"] is False
    # Mode defaults to the non-destructive one when nothing selects it.
    assert off["autotag_mode"] == "missing"


def test_caption_autotag_rejects_an_unknown_mode():
    """Fail at argv-parse time, not minutes into a GPU job."""
    import pytest

    from scripts.tasks.preprocess import _caption_correction_config

    with pytest.raises(SystemExit):
        _caption_correction_config(["--caption_autotag_mode", "clobber"])


def test_caption_autotag_request_always_applies(monkeypatch):
    """In-pipeline the user already opted in; a dry run there writes nothing."""
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.delenv("PREPROCESS_PATH_PATTERN", raising=False)
    req = preprocess._autotag_request({"autotag_mode": "missing"})
    assert (req.mode, req.apply) == ("missing", True)
    argv = req.to_argv()
    assert "--apply" in argv and "--mode" not in argv  # defaults stay unspelled
    # A zero floor is the request default, so the tagger's own thresholds rule.
    req = preprocess._autotag_request(
        {"autotag_mode": "merge", "autotag_min_confidence": 0.0}
    )
    assert "--min_confidence" not in req.to_argv()
    req = preprocess._autotag_request(
        {"autotag_mode": "merge", "autotag_min_confidence": 0.35}
    )
    assert (req.mode, req.min_confidence, req.apply) == ("merge", 0.35, True)


def test_preprocess_chains_autotag_first(monkeypatch):
    """Autotag *creates* the captions every later stage reads, so it goes first.

    Ordering is the contract: position clauses append to the master, correction
    and TE read it. An autotag that ran after any of them would leave this run
    encoding the un-tagged caption.
    """
    from scripts.tasks import preprocess

    order: list[str] = []

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_repa_pe_encoder", lambda: None)
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_resize", lambda *_a, **_k: order.append("resize")
    )
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_vae", lambda *_a, **_k: order.append("vae")
    )
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_te", lambda *_a, **_k: order.append("te")
    )
    monkeypatch.setattr(preprocess, "cmd_caption_index", lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess.os.path, "exists", lambda _p: True)

    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        order.append("autotag" if "autotag_captions" in _entry(cmd) else "position")

    _patch_run(monkeypatch, fake_run)
    monkeypatch.setenv("CAPTION_AUTOTAG", "1")
    monkeypatch.setenv("CAPTION_AUTOTAG_MODE", "merge")
    monkeypatch.setenv("CAPTION_POSITION_CLAUSES", "1")

    preprocess.cmd_preprocess([])

    assert order == ["resize", "autotag", "vae", "position", "te"]
    autotag_cmd = calls[0]
    assert autotag_cmd[:3] == [
        preprocess.PY,
        "-m",
        "anime_tools.stages.cli.autotag_captions",
    ]
    assert "--apply" in autotag_cmd
    assert autotag_cmd[autotag_cmd.index("--mode") + 1] == "merge"


def test_preprocess_autotag_blank_env_confidence_is_zero(monkeypatch):
    """The GUI writes ``""`` for an empty field — that must not raise."""
    from scripts.tasks.preprocess import _caption_correction_config

    monkeypatch.setenv("CAPTION_AUTOTAG", "1")
    monkeypatch.setenv("CAPTION_AUTOTAG_MIN_CONFIDENCE", "")

    config, _ = _caption_correction_config([])
    assert config["autotag_min_confidence"] == 0.0


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


def test_preprocess_task_wiring_forwards_drop_groups(monkeypatch):
    """``--caption_drop_groups`` (GH #95) threads through the task runner."""
    from scripts.tasks.preprocess import (
        _caption_correction_fields,
        _caption_correction_config,
        _caption_correction_enabled,
    )

    monkeypatch.delenv("CAPTION_DROP_GROUPS", raising=False)
    config, cleaned = _caption_correction_config(
        ["--caption_drop_groups", "artist,lighting", "--other", "x"]
    )
    assert config["drop_groups"] == "artist,lighting"
    assert cleaned == ["--other", "x"]
    assert _caption_correction_enabled({"drop_groups": "artist"})
    assert not _caption_correction_enabled({"drop_groups": "  "})
    assert _caption_correction_fields({"drop_groups": "artist,lighting"}) == {
        "caption_drop_groups": "artist,lighting"
    }

    monkeypatch.setenv("CAPTION_DROP_GROUPS", "pose")
    config, _ = _caption_correction_config([])
    assert config["drop_groups"] == "pose"
