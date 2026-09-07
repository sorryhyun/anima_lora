"""The model catalog and the ``make download-*`` targets that drive it.

The behaviour worth pinning is the *wiring*: rows land where loaders look, every
target resolves to real ids, the GUI's buttons run registered tasks, and a
present file is not re-fetched. Downloading itself is ``anime_tools``' tested
code — mock at the ``Asset.fetch`` boundary rather than at the hub.
"""

from __future__ import annotations

import pytest

from library import downloads as DL


def test_the_two_catalog_halves_have_disjoint_ids():
    """``by_id`` merges them; a collision would silently shadow a row."""
    trainer = {a.id for a in DL.catalog()}
    curation = {a.id for a in DL.curation_catalog()}
    assert not (trainer & curation)
    assert len(DL.by_id()) == len(trainer) + len(curation)


def test_every_group_and_the_default_set_name_real_rows():
    """A typo in ``GROUPS`` is a make target that downloads nothing."""
    known = set(DL.by_id())
    for group, ids in DL.GROUPS.items():
        assert set(ids) <= known, group
    assert set(DL.DEFAULT_SET) <= known


def test_the_first_run_set_leaves_the_gated_and_opt_in_rows_out():
    """v2 default: masking is opt-in, so SAM3 must not be in a fresh install's
    path — its gated repo was the first-run failure this removes."""
    assert "sam3" not in DL.DEFAULT_SET
    assert "vocab_pack" not in DL.DEFAULT_SET
    assert "mit_text" not in DL.DEFAULT_SET
    # …but the rows a default preprocess actually needs are.
    assert {"anima_dit", "anima_te", "anima_vae", "tagger"} <= set(DL.DEFAULT_SET)


def test_resolve_expands_groups_and_dedupes_in_catalog_order():
    ids = [a.id for a in DL.resolve(["pe", "pe_core", "anima"])]
    assert ids == ["anima_dit", "anima_te", "anima_vae", "pe_core", "pe_spatial"]


def test_resolve_raises_naming_the_unknown_token():
    with pytest.raises(KeyError) as exc:
        DL.resolve(["sam3", "nope"])
    assert "nope" in str(exc.value)


def test_rows_land_where_the_loaders_look():
    """The catalog's one rule: a path a loader owns separately is a Download
    button that writes where the loader will not look."""
    from library.anima import vocab_pack
    from library.vision import encoders

    rows = DL.by_id()
    assert encoders._default_pe_model_id() == str(DL.default_pe_core_path())
    assert rows["pe_core"].dest / DL.PE_CORE_FILENAME == DL.default_pe_core_path()

    assert vocab_pack.PACK_REPO == rows["vocab_pack"].repo
    assert vocab_pack.DEFAULT_PACK_PREFIX.startswith(f"models/{DL.VOCAB_PACK_DIR}/")
    assert rows["vocab_pack"].dest == DL.default_vocab_pack_dir()


def test_anima_rows_land_on_the_base_config_defaults():
    """``configs/base.toml`` points training at these three exact paths."""
    import tomllib
    from library.env import anima_home

    cfg = tomllib.loads((anima_home() / "configs" / "base.toml").read_text("utf-8"))
    rows = DL.by_id()
    configured = {
        "anima_dit": cfg["pretrained_model_name_or_path"],
        "anima_te": cfg["qwen3"],
        "anima_vae": cfg["vae"],
    }
    for row_id, rel in configured.items():
        row = rows[row_id]
        landed = row.dest / row.files[0].rsplit("/", 1)[-1]
        assert landed == anima_home() / rel, row_id


def test_prune_empty_drops_the_split_files_scaffolding(tmp_path):
    """``hf`` mirrors the repo layout under ``--local-dir``; after the flatten
    the empty tree must go, but a real file must not."""
    (tmp_path / "split_files" / "diffusion_models").mkdir(parents=True)
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "f.bin").write_bytes(b"x")

    DL._prune_empty(tmp_path)

    assert not (tmp_path / "split_files").exists()
    assert (tmp_path / "keep" / "f.bin").exists()


def test_fetch_skips_an_installed_row(monkeypatch):
    """Idempotency contract (GH #21): a re-run verifies, it does not re-fetch.

    Rows that move files out of ``hf``'s ``--local-dir`` layout would otherwise
    re-pull the whole repo, because the hub no longer sees them where it looks.
    """
    asset = DL.by_id()["sam3"]
    monkeypatch.setattr(type(asset), "installed", property(lambda _self: True))
    monkeypatch.setattr(
        type(asset), "fetch", lambda *_a, **_k: pytest.fail("re-fetched")
    )

    assert DL.fetch(asset, log=lambda _m: None) is False


def test_fetch_all_continues_past_a_failure_and_reports_it():
    """One gated repo without granted access must not abort the row beside it."""
    fetched: list[str] = []

    class _Row:
        def __init__(self, title, ok):
            self.title, self._ok, self.installed = title, ok, False
            self.repo = self.location = "-"
            self.dest = None

        def fetch(self, _log):
            if not self._ok:
                raise FileNotFoundError("gated")
            fetched.append(self.title)

    failed = DL.fetch_all(
        [_Row("a", True), _Row("b", False), _Row("c", True)], log=lambda _m: None
    )

    assert fetched == ["a", "c"]
    assert failed == ["b"]


def test_every_download_target_resolves(monkeypatch):
    """Each ``make download-<x>`` must name rows that exist — the failure mode
    is a target that prints nothing and exits 0."""
    import tasks

    picked: list[list[str]] = []
    monkeypatch.setattr(
        DL,
        "fetch_all",
        lambda assets, **_kw: picked.append([a.id for a in assets]) or [],
    )
    targets = [
        n
        for n in tasks.COMMANDS
        if n.startswith("download-")
        and n not in ("download-anima-variant", "download-list", "download-model")
    ]
    assert len(targets) >= 8
    for name in targets:
        tasks.COMMANDS[name][0]([])
    assert all(ids for ids in picked)


def test_models_dialog_rows_come_from_the_catalog():
    """Both tabs render Assets and run ``download-model <id>``; the id has to be
    one the task can resolve, and every labelled row has to be a real one."""
    pytest.importorskip("PySide6")
    import tasks
    from gui.system_dialog import _TITLE_KEYS

    assert "download-model" in tasks.COMMANDS
    assert set(_TITLE_KEYS) <= set(DL.by_id())


def test_the_two_tabs_show_the_two_catalog_halves():
    """One modal, one QProcess, one log — the split is a tab, so neither list
    grows long enough to push the log pane off the dialog."""
    import os

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from gui.system_dialog import ModelsDialog

    QApplication.instance() or QApplication([])
    dlg = ModelsDialog()
    try:
        assert dlg.tabs.count() == 2
        anima, curation = dlg._panels
        assert [a.id for a in anima.assets()] == [a.id for a in DL.catalog()]
        assert [a.id for a in curation.assets()] == [
            a.id for a in DL.curation_catalog()
        ]
        # Busy disables every button on *both* tabs: the dialog runs one job.
        dlg._set_busy(True)
        assert not any(b.isEnabled() for p in dlg._panels for _a, _s, b in p._rows)
        assert not any(p.all_btn.isEnabled() for p in dlg._panels)
    finally:
        dlg.close()


def test_every_title_key_exists_in_every_language():
    """A missing key silently falls back to English — worst for the KR/JA/ZH
    users who are most of the base."""
    pytest.importorskip("PySide6")
    from gui.i18n import TRANSLATIONS
    from gui.system_dialog import _TITLE_KEYS

    extra = (
        "curation_models_intro",
        "models_tab_anima",
        "models_tab_curation",
        "models_download_missing",
        "models_all_installed",
        "models_used_by",
    )
    for lang, table in TRANSLATIONS.items():
        for key in (*_TITLE_KEYS.values(), *extra):
            assert key in table, f"{lang}: {key}"
