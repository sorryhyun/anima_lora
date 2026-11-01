"""Regression tests for the three indefinite-hang guards added to the pipeline.

These close the "I clicked Train/Preprocess and it spins forever" class:
  1. ``safe_walk`` — a symlink cycle in the (symlinked) dataset tree used to
     make ``os.walk(followlinks=True)`` loop forever;
  2. ``ensure_hf_timeouts`` — the hub's read timeouts are pinned so a stuck
     download fails instead of spinning, without clobbering a user-set value;
  3. the daemon stall watchdog — a running job that stops producing output is
     detected and finalized with a "where it wedged" diagnostic.
"""

from __future__ import annotations

import os
import time
import types

import pytest


# ----- 1. safe_walk cycle guard -----


def _make_cyclic_tree(root: str) -> None:
    os.makedirs(os.path.join(root, "a"))
    open(os.path.join(root, "top.txt"), "w").close()
    open(os.path.join(root, "a", "img1.png"), "w").close()
    os.symlink(root, os.path.join(root, "a", "back"))  # a/back -> root (cycle)
    os.symlink(os.path.join(root, "a"), os.path.join(root, "a", "self2"))  # diamond


@pytest.mark.core
def test_safe_walk_terminates_on_symlink_cycle(tmp_path):
    from library.io.walk import safe_walk

    _make_cyclic_tree(str(tmp_path))
    files = []
    t0 = time.time()
    for _dp, _dn, fn in safe_walk(str(tmp_path), followlinks=True):
        files.extend(fn)
        assert time.time() - t0 < 10, "safe_walk did not terminate on a cycle"
    # Each real file surfaces exactly once despite the cycle + diamond link.
    assert sorted(files) == ["img1.png", "top.txt"]


# ----- 2. hub download timeouts -----


def test_ensure_hf_timeouts_pins_env(monkeypatch):
    from anime_tools._hf import ensure_hf_timeouts

    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.delenv("HF_HUB_ETAG_TIMEOUT", raising=False)
    ensure_hf_timeouts()
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]
    assert os.environ["HF_HUB_ETAG_TIMEOUT"]
    # Respects a user-set value rather than clobbering it.
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "5")
    ensure_hf_timeouts()
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "5"


# ----- 3. daemon stall watchdog -----


def _fake_job(stdout_path, started_at, kind="train"):
    return types.SimpleNamespace(
        id="job_test",
        kind=kind,
        started_at=started_at,
        stdout_path=stdout_path,
        progress_path=None,
    )


@pytest.mark.core
def test_stall_watchdog_fires_with_diagnostic(tmp_path, monkeypatch):
    from anima_daemon import config
    from anima_daemon.manager import JobManager

    stdout = tmp_path / "stdout.log"
    stdout.write_text("loading PE-Core...\rfetching sorryhyun/pe-core (one-time).\n")
    monkeypatch.setattr(config, "JOB_STALL_TIMEOUT", 900, raising=False)

    # Fresh output → alive.
    job = _fake_job(str(stdout), time.time())
    assert JobManager._stall_reason(job) is None

    # Backdate output + start beyond the timeout → stalled, naming last line.
    old = time.time() - 1000
    os.utime(stdout, (old, old))
    job.started_at = old
    reason = JobManager._stall_reason(job)
    assert reason is not None
    assert "no output for" in reason
    assert "fetching sorryhyun/pe-core" in reason  # the "where it wedged" hint


def test_stall_watchdog_budget_is_per_kind(tmp_path, monkeypatch):
    """With training unwatched by default, the same long silence is a stall for
    a command (preprocess) job but healthy for a train job."""
    from anima_daemon import config
    from anima_daemon.manager import JobManager

    monkeypatch.setattr(config, "CMD_STALL_TIMEOUT", 120, raising=False)
    monkeypatch.setattr(config, "JOB_STALL_TIMEOUT", 0, raising=False)

    stdout = tmp_path / "stdout.log"
    stdout.write_text("Caching latents: 40%\n")
    old = time.time() - 200  # 200s of silence
    os.utime(stdout, (old, old))

    cmd_job = _fake_job(str(stdout), old, kind="command")
    train_job = _fake_job(str(stdout), old, kind="train")
    assert JobManager._stall_reason(cmd_job) is not None  # past 120s → stalled
    assert JobManager._stall_reason(train_job) is None  # unwatched → alive


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
