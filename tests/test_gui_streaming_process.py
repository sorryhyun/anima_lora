"""StreamingProcess: line splitting, tail flush, UTF-8 across reads, env layering.

Runs a real ``python -c`` child under the offscreen Qt platform.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.jobs.process import StreamingProcess  # noqa: E402

_CHILD = r"""
import os, sys, time
han = "한".encode()
sys.stdout.buffer.write(b"one\ntwo\rthree " + han[:1]); sys.stdout.flush()
time.sleep(0.2)
sys.stdout.buffer.write(han[1:] + b"\n" + b"FOO=" + os.environ.get("FOO", "-").encode()
                        + b" GONE=" + os.environ.get("GONE", "-").encode()
                        + b" UNBUF=" + os.environ.get("PYTHONUNBUFFERED", "-").encode())
sys.stdout.flush()
sys.stderr.write("err1\nerr-tail"); sys.stderr.flush()
sys.exit(3)
"""


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _run(proc: StreamingProcess, args, **kw) -> int:
    loop = QEventLoop()
    codes: list[int] = []
    proc.finished.connect(lambda code: (codes.append(code), loop.quit()))
    QTimer.singleShot(15_000, loop.quit)
    assert proc.start(args, **kw)
    loop.exec()
    assert codes, "child did not finish"
    return codes[0]


def test_lines_tail_and_env(qapp, monkeypatch):
    monkeypatch.setenv("GONE", "present")
    proc = StreamingProcess()
    lines: list[tuple[str, bool]] = []
    chunks: list[tuple[str, bool]] = []
    proc.line.connect(lambda s, err: lines.append((s, err)))
    proc.chunk.connect(lambda s, err: chunks.append((s, err)))

    code = _run(proc, ["-c", _CHILD], env={"FOO": "bar", "GONE": None})

    assert code == 3
    out = [s for s, err in lines if not err]
    err = [s for s, err in lines if err]
    assert out == ["one", "two", "three 한", "FOO=bar GONE=- UNBUF=1"]
    assert err == ["err1", "err-tail"]
    assert "".join(s for s, e in chunks if not e).endswith("UNBUF=1")
    assert "�" not in "".join(s for s, _ in chunks)
    assert not proc.is_running()


def test_custom_separators_and_busy_guard(qapp):
    proc = StreamingProcess(separators="\n")
    lines: list[str] = []
    proc.line.connect(lambda s, err: lines.append(s) if not err else None)
    code = _run(proc, ["-c", "print('a\\rb'); print('c', end='')"])
    assert code == 0
    assert lines == ["a\rb", "c"]

    loop = QEventLoop()
    proc.finished.connect(lambda _c: loop.quit())
    assert proc.start(["-c", "import time; time.sleep(30)"])
    assert not proc.start(["-c", "pass"])
    proc.kill()
    QTimer.singleShot(10_000, loop.quit)
    loop.exec()
    assert not proc.is_running()
