"""Progress-bar drivers for the tabs: tqdm stdout lines and ``progress.jsonl``.

``TqdmProgressTracker`` matches tqdm's textual format and computes s/step from
the first completed step; ``JsonlProgressReader`` reads the trainer's
structured event stream.

Use as::

    self.tracker = TqdmProgressTracker(self.progress)
    ...
    line = parts[0]
    if not self.tracker.feed(line):
        self._log(line + "\\n")
"""

from __future__ import annotations

import json
import os
import re
import time

from PySide6.QtWidgets import QProgressBar

from gui.theme import tok

# The trailing "[...]" rate block is captured optionally so non-timed bars still parse.
TQDM_RE = re.compile(
    r"^(?P<label>.*?):?\s*(?P<pct>\d+)%\|[^|]*\|\s*(?P<cur>\d+)/(?P<tot>\d+)"
    r"(?:[^\[]*\[[^\]]*?(?P<rate>[\d.]+)(?P<unit>it/s|s/it)[^\]]*\])?"
)


def make_progress_bar() -> QProgressBar:
    """Build a QProgressBar styled to match the rest of the GUI.

    Returns a hidden bar — the tracker shows it on the first parsed update
    and ``TqdmProgressTracker.reset`` hides it again at run-end.
    """
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(True)
    bar.setFormat("")
    bar.setVisible(False)
    bar.setStyleSheet(
        f"QProgressBar {{ border: 1px solid {tok('border_dim')}; border-radius: 3px;"
        " text-align: center; padding: 1px; font-size: 11px; }"
        "QProgressBar::chunk { background: #27ae60; }"
    )
    return bar


class TqdmProgressTracker:
    """Parses tqdm output lines and drives a QProgressBar.

    Holds an anchor (timestamp + step) seeded from the *first completed*
    step of each new bar, so reported s/step doesn't include warm-up
    overhead (model load, compile, dataset scan).
    """

    def __init__(self, bar: QProgressBar) -> None:
        self._bar = bar
        # (monotonic_anchor_time, anchor_step, label, total)
        self._anchor: tuple[float, int, str, int] | None = None

    def reset(self) -> None:
        """Zero the bar, hide it, drop the rate anchor."""
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFormat("")
        self._bar.setVisible(False)
        self._anchor = None

    def mark_starting(self, label: str) -> None:
        """Show the bar in indeterminate "busy" mode (Qt animates range 0-0).

        Used between subprocess launch and the first tqdm line so Windows
        doesn't flag the GUI as "Not Responding" during the multi-second
        torch/accelerate import inside the child.
        """
        self._bar.setRange(0, 0)  # indeterminate / marquee
        self._bar.setFormat(label)
        self._bar.setVisible(True)
        self._anchor = None

    def feed(self, line: str) -> bool:
        """Try to parse *line* as a tqdm update. Returns True if matched.

        The caller passes non-matching lines to its log widget instead.
        """
        m = TQDM_RE.search(line)
        if not m:
            return False
        cur = int(m.group("cur"))
        tot = int(m.group("tot"))
        label = m.group("label").strip() or "progress"
        rate_str = self._update_rate(label, cur, tot)
        if tot > 0:
            # setRange(0, tot) leaves indeterminate mode, clearing the marquee animation.
            self._bar.setRange(0, tot)
            self._bar.setValue(cur)
            self._bar.setFormat(f"{label}: {cur}/{tot} (%p%){rate_str}")
            if not self._bar.isVisible():
                self._bar.setVisible(True)
        return True

    def _update_rate(self, label: str, cur: int, tot: int) -> str:
        now = time.monotonic()
        anchor = self._anchor
        # New bar (label/total changed, or progress rewound) → drop anchor.
        if anchor is None or anchor[2] != label or anchor[3] != tot or cur < anchor[1]:
            if cur >= 1:
                self._anchor = (now, cur, label, tot)
            else:
                self._anchor = None
            return ""
        anchor_time, anchor_step, _, _ = anchor
        steps = cur - anchor_step
        if steps <= 0:
            return ""
        spi = (now - anchor_time) / steps
        remaining = tot - cur
        if remaining <= 0:
            return f" — {spi:.2f}s/step"
        return f" — {spi:.2f}s/step — ETA {_format_duration(remaining * spi)}"


class JsonlProgressReader:
    """Drives a QProgressBar from a training ``progress.jsonl`` event stream.

    The trainer writes structured ``run_start`` / ``step`` / ``val`` /
    ``run_end`` events (``library/training/progress.py``) next to the
    checkpoint, and this reader tails that file. The caller keeps the tqdm
    :class:`TqdmProgressTracker` as a fallback and only hands the bar over once
    ``active`` flips True (first event seen). When the file never appears
    (progress disabled) the reader stays inert and tqdm drives the bar.

    Usage::

        self._jsonl_reader = JsonlProgressReader(self.progress)
        self._jsonl_reader.watch(progress_path)   # at launch
        # on a timer:
        self._jsonl_reader.poll()
        # in the stdout handler, suppress tqdm bar updates while active.
    """

    def __init__(self, bar: QProgressBar, *, on_run_start=None) -> None:
        self._bar = bar
        self._path: str | None = None
        self._pos = 0
        self._total_steps = 0
        self._active = False
        self._on_run_start = on_run_start
        # (anchor_ts, anchor_step) seeded from the first step event. Uses the
        # event's embedded ``ts`` (seconds since run start), not wall-clock —
        # see ``_rate`` for why that matters on GUI re-attach.
        self._anchor: tuple[float, int] | None = None

    @property
    def active(self) -> bool:
        return self._active

    def watch(self, path: str | None) -> None:
        """Point the reader at *path* and reset state. Pass ``None`` to disable
        (e.g. for test / preprocess runs that emit no progress.jsonl)."""
        self._path = path
        self._pos = 0
        self._total_steps = 0
        self._active = False
        self._anchor = None

    def reset(self) -> None:
        self.watch(None)

    def poll(self) -> None:
        """Read any complete new lines and update the bar. No-op while the
        file is absent (the trainer hasn't emitted ``run_start`` yet)."""
        if not self._path or not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                f.seek(self._pos)
                chunk = f.read()
                self._pos = f.tell()
        except OSError:
            return
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                # partial line written between our reads — rewind and retry next poll
                self._pos -= len(line.encode("utf-8"))
                return
            self._consume(ev)

    def _consume(self, ev: dict) -> None:
        kind = ev.get("ev")
        if kind == "run_start":
            self._active = True
            self._total_steps = int(ev.get("total_steps") or 0)
            self._anchor = None
            self._bar.setRange(0, self._total_steps or 0)
            self._bar.setValue(0)
            self._bar.setFormat("starting…")
            self._bar.setVisible(True)
            if self._on_run_start is not None:
                try:
                    self._on_run_start(ev)
                except Exception:
                    pass
        elif kind == "step":
            self._active = True
            self._update_bar(int(ev.get("global_step") or 0), ev.get("ts"))
        elif kind == "val":
            # CMMD val pass — keep the bar where it is, annotate if we have it.
            cmmd = ev.get("cmmd")
            if cmmd is not None and self._bar.isVisible():
                self._bar.setFormat(self._bar.format() + f" — CMMD {cmmd:.4f}")

    def _update_bar(self, cur: int, ts: float | None = None) -> None:
        tot = self._total_steps
        rate = self._rate(cur, ts)
        if tot > 0:
            self._bar.setRange(0, tot)
            self._bar.setValue(cur)
            self._bar.setFormat(f"step {cur}/{tot} (%p%){rate}")
        else:
            self._bar.setRange(0, 0)  # total unknown → indeterminate
            self._bar.setFormat(f"step {cur}{rate}")
        if not self._bar.isVisible():
            self._bar.setVisible(True)

    def _rate(self, cur: int, ts: float | None) -> str:
        """Compute s/step, anchoring on the event's embedded ``ts``.

        ``ts`` is seconds-since-run-start written by the trainer. Using it
        (rather than ``time.monotonic()``) makes the rate re-attach-safe: when
        the GUI reopens mid-run it re-reads the whole progress.jsonl in one
        burst, so wall-clock deltas between consecutive events collapse to ~0
        and s/step would read near zero (the "all steps done instantly" bug).
        The embedded ``ts`` carries the real step spacing no matter when we
        read the file. Falls back to wall-clock only if ``ts`` is absent.
        """
        clock = ts if ts is not None else time.monotonic()
        if self._anchor is None or cur < self._anchor[1]:
            self._anchor = (clock, cur)
            return ""
        anchor_time, anchor_step = self._anchor
        steps = cur - anchor_step
        if steps <= 0:
            return ""
        spi = (clock - anchor_time) / steps
        remaining = self._total_steps - cur
        if self._total_steps <= 0 or remaining <= 0:
            return f" — {spi:.2f}s/step"
        return f" — {spi:.2f}s/step — ETA {_format_duration(remaining * spi)}"


def _format_duration(seconds: float) -> str:
    """Render a duration as ``M:SS`` (under an hour) or ``H:MM:SS``.

    Matches tqdm's own remaining-time style so the ETA reads naturally next
    to the s/step rate.
    """
    s = max(0, int(round(seconds)))
    if s < 3600:
        return f"{s // 60}:{s % 60:02d}"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"
