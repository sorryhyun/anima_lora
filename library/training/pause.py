"""Cooperative "release" pause for a daemon-run training job.

The daemon's plain pause is a SIGSTOP tree-freeze: VRAM stays allocated and the
queue cannot advance. ``pause(release_model=True)`` instead asks train.py to
stop *itself* at the next optimizer step: it writes a resumable state (the same
``<output_name>-checkpoint-state`` dir auto-resume uses), acknowledges, and
exits with a ``run_end`` of status ``paused``. The GPU is fully released and the
queue moves on; ``resume`` re-enqueues the job with ``--resume <state_dir>``.

Protocol (all files live in the daemon job dir, ``ANIMA_DAEMON_JOB_DIR``):

- daemon writes ``pause.request`` (JSON, ``{"release_model": true}``)
- train.py polls it at each optimizer step (main process, throttled), saves,
  deletes the request, writes ``pause.ack.json`` and raises
  :class:`TrainingPaused`
- ``run_scope`` maps the exception to ``run_end status="paused"`` carrying
  ``state_dir``; the daemon parks the job as ``paused`` with no process.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

PAUSE_REQUEST_NAME = "pause.request"
PAUSE_ACK_NAME = "pause.ack.json"
JOB_DIR_ENV = "ANIMA_DAEMON_JOB_DIR"


class TrainingPaused(Exception):
    """Raised inside the training loop once the resumable state is on disk."""

    def __init__(self, state_dir: str, global_step: int, epoch: int) -> None:
        super().__init__(f"training paused at step {global_step}: {state_dir}")
        self.state_dir = state_dir
        self.global_step = global_step
        self.epoch = epoch


def request_path(job_dir: str) -> str:
    return os.path.join(job_dir, PAUSE_REQUEST_NAME)


def ack_path(job_dir: str) -> str:
    return os.path.join(job_dir, PAUSE_ACK_NAME)


class PauseWatcher:
    """Polls the job dir for a release-pause request.

    ``poll()`` is cheap enough to call every optimizer step: one ``stat`` at
    most every ``min_interval`` seconds on the main process, then the verdict is
    broadcast so every rank reaches the save collective together.
    """

    def __init__(self, job_dir: str, *, min_interval: float = 2.0) -> None:
        self.job_dir = job_dir
        self.min_interval = min_interval
        self._last_check = 0.0

    @classmethod
    def from_env(cls) -> Optional["PauseWatcher"]:
        job_dir = os.environ.get(JOB_DIR_ENV)
        if not job_dir:
            return None
        return cls(job_dir)

    def _requested_local(self) -> bool:
        now = time.monotonic()
        if now - self._last_check < self.min_interval:
            return False
        self._last_check = now
        return os.path.exists(request_path(self.job_dir))

    def poll(self, accelerator=None) -> bool:
        """True once a request is present. With more than one process the main
        process decides and the result is broadcast."""
        if accelerator is None or accelerator.num_processes == 1:
            return self._requested_local()
        flag = [self._requested_local() if accelerator.is_main_process else False]
        from accelerate.utils import broadcast_object_list

        flag = broadcast_object_list(flag, from_process=0)
        return bool(flag[0])

    def acknowledge(self, state_dir: str, *, global_step: int, epoch: int) -> None:
        """Consume the request and record where the state went (main only)."""
        try:
            os.remove(request_path(self.job_dir))
        except OSError:
            pass
        payload = {
            "state_dir": state_dir,
            "global_step": int(global_step),
            "epoch": int(epoch),
            "ts": time.time(),
        }
        try:
            with open(ack_path(self.job_dir), "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except OSError as exc:
            logger.warning("could not write %s: %s", ack_path(self.job_dir), exc)
