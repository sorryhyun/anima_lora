"""Daemon process entry point: ``python -m anima_daemon [port]``.

The long-lived, console-detached process ``make daemon`` spawns: takes the
single-daemon lock, binds the preferred port (falling back to an ephemeral
one if a stranger holds it), reconciles ``jobs/``, then serves until
``POST /shutdown``. See ``anima_daemon/README.md``.

A first argument naming a client verb (``submit``/``wait``/``status``, see
``cli.py``) dispatches to that verb and exits instead — the CLI front door
for talking *to* a daemon. Anything else is parsed as the port.
"""

from __future__ import annotations

import logging
import os
import sys

from . import cli, config, proc
from .manager import JobManager
from .server import serve_with_fallback


def _setup_logging() -> None:
    config.ensure_state_dirs()
    # ANIMA_DEBUG (GUI "Debug mode") turns on DEBUG logging so daemon.log
    # captures the launch/queue decisions a bug report needs.
    level = logging.DEBUG if os.environ.get("ANIMA_DEBUG") else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],  # stderr → daemon.log file
    )


def main() -> int:
    # Client verbs run before logging setup: they must print only JSON to
    # stdout, and don't start or drive a daemon.
    if len(sys.argv) > 1 and sys.argv[1] in cli.VERBS:
        return cli.main(sys.argv[1:])

    _setup_logging()
    log = logging.getLogger("anima.daemon")

    port = config.DEFAULT_PORT
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    # Refuse to start a second daemon over a live one.
    existing = proc.daemon_alive(config.PIDFILE)
    if existing is not None:
        log.error(
            "daemon already running (pid %s, port %s); refusing to start",
            existing.get("pid"),
            existing.get("port"),
        )
        return 3

    manager = JobManager()
    manager.start()

    # Boot-time source fingerprint for the pidfile + /health (see
    # config.source_fingerprint). Computed once: /health must echo the BOOT
    # value — re-hashing on-disk files would never report stale code.
    fingerprint = config.source_fingerprint()

    try:
        server = serve_with_fallback(manager, port=port, fingerprint=fingerprint)
    except OSError as exc:
        log.error("could not bind 127.0.0.1:%s (%s); another anima daemon?", port, exc)
        manager.shutdown(kill_jobs=False)
        return 3

    proc.write_pidfile(
        config.PIDFILE,
        pid=os.getpid(),
        port=server.server_address[1],
        root=config.ROOT,
        fingerprint=fingerprint,
    )
    # Mirror to a stable per-user path so a standalone install can discover us
    # without knowing the repo location. Best-effort.
    try:
        proc.write_pidfile(
            config.global_pidfile(),
            pid=os.getpid(),
            port=server.server_address[1],
            root=config.ROOT,
            fingerprint=fingerprint,
        )
    except OSError as exc:
        log.warning("could not write global pidfile mirror (%s)", exc)
    log.info(
        "anima training daemon up on http://%s:%s (pid %s)",
        config.HOST,
        server.server_address[1],
        os.getpid(),
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        manager.shutdown(kill_jobs=False)
    finally:
        for pid_path in (config.PIDFILE, config.global_pidfile()):
            try:
                pid_path.unlink()
            except OSError:
                pass
        log.info("daemon stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
