"""Local training-job daemon.

A single localhost process: a FIFO serial job queue + worker thread that spawns
``train.py`` runs and plain ``python <argv>`` command jobs as detached
subprocesses and follows train runs by tailing their ``progress.jsonl``.
Exposes a stdlib HTTP API (no auth, ``127.0.0.1`` only) consumed by the CLI
(``make daemon*``), the ComfyUI trainer node, and the MCP bridge (``mcp.py``).
Reference: ``anima_daemon/README.md``.

Public surface lives in submodules: ``client.DaemonClient`` (aliased ``Client``)
/ ``ensure_daemon`` for callers, ``manager.JobManager`` + ``server.serve`` for
the process itself (with ``launch`` / ``gpu_guard`` / ``watchdog`` / ``record``
/ ``tools``), ``cli`` for the ``python -m anima_daemon`` verbs.
"""
