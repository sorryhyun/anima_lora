"""Local training-job daemon.

A single localhost process: a FIFO serial job queue + worker thread that spawns
``train.py`` runs and plain ``python <argv>`` command jobs as detached
subprocesses and follows train runs by tailing their ``progress.jsonl``.
Exposes a stdlib HTTP API (no auth, ``127.0.0.1`` only) consumed by the CLI
(``make daemon*``), the ComfyUI trainer node, and the MCP bridge (``mcp.py``).

    python -m anima_daemon [port]                  # run the daemon (normally detached)
    python -m anima_daemon submit -- <argv…>       # enqueue a command job
    python -m anima_daemon wait <job_id>           # block until it's terminal
    python -m anima_daemon status [job_id]         # one-shot JSON status

Public surface lives in submodules: ``client.DaemonClient`` (aliased ``Client``)
/ ``ensure_daemon`` for callers, ``manager.JobManager`` + ``server.serve`` for
the process itself, ``cli`` for the submit/wait/status verbs above.
"""
