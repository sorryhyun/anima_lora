"""Local training-job daemon.

A single localhost process: a FIFO serial job queue + worker thread that spawns
``accelerate launch … train.py`` subprocesses (detached, so a console ctrl-C
can't reach them) and follows each run by tailing its ``progress.jsonl``.
Exposes a small stdlib HTTP API (no framework, no auth, ``127.0.0.1`` only)
consumed by the CLI (``make daemon*``), the ComfyUI trainer node, and the MCP
bridge (``mcp.py``).

    python -m anima_daemon [port]                  # run the daemon (normally detached)
    python -m anima_daemon submit -- <argv…>       # enqueue a command job
    python -m anima_daemon wait <job_id>           # block until it's terminal
    python -m anima_daemon status [job_id]         # one-shot JSON status

Public surface lives in submodules: ``client.DaemonClient`` (aliased ``Client``)
/ ``ensure_daemon`` for callers, ``manager.JobManager`` + ``server.serve`` for
the process itself, ``cli`` for the submit/wait/status verbs above.
"""
