---
name: daemon
description: Submit, monitor, and manage GPU jobs through the anima daemon (make daemon-*, make gen, make run-status, MCP bridge, discovery). Load before launching any GPU process as an agent, checking training-run progress, batch-generating images, or wiring a new daemon client.
---

# Daemon job queue & GPU work

Local FIFO job queue (`anima_daemon/`), auto-starts on first submit. Full HTTP contract: `anima_daemon/README.md`.

## Why agents must use it

**Agent-launched GPU work must go through the daemon.** GPU processes started from a Claude Code background Bash get killed by the harness sandbox layer after ~1 min (silent SIGKILL — no OS/OOM trace, no traceback; observed 2026-07-25). Daemon jobs also queue behind a live train run instead of OOM-colliding, and survive the terminal.

## Targets

`make daemon | daemon-run ARGS="<script.py> …" | daemon-wait [JOB=<id>] | daemon-attach [JOB=<id>] | daemon-pause [JOB=<id>] | daemon-resume [JOB=<id>] | daemon-kill | daemon-terminate | daemon-prune`

- Front door: **`make daemon-run ARGS="<script.py> [flags]"`** — attach-by-default, exits with the job's code; `--queue` detaches, `--inline` bypasses the daemon; `--stall-timeout S` where `0` = off. daemon-run's own `--label`/`--stall-timeout` go **before** the script path — after it every token reaches the child untouched (bench scripts take `--label` themselves), and `-- ` passes everything after it verbatim (run-mode flags included). No Python snippet needed; `python -m anima_daemon submit|wait|status` is the same thing without `tasks.py`.
- **`make daemon-wait [JOB=<id>]`** blocks to terminal and prints the record + result envelope, exiting with the job's code (`DaemonClient.wait()` programmatically) — don't hand-roll an HTTP poll loop.
- `daemon-pause` tree-freezes the running job (SIGSTOP — VRAM held, SM idle, resume instant; the queue does NOT advance past it; refuses `accelerate launch` runs).
- Append `--queue` to any train/distill target to enqueue instead of running inline (`make lora --queue`, `make turbo --queue`). GUI Train button, ComfyUI trainer node, and preprocessing all submit here.
- Long-quiet phases: prefer `--stall-timeout` over a heartbeat, else `bench/_common.py::start_heartbeat()` (the watchdog also spares a quiet-but-CPU-burning tree).

## Retention

Job dirs are retention-bounded: at boot, before `load_all()`, the daemon prunes terminal jobs older than 30d (keeping the newest 200). `make daemon-prune` is the manual sweep — dry-run unless `ARGS="--apply"`. Knobs: `ANIMA_DAEMON_JOB_RETENTION_DAYS` / `ANIMA_DAEMON_JOB_KEEP`.

## Discovery & agent surface

- Discovery is pidfile-based: `output/daemon/daemon.json` / `~/.anima/daemon.json` → `{port, root}`. **Never hardcode 8765** — the port falls back to ephemeral on collision.
- `make daemon-status` prints one JSON object (health + resolved `base_url` + compact job summaries, newest-first and capped, each with a derived `target` + `jobs_total`/`jobs_shown`). Filter via `ARGS="--running|--failed|--done|--state s|--limit N|--all"`; `--full` for raw records; `--job <id>`/`JOB=<id>` for one full record with its bench `result.json` inlined. Passive; exit 1 when down.
- The daemon self-describes at `GET /` (README) and `GET /tools` (JSON-Schema manifest). `anima_daemon/mcp.py` is a stdio MCP bridge over the same surface — register the script path as the MCP command; it discovers the daemon itself.

## Batch generation: `make gen`

The daemon-routed batch-generation front door — same argv + env levers as `make test`, but submitted as a GPU command job (attach-by-default; `--queue` detaches, `--inline` bypasses). Lands a `gen_manifest.json` in the job record: `inference.py`'s `write_gen_manifest` drops a `result_path.json` pointer when the daemon exports `ANIMA_DAEMON_JOB_DIR` (the proposal Phase 1a result-lift; a plain `python inference.py` is unaffected). Use it for eval grids / seed sweeps / ad-hoc renders; interactive single images stay on `make test` or the resident inference server.

## Run status: `make run-status`

"Where is this run at?" — `step N/total`, it/s, ETA, last losses, last ckpt, and `RUNNING`/`OK`/`ERROR`/`DEAD` (no `run_end` + dead pid), digested from the run's `progress.jsonl` (`library/training/progress.py::read_status` — importable; `scripts/run_status.py` is the CLI). Covers train.py methods **and** `make turbo`.

**Both launch paths are scanned**: an inline run's `output/logs/<name>.progress.jsonl` *and* a daemon job's `output/daemon/jobs/<id>/progress.jsonl` (the daemon overrides `--progress_jsonl` with a per-job path, so the run dir under `output/logs/` holds the snapshot + TB events but no stream). Defaults to the newest stream from either; `RUN=<output_name|job id|path>` selects — a daemon stream's filename is bare, so a run name there is matched against the `run_start` event inside it, and the header prints `(job <id>)` so the follow-up needs no lookup. `ARGS="--list"` for all, `ARGS="--json"` for the dict, `ARGS="--jobs-dir ''"` to skip the daemon dirs.

Don't export the TB events file to answer this; if you do need every scalar, `make export-logs RUN=output/logs/<run> SUMMARY=1` prints max-step + last value per tag (the raw payload is `{"run", "tags": {tag: [[step, wall_time, value], …]}}` — value is `row[2]`).
