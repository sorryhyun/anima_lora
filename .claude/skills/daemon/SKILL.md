---
name: daemon
description: Submit, monitor, and manage GPU jobs through the anima daemon (make daemon-*, make gen, make run-status, MCP bridge, discovery). Load before launching any GPU process as an agent, checking training-run progress, batch-generating images, or wiring a new daemon client.
---

# Daemon job queue & GPU work

Local FIFO job queue (`anima_daemon/`), auto-starts on first submit. Full HTTP contract:
`anima_daemon/README.md`.

**Agent-launched GPU work must go through the daemon.** A GPU process started from a
Claude Code background Bash gets killed by the harness sandbox layer after ~1 min (silent
SIGKILL — no OS/OOM trace, no traceback; observed 2026-07-25). Daemon jobs also queue
behind a live train run instead of OOM-colliding, and survive the terminal.

## Targets

`make daemon | daemon-run ARGS="<script.py> …" | daemon-wait [JOB=<id>] | daemon-attach [JOB=<id>] | daemon-jobs | daemon-log [JOB=<id>] | daemon-pause [JOB=<id>] | daemon-resume [JOB=<id>] | daemon-kill | daemon-terminate | daemon-prune`

- Front door: **`make daemon-run ARGS="<script.py> [flags]"`** — attach-by-default, exits
  with the job's code; `--queue` detaches, `--inline` bypasses the daemon;
  `--stall-timeout S` where `0` = off. daemon-run's own `--label`/`--stall-timeout` go
  **before** the script path — after it every token reaches the child untouched (bench
  scripts take `--label` themselves), and `--` passes everything after it verbatim.
- **`make daemon-wait [JOB=<id>]`** blocks to terminal and prints the record + result
  envelope, exiting with the job's code (`DaemonClient.wait()` programmatically) — don't
  hand-roll an HTTP poll loop.
- `daemon-pause` tree-freezes the running job (SIGSTOP — VRAM held, SM idle, resume
  instant; the queue does NOT advance past it; refuses `accelerate launch` runs).
- Append `--queue` to any train/distill target to enqueue instead of running inline
  (`make lora --queue`, `make turbo --queue`). GUI Train button, ComfyUI trainer node,
  and preprocessing all submit here.
- Long-quiet phases: prefer `--stall-timeout` over a heartbeat, else
  `bench/_common.py::start_heartbeat()` (the watchdog also spares a quiet-but-CPU-burning
  tree).

## Job environment

A job's env is **daemon-env ← `captured_env` ← `extra_env`**. `captured_env` is the
submitter's `ANIMA_*` / `CUDA_*` / `HF_*` / `PYTORCH_*` / `TORCH_*` / `NCCL_*` at submit
time (recorded in `job.json`); everything else — and every whitelisted var the submit
shell does **not** set — comes from the shell that booted the daemon, possibly days ago
(a stale-code respawn re-boots it from whichever shell submitted next).

- An unset var cannot override: if the daemon booted with
  `ANIMA_VOCAB_PACK=<preview pack>`, a later submit without the var trains on the
  preview pack. Set every env lever the job depends on in the submit shell, and confirm
  it from the job itself: `captured_env` in `job.json`, the value in
  `/proc/<pid>/environ`, or the line the job logs (the vocab pack logs its sha).
- **`make daemon-terminate` is the reset, and it is cheap when nothing is running** —
  all state is on disk and the next submit boots a fresh daemon from the current shell.
  Use it whenever `make daemon-jobs ARGS="--state queued,running,paused"` reads `0 of M`
  and the daemon's env is in doubt (an old session booted it, an env lever changed, a
  run is about to take hours). It kills the active job and discards the queue, so check
  that line first.

## Reading the queue

| question | command | what comes back |
|---|---|---|
| Is a job running / is the queue busy? | `make daemon-jobs ARGS="--state queued,running,paused"` | one line per unfinished job, then `N of M jobs` — a bare `0 of M` **is** the "nothing running" answer (exit 1 = daemon down) |
| How far along is the current run? | `make run-status` | `step N/total`, it/s, ETA, last losses, last ckpt (§ Run status below) |
| Did job `<id>` finish, and with what exit code? | `make daemon-jobs ARGS="--all" \| grep <id>` | one line: when · id · state · `rc=` · duration · target · first error line |
| …and block until it does? | `make daemon-wait JOB=<id>` | exits with the job's own code (record + envelope on stdout) |
| What argv did job `<id>` run? | `python -c "import json;print(' '.join(json.load(open('output/daemon/jobs/<id>/job.json'))['argv']))"` | the child argv on one line. **Command jobs only** — a train job persists `method`/`preset`/`overrides`/`extra` and builds its launch cmd at spawn, so its `argv` is empty |
| One job's full record + its bench `result.json`? | `python -m anima_daemon status <id>` | the whole record, envelope inlined under `result`; reads the on-disk `job.json` when the daemon is down |
| Is the daemon up, on which port, running stale code? | `python -m anima_daemon status` | `up`, resolved `base_url`, `stale_code`, `paused`, `active_job` (exit 1 when down) |

`daemon-jobs`, `daemon-log`, `run-status` and the `job.json` read all work with the
daemon down.

`daemon-jobs` prints **oldest first** (`| tail -5` = the five most recent), capped at 15;
filter with `ARGS="--running|--failed|--done|--state s[,s]|--limit N|--all"`. Jobs do not
always start in submit order (a chained job waits on its parent), so ask for pending work
by state rather than trusting the newest-15 slice to contain it. `make daemon-log
[JOB=<id>]` dumps a job's stdout from disk (`ARGS="-n 200"`; `-n 0` = all);
`daemon-attach` follows a *live* stream only, so it has nothing for a finished job.

## Discovery & agent surface

- Discovery is pidfile-based: `output/daemon/daemon.json` / `~/.anima/daemon.json` →
  `{port, root}`. **Never hardcode 8765** — the port falls back to ephemeral on collision.
- `python -m anima_daemon submit|wait|status` is the stdlib-only equivalent of the `make`
  targets, for callers that can't import `tasks.py`.
- The daemon self-describes at `GET /` (README) and `GET /tools` (JSON-Schema manifest).
  `anima_daemon/mcp.py` is a stdio MCP bridge over the same surface — register the script
  path as the MCP command; it discovers the daemon itself.
- Job dirs are retention-bounded. `make daemon-prune` is the manual sweep — dry-run
  unless `ARGS="--apply"`. Rules and knobs: `anima_daemon/README.md` § Retention.

## Batch generation: `make gen`

Daemon-routed batch generation — same argv + env levers as `make test`, submitted as a
GPU command job (attach-by-default; `--queue` detaches, `--inline` bypasses). Lands a
`gen_manifest.json` in the job record: `write_gen_manifest` drops a `result_path.json`
pointer when the daemon exports `ANIMA_DAEMON_JOB_DIR` (a plain `python inference.py` is
unaffected).

## Run status: `make run-status`

`step N/total`, it/s, ETA, last losses, last ckpt, and `RUNNING`/`OK`/`ERROR`/`DEAD` (no
`run_end` + dead pid), digested from the run's `progress.jsonl`
(`library/training/progress.py::read_status` — importable; `scripts/run_status.py` is the
CLI). Covers train.py methods **and** `make turbo`.

**Both launch paths are scanned**: an inline run's `output/logs/<name>.progress.jsonl`
*and* a daemon job's `output/daemon/jobs/<id>/progress.jsonl` (the daemon overrides
`--progress_jsonl` with a per-job path, so the run dir under `output/logs/` holds the
snapshot + TB events but no stream). Defaults to the newest stream from either;
`RUN=<output_name|job id|path>` selects — a daemon stream's filename is bare, so a run
name there is matched against the `run_start` event inside it, and the header prints
`(job <id>)`. `ARGS="--list"` for all, `ARGS="--json"` for the dict, `ARGS="--jobs-dir
''"` to skip the daemon dirs.

For every scalar instead: `make export-logs RUN=output/logs/<run> SUMMARY=1` prints
max-step + last value per tag (raw payload `{"run", "tags": {tag: [[step, wall_time,
value], …]}}` — value is `row[2]`).
