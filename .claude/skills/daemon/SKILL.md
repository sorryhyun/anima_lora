---
name: daemon
description: Submit, monitor, and manage GPU jobs through the anima daemon (make daemon-*, make gen, make run-status, MCP bridge, discovery). Load before launching any GPU process as an agent, checking training-run progress, batch-generating images, or wiring a new daemon client.
---

# Daemon job queue & GPU work

Local FIFO job queue (`anima_daemon/`), auto-starts on first submit. Reference (HTTP API,
CLI, env vars, on-disk layout, pause semantics, MCP bridge): `anima_daemon/README.md`.

**Agent-launched GPU work must go through the daemon.** A GPU process started from a
Claude Code background Bash gets killed by the harness sandbox layer after ~1 min (silent
SIGKILL — no OS/OOM trace, no traceback; observed 2026-07-25). Daemon jobs also queue
behind a live train run instead of OOM-colliding, and survive the terminal.

**Agent loop: submit detached, wait in the background.**

1. Submit with `--queue` (`make daemon-run ARGS="--queue [--stall-timeout S] <script.py> …"`,
   `make lora ARGS="--queue …"`). It prints the job id and returns. Without it the submit
   attaches and blocks a foreground Bash until the job ends.
2. Wait with one Bash `run_in_background: true` running `make daemon-wait JOB=<id>`. It
   notifies once when the job exits; read the result then (`make daemon-log JOB=<id>`,
   `make run-status`). Jobs queued back to back need only a wait on the last one whose
   result you act on.

Details and limits: § Waiting on a job.

## Targets

`make daemon | daemon-run ARGS="<script.py> …" | daemon-wait [JOB=<id>] | daemon-attach [JOB=<id>] | daemon-status [JOB=<id>] | daemon-jobs | daemon-log [JOB=<id>] | daemon-pause [JOB=<id>] | daemon-resume [JOB=<id>] | daemon-kill [JOB=<id>] | daemon-terminate | daemon-prune`

- **`make daemon-run ARGS="<script.py> [flags]"`** attaches by default and exits with the
  job's code (for a user at a terminal; agents pass `--queue`); `--queue` detaches, `--inline` bypasses the daemon; `--stall-timeout S`,
  `0` = off. daemon-run's own `--label`/`--stall-timeout` go **before** the script path:
  after it every token reaches the child (bench scripts take `--label` themselves), and
  `--` passes everything after it verbatim.
- **`make daemon-wait [JOB=<id>]`** blocks to terminal, prints the record + result
  envelope and exits with the job's code. Do not hand-roll an HTTP poll loop.
- `daemon-pause` freezes the running job (SIGSTOP: VRAM held, the queue does not advance,
  resume instant; refuses `accelerate launch` runs). `daemon-pause RELEASE=1` (train.py
  jobs) has the trainer save and exit, freeing the GPU and advancing the queue;
  `daemon-resume` relaunches it (reload + recompile).
- Train and distill targets take the same `--queue` / `--inline` (`make lora --queue`,
  `make turbo --queue`).
- Long-quiet phases: command jobs are killed after 120 s of silence by default. Prefer
  `--stall-timeout` over a heartbeat, else `bench/_common.py::start_heartbeat()` (the
  watchdog also spares a quiet-but-CPU-burning tree, up to 8× the budget).

## Waiting on a job (agent sessions)

The background `daemon-wait` of the agent loop above. Send step / ETA updates only when
the user asks for them.

- A background command has no time limit in an interactive local session, and
  `daemon-wait` is CPU-only, so the GPU-process kill does not apply to it.
- Linux reaps background shells under critical memory pressure once the session has
  idled 30 min (`CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1` turns it off): a waiter
  can vanish during a RAM-heavy run. `daemon-jobs` still has the answer.
- Not **Monitor**: its deadline is capped at 30 min and it kills the command at expiry.
  `ARGS="--timeout S"` makes `daemon-wait` exit 124 at S seconds with a JSON snapshot
  (state, last progress event, its staleness).

## Job environment

A job's env is **daemon-env ← `captured_env` ← `extra_env`**. `captured_env` holds the
whitelisted prefixes (README § `POST /jobs`) at submit time, recorded in `job.json`;
everything else, and every whitelisted var the submit shell does **not** set, comes from
the shell that booted the daemon, possibly days ago (a stale-code respawn re-boots it from
whichever shell submitted next).

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
| Is a job running / is the queue busy? | `make daemon-status` (health line + unfinished jobs), or `make daemon-jobs ARGS="--state queued,running,paused"` | one line per unfinished job; `0 of M` = nothing running (exit 1 = daemon down) |
| How far along is the current run? | `make run-status` | `step N/total`, it/s, ETA, last losses, last ckpt (§ Run status below) |
| Did job `<id>` finish, and with what exit code? | `make daemon-jobs ARGS="--all" \| grep <id>` | one line with state, `rc=` and the first error line |
| …and block until it does? | `make daemon-wait JOB=<id>` | exits with the job's own code |
| What argv did job `<id>` run? | `python -c "import json;print(' '.join(json.load(open('output/daemon/jobs/<id>/job.json'))['argv']))"` | the child argv on one line. **Command jobs only**: a train job's `argv` is empty |
| One job's full record + its bench `result.json`? | `python -m anima_daemon status <id>` | the whole record, envelope inlined under `result` |
| Is the daemon up, on which port, running stale code? | `python -m anima_daemon status` | `up`, resolved `base_url`, `stale_code`, `paused`, `active_job` (exit 1 when down) |

`daemon-jobs`, `daemon-log`, `run-status` and the `job.json` read all work with the
daemon down.

`daemon-jobs` prints **oldest first** (`| tail -5` = the five most recent), capped at 15;
filter with `ARGS="--running|--failed|--done|--state s[,s]|--limit N|--all"`. A chained
job waits on its parent, so ask for pending work by state rather than trusting the
newest-15 slice to contain it. `make daemon-log [JOB=<id>]` dumps a job's stdout from disk
(`ARGS="-n 200"`; `-n 0` = all); `daemon-attach` follows a live stream only.

## Discovery

- `output/daemon/daemon.json` / `~/.anima/daemon.json` → `{port, root}`. **Never hardcode
  8765**: the port falls back to ephemeral on collision.
- Callers that can't import `tasks.py` use `python -m anima_daemon submit|wait|status`.
  MCP clients use `anima_daemon/mcp.py` (README § MCP bridge).
- `make daemon-prune` sweeps old job dirs, dry-run unless `ARGS="--apply"` (README
  § Retention).

## Batch generation: `make gen`

`make test`'s argv and env levers as a GPU command job, with the same `--queue` /
`--inline`. The job's `result_path` points at its `gen_manifest.json`.

## Run status: `make run-status`

`step N/total`, it/s, ETA, last losses, last ckpt, and `RUNNING`/`OK`/`ERROR`/`DEAD` (no
`run_end` + dead pid), digested from the run's `progress.jsonl`
(`library/runtime/progress.py::read_status` — importable; `scripts/run_status.py` is the
CLI). Covers train.py methods, `make turbo` **and** moss train (`scripts/moss/train.py`); a bespoke loop that writes no
`progress.jsonl` (e.g. `project/cjk_anima_reseed/run.py`) shows nothing here — read the
step off `make daemon-log JOB=<id>`.

It scans an inline run's `output/logs/<name>.progress.jsonl` and a daemon job's
`output/daemon/jobs/<id>/progress.jsonl` (a daemon run's dir under `output/logs/` holds the
snapshot + TB events but no stream). It defaults to the newest stream from either, or
names the daemon's running job while that has no stream yet (a train loop still loading).
`RUN=<output_name|job id|path>` selects. `ARGS="--list"` for all, `ARGS="--json"` for the
dict, `ARGS="--jobs-dir ''"` to skip the daemon dirs.

For every scalar instead: `make export-logs RUN=output/logs/<run> SUMMARY=1` prints
max-step + last value per tag (raw payload `{"run", "tags": {tag: [[step, wall_time,
value], …]}}` — value is `row[2]`).
