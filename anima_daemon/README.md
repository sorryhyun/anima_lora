# Anima training daemon — REST + programmatic interface

A single localhost process that owns a **serial job queue**: submit a training
run (or a plain command), one job runs at a time, state survives restarts, and
observers poll or stream rather than holding the run open. The GUI Train button,
the ComfyUI trainer node, and `make … --queue` all submit here; this doc
describes the same surface for **direct use** — a script, an MCP server, or an
agent driving training without going through `make`.

Design in one line: `http://127.0.0.1:8765`, JSON in / JSON out, no auth, no
remote (localhost only, by design — see `config.py`). All state is on disk under
`output/daemon/`, so anything that can read files can observe a run even with the
HTTP port down.

**Self-describing.** The daemon serves its own docs: `GET /` returns this file
(markdown), `GET /tools` returns a machine-readable manifest — one entry per
operation with a JSON-Schema `input_schema`, the HTTP `method`+`path`, and a
description. An agent (or a thin MCP bridge) can discover the whole surface with
`curl 127.0.0.1:8765/tools` and needs nothing else.

## Start / discover the daemon

The daemon auto-starts on first submit — you rarely start it by hand. But:

```bash
python tasks.py daemon            # start it, detached, wait for /health
python -m anima_daemon          # equivalent (what the spawner runs)
python tasks.py daemon-status     # one JSON object: health + resolved base_url
                                  # + compact job summaries (newest first, capped)
curl -s 127.0.0.1:8765/health     # {"ok":true,"pid":…,"active_job":…,"paused":…}
```

**Submit → wait → read the result**, the whole agent/bench loop, without writing
any Python:

```bash
python tasks.py daemon-run bench/memorization/probe.py --n 5        # attach + stream
python tasks.py daemon-run --stall-timeout 0 my_quiet_loop.py       # quiet loop, no watchdog
python tasks.py daemon-run --queue long_sweep.py                    # detach instead
JOB=<id> python tasks.py daemon-wait          # block; print record + result envelope
JOB=<id> python tasks.py daemon-status        # one record, envelope inlined
```

`daemon-run` exits with the job's own exit code (so `&&` chains work), ctrl-C
detaches, and `--label NAME` overrides the auto-derived label. `daemon-run`'s
own flags (`--label`, `--stall-timeout`) are recognized only **before** the
script path — after it every token belongs to the child (bench scripts define
`--label` themselves), and a literal `--` makes everything after it child argv
verbatim, run-mode flags included. When the daemon side wasn't given a label,
the child's own `--label` is folded into the display one
(`run_bench --label ko3_a` → job `run_bench:ko3_a`), so a grid of N runs of the
same script doesn't render as N identical rows; the child argv is passed through
untouched either way. The same three
verbs exist inside the package for callers that can't import `tasks.py` (a
vendored node tree, a bare checkout):

```bash
python -m anima_daemon submit [--label L] [--stall-timeout S] [--wait] [--hold] -- <argv…>
python -m anima_daemon wait <job_id> [--timeout S]      # exit = job's exit code, 124 on timeout
python -m anima_daemon status [job_id]
```

`wait --timeout S` exits `124` (like `timeout(1)`) **and** prints a JSON
snapshot of where the job stands — `state`, the last `progress.jsonl` event and
its staleness — so a scripted caller that gives up still learns whether the run
is healthy-but-slow or wedged, instead of wrapping this in its own `timeout` and
losing the status to the attach buffer.

`daemon-status` is the one-shot answer to "is anything running and where do I
talk to it" — scripts and agents should start there instead of assuming a port.
Each compact job carries a derived `target` (what it operates on — the soup
name, the train `output_name`, a bench script's own `--label`) plus its
`returncode`, so a clean `done` and a "terminal but nonzero / signal-killed" row
are distinguishable in the listing rather than one `--job <id>` open per job.
`jobs_total`/`jobs_shown` report truncation.
The job list is **newest first** and capped to the most-recent 15 by default;
filter it with `ARGS="…"` (or pass the flags directly to the CLI):

```bash
python tasks.py daemon-status --running        # only running/paused jobs
python tasks.py daemon-status --failed         # only error/stopped
python tasks.py daemon-status --state done     # exact state(s), comma-separated
python tasks.py daemon-status --limit 40       # raise/lower the cap
python tasks.py daemon-status --all            # no cap (full history)
python tasks.py daemon-status --full           # raw records, not compact
python tasks.py daemon-status --job <id>       # ONE record, full, + its result envelope
```

`--state` validates its argument against the six real states and errors (exit 2)
on anything else — a typo (or a job id passed where a state belongs) used to
return `jobs_shown: 0`, which reads exactly like "the job vanished". To look up
one job use `--job <id>` (or `JOB=<id>`); that path also reads the on-disk record
when the daemon is down, so post-mortems don't need a live daemon.

The port is **not** guaranteed to be 8765: if a stranger holds that port the
daemon falls back to an OS-chosen one and records it in the pidfile. Always
resolve the real port from the pidfile rather than hardcoding — the Python
client does this for you (`DaemonClient()` with no arg), and `client._resolve_port()`
reads `output/daemon/daemon.json` → `port`.

## Two job kinds

| kind | what it runs | how it finalizes |
|------|--------------|------------------|
| `train` (default) | a `train.py` run built from `method` + `preset` + `overrides` + `extra` | `progress.jsonl` stream + exit code |
| `command` | a plain `python <argv>` task (preprocess, mask, a distill loop) | exit code only |

A `command` job can carry a **`chain_train`** spec — `{method, preset,
methods_subdir, overrides}`. When the command finishes successfully the daemon
auto-enqueues that training job. This is how "preprocess → train" survives the
submitter closing: the chain lives in the daemon, not the caller. The follow-on
job's id lands in the command job's `chained_job_id`.

## REST endpoints

Bodies are plain JSON dicts; there's no schema validation (trusted localhost
callers only). Field reference for the `Job` record is in `jobs.py`.

### `GET /` · `GET /tools` — self-description

`GET /` (alias `/readme`) serves this README as markdown. `GET /tools` returns
the operation manifest (`[{name, description, method, path, input_schema}, …]`)
— the same catalog an MCP bridge would register. Neither needs the rest of this
doc to be useful; they're the entry point for an agent discovering the daemon.

### `POST /jobs` — submit

Training job:
```json
{
  "method": "lora",
  "preset": "default",
  "methods_subdir": "gui-methods",
  "overrides": {"network_dim": 32, "max_train_epochs": 64},
  "extra": ["--some_flag"],
  "config_snapshot": null,
  "config_file": null,
  "start": true
}
```
Only `method` is required. `overrides` become `--key value` CLI args; `extra` is
appended verbatim. `config_snapshot` (a merged config dict) or `config_file` (a
path) pin the exact config instead of re-resolving the merge chain. Optional
`captured_env` is a whitelisted snapshot of the submitter's shell (`ANIMA_*`,
`CUDA_*`, `HF_*`, `PYTORCH_*`, `TORCH_*`, `NCCL_*`) that the daemon layers under
the job's env at spawn (**daemon-env ← captured_env ← extra_env**), so a queued
job runs with the caller's GPU/model/token settings, not the daemon's boot env.
The Python client and MCP bridge fill it automatically; pass `{}` to opt out.

Command job:
```json
{
  "kind": "command",
  "label": "preprocess",
  "argv": ["tasks.py", "preprocess-config", "..."],
  "extra_env": {"FOO": "bar"},
  "chain_train": {"method": "lora", "preset": "default", "methods_subdir": "gui-methods", "overrides": {}},
  "stall_timeout": 600,
  "start": true
}
```
`argv` is required (non-empty list). `chain_train` is optional.

`stall_timeout` (seconds) overrides the **stall watchdog** budget for this job —
the daemon kills a command job whose `stdout.log` + `progress.jsonl` have both
frozen for 120 s by default, so a wedged download can't park the queue. A
legitimately quiet loop (embed/eval between prints) should raise it, or pass `0`
to opt out entirely, rather than teach the script about the watchdog. Two other
guards make false kills unlikely: the watchdog also samples the job's
**process-tree CPU time** and spares a quiet-but-computing tree (up to 8× the
budget, after which a busy-spinning deadlock is still killed), and
`bench/_common.py::start_heartbeat()` is a one-line stdout keep-alive for scripts
that prefer to self-announce.

`start` controls the queue gate: `true` → run now (resume queue), `false` → add
but hold the queue paused, omitted/`null` → leave the gate as-is.

Response: `201 {"job_id": "20260611-142233-a1b2c3", "state": "queued"}`.

### `GET /jobs` — list

Returns `[job, …]` (full records, submission order). Each job has `state` ∈
`queued | running | paused | done | error | stopped` (`paused` = tree-frozen,
see `/jobs/{id}/pause` below).

### `GET /jobs/{id}` — status

The job record plus two live fields:
- `latest` — last event from `progress.jsonl` (training progress; `null` for command jobs)
- `stale_for` — seconds since the last progress tick (heartbeat staleness)

plus `result_path` / `result_summary` once terminal — see *Where did my run land*
below.

### `GET /jobs/{id}/progress` — query the structured progress stream

Filtered view of the job's `progress.jsonl` (train jobs only) — the surface to
**debug or analyze a run from**: loss/lr/metric curves (`step`), validation CMMD
(`val`), checkpoint saves (`ckpt`), mirrored WARNING+ log records (`log`), and
the run outcome (`run_start`/`run_end`). Query params, all optional:

| param | meaning |
|-------|---------|
| `events` | comma-separated `ev` kinds to keep, e.g. `events=step,val` or `events=log,run_end` |
| `since_step` | keep events at/after this `global_step` (step-less events inherit the preceding step) |
| `every_nth` | thin `step` events to every n-th (the latest step is always kept) |
| `last_n` | trailing cap on returned events (default 200) |

Returns `{job_id, state, progress_path, count, events}`. Example — the loss
curve at 1-in-50 resolution plus any warnings:

```
GET /jobs/{id}/progress?events=step,log&every_nth=50
```

### `POST /jobs/{id}/stop` — abort

Stops a running or queued job (tree-kills the process). Returns `{job_id, state}`.
The Python client's `stop()` with no id resolves the active job from `/health`.

### `POST /jobs/{id}/pause` · `POST /jobs/{id}/resume` — tree-freeze

`pause` SIGSTOPs the job's whole process tree (dataloader workers included);
`resume` SIGCONTs it back. Method-agnostic and zero-cooperation — works
identically on `train.py`, the bespoke turbo/spd/mod loops, bench, and
inference, with no per-loop wiring. The CUDA context and VRAM survive; only SM
scheduling stops, so resume is instant (no reload, no recompile, mid-step
optimizer state intact). Returns `{job_id, state, error?}` (`error` on a refusal;
404 only for an unknown id). Both client helpers (`pause_job`/`resume_job`) with
no id resolve the active job from `/health`.

Semantics:
- **The queue does NOT advance past a paused job** — it still owns its VRAM
  slot; `pause` is "hold my run", not "yield it". The worker stays parked
  monitoring it.
- **Refused** for anything not `running`, and for a multi-GPU `accelerate
  launch` run (a frozen NCCL rank trips the collective heartbeat) — the refusal
  rides an `error` field in the 200 body.
- `stale_for` freezes while paused (state is `paused`, not a wedged `running`),
  so observers don't false-alarm. Wall-clock throughput/ETA inside the run blips
  across the pause; accepted, not compensated.
- The freeze outlives a daemon restart (SIGSTOP persists) — boot reconcile
  re-adopts a `paused` job as-is. `stop`/`shutdown` on a paused job thaw the tree
  first so the kill lands promptly.
- Opportunistic side-runs go *around* the daemon, not through it: a paused train
  run holds only its allocated VRAM, so a small `--inline` inference fits in the
  remainder (poor-man's preemption — the human schedules the gap, not the queue).

### `POST /queue/pause` · `POST /queue/start`

Hold / resume the queue gate. A paused queue keeps accepting submissions but
launches nothing until started.

### `GET /jobs/{id}/logs` — SSE log tail

Server-Sent Events; each `data:` line is a line of the job's combined
stdout+stderr, from the start of the file. Emits a final `{"ev":"eof","state":…}`
once the job is terminal and the log is drained.

### `GET /events` — SSE daemon lifecycle

Daemon-level events (job start/finish, etc.), plus `: keepalive` comments while idle.

### `GET /health`

`{"ok", "pid", "port", "root", "fingerprint", "active_job", "paused",
"worker_alive", "worker_idle_for"}`. `root` is the checkout the daemon belongs
to — useful to confirm you're talking to *this* repo's daemon and not another
checkout's (see `daemon_matches_root` in `client.py`). `fingerprint` is the
content hash of `anima_daemon/*.py` the daemon **booted** with; if it differs
from the current on-disk source the daemon is running stale code and the next
`ensure_daemon()` submit restarts it eagerly (see *Disposable daemon* below). `worker_idle_for` is seconds since the job
worker thread last advanced; a large value while a job sits `queued` means the
worker is wedged behind a long-running job (normal) or — with `worker_alive`
false — has died (a bug worth a report).

### `POST /shutdown`

`{"kill_jobs": true}` → stop the daemon, optionally killing the running job.

## Python client (`anima_daemon.client`)

Pure stdlib (`urllib`) — imports without dragging in `library.*`/torch, so it's
safe to call from anywhere.

```python
from anima_daemon.client import DaemonClient, ensure_daemon   # `Client` is an alias

client = ensure_daemon()          # start-if-needed, returns a live client
# or: client = DaemonClient()     # attach only; assumes one is up

# submit a training run
r = client.submit(
    method="lora",
    preset="default",
    methods_subdir="gui-methods",
    overrides={"network_dim": 32, "max_train_epochs": 64},
    start=True,
)
job_id = r["job_id"]

# block until it's terminal (don't hand-roll a poll loop)
job = client.wait(job_id)                     # optional: poll=…, timeout=…
print(job["state"], job.get("error"), job.get("result_path"))

# stream logs instead of waiting (ends on its own once the job is terminal)
for line in client.stream_logs(job_id):
    print(line)

# control
client.pause_queue(); client.start_queue()        # queue gate
client.pause_job(job_id); client.resume_job(job_id)  # tree-freeze (no id → active job)
client.stop(job_id)               # or client.stop() for the active job
client.list_jobs()
```

`submit_command(label=…, argv=[…], chain_train=…, stall_timeout=…)` submits a
command job. All methods map 1:1 onto the endpoints above, with two client-side
extras:

- **`wait(job_id, poll=5.0, timeout=None)`** — block to a terminal state and
  return the final record. Interval ramps 0.25s → `poll`, so a one-second command
  job returns promptly and a 12-hour run costs one cheap request per `poll`.
  Raises `LookupError` for an unknown id and `TimeoutError` on `timeout` (a
  still-running job must never read as an outcome).
- **`job_record(job_id)`** — one record, falling back to the on-disk `job.json`
  when HTTP fails. This is why `wait` survives the eager stale-code restart
  mid-poll.

`ensure_daemon(expected_root=…)` refuses to attach to a daemon belonging to a
different checkout if that daemon still has live jobs — pass your repo root when
correctness across checkouts matters.

## Where did my run land — the result-envelope lift

A GPU job that produces an artifact record gets it **lifted onto the job record**
on the terminal transition (Phase 1a), so "where did my run land, and what did it
say" is one command instead of a job-dir spelunk. Two producers ship today:

| producer | writes | typical job |
|----------|--------|-------------|
| `bench/_common.py::write_result` | `bench/<m>/results/<ts>[-label]/result.json` | any bench / probe script |
| `inference.py::write_gen_manifest` | `<job_dir>/gen_manifest.json` | `make gen` batch generation |

The mechanism is one pointer file. The daemon exports `ANIMA_DAEMON_JOB_ID` /
`ANIMA_DAEMON_JOB_DIR` into every job's env; a producer that sees `JOB_DIR` drops
`<job_dir>/result_path.json` → `{"path": "<abs path to the envelope>"}`; the
monitor follows it and records `result_path` (absolute) plus `result_summary`
(`{label, metrics}`, lifted opaquely — the envelope schema stays bench-owned).
Both stay `null` for a job that wrote no envelope (training, plain inference), and
the artifacts themselves never move: the daemon holds a pointer and a digest.

**Writing one** needs no daemon-specific code — `write_result(run_dir,
script=__file__, args=args, metrics={…})` already drops the pointer when it's
running under the daemon, and is a plain envelope write when it isn't. **Reading
one back**:

```bash
JOB=<id> python tasks.py daemon-status   # full record + envelope inlined under "result"
JOB=<id> python tasks.py daemon-wait     # block first, then the same
python tasks.py daemon-status --all      # every job's result_path (pointer only)
```

The compact `daemon-status` list carries `result_path` but **not**
`result_summary`: a bench `metrics` blob can run hundreds of lines (per-pair
records, per-step curves) and would swamp the overview.

## Observing without HTTP

Everything is mirrored to disk, so a reader can skip the port entirely:

```
output/daemon/
  daemon.json            pidfile: {pid, create_time, port, root, fingerprint}
  daemon.log             the detached daemon's own stdout/stderr
  jobs/<id>/
    job.json             the full Job record (atomic-replaced on each change;
                         carries `returncode` once the job process exits)
    stdout.log           the subprocess's captured stdout+stderr
    progress.jsonl       structured training progress (train jobs only)
```

`job.json` → `state` is the fast, dependency-free way to check a job; the GUI
reads these files directly (`gui/daemon.py`) rather than polling HTTP in the Qt
thread.

### Retention — `jobs/` is bounded

Job dirs are pruned, not kept forever (a few hundred dirs / tens of MB of
`stdout.log` accumulate within a couple of months otherwise). `jobs.prune_jobs()`
runs at **daemon boot**, from `manager._reconcile` and deliberately *before*
`load_all()` — a pruned job never enters the in-memory table, so no later
`persist()` can recreate the dir. A dir is a candidate only when its `job.json`
parses, its state is terminal, it's older than `ANIMA_DAEMON_JOB_RETENTION_DAYS`
(by `ended_at`, falling back to `submitted_at` then dir mtime), and it's not
among the `ANIMA_DAEMON_JOB_RETENTION_KEEP` newest terminal jobs. Queued /
running / paused dirs and unreadable records are always left alone, and the whole
sweep is best-effort — it can never keep the daemon from booting.

For a manual sweep (a daemon that's been up for weeks, or a preview of what boot
would take), `make daemon-prune` — **dry-run by default**:

```bash
make daemon-prune                              # preview with the configured knobs
make daemon-prune ARGS="--apply"               # actually delete
make daemon-prune ARGS="--days 7 --keep 50 --apply --verbose"
```

It's pure filesystem (`python -m anima_daemon prune`) and works with the daemon
up or down. With one **up**, pruned jobs linger in its in-memory table until its
next restart — harmless, they're finished history — which is why boot remains the
primary path.

## Environment

| var | default | effect |
|-----|---------|--------|
| `ANIMA_DAEMON_PORT` | `8765` | preferred bind port |
| `ANIMA_DAEMON_PIDFILE` | `~/.anima/daemon.json` | per-user pidfile mirror (cross-checkout discovery) |
| `ANIMA_LORA_ROOT` | — | explicit repo root for pidfile discovery |
| `ANIMA_DAEMON_GPU_BUSY_FRAC` | `0.85` | pre-launch GPU guard: card treated as busy above this used/total fraction |
| `ANIMA_DAEMON_GPU_RETRIES` / `_DELAY` | `1` / `2.0` | guard wait before launching anyway |
| `ANIMA_DAEMON_JOB_RETENTION_DAYS` | `30` | boot prune: age above which a *terminal* job dir is deleted; `0` disables |
| `ANIMA_DAEMON_JOB_RETENTION_KEEP` | `200` | newest terminal job dirs always kept, whatever their age |

## Disposable daemon — trust + attach

The daemon is a **throwaway view over disk state**, not a durable service, so we
never have to ask "is the resident process trustworthy?" — we make the answer
irrelevant.

- **Eager restart on stale code.** Each daemon records a content fingerprint of
  its own `anima_daemon/*.py` at boot (in the pidfile + `/health`). Every
  submit goes through `ensure_daemon()`, which compares that against the current
  on-disk source; on a mismatch it `POST /shutdown {kill_jobs:false}` → respawns.
  The fresh daemon's boot reconcile re-adopts the still-running job and queued
  jobs persist on disk, so the restart is lossless (~1–2s, paid at most once per
  code change). `daemon-status` shows `stale_code` for a passive observer.
- **Submit-time env capture.** The submit chokepoints snapshot the caller's
  whitelisted env into the job record (`captured_env`, above), killing the
  "queued job silently ran with the daemon's week-old `CUDA_VISIBLE_DEVICES`"
  vector.
- **Attach by default (CLI).** `make lora` (and the other GPU targets) submit to
  the daemon and then **stream the job's stdout to your terminal**, exiting with
  the job's exit code (`returncode`). Ctrl-C **detaches** — the run survives (it
  prints the re-attach one-liners). `--queue` detaches immediately (the sweep
  producer); `--inline` runs the child directly with no daemon (the debugging
  path — pdb / py-spy / nsys). `ANIMA_RUN_MODE={attach,detach,inline}` sets the
  default; `PROFILE_STEPS` / `ANIMA_ACCELERATE_LAUNCH` force inline.

Corollary constraint: the daemon stays **stdlib-only forever**. The moment it
imports `library.*` or holds a model, restarts stop being ~1s and staleness
becomes real again.

## Gotchas

- **Localhost only.** No remote, no auth — the caller must run on the same machine.
- **Serial queue.** One job runs at a time; submitting while one runs enqueues.
- **No blocking wait *endpoint*.** The HTTP surface is poll-based
  (`GET /jobs/{id}`) or stream-based (`/jobs/{id}/logs`, which ends with an `eof`
  event and then closes the connection). Blocking lives on the client:
  `DaemonClient.wait()` / `make daemon-wait JOB=<id>`.
- **SSE responses are one-per-connection.** `_open_sse` sends `Connection: close`
  and sets `close_connection`: an SSE body has no `Content-Length` and no chunked
  framing, so the client's only EOF signal is the socket closing. Keeping the
  connection alive (the HTTP/1.1 default) made every consumer hang forever *after*
  the `eof` event — including `make daemon-attach` on an already-finished job.
  Don't "optimize" that header back to keep-alive.
- **Port drift.** Resolve from the pidfile, not a constant — the daemon may bind
  an ephemeral port. `DaemonClient()` and `ensure_daemon()` handle this.
- **`config_snapshot` vs re-resolve.** Without a snapshot/file the daemon
  re-runs the `base → preset → method → overrides` merge at launch; pin a
  snapshot when you need bit-stable config across a queued delay.
- **Command-job progress.** `latest`/`progress.jsonl` are training-only; a
  command job exposes only `state` + `stdout.log` until it exits.
- **Stall watchdog on quiet command jobs.** 120 s of frozen output is a kill by
  default (with a process-tree CPU cross-check). If your loop is legitimately
  silent for longer, submit with `stall_timeout` (`--stall-timeout` on
  `daemon-run` / `python -m anima_daemon submit`; `0` disables) or call
  `bench/_common.py::start_heartbeat()`. The kill shows up as `state: error` with
  `stalled: no output for …` in `error`.
- **Agent-launched GPU work must go through the daemon.** A GPU process started
  from an agent's background shell gets SIGKILLed by the harness sandbox after
  ~1 min with no trace; `daemon-run` (or `POST /jobs` kind=command) is the
  supported path.

## MCP bridge (`mcp.py`)

A stdio MCP server over this same surface — pure stdlib, newline-delimited
JSON-RPC, no new deps. Register it with any MCP client (Claude Code, Claude
Desktop, OpenClaw, …) as a **command, never an address**: the bridge resolves
the daemon itself via the pidfile, so it survives port drift and daemon
restarts without reconfiguration.

```bash
# Claude Code (use your checkout's absolute paths; any cwd works)
claude mcp add anima-daemon -- <repo>/.venv/Scripts/python.exe <repo>/anima_daemon/mcp.py
```

For other clients, the equivalent JSON config:

```json
{"mcpServers": {"anima-daemon": {
  "command": "<repo>/.venv/Scripts/python.exe",
  "args": ["<repo>/anima_daemon/mcp.py"]
}}}
```

The tool catalog **is** the `GET /tools` manifest (`server.TOOLS`, registered
verbatim — one source of truth), with two deviations:

- `tail_logs` (SSE) is replaced by **`tail_log`** `{id, lines=80}` — last N
  lines + current state in one call; it reads the on-disk `job.json` +
  `stdout.log` as fallback, so it answers even with the daemon down. tqdm
  `\r`-redraws are collapsed to their final rendering, so a progress bar
  counts as one line.
- **`get_progress`** is served from the on-disk `progress.jsonl` (same filters
  as the HTTP endpoint above), so it too answers even with the daemon down.
- Only `submit_training` / `submit_command` auto-start the daemon; every other
  tool is passive, so an agent asking "is anything running?" never boots a
  daemon as a side effect (`health` returns `{"up": false}` instead of erroring).
