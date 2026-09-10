# Anima training daemon — REST + programmatic interface

A single localhost process owning a **serial job queue**: submit a training run
(or a plain command), one job runs at a time, state survives restarts, observers
poll or stream rather than holding the run open. The GUI Train button, the
ComfyUI trainer node and `make … --queue` all submit here; this doc is the same
surface for direct use from a script, an MCP server, or an agent.

`http://127.0.0.1:8765`, JSON in / JSON out, no auth, localhost only
(`config.py`). All state is on disk under `output/daemon/`, so anything that can
read files can observe a run with the HTTP port down.

**Self-describing.** `GET /` returns this file; `GET /tools` returns a manifest
— one entry per operation with a JSON-Schema `input_schema`, HTTP
`method`+`path`, and a description. `curl 127.0.0.1:8765/tools` is enough to
discover the whole surface.

## Start / discover the daemon

The daemon auto-starts on first submit.

```bash
python tasks.py daemon            # start it, detached, wait for /health
python -m anima_daemon            # equivalent (what the spawner runs)
python tasks.py daemon-status     # one JSON object: health + resolved base_url + jobs
curl -s 127.0.0.1:8765/health     # {"ok":true,"pid":…,"active_job":…,"paused":…}
```

**Submit → wait → read the result**, without writing any Python:

```bash
python tasks.py daemon-run bench/memorization/probe.py --n 5   # attach + stream
python tasks.py daemon-run --stall-timeout 0 my_quiet_loop.py  # quiet loop, no watchdog
python tasks.py daemon-run --queue long_sweep.py               # detach instead
JOB=<id> python tasks.py daemon-wait          # block; print record + result envelope
JOB=<id> python tasks.py daemon-status        # one record, envelope inlined
```

`daemon-run` exits with the job's own exit code, and ctrl-C detaches. Its own
flags (`--label`, `--stall-timeout`) are recognized **only before the script
path** — after it every token belongs to the child (bench scripts define their
own `--label`), and a literal `--` passes everything after it verbatim. Absent a
daemon-side label, the child's `--label` is folded into the display one
(`run_bench --label ko3_a` → job `run_bench:ko3_a`) so a grid of N runs of one
script doesn't render as N identical rows.

The same verbs exist inside the package for callers that can't import `tasks.py`:

```bash
python -m anima_daemon submit [--label L] [--stall-timeout S] [--wait] [--hold] -- <argv…>
python -m anima_daemon wait <job_id> [--timeout S]   # exit = job's exit code, 124 on timeout
python -m anima_daemon status [job_id]
```

`wait --timeout S` exits `124` and prints a JSON snapshot — `state`, the last
`progress.jsonl` event, its staleness — so a caller that gives up still learns
whether the run is healthy-but-slow or wedged.

## Reading the queue

`daemon-status` is the machine surface: one JSON object, jobs **newest first**,
capped at 15, with every unfinished job (`queued`/`running`/`paused`) pinned in
even when it falls below the cap — jobs do not always start in submit order, so
without pinning a pending job can sit under 15 finished rows and the queue reads
as empty. `jobs_total`/`jobs_shown`/`jobs_pinned` report the truncation. Each
compact job carries a derived `target` (soup name, train `output_name`, a bench
script's `--label`) and its `returncode`.

`daemon-jobs` is the human surface: one greppable line per job, **oldest first**,
so `| tail -5` is the five most recent. (Tailing `daemon-status` shows the
*oldest* rows, cut mid-record.) `daemon-log` is the post-mortem counterpart to
`daemon-attach`, which follows a live stream and has nothing to show once a job
is terminal.

```bash
python tasks.py daemon-status --running        # only running/paused jobs
python tasks.py daemon-status --failed         # only error/stopped
python tasks.py daemon-status --state done     # exact state(s), comma-separated
python tasks.py daemon-status --limit 40       # raise/lower the cap
python tasks.py daemon-status --all            # no cap (full history)
python tasks.py daemon-status --full           # raw records, not compact
python tasks.py daemon-status --job <id>       # ONE record, full, + its result envelope
python tasks.py daemon-jobs                    # newest 15, newest LAST
python tasks.py daemon-jobs --failed --all     # every error/stopped job
JOB=<id> python tasks.py daemon-log            # that job's stdout, from disk
python tasks.py daemon-log -n 0                # newest job, whole log
```

`--state` validates against the six real states and exits 2 on anything else; a
typo would otherwise filter everything away and read as "the job vanished".
`--job`, `daemon-jobs` and `daemon-log` all fall back to the on-disk records, so
they answer with the daemon down.

## Two job kinds

| kind | what it runs | how it finalizes |
|------|--------------|------------------|
| `train` (default) | a `train.py` run built from `method` + `preset` + `overrides` + `extra` | `progress.jsonl` stream + exit code |
| `command` | a plain `python <argv>` task (preprocess, mask, a distill loop) | exit code only |

A `command` job can carry a **`chain_train`** spec — `{method, preset,
methods_subdir, overrides}` — and the daemon auto-enqueues that training job when
the command succeeds, so "preprocess → train" survives the submitter closing. The
follow-on id lands in the command job's `chained_job_id`.

## REST endpoints

Bodies are plain JSON dicts; no schema validation (trusted localhost callers).
Field reference for the `Job` record is in `jobs.py`.

### `GET /` · `GET /tools` — self-description

`GET /` (alias `/readme`) serves this README. `GET /tools` returns the operation
manifest (`[{name, description, method, path, input_schema}, …]`) — the catalog
an MCP bridge registers.

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
Only `method` is required. `overrides` become `--key value`; `extra` is appended
verbatim. `config_snapshot` (a merged config dict) or `config_file` (a path) pin
the exact config instead of re-resolving the merge chain. Optional
`captured_env` is a whitelisted snapshot of the submitter's shell (`ANIMA_*`,
`CUDA_*`, `HF_*`, `PYTORCH_*`, `TORCH_*`, `NCCL_*`) layered under the job's env at
spawn (**daemon-env ← captured_env ← extra_env**), so a queued job runs with the
caller's settings, not the daemon's boot env. The Python client and MCP bridge
fill it automatically; pass `{}` to opt out.

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
`argv` is required (non-empty list); `chain_train` is optional.

`stall_timeout` (seconds) overrides the **stall watchdog** for this job — the
daemon kills a command job whose `stdout.log` and `progress.jsonl` have both
frozen for 120 s by default, so a wedged download can't park the queue. Raise it
for a legitimately quiet loop, or pass `0` to opt out. The watchdog also samples
the job's process-tree CPU time and spares a quiet-but-computing tree (up to 8×
the budget); `bench/_common.py::start_heartbeat()` is a one-line stdout
keep-alive for scripts that prefer to self-announce.

`start` gates the queue: `true` → run now, `false` → add but hold the queue
paused, omitted/`null` → leave the gate as-is.

Response: `201 {"job_id": "20260611-142233-a1b2c3", "state": "queued"}`.

### `GET /jobs` — list

`[job, …]` (full records, submission order). `state` ∈ `queued | running |
paused | done | error | stopped`.

### `GET /jobs/{id}` — status

The job record plus two live fields:
- `latest` — last event from `progress.jsonl` (`null` for command jobs)
- `stale_for` — seconds since the last progress tick

plus `result_path` / `result_summary` once terminal.

### `GET /jobs/{id}/progress` — query the structured progress stream

Filtered view of the job's `progress.jsonl` (train jobs only): loss/lr/metric
curves (`step`), validation CMMD (`val`), checkpoint saves (`ckpt`), mirrored
WARNING+ log records (`log`), run outcome (`run_start`/`run_end`). Query params,
all optional:

| param | meaning |
|-------|---------|
| `events` | comma-separated `ev` kinds to keep, e.g. `events=step,val` |
| `since_step` | keep events at/after this `global_step` (step-less events inherit the preceding step) |
| `every_nth` | thin `step` events to every n-th (the latest step is always kept) |
| `last_n` | trailing cap on returned events (default 200) |

Returns `{job_id, state, progress_path, count, events}`.

```
GET /jobs/{id}/progress?events=step,log&every_nth=50
```

### `POST /jobs/{id}/stop` — abort

Tree-kills a running or queued job. Returns `{job_id, state}`. The client's
`stop()` with no id resolves the active job from `/health`.

### `POST /jobs/{id}/pause` · `POST /jobs/{id}/resume` — tree-freeze

`pause` SIGSTOPs the job's whole process tree (dataloader workers included);
`resume` SIGCONTs it. Method-agnostic and zero-cooperation — identical on
`train.py`, the bespoke turbo/spd/mod loops, bench, and inference. The CUDA
context and VRAM survive; only SM scheduling stops, so resume is instant (no
reload, no recompile, optimizer state intact). Returns `{job_id, state, error?}`
(`error` on a refusal; 404 only for an unknown id). `pause_job`/`resume_job` with
no id resolve the active job.

- **The queue does NOT advance past a paused job** — it still owns its VRAM slot.
- **Refused** for anything not `running`, and for a multi-GPU `accelerate launch`
  run (a frozen NCCL rank trips the collective heartbeat).
- `stale_for` freezes while paused. Wall-clock throughput/ETA inside the run blips
  across the pause; accepted, not compensated.
- The freeze outlives a daemon restart. `stop`/`shutdown` thaw the tree first.
- A paused run holds only its allocated VRAM, so a small `--inline` job fits in
  the remainder.

### `POST /queue/pause` · `POST /queue/start`

Hold / resume the queue gate. A paused queue keeps accepting submissions but
launches nothing.

### `GET /jobs/{id}/logs` — SSE log tail

Each `data:` line is a line of the job's combined stdout+stderr, from the start
of the file. Emits a final `{"ev":"eof","state":…}` once the job is terminal and
the log is drained.

### `GET /events` — SSE daemon lifecycle

Job start/finish and friends, plus `: keepalive` comments while idle.

### `GET /health`

`{"ok", "pid", "port", "root", "fingerprint", "active_job", "paused",
"worker_alive", "worker_idle_for"}`. `root` is the checkout the daemon belongs to
(`daemon_matches_root` in `client.py`). `fingerprint` is the content hash of
`anima_daemon/*.py` the daemon booted with; if it differs from the on-disk source
the next `ensure_daemon()` submit restarts it. `worker_idle_for` is seconds since
the job worker last advanced — large while a job sits `queued` means the worker
is behind a long run (normal), or with `worker_alive` false, has died (a bug).

### `POST /shutdown`

`{"kill_jobs": true}` → stop the daemon, optionally killing the running job.

## Python client (`anima_daemon.client`)

Pure stdlib (`urllib`) — imports without dragging in `library.*`/torch.

```python
from anima_daemon.client import DaemonClient, ensure_daemon   # `Client` is an alias

client = ensure_daemon()          # start-if-needed, returns a live client
# or: client = DaemonClient()     # attach only; assumes one is up

r = client.submit(
    method="lora",
    preset="default",
    methods_subdir="gui-methods",
    overrides={"network_dim": 32, "max_train_epochs": 64},
    start=True,
)
job_id = r["job_id"]

job = client.wait(job_id)                     # optional: poll=…, timeout=…
print(job["state"], job.get("error"), job.get("result_path"))

for line in client.stream_logs(job_id):       # ends once the job is terminal
    print(line)

client.pause_queue(); client.start_queue()           # queue gate
client.pause_job(job_id); client.resume_job(job_id)  # tree-freeze (no id → active)
client.stop(job_id)                                  # or stop() for the active job
client.list_jobs()
```

`submit_command(label=…, argv=[…], chain_train=…, stall_timeout=…)` submits a
command job. Methods map 1:1 onto the endpoints, with two client-side extras:

- **`wait(job_id, poll=5.0, timeout=None)`** — block to terminal, return the final
  record. Interval ramps 0.25s → `poll`. Raises `LookupError` for an unknown id
  and `TimeoutError` on `timeout`, so a still-running job never reads as an outcome.
- **`job_record(job_id)`** — one record, falling back to the on-disk `job.json`
  when HTTP fails, which is why `wait` survives a restart mid-poll.

`ensure_daemon(expected_root=…)` refuses to attach to another checkout's daemon
while that daemon has live jobs.

## Where did my run land — the result-envelope lift

A GPU job that produces an artifact record gets it lifted onto the job record on
the terminal transition. Two producers ship today:

| producer | writes | typical job |
|----------|--------|-------------|
| `bench/_common.py::write_result` | `bench/<m>/results/<ts>[-label]/result.json` | any bench / probe script |
| `inference.py::write_gen_manifest` | `<job_dir>/gen_manifest.json` | `make gen` batch generation |

One pointer file: the daemon exports `ANIMA_DAEMON_JOB_ID` /
`ANIMA_DAEMON_JOB_DIR` into every job's env, a producer that sees `JOB_DIR` drops
`<job_dir>/result_path.json` → `{"path": "<abs path>"}`, and the monitor records
`result_path` plus `result_summary` (`{label, metrics}`, lifted opaquely — the
schema stays bench-owned). Both stay `null` for a job that wrote no envelope; the
artifacts never move. `write_result(run_dir, script=__file__, args=args,
metrics={…})` drops the pointer under the daemon and is a plain envelope write
otherwise. Reading one back:

```bash
JOB=<id> python tasks.py daemon-status   # full record + envelope inlined under "result"
JOB=<id> python tasks.py daemon-wait     # block first, then the same
python tasks.py daemon-status --all      # every job's result_path (pointer only)
```

The compact list carries `result_path` but **not** `result_summary`: a bench
`metrics` blob can run hundreds of lines and would swamp the overview.

## Observing without HTTP

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

`job.json` → `state` is the fast, dependency-free check; the GUI reads these
files directly (`gui/daemon.py`) rather than polling HTTP in the Qt thread.

### Retention — `jobs/` is bounded

`jobs.prune_jobs()` runs at **daemon boot**, before `load_all()`. A dir is a
candidate only when its `job.json` parses, its state is terminal, it is older
than `ANIMA_DAEMON_JOB_RETENTION_DAYS` (by `ended_at`, else `submitted_at`, else
dir mtime), and it is not among the `ANIMA_DAEMON_JOB_RETENTION_KEEP` newest
terminal jobs. Queued / running / paused dirs and unreadable records are left
alone; the sweep is best-effort and never blocks boot.

```bash
make daemon-prune                              # dry-run preview
make daemon-prune ARGS="--apply"               # actually delete
make daemon-prune ARGS="--days 7 --keep 50 --apply --verbose"
```

Pure filesystem (`python -m anima_daemon prune`), works daemon up or down. With
one up, pruned jobs linger in its in-memory table until its next restart.

## Environment

| var | default | effect |
|-----|---------|--------|
| `ANIMA_DAEMON_PORT` | `8765` | preferred bind port |
| `ANIMA_DAEMON_PIDFILE` | `~/.anima/daemon.json` | per-user pidfile mirror (cross-checkout discovery) |
| `ANIMA_LORA_ROOT` | — | explicit repo root for pidfile discovery |
| `ANIMA_DAEMON_GPU_BUSY_FRAC` | `0.85` | pre-launch GPU guard: card busy above this used/total fraction |
| `ANIMA_DAEMON_GPU_RETRIES` / `_DELAY` | `1` / `2.0` | guard wait before launching anyway |
| `ANIMA_DAEMON_JOB_RETENTION_DAYS` | `30` | boot prune: age above which a *terminal* job dir is deleted; `0` disables |
| `ANIMA_DAEMON_JOB_RETENTION_KEEP` | `200` | newest terminal job dirs always kept, whatever their age |

## Disposable daemon

The daemon is a throwaway view over disk state, not a durable service.

- **Eager restart on stale code.** Each daemon fingerprints its own
  `anima_daemon/*.py` at boot; every submit goes through `ensure_daemon()`, which
  compares it to the on-disk source and on a mismatch does
  `POST /shutdown {kill_jobs:false}` → respawn. Boot reconcile re-adopts the
  running job and queued jobs persist, so the restart is lossless (~1–2s).
  `daemon-status` shows `stale_code`.
- **Submit-time env capture** keeps a queued job off the daemon's boot env.
- **Attach by default (CLI).** GPU targets submit and stream the job's stdout,
  exiting with its `returncode`; ctrl-C detaches and the run survives. `--queue`
  detaches immediately, `--inline` runs the child with no daemon (pdb / py-spy /
  nsys). `ANIMA_RUN_MODE={attach,detach,inline}` sets the default;
  `PROFILE_STEPS` / `ANIMA_ACCELERATE_LAUNCH` force inline.

Corollary: the daemon stays **stdlib-only forever** — importing `library.*` or
holding a model makes restarts slow and staleness real again.

## Gotchas

- **Localhost only.** No remote, no auth — the caller runs on the same machine.
- **Serial queue.** One job at a time; submitting while one runs enqueues.
- **No blocking wait *endpoint*.** HTTP is poll-based (`GET /jobs/{id}`) or
  stream-based (`/jobs/{id}/logs`). Blocking lives on the client:
  `DaemonClient.wait()` / `make daemon-wait JOB=<id>`.
- **SSE responses are one-per-connection.** `_open_sse` sends `Connection: close`
  and sets `close_connection` — an SSE body has no `Content-Length`, so the
  client's only EOF signal is the socket closing. Do not make it keep-alive:
  every consumer then hangs after the `eof` event.
- **Port drift.** Resolve from the pidfile, not a constant. `DaemonClient()` and
  `ensure_daemon()` handle it.
- **`config_snapshot` vs re-resolve.** Without a snapshot/file the daemon re-runs
  the `base → preset → method → overrides` merge at launch; pin a snapshot when
  you need bit-stable config across a queued delay.
- **Command-job progress.** `latest`/`progress.jsonl` are training-only; a
  command job exposes only `state` + `stdout.log` until it exits.
- **Agent-launched GPU work must go through the daemon.** A GPU process started
  from an agent's background shell gets SIGKILLed by the harness sandbox after
  ~1 min with no trace.

## MCP bridge (`mcp.py`)

A stdio MCP server over the same surface — pure stdlib, newline-delimited
JSON-RPC. Register it as a **command, never an address**: the bridge resolves the
daemon via the pidfile, so it survives port drift and restarts.

```bash
# absolute paths; any cwd works. Other clients: the same command/args as JSON.
claude mcp add anima-daemon -- <repo>/.venv/bin/python <repo>/anima_daemon/mcp.py
```

The tool catalog **is** the `GET /tools` manifest (`server.TOOLS`, registered
verbatim), with three deviations:

- `tail_logs` (SSE) is replaced by **`tail_log`** `{id, lines=80}` — last N lines
  + current state in one call, reading the on-disk `job.json` + `stdout.log` as
  fallback. tqdm `\r`-redraws collapse to their final rendering.
- **`get_progress`** is served from the on-disk `progress.jsonl` (same filters as
  the HTTP endpoint), so it answers with the daemon down.
- Only `submit_training` / `submit_command` auto-start the daemon; every other
  tool is passive, so "is anything running?" never boots one (`health` returns
  `{"up": false}` instead of erroring).
