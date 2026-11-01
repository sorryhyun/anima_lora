# Anima training daemon — REST + programmatic interface

A single localhost process owning a **serial job queue**: submit a training run
(or a plain command), one job runs at a time, state survives restarts, observers
poll or stream rather than holding the run open. The GUI Train button, the
ComfyUI trainer node and `make … --queue` all submit here; this doc is the same
surface for direct use from a script, an MCP server, or an agent.

`http://127.0.0.1:<port>`, JSON in / JSON out, no auth, localhost only
(`config.py`). The port is 8765 unless taken, then ephemeral, so resolve it from
the pidfile; `DaemonClient()` and `ensure_daemon()` do. All state is on disk under
`output/daemon/`, so anything that can read files can observe a run with the HTTP
port down.

## Start / discover the daemon

The daemon auto-starts on first submit.

```bash
python tasks.py daemon            # start it, detached, wait for /health
python -m anima_daemon            # equivalent (what the spawner runs)
python -m anima_daemon status     # health + resolved base_url + stale_code
curl -s 127.0.0.1:8765/health     # {"ok":true,"pid":…,"active_job":…,"paused":…}
```

**Submit → wait → read the result**, without writing any Python:

```bash
python tasks.py daemon-run bench/memorization/probe.py --n 5   # attach + stream
python tasks.py daemon-run --stall-timeout 0 my_quiet_loop.py  # quiet loop, no watchdog
python tasks.py daemon-run --queue long_sweep.py               # detach instead
JOB=<id> python tasks.py daemon-wait          # block; print record + result envelope
python -m anima_daemon status <id>            # one record, envelope inlined
```

`daemon-run` exits with the job's own exit code, and ctrl-C detaches. Its own
flags (`--label`, `--stall-timeout`) are recognized **only before the script
path** — after it every token belongs to the child (bench scripts define their
own `--label`), and a literal `--` passes everything after it verbatim. Absent a
daemon-side label, the child's `--label` is folded into the display one
(`run_bench --label ko3_a` → job `run_bench:ko3_a`).

The same verbs exist inside the package for callers that can't import `tasks.py`:

```bash
python -m anima_daemon submit [--label L] [--stall-timeout S] [--wait] [--hold] -- <argv…>
python -m anima_daemon wait <job_id> [--timeout S]   # exit = job's exit code
python -m anima_daemon status [job_id]
```

`wait --timeout S` exits `124` on timeout and prints a JSON snapshot: `state`, the
last `progress.jsonl` event, its staleness.

## Reading the queue

Use `daemon-jobs` to see what has run or is queued, `run-status` for how far the
current run has got, `daemon-wait` to block on one job, `daemon-log` for a job's
stdout, and `python -m anima_daemon status [id]` for daemon health or one full
record with its result envelope.

`daemon-jobs` prints one greppable line per job — when · id · state · `rc=` ·
duration · derived `target` (soup name, train `output_name`, a bench script's
`--label`) · first error line — **oldest first**, capped at the newest 15, with a
trailing `N of M jobs`. `--state queued,running,paused` shows live jobs (`0 of
M` = nothing running). `daemon-log` reads the log off disk, so unlike
`daemon-attach` (a live stream) it works for a finished job.

```bash
python tasks.py daemon-jobs                    # newest 15, newest LAST
python tasks.py daemon-jobs --running          # only running/paused jobs
python tasks.py daemon-jobs --failed --all     # every error/stopped job
python tasks.py daemon-jobs --state done       # exact state(s), comma-separated
python tasks.py daemon-jobs --limit 40         # raise/lower the cap
python tasks.py daemon-jobs --all              # no cap (full history)
python tasks.py daemon-log -n 0                # newest job, whole log
```

`--state` validates against the six states and exits 2 on anything else.
`daemon-jobs`, `daemon-log` and `python -m anima_daemon status <id>` all fall
back to the on-disk records, so they answer with the daemon down.

## Two job kinds

| kind | what it runs | how it finalizes |
|------|--------------|------------------|
| `train` (default) | a `train.py` run built from `method` + `preset` + `overrides` + `extra` | `progress.jsonl` stream + exit code |
| `command` | a plain `python <argv>` task (preprocess, mask, a distill loop) | exit code; a loop that writes `$ANIMA_DAEMON_JOB_DIR/progress.jsonl` (moss train, via `library.runtime.progress.command_job_path`) is read like a train job's, but a nonzero exit outranks its `run_end` "ok" |

A train job's record keeps `method`/`preset`/`overrides`/`extra` and builds its
launch argv at spawn, so its `argv` field stays empty; a command job's `argv` is
literally what runs (`jobs/<id>/job.json`).

A `command` job can carry a **`chain_train`** spec — `{method, preset,
methods_subdir, overrides}` — and the daemon auto-enqueues that training job when
the command succeeds, even if the submitter has exited. The follow-on id lands in
the command job's `chained_job_id`.

## REST endpoints

Bodies are plain JSON dicts; no schema validation. Field reference for the `Job`
record is in `jobs.py`.

### `GET /` · `GET /tools` — self-description

`GET /` (alias `/readme`) serves this README. `GET /tools` returns the operation
manifest (`[{name, description, method, path, input_schema}, …]`, JSON-Schema
inputs), which is what the MCP bridge registers.

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
the exact config; without one the daemon re-runs the `base → preset → method →
overrides` merge at launch, so a config edit made while the job is queued reaches
it. Optional `captured_env` is a whitelisted snapshot of the submitter's shell (`ANIMA_*`,
`CUDA_*`, `HF_*`, `PYTORCH_*`, `TORCH_*`, `NCCL_*`) layered under the job's env at
spawn (**daemon-env ← captured_env ← extra_env**). The Python client and MCP
bridge fill it automatically; pass `{}` to opt out.

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
frozen for 120 s by default. Raise it for a legitimately quiet loop, or pass `0`
to opt out. The watchdog also samples the job's process-tree CPU time and spares
a quiet-but-computing tree (up to 8× the budget);
`bench/_common.py::start_heartbeat()` is a one-line stdout keep-alive.

`start` gates the queue: `true` → run now (resumes the gate), `false` → pause
the gate only if nothing else is queued or running (otherwise the job advances
behind them), omitted/`null` → leave the gate as-is.

Response: `201 {"job_id": "20260611-142233-a1b2c3", "state": "queued"}`.

### `GET /jobs` — list

`[job, …]` (full records, submission order). `state` ∈ `queued | running |
paused | done | error | stopped`.

### `GET /jobs/{id}` — status

The job record plus two live fields:
- `latest` — last event from `progress.jsonl` (`null` for a command job that writes none)
- `stale_for` — seconds since the last progress tick

plus `result_path` / `result_summary` once terminal.

### `GET /jobs/{id}/progress` — query the structured progress stream

Filtered view of the job's `progress.jsonl` (train jobs, and command jobs that write one): loss/lr/metric
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
`resume` SIGCONTs it. Works on any job with no cooperation from the child. The
CUDA context and VRAM survive, so resume is instant. Returns
`{job_id, state, error?}` (`error` on a refusal; 404 only for an unknown id).
`pause_job`/`resume_job` with no id resolve the active job.

- **The queue does NOT advance past a paused job** — it still owns its VRAM slot.
- **Refused** for anything not `running`, and for a multi-GPU `accelerate launch`
  run (a frozen NCCL rank trips the collective heartbeat).
- `stale_for` freezes while paused. Wall-clock throughput/ETA inside the run
  includes the pause.
- The freeze outlives a daemon restart. `stop`/`shutdown` thaw the tree first.
- A paused run holds only its allocated VRAM, so a small `--inline` job fits in
  the remainder.

**`{"release_model": true}`** in the body (`make daemon-pause RELEASE=1`,
`client.pause_job(id, release_model=True)`) is the cooperative variant for
**train.py jobs only**: the daemon drops `pause.request` in the job dir, the
trainer picks it up at its next optimizer step (`library/runtime/pause.py`),
writes the resumable `<output_name>-checkpoint` weights + `-checkpoint-state`
dir (the same files auto-resume uses, regardless of `checkpointing_epochs`),
writes `pause.ack.json`, and exits with `run_end status="paused"` carrying
`state_dir`. The daemon then parks the job as `paused` with `released: true`,
`pid: null` and `resume_state_dir` set. Returns
`{job_id, state: "running", release_requested: true}` immediately — the job
stays `running` until the trainer has actually exited (`status_detail` says
so). A frozen (SIGSTOP) job is thawed first so it can act on the request.

- **The GPU is fully released and the queue DOES advance** past a released job.
- `resume` on a released job re-enqueues it at the **front** of the queue with
  `--resume <state_dir> --skip_until_initial_step` appended to `extra`
  (`resume_count` increments); it relaunches when the slot frees — model reload
  + recompile, **not** instant. The resumed run removes the state dir on
  completion.
- `stop` on a released job finalizes it `stopped` (nothing to kill). A released
  job survives a daemon restart as-is. Refused for command jobs (no
  cooperative checkpoint) — use the plain freeze there.
- Not refused for `accelerate launch` runs: every rank reaches the save
  collective together (the main process's verdict is broadcast).

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
`anima_daemon/*.py` the daemon booted with (see Restarts and run modes).
`worker_idle_for` is seconds since
the job worker last advanced — large while a job sits `queued` means the worker
is behind a long run, or, with `worker_alive` false, has died.

### `POST /shutdown`

`{"kill_jobs": true}` → stop the daemon, optionally killing the running job.

## Python client (`anima_daemon.client`)

`urllib`-based, no `library.*` / torch imports.

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
  and `TimeoutError` on `timeout`.
- **`job_record(job_id)`** — one record, falling back to the on-disk `job.json`
  when HTTP fails, so `wait` survives a restart mid-poll.

`ensure_daemon(expected_root=…)` refuses to attach to another checkout's daemon
while that daemon has live jobs.

## Result envelopes

A GPU job that produces an artifact record gets it lifted onto the job record on
the terminal transition. Producers:

| producer | writes | typical job |
|----------|--------|-------------|
| `bench/_common.py::write_result` | `bench/<m>/results/<ts>[-label]/result.json` | any bench / probe script |
| `inference.py::write_gen_manifest` | `<job_dir>/gen_manifest.json` | `make gen` batch generation |

One pointer file: the daemon exports `ANIMA_DAEMON_JOB_ID` /
`ANIMA_DAEMON_JOB_DIR` into every job's env, a producer that sees `JOB_DIR` drops
`<job_dir>/result_path.json` → `{"path": "<abs path>"}`, and the monitor records
`result_path` plus `result_summary` (`{label, metrics}`, lifted opaquely — the
schema is bench-owned). Both stay `null` for a job that wrote no envelope; the
artifacts never move. `write_result(run_dir, script=__file__, args=args,
metrics={…})` drops the pointer under the daemon and is a plain envelope write
otherwise. `python -m anima_daemon status <id>` and `daemon-wait` inline the
envelope under `result`.

## Observing without HTTP

```
output/daemon/
  daemon.json            pidfile: {pid, create_time, port, root, fingerprint}
  daemon.log             the detached daemon's own stdout/stderr
  current -> jobs/<id>/  the running job (last launched when idle); relative
  current.log -> jobs/<id>/stdout.log    symlinks, retargeted at launch/adopt,
                         so `tail -f output/daemon/current.log` follows whatever
                         the queue is driving. Best-effort (skipped where the
                         host refuses symlinks).
  jobs/<id>/
    job.json             the full Job record (atomic-replaced on each change;
                         carries `returncode` once the job process exits)
    stdout.log           the subprocess's captured stdout+stderr
    progress.jsonl       structured training progress (train jobs; command jobs that write one)
```

`state` in `job.json` is the fast check; the GUI reads these files directly
(`gui/jobs/daemon.py`).

### Retention

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

## Restarts and run modes

- **Eager restart on stale code.** Each daemon fingerprints its own
  `anima_daemon/*.py` at boot; every submit goes through `ensure_daemon()`, which
  compares it to the on-disk source and on a mismatch does
  `POST /shutdown {kill_jobs:false}` → respawn. Boot reconcile re-adopts the
  running job and queued jobs persist, so the restart is lossless (~1–2s).
  `python -m anima_daemon status` shows `stale_code`.
- **Attach by default (CLI).** GPU targets submit and stream the job's stdout,
  exiting with its `returncode`; ctrl-C detaches and the run survives. `--queue`
  detaches immediately, `--inline` runs the child with no daemon (pdb / py-spy /
  nsys). `ANIMA_RUN_MODE={attach,detach,inline}` sets the default;
  `PROFILE_STEPS` / `ANIMA_ACCELERATE_LAUNCH` force inline unless a mode flag
  is given.

The daemon package never imports `library` / `networks` / `torch` and never
holds a model (`tests/test_daemon.py` enforces the imports).

## Gotchas

- **No blocking wait endpoint.** Poll `GET /jobs/{id}` or stream
  `/jobs/{id}/logs`; blocking lives on the client (`DaemonClient.wait()` /
  `make daemon-wait JOB=<id>`).
- **SSE responses are one-per-connection** (`Connection: close`); the socket
  closing is the client's EOF signal.
- **Agent-launched GPU work must go through the daemon.** A GPU process started
  from an agent's background shell gets SIGKILLed by the harness sandbox after
  ~1 min with no trace.

## MCP bridge (`mcp.py`)

A stdio MCP server over the same surface, newline-delimited JSON-RPC. Register it
as a command: the bridge resolves the daemon via the pidfile, so it survives port
changes and restarts.

```bash
# absolute paths; any cwd works. Other clients: the same command/args as JSON.
claude mcp add anima-daemon -- <repo>/.venv/bin/python <repo>/anima_daemon/mcp.py
```

The tool catalog is the `GET /tools` manifest (`tools.TOOLS`), with three
deviations:

- `tail_logs` (SSE) is replaced by **`tail_log`** `{id, lines=80}` — last N lines
  + current state in one call, reading the on-disk `job.json` + `stdout.log` as
  fallback. tqdm `\r`-redraws collapse to their final rendering.
- **`get_progress`** is served from the on-disk `progress.jsonl` (same filters as
  the HTTP endpoint), so it answers with the daemon down.
- Only `submit_training` / `submit_command` auto-start the daemon; every other
  tool is passive (`health` returns `{"up": false}` instead of erroring).

## RunPod pods (`anima_daemon.runpod`)

Not wired into the queue yet: a stdlib REST client (`api.RunPodClient`) and a
CLI for renting a pod by hand. `create` goes through v2 (`POST
api.runpod.io/v2/pods`, the only one that takes `gpu.minCudaVersion` past 13.0);
reading, listing, start / stop / terminate and volumes through the v1 REST API
(`rest.runpod.io/v1`), which reads v2-created pods; GPU price and stock through
the v2 catalog (`api.runpod.io/v2/catalog/gpus`). Key: `RUNPOD_API_KEY` from the environment,
then the repo `.env`, then `apikey` in `~/.runpod/config.toml`. Requests send
their own User-Agent — Cloudflare in front of `api.runpod.io` 403s urllib's
default `Python-urllib/x`.

```bash
make runpod ARGS="gpus --min-vram 24"                  # price + stock, cheapest first
make runpod ARGS="create --gpu 'NVIDIA A40' --gpu 'NVIDIA L4' --dry-run"
make runpod ARGS="create --gpu … --wait"               # billing starts; blocks until ssh answers
make runpod ARGS="ls"                                  # anima-* pods (--all for every pod)
make runpod ARGS="ssh <pod_id> -- nvidia-smi"
make runpod ARGS="terminate <pod_id> --yes"            # pod and its disks deleted
```

`create` defaults to `sorryhyun96/anima_lora:latest` (`docker/Dockerfile`) on a
200 GB container disk with no pod volume, on a host with CUDA ≥ 13.2 (the
image's torch is cu132; `--min-cuda V`, `any` = no filter — `gpus` takes the same
flag, so its stock column matches). A driver-580 (CUDA 13.0) host has run the
image's `torch.compile` and FA2. The container disk is wiped on stop, so a pod
is terminated after its job rather than stopped. v2 takes one GPU type per
request; `--gpu` repeats and `create` tries each in order until one is placed.
`--spot` (interruptible) exists only in v1, whose `allowedCudaVersions` stops at
13.0, so it needs `--min-cuda any` (or ≤ 13.0). `create` prefixes the name with
`anima-` and sets `PUBLIC_KEY` from `~/.ssh/id_*.pub`, replacing the account's
keys for that pod. `create --wait` and `ls` print the pod's ssh `ip` and `port`;
`ssh` pins no host key (pod IP:port pairs are recycled).

The image carries the CUDA toolchain, `openssh-server` / `rsync`, `libraqm0` and the
Noto CJK fonts; its entrypoint sets `OMP_NUM_THREADS` from the container's CPU quota.
With `PUBLIC_KEY` set it authorizes the key, starts sshd and writes the image env (PATH / `ANIMA_*` / `HF_*` / `OMP_*` …) to
`/etc/environment` (pam_env overrides `~/.ssh/environment`). It then fetches the
first-run models into `/app/models` unless `ANIMA_SKIP_DOWNLOAD=1`, and idles on
`sleep infinity` (no TTY on a pod — a bare `bash` would exit and restart the
container). An ssh session lands with `/app/.venv/bin/python` on PATH; `cd /app`
for `tasks.py`.

Operating a pod — staging a run, moving data, launching, waiting, terminating:
the `runpod` skill (`.claude/skills/runpod/SKILL.md`).
