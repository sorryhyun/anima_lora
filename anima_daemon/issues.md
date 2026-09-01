# Daemon — known issues & friction log

Running list of daemon bugs and UX friction, collected from real sessions.
Each entry: symptom → repro → suggestion. Severity: **bug** (wrong/broken),
**friction** (works but fights the user), **idea** (missing capability).

Last updated: 2026-09-01 (§1–§5 fixed; §6 still open).

---

## 1. `make run-status` is blind to daemon-launched train runs — **bug** · FIXED 2026-09-01

The default launch path now routes every train job through the daemon, and a
daemon job writes its progress stream to
`output/daemon/jobs/<id>/progress.jsonl` (the record's `progress_path`).
`scripts/run_status.py` only scanned `output/logs/`, where the run dir (e.g.
`output/logs/cjk_unmask_a_20260901-1802/`) holds the snapshot TOML and the TB
events but **no `progress.jsonl`** — so `make run-status RUN=<name>` exited 1
with "no progress stream" while the run was training fine.

- Repro (2026-09-01): `make lora-gui CUSTOM=cjk_unmask_a.toml ARGS="--queue"`,
  wait until `daemon-status` shows `global_step` advancing, then
  `make run-status RUN=cjk_unmask_a` → error, every time.
- Effect: the documented "where is this run at?" front door failed for the
  *default* launch mode; you had to fall back to
  `make daemon-status ARGS="--job <id>"` and read the raw `latest` event.

**Fix** (scanner side — the daemon's per-job path stays authoritative):
`run_status.py::_find_streams` now scans `output/daemon/jobs/*/progress.jsonl`
alongside `output/logs/**/*.progress.jsonl`, newest-mtime first across both. A
daemon stream's filename is bare (`progress.jsonl`), so the run name is matched
against the `run_start` event *inside* the stream (`_stream_run_name`) — and the
job id resolves as a target too, since that's what a `daemon-status` listing
gives you. The header prints `(job <id>)` for a daemon run so the follow-up
(`daemon-attach JOB=…`) needs no lookup step. `--jobs-dir ''` opts out.
Command jobs never materialize a stream, so `is_file()` filters them.
Regression: `tests/test_progress_sink.py::test_run_status_finds_daemon_job_streams`.

## 2. Two `--label`s in `daemon-run`; the job record usually ends up unlabeled — **friction** · FIXED 2026-09-01

`daemon-run`'s own `--label` must come **before** the script path; anything
after it goes to the child. Bench scripts take their own `--label`, which is
where muscle memory puts it — result: the *bench run dir* was labeled but the
*daemon job record* was not. `daemon-status` listings then showed a column of
identical `bench/cjk_adapter/run_bench.py` rows with empty labels (see the
2026-09-01 grid session: ~10 unlabeled rows distinguishable only by timestamp).

**Fix** (display only — the child argv is still passed through untouched):
`cli._label_for` folds the child's own `--label` into the derived job label
(`run_bench --label ko3_a` → `run_bench:ko3_a`), and `_job_target` in
`scripts/tasks/daemon.py` scans `--label` last (after `--name` /
`--path_pattern` / `--output_name`, so soup/preprocess keep their names). The
`target` is derived at read time, so it also rescues job records already on
disk. Both `--label V` and `--label=V` forms.

## 3. Compact `daemon-status` rows omit the exit code — **friction** · FIXED 2026-09-01

The compact summaries showed `state` but not `returncode`, so a `done` row and a
"done but rc=1" row looked identical; you had to open `--job <id>` per job to
tell them apart. Batch workflows (queue N, check later) want failure visibility
in the listing.

**Fix**: `returncode` added to `_STATUS_JOB_FIELDS`. Note the originally-stated
case turned out not to occur in 605 jobs of history — the manager already maps a
nonzero exit to `state: error`. What the field actually buys is the **residual**:
`stopped` rows with `rc=0` (48 in history), `error` rows with `rc=None` (adopted
orphans, 4), and signal deaths (negative rc, e.g. `-9` = SIGKILL), which state
alone cannot separate.

## 4. Submission prints 5 lines of boilerplate per job — **nit** · FIXED 2026-09-01

`_print_queued` emitted the attach/kill/terminate cheat-sheet on every submit.
Queueing several jobs in a loop buried the one informative line
(`queued job <id>`) under repeated hints; a user watching the scrollback asked
"what did the queue just get?" mid-session (2026-09-01).

**Fix**: `scripts/tasks/_common.py` prints the cheat-sheet **once per process**
(`_QUEUED_HINTS_SHOWN`); every later submit prints only its own
`queued job <id> (<desc>)` line. Caveat: a shell loop of separate `make …
--queue` invocations is N processes, so it still gets N cheat-sheets — the fix
covers the in-process sweep/GUI/grid path, which is where the pile-up came from.

## 5. `daemon-wait` has no max-wait / summary-on-timeout — **idea** · FIXED 2026-09-01

`daemon-wait` blocks until terminal. For agent/scripted use a
`--timeout <s>` that exits with a distinct code and prints the latest progress
event would avoid the caller having to wrap it in its own timeout and lose the
in-flight status (observed: harness kills the wait at its own limit and the
partial output is empty because attach buffers).

**Fix**: `--timeout` and the `124` exit already existed; what was missing was the
summary. `cli._wait_and_report` now prints a JSON snapshot on timeout —
`{job_id, timed_out, state, started_at, stdout_path, progress_path, latest,
stale_for}` — so the caller can tell healthy-but-slow from wedged. `latest` comes
from the daemon's `GET /jobs/{id}`, falling back to reading `progress.jsonl`
directly so the snapshot still works when the daemon is the wedged thing.
Absent keys are omitted (a command job has no progress stream).

## 6. Standing issues (recorded in project memory, kept here for one place) — **OPEN**

Both need a root-cause pass, not a patch; neither was touched on 2026-09-01.

- **`daemon-pause` unreliable in practice** — docs describe the SIGSTOP
  tree-freeze as safe/instant, but in practice pausing live train jobs has
  burned runs; policy has been "don't risk live jobs on it". Needs a
  root-cause pass before it's trusted (interaction with CUDA/NCCL watchdogs
  suspected).
- **Stall watchdog kills first-run HF downloads** — a long quiet
  `hf_hub_download` inside a job trips the stall timeout and the job is
  killed mid-download; workaround is `--stall-timeout 0` on any job that may
  fetch models. Suggestion: treat growing files under `~/.cache/huggingface`
  (or child net I/O) as liveness, or default the watchdog off for `download`
  targets.
