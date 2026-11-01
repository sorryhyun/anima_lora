"""The operation manifest: served at ``GET /tools``, registered by ``mcp.py``."""

from __future__ import annotations

# Machine-readable manifest served at GET /tools and registered by mcp.py — one
# entry per operation with a JSON-Schema `input_schema`. Kept in sync with the
# handlers by hand.
TOOLS = [
    {
        "name": "submit_training",
        "description": (
            "Enqueue a train.py run built from method + preset + overrides. "
            "Runs when it reaches the front of the serial queue. Returns {job_id, state}."
        ),
        "method": "POST",
        "path": "/jobs",
        "input_schema": {
            "type": "object",
            "required": ["method"],
            "properties": {
                "method": {
                    "type": "string",
                    "description": "Method/adapter config name (e.g. 'lora', 'soft_tokens', 'easycontrol').",
                },
                "preset": {
                    "type": "string",
                    "default": "default",
                    "description": "Hardware preset: default | low_vram | half | quarter | tenth | graft | debug (configs/presets.toml).",
                },
                "methods_subdir": {
                    "type": "string",
                    "description": "Config subdir, e.g. 'gui-methods' for the clean per-variant tree. Omit for 'methods'.",
                },
                "overrides": {
                    "type": "object",
                    "description": '{key: value} → --key value CLI overrides, e.g. {"network_dim": 32, "max_train_epochs": 64}.',
                },
                "extra": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra CLI args appended verbatim.",
                },
                "config_snapshot": {
                    "type": "object",
                    "description": "A fully-merged config dict to pin instead of re-resolving the base→preset→method→overrides chain at launch.",
                },
                "config_file": {
                    "type": "string",
                    "description": "Path to a config snapshot file (alternative to config_snapshot).",
                },
                "start": {
                    "type": "boolean",
                    "description": "true → run now (resume queue); false → add to queue (held for Start Queue only if the queue is idle; auto-advances behind a job that's already running); omit → leave the gate as-is.",
                },
            },
        },
    },
    {
        "name": "submit_command",
        "description": (
            "Enqueue a plain `python <argv>` task (preprocess, mask, a distill loop). "
            "Optionally carries a chain_train spec the daemon auto-enqueues on success "
            "(how 'preprocess → train' survives the caller closing). Returns {job_id, state}."
        ),
        "method": "POST",
        "path": "/jobs",
        "input_schema": {
            "type": "object",
            "required": ["label", "argv"],
            "properties": {
                "kind": {
                    "type": "string",
                    "const": "command",
                    "description": "Must be 'command'.",
                },
                "label": {
                    "type": "string",
                    "description": "Display label (doubles as the job's 'method' field).",
                },
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Argv run under the anima venv, e.g. ['tasks.py', 'preprocess-config', …].",
                },
                "extra_env": {
                    "type": "object",
                    "description": "Extra environment variables for the subprocess.",
                },
                "chain_train": {
                    "type": "object",
                    "description": "Training spec {method, preset, methods_subdir, overrides} auto-enqueued when this command finishes successfully.",
                },
                "stall_timeout": {
                    "type": "number",
                    "description": (
                        "Per-job stall-watchdog budget in seconds, overriding the "
                        "120s command-job default; 0 disables it. Raise (or disable) "
                        "for a legitimately quiet embed/eval loop."
                    ),
                },
                "config_snapshot": {
                    "type": "object",
                    "description": "Config dict to pin (forwarded to the chained train job).",
                },
                "config_file": {
                    "type": "string",
                    "description": "Config snapshot file path (alternative to config_snapshot).",
                },
                "start": {
                    "type": "boolean",
                    "description": "true → run now; false → add to queue (held only if the queue is idle; auto-advances behind a running job); omit → leave gate as-is.",
                },
            },
        },
    },
    {
        "name": "list_jobs",
        "description": "List all jobs (full records, submission order). Each has state ∈ queued|running|paused|done|error|stopped.",
        "method": "GET",
        "path": "/jobs",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_job",
        "description": (
            "Fetch one job record plus live fields: 'latest' (last progress.jsonl event, "
            "train jobs only) and 'stale_for' (seconds since last progress tick). Poll this for completion."
        ),
        "method": "GET",
        "path": "/jobs/{id}",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "Job id from submit_*."}
            },
        },
    },
    {
        "name": "get_progress",
        "description": (
            "Query a train job's structured progress.jsonl — events are "
            "run_start | step (loss/lr/metrics) | val (cmmd) | ckpt | log "
            "(mirrored WARNING+ records) | run_end. Filters keep the payload "
            "small on long runs; returns {job_id, state, count, events}. "
            "This is the surface to debug/analyze a run from (loss curve, "
            "warnings, checkpoints) — stdout is only for raw crash output."
        ),
        "method": "GET",
        "path": "/jobs/{id}/progress",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "Job id from submit_*."},
                "events": {
                    "type": "string",
                    "description": "Comma-separated ev kinds to keep, e.g. 'step,val' or 'log,run_end'. Omit for all.",
                },
                "since_step": {
                    "type": "integer",
                    "description": "Keep events at/after this global_step (step-less events inherit the preceding step).",
                },
                "every_nth": {
                    "type": "integer",
                    "description": "Thin step events to every n-th (latest step always kept).",
                },
                "last_n": {
                    "type": "integer",
                    "default": 200,
                    "description": "Trailing cap on returned events.",
                },
            },
        },
    },
    {
        "name": "stop_job",
        "description": "Abort a running or queued job (tree-kills the process). Returns {job_id, state}.",
        "method": "POST",
        "path": "/jobs/{id}/stop",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string", "description": "Job id to stop."}},
        },
    },
    {
        "name": "pause_job",
        "description": (
            "Freeze a running job's process tree (SIGSTOP) — VRAM/CUDA context "
            "stay put, SM utilisation drops to zero, resume is instant. The queue "
            "does NOT advance past it (it still owns its slot). Refuses anything "
            "not running, and refuses a multi-GPU accelerate-launch run. Returns "
            "{job_id, state, error?}. With release_model=true (train jobs only) "
            "the trainer instead saves a resumable state at its next optimizer "
            "step and exits: the GPU is freed, the queue advances, and the job "
            "parks as paused with released=true until resume_job relaunches it "
            "with --resume (model reload + recompile, not instant)."
        ),
        "method": "POST",
        "path": "/jobs/{id}/pause",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "Job id to pause."},
                "release_model": {
                    "type": "boolean",
                    "description": "Cooperative release: save resumable state, exit, free the GPU (train.py jobs only). Default false = SIGSTOP freeze.",
                },
            },
        },
    },
    {
        "name": "resume_job",
        "description": (
            "Thaw a paused job's process tree (SIGCONT) back to running. A "
            "release-paused job (released=true) is re-enqueued at the front of "
            "the queue with --resume <state_dir> instead. Returns "
            "{job_id, state, error?} (error if the job isn't paused)."
        ),
        "method": "POST",
        "path": "/jobs/{id}/resume",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "Job id to resume."}
            },
        },
    },
    {
        "name": "tail_logs",
        "description": (
            "SSE stream of a job's combined stdout+stderr from the start of the file; "
            "ends with a {ev:'eof', state} event once the job is terminal and drained, "
            "then closes the connection. There is no blocking server-side 'wait' "
            "endpoint — tail this, poll get_job, or use the client-side "
            "DaemonClient.wait / `make daemon-wait JOB=<id>`."
        ),
        "method": "GET",
        "path": "/jobs/{id}/logs",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string", "description": "Job id to tail."}},
        },
    },
    {
        "name": "pause_queue",
        "description": "Hold the queue gate — queued jobs wait until start_queue. Submissions still accepted.",
        "method": "POST",
        "path": "/queue/pause",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "start_queue",
        "description": "Resume a paused queue — the worker launches queued jobs in order.",
        "method": "POST",
        "path": "/queue/start",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "health",
        "description": "Daemon liveness: {ok, pid, port, root, fingerprint, active_job, paused, worker_alive, worker_idle_for}. 'root' is the checkout it belongs to; 'fingerprint' is the daemon-source hash it booted with (stale if it differs from current on-disk source); worker_idle_for is seconds since the job worker last advanced.",
        "method": "GET",
        "path": "/health",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "shutdown",
        "description": "Stop the daemon. Destructive: kill_jobs=true (default) tree-kills the running job too.",
        "method": "POST",
        "path": "/shutdown",
        "input_schema": {
            "type": "object",
            "properties": {
                "kill_jobs": {
                    "type": "boolean",
                    "default": True,
                    "description": "Kill the running job on the way down.",
                }
            },
        },
    },
]
