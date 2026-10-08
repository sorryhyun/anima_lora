# bench/

Benchmark and probe scripts — the reproducible evidence behind method claims.
CONTRIBUTING.md requires one for every numerics/efficiency change (Tier 1.5) and
every new method (Tier 2): a runnable script that reports the headline number(s)
the change claims to move, **before and after**. This directory is where those
scripts and their results live.

`bench/` is **not an installed package** (only `anima_lora` / `library` /
`networks` are) — every script starts with a two-line `sys.path` bootstrap so
`uv run python bench/<method>/<script>.py` works from the repo root:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from bench._common import make_run_dir, write_result  # noqa: E402
```

## Layout

```
bench/
  _common.py            # result.json envelope: make_run_dir + write_result
  _anima.py             # bench-facing CLI surface + model-loading re-exports
  _template/            # copy-me skeleton for a new bench (see below)
  <method>/             # one dir per method / topic
    README.md           # REQUIRED: what it measures, run command, interpretation
    <script>.py         # probes / sweeps / A-B runners
    results/            # gitignored — runs land here, numbers go in the PR text
      <YYYYMMDD-HHMM>[-<label>]/
        result.json     # the standard envelope
        <artifacts>     # CSVs, PNGs, NPZs referenced by the envelope
```

Each `bench/<method>/README.md` must cover: what the bench measures, a
copy-pasteable run command, the output layout, a baseline run's numbers, and how
to interpret them. `bench/dave/` and `bench/memorization/` are the fleshed-out
shape templates. `results/` is **gitignored** — evidence stays local; paste the
before/after output into the PR description (reviewers reproduce with one
command).

## The two shared modules

**`bench/_common.py` — the result envelope.** Every script writes a
`result.json` with one outer schema (script, label, UTC timestamp, git sha/dirty,
python/torch/CUDA/GPU env, full args, script-specific `metrics`, artifact list)
so runs from different methods can be compared and re-found. Usage:

```python
from bench._common import make_run_dir, write_result

run_dir = make_run_dir("spectrum", label=args.label)
# ... write artifacts into run_dir ...
write_result(run_dir, script=__file__, args=args,
             metrics={...}, label=args.label, artifacts=["curves.png"])
```

Gotcha: everything after `run_dir` is **keyword-only** — a positional
`write_result(run_dir, metrics)` fails. And `make_run_dir` is
`<YYYYMMDD-HHMM>[-label]` with `exist_ok=True`: two runs of one script in the same
minute with the same (or no) label share a dir and the second overwrites
`result.json` — pass a distinct `--label` per run. When a script runs as a daemon
job the envelope path is auto-lifted into the job record (`ANIMA_DAEMON_JOB_DIR`); a
plain inline run is unaffected — zero daemon coupling.

**`bench/_anima.py` — the model-loading surface.** If your bench loads the DiT,
use this instead of hand-rolling paths and load order:

- `add_model_args(p)` — injects `--dit/--vae/--text_encoder` at the canonical
  `default_checkpoints()` paths (honors `ANIMA_DIT`/… env + `.env` + base.toml).
- `add_common_args(p)` — injects `--label/--seed/--device/--dtype/--attn_mode/
  --gradient_checkpointing/--compile/--compile_mode`. Every DiT-loading bench
  must expose `--compile` — this is how.
- `build_anima(args, adapter=…, train_mode=…)` — the shared harness (re-exported
  from `library/runtime/harness.py`) encoding the **compile-after-apply**
  invariant: `compile_blocks()` runs after `network.apply_to` + `load_weights`,
  or torch.compile traces the wrong forward.
- `DEFAULT_PROMPT` / `DEFAULT_NEG` — the canonical `make test` prompt pair.

For raw cache reads (latents / text embeddings under `post_image_dataset/lora/`)
use `library/io/cache.py` (`load_cached_latents`, `load_cached_text_features`,
`discover_latents_by_stem`, …) — don't hand-roll the NPZ key scan.

## Starting a new bench

1. Copy `bench/_template/` to `bench/<your_topic>/` (or add a script to an
   existing `bench/<method>/` if the change is scoped to that method).
2. Replace the placeholder metric with the number your change claims to move;
   run it on both the before and the after tree.
3. Write the dir's `README.md` (use `bench/dave/` as the model).
4. Paste both runs' output into the PR description and link the script.

Older scripts predate the envelope and don't call `make_run_dir`/`write_result`;
converting a holdout is a self-contained Tier 1 PR (see CONTRIBUTING §bench
gaps).
