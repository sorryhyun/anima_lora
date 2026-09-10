#!/usr/bin/env python3
"""bench/torch_bump — A/B a scratch torch venv against ``.venv`` on a real compiled LoRA run.

Trial recipe from ``docs/proposal/torch214.md`` step 4: same seed, same dataset
subset, ``torch_compile`` on, one short ``train.py --method lora`` per arm,
submitted through the daemon (agent-launched GPU work must go through it).
Every arm gets a **cold, private** Inductor + Triton cache so first-step wall
is comparable; a repeated arm reuses its first instance's cache, which also
measures the warm-cache load.

Arm spec ``VER:MODE[:K=V;K=V]`` — ``VER`` picks the interpreter (``--venv
212=.venv --venv 214=.venv-t214``), ``MODE`` is ``--compile_inductor_mode``
(``default`` = flag omitted), the optional env block is layered on the job;
a ``pin.<inductor_key>=<val>`` entry there becomes a ``--pin`` on the shim
(a ``torch._inductor.config`` pin for knobs that have no env var).
Order is the queue order; repeat an arm to get an in-batch noise floor
([[project_deterministic_flag_chaos_floor]]: never quote a floor across setups).

Usage::

    .venv/bin/python bench/torch_bump/run_bench.py submit --label t214 \
        --arms 212:default 214:default 212:max-autotune 214:max-autotune 212:default 214:default
    .venv/bin/python bench/torch_bump/run_bench.py analyze bench/torch_bump/results/<dir>

``submit`` writes ``manifest.json`` into the run dir and returns; ``analyze``
blocks on the daemon (``DaemonClient.wait``) then writes ``result.json`` +
``summary.md``. Metrics per run: time to first step (load + compile), warm
s/it (mean over steps ≥ ``--warm``, plus per-event median), peak VRAM (shim),
final-loss mean, the per-band ``mark_dynamic`` line, recompile-log count.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))  # bench/ is not an installed package

from bench._common import make_run_dir, write_result  # noqa: E402

SHIM = Path(__file__).resolve().parent / "vram_shim.py"


def parse_arm(spec: str) -> dict:
    parts = spec.split(":", 2)
    if len(parts) < 2:
        raise SystemExit(f"bad arm spec {spec!r}: want VER:MODE[:K=V;K=V]")
    ver, mode = parts[0], parts[1]
    env = {}
    if len(parts) == 3 and parts[2]:
        for kv in parts[2].split(";"):
            k, _, v = kv.partition("=")
            env[k.strip()] = v.strip()
    key = f"{ver}-{mode}" + (("-" + "-".join(sorted(env))) if env else "")
    return {"ver": ver, "mode": mode, "env": env, "key": key}


def torch_version(py: str) -> str:
    out = subprocess.run(
        [
            py,
            "-c",
            "import torch, triton; print(torch.__version__, triton.__version__)",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return out.stdout.strip() or out.stderr.strip()[-200:]


def cmd_submit(args: argparse.Namespace) -> None:
    from anima_daemon.client import DaemonClient, ensure_daemon

    venvs = dict(kv.split("=", 1) for kv in args.venv)
    run_dir = make_run_dir("torch_bump", label=args.label)
    ensure_daemon()
    client = DaemonClient()
    arms = [parse_arm(a) for a in args.arms]
    seen: dict[str, int] = {}
    jobs = []
    for arm in arms:
        rep = seen.get(arm["key"], 0)
        seen[arm["key"]] = rep + 1
        py = str(
            REPO / venvs[arm["ver"]] / "bin" / "python"
        )  # no resolve(): the venv symlink must be invoked as-is
        name = f"tb_{args.label}_{arm['key']}_r{rep}"
        progress = run_dir / f"{name}.progress.jsonl"
        cache = run_dir / f"cache-{arm['key']}"
        pins = [
            f"--pin {k[4:]}={v}".split()
            for k, v in arm["env"].items()
            if k.startswith("pin.")
        ]
        argv = [
            str(SHIM),
            *[tok for pin in pins for tok in pin],
            py,
            "train.py",
            "--method",
            args.method,
            "--preset",
            args.preset,
            "--path_pattern",
            args.path_pattern,
            "--max_train_epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--output_name",
            name,
            "--progress_jsonl",
            str(progress),
            *args.extra,
        ]
        if arm["mode"] != "default":
            argv += ["--compile_inductor_mode", arm["mode"]]
        env = {
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            **{k: v for k, v in arm["env"].items() if not k.startswith("pin.")},
        }
        rec = client.submit_command(
            label=name, argv=argv, extra_env=env, stall_timeout=0
        )
        job_id = rec.get("id") or rec.get("job_id") or rec.get("job", {}).get("id")
        if not job_id:
            raise SystemExit(f"could not find job id in daemon response: {rec}")
        jobs.append(
            {
                **arm,
                "rep": rep,
                "name": name,
                "job_id": job_id,
                "python": py,
                "torch": torch_version(py),
                "progress": str(progress),
                "cache_dir": str(cache),
                "warm_cache": rep > 0,
                "argv": argv,
                "extra_env": env,
            }
        )
        print(f"submitted {name} -> {job_id}")
    manifest = {"label": args.label, "args": vars(args), "jobs": jobs}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {run_dir / 'manifest.json'}")
    print(f"then: {sys.executable} {__file__} analyze {run_dir}")


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def analyze_run(job: dict, record: dict, warm: int) -> dict:
    events = _read_jsonl(Path(job["progress"]))
    start = next((e for e in events if e.get("ev") == "run_start"), {})
    end = next((e for e in reversed(events) if e.get("ev") == "run_end"), None)
    steps = [e for e in events if e.get("ev") == "step"]
    m: dict = {
        "state": record.get("state"),
        "returncode": record.get("returncode"),
        "error": record.get("error"),
        "wall_total_s": (record.get("ended_at") or 0) - (record.get("started_at") or 0),
        "total_steps": start.get("total_steps"),
        "steps_logged": len(steps),
    }
    if steps and start:
        m["t_to_first_step_s"] = steps[0]["ts"] - start["ts"]
        tail = [e for e in steps if e["global_step"] >= warm]
        if len(tail) >= 2:
            d_step = tail[-1]["global_step"] - tail[0]["global_step"]
            d_ts = tail[-1]["ts"] - tail[0]["ts"]
            m["s_per_it_warm_mean"] = d_ts / d_step if d_step else None
            per = [
                (b["ts"] - a["ts"]) / (b["global_step"] - a["global_step"])
                for a, b in zip(tail, tail[1:])
                if b["global_step"] > a["global_step"]
            ]
            m["s_per_it_warm_median"] = statistics.median(per) if per else None
            m["s_per_it_warm_p90"] = (
                sorted(per)[int(0.9 * (len(per) - 1))] if per else None
            )
        losses = [
            e.get("loss/current", e.get("loss"))
            for e in steps[-25:]
            if isinstance(e.get("loss/current", e.get("loss")), (int, float))
        ]
        m["loss_mean_last"] = statistics.fmean(losses) if losses else None
        if end:
            m["run_status"] = end.get("status")
            m["wall_train_s"] = end["ts"] - start["ts"]
    stdout = Path(record.get("stdout_path") or "")
    if stdout.exists():
        text = stdout.read_text(errors="replace")
        vm = re.search(r"VRAM_SHIM peak_used_mib=(\d+)", text)
        m["peak_vram_used_mib"] = int(vm.group(1)) if vm else None
        band = re.search(r"dynamic-seq per-band mark_dynamic[^\n]*", text)
        m["band_line"] = band.group(0).strip() if band else None
        m["recompile_log_lines"] = len(re.findall(r"(?i)recompil", text))
        m["stdout_path"] = str(stdout)
    return m


def cmd_analyze(args: argparse.Namespace) -> None:
    from anima_daemon.client import DaemonClient

    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    client = DaemonClient()
    runs = []
    for job in manifest["jobs"]:
        record = client.wait(job["job_id"], timeout=args.timeout)
        metrics = analyze_run(job, record, args.warm)
        runs.append(
            {
                k: job[k]
                for k in (
                    "key",
                    "ver",
                    "mode",
                    "rep",
                    "name",
                    "job_id",
                    "torch",
                    "warm_cache",
                )
            }
            | metrics
        )
        print(
            f"{job['name']}: {json.dumps({k: metrics.get(k) for k in ('state', 's_per_it_warm_mean', 't_to_first_step_s', 'peak_vram_used_mib')})}"
        )

    by_arm: dict[str, dict] = {}
    for r in runs:
        if r.get("s_per_it_warm_mean") is None:
            continue
        by_arm.setdefault(
            r["key"],
            {"s_per_it": [], "t_first_cold": [], "t_first_warm": [], "vram": []},
        )
        a = by_arm[r["key"]]
        a["s_per_it"].append(r["s_per_it_warm_mean"])
        (a["t_first_warm"] if r["warm_cache"] else a["t_first_cold"]).append(
            r.get("t_to_first_step_s")
        )
        if r.get("peak_vram_used_mib"):
            a["vram"].append(r["peak_vram_used_mib"])
    summary = {}
    for key, a in by_arm.items():
        summary[key] = {
            "s_per_it_mean": statistics.fmean(a["s_per_it"]),
            "s_per_it_spread_pct": (
                100
                * (max(a["s_per_it"]) - min(a["s_per_it"]))
                / statistics.fmean(a["s_per_it"])
                if len(a["s_per_it"]) > 1
                else None
            ),
            "t_first_cold_s": statistics.fmean(a["t_first_cold"])
            if a["t_first_cold"]
            else None,
            "t_first_warm_s": statistics.fmean(a["t_first_warm"])
            if a["t_first_warm"]
            else None,
            "peak_vram_used_mib": max(a["vram"]) if a["vram"] else None,
            "n": len(a["s_per_it"]),
        }
    # Δ% vs the same-mode baseline (first VER seen in the arm list = baseline).
    base_ver = manifest["jobs"][0]["ver"]
    for key, s in summary.items():
        ver, _, rest = key.partition("-")
        base = summary.get(f"{base_ver}-{rest}")
        s["delta_pct_vs_base"] = (
            100 * (s["s_per_it_mean"] / base["s_per_it_mean"] - 1)
            if base and ver != base_ver
            else None
        )

    lines = [
        f"# torch_bump — {manifest['label']}",
        "",
        "| arm | torch | n | s/it (warm) | Δ% vs base | spread % | first step cold (s) | first step warm (s) | peak VRAM used (MiB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, s in summary.items():
        tv = next(r["torch"] for r in runs if r["key"] == key)
        f = lambda v, p=3: "—" if v is None else f"{v:.{p}f}"  # noqa: E731
        lines.append(
            f"| {key} | {tv} | {s['n']} | {f(s['s_per_it_mean'])} | {f(s['delta_pct_vs_base'], 2)} | "
            f"{f(s['s_per_it_spread_pct'], 2)} | {f(s['t_first_cold_s'], 1)} | {f(s['t_first_warm_s'], 1)} | {s['peak_vram_used_mib'] or '—'} |"
        )
    lines += ["", "## runs", ""]
    for r in runs:
        lines.append(
            f"- `{r['name']}` job `{r['job_id']}`: {r.get('state')} rc={r.get('returncode')} "
            f"steps={r.get('steps_logged')}/{r.get('total_steps')} s/it={r.get('s_per_it_warm_mean')} "
            f"median={r.get('s_per_it_warm_median')} p90={r.get('s_per_it_warm_p90')} "
            f"first_step={r.get('t_to_first_step_s')} vram={r.get('peak_vram_used_mib')} "
            f"loss={r.get('loss_mean_last')} recompile_lines={r.get('recompile_log_lines')}"
        )
        if r.get("band_line"):
            lines.append(f"  - {r['band_line']}")
        if r.get("error"):
            lines.append(f"  - error: {r['error']}")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    write_result(
        run_dir,
        script=__file__,
        args=manifest["args"],
        label=manifest["label"],
        metrics={"summary": summary, "runs": runs},
        artifacts=["summary.md", "manifest.json"],
    )
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--label", required=True)
    s.add_argument(
        "--arms",
        nargs="+",
        default=[
            "212:default",
            "214:default",
            "212:max-autotune",
            "214:max-autotune",
            "212:default",
            "214:default",
        ],
    )
    s.add_argument(
        "--venv",
        nargs="+",
        default=["212=.venv", "214=.venv-t214"],
        help="VER=venv-dir pairs",
    )
    s.add_argument("--method", default="lora")
    s.add_argument("--preset", default="default")
    s.add_argument(
        "--path_pattern",
        default="mikozin/*",
        help="dataset subset (70 images → 3 epochs ≈ 210 steps)",
    )
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument(
        "--extra", nargs="*", default=[], help="extra train.py flags, verbatim"
    )
    a = sub.add_parser("analyze")
    a.add_argument("run_dir")
    a.add_argument(
        "--warm", type=int, default=40, help="ignore steps below this for s/it"
    )
    a.add_argument("--timeout", type=float, default=None)
    args = p.parse_args()
    (cmd_submit if args.cmd == "submit" else cmd_analyze)(args)


if __name__ == "__main__":
    main()
