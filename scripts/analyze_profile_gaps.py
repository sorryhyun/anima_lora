"""Find GPU-idle gaps in a torch.profiler chrome trace and identify what
the CPU was doing during them.

Usage:
    python scripts/analyze_profile_gaps.py output/ckpt/profile_trace.json
    python scripts/analyze_profile_gaps.py path/to/trace.json --gap_ms 3 --top 8

The trace is produced by training with `--profile_steps START-END` (see
`train.py:1537`). Each "gap" is a span where no GPU kernel/memcpy was
executing for at least `--gap_ms` milliseconds. For every gap we list the
CPU events that overlap it, ranked by total time spent inside the gap —
that ranking is the answer to "what is starving the GPU".

Periodicity is reported in two ways: in microseconds (raw inter-gap delta)
and in steps if `ProfilerStep#N` annotations are present in the trace
(they are, by default, when torch.profiler is used as a context manager).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


# ─── chrome-trace event categorization ────────────────────────────────────

# PyTorch's chrome export uses these categories for GPU-side activity. We
# deliberately *exclude* `gpu_user_annotation`: those wrap many kernels under
# one event and would mask kernel-to-kernel idle when computed via running-max.
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}

# CPU events we never want to credit as "what was the CPU doing in the gap" —
# they are framework bookkeeping or generic dispatcher wrappers that span huge
# durations and overlap every gap by construction, drowning out the real
# culprit.
CPU_NOISE_PREFIXES = (
    "ProfilerStep",
    "[memory]",
    "Optimizer.zero_grad",
    "PyTorch Profiler",
    "<built-in method ",  # pybind11 dispatcher noise
    "<built-in function ",
    "## Call CompiledFxGraph",
    "torch/_dynamo/",
    "torch/_inductor/",
    "torch/_functorch/",
    "torch/autograd/",
    "torch/nn/modules/module.py",
    "FunctionMeta",
)


def load_events(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return data


def split_events(events: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (gpu_events, cpu_events, step_events).

    - gpu_events: kernels / memcpys, sorted by ts.
    - cpu_events: everything else with a duration; used to attribute gaps.
    - step_events: ProfilerStep#N annotations, used to translate gap timing
      into "step N" coordinates.
    """
    gpu, cpu, steps = [], [], []
    for e in events:
        if e.get("ph") != "X":
            continue
        if "dur" not in e or e.get("dur", 0) <= 0:
            continue
        cat = e.get("cat", "") or ""
        name = e.get("name", "") or ""
        if cat in GPU_CATS:
            gpu.append(e)
        elif name.startswith("ProfilerStep"):
            steps.append(e)
        else:
            cpu.append(e)
    gpu.sort(key=lambda e: e["ts"])
    cpu.sort(key=lambda e: e["ts"])
    steps.sort(key=lambda e: e["ts"])
    return gpu, cpu, steps


def find_gaps(gpu_events: list[dict], gap_us: float) -> list[tuple[float, float]]:
    """Return list of (gap_start_us, gap_end_us) where the GPU was idle for at
    least `gap_us` microseconds. Uses a running max end-time to handle multiple
    overlapping streams correctly (otherwise a long memcpy hides concurrent
    kernels and inflates the gap count)."""
    gaps: list[tuple[float, float]] = []
    if not gpu_events:
        return gaps
    running_end = gpu_events[0]["ts"] + gpu_events[0]["dur"]
    for e in gpu_events[1:]:
        ts = e["ts"]
        if ts > running_end + gap_us:
            gaps.append((running_end, ts))
        running_end = max(running_end, ts + e["dur"])
    return gaps


def is_noise(name: str) -> bool:
    return any(name.startswith(p) for p in CPU_NOISE_PREFIXES)


def cpu_overlap(
    cpu_events: list[dict], gap_start: float, gap_end: float
) -> list[tuple[str, float]]:
    """Return [(name, overlap_us), …] for every CPU event that overlaps the
    gap. Uses bisect-driven scan — cpu_events is sorted by ts.

    Crucially: we cap each event's overlap at the gap duration (not at the
    event's own duration), and we skip "wrapper" events whose duration is at
    least 90% of the gap — those are outer dispatcher frames that just contain
    the real culprit. The point is to surface *leaf-ish* events that actually
    explain why the GPU was idle.
    """
    out: list[tuple[str, float]] = []
    gap_dur = gap_end - gap_start
    for e in cpu_events:
        ts = e["ts"]
        if ts >= gap_end:
            break
        end = ts + e["dur"]
        if end <= gap_start:
            continue
        ov_start = max(ts, gap_start)
        ov_end = min(end, gap_end)
        ov = ov_end - ov_start
        if ov <= 0:
            continue
        name = e.get("name", "") or "<unnamed>"
        if is_noise(name):
            continue
        # Skip wrapper events that span essentially the entire gap — they
        # encapsulate the real culprit one or more frames deeper.
        if ov >= 0.95 * gap_dur and e["dur"] >= 5 * gap_dur:
            continue
        out.append((name, ov))
    return out


def step_at(ts: float, steps: list[dict]) -> int | None:
    """Translate a microsecond timestamp into a ProfilerStep index. Returns
    None if no step annotation covers `ts`."""
    for s in steps:
        if s["ts"] <= ts <= s["ts"] + s["dur"]:
            name = s.get("name", "")
            # name looks like "ProfilerStep#42"
            if "#" in name:
                try:
                    return int(name.rsplit("#", 1)[1])
                except ValueError:
                    return None
    return None


def fmt_us(us: float) -> str:
    if us < 1_000:
        return f"{us:.0f}us"
    if us < 1_000_000:
        return f"{us / 1_000:.2f}ms"
    return f"{us / 1_000_000:.2f}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path, help="path to profile_trace.json")
    ap.add_argument(
        "--gap_ms",
        type=float,
        default=3.0,
        help="min GPU-idle duration to count as a gap (default: 3ms)",
    )
    ap.add_argument(
        "--top", type=int, default=10, help="top-N CPU events to print per summary"
    )
    ap.add_argument(
        "--per_gap",
        type=int,
        default=5,
        help="show per-gap detail for the first N gaps (0 disables)",
    )
    args = ap.parse_args()

    if not args.trace.exists():
        sys.exit(f"trace not found: {args.trace}")

    print(f"loading {args.trace}…")
    events = load_events(args.trace)
    gpu, cpu, steps = split_events(events)

    if not gpu:
        sys.exit("no GPU events in trace — was the profiler running on CUDA?")

    trace_start = gpu[0]["ts"]
    trace_end = max(e["ts"] + e["dur"] for e in gpu)
    trace_dur = trace_end - trace_start
    # Union of busy intervals (sum of durations would double-count overlapping
    # streams; we already filtered annotation wrappers but kernels can still
    # overlap with memcpy on a multi-stream setup).
    busy = 0.0
    cur_end = gpu[0]["ts"]
    for e in gpu:
        ts = e["ts"]
        end = ts + e["dur"]
        if ts > cur_end:
            busy += end - ts
            cur_end = end
        elif end > cur_end:
            busy += end - cur_end
            cur_end = end
    util = 100.0 * busy / trace_dur if trace_dur else 0.0
    print(
        f"  {len(events):,} events  ·  {len(gpu):,} GPU  ·  {len(cpu):,} CPU  "
        f"·  {len(steps)} ProfilerSteps"
    )
    print(
        f"  trace span: {fmt_us(trace_dur)}  ·  GPU busy: {fmt_us(busy)}  "
        f"·  utilization: {util:.1f}%"
    )

    gap_us = args.gap_ms * 1_000.0
    gaps = find_gaps(gpu, gap_us)
    if not gaps:
        print(f"\nno gaps ≥ {args.gap_ms}ms found.")
        return

    durations = [b - a for a, b in gaps]
    print(
        f"\n{len(gaps)} gaps ≥ {args.gap_ms}ms  ·  "
        f"total idle: {fmt_us(sum(durations))}  "
        f"({100 * sum(durations) / trace_dur:.1f}% of trace)"
    )
    print(
        f"  gap dur: min={fmt_us(min(durations))} "
        f"median={fmt_us(median(durations))} "
        f"mean={fmt_us(mean(durations))} "
        f"max={fmt_us(max(durations))}"
    )

    # Cadence
    if len(gaps) >= 2:
        deltas_us = [gaps[i][0] - gaps[i - 1][0] for i in range(1, len(gaps))]
        print(
            f"  inter-gap spacing: median={fmt_us(median(deltas_us))} "
            f"mean={fmt_us(mean(deltas_us))}"
        )
        if steps:
            gap_step_idx = [step_at(g[0], steps) for g in gaps]
            gap_step_idx = [s for s in gap_step_idx if s is not None]
            if len(gap_step_idx) >= 2:
                step_deltas = [
                    gap_step_idx[i] - gap_step_idx[i - 1]
                    for i in range(1, len(gap_step_idx))
                ]
                print(
                    f"  inter-gap spacing in steps: median={median(step_deltas):.1f} "
                    f"mean={mean(step_deltas):.2f}  "
                    f"(over {len(gap_step_idx)} located gaps)"
                )

    # Aggregated CPU-event blame
    blame: Counter[str] = Counter()
    blame_count: Counter[str] = Counter()
    per_gap_top: list[list[tuple[str, float]]] = []
    for gs, ge in gaps:
        ovs = cpu_overlap(cpu, gs, ge)
        # Per-gap, fold occurrences of the same name within the gap
        per_name: dict[str, float] = defaultdict(float)
        for n, d in ovs:
            per_name[n] += d
        ranked = sorted(per_name.items(), key=lambda x: x[1], reverse=True)
        per_gap_top.append(ranked)
        for n, d in per_name.items():
            blame[n] += d
            blame_count[n] += 1

    print(f"\ntop {args.top} CPU events overlapping gaps (by total overlap time):")
    total_blame = sum(blame.values()) or 1.0
    for name, dur in blame.most_common(args.top):
        share = 100 * dur / total_blame
        hits = blame_count[name]
        print(
            f"  {share:5.1f}%  {fmt_us(dur):>8}  in {hits:>3}/{len(gaps)} gaps   {name}"
        )

    if args.per_gap > 0:
        print(f"\nfirst {min(args.per_gap, len(gaps))} gaps (top-3 events each):")
        for i, (gs, ge) in enumerate(gaps[: args.per_gap]):
            step = step_at(gs, steps) if steps else None
            tag = f"step {step}" if step is not None else f"t={fmt_us(gs - trace_start)}"
            top = per_gap_top[i][:3]
            descr = (
                ", ".join(f"{n} ({fmt_us(d)})" for n, d in top) if top else "<no CPU>"
            )
            print(f"  gap {i + 1:>2}  {tag:<14}  dur={fmt_us(ge - gs):>8}   {descr}")


if __name__ == "__main__":
    main()
