"""train.py with the block partition, per-step FA2 and int8 kernel calls and, with
ANIMA_PROBE_PROFILE=1, a kernel split of steps 14-19. Stops at step ANIMA_PROBE_STOP
(default 30) with no checkpoint. The daemon passes only ``ANIMA_*`` variables.

    make daemon-run ARGS="--stall-timeout 400 project/int8_train/src/anima_partition_probe.py \\
        --method lora --preset default --artist_filter 'ama mitsuki' --seed 1408198296 \\
        --max_train_steps 30 --int8 stream --activation_memory_budget 0.3"
"""

import collections
import os
import runpy
import statistics
import sys
import time
import types

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import comfy_kitchen  # noqa: E402
import torch  # noqa: E402
import torch._functorch.partitioners as P  # noqa: E402
from flash_attn import flash_attn_interface as fi  # noqa: E402

import library.anima.models as m  # noqa: E402


def name_of(node) -> str:
    t = node.target
    return getattr(t, "__name__", None) or str(t)


def gb(nodes) -> float:
    return sum(P._size_of(n) for n in nodes) / 1e9


def group(nodes) -> str:
    by = collections.Counter()
    for n in nodes:
        by[name_of(n)] += P._size_of(n)
    return ", ".join(f"{k}={v / 1e6:.1f}MB" for k, v in by.most_common(14))


def is_custom(k: str) -> bool:
    return any(
        t in k
        for t in (
            "linear_stream",
            "linear_full",
            "linear_fwd",
            "dx_stream",
            "flash_attn",
            "auto_functionalized",
        )
    )


print(
    "PROBE env",
    {k: v for k, v in os.environ.items() if k.startswith("ANIMA_PROBE")},
    flush=True,
)
cuts = []
orig_solve = P.solve_min_cut


def solve(graph, info, options, *a, **k):
    out = orig_solve(graph, info, options, *a, **k)
    saved = out[0]
    cuts.append((options, saved))
    return out


P.solve_min_cut = solve
orig_choose = P.choose_saved_values_set
partition_no = [0]


def choose(joint_graph, node_info, memory_budget=1, **k):
    cuts.clear()
    saved = orig_choose(joint_graph, node_info, memory_budget=memory_budget, **k)
    partition_no[0] += 1
    names = collections.Counter(
        name_of(n) for n in joint_graph.nodes if n.op == "call_function"
    )
    custom = {k: v for k, v in names.items() if is_custom(k)}
    print(f"\nPROBE partition {partition_no[0]} budget={memory_budget}", flush=True)
    print(f"PROBE joint custom ops {custom}", flush=True)
    print(f"PROBE inputs {gb(node_info.inputs):.4f} GB", flush=True)
    for options, cut in cuts:
        same = {id(n) for n in cut} == {id(n) for n in saved}
        print(
            f"PROBE cut saved={gb(cut):.4f} GB returned={same} "
            f"far={options.ban_if_used_far_apart} "
            f"chain={options.ban_if_long_fusible_chains} "
            f"mat={options.ban_if_materialized_backward} "
            f"allow={options.ban_if_not_in_allowlist}",
            flush=True,
        )
    print(f"PROBE saved {gb(saved):.4f} GB: {group(saved)}", flush=True)
    return saved


P.choose_saved_values_set = choose
orig_extract = P._extract_fwd_bwd_modules


def extract(*a, **k):
    fw, bw = orig_extract(*a, **k)
    for tag, gm in (("fw", fw), ("bw", bw)):
        names = collections.Counter(
            name_of(n) for n in gm.graph.nodes if n.op == "call_function"
        )
        custom = {k: v for k, v in names.items() if is_custom(k)}
        casts = collections.Counter()
        for n in gm.graph.nodes:
            if n.op == "call_function" and "convert_element" in name_of(n):
                src = n.args[0].meta["val"].dtype
                casts[f"{src}->{n.args[1]}"] += 1
                user = next(iter(n.users), None)
                if is_custom(
                    name_of(n.args[0]) if n.args[0].op == "call_function" else ""
                ) or (
                    user is not None
                    and user.op == "call_function"
                    and is_custom(name_of(user))
                ):
                    casts[f"[next to custom] {src}->{n.args[1]}"] += 1
        print(
            f"PROBE {tag} custom {custom} casts {dict(casts)} "
            f"ops {sum(names.values())}",
            flush=True,
        )
    return fw, bw


P._extract_fwd_bwd_modules = extract

STOP = int(os.environ.get("ANIMA_PROBE_STOP", "30"))
PROFILE = os.environ.get("ANIMA_PROBE_PROFILE") == "1"
prof = [None]
PROF_FROM, PROF_N = 14, 6
counts = collections.Counter()
orig_int8 = comfy_kitchen.int8_linear


def int8_linear(x, wq, s, **k):
    counts[f"int8 {x.dtype} N={wq.shape[0]}"] += 1
    return orig_int8(x, wq, s, **k)


comfy_kitchen.int8_linear = int8_linear
real = fi.flash_attn_gpu


def counted(name):
    fn = getattr(real, name)

    def call(*a, **k):
        counts[f"fa2 {name}"] += 1
        return fn(*a, **k)

    return call


fi.flash_attn_gpu = types.SimpleNamespace(fwd=counted("fwd"), bwd=counted("bwd"))

orig = m.Block.forward
seen = [0]
stamps = []
sizes = []
snap = []


def forward(self, x, emb, ctx, *a, **k):
    if seen[0] == 0:
        print(
            f"PROBE block input x {x.dtype} emb {emb.dtype} ctx {ctx.dtype} "
            f"autocast {torch.is_autocast_enabled('cuda')} shape {tuple(x.shape)}",
            flush=True,
        )
    if seen[0] % 28 == 0:
        step = seen[0] // 28
        if PROFILE and step == PROF_FROM:
            prof[0] = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CUDA,
                    torch.profiler.ProfilerActivity.CPU,
                ]
            )
            prof[0].__enter__()
        if PROFILE and step == PROF_FROM + PROF_N:
            prof[0].__exit__(None, None, None)
            kernel_split()
        if step == STOP:
            report()
            sys.stdout.flush()
            os._exit(0)
        stamps.append(time.perf_counter())
        sizes.append(x.shape[2])
        snap.append(dict(counts))
    out = orig(self, x, emb, ctx, *a, **k)
    seen[0] += 1
    return out


def kernel_split():
    import re

    cats = [
        ("rotate_quant", r"quantize_int8|convrot"),
        ("int8_gemm", r"signed char"),
        ("attention", r"flash|fmha|attention"),
        ("memcpy", r"^Memcpy|^Memset"),
        ("gemm", r"gemm|gemv|cutlass|sm\d+_xmma"),
        ("optimizer", r"multi_tensor|foreach"),
        ("triton_fused", r"^triton_"),
        ("copy_cast", r"copy|elementwise_kernel"),
    ]
    by = collections.defaultdict(lambda: [0.0, 0])
    tot = collections.Counter()
    n_kernels = 0
    for e in prof[0].events():
        if e.device_type != torch.autograd.DeviceType.CUDA or getattr(
            e, "is_user_annotation", False
        ):
            continue
        by[e.name][0] += e.device_time
        by[e.name][1] += 1
        n_kernels += 1
        cat = next((c for c, pat in cats if re.search(pat, e.name)), "other")
        tot[cat] += e.device_time / 1e3 / PROF_N
    print(f"PROBE kernels/step {n_kernels / PROF_N:.0f}", flush=True)
    print(
        "PROBE split ms/step "
        + ", ".join(f"{k}={v:.1f}" for k, v in tot.most_common()),
        flush=True,
    )
    print(f"PROBE split total {sum(tot.values()):.1f} ms", flush=True)
    top = sorted(by.items(), key=lambda kv: -kv[1][0])[:28]
    for name, (us, cnt) in top:
        print(
            f"PROBE kernel {us / 1e3 / PROF_N:7.2f} ms {cnt / PROF_N:7.1f}/step {name[:110]}",
            flush=True,
        )


def report():
    if len(snap) > 8:
        delta = {k: snap[8].get(k, 0) - snap[7].get(k, 0) for k in snap[8]}
        print(f"PROBE calls in step 7: {delta}", flush=True)
    window = stamps[4:]
    if PROFILE:
        window = stamps[4:PROF_FROM]
    dt = [b - a for a, b in zip(window, window[1:])]
    if dt:
        print(
            f"PROBE steps {len(dt)} median {statistics.median(dt) * 1000:.1f} ms "
            f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB "
            f"tokens {min(sizes)}..{max(sizes)}",
            flush=True,
        )


m.Block.forward = forward
sys.argv = ["train.py", *sys.argv[1:]]
runpy.run_path("train.py", run_name="__main__")
